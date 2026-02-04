from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv

# Import modular components
from config import initialize_backend, get_context_window, remember_exchange, build_messages_with_context
from services.embeddings import retrieve_context
from services.llm_clients import send_ollama_chat, send_ollama_chat_stream
from handlers.video_handler import process_video
from handlers.audio_handler import process_audio
from handlers.pdf_handler import process_pdf
from handlers.image_handler import process_image
from handlers.csv_handler import process_csv
from services.database import (
    get_db_schema, execute_sql, enforce_sql_safety, 
    repair_sql, format_sql_table, generate_sql_with_llm
)

script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, ".env")
load_dotenv()

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize all components once at startup
print("\n" + "="*70)
print("INITIALIZING RAG CHATBOT BACKEND")
print("="*70)

backend_config = initialize_backend()

# Extract components
collections = backend_config["collections"]
video_collection = collections.get("video")
audio_collection = collections.get("audio")
pdf_collection = collections.get("pdf")
image_collection = collections.get("image")

ollama_client = backend_config["ollama_client"]
OLLAMA_MODEL = backend_config["ollama_model"]
OLLAMA_HOST = os.getenv("OLLAMA_URL", "http://localhost:11434")

DB_PATH = backend_config["db_path"]

print(f"\n✅ Backend ready to accept requests!\n")

# Configuration
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "30"))


def format_reply_with_model(reply_text: str) -> str:
    return f"Model: {OLLAMA_MODEL}\n{reply_text}"


class ChatRequest(BaseModel):
    message: str
    use_video: bool = False
    use_audio: bool = False
    use_pdf: bool = False
    use_image: bool = False
    use_sql: bool = False


@app.get("/")
async def root():
    return {"status": "Chatbot API running"}


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Transcribe video using local Whisper and store chunks in ChromaDB"""
    try:
        file_bytes = await file.read()
        result = await process_video(file_bytes, file.filename, video_collection)
        return result
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Analyze image via Ollama vision model, chunk text and store in ChromaDB"""
    try:
        file_bytes = await file.read()
        result = await process_image(
            file_bytes, 
            file.filename, 
            image_collection,
            ollama_client=ollama_client,
            vision_model="moondream:1.8b"
        )
        return result
    except Exception as e:
        print(f"Image Upload Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """Transcribe audio using local Whisper and store chunks in ChromaDB"""
    try:
        file_bytes = await file.read()
        result = await process_audio(file_bytes, file.filename, audio_collection)
        return result
    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV and create SQL table for querying."""
    try:
        file_bytes = await file.read()
        result = await process_csv(file_bytes, file.filename, str(DB_PATH))
        return result
    except Exception as e:
        print(f"CSV Upload Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Extract text from PDF and store chunks in ChromaDB (with intelligent OCR)"""
    try:
        file_bytes = await file.read()
        result = await process_pdf(
            file_bytes, 
            file.filename, 
            pdf_collection, 
            MAX_PDF_SIZE_MB
        )
        return result
    except Exception as e:
        print(f"PDF Upload Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: ChatRequest):
    """Send message with optional video, audio, pdf, image, or SQL context"""
    if not request.message.strip():
        return {"error": "Message cannot be empty"}

    # --- SQL MODE ---
    if request.use_sql:
        try:
            schema = get_db_schema(DB_PATH)
            if "No database found" in schema or "Database is empty" in schema:
                return {
                    "reply": "No SQL database available. Please upload a CSV file first.",
                    "source": "SQL Context"
                }
            
            # Build messages with context for SQL generation
            messages_with_context = build_messages_with_context(request.message)
            
            # Generate SQL WITH prior conversation context for follow-up questions
            raw_sql = generate_sql_with_llm(
                request.message, 
                schema,
                ollama_client,
                OLLAMA_MODEL,
                messages_with_context
            )
            
            # Repair and enforce safety
            fixed_sql = repair_sql(raw_sql)
            safe_sql = enforce_sql_safety(fixed_sql)
            
            # Execute query
            rows = execute_sql(safe_sql, DB_PATH)
            table = format_sql_table(rows)
            
            # Format response with proper markdown (clean spacing for tables)
            response_text = (
                f"**SQL Query:**\n" \
                f"```sql\n{safe_sql}\n```\n\n" \
                f"**Results:** ({len(rows)} rows)\n\n" \
                f"{table}"
            )
            formatted_response = format_reply_with_model(response_text)
            
            # Remember SQL exchange in conversation history
            remember_exchange(request.message, formatted_response)
            
            return {
                "reply": formatted_response,
                "source": "SQL Query",
                "sql": safe_sql,
                "row_count": len(rows),
                "rows": rows[:10]  # Limit to first 10 for response size
            }
        except Exception as e:
            print(f"SQL Query Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "error": f"SQL query failed: {str(e)}",
                "source": "SQL Error"
            }

    # --- VIDEO MODE ---
    if request.use_video:
        context, search_method = retrieve_context(request.message, video_collection, n_results=5)

        if not context:
            return {
                "reply": "Question is not in this video (no matching transcript context).",
                "source": "Video Context"
            }

        system_instr = "Answer the question ONLY based on the video transcript provided. If not found, say question is not related to the video."
        user_msg = f"Video Transcript Context:\n{context}\n\nQuestion: {request.message}"

        messages = build_messages_with_context(user_msg, system_instr)
        ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
        formatted_reply = format_reply_with_model(ollama_reply)
        remember_exchange(request.message, formatted_reply)
        return {"reply": formatted_reply, "source": "Ollama (Video)", "search_method": search_method}

    # --- AUDIO MODE ---
    if request.use_audio:
        context, search_method = retrieve_context(request.message, audio_collection, n_results=5)

        if not context:
            return {
                "reply": "Question is not in this audio (no matching transcript context).",
                "source": "Audio Context"
            }

        system_instr = "Answer the question ONLY based on the audio transcript provided. If not found, say question is not related to the audio."
        user_msg = f"Audio Transcript Context:\n{context}\n\nQuestion: {request.message}"

        try:
            messages = build_messages_with_context(user_msg, system_instr)
            ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
            formatted_reply = format_reply_with_model(ollama_reply)
            remember_exchange(request.message, formatted_reply)
            return {"reply": formatted_reply, "source": "Ollama (Audio)", "search_method": search_method}
        except Exception as e:
            return {"error": f"Audio query failed: {str(e)}"}

    # --- PDF MODE ---
    if request.use_pdf:
        context, search_method = retrieve_context(request.message, pdf_collection, n_results=5)

        if not context:
            return {
                "reply": "Question is not in this PDF (no matching document context).",
                "source": "PDF Context"
            }

        system_instr = "Answer the question ONLY based on the PDF document provided. If not found, say question is not related to the PDF."
        user_msg = f"PDF Context:\n{context}\n\nQuestion: {request.message}"

        try:
            messages = build_messages_with_context(user_msg, system_instr)
            ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
            formatted_reply = format_reply_with_model(ollama_reply)
            remember_exchange(request.message, formatted_reply)
            return {"reply": formatted_reply, "source": "Ollama (PDF)", "search_method": search_method}
        except Exception as e:
            return {"error": f"PDF query failed: {str(e)}"}

    # --- IMAGE MODE ---
    if request.use_image:
        print(f"\n🖼️  IMAGE MODE QUERY: {request.message}")
        context, search_method = retrieve_context(request.message, image_collection, n_results=5)

        if not context or not str(context).strip():
            print("⚠️  No context retrieved from image collection")
            return {
                "reply": "No image has been uploaded yet, or the question cannot be matched with the uploaded image content. Please upload an image first.",
                "source": "Image Context"
            }

        system_instr = "Answer the question based on the image description and extracted text provided below. Be helpful and informative."
        user_msg = f"Image Context:\n{context}\n\nQuestion: {request.message}"

        try:
            messages = build_messages_with_context(user_msg, system_instr)
            ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
            formatted_reply = format_reply_with_model(ollama_reply)
            remember_exchange(request.message, formatted_reply)
            return {"reply": formatted_reply, "source": "Ollama (Image)", "search_method": search_method}
        except Exception as e:
            return {"error": f"Image query failed: {str(e)}"}

    # --- NORMAL MODE: Ollama only ---
    try:
        # Build messages with recent context window
        messages = build_messages_with_context(request.message, system_prompt="Use the following recent conversation for context and coherence.")
        ollama_reply = send_ollama_chat(ollama_client, OLLAMA_MODEL, messages)
        formatted_reply = format_reply_with_model(ollama_reply)
        remember_exchange(request.message, formatted_reply)
        return {"reply": formatted_reply, "source": "Ollama"}

    except Exception as ollama_err:
        return {"error": f"Ollama request failed: {str(ollama_err)}"}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses in real-time using Server-Sent Events"""
    if not request.message.strip():
        return {"error": "Message cannot be empty"}

    def generate():
        try:
            # SQL MODE (no streaming for SQL)
            if request.use_sql:
                try:
                    schema = get_db_schema(DB_PATH)
                    if "No database found" in schema or "Database is empty" in schema:
                        yield f"data: {json.dumps({'content': 'No SQL database available. Please upload a CSV file first.', 'done': True})}\n\n"
                        return
                    
                    messages_with_context = build_messages_with_context(request.message)
                    raw_sql = generate_sql_with_llm(
                        request.message, 
                        schema,
                        ollama_client,
                        OLLAMA_MODEL,
                        messages_with_context
                    )
                    
                    fixed_sql = repair_sql(raw_sql)
                    safe_sql = enforce_sql_safety(fixed_sql)
                    rows = execute_sql(safe_sql, DB_PATH)
                    
                    table = format_sql_table(rows)
                    
                    response_text = (
                        f"**SQL Query:**\n"
                        f"```sql\n{safe_sql}\n```\n\n"
                        f"**Results:** ({len(rows)} rows)\n\n"
                        f"{table}"
                    )
                    formatted_response = format_reply_with_model(response_text)
                    remember_exchange(request.message, formatted_response)
                    
                    yield f"data: {json.dumps({'content': formatted_response, 'done': True})}\n\n"
                    return
                    
                except Exception as sql_error:
                    yield f"data: {json.dumps({'content': f'SQL Error: {str(sql_error)}', 'done': True})}\n\n"
                    return

            # Determine context and system instruction
            context = None
            system_instr = None
            
            if request.use_video:
                context, search_method = retrieve_context(request.message, video_collection, n_results=5)
                if context:
                    system_instr = "Answer the question based on the video transcript provided. If the answer is not in the transcript, use your own knowledge to provide a helpful response."
                    user_msg = f"Video Transcript Context:\n{context}\n\nQuestion: {request.message}"
                else:
                    system_instr = "Answer the question using your own knowledge. No video context is available."
                    user_msg = request.message
            
            elif request.use_audio:
                context, search_method = retrieve_context(request.message, audio_collection, n_results=5)
                if context:
                    system_instr = "Answer the question based on the audio transcript provided. If the answer is not in the transcript, use your own knowledge to provide a helpful response."
                    user_msg = f"Audio Transcript Context:\n{context}\n\nQuestion: {request.message}"
                else:
                    system_instr = "Answer the question using your own knowledge. No audio context is available."
                    user_msg = request.message
            
            elif request.use_pdf:
                context, search_method = retrieve_context(request.message, pdf_collection, n_results=5)
                if context:
                    system_instr = "Answer the question based on the PDF document provided. If the answer is not in the document, use your own knowledge to provide a helpful response."
                    user_msg = f"PDF Context:\n{context}\n\nQuestion: {request.message}"
                else:
                    system_instr = "Answer the question using your own knowledge. No PDF context is available."
                    user_msg = request.message
            
            elif request.use_image:
                context, search_method = retrieve_context(request.message, image_collection, n_results=5)
                if context and str(context).strip():
                    system_instr = "Answer the question based on the image description and extracted text provided. If the answer is not in the image, use your own knowledge to provide a helpful response."
                    user_msg = f"Image Context:\n{context}\n\nQuestion: {request.message}"
                else:
                    system_instr = "Answer the question using your own knowledge. No image context is available."
                    user_msg = request.message
            
            else:
                # Normal mode - use model's own knowledge
                system_instr = "You are a helpful AI assistant. Answer questions accurately and concisely. Provide clear explanations with examples when relevant."
                user_msg = request.message
            
            # Build messages and stream response
            messages = build_messages_with_context(user_msg, system_instr)
            
            # Send model name header with clear line break
            yield f"data: {{\"content\": \"**Model:** {OLLAMA_MODEL}\\n\\n\", \"done\": false}}\n\n"
            
            full_response = ""
            for chunk in send_ollama_chat_stream(ollama_client, OLLAMA_MODEL, messages):
                full_response += chunk
                # Escape for JSON and send
                escaped_chunk = chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                yield f"data: {{\"content\": \"{escaped_chunk}\", \"done\": false}}\n\n"
            
            # Remember exchange (without model name since it's already sent)
            remember_exchange(request.message, full_response)
            
            yield f"data: {{\"done\": true}}\n\n"
            
        except Exception as e:
            error_msg = f"Error: {str(e)}".replace('"', '\\"')
            yield f"data: {{\"content\": \"{error_msg}\", \"done\": true}}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)