# Multimodal RAG Chatbot (Offline Version)

A fully **offline-capable** Retrieval-Augmented Generation (RAG) chatbot supporting multiple file formats with local AI models.

## ✨ Features

### **Supported File Types**
- 📄 **PDF** - Text extraction with 3-tier fallback (PyMuPDF → pdfplumber → PyPDF)
- 🖼️ **Images** - Vision model analysis using Ollama's moondream:1.8b (describes images + extracts text)
- 🎥 **Video** - Audio transcription with Whisper + FAISS storage
- 🎵 **Audio** - Direct transcription with Whisper model
- 📊 **CSV** - SQL table creation for natural language queries

### **AI Capabilities**
- 🤖 **Local LLM** - Ollama models (no API keys required)
- 🔍 **Semantic Search** - FAISS vector store with SentenceTransformers
- 🎨 **Vision Understanding** - Moondream VLM for image analysis
- 🗣️ **Voice Input** - Real-time audio recording and transcription
- 💬 **Context-Aware Chat** - Maintains conversation history
- 📈 **SQL Queries** - Natural language to SQL conversion

### **Architecture**
- **Backend**: FastAPI with modular structure (handlers/services)
- **Frontend**: Streamlit with dark theme UI
- **Storage**: FAISS for embeddings, SQLite for CSV data
- **Models**: All run locally (Whisper, SentenceTransformers, Ollama)

---

## 🚀 Quick Start

### **Prerequisites**

1. **Python 3.8+**
2. **Ollama** - Download from https://ollama.ai
3. **FFmpeg** - For audio/video processing
   ```powershell
   # Windows (with Chocolatey)
   choco install ffmpeg
   
   # Or download from: https://ffmpeg.org/download.html
   ```

### **Installation**

#### **Step 1: Install Dependencies**
```powershell
cd offline
pip install -r requirements.txt
```

#### **Step 2: Pull Ollama Models**
```powershell
# Text generation model (required)
ollama pull gemma2:2b

# Vision model for images (required for image uploads)
ollama pull moondream:1.8b
```
**Note:** Models will auto-download on first use if not present.

#### **Step 3: Configure Environment**
Edit `backend/.env` to set your preferred model:
```env
OLLAMA_MODEL=gemma2:2b
OLLAMA_VLM_MODEL = moondream:1.8b
```

---

## 🎯 Running the Application

### **Start Backend**
```powershell
cd offline/backend
python main.py
```
Backend runs at `http://localhost:8001`

### **Start Frontend** (in separate terminal)
```powershell
cd offline/frontend
streamlit run app.py
```
Frontend opens at `http://localhost:8501`

---

## 📁 Project Structure

```
offline/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration & initialization
│   ├── .env                 # Environment variables
│   │
│   ├── handlers/            # File processors
│   │   ├── audio_handler.py
│   │   ├── video_handler.py
│   │   ├── image_handler.py # Uses moondream vision model
│   │   ├── pdf_handler.py
│   │   └── csv_handler.py
│   │
│   ├── services/            # Shared utilities
│   │   ├── embeddings.py    # FAISS storage & retrieval
│   │   ├── faiss_store.py   # Vector store implementation
│   │   ├── database.py      # SQL utilities
│   │   ├── llm_clients.py   # Ollama integration + vision
│   │   └── whisper_manager.py
│   │
│   ├── faiss_db/            # Persisted FAISS indices
│   ├── cache/               # Model cache
│   └── sql_data.db          # SQLite database (created on CSV upload)
│
├── frontend/
│   └── app.py               # Streamlit UI
│
└── requirements.txt         # Python dependencies
```

---

## 🔧 Usage Guide

### **1. Upload Files**
Use the sidebar to upload:
- PDF documents
- Images (screenshots, photos, diagrams)
- Audio/Video files
- CSV files for SQL queries

### **2. Select Query Mode**
- **Normal Query** - Chat with LLM using general knowledge
- **📄 PDF Context** - Query uploaded PDF content
- **🖼️ Image Context** - Ask about uploaded images
- **🎥 Video Context** - Query video transcripts
- **🎵 Audio Context** - Query audio transcripts
- **📊 SQL Query** - Natural language queries on CSV data

### **3. Ask Questions**
Type or use 🎤 voice input:
- "What's in the image?"
- "Summarize the PDF"
- "Show me all students with marks > 90"
- "What was discussed in the video?"

---

## 🛠️ Testing & Verification

### **Test SQL Functionality**
```powershell
cd offline/backend
python test_sql.py
```

### **Test Vision Model**
```powershell
cd offline/backend
python test_vision.py
```

---

## 📝 Key Improvements

1. **Vision Model Integration** - Replaced EasyOCR with Ollama moondream:1.8b
   - Better image understanding + text extraction
   - No separate OCR dependencies

2. **Cleaner Structure** - Reorganized into handlers/services packages
   - Easier to maintain and scale

3. **Optimized Logging** - Removed verbose debug logs
   - Production-ready output

4. **SQL Safety** - Enhanced query validation
   - Auto-limit enforcement

5. **Auto Model Download** - Vision and text models auto-pull if missing

---

## ⚙️ Configuration

**Environment Variables** (`backend/.env`):
```env
OLLAMA_MODEL=mistral              # Text generation model
OLLAMA_URL=http://localhost:11434
MAX_PDF_SIZE_MB=30                # PDF upload limit
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Ollama connection error | Run `ollama serve` in terminal |
| Model not found | Run `ollama pull mistral` and `ollama pull moondream:1.8b` |
| FFmpeg not found | Install FFmpeg and add to PATH |
| Vision model fails | Ensure moondream:1.8b is pulled |
| SQL errors | Upload CSV file first, select SQL Query mode |

---

## 🎓 Models Used

1. **Text Generation**: `mistral` (~4.1GB) or `tinyllama:1.1b` (~637MB)
2. **Vision**: `moondream:1.8b` (~1.1GB) - Auto-downloads on first image upload
3. **Audio**: `whisper tiny` (~75MB)
4. **Embeddings**: `all-MiniLM-L6-v2` (~80MB)

---

**Fully Offline** ✅ | **No API Keys** ✅ | **Local Models** ✅ | **Privacy First** ✅

- pip
- Optional but recommended: virtual environment (venv)

## Repository Structure

- `backend/` — Python backend (contains `main.py`)
- `frontend/` — Streamlit frontend (contains `app.py`)
- `README.md` — This file

## Environment variables (.env)

This project expects a `.env` file (or environment variables set in your environment) to supply local model configurations. Do NOT commit your real `.env` file to source control.

Recommended variables (example template):

```env
# Local Ollama
OLLAMA_MODEL=tinyllama:1.1b
OLLAMA_URL=http://localhost:11434
```

Notes:
- Fill the values on the right of `=` with your actual API keys or model names.
- `OLLAMA_URL` defaults to `http://localhost:11434` for a local Ollama server.
- Keep secrets out of git. Instead, add a `.env.example` (with empty values or placeholders) and add `.env` to `.gitignore`.

Suggested steps to add env files safely:

1. Create a local `.env` (from the repository root):
   ```bash
   cat > .env <<'EOT'
   OLLAMA_MODEL=tinyllama:1.1b
   OLLAMA_URL=http://localhost:11434
   EOT
   ```

2. Add `.env` to `.gitignore` (if not already present):
   ```bash
   echo ".env" >> .gitignore
   git add .gitignore
   git commit -m "Ignore local .env files"
   ```

3. Add a `.env.example` to the repo (safe to commit) so contributors know which variables are required:
   ```env
   # .env.example
   OLLAMA_MODEL=tinyllama:1.1b
   OLLAMA_URL=http://localhost:11434
   ```

4. In CI / production, set environment variables securely through your platform (GitHub Actions secrets, Docker secrets, host environment, etc.) rather than a committed file.

## Setup & Run

Follow these steps to run the project locally.

### Backend

1. Create and activate a virtual environment:
   - macOS / Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure local dependencies are installed:
   - Install Ollama and pull the model:
     ```bash
     ollama pull tinyllama:1.1b
     ```
   - Install Tesseract OCR and ensure `tesseract` is on your PATH.
   - Install FFmpeg so `faster-whisper` can decode audio/video files.

4. Open a terminal and change into the backend directory:
   ```bash
   cd backend
   ```

5. Run the backend server:
   ```bash
   python main.py
   ```
   The backend should start and listen on the configured host/port (check `main.py` for defaults).

### Frontend

1. Open a new terminal and change into the frontend directory:
   ```bash
   cd frontend
   ```

2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   This will open the frontend in your browser (typically at `http://localhost:8501`).

## Usage

- Ensure API keys and model variables are set in your `.env` or environment before running; see the Environment variables (.env) section.
- Start the backend first, then the frontend.
- Use the Streamlit interface to interact with the RAG chatbot. The frontend will communicate with the backend API endpoints defined in `backend/main.py`.