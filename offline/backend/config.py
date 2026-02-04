"""
Backend Configuration and Initialization Module
Environment variables, API clients, and database setup
"""

import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ollama import Client as OllamaClient
from collections import deque
from services.faiss_store import FAISSCollection

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir, ".env")
load_dotenv(dotenv_path)

# Configuration constants
MAX_SQL_LIMIT = 100
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "30"))
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_DIR = os.path.join(script_dir, "faiss_db")
CACHE_DIR = os.path.join(script_dir, "cache")
MODEL_CACHE_PATH = os.path.join(CACHE_DIR, "embedding_model.pkl")

# Conversation memory: last 10 messages (5 user + 5 assistant)
chat_memory = deque(maxlen=10)


def load_embedding_model():
    """Load embedding model from cache or download if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_CACHE_PATH):
        try:
            with open(MODEL_CACHE_PATH, 'rb') as f:
                embed_fn = pickle.load(f)
            return embed_fn
        except Exception as e:
            pass
    
    try:
        # Try loading from local cache first (offline mode)
        embed_fn = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception as e:
        try:
            # Allow one-time download if not cached
            embed_fn = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as download_err:
            print(f"❌ Failed to download model: {download_err}")
            print(f"💡 Manual fix: Run this while online:")
            print(f"   python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('{EMBEDDING_MODEL_NAME}')\"")
            raise RuntimeError(f"Embedding model '{EMBEDDING_MODEL_NAME}' not available offline and download failed")
    
    try:
        with open(MODEL_CACHE_PATH, 'wb') as f:
            pickle.dump(embed_fn, f)
    except Exception as e:
        pass
    
    return embed_fn


def ensure_ollama_server_running():
    """
    Ensure Ollama server is running by executing 'ollama list'.
    This command will start the Ollama server if it's not already running.
    """
    import subprocess
    
    try:
        # Run 'ollama list' which will start the server if not running
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )
    except FileNotFoundError:
        print("⚠️  Ollama not found. Please install Ollama from https://ollama.ai")
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass


def initialize_backend():
    """Initialize all backend components."""
    
    # Setup directories
    os.makedirs(FAISS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load embedding model
    embed_fn = load_embedding_model()
    
    # Initialize FAISS collections
    collections = {}
    collection_names = {
        "video": "chat_video_context",
        "audio": "chat_audio_context",
        "pdf": "chat_pdf_context",
        "image": "chat_image_context"
    }
    
    for col_type, col_name in collection_names.items():
        try:
            collections[col_type] = FAISSCollection(
                name=col_name,
                persist_dir=FAISS_DIR,
                embedding_model=embed_fn
            )
        except Exception as e:
            pass
    
    # Setup Ollama (local)
    # Try to start Ollama server if not running
    ensure_ollama_server_running()
    
    ollama_client = None
    ollama_model = os.getenv("OLLAMA_MODEL", "tinyllama:1.1b")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    if OLLAMA_URL:
        try:
            headers = {'Authorization': f'Bearer {OLLAMA_API_KEY}'} if OLLAMA_API_KEY else None
            ollama_client = OllamaClient(host=OLLAMA_URL, headers=headers)
        except Exception as e:
            pass
    
    # Setup database
    db_path = Path(script_dir) / "sql_data.db"
    
    return {
        "collections": collections,
        "ollama_client": ollama_client,
        "ollama_model": ollama_model,
        "db_path": db_path,
    }


def get_context_window():
    """Get conversation history."""
    return list(chat_memory)


def remember_exchange(user_text: str, assistant_text: str):
    """Store conversation exchange in memory."""
    try:
        chat_memory.append({"role": "user", "content": user_text})
        chat_memory.append({"role": "assistant", "content": assistant_text})
    except Exception:
        pass


def build_messages_with_context(user_message: str, system_prompt: str | None = None):
    """Build message list with conversation context."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in get_context_window():
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages
