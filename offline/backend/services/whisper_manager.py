"""
Whisper Model Manager
Handles caching and loading of the Whisper model to avoid reloading on every call
"""

import whisper
import os

# Global model cache
_whisper_model = None
MODEL_NAME = "tiny"


def get_whisper_model():
    """
    Get or load the cached Whisper model.
    Loads the model once and reuses it for subsequent calls.
    
    Returns:
        Whisper model instance
    """
    global _whisper_model
    
    if _whisper_model is None:
        print(f"📥 Loading Whisper {MODEL_NAME} model (this may take a moment)...")
        _whisper_model = whisper.load_model(MODEL_NAME)
        print(f"✅ Whisper model loaded and cached!")
    
    return _whisper_model


def transcribe_audio(file_path: str, language: str = "en") -> str:
    """
    Transcribe audio/video file using cached Whisper model.
    
    Args:
        file_path: Path to audio or video file
        language: Language code (default: "en" for English)
    
    Returns:
        Transcribed text
    """
    model = get_whisper_model()
    
    # Ensure file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"🎙️  Transcribing: {os.path.basename(file_path)}...")
    print(f"   File size: {os.path.getsize(file_path)} bytes")
    
    # Transcribe with fp16=False to avoid FP16 warnings and compatibility issues
    try:
        result = model.transcribe(
            file_path, 
            language=language,
            fp16=False,  # Use FP32 on CPU
            verbose=False  # Reduce logging noise
        )
        return result["text"].strip()
    except Exception as e:
        print(f"   Error during transcription: {e}")
        raise
