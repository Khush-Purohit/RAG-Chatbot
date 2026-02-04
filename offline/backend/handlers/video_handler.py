"""
Video Processing Module
Handles video transcription using Whisper and storage
"""

import os
import tempfile
from services.embeddings import chunk_and_store
from services.whisper_manager import transcribe_audio


async def process_video(file_bytes: bytes, filename: str, video_collection):
    """
    Transcribe video using Whisper (tiny model) and store chunks in FAISS.
    
    Args:
        file_bytes: Video file bytes
        filename: Name of the video file
        video_collection: FAISS video collection
    
    Returns:
        Dict with status and message
    """
    suffix = os.path.splitext(filename)[1] or ".mp4"
    tmp_path = None
    
    try:
        # Create a temporary file with unique name
        temp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(temp_dir, f"video_{os.urandom(8).hex()}{suffix}")
        
        # Write file bytes to temporary location
        with open(tmp_path, 'wb') as f:
            f.write(file_bytes)
        
        print(f"📁 Video saved to: {tmp_path}")
        
        print("🎬 Transcribing video with Whisper...")
        transcript = transcribe_audio(tmp_path, language="en")
        print("✅ Transcribed video with Whisper")
        
        if not transcript:
            return {
                "status": "error",
                "message": "Could not transcribe video. Ensure audio content exists and try uploading a different video.",
                "duplicate": False
            }
        
        if not transcript.strip():
            return {
                "status": "error",
                "message": "No speech detected in video file",
                "duplicate": False
            }
        
        # Chunk and store in FAISS
        chunk_count, already_exists = chunk_and_store(transcript, video_collection, source=filename)
        
        if already_exists:
            return {
                "status": "success",
                "message": f"Video '{filename}' already exists in database. Skipped processing.",
                "duplicate": True
            }
        
        return {
            "status": "success",
            "message": f"Processed {chunk_count} chunks from video.",
            "duplicate": False
        }
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Safely remove temporary files
        for path in [tmp_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    import time
                    time.sleep(0.2)
                    try:
                        os.remove(path)
                    except OSError as cleanup_error:
                        print(f"Warning: Could not remove temp file {path}: {cleanup_error}")
