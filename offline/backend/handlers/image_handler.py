"""
Image Handler Module
Handles image upload, vision model analysis, and FAISS storage.
Uses Ollama's moondream:1.8b vision model for image description and text extraction.
"""

import io
import numpy as np
from PIL import Image

from services.embeddings import chunk_and_store
from services.llm_clients import ensure_ollama_model, send_ollama_vision


async def process_image(
    file_bytes: bytes,
    filename: str,
    image_collection,
    ollama_client=None,
    vision_model: str = "moondream:1.8b"
):
    """
    Process uploaded image: analyze with Ollama vision model, chunk, and store.
    
    Args:
        file_bytes: Raw image file bytes
        filename: Original filename
        image_collection: ChromaDB collection for image storage
        ollama_client: Ollama client instance
        vision_model: Vision model to use (default: moondream:1.8b)
    Returns:
        dict: Status with message and chunk count
        
    Raises:
        ValueError: If vision model analysis fails
    """
    # Open image to verify it's valid
    img = Image.open(io.BytesIO(file_bytes))
    
    # Ensure vision model is available
    if ollama_client:
        model_ready = ensure_ollama_model(ollama_client, vision_model)
        if not model_ready:
            raise ValueError(f"Vision model '{vision_model}' is not available. Please pull it with: ollama pull {vision_model}")
    else:
        raise ValueError("Ollama client not available for image analysis")
    
    # Analyze image with vision model
    print(f"🤖 Analyzing image with {vision_model}...")
    prompt = (
        "Describe this image in detail. "
        "Extract text, layout, objects, and data."
    )
    
    try:
        vision_response = send_ollama_vision(
            ollama_client,
            vision_model,
            prompt,
            file_bytes
        )
        print(f"✅ Vision analysis complete. Response length: {len(vision_response)} characters")
    except Exception as e:
        print(f"❌ Vision analysis failed: {str(e)}")
        raise ValueError(f"Failed to analyze image with vision model: {str(e)}")
    
    if not vision_response.strip():
        raise ValueError("Vision model returned empty response")
    
    # Add metadata to the vision response
    full_text = f"Image: {filename}\n\nAnalysis:\n{vision_response}"
    
    # Chunk and store using reusable utility
    chunk_count, already_exists = chunk_and_store(
        full_text, 
        image_collection, 
        source=filename, 
        chunk_size=1000, 
        chunk_overlap=200
    )
    
    if already_exists:
        return {
            "status": "success",
            "message": f"Image '{filename}' already exists in database. Skipped processing.",
            "duplicate": True
        }
    
    return {
        "status": "success",
        "message": f"Processed {chunk_count} chunks from image using {vision_model}.",
        "duplicate": False,
        "analysis_preview": vision_response[:200] + "..." if len(vision_response) > 200 else vision_response
    }
