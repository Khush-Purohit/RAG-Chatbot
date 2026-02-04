"""
LLM Client Interaction Module
Handles communication with local Ollama API
"""


def ensure_ollama_model(ollama_client, model_name: str) -> bool:
    """
    Check if Ollama model exists, pull it if not.
    
    Args:
        ollama_client: Ollama client instance
        model_name: Model name to check/pull
    
    Returns:
        True if model is ready, False otherwise
    """
    try:
        if not ollama_client:
            return False
        
        # Try to list available models
        try:
            models_response = ollama_client.list()
            
            # Handle both dict and list responses
            models_list = []
            if isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            elif isinstance(models_response, list):
                models_list = models_response
            
            # Extract model names safely
            model_names = []
            for m in models_list:
                if isinstance(m, dict):
                    name = m.get('name', '')
                else:
                    name = str(m)
                if name:
                    model_names.append(name)
            
            if model_names:
                # Check if model exists
                if any(model_name in name or name in model_name for name in model_names):
                    return True
        except Exception as list_err:
            pass
        
        # Try to use the model directly (most reliable check)
        try:
            response = ollama_client.chat(
                model=model_name, 
                messages=[{"role": "user", "content": "test"}],
                stream=False
            )
            return True
        except Exception as chat_err:
            error_msg = str(chat_err)
            # If model not found, try pulling
            if "not found" in error_msg.lower() or "pull" in error_msg.lower():
                try:
                    ollama_client.pull(model_name)
                    return True
                except Exception as pull_err:
                    print(f"❌ Failed to pull model '{model_name}': {str(pull_err)}")
                    return False
            else:
                return False
        
    except Exception as e:
        return False


def send_ollama_vision(ollama_client, model: str, prompt: str, image_bytes: bytes) -> str:
    """
    Send vision request to Ollama API with image.
    
    Args:
        ollama_client: Ollama client instance
        model: Vision model name (e.g., 'moondream:1.8b')
        prompt: Text prompt for the image
        image_bytes: Image data as bytes
    
    Returns:
        Response text describing the image
    """
    try:
        import base64
        
        # Encode image to base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Send request to Ollama with image
        response = ollama_client.chat(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_b64]
                }
            ]
        )
        
        # Extract message content - handle different response formats
        if isinstance(response, dict):
            # Try to get message content
            message = response.get('message', {})
            if isinstance(message, dict):
                content = message.get('content', '')
                if content:
                    return content.strip()
            
            # Fallback: try direct content key
            if 'content' in response:
                content = response['content']
                if content:
                    return str(content).strip()
        
        # If response is a string or other type
        response_str = str(response)
        
        # Clean up metadata from response if present
        if 'model=' in response_str and 'content=' not in response_str:
            raise ValueError("Vision model returned metadata without content. Response may be incomplete.")
        
        return response_str.strip()
        
    except Exception as e:
        raise ValueError(f"Vision model failed: {str(e)}")


def send_ollama_chat(ollama_client, ollama_model: str, messages: list[dict]) -> str:
    """
    Send chat request to Ollama API.
    
    Args:
        ollama_client: Ollama client instance
        ollama_model: Model name
        messages: List of message dictionaries
    
    Returns:
        Response text
    """
    try:
        if not ollama_client:
            return "Ollama Fallback Error: Client unavailable"
        
        # Ensure model is available
        ensure_ollama_model(ollama_client, ollama_model)
        
        response = ollama_client.chat(model=ollama_model, messages=messages, stream=False)
        return response['message']['content']
    except Exception as e:
        return f"Ollama Fallback Error: {str(e)}"


def send_ollama_chat_stream(ollama_client, ollama_model: str, messages: list[dict]):
    """
    Send streaming chat request to Ollama API.
    
    Args:
        ollama_client: Ollama client instance
        ollama_model: Model name
        messages: List of message dictionaries
    
    Yields:
        Response chunks as strings
    """
    try:
        if not ollama_client:
            yield "Ollama Error: Client unavailable"
            return
        
        # Ensure model is available
        if not ensure_ollama_model(ollama_client, ollama_model):
            yield f"Ollama Error: Could not load model '{ollama_model}'"
            return
        
        stream = ollama_client.chat(model=ollama_model, messages=messages, stream=True)
        
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content
                
    except Exception as e:
        yield f"\n\nOllama Error: {str(e)}"
