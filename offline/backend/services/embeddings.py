"""
Embedding and Context Retrieval Module
Handles text chunking and context retrieval from ChromaDB collections
"""

import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


def get_text_splitter(chunk_size: int, chunk_overlap: int):
    method = os.getenv("CHUNKING_METHOD", "token").strip().lower()
    if method == "token":
        try:
            return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except Exception:
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def chunk_and_store(text: str, collection, source: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Chunk text and store in FAISS collection.
    Checks for duplicate sources before storing.
    
    Args:
        text: Text to chunk
        collection: FAISS collection to store in
        source: Source identifier (e.g., filename) to check for duplicates
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        Tuple of (chunks_created, already_exists)
        - chunks_created: Number of chunks created (0 if duplicate)
        - already_exists: Boolean indicating if content already exists
    """
    print(f"\n📝 Processing text from source: {source}")
    print(f"   Text length: {len(text)} characters")
    
    # Check if this source already exists in the collection
    if source:
        try:
            existing = collection.get(where={"source": source})
            if existing and len(existing.get('ids', [])) > 0:
                return 0, True  # Already exists, return 0 chunks and True flag
        except Exception as e:
            # If collection is empty or error occurs, proceed with storing
            pass
    
    text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]
    
    # Prepare metadata with source information
    metadatas = [{"source": source} if source else {} for _ in chunks]
    
    # Store in FAISS - this will generate embeddings and persist
    try:
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    except Exception as e:
        raise
        raise
    
    return len(chunks), False


def retrieve_context(query: str, collection, n_results: int = 5, fallback_text: str = None):
    """
    Retrieve relevant context from FAISS collection using hybrid search.
    First tries semantic search, then falls back to keyword search if needed.
    
    Inspired by: https://github.com/anurag0802/rag-based-voice-enabled-Chatbot/blob/main/backend/rag_pdf_backend/rag.py
    
    Args:
        query: Query text
        collection: FAISS collection to search
        n_results: Number of results to retrieve
        fallback_text: Optional full text for keyword fallback search
    
    Returns:
        Tuple of (context_string, search_method)
        - context_string: Concatenated context from search
        - search_method: "semantic" or "keyword" indicating which method was used
    """
    try:
        # Check if collection has any documents
        count = collection.count()
        
        if count == 0:
            return "", "none"
        
        # 1️⃣ PRIMARY: Semantic search with relevance threshold
        results = collection.query(query_texts=[query], n_results=min(n_results * 2, count))
        documents = results.get("documents", [[]])[0]
        
        # Filter by relevance - if we got results, use them
        if documents and len(documents) > 0:
            # Take top N results
            context_docs = documents[:n_results]
            context = "\n\n".join(context_docs)
            return context, "semantic"
        
        # 2️⃣ FALLBACK: Keyword search (if provided and semantic found nothing)
        if fallback_text:
            keyword_context = keyword_search(fallback_text, query, top_k=n_results)
            if keyword_context:
                return keyword_context, "keyword"
        
        return "", "none"
    
    except Exception as e:
        return "", "error"


def keyword_search(full_text: str, query: str, top_k: int = 3) -> str:
    """
    Simple keyword-based search as fallback when semantic search fails.
    Searches for lines containing query keywords.
    
    Args:
        full_text: Full text to search in
        query: Query string
        top_k: Number of top matching lines to return
    
    Returns:
        Concatenated matching text
    """
    try:
        query_lower = query.lower()
        lines = full_text.split('\n')
        
        # Score lines by keyword matches
        scored_lines = []
        for line in lines:
            if len(line.strip()) > 0:
                # Count keyword occurrences
                score = sum(1 for word in query_lower.split() if word in line.lower())
                if score > 0:
                    scored_lines.append((score, line))
        
        if scored_lines:
            # Sort by score (descending) and take top K
            scored_lines.sort(key=lambda x: x[0], reverse=True)
            matched = [line for _, line in scored_lines[:top_k]]
            return "\n".join(matched)
        
        return ""
    except Exception as e:
        return ""
