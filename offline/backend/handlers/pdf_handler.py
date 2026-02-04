"""
PDF Handler Module
Handles PDF upload, text extraction, table detection, and FAISS storage.
Uses PyMuPDF + PyPDF + pdfplumber for reliable PDF processing.
"""

import os
import tempfile

from services.embeddings import chunk_and_store


async def process_pdf(
    file_bytes: bytes,
    filename: str,
    pdf_collection,
    max_pdf_size_mb: int = 10
):
    """
    Process uploaded PDF file using PyMuPDF + pdfplumber for reliable extraction.
    
    Args:
        file_bytes: Raw PDF file bytes
        filename: Original filename
        pdf_collection: ChromaDB collection for PDF storage
        max_pdf_size_mb: Maximum allowed PDF size in MB
        
    Returns:
        dict: Status with message, chunk count, and pages processed
        
    Raises:
        ValueError: If PDF is too large or no text extracted
    """
    # Enforce upload size limit
    if len(file_bytes) > max_pdf_size_mb * 1024 * 1024:
        raise ValueError(f"PDF too large (> {max_pdf_size_mb} MB)")
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        print(f"📄 Processing PDF: {filename}")
        
        import fitz  # PyMuPDF
        import pdfplumber
        
        # Method 1: Try PyMuPDF first (best for text PDFs)
        print("🔍 Extracting text with PyMuPDF...")
        pdf_text = ""
        page_count = 0
        
        try:
            doc = fitz.open(tmp_path)
            page_count = len(doc)
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                if text.strip():
                    pdf_text += f"\n--- Page {page_num} ---\n" + text
            
            doc.close()
        except Exception as e:
            print(f"⚠️  PyMuPDF extraction failed: {e}")
        
        # Method 2: If PyMuPDF got nothing, try pdfplumber
        if not pdf_text.strip():
            print("🔄 Retrying with pdfplumber...")
            try:
                with pdfplumber.open(tmp_path) as pdf:
                    page_count = len(pdf.pages)
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text and text.strip():
                            pdf_text += f"\n--- Page {page_num} ---\n" + text
            except Exception as e:
                print(f"⚠️  pdfplumber extraction failed: {e}")
        
        # Method 3: If still no text, try PyPDF as fallback
        if not pdf_text.strip():
            print("🔄 Trying PyPDF fallback...")
            try:
                from pypdf import PdfReader
                reader = PdfReader(tmp_path)
                page_count = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        pdf_text += f"\n--- Page {page_num} ---\n" + text
            except Exception as e:
                print(f"⚠️  PyPDF extraction failed: {e}")
        
        if not pdf_text.strip():
            raise ValueError("No text could be extracted from PDF using any method")
        
        print(f"✅ Extracted {len(pdf_text)} characters from {page_count} page(s)")
        
        # Try to extract tables with pdfplumber
        table_text = ""
        try:
            print("📊 Attempting to extract tables...")
            with pdfplumber.open(tmp_path) as pdf:
                table_count = 0
                for page_num, page in enumerate(pdf.pages, 1):
                    if page.tables:
                        for table_idx, table in enumerate(page.tables, 1):
                            table_count += 1
                            df = page.extract_table(table.bbox)
                            if df:
                                table_text += f"\n--- Table {table_idx} on Page {page_num} ---\n"
                                table_text += str(df)
                
                if table_count > 0:
                    print(f"✅ Extracted {table_count} table(s)")
        except Exception as e:
            print(f"ℹ️  Table extraction not available: {e}")
        
        # Combine text and tables
        full_text = pdf_text
        if table_text:
            full_text += "\n\n" + table_text
        
        # Chunk and store
        chunk_count, already_exists = chunk_and_store(
            full_text,
            pdf_collection,
            source=filename,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        if already_exists:
            return {
                "status": "success",
                "message": f"PDF '{filename}' already exists in database. Skipped processing.",
                "duplicate": True
            }
        
        return {
            "status": "success",
            "message": f"Processed {chunk_count} chunks from {page_count} page(s).",
            "pages_processed": page_count,
            "duplicate": False
        }
    
    except Exception as e:
        print(f"⚠️  PDF processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Failed to process PDF: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
