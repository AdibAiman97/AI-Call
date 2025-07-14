from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create router instance
router = APIRouter(
    prefix="/pdf",
    tags=["PDF Upload"]
)

# We'll get the RAG system from the main app
async def get_rag_system():
    """Dependency to get the RAG system from main app"""
    from main import rag_sys
    if not rag_sys:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_sys

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    rag_system = Depends(get_rag_system)
):
    """
    Upload a PDF file to MongoDB with text embedding using Vertex AI
    
    This endpoint:
    1. Accepts a PDF file upload
    2. Processes the PDF into text chunks
    3. Creates embeddings using Vertex AI (same as the main RAG system)
    4. Stores the embeddings in MongoDB Atlas Vector Search
    5. Returns processing statistics
    
    **How to use:**
    - Content-Type: multipart/form-data
    - Field name: "file"
    - File type: PDF only
    - Max size: 10MB
    """
    try:
        # Debug: Log incoming file information
        print(f"🔍 Received file upload:")
        print(f"   - Filename: {file.filename}")
        print(f"   - Content type: {file.content_type}")
        print(f"   - File object type: {type(file)}")
        
        # Step 1: Validate file object
        if not hasattr(file, 'filename') or file.filename is None:
            raise HTTPException(
                status_code=400,
                detail="No file was uploaded. Please ensure you're sending a file in the 'file' field with multipart/form-data."
            )
        
        # Step 2: Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"Only PDF files are allowed. Received: {file.filename}"
            )
        
        # Step 3: Validate content type (optional but helpful)
        if file.content_type and not file.content_type.startswith(('application/pdf', 'application/octet-stream')):
            print(f"⚠️ Warning: Unexpected content type: {file.content_type}")
        
        # Step 4: Check file size (optional - limit to 10MB)
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )
        
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is 10MB, received: {round(file_size / (1024 * 1024), 2)}MB"
            )
        
        # Step 5: Verify RAG system is initialized
        if not rag_system.embeddings or not rag_system.vector_store:
            raise HTTPException(
                status_code=503,
                detail="RAG system not fully initialized. Please wait and try again."
            )
        
        # Step 6: Process the PDF
        print(f"📄 Processing PDF upload: {file.filename} ({round(file_size / (1024 * 1024), 2)}MB)")
        
        result = await rag_system.upload_pdf_from_bytes(
            pdf_bytes=file_content,
            filename=file.filename
        )
        
        # Step 7: Return results
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "PDF uploaded successfully",
                    "filename": file.filename,
                    "file_size_mb": round(file_size / (1024 * 1024), 2),
                    "documents_processed": result["documents_processed"],
                    "chunks_created": result["chunks_created"],
                    "embedding_model": result["embedding_model"],
                    "vector_store_stats": result.get("vector_store_stats", {})
                }
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"PDF processing failed: {result['message']}"
            )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        print(f"❌ Unexpected error during PDF upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during PDF upload: {str(e)}"
        )

@router.get("/stats")
async def get_pdf_stats(rag_system = Depends(get_rag_system)):
    """
    Get statistics about the PDF documents in the vector store
    """
    try:
        stats = rag_system.get_vector_store_stats()
        return JSONResponse(
            status_code=200,
            content={
                "message": "Vector store statistics",
                "stats": stats
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting PDF stats: {str(e)}"
        )

@router.delete("/clear")
async def clear_pdf_collection(rag_system = Depends(get_rag_system)):
    """
    Clear all documents from the PDF collection
    WARNING: This will delete all uploaded PDFs!
    """
    try:
        # Check if MongoDB collection is properly initialized
        if rag_system.mongodb_collection is None:
            raise HTTPException(
                status_code=503,
                detail="MongoDB collection not initialized"
            )
        
        # Test connection first
        if not rag_system.test_connection():
            raise HTTPException(
                status_code=503,
                detail="MongoDB connection failed"
            )
        
        # Count documents before deletion
        doc_count = rag_system.mongodb_collection.count_documents({})
        
        # Delete all documents
        result = rag_system.mongodb_collection.delete_many({})
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF collection cleared successfully",
                "documents_deleted": result.deleted_count,
                "original_count": doc_count
            }
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"❌ Error clearing PDF collection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing PDF collection: {str(e)}"
        ) 