"""
GCP Vertex AI RAG (Retrieval-Augmented Generation) System
A complete implementation with embedding and LLM capabilities
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from VertexRagSystem.rag_class import VertexRAGSystem, RAGConfig

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_sys: Optional[VertexRAGSystem] = None
init_status = {"status": "not_started", "message" : ""}

@app.on_event("startup")
async def startup_event():
    """Init Rag System"""
    global rag_sys, init_status

    try: 
        init_status["status"] = "Init...."
        initi_status["message"] = "Starting Rag System...."

        config = RAGConfig(
            project_id="voxis-ai",
            location="us-central1"
        )

        rag_sys = VertexRAGSystem(config)

        await rag_sys.initialize()

        sample_docs = [
            {
                "id": "doc1",
                "title": "Machine Learning Basics",
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns."
            },
            {
                "id": "doc2", 
                "title": "Deep Learning Overview",
                "content": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition."
            },
            {
                "id": "doc3",
                "title": "RAG Systems",
                "content": "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It first retrieves relevant documents from a knowledge base, then uses those documents as context to generate more accurate and informed responses."
            }
        ]

        rag_sys.add_documents(sample_docs)

        await rag_sys.embed_documents()

        init_status["status"] = "completed"
        init_status["message"] = "RAG System initialized"

    except Exception as e:
        init_status["status"] = "failed"
        init_status["message"] = f"Init failed: {str(e)}"

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Vertex AI RAG System API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "API information",

            "POST /query": "Perform RAG query",
            "POST /retrieve": "Retrieve relevant documents",
            "POST /generate": "Generate response with LLM",
            "POST /add_documents": "Add documents to knowledge base",
            "POST /embed_documents": "Generate embeddings for documents",
            # "POST /save_embeddings": "Save embeddings to file",
            # "POST /load_embeddings": "Load embeddings from file"
        }
    }

@app.post("/query")
async def rag_query(
    query: str,
    include_sources: bool = True,
    top_k: Optional[int] = None
):
    """
    Perform complete RAG query: retrieve relevant documents and generate response
    
    Args:
        query: The question or query string
        include_sources: Whether to include source documents in response
        top_k: Number of top documents to retrieve (optional)
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # Temporarily set top_k if provided
        original_top_k = rag_system.config.top_k_docs
        if top_k:
            rag_system.config.top_k_docs = top_k
        
        result = await rag_system.rag_query(query, include_sources)
        
        # Restore original top_k
        rag_system.config.top_k_docs = original_top_k
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/generate")
async def generate_response(
    prompt: str,
    context: Optional[str] = None
):
    """
    Generate response using LLM with optional context
    
    Args:
        prompt: The input prompt
        context: Optional context information
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    try:
        response = await rag_system.generate_response(prompt, context)
        return {
            "prompt": prompt,
            "context": context,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

@app.post("/retrieve")
async def retrieve_documents(
    query: str,
    top_k: int = 3
):
    """
    Retrieve relevant documents without generating response
    
    Args:
        query: The search query
        top_k: Number of top documents to retrieve
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")
    
    try:
        docs = await rag_system.retrieve_relevant_docs(query, top_k)
        return {
            "query": query,
            "top_k": top_k,
            "documents": docs,
            "count": len(docs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

@app.post("/add_documents")
async def add_documents(documents: List[Dict[str, str]]):
    """
    Add documents to the knowledge base
    
    Args:
        documents: List of documents with id, title, and content fields
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not documents:
        raise HTTPException(status_code=400, detail="Documents list cannot be empty")
    
    # Validate document format
    for i, doc in enumerate(documents):
        required_fields = ["id", "title", "content"]
        for field in required_fields:
            if field not in doc:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Document {i} missing required field: {field}"
                )
    
    try:
        rag_system.add_documents(documents)
        return {
            "message": f"Successfully added {len(documents)} documents",
            "total_documents": len(rag_system.knowledge_base),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

@app.post("/embed_documents")
async def embed_documents(batch_size: int = 10):
    """
    Generate embeddings for all documents in the knowledge base
    
    Args:
        batch_size: Number of documents to process in each batch
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be greater than 0")
    
    try:
        await rag_system.embed_documents(batch_size)
        return {
            "message": f"Successfully embedded {len(rag_system.embedded_docs)} documents",
            "embedded_count": len(rag_system.embedded_docs),
            "total_documents": len(rag_system.knowledge_base),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )