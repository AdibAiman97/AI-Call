from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Optional, Tuple
import json

async def generate_stream(
    rag_sys,
    query,
    call_summary: Optional[str] = None,
    include_sources: bool = True, 
    top_k: Optional[int] = None,
):
    try:
        print(f"📝 Stream RAG Query received: {query}")
        print(f"🔧 RAG system initialized: {rag_sys is not None}")
        
        # Check RAG system status
        if not rag_sys:
            error_msg = "❌ RAG system not initialized"
            print(error_msg)
            yield error_msg
            return
            
        # Test MongoDB connection before proceeding
        print("🔍 Testing MongoDB connection...")
        if rag_sys.test_connection():
            # print("✅ MongoDB connection verified in stream_rag")
            stats = rag_sys.get_vector_store_stats()
            # print(f"📊 Collection stats: {stats}")
        else:
            print("❌ MongoDB connection failed in stream_rag")
        
        # Save and update top_k if provided
        original_top_k = rag_sys.config.top_k_docs
        if top_k:
            print(f"🔧 Updating top_k from {original_top_k} to {top_k}")
            rag_sys.config.top_k_docs = top_k
            # Update retriever search kwargs
            if rag_sys.retriever:
                rag_sys.retriever.search_kwargs["k"] = top_k
                print(f"✅ Retriever top_k updated to {top_k}")
            else:
                print("⚠️ Retriever not available for top_k update")

        # print("🚀 Starting MongoDB-based streaming generation...")
        chunk_count = 0
        
        # Use the new MongoDB-based streaming generation
        async for chunk in rag_sys.generate_response_stream(
            query, 
            use_memory=True, 
            call_summary=call_summary
        ):
            if chunk:
                chunk_count += 1
                yield chunk

        # print(f"✅ Stream completed: {chunk_count} chunks generated")

        # Restore original top_k
        # print(f"🔧 Restoring original top_k: {original_top_k}")
        rag_sys.config.top_k_docs = original_top_k
        if rag_sys.retriever:
            rag_sys.retriever.search_kwargs["k"] = original_top_k

    except Exception as e:
        error_msg = f"❌ Stream processing failed: {str(e)}"
        print(error_msg)
        # Yield error message to the stream
        yield f"\n[Error: {str(e)}]"