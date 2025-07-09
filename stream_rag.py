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
        print(f"📝 Query received: {query}")
        
        # Save and update top_k if provided
        original_top_k = rag_sys.config.top_k_docs
        if top_k:
            rag_sys.config.top_k_docs = top_k

        # Get relevant documents
        relevant_docs = await rag_sys.retrieve_relevant_docs(query)

        # Handle case with no relevant docs found
        if not relevant_docs:
            return
            
        # Build context from documents
        context = "\n".join(
            f"Document {i+1} (ID: {doc['doc_id']}):\nTitle: {doc['title']}\nContent: {doc['content']}\n"
            for i, doc in enumerate(relevant_docs)
        )

        # Stream only the LLM response
        async for chunk in rag_sys.generate_response_stream(query, context, call_summary=call_summary):
            if chunk:
                # print(f"LLM Response:{chunk}")
                yield chunk

        # Restore original top_k
        rag_sys.config.top_k_docs = original_top_k

    except Exception as e:
        print(f"Query processing failed: {str(e)}")