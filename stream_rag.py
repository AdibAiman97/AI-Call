from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Optional, Tuple

import json

async def generate_stream(
    rag_sys, 
    query,
    stream_queue,
    include_sources: bool = True,
    top_k: Optional[int] = None
    ):
    try:
        print(f"📝 Query received: {query}")
        # Temporarily set top_k if provided
        original_top_k = rag_sys.config.top_k_docs
        if top_k:
            rag_sys.config.top_k_docs = top_k
        
        # Step 1: Retrieve relevant documents
        relevant_docs = await rag_sys.retrieve_relevant_docs(query)
        
        # Send initial metadata if sources are requested
        if include_sources and relevant_docs:
            sources = []
            for doc in relevant_docs:
                sources.append({
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "similarity": doc["similarity"]
                })
            
            # Send sources as first chunk
            sources_chunk = f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
            await stream_queue.put(sources_chunk)
            # yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        if not relevant_docs:
            err_chunk = f"data: {json.dumps({'type': 'content', 'data': 'I could not find any relevant information in the knowledge base.'})}\n\n"
            await stream_queue.put(err_chunk)
            
            done_chunk = f"data: {json.dumps({'type': 'done'})}\n\n"
            await stream_queue.put(done_chunk)

            # yield f"data: {json.dumps({'type': 'content', 'data': 'I could not find any relevant information in the knowledge base.'})}\n\n"
            # yield f"data: {json.dumps({'type': 'done'})}\n\n"
            await stream_queue.put(None)
            return
        
        # Step 2: Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            context_parts.append(f"Document {i+1} (ID: {doc['doc_id']}):\nTitle: {doc['title']}\nContent: {doc['content']}\n")
        
        context = "\n".join(context_parts)
        
        # Step 3: Stream the response
        start_chunk = f"data: {json.dumps({'type': 'start'})}\n\n"
        await stream_queue.put(start_chunk)
        # yield f"data: {json.dumps({'type': 'start'})}\n\n"
        
        async for chunk in rag_sys.generate_response_stream(query, context):
            print("LLM Response: !") 
            if chunk:
                print(f"{chunk}")
                content_chunk = f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
                await stream_queue.put(content_chunk)
                # yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"

        # Signal completion
        done_chunk = f"data: {json.dumps({'type': 'done'})}\n\n"
        await stream_queue.put(done_chunk)
        await stream_queue.put(None)  # Signal end of stream
        
        # Restore original top_k
        rag_sys.config.top_k_docs = original_top_k
        
    except Exception as e:
        err_chunk = f"data: {json.dumps({'type': 'error', 'data': f'Query processing failed: {str(e)}'})}\n\n"
        await stream_queue.put(err_chunk)
        await stream_queue.put(None)
        # yield f"data: {json.dumps({'type': 'error', 'data': f'Query processing failed: {str(e)}'})}\n\n"