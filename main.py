from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.websockets import WebSocket
from pydantic import BaseModel
from typing import Optional
# Files
from stt import AudioBuffer, TranscriptManager, audio_generator, audio_receiver, speech_processor
from VertexRagSystem.rag_class import VertexRAGSystem, RAGConfig
from google.cloud import speech
from collections import deque

from api.customer_router import router as customer_router
from api.call_session_router import router as call_session_router
from api.transcript_router import router as transcript_router
from api.appointment_router import router as appointment_router
from api.property_router import router as property_router
from database.connection import engine, Base

import uvicorn
import asyncio
import json
import threading

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

app.include_router(customer_router)
app.include_router(call_session_router)
app.include_router(transcript_router)
app.include_router(appointment_router)
app.include_router(property_router)

# app.include_router(stt)

# class QueryRequest(BaseModel):
#     query: str
    # include_sources: bool = True
    # top_k: Optional[int] = None

rag_sys: Optional[VertexRAGSystem] = None
init_status = {"status": "not_started", "message" : ""}

@app.on_event("startup")
async def startup_event():
    """Init Rag System"""
    global rag_sys, init_status

    try: 
        init_status["status"] = "Init...."
        init_status["message"] = "Starting Rag System...."

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

async def start_speech_session(
    ws: WebSocket, 
    rag_sys, 
    speech_client, 
    config, 
    streaming_config, 
    tts_state_manager=None,
    call_session_id=None,
    ):
    """Start a single speech recognition session with proper task management"""
    from stt import TTSStateManager
    
    audio_buffer = AudioBuffer()
    transcript_manager = TranscriptManager()
    
    # Create or use provided TTS state manager
    if tts_state_manager is None:
        tts_state_manager = TTSStateManager()
    
    # Create tasks that we can cancel if needed
    audio_task = None
    speech_task = None
    
    try:
        print("🔧 Creating audio receiver and speech processor tasks...")
        
        # Create tasks explicitly so we can manage them
        audio_task = asyncio.create_task(
            audio_receiver(ws, audio_buffer)
        )
        speech_task = asyncio.create_task(
            speech_processor(
                speech_client, 
                streaming_config, 
                transcript_manager, 
                audio_buffer, 
                speech, 
                ws, 
                rag_sys,
                tts_state_manager,
                call_session_id
            )
        )
        
        # Wait for tasks to complete - don't use return_exceptions so exceptions are raised
        try:
            results = await asyncio.gather(
                audio_task,
                speech_task
            )
            print("✅ Both audio receiver and speech processor completed successfully")
            return transcript_manager
            
        except Exception as task_error:
            error_message = str(task_error)
            print(f"❌ Task failed with exception: {error_message}")
            
            # Check if it's a timeout error that we should retry
            if ("Audio Timeout Error" in error_message or 
                "400" in error_message or 
                "Long duration elapsed without audio" in error_message):
                print("🔄 Audio timeout detected in task - will retry session")
                # This will trigger the finally block for cleanup
                raise task_error
            else:
                print(f"💥 Non-timeout error in task: {error_message}")
                raise task_error
        
    except asyncio.CancelledError:
        print("🛑 Speech session was cancelled")
        raise
    except Exception as e:
        print(f"❌ Exception in speech session: {e}")
        raise
        
    finally:
        print("🧹 Cleaning up speech session tasks...")
        
        # Always finish audio buffer first
        audio_buffer.finish()
        print("🛑 Audio buffer marked as finished")
        
        # Cancel any running tasks
        tasks_to_cancel = []
        if audio_task and not audio_task.done():
            print("🛑 Cancelling audio receiver task...")
            audio_task.cancel()
            tasks_to_cancel.append(("audio_receiver", audio_task))
        
        if speech_task and not speech_task.done():
            print("🛑 Cancelling speech processor task...")
            speech_task.cancel()
            tasks_to_cancel.append(("speech_processor", speech_task))
        
        # Wait for all cancelled tasks to complete
        for task_name, task in tasks_to_cancel:
            try:
                await task
                print(f"✅ {task_name} task cancelled successfully")
            except asyncio.CancelledError:
                print(f"✅ {task_name} task cancelled")
            except Exception as e:
                print(f"⚠️ {task_name} task cancellation error: {e}")
        
        # Clear TTS state if still active
        if tts_state_manager and tts_state_manager.is_active():
            print("🔇 Clearing TTS state during cleanup")
            tts_state_manager.end_tts()
        
        print("✅ Speech session cleanup complete")

@app.websocket("/stt/{call_session_id}")
async def rag_query_stream(ws: WebSocket, query: Optional[str] = None, call_session_id: Optional[int] = None):

    # Call Session ID in
    # What is the phone number?

    # If PN, get latest call session@summarized
    # pass into RAG

    await ws.accept()
    await ws.send_text("✅ WebSocket connected to Google STT")

    if not rag_sys:
        await ws.send_text(json.dumps({
            "type": "error",
            "message": "RAG system not initialized"
        }))
        return

    # Create Google Speech client and config (reusable)
    speech_client = speech.SpeechClient()
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    # Keep track of overall conversation and TTS state
    from stt import TTSStateManager
    conversation_transcript_manager = TranscriptManager()
    tts_state_manager = TTSStateManager()
    retry_count = 0
    max_retries = 5  # Prevent infinite loops
    consecutive_tts_timeouts = 0  # Track TTS-related timeouts
    
    while retry_count < max_retries:
        try:
            # Check WebSocket connection before starting
            if ws.application_state == 3:  # DISCONNECTED
                print("❌ WebSocket disconnected - ending session")
                break
                
            print(f"🎤 Starting speech session (attempt {retry_count + 1})")
            
            # Notify frontend about session start/restart
            if retry_count > 0:
                try:
                    await ws.send_text(json.dumps({
                        "type": "session_restart",
                        "message": f"Restarting speech recognition (attempt {retry_count + 1})",
                        "retry_count": retry_count
                    }))
                    print(f"📱 Sent restart notification to frontend (attempt {retry_count + 1})")
                except Exception as send_error:
                    print(f"❌ Failed to send restart notification: {send_error}")
                    break  # If we can't send to frontend, connection is likely broken
            
            # Start speech session WITH RAG 
            session_transcript_manager = await start_speech_session(
                ws, rag_sys, speech_client, config, streaming_config, tts_state_manager,
                call_session_id
            )
            
            # If we get here, session completed successfully
            print("✅ Speech session completed successfully")
            
            # Merge session transcript into conversation transcript
            if session_transcript_manager:
                final_text = session_transcript_manager.get_final_only()
                if final_text:
                    conversation_transcript_manager.add_final(final_text)
            
            break  # Exit retry loop on success
            
        except Exception as e:
            error_message = str(e)
            print(f"❌ Speech session error: {error_message}")
            
            # Check WebSocket state before deciding to retry
            if ws.application_state == 3:  # DISCONNECTED
                print("❌ WebSocket disconnected during error - ending session")
                break
            
            # Check if it's a timeout error that we should retry
            if ("Audio Timeout Error" in error_message or 
                "400" in error_message or 
                "Long duration elapsed without audio" in error_message):
                
                retry_count += 1
                
                # Check if this was a TTS-related timeout
                was_tts_timeout = "Speech timeout during TTS" in error_message or tts_state_manager.is_active()
                if was_tts_timeout:
                    consecutive_tts_timeouts += 1
                    print(f"🔊 TTS-related timeout detected (consecutive: {consecutive_tts_timeouts})")
                    # End TTS state to ensure clean restart
                    tts_state_manager.end_tts()
                else:
                    consecutive_tts_timeouts = 0  # Reset if not TTS-related
                
                # Allow more retries for TTS-related timeouts since they're expected during AI responses
                effective_max_retries = max_retries + 3 if consecutive_tts_timeouts > 2 else max_retries
                
                if retry_count >= effective_max_retries:
                    if consecutive_tts_timeouts > 2:
                        print(f"💀 Maximum retries ({effective_max_retries}) exceeded (many TTS timeouts)")
                    else:
                        print(f"💀 Maximum retries ({max_retries}) exceeded")
                    break
                
                print(f"🔄 Retrying speech session ({retry_count}/{max_retries})")
                
                # Notify frontend about the retry
                try:
                    retry_message = "Speech timeout - restarting recognition..."
                    if was_tts_timeout:
                        retry_message = "Speech timeout during AI response - restarting recognition..."
                    
                    await ws.send_text(json.dumps({
                        "type": "timeout_retry",
                        "message": retry_message,
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                        "was_tts_timeout": was_tts_timeout
                    }))
                    print(f"📱 Sent timeout retry notification (attempt {retry_count})")
                except Exception as send_error:
                    print(f"❌ Failed to send retry notification: {send_error}")
                    break  # Connection likely broken
                
                # Longer delay for TTS-related timeouts to let audio finish playing
                delay = 3 if was_tts_timeout else 2
                print(f"⏳ Waiting {delay} seconds before retry...")
                await asyncio.sleep(delay)
                continue
            else:
                # Non-timeout error - don't retry
                print(f"💥 Non-recoverable error (not retrying): {error_message}")
                break
    
    # Final cleanup and session summary
    try:
        if ws.application_state != 3:  # Not disconnected
            final_complete = conversation_transcript_manager.get_final_only()
            
            if retry_count >= max_retries:
                await ws.send_text(json.dumps({
                    "type": "session_failed",
                    "message": "Maximum retries exceeded. Please refresh and try again.",
                    "final_transcript": final_complete
                }))
            else:
                await ws.send_text(json.dumps({
                    "type": "session_complete",
                    "final_transcript": final_complete,
                    "total_retries": retry_count,
                    "tts_timeouts": consecutive_tts_timeouts
                }))

               
            if final_complete:
                # Save to database
                from database.connection import SessionLocal
                from database.transcript_crud import TranscriptCRUD
                from database.schemas import TranscriptCreate

                db = SessionLocal()
                try:
                    transcript_data = TranscriptCreate(
                        session_id=call_session_id,
                        message=final_complete,
                        message_by="System",
                    )
                    TranscriptCRUD.create_transcript(db, transcript=transcript_data)
                finally:
                    db.close()
        else:
            print("WebSocket already closed, skipping final message")
    except Exception as e:
        print(f"Error sending final transcript: {e}")
    
    print(f"🏁 STT WebSocket session ended (total retries: {retry_count})")




# @app.post("/query")
# async def rag_query(
#     query: str,
#     include_sources: bool = True,
#     top_k: Optional[int] = None
# ):
#     """
#     Perform complete RAG query: retrieve relevant documents and generate response
    
#     Args:
#         query: The question or query string
#         include_sources: Whether to include source documents in response
#         top_k: Number of top documents to retrieve (optional)
#     """
#     if not rag_sys:
#         raise HTTPException(status_code=503, detail="RAG system not initialized")
    
#     if not query.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty")
    
#     try:
#         # Temporarily set top_k if provided
#         original_top_k = rag_sys.config.top_k_docs
#         if top_k:
#             rag_sys.config.top_k_docs = top_k
        
#         result = await rag_sys.rag_query(query, include_sources)
        
#         # Restore original top_k
#         rag_sys.config.top_k_docs = original_top_k
        
#         return result
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# @app.post("/generate")
# async def generate_response(
#     prompt: str,
#     context: Optional[str] = None
# ):
#     """
#     Generate response using LLM with optional context
    
#     Args:
#         prompt: The input prompt
#         context: Optional context information
#     """
#     if not rag_sys:
#         raise HTTPException(status_code=503, detail="RAG system not initialized")
    
#     if not prompt.strip():
#         raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
#     try:
#         response = await rag_sys.generate_response(prompt, context)
#         return {
#             "prompt": prompt,
#             "context": context,
#             "response": response
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

# @app.post("/retrieve")
# async def retrieve_documents(
#     query: str,
#     top_k: int = 3
# ):
#     """
#     Retrieve relevant documents without generating response
    
#     Args:
#         query: The search query
#         top_k: Number of top documents to retrieve
#     """
#     if not rag_sys:
#         raise HTTPException(status_code=503, detail="RAG system not initialized")
    
#     if not query.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty")
    
#     if top_k <= 0:
#         raise HTTPException(status_code=400, detail="top_k must be greater than 0")
    
#     try:
#         docs = await rag_sys.retrieve_relevant_docs(query, top_k)
#         return {
#             "query": query,
#             "top_k": top_k,
#             "documents": docs,
#             "count": len(docs)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Document retrieval failed: {str(e)}")

# @app.post("/add_documents")
# async def add_documents(documents: List[Dict[str, str]]):
#     """
#     Add documents to the knowledge base
    
#     Args:
#         documents: List of documents with id, title, and content fields
#     """
#     if not rag_sys:
#         raise HTTPException(status_code=503, detail="RAG system not initialized")
    
#     if not documents:
#         raise HTTPException(status_code=400, detail="Documents list cannot be empty")
    
#     # Validate document format
#     for i, doc in enumerate(documents):
#         required_fields = ["id", "title", "content"]
#         for field in required_fields:
#             if field not in doc:
#                 raise HTTPException(
#                     status_code=400, 
#                     detail=f"Document {i} missing required field: {field}"
#                 )
    
#     try:
#         rag_sys.add_documents(documents)
#         return {
#             "message": f"Successfully added {len(documents)} documents",
#             "total_documents": len(rag_sys.knowledge_base),
#             "status": "success"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to add documents: {str(e)}")

# @app.post("/embed_documents")
# async def embed_documents(batch_size: int = 10):
#     """
#     Generate embeddings for all documents in the knowledge base
    
#     Args:
#         batch_size: Number of documents to process in each batch
#     """
#     if not rag_sys:
#         raise HTTPException(status_code=503, detail="RAG system not initialized")
    
#     if batch_size <= 0:
#         raise HTTPException(status_code=400, detail="batch_size must be greater than 0")
    
#     try:
#         await rag_sys.embed_documents(batch_size)
#         return {
#             "message": f"Successfully embedded {len(rag_sys.embedded_docs)} documents",
#             "embedded_count": len(rag_sys.embedded_docs),
#             "total_documents": len(rag_sys.knowledge_base),
#             "status": "success"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

# @app.exception_handler(404)
# async def not_found_handler(request, exc):
#     return JSONResponse(
#         status_code=404,
#         content={"detail": "Endpoint not found"}
#     )

# @app.exception_handler(500)
# async def internal_error_handler(request, exc):
#     return JSONResponse(
#         status_code=500,
#         content={"detail": "Internal server error"}
#     )