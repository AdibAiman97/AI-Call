from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import base64
import os
import numpy as np
from dotenv import load_dotenv

# Import existing routers
from api.customer_router import router as customer_router
from api.call_session_router import router as call_session_router
from api.transcript_router import router as transcript_router
from api.appointment_router import router as appointment_router
from api.property_router import router as property_router
from api.pdf_router import router as pdf_router

# Import database components
from database.connection import engine, Base, get_db
from services.call_session import CallSessionService
from database.schemas import CallSessionBase
from datetime import datetime

# Import RAG system
from VertexRagSystem.rag_class import VertexRAGSystem, RAGConfig

# Import the new Gemini Live system
from gemini_live_websocket import GeminiLiveManager, GeminiLiveConfig, get_gemini_live_manager

# Import the simple WebSocket implementation
from simple_websocket_endpoint import handle_simple_gemini_websocket

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# Include existing routers
app.include_router(customer_router)
app.include_router(call_session_router)
app.include_router(transcript_router)
app.include_router(appointment_router)
app.include_router(property_router)
app.include_router(pdf_router)

# Global variables
rag_sys: Optional[VertexRAGSystem] = None
gemini_live_manager: Optional[GeminiLiveManager] = None
init_status = {"status": "not_started", "message": ""}

# Note: Voice Activity Detection (VAD) is now handled by Gemini Live's built-in automatic activity detection

@app.on_event("startup")
async def startup_event():
    """Initialize RAG System and Gemini Live Manager"""
    global rag_sys, gemini_live_manager, init_status
    
    from config import DB_NAME, COLLECTION_NAME, ATLAS_VECTOR_SEARCH_INDEX_NAME, GCP_PROJECT_ID, GCP_LOCATION
    
    load_dotenv()
    
    try:
        init_status["status"] = "initializing"
        init_status["message"] = "Starting RAG and Gemini Live systems..."
        
        # Initialize RAG System
        print("🔄 Initializing RAG System...")
        MONGO_DB_CONNECTION_STRING = os.getenv("MONGO_DB") 
        
        # Configure RAG System
        rag_config = RAGConfig(
            project_id=GCP_PROJECT_ID,
            location=GCP_LOCATION,
            mongo_db_connection_string=MONGO_DB_CONNECTION_STRING,
            db_name=DB_NAME,
            collection_name=COLLECTION_NAME,
            atlas_vector_search_index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )

        rag_sys = VertexRAGSystem(rag_config)
        await rag_sys.initialize()

        # Test RAG connection
        if rag_sys.test_connection():
            stats = rag_sys.get_vector_store_stats()
            print(f"📊 RAG System - Vector store stats: {stats}")
        else:
            raise Exception("RAG System - MongoDB connection test failed")
        
        # Initialize Gemini Live Manager
        print("🔄 Initializing Gemini Live Manager...")
        gemini_config = GeminiLiveConfig(
            project_id=GCP_PROJECT_ID,
            location=GCP_LOCATION,
            model_name="gemini-2.0-flash-live-preview-04-09",  # Official Vertex AI Live model
            voice_name="Aoede",  # Professional female voice
            max_output_tokens=1000,
            temperature=0.7
        )
        
        gemini_live_manager = GeminiLiveManager(gemini_config)

        init_status["status"] = "completed"
        init_status["message"] = "RAG System and Gemini Live initialized successfully"
        
        print("✅ All systems initialized successfully")

    except Exception as e:
        init_status["status"] = "failed"
        init_status["message"] = f"Initialization failed: {str(e)}"
        print(f"❌ Initialization failed: {e}")

@app.websocket("/stt/{call_session_id}")
async def gemini_live_websocket(websocket: WebSocket, call_session_id: int):
    """
    WebSocket endpoint for Gemini Live 2.5 Flash Preview native audio dialog
    Replaces the previous STT/TTS pipeline with unified voice conversation
    """
    gemini_session = None
    db = None
    response_task = None
    
    # Note: Turn management is now handled by Gemini Live's automatic activity detection
    
    try:
        # Check if systems are initialized BEFORE accepting websocket
        if not rag_sys or not gemini_live_manager:
            await websocket.close(code=1011, reason="Systems not initialized")
            return
        
        # Get or create call session BEFORE accepting websocket
        print("📞 Setting up call session...")
        db_gen = get_db()
        db = next(db_gen)
        service = CallSessionService(db)
        
        call_session = service.get_by_id(call_session_id)
        call_summary = call_session.summarized_content if call_session else None
        
        # Create new call session if not exists
        if not call_session:
            create_call_session = service.create(
                CallSessionBase(cust_id="0123334444")
            )
            call_session = service.get_by_id(create_call_session.id)
            call_session_id = call_session.id
            
        print(f"📞 Call session ID: {call_session_id}")
        
        # Create and prepare Gemini Live session BEFORE accepting websocket
        print("🎤 Creating Gemini Live session...")
        gemini_session = await gemini_live_manager.create_session(
            call_session_id=str(call_session_id),
            rag_system=rag_sys,
            call_summary=call_summary
        )
        
        # Prepare Gemini Live configuration
        print("🔄 Preparing Gemini Live configuration...")
        config_success = await gemini_session.connect()
        if not config_success:
            await websocket.close(code=1011, reason="Failed to prepare Gemini Live configuration")
            return
        
        print("🔗 Establishing Gemini Live connection (blocking)...")
        # Establish the actual connection before accepting websocket
        async with gemini_session.client.aio.live.connect(
            model=gemini_session.config.model_name,
            config=gemini_session.connection_config
        ) as session:
            print("✅ Gemini Live connected successfully! Now accepting websocket...")
            gemini_session.session = session  # Store session reference
            
            # NOW accept the websocket after everything is ready
            await websocket.accept()
            
            # Send immediate success message
            try:
                await websocket.send_json({
                    "type": "connected",
                    "message": "Connected to Gemini Live. You can start speaking now.",
                    "session_id": call_session_id
                })
                print("📤 Success message sent to client")
            except Exception as send_error:
                print(f"⚠️ Client disconnected before receiving success message: {send_error}")
                return
            
            # Send initial greeting trigger to Gemini Live
            try:
                from google.genai.types import Content, Part
                print("🎯 Sending initial greeting trigger to Gemini Live...")
                await session.send_client_content(
                    turns=Content(
                        role="user", 
                        parts=[Part(text="Hello, I'm interested in learning about properties at Gamuda Cove.")]
                    )
                )
                print("✅ Initial greeting trigger sent")
            except Exception as trigger_error:
                print(f"⚠️ Failed to send initial greeting trigger: {trigger_error}")
                # Continue anyway - not critical
            
            # Start background listener for Gemini Live responses
            async def handle_gemini_responses():
                """Handle responses from Gemini Live"""
                try:
                    audio_buffer = []
                    
                    async for message in session.receive():
                        if not gemini_session.session_active:
                            break
                        
                        try:
                            # Handle server content (audio/text responses)
                            if message.server_content:
                                server_content = message.server_content
                                
                                # Handle model turn (audio response)
                                if server_content.model_turn and server_content.model_turn.parts:
                                    for part in server_content.model_turn.parts:
                                        if part.inline_data:
                                            # Collect audio data
                                            audio_data = np.frombuffer(part.inline_data.data, dtype=np.int16)
                                            audio_buffer.append(audio_data)
                                        
                                        elif part.text:
                                            # Handle text response (for debugging/logging)
                                            text_content = part.text
                                            print(f"📝 Gemini response: {text_content}")
                                            
                                            # Save to database
                                            try:
                                                from services.transcript_crud import create_session_message
                                                create_session_message(
                                                    db,
                                                    session_id=str(call_session_id),
                                                    message=text_content,
                                                    message_by="AI"
                                                )
                                            except Exception as e:
                                                print(f"❌ Error saving to database: {e}")
                                
                                # Handle turn completion
                                if server_content.turn_complete:
                                    # Send accumulated audio to client
                                    if audio_buffer:
                                        # Concatenate all audio chunks
                                        full_audio = np.concatenate(audio_buffer)
                                        # Convert to base64 for transmission
                                        audio_base64 = base64.b64encode(full_audio.tobytes()).decode('utf-8')
                                        
                                        try:
                                            await websocket.send_json({
                                                "type": "audio_response",
                                                "audio_data": audio_base64,
                                                "encoding": "base64",
                                                "sample_rate": 24000,
                                                "dtype": "int16"
                                            })
                                            print(f"📤 Sent {len(full_audio)} audio samples to client")
                                            
                                        except Exception as e:
                                            print(f"❌ Error sending audio to client: {e}")
                                        
                                        # Clear buffer
                                        audio_buffer = []
                                    
                                    # Send turn complete signal
                                    try:
                                        await websocket.send_json({
                                            "type": "turn_complete",
                                            "message": "AI finished speaking"
                                        })
                                    except Exception as e:
                                        print(f"❌ Error sending turn complete: {e}")
                            
                            # Handle tool calls (function calls)
                            if message.tool_call:
                                tool_call = message.tool_call
                                
                                for function_call in tool_call.function_calls:
                                    # Process the function call
                                    response = await gemini_session.handle_function_call({
                                        "name": function_call.name,
                                        "args": function_call.args
                                    })
                                    
                                    # Send function response back to Gemini Live
                                    await session.send_tool_response(
                                        function_response=response
                                    )
                                    print(f"📤 Sent tool response: {response['name']}")
                            
                        except Exception as e:
                            print(f"❌ Error processing Gemini message: {e}")
                            continue
                
                except Exception as e:
                    print(f"❌ Error in Gemini response handler: {e}")
            
            # Start the response handler as a background task
            response_task = asyncio.create_task(handle_gemini_responses())
            
            # Handle incoming messages from client
            async for message in websocket.iter_text():
                try:
                    # Ensure message is properly decoded as string
                    if isinstance(message, bytes):
                        message = message.decode('utf-8')
                    elif not isinstance(message, str):
                        print(f"⚠️ Unexpected message type: {type(message)}")
                        continue
                    
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "audio_chunk":
                        # Handle audio input from client
                        audio_data_b64 = data.get("audio_data", "")
                        
                        if audio_data_b64:
                            # Decode base64 audio data
                            audio_bytes = base64.b64decode(audio_data_b64)
                            
                            # Send all audio to Gemini Live (VAD disabled for testing)
                            await gemini_session.send_audio_chunk(audio_bytes)
                    
                    elif message_type == "start_speaking":
                        # Optional: Manual client control (for debugging)
                        print("🎤 Client: Manual start speaking signal (using built-in VAD)")
                        
                    elif message_type == "stop_speaking":
                        # Optional: Manual client control (for debugging)
                        print("🔇 Client: Manual stop speaking signal (using built-in VAD)")
                    
                    elif message_type == "text_message":
                        # Handle text input (for debugging or fallback)
                        text_content = data.get("text", "")
                        print(f"📝 Received text: {text_content}")
                        
                        # You can send text to Gemini Live if needed
                        # For now, we'll focus on audio-only interaction
                        
                    elif message_type == "end_session":
                        # Clean session termination
                        print("🔚 Client requested session end")
                        break
                    
                    else:
                        print(f"❓ Unknown message type: {message_type}")
                        
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON received: {message[:100]}... Error: {e}")
                    continue
                except UnicodeDecodeError as e:
                    print(f"❌ Text decoding error: {e}")
                    continue
                except Exception as e:
                    print(f"❌ Error processing message: {e}")
                    continue
    
    except Exception as e:
        print(f"❌ Error in websocket setup or communication: {e}")
        # Try to close websocket if it was accepted
        try:
            await websocket.close(code=1011, reason=f"Setup error: {str(e)}")
        except:
            pass
    
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    
    finally:
        # Clean up
        print("🧹 Cleaning up Gemini Live session...")
        
        # Cancel response task
        if response_task and not response_task.done():
            response_task.cancel()
            try:
                await response_task
            except asyncio.CancelledError:
                pass
        
        # Close Gemini Live session
        if gemini_session:
            await gemini_session.close()
        
        # Close database connection
        if db:
            db.close()
        
        print("✅ Gemini Live session cleanup completed")

@app.websocket("/simple-stt/{call_session_id}")
async def simple_gemini_websocket(websocket: WebSocket, call_session_id: int):
    """
    Simple WebSocket endpoint for Gemini Live using direct Google AI API connection
    This is a simplified version that bypasses Vertex AI complexity
    """
    await handle_simple_gemini_websocket(websocket, call_session_id)

@app.get("/test-rag")
async def test_rag_endpoint():
    """Test endpoint to verify RAG functionality"""
    if not rag_sys:
        return JSONResponse(
            status_code=503, 
            content={"error": "RAG system not initialized"}
        )
    
    test_query = "What properties are available at Gamuda Cove?"
    
    try:
        result = await rag_sys.rag_query(
            query=test_query,
            include_sources=True,
            use_memory=False
        )
        
        result["system_stats"] = {
            "vector_store_stats": rag_sys.get_vector_store_stats(),
            "connection_test": rag_sys.test_connection(),
            "config": {
                "database": rag_sys.config.db_name,
                "collection": rag_sys.config.collection_name,
                "index_name": rag_sys.config.atlas_vector_search_index_name,
                "top_k": rag_sys.config.top_k_docs
            }
        }
        
        return result
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Test query failed: {str(e)}",
                "query": test_query,
                "system_stats": {
                    "vector_store_stats": rag_sys.get_vector_store_stats() if rag_sys else None,
                    "connection_test": rag_sys.test_connection() if rag_sys else False
                }
            }
        )

@app.get("/gemini-live-status")
async def gemini_live_status():
    """Get Gemini Live system status"""
    if not gemini_live_manager:
        return {
            "status": "not_initialized",
            "message": "Gemini Live system not initialized"
        }
    
    return {
        "status": "initialized",
        "message": "Gemini Live system ready",
        "active_sessions": len(gemini_live_manager.active_sessions),
        "model": gemini_live_manager.config.model_name,
        "voice": gemini_live_manager.config.voice_name
    }

@app.get("/system-status")
async def system_status():
    """Get overall system status"""
    return {
        "initialization": init_status,
        "rag_system": {
            "status": "initialized" if rag_sys else "not_initialized",
            "mongodb_connection": rag_sys.test_connection() if rag_sys else False,
            "vector_store_stats": rag_sys.get_vector_store_stats() if rag_sys else None
        },
        "gemini_live": {
            "status": "initialized" if gemini_live_manager else "not_initialized",
            "active_sessions": len(gemini_live_manager.active_sessions) if gemini_live_manager else 0
        }
    }

@app.get("/check-documents")
async def check_documents():
    """Check what documents are in the MongoDB collection"""
    if not rag_sys:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized"}
        )
    
    try:
        collection = rag_sys.mongodb_collection
        
        total_docs = collection.count_documents({})
        sample_docs = list(collection.find().limit(3))
        
        for doc in sample_docs:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
        
        docs_with_embeddings = collection.count_documents({"embedding": {"$exists": True}})
        sample_fields = list(sample_docs[0].keys()) if sample_docs else []
        
        return {
            "total_documents": total_docs,
            "documents_with_embeddings": docs_with_embeddings,
            "sample_fields": sample_fields,
            "sample_documents": sample_docs,
            "database": rag_sys.config.db_name,
            "collection": rag_sys.config.collection_name,
            "message": "Collection is empty" if total_docs == 0 else f"Found {total_docs} documents"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to check documents: {str(e)}"}
        )

@app.get("/test-retrieval/{query}")
async def test_retrieval(query: str):
    """Test document retrieval from MongoDB Atlas Vector Search"""
    if not rag_sys:
        return JSONResponse(
            status_code=503,
            content={"error": "RAG system not initialized"}
        )
    
    try:
        print(f"🔍 Testing retrieval for query: {query}")
        
        connection_ok = rag_sys.test_connection()
        stats = rag_sys.get_vector_store_stats()
        retrieved_docs = await rag_sys.retrieve_relevant_docs(query, top_k=5)
        
        return {
            "query": query,
            "connection_status": connection_ok,
            "collection_stats": stats,
            "retrieved_documents_count": len(retrieved_docs),
            "retrieved_documents": retrieved_docs,
            "retriever_config": {
                "search_kwargs": rag_sys.retriever.search_kwargs if rag_sys.retriever else None,
                "index_name": rag_sys.config.atlas_vector_search_index_name
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Retrieval test failed: {str(e)}",
                "query": query,
                "connection_status": rag_sys.test_connection() if rag_sys else False
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)