"""
Simple WebSocket endpoint for Gemini Live
Replaces the complex Vertex AI implementation with direct WebSocket approach
"""

import asyncio
import json
import base64
import os
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
import logging

from gemini_websocket_simple import SimpleGeminiLive

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SimpleWebSocketEndpoint")

# Load environment variables
load_dotenv()

async def handle_simple_gemini_websocket(websocket: WebSocket, call_session_id: int):
    """
    Simple WebSocket endpoint for Gemini Live using direct Google AI API connection
    """
    gemini_session: Optional[SimpleGeminiLive] = None
    response_task: Optional[asyncio.Task] = None
    
    try:
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            await websocket.close(code=1011, reason="GEMINI_API_KEY not configured")
            return
        
        logger.info(f"📞 Starting simple Gemini Live session for call {call_session_id}")
        
        # Accept WebSocket connection
        await websocket.accept()
        
        # Create Gemini Live session
        gemini_session = SimpleGeminiLive(api_key)
        
        # Setup connection to Gemini Live
        connection_success = await gemini_session.setup_connection(websocket)
        if not connection_success:
            await websocket.close(code=1011, reason="Failed to connect to Gemini Live")
            return
        
        # Send success message to client
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Gemini Live. You can start speaking!",
            "session_id": call_session_id
        })
        logger.info("📤 Sent connection success message to client")
        
        # Send initial greeting
        await gemini_session.send_text_message("Hello! I'm your AI assistant. How can I help you today?")
        
        # Start background task to handle Gemini responses
        response_task = asyncio.create_task(gemini_session.handle_gemini_responses())
        
        # Handle incoming messages from client
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                message_type = data.get("type")
                
                if message_type == "audio_chunk":
                    # Handle audio input from client
                    audio_data_b64 = data.get("audio_data", "")
                    
                    if audio_data_b64:
                        # Decode base64 audio data
                        try:
                            audio_bytes = base64.b64decode(audio_data_b64)
                            await gemini_session.send_audio_chunk(audio_bytes)
                        except Exception as e:
                            logger.error(f"❌ Error processing audio: {e}")
                
                elif message_type == "text_message":
                    # Handle text input from client
                    text_content = data.get("text", "")
                    if text_content.strip():
                        logger.info(f"📝 Received text from client: {text_content}")
                        await gemini_session.send_text_message(text_content)
                
                elif message_type == "turn_complete":
                    # Signal end of user turn
                    await gemini_session.send_turn_complete()
                    logger.info("📋 Received turn complete from client")
                
                elif message_type == "end_session":
                    # Client wants to end the session
                    logger.info("🔚 Client requested session end")
                    break
                    
                else:
                    logger.warning(f"❓ Unknown message type: {message_type}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON received: {message[:100]}...")
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
    
    except WebSocketDisconnect:
        logger.info("🔌 Client disconnected")
    except Exception as e:
        logger.error(f"❌ Error in WebSocket handler: {e}")
        try:
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")
        except:
            pass
    
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up simple Gemini Live session...")
        
        # Cancel response task
        if response_task and not response_task.done():
            response_task.cancel()
            try:
                await response_task
            except asyncio.CancelledError:
                pass
        
        # Close Gemini session
        if gemini_session:
            await gemini_session.close()
        
        logger.info("✅ Simple Gemini Live session cleanup completed")