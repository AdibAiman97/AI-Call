"""
Simple Gemini Live WebSocket Implementation
Based on Google's get_started_liveapi_tools.py example
Uses direct WebSocket connection to Google AI API
"""

import asyncio
import json
import base64
import os
import logging
from typing import Optional, Dict, Any
import websockets
from fastapi import WebSocket

# Configure logging
logger = logging.getLogger("GeminiWebSocket")
logger.setLevel(logging.INFO)

class SimpleGeminiLive:
    def __init__(self, api_key: str, model: str = "models/gemini-2.0-flash-exp"):
        self.api_key = api_key
        self.model = model
        self.ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
        self.client_websocket: Optional[WebSocket] = None
        self.gemini_websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.session_active = True

    async def setup_connection(self, client_ws: WebSocket):
        """Setup connection to both client and Gemini Live"""
        self.client_websocket = client_ws
        
        try:
            # Connect to Gemini Live WebSocket
            self.gemini_websocket = await websockets.connect(
                self.ws_url,
                additional_headers={"Content-Type": "application/json"}
            )
            
            # Send setup message to Gemini
            await self.send_setup_message()
            
            # Wait for setup response
            setup_response = await self.gemini_websocket.recv()
            response_data = json.loads(setup_response)
            logger.info(f"Setup response: {response_data}")
            
            self.is_connected = True
            logger.info("✅ Connected to Gemini Live WebSocket")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Gemini Live: {e}")
            return False

    async def send_setup_message(self):
        """Send initial setup configuration to Gemini"""
        setup = {
            "setup": {
                "model": self.model,
                "generation_config": {
                    "response_modalities": ["AUDIO", "TEXT"]
                },
                "system_instruction": {
                    "parts": [
                        {
                            "text": """You are a helpful AI assistant. Respond naturally to user questions and provide helpful information. Keep your responses conversational and engaging."""
                        }
                    ]
                }
            }
        }
        
        await self.gemini_websocket.send(json.dumps(setup))
        logger.info("📤 Sent setup message to Gemini Live")

    async def send_text_message(self, text: str):
        """Send a text message to Gemini Live"""
        if not self.gemini_websocket or not self.is_connected:
            logger.warning("⚠️ Not connected to Gemini Live")
            return
            
        message = {
            "client_content": {
                "turns": [{"role": "user", "parts": [{"text": text}]}],
                "turn_complete": True,
            }
        }
        
        await self.gemini_websocket.send(json.dumps(message))
        logger.info(f"📤 Sent text message: {text}")

    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to Gemini Live"""
        if not self.gemini_websocket or not self.is_connected:
            logger.warning("⚠️ Not connected to Gemini Live")
            return
            
        # Convert audio bytes to base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        message = {
            "client_content": {
                "turns": [{
                    "role": "user", 
                    "parts": [{
                        "inline_data": {
                            "mime_type": "audio/pcm;rate=16000",
                            "data": audio_base64
                        }
                    }]
                }],
                "turn_complete": False,  # Keep turn open for more audio
            }
        }
        
        await self.gemini_websocket.send(json.dumps(message))
        logger.debug(f"📤 Sent {len(audio_data)} bytes of audio")

    async def send_turn_complete(self):
        """Signal that the current turn is complete"""
        if not self.gemini_websocket or not self.is_connected:
            return
            
        message = {
            "client_content": {
                "turn_complete": True
            }
        }
        
        await self.gemini_websocket.send(json.dumps(message))
        logger.info("📋 Sent turn complete signal")

    async def handle_gemini_responses(self):
        """Handle responses from Gemini Live WebSocket"""
        try:
            async for raw_response in self.gemini_websocket:
                if not self.session_active:
                    break
                    
                try:
                    response = json.loads(raw_response)
                    logger.debug(f"📥 Received response: {str(response)[:200]}...")
                    
                    # Handle server content (audio/text responses)
                    server_content = response.get("serverContent")
                    if server_content:
                        await self.handle_server_content(server_content)
                    
                    # Handle tool calls (if implemented later)
                    tool_call = response.get("toolCall")
                    if tool_call:
                        await self.handle_tool_call(tool_call)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse Gemini response: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing Gemini response: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error in Gemini response handler: {e}")

    async def handle_server_content(self, server_content: Dict[str, Any]):
        """Process server content from Gemini"""
        model_turn = server_content.get("modelTurn")
        if not model_turn:
            return
            
        parts = model_turn.get("parts", [])
        
        for part in parts:
            # Handle text response
            text = part.get("text")
            if text:
                logger.info(f"📝 Gemini text response: {text}")
                await self.send_to_client({
                    "type": "text_response",
                    "text": text
                })
            
            # Handle audio response
            inline_data = part.get("inlineData")
            if inline_data:
                mime_type = inline_data.get("mimeType", "")
                if "audio" in mime_type:
                    audio_data = inline_data.get("data")
                    logger.info(f"🎵 Received audio data: {len(audio_data) if audio_data else 0} chars")
                    
                    await self.send_to_client({
                        "type": "tts_audio",
                        "audio_data": audio_data,
                        "text": text or ""
                    })
        
        # Check if turn is complete
        if server_content.get("turnComplete"):
            logger.info("✅ Gemini turn completed")
            await self.send_to_client({
                "type": "turn_complete",
                "message": "AI finished responding"
            })

    async def handle_tool_call(self, tool_call: Dict[str, Any]):
        """Handle tool/function calls from Gemini (placeholder for future RAG integration)"""
        logger.info(f"🔧 Tool call received: {tool_call}")
        
        # For now, just send a simple response
        function_calls = tool_call.get("functionCalls", [])
        for fc in function_calls:
            response_msg = {
                "tool_response": {
                    "function_responses": [{
                        "id": fc.get("id"),
                        "name": fc.get("name"),
                        "response": {"result": {"string_value": "Function call received but not implemented yet"}}
                    }]
                }
            }
            
            await self.gemini_websocket.send(json.dumps(response_msg))
            logger.info(f"📤 Sent tool response for {fc.get('name')}")

    async def send_to_client(self, message: Dict[str, Any]):
        """Send message to frontend client"""
        if self.client_websocket:
            try:
                await self.client_websocket.send_json(message)
                logger.debug(f"📤 Sent to client: {message.get('type')}")
            except Exception as e:
                logger.error(f"❌ Failed to send to client: {e}")

    async def close(self):
        """Close connections and cleanup"""
        self.session_active = False
        
        if self.gemini_websocket:
            await self.gemini_websocket.close()
            logger.info("🔌 Closed Gemini WebSocket connection")
            
        self.is_connected = False
        logger.info("✅ SimpleGeminiLive session closed")


async def create_simple_gemini_session(api_key: str) -> SimpleGeminiLive:
    """Factory function to create a new SimpleGeminiLive session"""
    return SimpleGeminiLive(api_key)