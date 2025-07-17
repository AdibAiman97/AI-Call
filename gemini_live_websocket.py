import asyncio
import json
import base64
import time
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import threading
import numpy as np

# Official Google Gen AI SDK imports
from google import genai
from google.genai.types import (
    AudioTranscriptionConfig,
    AutomaticActivityDetection,
    Blob,
    Content,
    EndSensitivity,
    LiveConnectConfig,
    Part,
    PrebuiltVoiceConfig,
    RealtimeInputConfig,
    SpeechConfig,
    StartSensitivity,
    Tool,
    VoiceConfig,
    FunctionDeclaration,
    Schema,
    Type
)

# Import existing components
from database.connection import get_db
from services.transcript_crud import create_session_message

# Configuration for Gemini Live API via official SDK
@dataclass
class GeminiLiveConfig:
    """Configuration for Gemini Live API connection with Vertex AI using official SDK"""
    project_id: str
    location: str = "us-central1"
    model_name: str = "gemini-2.0-flash-live-preview-04-09"  # Official Vertex AI Live model
    voice_name: str = "Aoede"  # Available voices: Aoede, Charon, Fenrir, Kore, Puck
    max_output_tokens: int = 1000
    temperature: float = 0.7

class GeminiLiveSession:
    """Handles Gemini Live connection using official Google Gen AI SDK with Vertex AI"""
    
    def __init__(self, config: GeminiLiveConfig, rag_system, call_session_id: str, call_summary: Optional[str] = None):
        self.config = config
        self.rag_system = rag_system
        self.call_session_id = call_session_id
        self.call_summary = call_summary
        
        # Initialize the official Google Gen AI client with Vertex AI backend
        self.client = genai.Client(
            vertexai=True, 
            project=self.config.project_id, 
            location=self.config.location
        )
        print(f"✅ Google Gen AI Client initialized with Vertex AI for project: {self.config.project_id}")
        
        # Session state
        self.session = None
        self.session_active = False
        self.conversation_history = []
        
        # Threading
        self.lock = threading.Lock()
        
        # Database
        self.db = next(get_db())
    
    def _create_rag_tool(self) -> Tool:
        """Create the RAG tool for property database queries using official SDK format"""
        return Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="query_property_database",
                    description="Query the property database for information about properties, pricing, amenities, and availability at Gamuda Cove",
                    parameters=Schema(
                        type=Type.OBJECT,
                        properties={
                            "query": Schema(
                                type=Type.STRING,
                                description="The search query to find relevant property information"
                            )
                        },
                        required=["query"]
                    )
                )
            ]
        )
    
    def _create_appointment_tool(self) -> Tool:
        """Create the appointment scheduling tool using official SDK format"""
        return Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="schedule_appointment",
                    description="Schedule a viewing appointment at Gamuda Cove sales gallery",
                    parameters=Schema(
                        type=Type.OBJECT,
                        properties={
                            "customer_name": Schema(
                                type=Type.STRING,
                                description="Full name of the customer"
                            ),
                            "property_type": Schema(
                                type=Type.STRING,
                                description="Type of property interested in (Semi-detached, Terrace, Bungalow, Apartments)"
                            ),
                            "purchase_purpose": Schema(
                                type=Type.STRING,
                                description="Purpose of purchase (investment, own stay, family, etc.)"
                            ),
                            "preferred_datetime": Schema(
                                type=Type.STRING,
                                description="Preferred appointment date and time"
                            ),
                            "phone_number": Schema(
                                type=Type.STRING,
                                description="Customer's contact phone number"
                            )
                        },
                        required=["customer_name", "property_type"]
                    )
                )
            ]
        )
    
    def _build_system_instructions(self) -> str:
        """Build comprehensive system instructions for the voice assistant"""
        
        base_instructions = """
        You are Gina, a friendly and professional sales consultant for Gamuda Cove sales gallery located in Bandar Gamuda Cove, Kuala Langat, Selangor.
        
        VOICE CONVERSATION GUIDELINES:
        - This is a voice conversation, so keep responses conversational and natural
        - Use casual language with phrases like "Well...", "You know...", "I mean..."
        - Keep responses concise and engaging - aim for 1-2 sentences per response
        - Mirror the customer's energy and speaking style
        - Always sound enthusiastic about the properties
        
        YOUR ROLE:
        - Help customers learn about properties at Gamuda Cove
        - Guide conversations toward scheduling a viewing appointment
        - Answer questions about townships, property details, pricing, and amenities
        - Use the query_property_database function to get specific property information
        - Use the schedule_appointment function when customers want to book a viewing
        
        CONVERSATION FLOW:
        1. Greet customers warmly and introduce yourself
        2. Ask how you can help them today
        3. Listen to their needs and use the property database to provide relevant information
        4. Guide them toward booking an appointment to visit the sales gallery
        
        APPOINTMENT BOOKING PROCESS:
        When customers show interest, gather:
        1. Their full name
        2. Property type preference (Semi-detached, Terrace, Bungalow, Apartments)
        3. Purpose of purchase (investment, own stay, family)
        4. Preferred appointment time
        5. Contact phone number
        
        Then use the schedule_appointment function to book their visit.
        
        IMPORTANT RULES:
        - NEVER say "I don't have information" or "I don't know"
        - Always use the query_property_database function to find relevant information
        - If specific details aren't available, redirect to appointment booking
        - Convert all numbers to words in speech (e.g., "two thousand" not "2000")
        - Keep responses under 50 words for natural conversation flow
        - When customers provide appointment details, immediately use schedule_appointment function
        """
        
        # Add call summary if available
        if self.call_summary:
            base_instructions += f"\n\nPREVIOUS CALL SUMMARY:\n{self.call_summary}"
        
        return base_instructions
    
    async def connect(self):
        """Prepare Gemini Live configuration and validate it"""
        try:
            print("🔄 Preparing Gemini Live configuration...")
            
            # Create the live connection configuration with built-in VAD
            config = LiveConnectConfig(
                response_modalities=["AUDIO"],
                # TEMPORARILY DISABLE VAD FOR TESTING
                # realtime_input_config=RealtimeInputConfig(
                #     automatic_activity_detection=AutomaticActivityDetection(
                #         disabled=True,  # DISABLE VAD for testing
                #         start_of_speech_sensitivity=StartSensitivity.START_SENSITIVITY_LOW,
                #         end_of_speech_sensitivity=EndSensitivity.END_SENSITIVITY_LOW,
                #         prefix_padding_ms=200,  # Capture complete words
                #         silence_duration_ms=1000,  # 2 seconds for natural conversation pauses
                #     )
                # ),
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(
                            voice_name=self.config.voice_name
                        )
                    )
                ),
                system_instruction=Content(
                    parts=[Part(text=self._build_system_instructions())]
                ),
                tools=[
                    self._create_rag_tool(),
                    self._create_appointment_tool()
                ],
                generation_config={
                    "max_output_tokens": self.config.max_output_tokens,
                    "temperature": self.config.temperature
                }
            )
            
            # Store the connection config and mark as ready
            self.connection_config = config
            self.session_active = True
            
            print("✅ Gemini Live configuration prepared successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to prepare Gemini Live configuration: {e}")
            return False
    
    async def send_audio_chunk(self, audio_data: bytes):
        """Send audio chunk to Gemini Live using official SDK"""
        if not hasattr(self, 'session') or not self.session or not self.session_active:
            print("⚠️ Session not ready, skipping audio chunk")
            return
        
        try:
            # Send audio to Gemini Live session with proper format
            # Note: audio_data should be raw bytes in 16kHz, 16-bit PCM format
            await self.session.send_realtime_input(
                audio=Blob(data=audio_data, mime_type="audio/pcm;rate=16000")
            )
            
        except Exception as e:
            print(f"❌ Error sending audio chunk: {e}")
    
    async def handle_function_call(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        """Handle function calls from Gemini Live (RAG queries and appointments)"""
        function_name = function_call.get("name")
        function_args = function_call.get("args", {})
        
        if function_name == "query_property_database":
            query = function_args.get("query", "")
            
            try:
                print(f"🔍 RAG Query: {query}")
                
                # Use the existing RAG system to get property information
                if self.rag_system:
                    # Use the streaming method to get response
                    response_parts = []
                    async for chunk in self.rag_system.generate_response_stream(
                        prompt=query,
                        use_memory=False,  # Don't use RAG memory, let Gemini Live handle conversation memory
                        call_summary=self.call_summary
                    ):
                        response_parts.append(chunk)
                    
                    result = "".join(response_parts)
                    
                    # Log the RAG response
                    print(f"📋 RAG Response: {result[:100]}...")
                    
                    return {
                        "name": function_name,
                        "response": {
                            "result": result
                        }
                    }
                else:
                    return {
                        "name": function_name,
                        "response": {
                            "result": "Property database is currently unavailable. Please visit our sales gallery for detailed information."
                        }
                    }
                    
            except Exception as e:
                print(f"❌ Error in RAG query: {e}")
                return {
                    "name": function_name,
                    "response": {
                        "result": "I'd be happy to discuss that when you visit our sales gallery. When would work best for you?"
                    }
                }
        
        elif function_name == "schedule_appointment":
            try:
                print(f"📅 Scheduling appointment with details: {function_args}")
                
                # Log the appointment request to database
                appointment_details = {
                    "customer_name": function_args.get("customer_name", ""),
                    "property_type": function_args.get("property_type", ""),
                    "purchase_purpose": function_args.get("purchase_purpose", ""),
                    "preferred_datetime": function_args.get("preferred_datetime", ""),
                    "phone_number": function_args.get("phone_number", "")
                }
                
                # Save appointment request to database
                appointment_summary = f"Appointment request: {appointment_details['customer_name']} interested in {appointment_details['property_type']} for {appointment_details['purchase_purpose']}, preferred time: {appointment_details['preferred_datetime']}"
                
                create_session_message(
                    self.db,
                    session_id=self.call_session_id,
                    message=appointment_summary,
                    message_by="SYSTEM"
                )
                
                return {
                    "name": function_name,
                    "response": {
                        "result": f"Perfect! I've scheduled your appointment for {appointment_details['preferred_datetime']}. Our team will contact you at {appointment_details['phone_number']} to confirm the details. Looking forward to seeing you at our Gamuda Cove sales gallery!"
                    }
                }
                
            except Exception as e:
                print(f"❌ Error scheduling appointment: {e}")
                return {
                    "name": function_name,
                    "response": {
                        "result": "I'd be happy to help you schedule an appointment. Let me connect you with our booking team to finalize the details."
                    }
                }
        
        return {
            "name": function_name,
            "response": {
                "result": "Function not implemented"
            }
        }
    
    async def connect_and_listen(self, client_websocket):
        """Connect to Gemini Live and listen for responses (atomic operation)"""
        try:
            if not hasattr(self, 'connection_config'):
                print("❌ No connection config prepared")
                return False
            
            print("🔗 Establishing Gemini Live connection and starting listener...")
            
            # Use the official SDK's async context manager pattern
            async with self.client.aio.live.connect(
                model=self.config.model_name,
                config=self.connection_config
            ) as session:
                print("✅ Gemini Live connected successfully!")
                self.session = session  # Store for other methods
                
                # Send success message immediately after real connection
                try:
                    await client_websocket.send_json({
                        "type": "connected",
                        "message": "Connected to Gemini Live. You can start speaking now.",
                        "session_id": self.call_session_id
                    })
                    print("📤 Success message sent to client")
                except Exception as send_error:
                    print(f"⚠️ Client disconnected before receiving success message: {send_error}")
                    return False
                
                audio_buffer = []
                
                async for message in session.receive():
                    if not self.session_active:
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
                                            create_session_message(
                                                self.db,
                                                session_id=self.call_session_id,
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
                                        await client_websocket.send_json({
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
                                    await client_websocket.send_json({
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
                                response = await self.handle_function_call({
                                    "name": function_call.name,
                                    "args": function_call.args
                                })
                                
                                # Send function response back to Gemini Live
                                await session.send_tool_response(
                                    function_response=response
                                )
                                print(f"📤 Sent tool response: {response['name']}")
                        
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        continue
                
                # If we exit the async for loop normally, return True
                return True
                        
        except Exception as e:
            print(f"❌ Error in connect_and_listen: {e}")
            return False
        finally:
            self.session_active = False
            self.session = None
    
    async def close(self):
        """Close the Gemini Live session"""
        self.session_active = False
        
        # Clear session references (context manager handles cleanup automatically)
        self.session = None
        
        # Close database connection
        if hasattr(self, 'db'):
            self.db.close()
            
        print("🔌 Gemini Live session cleaned up")

class GeminiLiveManager:
    """Manages multiple Gemini Live sessions using official SDK"""
    
    def __init__(self, config: GeminiLiveConfig):
        self.config = config
        self.active_sessions: Dict[str, GeminiLiveSession] = {}
        self.lock = threading.Lock()
    
    async def create_session(self, call_session_id: str, rag_system, call_summary: Optional[str] = None) -> GeminiLiveSession:
        """Create a new Gemini Live session"""
        
        with self.lock:
            # Clean up any existing session
            if call_session_id in self.active_sessions:
                old_session = self.active_sessions[call_session_id]
                asyncio.create_task(old_session.close())
            
            # Create new session
            session = GeminiLiveSession(
                config=self.config,
                rag_system=rag_system,
                call_session_id=call_session_id,
                call_summary=call_summary
            )
            
            self.active_sessions[call_session_id] = session
            
        return session
    
    async def close_session(self, call_session_id: str):
        """Close a specific session"""
        with self.lock:
            if call_session_id in self.active_sessions:
                session = self.active_sessions.pop(call_session_id)
                await session.close()
    
    async def close_all_sessions(self):
        """Close all active sessions"""
        with self.lock:
            sessions_to_close = list(self.active_sessions.values())
            self.active_sessions.clear()
        
        for session in sessions_to_close:
            await session.close()

# Global manager instance
gemini_live_manager = None

def get_gemini_live_manager() -> GeminiLiveManager:
    """Get the global Gemini Live manager instance"""
    global gemini_live_manager
    
    if gemini_live_manager is None:
        # Import configuration from config.py
        from config import GCP_PROJECT_ID, GCP_LOCATION
        
        # Initialize with Vertex AI configuration using official SDK
        config = GeminiLiveConfig(
            project_id=GCP_PROJECT_ID,
            location=GCP_LOCATION,
            model_name="gemini-2.0-flash-live-preview-04-09"  # Official Vertex AI Live model
        )
        
        gemini_live_manager = GeminiLiveManager(config)
    
    return gemini_live_manager 