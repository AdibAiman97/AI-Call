#!/usr/bin/env python3
"""
FastAPI Server for Gemini Live API
Run with: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import json
import os
import time
import wave
from typing import Optional, List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
import logging
from websockets.asyncio.client import connect
import numpy as np
from rag_integration import rag_service, initialize_rag, process_rag_query, get_rag_health
import io
import tempfile
import ssl
import certifi

# Import existing routers
from api.customer_router import router as customer_router
from api.call_session_router import router as call_session_router
from api.transcript_router import router as transcript_router
from api.appointment_router import router as appointment_router
from api.property_router import router as property_router
from api.pdf_router import router as pdf_router

# Import database components
from database.connection import engine, Base, get_db, SessionLocal
from services.call_session import CallSessionService
from database.schemas import CallSessionBase
from services.transcript_crud import create_session_message
from database.models.call_session import CallSession
from database.models.transcript import Transcript

# Import AI Services
from ai_services.call_suggestion_admin import get_suggestion_from_agent
from ai_services.call_suggestion_customer import generate_caller_suggestions
from ai_services.call_summarized_context import summarize_text
from ai_services.sentiment_tool import analyze_sentiment_from_transcript

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_live.log'),
        logging.StreamHandler()  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Load API key from environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# =============================================================================
# AI SERVICES INTEGRATION (USING IMPORTS)
# =============================================================================
# All AI services are now imported from their respective files in ai_services/
# This keeps the server.py clean and maintains separation of concerns

def get_session_conversation(session_id: int) -> dict:
    """
    Get session conversation directly from database.
    Returns dict with success status and conversation data.
    """
    db = SessionLocal()
    try:
        transcripts = (
            db.query(Transcript)
            .filter(Transcript.session_id == session_id)
            .order_by(Transcript.created_at)
            .all()
        )

        if not transcripts:
            return {
                "success": False,
                "error": f"No transcripts found for session ID {session_id}",
            }

        conversation = [
            {
                "message_by": t.message_by,
                "message": t.message,
                "created_at": t.created_at.isoformat(),
            }
            for t in transcripts
        ]

        return {"success": True, "session_id": session_id, "conversation": conversation}

    except Exception as e:
        return {"success": False, "error": f"Database error: {str(e)}"}
    finally:
        db.close()

# =============================================================================

# =============================================================================
# POST-CALL AI PROCESSING FUNCTION
# =============================================================================

async def process_call_session_ai(call_session_id: int):
    """
    Process a call session with all AI services after the call ends.
    This runs asynchronously to avoid blocking the WebSocket cleanup.
    """
    try:
        logger.info(f"🤖 Processing call session {call_session_id} with AI services")
        print(f"🤖 Processing call session {call_session_id} with AI services")
        
        # Get session conversation first
        conversation_data = get_session_conversation(call_session_id)
        
        if not conversation_data.get("success"):
            logger.warning(f"No conversation found for session {call_session_id}")
            print(f"⚠️ No conversation found for session {call_session_id}")
            return
        
        conversation = conversation_data["conversation"]
        
        if len(conversation) == 0:
            logger.warning(f"Empty conversation for session {call_session_id}")
            print(f"⚠️ Empty conversation for session {call_session_id}")
            return
        
        # Format conversation for processing
        formatted_conversation = [
            {"role": msg["message_by"], "content": msg["message"]} 
            for msg in conversation
        ]
        
        logger.info(f"🤖 Found {len(conversation)} messages in session {call_session_id}")
        print(f"🤖 Found {len(conversation)} messages in session {call_session_id}")
        
        # 1. Generate and store conversation summary
        try:
            logger.info(f"📝 Generating summary for session {call_session_id}")
            print(f"📝 Generating summary for session {call_session_id}")
            summary = summarize_text(formatted_conversation, call_session_id)
            logger.info(f"✅ Summary generated and stored for session {call_session_id}")
            print(f"✅ Summary generated: {summary[:100]}...")
        except Exception as e:
            logger.error(f"❌ Error generating summary for session {call_session_id}: {e}")
            print(f"❌ Error generating summary: {e}")
        
        # 2. Generate and store customer suggestions  
        try:
            logger.info(f"💡 Generating customer suggestions for session {call_session_id}")
            print(f"💡 Generating customer suggestions for session {call_session_id}")
            customer_suggestions = generate_caller_suggestions(formatted_conversation, call_session_id)
            logger.info(f"✅ Customer suggestions generated and stored for session {call_session_id}")
            print(f"✅ Customer suggestions generated: {customer_suggestions[:100]}...")
        except Exception as e:
            logger.error(f"❌ Error generating customer suggestions for session {call_session_id}: {e}")
            print(f"❌ Error generating customer suggestions: {e}")
        
        # 3. Analyze sentiment for each transcript
        try:
            logger.info(f"😊 Analyzing sentiment for session {call_session_id}")
            print(f"😊 Analyzing sentiment for session {call_session_id}")
            
            sentiment_results = []
            db = SessionLocal()
            try:
                transcripts = (
                    db.query(Transcript)
                    .filter(Transcript.session_id == call_session_id)
                    .order_by(Transcript.created_at)
                    .all()
                )
                
                for transcript in transcripts:
                    try:
                        sentiment_result = analyze_sentiment_from_transcript(transcript.id)
                        sentiment_data = json.loads(sentiment_result)
                        if sentiment_data.get("success"):
                            sentiment_results.append(sentiment_data)
                            sentiment_info = sentiment_data.get("sentiment_analysis", {})
                            sentiment = sentiment_info.get("sentiment", "Unknown")
                            confidence = sentiment_info.get("confidence", 0)
                            print(f"   📊 Transcript {transcript.id}: {sentiment} (confidence: {confidence:.2f})")
                        else:
                            logger.warning(f"Sentiment analysis failed for transcript {transcript.id}")
                    except Exception as transcript_error:
                        logger.error(f"Error analyzing sentiment for transcript {transcript.id}: {transcript_error}")
                        
            finally:
                db.close()
            
            logger.info(f"✅ Sentiment analysis completed for {len(sentiment_results)} transcripts in session {call_session_id}")
            print(f"✅ Sentiment analysis completed for {len(sentiment_results)} transcripts")
            
        except Exception as e:
            logger.error(f"❌ Error analyzing sentiment for session {call_session_id}: {e}")
            print(f"❌ Error analyzing sentiment: {e}")
        
        # 4. Update call session with end time, duration, and sentiment counts
        try:
            logger.info(f"📊 Updating call session {call_session_id} with call data")
            print(f"📊 Updating call session {call_session_id} with call data")
            
            db = SessionLocal()
            call_session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
            
            if call_session:
                # Import datetime for end time calculation
                from datetime import datetime, timezone
                
                # Set end time to current time
                end_time = datetime.now(timezone.utc)
                call_session.end_time = end_time
                
                # Calculate duration in seconds
                if call_session.start_time:
                    duration = end_time - call_session.start_time
                    call_session.duration_secs = int(duration.total_seconds())
                    print(f"   ⏱️ Call duration: {call_session.duration_secs} seconds")
                
                # Calculate sentiment counts from sentiment analysis results
                positive_count = 0
                neutral_count = 0
                negative_count = 0
                
                for sentiment_data in sentiment_results:
                    sentiment_info = sentiment_data.get("sentiment_analysis", {})
                    sentiment = sentiment_info.get("sentiment", "").lower()
                    
                    if sentiment == "positive":
                        positive_count += 1
                    elif sentiment == "negative":
                        negative_count += 1
                    else:
                        neutral_count += 1
                
                call_session.positive = positive_count
                call_session.neutral = neutral_count
                call_session.negative = negative_count
                
                print(f"   😊 Sentiment counts - Positive: {positive_count}, Neutral: {neutral_count}, Negative: {negative_count}")
                
                # Extract key words from conversation (simple approach)
                try:
                    # Create a list of all words from the conversation
                    all_text = " ".join([msg["content"] for msg in formatted_conversation])
                    # Simple keyword extraction - you could enhance this with NLP
                    words = all_text.split()
                    # Get unique words longer than 3 characters, limit to first 10
                    key_words_list = list(set([word.strip('.,!?').lower() for word in words 
                                             if len(word.strip('.,!?')) > 3]))[:10]
                    call_session.key_words = ", ".join(key_words_list)
                    print(f"   🔑 Key words: {call_session.key_words}")
                except Exception as kw_error:
                    print(f"   ⚠️ Could not extract key words: {kw_error}")
                
                # Commit all changes
                db.commit()
                db.refresh(call_session)
                
                logger.info(f"✅ Call session {call_session_id} updated successfully")
                print(f"✅ Call session updated successfully")
                print(f"   📞 End time: {call_session.end_time}")
                print(f"   ⏱️ Duration: {call_session.duration_secs} seconds")
                print(f"   📊 Sentiment: +{positive_count} ={neutral_count} -{negative_count}")
                
            else:
                logger.warning(f"Call session {call_session_id} not found for update")
                print(f"⚠️ Call session {call_session_id} not found for update")
                
            db.close()
            
        except Exception as e:
            logger.error(f"❌ Error updating call session {call_session_id}: {e}")
            print(f"❌ Error updating call session: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. Log completion
        logger.info(f"🎉 AI processing and call session update completed for session {call_session_id}")
        print(f"🎉 AI processing and call session update completed for session {call_session_id}")
        
    except Exception as e:
        logger.error(f"❌ Critical error in AI processing for session {call_session_id}: {e}")
        print(f"❌ Critical error in AI processing: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# END POST-CALL AI PROCESSING FUNCTION
# =============================================================================

# Configuration
GEMINI_HOST = 'generativelanguage.googleapis.com'
GEMINI_MODEL = 'models/gemini-2.0-flash-live-001'

# VAD Configuration - Set to False if having voice detection issues
ENABLE_VAD = True  # Change to False to disable VAD temporarily

# Audio configuration
class AudioConfig:
    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.format = 'S16_LE'
        self.channels = channels
    
    @property
    def sample_size(self):
        return 2  # 16-bit
    
    @property
    def frame_size(self):
        return self.channels * self.sample_size

# Audio configuration as per Google documentation
# Input: 16-bit PCM, 16kHz, mono
# Output: 24kHz
INPUT_AUDIO_CONFIG = AudioConfig(sample_rate=16000, channels=1)  # Input to Gemini
OUTPUT_AUDIO_CONFIG = AudioConfig(sample_rate=24000, channels=1)  # Output from Gemini

# Add CORS for Vue frontend
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting up FastAPI server...")
    rag_initialized = await initialize_rag()
    if rag_initialized:
        logger.info("RAG service initialized successfully")
    else:
        logger.warning("RAG service failed to initialize - continuing without RAG")
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI server...")
    rag_service.close()

# Create FastAPI app
app = FastAPI(title="Gemini Live API Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],  # Vue dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(customer_router)
app.include_router(call_session_router)
app.include_router(transcript_router)
app.include_router(appointment_router)
app.include_router(property_router)
app.include_router(pdf_router)

def encode_audio_input(data: bytes, config: AudioConfig) -> dict:
    """Build message with user input audio bytes."""
    return {
        'realtimeInput': {
            'mediaChunks': [{
                'mimeType': f'audio/pcm;rate={config.sample_rate}',
                'data': base64.b64encode(data).decode('UTF-8'),
            }],
        },
    }

def encode_text_input(text: str) -> dict:
    """Builds message with user input text."""
    return {
        'clientContent': {
            'turns': [{
                'role': 'USER',
                'parts': [{'text': text}],
            }],
            'turnComplete': True,
        },
    }

def decode_audio_output(input_msg: dict) -> bytes:
    """Returns byte string with model output audio."""
    result = []
    content_input = input_msg.get('serverContent', {})
    content = content_input.get('modelTurn', {})
    for part in content.get('parts', []):
        data = part.get('inlineData', {}).get('data', '')
        if data:
            result.append(base64.b64decode(data))
    return b''.join(result)

def extract_text_response(input_msg: dict) -> str:
    """Extract text response from Gemini message."""
    content_input = input_msg.get('serverContent', {})
    content = content_input.get('modelTurn', {})
    for part in content.get('parts', []):
        if 'text' in part:
            return part['text']
    return ""

def save_audio_to_file(audio_data: bytes, config: AudioConfig, filename: str):
    """Save audio data to WAV file."""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(config.channels)
        wav_file.setsampwidth(config.sample_size)
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(audio_data)

# Store active connections
active_connections = {}

class GeminiLiveConnection:
    """Manages connection to Gemini Live API for a WebSocket client."""
    
    def __init__(self, client_websocket: WebSocket, call_session_id: int):
        self.client_ws = client_websocket
        self.gemini_ws: Optional[object] = None
        self.is_connected = False
        self.audio_chunks = []
        self.user_transcript_buffer = ""  # Buffer for accumulating user transcript chunks
        self.ai_transcript_buffer = ""    # Buffer for accumulating AI transcript chunks
        self.last_user_transcript_time = 0
        self.last_ai_transcript_time = 0
        self.last_sent_transcript = ""    # Store last sent transcript for supplementation  
        self.pending_transcript_task = None  # Track pending delayed transcript
        self.pending_transcript_content = ""  # Store content of pending transcript
        self.proper_names = ["Mori Pines", "Gamuda Cove", "Enso Woods"]  # Key property names to preserve
        self.transcript_timeout = 2.0  # Seconds to wait before sending incomplete transcript
        self.last_rag_result = ""  # Store last RAG result for verification
        self.last_rag_query = ""   # Store last RAG query for verification
        self.call_session_id = call_session_id  # Store call session ID for database operations
        
    async def connect_to_gemini(self):
        """Connect to Gemini Live API."""
        try:
            ws_url = f'wss://{GEMINI_HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}'
            print(f"🔗 Connecting to Gemini Live...")
            logger.info("Connecting to Gemini Live API...")

            ssl_context = ssl.create_default_context(cafile=certifi.where())
            self.gemini_ws = await connect(ws_url, ssl=ssl_context)
            
            # Define RAG function for Gemini to call
            rag_function_declaration = {
                "name": "search_knowledge_base", 
                "description": "MANDATORY: Search the comprehensive knowledge base containing information about properties, real estate projects, pricing, amenities, layouts, and related details. You MUST call this function whenever users mention ANY property-related information, including property names like 'Mori Pines', 'Gamuda Cove', pricing questions, amenities, or any real estate topics. CRITICAL: You must use the returned results as your response - never say you don't have information if this function returns data.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's property-related question or search query. Pass the exact user input that contains property information."
                        }
                    },
                    "required": ["query"]
                }
            }
            
            # Send initial setup with audio response modality, transcription, optional VAD, and RAG tool
            setup_message = {
                "setup": {
                    "model": GEMINI_MODEL,
                    "generation_config": {
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {"prebuilt_voice_config": {"voice_name": "Aoede"}}
                        }
                    },
                    "output_audio_transcription": {},
                    "input_audio_transcription": {},
                }
            }
            
            # Add VAD configuration only if enabled
            if ENABLE_VAD:
                setup_message["setup"]["realtime_input_config"] = {
                    # Voice Activity Detection (VAD) configuration - More sensitive settings
                    # Based on Example 8 from Google's Gemini Live API reference
                    "automatic_activity_detection": {
                        "disabled": False,  # Enable VAD
                        "start_of_speech_sensitivity": "START_SENSITIVITY_HIGH",  # Very sensitive to speech start
                        "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",     # More sensitive to speech end
                        "prefix_padding_ms": 100,     # Include 100ms before detected speech start
                        "silence_duration_ms": 300    # Wait 300ms of silence before considering speech ended
                    }
                }
                print(f"🎙️ VAD enabled with HIGH sensitivity settings")
            else:
                print(f"🎙️ VAD disabled - using continuous audio streaming")
            
            # Add system instruction and tools
            setup_message["setup"]["system_instruction"] = {
                "parts": [{
                    "text": """
                            You are a helpful AI assistant with access to a comprehensive knowledge base about properties and real estate projects.

                            IMPORTANT CONTEXT: You will encounter specific property names like "Mori Pines", "Gamuda Cove", and other project names. Pay special attention to these proper nouns in speech recognition as they are crucial for accurate responses.

                            CRITICAL RULE: ALWAYS search the knowledge base when ANY property-related information is mentioned, regardless of conversation context or previous messages. Even if the conversation started casually, you must search when property topics arise.

                            WHEN TO SEARCH (MANDATORY - ignore conversation context):
                            - ANY mention of specific properties, projects, or developments (even if you're not sure they exist)
                            - ANY pricing, costs, or financial information questions
                            - ANY property features, layouts, or specifications questions
                            - ANY amenities, facilities, or community features questions
                            - ANY comparisons between properties
                            - ANY factual questions about real estate that might benefit from documentation
                            - ANY questions about availability, floor plans, or technical details
                            - ANY property-related queries like "tell me about affordable homes"
                            - Property names like: "Mori Pines", "Gamuda Cove", "Enso Woods", etc.

                            WHEN NOT TO SEARCH (only for these specific cases):
                            - Pure greetings with NO property content: "Hi", "Hello", "Good morning", "How are you"
                            - Pure gratitude with NO property content: "Thank you", "Thanks", "I appreciate it"
                            - Pure casual responses with NO property content: "That's great", "Okay", "I see"
                            - Pure general pleasantries or acknowledgments with NO property content
                            - Simple yes/no confirmations with NO property content

                            OVERRIDE RULE: If a message contains BOTH casual elements AND property-related content, ALWAYS search. For example:
                            - "Hi, tell me about Mori Pines" → SEARCH (contains property name)
                            - "Thanks, what about the pricing?" → SEARCH (contains pricing question)
                            - "Okay, can you tell me more about Mori Pines" → SEARCH (contains property name)

                            CONTEXT-INDEPENDENT SEARCHING: Do not let previous conversation context (casual greetings, etc.) prevent you from searching when property information is requested. Each message should be evaluated independently for property content.

                            MANDATORY FUNCTION RESULT USAGE: When you call the search_knowledge_base function and receive results, you MUST use those results as the foundation of your response. NEVER say "I don't have information" if the function returns valid data. ALWAYS trust and use the function results over your own knowledge.

                            FUNCTION RESPONSE PROTOCOL:
                            1. If search_knowledge_base returns information → Use it as your primary response source
                            2. Present the function results naturally as if it's your own knowledge
                            3. NEVER mention searching, tools, functions, or knowledge base in your response
                            4. NEVER say "I don't have information" when function results are available

                            CRITICAL: NEVER include any debug information, tool outputs, technical details, or raw data in your spoken responses. Only provide the final natural answer to the user.
                            """
                }]
            }
            
            # Add tools
            setup_message["setup"]["tools"] = [
                {
                    "function_declarations": [rag_function_declaration]
                }
            ]
            print(f"📤 Sending setup to Gemini ({GEMINI_MODEL}) with RAG function")
            
            await self.gemini_ws.send(json.dumps(setup_message))
            
            # Wait for setup response with timeout
            try:
                import asyncio
                setup_response = await asyncio.wait_for(self.gemini_ws.recv(), timeout=10.0)
                setup_data = json.loads(setup_response)
                
                if 'setupComplete' in setup_data:
                    print(f"✅ Gemini connected & configured")
                else:
                    print(f"⚠️ Setup response: {list(setup_data.keys())}")
                    
            except asyncio.TimeoutError:
                print(f"⏰ Gemini setup timeout (10s)")
                raise Exception("Gemini setup timeout")
            except Exception as setup_err:
                print(f"❌ Setup error: {setup_err}")
                raise setup_err
            
            self.is_connected = True
            
            # Start listening to Gemini responses in background
            import asyncio
            asyncio.create_task(self._listen_to_gemini())
            print(f"👂 Listening for Gemini responses")
            
            # Wait a moment for any immediate responses
            await asyncio.sleep(1)
            print(f"✅ Setup complete, ready for interactions")
            
        except Exception as e:
            logger.error(f"Error connecting to Gemini: {e}")
            print(f"❌ GEMINI CONNECTION FAILED: {e}")
            await self.client_ws.send_json({"type": "error", "message": f"Failed to connect to Gemini: {e}"})
    
    async def _listen_to_gemini(self):
        """Listen for messages from Gemini Live API."""
        try:
            async for message in self.gemini_ws:
                msg_data = json.loads(message)
                
                # Debug: Log all Gemini messages to understand the structure
                # print(f"🔍 FULL Gemini message: {json.dumps(msg_data, indent=2)}")
                
                # Debug: Check specifically for transcription fields
                server_content = msg_data.get('serverContent', {})
                # if 'input_transcription' in server_content:
                #     print(f"🎯 Found input_transcription: {server_content['input_transcription']}")
                # if 'output_transcription' in server_content:
                #     print(f"🎯 Found output_transcription: {server_content['output_transcription']}")
                # if 'inputTranscription' in server_content:
                #     print(f"🎯 Found inputTranscription: {server_content['inputTranscription']}")
                # if 'outputTranscription' in server_content:
                #     print(f"🎯 Found outputTranscription: {server_content['outputTranscription']}")
                
                # # Check all keys in serverContent
                # if server_content:
                #     print(f"🔑 serverContent keys: {list(server_content.keys())}")
                
                # Check for potential RAG-triggering content without function call
                model_turn = server_content.get('modelTurn', {})
                if model_turn and 'toolCall' not in msg_data:
                    parts = model_turn.get('parts', [])
                    for part in parts:
                        if 'text' in part:
                            text_content = part['text']
                            # Check if Gemini is saying "I don't have information" without calling RAG
                            if any(phrase in text_content.lower() for phrase in ["don't have", "no information", "not sure", "can't help"]):
                                print(f"⚠️ GEMINI CLAIMS NO INFO WITHOUT RAG CALL: '{text_content}'")
                                print(f"⚠️ This suggests Gemini didn't call search_knowledge_base function!")
                                print(f"⚠️ Full message: {json.dumps(msg_data, indent=2)}")
                            
                            # Check if Gemini is ignoring function results (more serious issue)
                            if any(phrase in text_content.lower() for phrase in ["don't have", "no information"]) and any(keyword in text_content.lower() for keyword in ["mori pines", "gamuda", "property"]):
                                print(f"🚨 CRITICAL: GEMINI IGNORING FUNCTION RESULTS!")
                                print(f"🚨 Gemini claims no info about property topics that should trigger RAG")
                                print(f"🚨 Response: '{text_content}'")
                                print(f"🚨 This indicates Gemini is not using function call results properly!")
                                
                                # Check if we recently sent RAG results that are being ignored
                                if hasattr(self, 'last_rag_result') and self.last_rag_result:
                                    print(f"🚨 RECENT RAG RESULT WAS IGNORED!")
                                    print(f"🚨 Last RAG query: '{self.last_rag_query}'")
                                    print(f"🚨 Last RAG result: '{self.last_rag_result[:200]}...'")
                                    print(f"🚨 Gemini should have used this information but ignored it!")
                
                # Extract transcripts using SDK-like patterns
                await self._extract_transcripts_sdk_style(msg_data)
                
                # Handle different types of server content
                if 'serverContent' in msg_data:
                    server_content = msg_data['serverContent']
                    
                    # Check for user transcript in grounding metadata
                    if 'groundingMetadata' in server_content:
                        grounding = server_content['groundingMetadata']
                        if 'groundingSupports' in grounding:
                            for support in grounding['groundingSupports']:
                                if 'segment' in support and 'text' in support['segment']:
                                    user_text = support['segment']['text']
                                    print(f"👤 User said (from grounding): \"{user_text}\"")
                                    await self.client_ws.send_json({
                                        "type": "user_transcript", 
                                        "content": user_text
                                    })
                    
                    # Check for user transcript (clientContent) - alternative location
                    if 'clientContent' in server_content:
                        client_content = server_content['clientContent']
                        turns = client_content.get('turns', [])
                        
                        for turn in turns:
                            if turn.get('role') == 'USER':
                                parts = turn.get('parts', [])
                                for part in parts:
                                    if 'text' in part:
                                        user_text = part['text']
                                        print(f"👤 User said: \"{user_text}\"")
                                        
                                        # Send user transcript to frontend
                                        await self.client_ws.send_json({
                                            "type": "user_transcript",
                                            "content": user_text
                                        })
                    
                    # Handle AI model responses
                    model_turn = server_content.get('modelTurn', {})
                    if model_turn:
                        parts = model_turn.get('parts', [])
                        
                        if parts:
                            text_parts = [p for p in parts if 'text' in p]
                            audio_parts = [p for p in parts if 'inlineData' in p]
                            
                            # Extract and send any text responses to frontend
                            if text_parts:
                                text_content = text_parts[0]['text']
                                text_preview = text_content[:60] + ('...' if len(text_content) > 60 else '')
                                print(f"📨 Gemini text: \"{text_preview}\"")
                                
                                # Debug: Check if text contains tool output artifacts
                                if 'tool_outputs' in text_content or 'hits' in text_content:
                                    print(f"⚠️ FOUND DEBUG ARTIFACTS IN GEMINI TEXT: {text_content}")
                                    print(f"⚠️ Full model turn: {json.dumps(model_turn, indent=2)}")
                                
                                # Clean any debug artifacts from the text
                                cleaned_text = self._clean_debug_artifacts(text_content)
                                
                                # Send text to frontend for transcript
                                await self.client_ws.send_json({
                                    "type": "text",
                                    "content": cleaned_text
                                })
                            
                            if audio_parts:
                                audio_size = len(audio_parts[0]['inlineData'].get('data', ''))
                                # print(f"🔊 Gemini audio: {audio_size} chars")
                    
                    if server_content.get('turnComplete'):
                        print(f"✅ Turn complete")
                    if server_content.get('interrupted'):
                        print(f"⏸️ AI response interrupted by user")
                        # Forward interruption to frontend immediately
                        await self.client_ws.send_json({
                            "type": "interrupted",
                            "message": "AI response interrupted by user speech"
                        })
                
                # Handle function calls from Gemini
                if 'toolCall' in msg_data:
                    await self._handle_tool_call(msg_data['toolCall'])
                else:
                    await self._process_gemini_message(msg_data)
                
        except Exception as e:
            logger.error(f"Error listening to Gemini: {e}")
            print(f"❌ GEMINI LISTENING ERROR: {e}")
            await self.client_ws.send_json({"type": "error", "message": f"Gemini connection error: {e}"})
    
    async def _extract_transcripts_sdk_style(self, msg_data: dict):
        """Extract transcripts using various possible field names."""
        try:
            server_content = msg_data.get('serverContent', {})
            
            # Check multiple possible field names for input transcription (user speech)
            input_transcript_fields = ['input_transcription', 'inputTranscription', 'userTranscript', 'speechRecognition']
            for field in input_transcript_fields:
                if field in server_content:
                    transcript_data = server_content[field]
                    print(f"🎯 Found user transcript field '{field}': {transcript_data}")
                    
                    # Try different text field names
                    text_content = None
                    for text_field in ['text', 'transcript', 'content']:
                        if text_field in transcript_data:
                            text_content = transcript_data[text_field]
                            break
                    
                    if text_content:
                        # Accumulate transcript chunks
                        self.user_transcript_buffer += text_content
                        print(f"👤 User transcript chunk: \"{text_content}\" (buffer: \"{self.user_transcript_buffer}\")")
                        print(f"👤 Chunk length: {len(text_content)}, Buffer length: {len(self.user_transcript_buffer)}")
                        
                        # Debug: Check if this chunk contains key words
                        key_words = ['mori', 'pines', 'gamuda', 'cove']
                        for word in key_words:
                            if word in text_content.lower():
                                print(f"🎯 FOUND KEY WORD '{word}' in transcript chunk: \"{text_content}\"")
                        
                        # Check for potential language detection issues
                        if any('\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' for char in text_content):
                            print(f"⚠️ Japanese characters detected in user speech: \"{text_content}\"")
                        if any('\u4E00' <= char <= '\u9FFF' for char in text_content):
                            print(f"⚠️ Chinese characters detected in user speech: \"{text_content}\"")
                        
                        # Improved transcript handling with proper name preservation
                        current_time = time.time()
                        should_send = False
                        
                        # Check if buffer contains proper names - if so, wait longer to get full context
                        contains_proper_name = any(name.lower() in self.user_transcript_buffer.lower() 
                                                 for name in self.proper_names)
                        
                        # Send conditions (more conservative to preserve proper names)
                        if (text_content.endswith('.') or text_content.endswith('?') or text_content.endswith('!')):
                            should_send = True
                            print(f"👤 Detected sentence ending punctuation: '{text_content[-1]}'")
                        elif len(self.user_transcript_buffer.strip().split()) >= 8:  # Increased from 4 to 8 words
                            should_send = True
                            print(f"👤 Sending transcript after {len(self.user_transcript_buffer.strip().split())} words")
                        elif contains_proper_name and len(self.user_transcript_buffer.strip().split()) >= 6:
                            should_send = True
                            print(f"👤 Sending transcript with proper name after {len(self.user_transcript_buffer.strip().split())} words")
                        elif current_time - self.last_user_transcript_time > self.transcript_timeout:
                            should_send = True
                            print(f"👤 Sending transcript due to timeout ({self.transcript_timeout}s)")
                        
                        if should_send and self.user_transcript_buffer.strip():
                            # Clean up the transcript before sending
                            cleaned_transcript = self._clean_transcript(self.user_transcript_buffer.strip())
                            print(f"👤 Sending complete user transcript: \"{cleaned_transcript}\"")
                            
                            # Store the transcript for potential supplementation
                            self.last_sent_transcript = cleaned_transcript
                            
                            # Save user transcript to database
                            try:
                                db = next(get_db())
                                create_session_message(
                                    db=db,
                                    session_id=self.call_session_id,
                                    message=cleaned_transcript,
                                    message_by="User"
                                )
                                print(f"💾 Saved user transcript to database")
                            except Exception as e:
                                print(f"❌ Error saving user transcript to database: {e}")
                            finally:
                                if db:
                                    db.close()
                            
                            # Send to frontend
                            await self.client_ws.send_json({
                                "type": "user_transcript",
                                "content": cleaned_transcript
                            })
                            self.user_transcript_buffer = ""  # Clear buffer
                        else:
                            print(f"👤 Buffering transcript: \"{self.user_transcript_buffer}\"")
                        
                        self.last_user_transcript_time = current_time
            
            # Check multiple possible field names for output transcription (AI speech)
            output_transcript_fields = ['output_transcription', 'outputTranscription', 'aiTranscript', 'speechOutput']
            for field in output_transcript_fields:
                if field in server_content:
                    transcript_data = server_content[field]
                    # print(f"🎯 Found AI transcript field '{field}': {transcript_data}")
                    
                    # Try different text field names
                    text_content = None
                    for text_field in ['text', 'transcript', 'content']:
                        if text_field in transcript_data:
                            text_content = transcript_data[text_field]
                            break
                    
                    if text_content:
                        # Accumulate AI transcript chunks
                        self.ai_transcript_buffer += text_content
                        # print(f"🤖 AI transcript chunk: \"{text_content}\" (buffer: \"{self.ai_transcript_buffer}\")")
                        
                        # Only send transcript when we detect definitive end of sentence
                        # Remove timing-based sending to avoid premature messages
                        if (text_content.endswith('.') or text_content.endswith('?') or 
                            text_content.endswith('!') or text_content.endswith('\n')):
                            
                            if self.ai_transcript_buffer.strip():
                                print(f"🤖 Sending complete AI transcript: \"{self.ai_transcript_buffer.strip()}\"")
                                
                                # Save AI transcript to database
                                try:
                                    db = next(get_db())
                                    create_session_message(
                                        db=db,
                                        session_id=self.call_session_id,
                                        message=self.ai_transcript_buffer.strip(),
                                        message_by="AI"
                                    )
                                    print(f"💾 Saved AI transcript to database")
                                except Exception as e:
                                    print(f"❌ Error saving AI transcript to database: {e}")
                                finally:
                                    if db:
                                        db.close()
                                
                                # Send to frontend
                                await self.client_ws.send_json({
                                    "type": "text",
                                    "content": self.ai_transcript_buffer.strip()
                                })
                                self.ai_transcript_buffer = ""  # Clear buffer
                        
                        self.last_ai_transcript_time = time.time()
            
            # Handle audio chunks (for playback)
            if 'audio_chunk' in server_content:
                audio_chunk = server_content['audio_chunk']
                if 'data' in audio_chunk:
                    print(f"🔊 Audio chunk received: {len(audio_chunk['data'])} chars")
            
            # Handle Voice Activity Detection (VAD) events
            if 'activity_start' in msg_data or 'activityStart' in msg_data:
                print(f"🎙️ VAD: User started speaking - will interrupt AI if speaking")
                # Send activity start notification to frontend
                await self.client_ws.send_json({
                    "type": "activity_start", 
                    "message": "User started speaking"
                })
                # Send interruption signal to stop AI speech immediately
                await self.client_ws.send_json({
                    "type": "interrupted",
                    "message": "AI speech interrupted by user"
                })
                
                # Send audio stream end to flush any buffered audio  
                if self.gemini_ws and self.is_connected:
                    try:
                        audio_stream_end_message = {
                            "realtimeInput": {
                                "audioStreamEnd": True
                            }
                        }
                        await self.gemini_ws.send(json.dumps(audio_stream_end_message))
                        print(f"📤 Sent audio stream end signal to Gemini")
                    except Exception as e:
                        print(f"❌ Error sending audio stream end: {e}")
                # Clear any old transcript buffer when user starts speaking
                self.user_transcript_buffer = ""
                # Clear last sent transcript to ensure we don't supplement old messages
                self.last_sent_transcript = ""
                
            if 'activity_end' in msg_data or 'activityEnd' in msg_data:
                print(f"🎙️ VAD: User stopped speaking")
                # Send activity end notification to frontend
                await self.client_ws.send_json({
                    "type": "activity_end", 
                    "message": "User stopped speaking"
                })
                # Always flush user transcript when user stops speaking, with delay to prevent UI flickering
                if self.user_transcript_buffer.strip():
                    print(f"👤 Scheduling delayed user transcript send: \"{self.user_transcript_buffer.strip()}\"")
                    # Store the current transcript for delayed sending (cleaned)
                    pending_transcript = self._clean_transcript(self.user_transcript_buffer.strip())
                    self.user_transcript_buffer = ""
                    self.pending_transcript_content = pending_transcript  # Store for potential supplementation
                    
                    # Cancel any existing pending transcript task
                    if self.pending_transcript_task and not self.pending_transcript_task.done():
                        self.pending_transcript_task.cancel()
                    
                    # Schedule delayed send (800ms delay to allow for potential RAG supplementation)
                    self.pending_transcript_task = asyncio.create_task(self._send_delayed_transcript(pending_transcript))
                else:
                    print(f"👤 No user transcript to flush (buffer was empty)")
                    
            # Handle VAD speech detected event (indicates audio was detected but not necessarily speech)
            if 'speechDetected' in msg_data:
                speech_detected = msg_data['speechDetected']
                print(f"🎙️ VAD: Speech detected = {speech_detected}")
                await self.client_ws.send_json({
                    "type": "speech_detected",
                    "detected": speech_detected
                })
                    
                
            # Removed generic text field search to reduce debug noise
                
        except Exception as e:
            print(f"❌ Transcript extraction error: {e}")
    
    async def _send_delayed_transcript(self, transcript_content: str):
        """Send transcript after a delay, unless cancelled by supplementation."""
        try:
            # Wait for 800ms to allow potential RAG supplementation
            await asyncio.sleep(0.8)
            
            # Check if this transcript is still the latest (not replaced by supplementation)
            if self.pending_transcript_task and not self.pending_transcript_task.cancelled():
                print(f"👤 Sending delayed transcript: \"{transcript_content}\"")
                await self.client_ws.send_json({
                    "type": "user_transcript",
                    "content": transcript_content
                })
                self.last_sent_transcript = transcript_content
                self.pending_transcript_task = None
                self.pending_transcript_content = ""  # Clear pending content
            else:
                print(f"👤 Delayed transcript cancelled (replaced by supplementation)")
                
        except asyncio.CancelledError:
            print(f"👤 Delayed transcript task was cancelled")
        except Exception as e:
            print(f"❌ Error sending delayed transcript: {e}")
    
    async def _handle_tool_call(self, tool_call):
        """Handle function calls from Gemini."""
        try:
            print(f"🔧 Function call received")
            
            for function_call in tool_call.get('functionCalls', []):
                function_name = function_call.get('name')
                function_args = function_call.get('args', {})
                function_id = function_call.get('id')
                
                print(f"🔍 Calling function: {function_name}")
                print(f"📋 Args: {function_args}")
                
                if function_name == 'search_knowledge_base':
                    # Execute RAG search
                    query = function_args.get('query', '')
                    print(f"🔍 Searching RAG: '{query}'")
                    print(f"🔍 RAG service chain available: {bool(rag_service.chain)}")
                    
                    # Supplement incomplete transcript ONLY if we have a pending transcript from current user input
                    # This ensures we don't accidentally modify old messages
                    if self.pending_transcript_task and not self.pending_transcript_task.done() and self.pending_transcript_content:
                        # We have a pending transcript from the current user input - supplement it
                        current_transcript = self.pending_transcript_content
                        print(f"🔍 Found pending transcript task - will supplement before sending")
                        print(f"   Pending transcript: \"{current_transcript}\"")
                        print(f"   RAG query: \"{query}\"")
                        
                        # Cancel the pending task so we can send supplemented version immediately
                        self.pending_transcript_task.cancel()
                        
                        # Check if the query contains information missing from transcript
                        transcript_lower = current_transcript.lower()
                        query_lower = query.lower()
                        
                        # If query has content not in transcript, create a supplemented version
                        missing_words = [word for word in query_lower.split() if word not in transcript_lower]
                        if missing_words:
                            # Insert the missing words before the punctuation
                            if current_transcript.endswith('?'):
                                supplemented_transcript = f"{current_transcript[:-1]} {' '.join(missing_words)}?"
                            else:
                                supplemented_transcript = f"{current_transcript} {' '.join(missing_words)}"
                                
                            print(f"🔧 Supplementing current incomplete transcript:")
                            print(f"   Original: \"{current_transcript}\"")
                            print(f"   Missing words: {missing_words}")
                            print(f"   Result: \"{supplemented_transcript}\"")
                            
                            # Send as initial transcript (replacing the delayed one)
                            await self.client_ws.send_json({
                                "type": "user_transcript",
                                "content": supplemented_transcript
                            })
                            
                            self.last_sent_transcript = supplemented_transcript
                            self.pending_transcript_task = None  # Clear pending task reference
                            self.pending_transcript_content = ""  # Clear pending content
                        else:
                            # No missing words, just send the original transcript
                            print(f"🔍 No supplementation needed, sending original transcript")
                            await self.client_ws.send_json({
                                "type": "user_transcript",
                                "content": current_transcript
                            })
                            self.last_sent_transcript = current_transcript
                            self.pending_transcript_task = None
                            self.pending_transcript_content = ""
                    else:
                        print(f"🔍 No pending transcript to supplement - RAG query processed independently")
                        
                        # Alternative: If no pending transcript but we have a RAG query, 
                        # it means Gemini understood audio that STT might have missed
                        if query and not self.last_sent_transcript:
                            print(f"🔧 STT might have missed audio - Gemini understood: '{query}'")
                            print(f"🔧 Consider this as evidence of STT quality issues")
                            
                            # Log for analysis
                            print(f"📊 AUDIO PROCESSING DISCREPANCY:")
                            print(f"   STT Result: {self.last_sent_transcript or 'NONE'}")
                            print(f"   Gemini Understood: {query}")
                            print(f"   This suggests STT quality issues with proper names/complex phrases")
                    
                    rag_result = await process_rag_query(query)
                    
                    print(f"🔍 RAG raw result: {rag_result}")
                    print(f"🔍 Sources count: {rag_result.get('sources_count', 'not found')}")
                    print(f"🔍 Answer: {rag_result.get('answer', 'not found')}")
                    
                    # Format the response naturally (without mentioning "sources" or "tool")
                    answer_text = rag_result.get('answer', '').strip()
                    sources_count = rag_result.get('sources_count', 0)
                    
                    if answer_text and answer_text != "I couldn't process your question.":
                        # Use the RAG answer regardless of source count, as long as it's meaningful
                        # Remove any potential technical language from the RAG response
                        result_text = answer_text.replace("According to", "").replace("Based on", "").replace("The documents show", "").replace("The information indicates", "").strip()
                        
                        # Ensure the result is not too long and properly formatted for JSON
                        if len(result_text) > 500:
                            result_text = result_text[:500] + "..."
                        
                        # Clean any potential JSON-breaking characters
                        result_text = result_text.replace('"', "'").replace('\n', ' ').replace('\r', ' ')
                        
                        print(f"📚 RAG result: {sources_count} sources found, using answer: '{result_text[:50]}...'")
                    else:
                        result_text = "I don't have specific information about that."
                        print(f"📚 No meaningful RAG answer found")
                    
                    # Send function response back to Gemini (using exact Google format from reference)
                    function_response = {
                        "tool_response": {
                            "function_responses": [{
                                "id": function_id,
                                "name": function_name,
                                "response": {"result": {"string_value": result_text}}
                            }]
                        }
                    }
                    
                    print(f"📤 Sending RAG result to Gemini: '{result_text[:100]}...'")
                    print(f"📤 Function response: {json.dumps(function_response)}")
                    
                    # Send function response to Gemini
                    await self.gemini_ws.send(json.dumps(function_response))
                    
                    # Store the RAG result for potential verification
                    self.last_rag_result = result_text
                    self.last_rag_query = query
                    print(f"✅ Function response sent, stored RAG result for verification")
                    
                else:
                    print(f"❓ Unknown function: {function_name}")
                    
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")
            print(f"❌ Tool call error: {e}")
    
    async def _process_gemini_message(self, msg_data: dict):
        """Process message from Gemini and forward to client."""
        try:
            # Handle audio output
            if audio_data := decode_audio_output(msg_data):
                self.audio_chunks.append(audio_data)
                
                # Send audio data to client
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                await self.client_ws.send_json({
                    "type": "audio",
                    "data": audio_b64,
                    "config": {
                        "sample_rate": OUTPUT_AUDIO_CONFIG.sample_rate,
                        "channels": OUTPUT_AUDIO_CONFIG.channels
                    }
                })
            
            # Handle text responses
            if text_response := extract_text_response(msg_data):
                # Clean the response before sending
                cleaned_response = self._clean_debug_artifacts(text_response)
                if cleaned_response:  # Only send if we have content after cleaning
                    await self.client_ws.send_json({
                        "type": "text",
                        "content": cleaned_response
                    })
            
            # Handle turn completion
            if 'turnComplete' in msg_data.get('serverContent', {}):
                # Flush any remaining transcript buffers
                if self.ai_transcript_buffer.strip():
                    print(f"🤖 Flushing final AI transcript: \"{self.ai_transcript_buffer.strip()}\"")
                    await self.client_ws.send_json({
                        "type": "text",
                        "content": self.ai_transcript_buffer.strip()
                    })
                    self.ai_transcript_buffer = ""
                
                if self.user_transcript_buffer.strip():
                    print(f"👤 Flushing final user transcript: \"{self.user_transcript_buffer.strip()}\"")
                    await self.client_ws.send_json({
                        "type": "user_transcript",
                        "content": self.user_transcript_buffer.strip()
                    })
                    self.user_transcript_buffer = ""
                
                self.audio_chunks = []
                await self.client_ws.send_json({
                    "type": "turn_complete",
                    "message": "Turn complete - you can speak now"
                })
            
            # Handle interruption
            elif 'interrupted' in msg_data.get('serverContent', {}):
                await self.client_ws.send_json({
                    "type": "interrupted",
                    "message": "Response interrupted"
                })
                
        except Exception as e:
            logger.error(f"Error processing Gemini message: {e}")
    
    def _clean_transcript(self, transcript: str) -> str:
        """Clean and improve transcript quality, especially for proper names."""
        cleaned = transcript.strip()
        
        # Remove non-English characters and noise words
        import re
        # Remove Thai, Chinese, Japanese characters
        cleaned = re.sub(r'[\u0E00-\u0E7F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]', '', cleaned)
        
        # Remove common noise patterns
        cleaned = re.sub(r'<noise>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        
        # Attempt to restore proper names using fuzzy matching
        for proper_name in self.proper_names:
            # Create patterns for common speech recognition errors
            name_parts = proper_name.lower().split()
            
            # Look for partial matches and try to restore
            if len(name_parts) == 2:  # e.g., "Mori Pines"
                part1, part2 = name_parts
                
                # Common patterns where proper names get mangled
                patterns = [
                    f"{part1}.*{part2}",  # "mori something pines"
                    f"{part1[:-1]}.*{part2}",  # "mor something pines" 
                    f"{part1}.*{part2[:-1]}",  # "mori something pine"
                    f"more.*{part2}",  # "more pines" -> "Mori Pines"
                    f"{part1}.*pine",  # "mori pine" -> "Mori Pines"
                ]
                
                for pattern in patterns:
                    if re.search(pattern, cleaned.lower()):
                        # Replace with correct proper name
                        cleaned = re.sub(pattern, proper_name, cleaned, flags=re.IGNORECASE)
                        print(f"🔧 Restored proper name: '{pattern}' -> '{proper_name}'")
                        break
        
        return cleaned.strip()
    
    def _clean_debug_artifacts(self, text: str) -> str:
        """Remove debug artifacts from Gemini responses."""
        import re
        
        # Remove common debug patterns
        cleaned = text
        
        # Remove tool_outputs code blocks
        cleaned = re.sub(r'```tool_outputs\s*\n?.*?\n?```', '', cleaned, flags=re.DOTALL)
        
        # Remove JSON-like answer structures
        cleaned = re.sub(r'\{"answer":\s*"([^"]+)"\}', r'\1', cleaned)
        cleaned = re.sub(r'\{"answer":\s*"([^"]+)"\}', r'\1', cleaned)
        
        # Remove any remaining tool output patterns
        cleaned = re.sub(r'tool_outputs\s*{[^}]*}', '', cleaned)
        cleaned = re.sub(r'\{"hits":[^\}]*\}', '', cleaned)
        
        # Remove any remaining backticks and code block markers
        cleaned = re.sub(r'```[a-zA-Z]*\n?', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)
        
        # Clean up any JSON artifacts
        cleaned = re.sub(r'\{"[^"]+":"([^"]+)"\}', r'\1', cleaned)
        cleaned = re.sub(r"{'[^']+':'([^']+)'}", r'\1', cleaned)
        
        # Clean up extra whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        if cleaned != text:
            print(f"🧹 Cleaned debug artifacts from AI response:")
            print(f"   Before: {text}")
            print(f"   After:  {cleaned}")
        
        return cleaned
    
    async def _generate_user_transcript(self):
        """Generate transcript from accumulated user audio buffer."""
        if not self.user_audio_buffer:
            return
            
        try:
            # Combine all audio chunks
            combined_audio = b''.join(self.user_audio_buffer)
            self.user_audio_buffer = []  # Clear buffer
            
            if len(combined_audio) < 1000:  # Skip very short audio
                return
                
            # Use Gemini's own speech recognition by sending the audio as a query
            # This is a workaround since we can't get direct transcripts
            print(f"🎯 Generating transcript for {len(combined_audio)} bytes of user audio")
            
            # For now, just send a placeholder transcript
            # In production, you could integrate Google Speech-to-Text API here
            current_time = time.time()
            if current_time - self.last_user_transcript_time > 2:  # Avoid spam
                await self.client_ws.send_json({
                    "type": "user_transcript",
                    "content": "[User spoke - transcript not available in audio-only mode]"
                })
                self.last_user_transcript_time = current_time
                
        except Exception as e:
            print(f"❌ Transcript generation error: {e}")
    
    async def send_audio_to_gemini(self, audio_data: bytes):
        """Send audio data to Gemini."""
        if self.gemini_ws and self.is_connected:
            try:
                message = encode_audio_input(audio_data, INPUT_AUDIO_CONFIG)
                await self.gemini_ws.send(json.dumps(message))
                # print(f"🎵 Audio sent: {len(audio_data)} bytes")
                
            except Exception as e:
                logger.error(f"Error sending audio to Gemini: {e}")
                print(f"❌ Audio send error: {e}")
        else:
            print(f"⚠️ Not connected to Gemini")
    
    async def send_text_to_gemini(self, text: str):
        """Send text message to Gemini (RAG will be handled by function calling)."""
        if self.gemini_ws and self.is_connected:
            try:
                # Show full user input for debugging
                if len(text) > 50:
                    print(f"💬 User: \"{text[:50]}...\" (full: {len(text)} chars)")
                    print(f"   Full text: {text}")
                else:
                    print(f"💬 User: \"{text}\"")
                
                # Send text directly to Gemini - it will call RAG function if needed
                message = encode_text_input(text)
                await self.gemini_ws.send(json.dumps(message))
                print(f"🚀 Sent to Gemini (RAG available via function calling)")
                
            except Exception as e:
                logger.error(f"Error sending text to Gemini: {e}")
                print(f"❌ Text send error: {e}")
    
    async def disconnect(self):
        """Disconnect from Gemini."""
        self.is_connected = False
        if self.gemini_ws:
            await self.gemini_ws.close()
            logger.info("Disconnected from Gemini")

# =============================================================================
# AI SERVICES API ENDPOINTS
# =============================================================================

class AdminSuggestionRequest(BaseModel):
    session_id: int
    query: str
    message_by: str

class CustomerSuggestionRequest(BaseModel):
    conversation: List[Dict[str, str]]
    call_session_id: Optional[int] = None

class SummarizationRequest(BaseModel):
    conversation: List[Dict[str, str]]
    call_session_id: int

class SentimentAnalysisRequest(BaseModel):
    transcript_id: int

class SessionConversationRequest(BaseModel):
    session_id: int

@app.post("/api/ai/admin-suggestion")
async def admin_suggestion_endpoint(request: AdminSuggestionRequest):
    """Generate admin suggestions using LangChain agent."""
    try:
        suggestion = get_suggestion_from_agent(
            session_id=request.session_id,
            query=request.query,
            message_by=request.message_by
        )
        return {
            "success": True,
            "suggestion": suggestion,
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Admin suggestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating admin suggestion: {str(e)}")

@app.post("/api/ai/customer-suggestions")
async def customer_suggestions_endpoint(request: CustomerSuggestionRequest):
    """Generate customer suggestions for post-call actions."""
    try:
        suggestions = generate_caller_suggestions(
            conversation=request.conversation,
            call_session_id=request.call_session_id
        )
        return {
            "success": True,
            "suggestions": suggestions,
            "call_session_id": request.call_session_id
        }
    except Exception as e:
        logger.error(f"Customer suggestions error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating customer suggestions: {str(e)}")

@app.post("/api/ai/summarize")
async def summarize_conversation_endpoint(request: SummarizationRequest):
    """Summarize a conversation and store it in the database."""
    try:
        summary = summarize_text(
            conversation=request.conversation,
            call_session_id=request.call_session_id
        )
        return {
            "success": True,
            "summary": summary,
            "call_session_id": request.call_session_id
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Error summarizing conversation: {str(e)}")

@app.post("/api/ai/sentiment-analysis")
async def sentiment_analysis_endpoint(request: SentimentAnalysisRequest):
    """Analyze sentiment of a specific transcript."""
    try:
        result = analyze_sentiment_from_transcript(request.transcript_id)
        result_data = json.loads(result)
        
        if result_data.get("success"):
            return result_data
        else:
            raise HTTPException(status_code=404, detail=result_data.get("error", "Unknown error"))
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail="Error parsing sentiment analysis result")
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.post("/api/ai/session-conversation")
async def session_conversation_endpoint(request: SessionConversationRequest):
    """Retrieve full conversation for a session."""
    try:
        result_data = get_session_conversation(request.session_id)
        
        if result_data.get("success"):
            return result_data
        else:
            raise HTTPException(status_code=404, detail=result_data.get("error", "Unknown error"))
    except Exception as e:
        logger.error(f"Session conversation error: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving session conversation: {str(e)}")

# Bulk AI processing endpoint
@app.post("/api/ai/process-session")
async def process_session_endpoint(session_id: int):
    """Process a complete session with all AI services (summary, sentiment, suggestions)."""
    try:
        # Get session conversation
        conversation_data = get_session_conversation(session_id)
        
        if not conversation_data.get("success"):
            raise HTTPException(status_code=404, detail="Session not found or has no conversation")
        
        conversation = conversation_data["conversation"]
        
        # Format conversation for processing
        formatted_conversation = [
            {"role": msg["message_by"], "content": msg["message"]} 
            for msg in conversation
        ]
        
        # Generate summary
        summary = summarize_text(formatted_conversation, session_id)
        
        # Generate customer suggestions
        customer_suggestions = generate_caller_suggestions(formatted_conversation, session_id)
        
        # Analyze sentiment for each transcript (if needed)
        sentiment_results = []
        db = SessionLocal()
        try:
            transcripts = (
                db.query(Transcript)
                .filter(Transcript.session_id == session_id)
                .order_by(Transcript.created_at)
                .all()
            )
            
            for transcript in transcripts:
                sentiment_result = analyze_sentiment_from_transcript(transcript.id)
                sentiment_data = json.loads(sentiment_result)
                if sentiment_data.get("success"):
                    sentiment_results.append(sentiment_data)
        finally:
            db.close()
        
        return {
            "success": True,
            "session_id": session_id,
            "summary": summary,
            "customer_suggestions": customer_suggestions,
            "sentiment_analysis": sentiment_results,
            "conversation_count": len(conversation)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing session: {str(e)}")

@app.get("/api/ai/health")
async def ai_services_health():
    """Health check for AI services."""
    try:
        # Test AI services by calling them with simple test data
        test_conversation = [{"role": "User", "content": "Hello"}, {"role": "AI", "content": "Hi there"}]
        
        # Test summarization service
        try:
            summary_test = summarize_text(test_conversation, call_session_id=0)  # Use 0 for test (won't store)
            summarization_status = "healthy" if summary_test and not summary_test.startswith("Error") else "error"
        except Exception as e:
            summarization_status = f"error: {str(e)}"
        
        # Test customer suggestions service
        try:
            suggestions_test = generate_caller_suggestions(test_conversation, call_session_id=None)
            suggestions_status = "healthy" if suggestions_test and not suggestions_test.startswith("Error") else "error"
        except Exception as e:
            suggestions_status = f"error: {str(e)}"
        
        # Test database connectivity for AI services
        db = SessionLocal()
        try:
            # Simple query to test database
            transcript_count = db.query(Transcript).count()
            db_status = "healthy"
        except Exception as e:
            db_status = f"error: {str(e)}"
            transcript_count = 0
        finally:
            db.close()
        
        # Test sentiment analysis (requires a real transcript, so just check import)
        try:
            # Just check if the function is callable
            sentiment_status = "healthy" if callable(analyze_sentiment_from_transcript) else "error"
        except Exception as e:
            sentiment_status = f"error: {str(e)}"
        
        return {
            "status": "healthy" if all([
                summarization_status == "healthy",
                suggestions_status == "healthy", 
                db_status == "healthy",
                sentiment_status == "healthy"
            ]) else "partial",
            "services": {
                "summarization": summarization_status,
                "customer_suggestions": suggestions_status,
                "sentiment_analysis": sentiment_status,
                "database": db_status,
                "transcript_count": transcript_count
            },
            "api_endpoints": [
                "/api/ai/admin-suggestion",
                "/api/ai/customer-suggestions", 
                "/api/ai/summarize",
                "/api/ai/sentiment-analysis",
                "/api/ai/session-conversation",
                "/api/ai/process-session",
                "/api/ai/process-session-manual/{session_id}",
                "/api/call-session/{session_id}/status"
            ]
        }
    except Exception as e:
        logger.error(f"AI services health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "services": {
                "summarization": "unknown",
                "customer_suggestions": "unknown", 
                "sentiment_analysis": "unknown",
                "database": "unknown"
            }
        }

@app.post("/api/ai/process-session-manual/{session_id}")
async def process_session_manual_trigger(session_id: int):
    """
    Manually trigger AI processing for a specific session.
    Useful for reprocessing sessions or testing.
    """
    try:
        logger.info(f"Manual AI processing triggered for session {session_id}")
        
        # Check if session exists and has conversation
        conversation_data = get_session_conversation(session_id)
        
        if not conversation_data.get("success"):
            raise HTTPException(status_code=404, detail="Session not found or has no conversation")
        
        conversation = conversation_data["conversation"]
        if len(conversation) == 0:
            raise HTTPException(status_code=400, detail="Session has no messages to process")
        
        # Start AI processing in background
        asyncio.create_task(process_call_session_ai(session_id))
        
        return {
            "success": True,
            "message": f"AI processing started for session {session_id}",
            "session_id": session_id,
            "message_count": len(conversation),
            "note": "Processing is running in the background. Check logs for progress."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error manually triggering AI processing for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting AI processing: {str(e)}")

@app.get("/api/call-session/{session_id}/status")
async def get_call_session_status(session_id: int):
    """
    Get the current status of a call session including AI processing results.
    Shows the automatically updated call session data.
    """
    try:
        db = SessionLocal()
        call_session = db.query(CallSession).filter(CallSession.id == session_id).first()
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")
        
        # Get conversation count
        conversation_count = db.query(Transcript).filter(Transcript.session_id == session_id).count()
        
        # Check if AI processing has been completed (has end_time and summary)
        ai_processed = bool(call_session.end_time and call_session.summarized_content)
        
        db.close()
        
        return {
            "success": True,
            "call_session": {
                "id": call_session.id,
                "cust_id": call_session.cust_id,
                "start_time": call_session.start_time.isoformat() if call_session.start_time else None,
                "end_time": call_session.end_time.isoformat() if call_session.end_time else None,
                "duration_secs": call_session.duration_secs,
                "duration_formatted": f"{call_session.duration_secs // 60}m {call_session.duration_secs % 60}s" if call_session.duration_secs else None,
                "sentiment_counts": {
                    "positive": call_session.positive,
                    "neutral": call_session.neutral,
                    "negative": call_session.negative,
                    "total": (call_session.positive or 0) + (call_session.neutral or 0) + (call_session.negative or 0)
                },
                "key_words": call_session.key_words,
                "summarized_content": call_session.summarized_content,
                "customer_suggestions": call_session.customer_suggestions,
                "admin_suggestions": call_session.admin_suggestions,
                "conversation_count": conversation_count,
                "ai_processed": ai_processed,
                "call_status": "completed" if call_session.end_time else "in_progress"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call session status for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving call session status: {str(e)}")

@app.post("/api/call-session/{session_id}/test-end-call")
async def test_end_call(session_id: int):
    """
    Test endpoint to simulate ending a call and triggering automatic AI processing.
    This is useful for testing the automatic call session updates.
    """
    try:
        logger.info(f"Test: Simulating call end for session {session_id}")
        
        # Check if session exists
        db = SessionLocal()
        call_session = db.query(CallSession).filter(CallSession.id == session_id).first()
        
        if not call_session:
            db.close()
            raise HTTPException(status_code=404, detail="Call session not found")
        
        # Check if session has any conversation
        conversation_count = db.query(Transcript).filter(Transcript.session_id == session_id).count()
        
        if conversation_count == 0:
            db.close()
            raise HTTPException(status_code=400, detail="Session has no conversation to process")
        
        db.close()
        
        # Trigger the same AI processing that happens when WebSocket closes
        asyncio.create_task(process_call_session_ai(session_id))
        
        return {
            "success": True,
            "message": f"Simulated call end for session {session_id}",
            "session_id": session_id,
            "conversation_count": conversation_count,
            "note": "AI processing and call session update started in background. Use GET /api/call-session/{session_id}/status to check results.",
            "test_workflow": [
                "1. Call ended (simulated)",
                "2. AI processing started automatically",
                "3. Conversation summarized",
                "4. Customer suggestions generated", 
                "5. Sentiment analysis performed",
                "6. Call session updated with end time, duration, sentiment counts",
                "7. Check status with GET /api/call-session/{session_id}/status"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test end call for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error simulating call end: {str(e)}")

# =============================================================================
# END AI SERVICES API ENDPOINTS
# =============================================================================

@app.websocket("/ws/{call_session_id}")
async def websocket_endpoint(websocket: WebSocket, call_session_id: int):
    """WebSocket endpoint for client connections."""
    
    db = None
    
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
    
    await websocket.accept()
    print(f"🔌 Client connected ({len(active_connections)+1} total)")
    logger.info("Client connected to WebSocket")
    
    # Create Gemini connection for this client
    connection = GeminiLiveConnection(websocket, call_session_id)
    active_connections[websocket] = connection
    
    # Start listening for client messages immediately (don't wait for Gemini connection)
    try:
        # Create Gemini connection task in background
        asyncio.create_task(connection.connect_to_gemini())
        
        # Listen for client messages immediately
        while True:
            try:
                data = await websocket.receive_json()
                await handle_client_message(connection, data)
            except WebSocketDisconnect:
                logger.info("Client disconnected normally")
                break
            except Exception as e:
                logger.error(f"Error handling client message: {e}")
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except:
                    # Connection might be closed
                    break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        # Cleanup
        await connection.disconnect()
        if websocket in active_connections:
            del active_connections[websocket]
        logger.info("Client connection cleaned up")
        
        # =============================================================================
        # POST-CALL AI PROCESSING
        # =============================================================================
        
        # Process the call session with AI services after connection closes
        if call_session_id:
            try:
                logger.info(f"Starting post-call AI processing for session {call_session_id}")
                print(f"🤖 Starting post-call AI processing for session {call_session_id}")
                
                # Run AI processing in background to avoid blocking cleanup
                asyncio.create_task(process_call_session_ai(call_session_id))
                
            except Exception as e:
                logger.error(f"Error starting post-call AI processing: {e}")
                print(f"❌ Error starting post-call AI processing: {e}")

async def handle_client_message(connection: GeminiLiveConnection, data: dict):
    """Handle message from client."""
    try:
        message_type = data.get('type')
        
        if message_type == 'audio':
            # Handle audio data from client
            audio_b64 = data.get('data', '')
            if audio_b64:
                try:
                    # Decode base64 audio data
                    audio_data = base64.b64decode(audio_b64)
                    # print(f"🎤 Received {len(audio_data)} bytes")
                    
                    # Convert the audio data to PCM format
                    pcm_data = await convert_audio_to_pcm(audio_data)
                    if pcm_data:
                        await connection.send_audio_to_gemini(pcm_data)
                    else:
                        print(f"❌ Audio conversion failed")
                        
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    print(f"❌ Audio error: {e}")
        
        elif message_type == 'text':
            # Handle text message from client
            text_content = data.get('content', '')
            if text_content:
                print(f"📝 Text received: \"{text_content}\"")
                await connection.send_text_to_gemini(text_content)
            else:
                print(f"⚠️ Empty text message")
        
        elif message_type == 'ping':
            # Handle ping message for testing
            print(f"🏓 Ping")
            await connection.client_ws.send_json({"type": "pong", "message": "Backend is working!"})
        
        else:
            print(f"❓ Unknown message type: {message_type}")
            
    except Exception as e:
        logger.error(f"Error handling client message: {e}")
        print(f"❌ Message error: {e}")

async def convert_audio_to_pcm(audio_data: bytes) -> bytes:
    """Convert audio data to 16-bit PCM, 16kHz, mono format as required by Gemini."""
    try:
        if len(audio_data) == 0:
            return None
        
        # Check if the data length is even (16-bit samples are 2 bytes each)
        if len(audio_data) % 2 != 0:
            audio_data = audio_data + b'\x00'
        
        # Convert bytes to numpy array for processing
        try:
            # Try to interpret as int16 PCM data (little-endian)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Ensure we have valid audio data
            if len(audio_array) > 0:
                # Convert back to bytes in the format Gemini expects
                pcm_bytes = audio_array.astype(np.int16).tobytes()
                return pcm_bytes
            else:
                return None
                
        except Exception as e:
            # Fallback: assume it's already in the right format
            return audio_data
        
    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        return None


"""NOT IMPORTANT: FOR DEBUGGING PURPOSE ONLY"""

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "connections": len(active_connections)}

@app.get("/api/status")
async def api_status():
    """API status endpoint for Vue frontend."""
    rag_health = await get_rag_health()
    
    # Check AI services health
    ai_services_healthy = True
    try:
        # Quick test of AI services by checking if they're importable and callable
        ai_services_healthy = all([
            callable(get_suggestion_from_agent),
            callable(generate_caller_suggestions),
            callable(summarize_text),
            callable(analyze_sentiment_from_transcript)
        ])
    except Exception:
        ai_services_healthy = False
    
    return {
        "status": "ready",
        "websocket_url": "/ws",
        "active_connections": len(active_connections),
        "api_key_configured": bool(GOOGLE_API_KEY),
        "rag_status": rag_health,
        "ai_services": {
            "status": "healthy" if ai_services_healthy else "error",
                         "endpoints_available": [
                 "/api/ai/admin-suggestion",
                 "/api/ai/customer-suggestions", 
                 "/api/ai/summarize",
                 "/api/ai/sentiment-analysis",
                 "/api/ai/session-conversation",
                 "/api/ai/process-session",
                 "/api/ai/process-session-manual/{session_id}",
                 "/api/ai/health",
                 "/api/call-session/{session_id}/status"
             ]
        }
    }

@app.post("/api/rag/query")
async def rag_query_endpoint(query_data: dict):
    """Direct RAG query endpoint for testing."""
    query = query_data.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    result = await process_rag_query(query)
    return result

@app.get("/api/rag/health")
async def rag_health_endpoint():
    """RAG service health check endpoint."""
    return await get_rag_health()

@app.post("/api/test-text")
async def test_text_endpoint(request_data: dict):
    """Test endpoint for frontend to send text without WebSocket."""
    text = request_data.get("text", "")
    print(f"\n📝 TEST TEXT ENDPOINT RECEIVED:")
    print(f"  Text: '{text}'")
    print(f"  Length: {len(text)} characters")
    
    if not text:
        return {"error": "No text provided"}
    
    # Test RAG processing
    if rag_service.chain:
        rag_result = await process_rag_query(text)
        print(f"📝 RAG result: {rag_result}")
        return {
            "received_text": text,
            "rag_answer": rag_result.get("answer", ""),
            "rag_sources": rag_result.get("sources_count", 0),
            "rag_raw_result": rag_result,
            "status": "processed"
        }
    else:
        return {
            "received_text": text,
            "status": "no_rag"
        }

@app.post("/api/test-rag")
async def test_rag_endpoint(request_data: dict):
    """Direct RAG test endpoint."""
    query = request_data.get("query", "Gamuda Cove")
    print(f"\n🧪 TESTING RAG DIRECTLY:")
    print(f"  Query: '{query}'")
    print(f"  RAG chain available: {bool(rag_service.chain)}")
    
    if rag_service.chain:
        try:
            # Test multiple search terms from the sample documents
            test_queries = [query, "Mori Pines", "Enso Woods", "terrace", "property", "amenities"]
            results = {}
            
            for test_query in test_queries:
                print(f"  Testing query: '{test_query}'")
                if rag_service.retriever:
                    docs = rag_service.retriever.get_relevant_documents(test_query)
                    print(f"    Retrieved {len(docs)} documents for '{test_query}'")
                    if docs:
                        print(f"    First doc preview: {docs[0].page_content[:150]}...")
                    results[test_query] = {
                        "doc_count": len(docs),
                        "preview": docs[0].page_content[:200] if docs else "No documents found"
                    }
            
            # Run the original query through RAG
            rag_result = await process_rag_query(query)
            print(f"  RAG result: {rag_result}")
            
            return {
                "query": query,
                "result": rag_result,
                "test_results": results,
                "status": "success"
            }
        except Exception as e:
            print(f"  RAG error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "error": str(e),
                "status": "error"
            }
    else:
        return {
            "query": query,
            "error": "RAG not initialized",
            "status": "not_initialized"
        }

@app.get("/api/test-mongodb")
async def test_mongodb_endpoint():
    """Test MongoDB connection and data."""
    try:
        if rag_service.client:
            # Test connection
            rag_service.client.admin.command('ping')
            
            # Get collection
            collection = rag_service.client[rag_service.DB_NAME][rag_service.COLLECTION_NAME]
            
            # Get document count
            doc_count = collection.count_documents({})
            
            # Search for documents containing "Mori Pines" (handling line breaks and spaces)
            mori_pines_docs = list(collection.find({"text": {"$regex": "Mori[\\s\\n]+Pines", "$options": "i"}}).limit(5))
            
            # Get sample documents
            sample_docs = list(collection.find({}).limit(3))
            
            # Check if documents have embeddings
            docs_with_embeddings = collection.count_documents({"embedding": {"$exists": True}})
            
            # Check embedding field structure
            embedding_info = {}
            if sample_docs:
                first_doc = sample_docs[0]
                if "embedding" in first_doc:
                    embedding = first_doc["embedding"]
                    embedding_info = {
                        "embedding_type": str(type(embedding)),
                        "embedding_length": len(embedding) if hasattr(embedding, '__len__') else "unknown",
                        "first_few_values": embedding[:5] if hasattr(embedding, '__getitem__') else "unknown",
                        "is_list": isinstance(embedding, list),
                        "is_array": str(type(embedding))
                    }
            
            return {
                "status": "connected",
                "database": rag_service.DB_NAME,
                "collection": rag_service.COLLECTION_NAME,
                "document_count": doc_count,
                "docs_with_embeddings": docs_with_embeddings,
                "embedding_info": embedding_info,
                "mori_pines_found": len(mori_pines_docs),
                "mori_pines_docs": [
                    {
                        "id": str(doc.get("_id", "")),
                        "text_preview": doc.get("text", "")[:300] + "..." if len(doc.get("text", "")) > 300 else doc.get("text", ""),
                        "has_mori_pines": "Mori Pines" in doc.get("text", "")
                    }
                    for doc in mori_pines_docs
                ],
                "sample_docs": [
                    {
                        "id": str(doc.get("_id", "")),
                        "content_preview": str(doc).replace(rag_service.COLLECTION_NAME, "")[:200] + "...",
                        "has_embedding": "embedding" in doc,
                        "doc_keys": list(doc.keys())
                    }
                    for doc in sample_docs
                ],
                "index_name": rag_service.INDEX_NAME
            }
        else:
            return {
                "status": "not_connected",
                "error": "MongoDB client not available"
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/api/test-gemini-live")
async def test_gemini_live_endpoint(request_data: dict):
    """Test endpoint to send text directly to Gemini Live API."""
    text = request_data.get("text", "Hello Gemini!")
    print(f"\n🧪 TESTING GEMINI LIVE CONNECTION WITH TEXT:")
    print(f"  Text: '{text}'")
    
    # Create a temporary connection to test Gemini Live
    try:
        from websockets.asyncio.client import connect
        ws_url = f'wss://{GEMINI_HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}'
        
        async with connect(ws_url) as ws:
            print("✅ Connected to Gemini Live for testing")
            
            # Send setup
            setup_message = {
                "setup": {
                    "model": GEMINI_MODEL,
                    "generation_config": {
                        "response_modalities": ["TEXT"]
                    }
                }
            }
            await ws.send(json.dumps(setup_message))
            setup_response = await ws.recv()
            setup_data = json.loads(setup_response)
            print(f"📥 Setup response: {setup_data}")
            
            # Send text message
            text_message = encode_text_input(text)
            await ws.send(json.dumps(text_message))
            print("📤 Text message sent to Gemini Live")
            
            # Wait for response
            response = await ws.recv()
            response_data = json.loads(response)
            print(f"📨 Gemini response: {response_data}")
            
            return {
                "status": "success",
                "setup_response": setup_data,
                "gemini_response": response_data
            }
            
    except Exception as e:
        print(f"❌ Gemini Live test failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")