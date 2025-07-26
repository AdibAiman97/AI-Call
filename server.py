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
from rag_integration import (
    rag_service,
    initialize_rag,
    process_rag_query,
    get_rag_health,
)
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
from services.appointment_crud import AppointmentCRUD
from datetime import datetime, timedelta
import re

# Setup logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gemini_live.log"),
        logging.StreamHandler(),  # Console output
    ],
)
logger = logging.getLogger(__name__)

# Load API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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

        logger.info(
            f"🤖 Found {len(conversation)} messages in session {call_session_id}"
        )
        print(f"🤖 Found {len(conversation)} messages in session {call_session_id}")

        # 1. Generate conversation summary
        summary = None
        try:
            logger.info(f"📝 Generating summary for session {call_session_id}")
            print(f"📝 Generating summary for session {call_session_id}")
            summary = summarize_text(formatted_conversation, call_session_id)
            print(f"✅ Summary generated: {summary[:100]}...")
        except Exception as e:
            logger.error(
                f"❌ Error generating summary for session {call_session_id}: {e}"
            )
            print(f"❌ Error generating summary: {e}")

        # 2. Generate customer suggestions
        customer_suggestions = None
        try:
            logger.info(
                f"💡 Generating customer suggestions for session {call_session_id}"
            )
            print(f"💡 Generating customer suggestions for session {call_session_id}")
            customer_suggestions = generate_caller_suggestions(
                formatted_conversation, call_session_id
            )
            print(f"✅ Customer suggestions generated: {customer_suggestions[:100]}...")
        except Exception as e:
            logger.error(
                f"❌ Error generating customer suggestions for session {call_session_id}: {e}"
            )
            print(f"❌ Error generating customer suggestions: {e}")

        # 3. Analyze sentiment for a sample of transcripts to avoid rate limiting
        sentiment_results = []
        try:
            logger.info(f"😊 Analyzing sentiment for session {call_session_id}")
            print(f"😊 Analyzing sentiment for session {call_session_id}")

            db = SessionLocal()
            try:
                transcripts = (
                    db.query(Transcript)
                    .filter(Transcript.session_id == call_session_id)
                    .order_by(Transcript.created_at)
                    .all()
                )

                # Limit sentiment analysis to a max of 15 transcripts to avoid rate limits
                max_sentiment_checks = 15
                transcripts_to_check = transcripts[:max_sentiment_checks]

                logger.info(
                    f"🔬 Analyzing {len(transcripts_to_check)} of {len(transcripts)} transcripts for sentiment."
                )
                print(
                    f"🔬 Analyzing {len(transcripts_to_check)} of {len(transcripts)} transcripts for sentiment."
                )

                for i, transcript in enumerate(transcripts_to_check):
                    try:
                        sentiment_result = analyze_sentiment_from_transcript.invoke(
                            {"transcript_id": transcript.id}
                        )

                        # Add a small delay to respect rate limits
                        if i < len(transcripts_to_check) - 1:
                            await asyncio.sleep(1)  # 1-second delay between calls

                        try:
                            sentiment_data = json.loads(sentiment_result)
                            if sentiment_data.get("success"):
                                sentiment_results.append(sentiment_data)
                                sentiment_info = sentiment_data.get(
                                    "sentiment_analysis", {}
                                )
                                sentiment = sentiment_info.get("sentiment", "Unknown")
                                confidence = sentiment_info.get("confidence", 0)
                                print(
                                    f"   📊 Transcript {transcript.id}: {sentiment} (confidence: {confidence:.2f})"
                                )
                            else:
                                logger.warning(
                                    f"Sentiment analysis failed for transcript {transcript.id}: {sentiment_data.get('error')}"
                                )
                        except json.JSONDecodeError:
                            logger.error(
                                f"Could not decode sentiment JSON for transcript {transcript.id}: {sentiment_result}"
                            )

                    except Exception as transcript_error:
                        logger.error(
                            f"Error analyzing sentiment for transcript {transcript.id}: {transcript_error}"
                        )

            finally:
                db.close()

            logger.info(
                f"✅ Sentiment analysis completed for {len(sentiment_results)} transcripts in session {call_session_id}"
            )
            print(
                f"✅ Sentiment analysis completed for {len(sentiment_results)} transcripts"
            )

        except Exception as e:
            logger.error(
                f"❌ Error analyzing sentiment for session {call_session_id}: {e}"
            )
            print(f"❌ Error analyzing sentiment: {e}")

        # 4. Update call session with all collected data using the service
        try:
            logger.info(f"📊 Updating call session {call_session_id} via service")
            print(f"📊 Updating call session {call_session_id} via service")

            db = SessionLocal()
            service = CallSessionService(db)

            call_session = service.get_by_id(call_session_id)

            if call_session:
                from datetime import datetime, timezone
                from database.schemas import CallSessionUpdate

                end_time = datetime.now(timezone.utc)
                duration_secs = None
                if call_session.start_time:
                    duration = end_time - call_session.start_time
                    duration_secs = int(duration.total_seconds())

                positive_count = sum(
                    1
                    for r in sentiment_results
                    if r.get("sentiment_analysis", {}).get("sentiment", "").lower()
                    == "positive"
                )
                negative_count = sum(
                    1
                    for r in sentiment_results
                    if r.get("sentiment_analysis", {}).get("sentiment", "").lower()
                    == "negative"
                )
                neutral_count = len(sentiment_results) - positive_count - negative_count

                key_words = None
                try:
                    all_text = " ".join(
                        [msg["content"] for msg in formatted_conversation]
                    )
                    words = all_text.split()
                    key_words_list = list(
                        set(
                            [
                                word.strip(".,!?").lower()
                                for word in words
                                if len(word.strip(".,!?")) > 3
                            ]
                        )
                    )[:10]
                    key_words = ", ".join(key_words_list)
                except Exception as kw_error:
                    print(f"   ⚠️ Could not extract key words: {kw_error}")

                update_data = CallSessionUpdate(
                    end_time=end_time.isoformat(),
                    duration_secs=duration_secs,
                    summarized_content=summary,
                    customer_suggestions=customer_suggestions,
                    positive=positive_count,
                    neutral=neutral_count,
                    negative=negative_count,
                    key_words=key_words,
                )

                updated_session = service.update(call_session_id, update_data)

                if updated_session:
                    logger.info(
                        f"✅ Call session {call_session_id} updated successfully via service"
                    )
                    print(f"✅ Call session updated successfully via service")
                else:
                    logger.warning(
                        f"Call session {call_session_id} not found for update via service"
                    )
                    print(
                        f"⚠️ Call session {call_session_id} not found for update via service"
                    )
            else:
                logger.warning(
                    f"Call session {call_session_id} not found for duration calculation"
                )
                print(
                    f"⚠️ Call session {call_session_id} not found for duration calculation"
                )

            db.close()

        except Exception as e:
            logger.error(
                f"❌ Error updating call session {call_session_id} via service: {e}"
            )
            print(f"❌ Error updating call session via service: {e}")
            import traceback

            traceback.print_exc()

        # 5. Log completion
        logger.info(
            f"🎉 AI processing and call session update completed for session {call_session_id}"
        )
        print(
            f"🎉 AI processing and call session update completed for session {call_session_id}"
        )

    except Exception as e:
        logger.error(
            f"❌ Critical error in AI processing for session {call_session_id}: {e}"
        )
        print(f"❌ Critical error in AI processing: {e}")
        import traceback

        traceback.print_exc()


# =============================================================================
# END POST-CALL AI PROCESSING FUNCTION
# =============================================================================

# Configuration
GEMINI_HOST = 'generativelanguage.googleapis.com'
# GEMINI_MODEL = 'models/gemini-2.0-flash-live-001'
GEMINI_MODEL = 'models/gemini-2.5-flash-preview-native-audio-dialog'
# GEMINI_MODEL = 'models/gemini-live-2.5-flash-preview'

# VAD Configuration - Set to False if having voice detection issues
ENABLE_VAD = True  # Change to False to disable VAD temporarily


# Audio configuration
class AudioConfig:
    def __init__(self, sample_rate=24000, channels=1):
        self.sample_rate = sample_rate
        self.format = "S16_LE"
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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ],  # Vue dev servers
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
app.include_router(pdf_router)


def encode_audio_input(data: bytes, config: AudioConfig) -> dict:
    """Build message with user input audio bytes."""
    return {
        "realtimeInput": {
            "mediaChunks": [
                {
                    "mimeType": f"audio/pcm;rate={config.sample_rate}",
                    "data": base64.b64encode(data).decode("UTF-8"),
                }
            ],
        },
    }


def encode_text_input(text: str) -> dict:
    """Builds message with user input text."""
    return {
        "clientContent": {
            "turns": [
                {
                    "role": "USER",
                    "parts": [{"text": text}],
                }
            ],
            "turnComplete": True,
        },
    }


def decode_audio_output(input_msg: dict) -> bytes:
    """Returns byte string with model output audio."""
    result = []
    content_input = input_msg.get("serverContent", {})
    content = content_input.get("modelTurn", {})
    for part in content.get("parts", []):
        data = part.get("inlineData", {}).get("data", "")
        if data:
            result.append(base64.b64decode(data))
    return b"".join(result)


def extract_text_response(input_msg: dict) -> str:
    """Extract text response from Gemini message."""
    content_input = input_msg.get('serverContent', {})
    content = content_input.get('modelTurn', {})
    for part in content.get('parts', []):
        if 'text' in part:
            raw_text = part['text']
            
            # More aggressive filtering of tool artifacts
            if raw_text and any(pattern in raw_text.lower() for pattern in [
                'tool_response', 'function_responses', 'string_value', 
                '"id":', '"name":', '"response":', 'tool_outputs',
                '```tool_outputs', '{"answer"', '"answer":', 'tool_',
                'function_', '```json', '"result":'
            ]):
                print(f"🗑️ Filtering out text response containing tool artifacts: '{raw_text[:50]}...'")
                return ""
            
            # Return raw text for further cleaning by _clean_debug_artifacts
            return raw_text if raw_text else ""
    return ""


def save_audio_to_file(audio_data: bytes, config: AudioConfig, filename: str):
    """Save audio data to WAV file."""
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(config.channels)
        wav_file.setsampwidth(config.sample_size)
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(audio_data)


# Store active connections
active_connections = {}


def parse_appointment_string(appointment_str: str, call_session_id: int) -> dict:
    """
    Parse appointment string into database-compatible format.
    
    Expected format: "Customer: [Name], Phone: [Number], Date: [Date], Time: [Time], Purpose: [Purpose], Notes: [Notes]"
    Returns dict compatible with AppointmentCRUD.create_appointment()
    """
    try:
        # Initialize default values
        appointment_data = {
            "call_session_id": call_session_id,
            "title": "Property Appointment",
            "start_time": None,
            "end_time": None
        }
        
        # Parse customer name
        customer_match = re.search(r'Customer:\s*([^,]+)', appointment_str, re.IGNORECASE)
        customer_name = customer_match.group(1).strip() if customer_match else "Unknown Customer"
        
        # Parse phone number
        phone_match = re.search(r'Phone:\s*([^,]+)', appointment_str, re.IGNORECASE)
        phone_number = phone_match.group(1).strip() if phone_match else ""
        
        # Parse date
        date_match = re.search(r'Date:\s*([^,]+)', appointment_str, re.IGNORECASE)
        date_str = date_match.group(1).strip() if date_match else ""
        
        # Parse time
        time_match = re.search(r'Time:\s*([^,]+)', appointment_str, re.IGNORECASE)
        time_str = time_match.group(1).strip() if time_match else ""
        
        # Parse purpose
        purpose_match = re.search(r'Purpose:\s*([^,]+)', appointment_str, re.IGNORECASE)
        purpose = purpose_match.group(1).strip() if purpose_match else "Property viewing"
        
        # Parse notes
        notes_match = re.search(r'Notes:\s*(.+)', appointment_str, re.IGNORECASE)
        notes = notes_match.group(1).strip() if notes_match else ""
        
        # Create title
        appointment_data["title"] = f"{purpose} - {customer_name}"
        if phone_number:
            appointment_data["title"] += f" ({phone_number})"
        if notes:
            appointment_data["title"] += f" - {notes[:50]}"
        
        # Parse and convert date/time
        if date_str and time_str:
            # Try to parse different date formats
            date_formats = [
                "%Y-%m-%d",    # 2025-01-30
                "%m/%d/%Y",    # 01/30/2025
                "%d/%m/%Y",    # 30/01/2025
                "%B %d, %Y",   # January 30, 2025
                "%b %d, %Y",   # Jan 30, 2025
            ]
            
            parsed_date = None
            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, date_format).date()
                    break
                except ValueError:
                    continue
            
            if not parsed_date:
                # Default to tomorrow if parsing fails
                parsed_date = (datetime.now() + timedelta(days=1)).date()
                print(f"⚠️ Could not parse date '{date_str}', defaulting to tomorrow")
            
            # Parse time
            time_formats = [
                "%H:%M",       # 14:00
                "%I:%M %p",    # 2:00 PM
                "%I %p",       # 2 PM
                "%H:%M:%S",    # 14:00:00
            ]
            
            parsed_time = None
            for time_format in time_formats:
                try:
                    parsed_time = datetime.strptime(time_str, time_format).time()
                    break
                except ValueError:
                    continue
            
            if not parsed_time:
                # Default to 10:00 AM if parsing fails
                parsed_time = datetime.strptime("10:00", "%H:%M").time()
                print(f"⚠️ Could not parse time '{time_str}', defaulting to 10:00 AM")
            
            # Combine date and time
            appointment_data["start_time"] = datetime.combine(parsed_date, parsed_time)
            
            # Set end time (1 hour later by default)
            appointment_data["end_time"] = appointment_data["start_time"] + timedelta(hours=1)
            
        else:
            # Default to tomorrow at 10:00 AM if no date/time provided
            tomorrow = datetime.now() + timedelta(days=1)
            appointment_data["start_time"] = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0)
            appointment_data["end_time"] = appointment_data["start_time"] + timedelta(hours=1)
            print(f"⚠️ No date/time provided, defaulting to tomorrow 10:00 AM")
        
        print(f"📝 Parsed appointment: {customer_name} on {appointment_data['start_time']} - {appointment_data['title']}")
        return appointment_data
        
    except Exception as e:
        print(f"❌ Error parsing appointment string: {e}")
        print(f"❌ Original string: {appointment_str}")
        raise e

class GeminiLiveConnection:
    """Manages connection to Gemini Live API for a WebSocket client."""

    def __init__(self, client_websocket: WebSocket, call_session_id: int):
        self.client_ws = client_websocket
        self.gemini_ws: Optional[object] = None
        self.is_connected = False
        self.audio_chunks = []
        self.user_transcript_buffer = (
            ""  # Buffer for accumulating user transcript chunks
        )
        self.ai_transcript_buffer = ""  # Buffer for accumulating AI transcript chunks
        self.last_user_transcript_time = 0
        self.last_ai_transcript_time = 0
        self.last_sent_transcript = ""  # Store last sent transcript for supplementation
        self.pending_transcript_task = None  # Track pending delayed transcript
        self.pending_transcript_content = ""  # Store content of pending transcript
        self.proper_names = [
            "Mori Pines",
            "Gamuda Cove",
            "Enso Woods",
        ]  # Key property names to preserve
        self.transcript_timeout = (
            2.0  # Seconds to wait before sending incomplete transcript
        )
        self.last_rag_result = ""  # Store last RAG result for verification
        self.last_rag_query = ""   # Store last RAG query for verification
        self.call_session_id = call_session_id  # Store call session ID for database operations
        self._processing_response = False  # Flag to prevent duplicate processing
        self._initial_greeting_sent = False  # Flag to prevent duplicate initial greeting
        self.pending_appointment_data = ""  # Store appointment details during call session
        
    async def connect_to_gemini(self):
        """Connect to Gemini Live API."""
        try:
            ws_url = f"wss://{GEMINI_HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}"
            print(f"🔗 Connecting to Gemini Live...")
            logger.info("Connecting to Gemini Live API...")

            ssl_context = ssl.create_default_context(cafile=certifi.where())
            self.gemini_ws = await connect(ws_url, ssl=ssl_context)

            # Define RAG function for Gemini to call
            rag_function_declaration = {
                "name": "search_knowledge_base", 
                "description": """Search the knowledge base for property information.

                Use this function when users ask about:
                - Specific properties: Mori Pines, Gamuda Cove, Enso Woods
                - Property features, amenities, layouts, specifications  
                - Pricing, cost, or budget questions
                - General property inquiries or comparisons
                - Any real estate information requests
                
                Always call this function for property-related questions to get accurate, up-to-date information. Use only the information returned by this function in your response.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's complete question or any property-related terms they mentioned. Include context like 'Tell me about Mori Pines' or 'What properties are available'."
                        }
                    },
                    "required": ["query"],
                },
            }
            
            # Define appointment retrieval function for Gemini to call
            appointment_function_declaration = {
                "name": "get_current_appointments",
                "description": """
                Use this function to retrieve current appointment data from the database.
                
                WHEN TO CALL:
                1. When customer asks about available time slots
                2. When customer wants to schedule an appointment
                3. When customer asks "when are you available?"
                4. When customer mentions wanting to book a viewing
                5. When customer asks about appointment availability
                
                This function returns all current appointments so you can determine:
                - Which time slots are already booked
                - What times are available for new appointments
                - Current appointment schedule to avoid conflicts
                """,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Define appointment storage function for Gemini to call
            store_appointment_function_declaration = {
                "name": "store_appointment_details",
                "description": """
                Use this function to store confirmed appointment details during the call.
                
                WHEN TO CALL:
                1. When customer has confirmed they want to book an appointment
                2. After you have gathered customer name, phone number, and preferred date/time
                3. When customer says "yes" to booking confirmation
                4. Before telling customer "your appointment is confirmed"
                
                Store all appointment information in one complete string including:
                - Customer full name
                - Customer phone number  
                - Appointment date and time
                - Purpose/type of appointment (property viewing, consultation, etc.)
                - Any special notes or requests
                
                This stores the data temporarily during the call for processing after the call ends.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_details": {
                            "type": "string",
                            "description": "Complete appointment information in format: 'Customer: [Name], Phone: [Number], Date: [Date], Time: [Time], Purpose: [Property viewing/consultation], Notes: [Any additional info]'"
                        }
                    },
                    "required": ["appointment_details"]
                }
            }
            
            # Send initial setup with audio response modality, transcription, optional VAD, and RAG tool
            setup_message = {
                "setup": {
                    "model": GEMINI_MODEL,
                    "generation_config": {
                        "response_modalities": ["AUDIO"],
                        "speech_config": {
                            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}
                        },
                        # "temperature": 0.7,  # Higher temperature for more natural, varied speech
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
                        "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",  # More sensitive to speech end
                        "prefix_padding_ms": 100,  # Include 100ms before detected speech start
                        "silence_duration_ms": 300,  # Wait 300ms of silence before considering speech ended
                    }
                }
                print(f"🎙️ VAD enabled with HIGH sensitivity settings")
            else:
                print(f"🎙️ VAD disabled - using continuous audio streaming")
            
            # 2. Property type preference (Semi-detached, Terrace, Bungalow, Apartments)
            # 3. Purpose of purchase (investment, own stay, family)
            
            # Add system instruction and tools
            setup_message["setup"]["system_instruction"] = {
                "parts": [{
                    "text": """
                            You are Gina, a friendly and professional sales consultant for Gamuda Cove sales gallery located in Bandar Gamuda Cove, Kuala Langat, Selangor.

                            IMPORTANT: Always use your available functions to get accurate information.
                            
                            For property questions: Use search_knowledge_base() to get current property details.
                            For appointment questions: Use get_current_appointments() to check availability.
                            
                            Respond naturally and conversationally after getting function results.

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
                            - Use the search_knowledge_base function to get specific property information
                            - Use the get_current_appointments function to check available time slots
                            - Use the store_appointment_details function to save confirmed bookings
                            - Guide customers through the appointment booking process

                            CONVERSATION FLOW:
                            1. When prompted to greet, provide a warm welcome as Gina from Gamuda Cove sales gallery
                            2. For property questions, immediately call search_knowledge_base() and provide specific information
                            3. Guide them toward booking an appointment after providing the requested information

                            APPOINTMENT BOOKING PROCESS:
                            When customers show interest in booking:
                            1. Call get_current_appointments() to check availability
                            2. Gather customer details: full name, phone number, preferred date/time
                            3. When customer confirms booking, immediately call store_appointment_details() with format:
                               "Customer: [Full Name], Phone: [Phone Number], Date: [Date], Time: [Time], Purpose: [Property viewing/consultation], Notes: [Any special requests]"
                            4. Confirm to customer that their appointment is booked

                            MANDATORY FUNCTION CALLING RULES:
                            🚨 MUST CALL search_knowledge_base() FOR:
                            - ANY mention of "Mori Pines", "Gamuda Cove", "Enso Woods", or any property name
                            - ANY words like "property", "development", "project", "township"
                            - ANY pricing, cost, budget, or "how much" questions
                            - ANY features, amenities, layouts, specifications questions
                            - ANY comparisons between properties or projects
                            - ANY general questions like "what do you have", "tell me about"
                            - ANY questions about availability, floor plans, or details
                            - EVERY real estate information request

                            🚨 MUST CALL get_current_appointments() FOR:
                            - ANY mention of "appointment", "schedule", "book", "available", "when"
                            - ANY questions about time slots or availability
                            - ANY interest in viewing or meeting
                            
                            🚨 MUST CALL store_appointment_details() FOR:
                            - When customer confirms they want to book an appointment
                            - After gathering customer name, phone, date, and time
                            - Before telling customer "your appointment is confirmed"
                            - To save the booking details during the call

                            CRITICAL ANTI-HALLUCINATION RULES:
                            - You can ONLY use information returned by function calls
                            - FORBIDDEN: Creating any property details not in function response
                            - FORBIDDEN: Adding bedroom counts, prices, sizes not provided by functions
                            - MANDATORY: If function says "1,785 to 2,973 sq ft" - use that exact range
                            - MANDATORY: If function says "3 to 5 bedrooms" - use that exact range
                            - NEVER mix function data with your own knowledge
                            - If function returns no data, say "I don't have that information available"
                            """
                }]
            }

            # Add tools
            setup_message["setup"]["tools"] = [
                {
                    "function_declarations": [rag_function_declaration, appointment_function_declaration, store_appointment_function_declaration]
                }
            ]
            print(f"📤 Sending setup to Gemini ({GEMINI_MODEL}) with RAG, appointment retrieval, and appointment storage functions")
            
            await self.gemini_ws.send(json.dumps(setup_message))

            # Wait for setup response with timeout
            try:
                import asyncio

                setup_response = await asyncio.wait_for(
                    self.gemini_ws.recv(), timeout=10.0
                )
                setup_data = json.loads(setup_response)

                if "setupComplete" in setup_data:
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
            
            # Send initial greeting to trigger Gina's response (only once)
            if not self._initial_greeting_sent:
                # Send a simple prompt to trigger AI greeting without user input
                initial_prompt = "Please greet the customer as Gina from Gamuda Cove sales gallery."
                await self.send_text_to_gemini(initial_prompt)
                print(f"👋 Sent greeting prompt to trigger Gina's welcome message")
                self._initial_greeting_sent = True
            
            # Wait a moment for any immediate responses
            await asyncio.sleep(1)
            print(f"✅ Setup complete, ready for interactions")

        except Exception as e:
            logger.error(f"Error connecting to Gemini: {e}")
            print(f"❌ GEMINI CONNECTION FAILED: {e}")
            await self.client_ws.send_json(
                {"type": "error", "message": f"Failed to connect to Gemini: {e}"}
            )

    async def _listen_to_gemini(self):
        """Listen for messages from Gemini Live API."""
        try:
            async for message in self.gemini_ws:
                msg_data = json.loads(message)

                # Debug: Log all Gemini messages to understand the structure
                # print(f"🔍 FULL Gemini message: {json.dumps(msg_data, indent=2)}")

                # Debug: Check specifically for transcription fields
                server_content = msg_data.get("serverContent", {})
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
                model_turn = server_content.get("modelTurn", {})
                if model_turn and "toolCall" not in msg_data:
                    parts = model_turn.get("parts", [])
                    for part in parts:
                        if 'text' in part:
                            text_content = part['text']
                            
                            # Check for property keywords that should trigger function calls
                            property_keywords = ["mori pines", "gamuda cove", "enso woods", "property", "development", "project", "township", "amenities", "features", "price", "cost", "bedroom", "bathroom", "size", "layout"]
                            appointment_keywords = ["appointment", "schedule", "book", "available", "when", "viewing", "visit"]
                            
                            has_property_keywords = any(keyword in text_content.lower() for keyword in property_keywords)
                            has_appointment_keywords = any(keyword in text_content.lower() for keyword in appointment_keywords)
                            
                            # Check if Gemini is providing info without function call
                            if has_property_keywords:
                                print(f"🚨 CRITICAL: GEMINI BYPASSED FUNCTION CALLING!")
                                print(f"🚨 Text contains property keywords but NO function call made")
                                print(f"🚨 Response: '{text_content[:100]}...'")
                                print(f"🚨 Keywords found: {[kw for kw in property_keywords if kw in text_content.lower()]}")
                                print(f"🚨 GEMINI MUST CALL search_knowledge_base() for ANY property question!")
                                
                            if has_appointment_keywords:
                                print(f"🚨 CRITICAL: GEMINI BYPASSED APPOINTMENT FUNCTION!")
                                print(f"🚨 Text contains appointment keywords but NO function call made")
                                print(f"🚨 Response: '{text_content[:100]}...'")
                                print(f"🚨 Keywords found: {[kw for kw in appointment_keywords if kw in text_content.lower()]}")
                                print(f"🚨 GEMINI MUST CALL get_current_appointments() for ANY appointment question!")
                            
                            # Check for fake "searching" behavior
                            fake_search_phrases = ["let me search", "i'm checking", "querying", "initiating", "retrieving", "looking up"]
                            if any(phrase in text_content.lower() for phrase in fake_search_phrases):
                                print(f"🚨 FAKE SEARCH DETECTED!")
                                print(f"🚨 Gemini is pretending to search instead of calling functions")
                                print(f"🚨 Response: '{text_content}'")
                                print(f"🚨 This is PROHIBITED behavior - functions must be called immediately!")
                            
                            # Check if Gemini is saying "I don't have information" without calling RAG
                            if any(phrase in text_content.lower() for phrase in ["don't have", "no information", "not sure", "can't help"]):
                                print(f"⚠️ GEMINI CLAIMS NO INFO WITHOUT RAG CALL: '{text_content}'")
                                print(f"⚠️ This suggests Gemini didn't call search_knowledge_base function!")
                                print(f"⚠️ Full message: {json.dumps(msg_data, indent=2)}")
                            
                            # Check if we recently sent RAG results that are being ignored
                            if hasattr(self, 'last_rag_result') and self.last_rag_result:
                                print(f"🚨 RECENT RAG RESULT WAS IGNORED!")
                                print(f"🚨 Last RAG query: '{self.last_rag_query}'")
                                print(f"🚨 Last RAG result: '{self.last_rag_result[:200]}...'")
                                print(f"🚨 Gemini should have used this information but ignored it!")
                
                # Extract transcripts using SDK-like patterns
                await self._extract_transcripts_sdk_style(msg_data)

                # Handle different types of server content
                if "serverContent" in msg_data:
                    server_content = msg_data["serverContent"]

                    # Check for user transcript in grounding metadata
                    if "groundingMetadata" in server_content:
                        grounding = server_content["groundingMetadata"]
                        if "groundingSupports" in grounding:
                            for support in grounding["groundingSupports"]:
                                if (
                                    "segment" in support
                                    and "text" in support["segment"]
                                ):
                                    user_text = support["segment"]["text"]
                                    print(
                                        f'👤 User said (from grounding): "{user_text}"'
                                    )
                                    await self.client_ws.send_json(
                                        {
                                            "type": "user_transcript",
                                            "content": user_text,
                                        }
                                    )

                    # Check for user transcript (clientContent) - alternative location
                    if "clientContent" in server_content:
                        client_content = server_content["clientContent"]
                        turns = client_content.get("turns", [])

                        for turn in turns:
                            if turn.get("role") == "USER":
                                parts = turn.get("parts", [])
                                for part in parts:
                                    if "text" in part:
                                        user_text = part["text"]
                                        print(f'👤 User said: "{user_text}"')

                                        # Send user transcript to frontend
                                        await self.client_ws.send_json(
                                            {
                                                "type": "user_transcript",
                                                "content": user_text,
                                            }
                                        )

                    # Handle AI model responses
                    model_turn = server_content.get("modelTurn", {})
                    if model_turn:
                        parts = model_turn.get("parts", [])

                        if parts:
                            text_parts = [p for p in parts if "text" in p]
                            audio_parts = [p for p in parts if "inlineData" in p]

                            # Extract and send any text responses to frontend
                            if text_parts and not self._processing_response:
                                self._processing_response = True
                                try:
                                    text_content = text_parts[0]['text']
                                    
                                    # Clean the text content to remove tool artifacts
                                    cleaned_content = self._clean_debug_artifacts(text_content)
                                    
                                    if cleaned_content and cleaned_content.strip():
                                        text_preview = cleaned_content[:60] + ('...' if len(cleaned_content) > 60 else '')
                                        print(f"📨 Gemini text (cleaned): \"{text_preview}\"")
                                        
                                        # Send cleaned text to frontend
                                        await self.client_ws.send_json({
                                            "type": "text",
                                            "content": cleaned_content
                                        })
                                    else:
                                        print(f"🗑️ Filtered out text response containing only tool artifacts")
                                finally:
                                    self._processing_response = False
                            
                            if audio_parts:
                                audio_size = len(
                                    audio_parts[0]["inlineData"].get("data", "")
                                )
                                # print(f"🔊 Gemini audio: {audio_size} chars")

                    if server_content.get("turnComplete"):
                        print(f"✅ Turn complete")
                    if server_content.get("interrupted"):
                        print(f"⏸️ AI response interrupted by user")
                        # Forward interruption to frontend immediately
                        await self.client_ws.send_json(
                            {
                                "type": "interrupted",
                                "message": "AI response interrupted by user speech",
                            }
                        )

                # Handle function calls from Gemini
                if "toolCall" in msg_data:
                    await self._handle_tool_call(msg_data["toolCall"])
                else:
                    await self._process_gemini_message(msg_data)

        except Exception as e:
            logger.error(f"Error listening to Gemini: {e}")
            print(f"❌ GEMINI LISTENING ERROR: {e}")
            await self.client_ws.send_json(
                {"type": "error", "message": f"Gemini connection error: {e}"}
            )

    async def _extract_transcripts_sdk_style(self, msg_data: dict):
        """Extract transcripts using various possible field names."""
        try:
            server_content = msg_data.get("serverContent", {})

            # Check multiple possible field names for input transcription (user speech)
            input_transcript_fields = [
                "input_transcription",
                "inputTranscription",
                "userTranscript",
                "speechRecognition",
            ]
            for field in input_transcript_fields:
                if field in server_content:
                    transcript_data = server_content[field]
                    print(
                        f"🎯 Found user transcript field '{field}': {transcript_data}"
                    )

                    # Try different text field names
                    text_content = None
                    for text_field in ["text", "transcript", "content"]:
                        if text_field in transcript_data:
                            text_content = transcript_data[text_field]
                            break

                    if text_content:
                        # Accumulate transcript chunks
                        self.user_transcript_buffer += text_content
                        print(
                            f'👤 User transcript chunk: "{text_content}" (buffer: "{self.user_transcript_buffer}")'
                        )
                        print(
                            f"👤 Chunk length: {len(text_content)}, Buffer length: {len(self.user_transcript_buffer)}"
                        )

                        # Debug: Check if this chunk contains key words
                        key_words = ['mori', 'pines', 'gamuda', 'cove', 'property', 'project', 'development', 'township', 'estate', 'estate', 'budget']
                        for word in key_words:
                            if word in text_content.lower():
                                print(
                                    f"🎯 FOUND KEY WORD '{word}' in transcript chunk: \"{text_content}\""
                                )

                        # Check for potential language detection issues
                        if any(
                            "\u3040" <= char <= "\u309f" or "\u30a0" <= char <= "\u30ff"
                            for char in text_content
                        ):
                            print(
                                f'⚠️ Japanese characters detected in user speech: "{text_content}"'
                            )
                        if any("\u4e00" <= char <= "\u9fff" for char in text_content):
                            print(
                                f'⚠️ Chinese characters detected in user speech: "{text_content}"'
                            )

                        # Improved transcript handling with proper name preservation
                        current_time = time.time()
                        should_send = False

                        # Check if buffer contains proper names - if so, wait longer to get full context
                        contains_proper_name = any(
                            name.lower() in self.user_transcript_buffer.lower()
                            for name in self.proper_names
                        )

                        # Send conditions (more conservative to preserve proper names)
                        if (
                            text_content.endswith(".")
                            or text_content.endswith("?")
                            or text_content.endswith("!")
                        ):
                            should_send = True
                            print(
                                f"👤 Detected sentence ending punctuation: '{text_content[-1]}'"
                            )
                        elif (
                            len(self.user_transcript_buffer.strip().split()) >= 8
                        ):  # Increased from 4 to 8 words
                            should_send = True
                            print(
                                f"👤 Sending transcript after {len(self.user_transcript_buffer.strip().split())} words"
                            )
                        elif (
                            contains_proper_name
                            and len(self.user_transcript_buffer.strip().split()) >= 6
                        ):
                            should_send = True
                            print(
                                f"👤 Sending transcript with proper name after {len(self.user_transcript_buffer.strip().split())} words"
                            )
                        elif (
                            current_time - self.last_user_transcript_time
                            > self.transcript_timeout
                        ):
                            should_send = True
                            print(
                                f"👤 Sending transcript due to timeout ({self.transcript_timeout}s)"
                            )

                        if should_send and self.user_transcript_buffer.strip():
                            # Clean up the transcript before sending
                            cleaned_transcript = self._clean_transcript(
                                self.user_transcript_buffer.strip()
                            )
                            print(
                                f'👤 Sending complete user transcript: "{cleaned_transcript}"'
                            )

                            # Store the transcript for potential supplementation
                            self.last_sent_transcript = cleaned_transcript

                            # Save user transcript to database
                            try:
                                db = next(get_db())
                                create_session_message(
                                    db=db,
                                    session_id=self.call_session_id,
                                    message=cleaned_transcript,
                                    message_by="User",
                                )
                                print(f"💾 Saved user transcript to database")
                            except Exception as e:
                                print(
                                    f"❌ Error saving user transcript to database: {e}"
                                )
                            finally:
                                if db:
                                    db.close()

                            # Send to frontend
                            await self.client_ws.send_json(
                                {
                                    "type": "user_transcript",
                                    "content": cleaned_transcript,
                                }
                            )
                            self.user_transcript_buffer = ""  # Clear buffer
                        else:
                            print(
                                f'👤 Buffering transcript: "{self.user_transcript_buffer}"'
                            )

                        self.last_user_transcript_time = current_time

            # Check multiple possible field names for output transcription (AI speech)
            output_transcript_fields = [
                "output_transcription",
                "outputTranscription",
                "aiTranscript",
                "speechOutput",
            ]
            for field in output_transcript_fields:
                if field in server_content:
                    transcript_data = server_content[field]
                    # print(f"🎯 Found AI transcript field '{field}': {transcript_data}")

                    # Try different text field names
                    text_content = None
                    for text_field in ["text", "transcript", "content"]:
                        if text_field in transcript_data:
                            text_content = transcript_data[text_field]
                            break

                    if text_content:
                        # No filtering needed - process all content
                        
                        # Accumulate AI transcript chunks
                        self.ai_transcript_buffer += text_content
                        # print(f"🤖 AI transcript chunk: \"{text_content}\" (buffer: \"{self.ai_transcript_buffer}\")")

                        # Only send transcript when we detect definitive end of sentence
                        # Remove timing-based sending to avoid premature messages
                        if (text_content.endswith('.') or text_content.endswith('?') or 
                            text_content.endswith('!') or text_content.endswith('\n')):
                            
                            if self.ai_transcript_buffer.strip() and not self._processing_response:
                                self._processing_response = True
                                try:
                                    # Get the complete transcript
                                    transcript = self.ai_transcript_buffer.strip()
                                    
                                    print(f"🤖 Sending complete AI transcript: \"{transcript}\"")
                                    
                                    # Save AI transcript to database
                                    try:
                                        db = next(get_db())
                                        create_session_message(
                                            db=db,
                                            session_id=self.call_session_id,
                                            message=transcript,
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
                                        "content": transcript
                                    })
                                    
                                    self.ai_transcript_buffer = ""  # Clear buffer
                                finally:
                                    self._processing_response = False
                        
                        self.last_ai_transcript_time = time.time()

            # Handle audio chunks (for playback)
            if "audio_chunk" in server_content:
                audio_chunk = server_content["audio_chunk"]
                if "data" in audio_chunk:
                    print(f"🔊 Audio chunk received: {len(audio_chunk['data'])} chars")

            # Handle Voice Activity Detection (VAD) events
            if "activity_start" in msg_data or "activityStart" in msg_data:
                print(f"🎙️ VAD: User started speaking - will interrupt AI if speaking")
                # Send activity start notification to frontend
                await self.client_ws.send_json(
                    {"type": "activity_start", "message": "User started speaking"}
                )
                # Send interruption signal to stop AI speech immediately
                await self.client_ws.send_json(
                    {"type": "interrupted", "message": "AI speech interrupted by user"}
                )

                # Send audio stream end to flush any buffered audio
                if self.gemini_ws and self.is_connected:
                    try:
                        audio_stream_end_message = {
                            "realtimeInput": {"audioStreamEnd": True}
                        }
                        await self.gemini_ws.send(json.dumps(audio_stream_end_message))
                        print(f"📤 Sent audio stream end signal to Gemini")
                    except Exception as e:
                        print(f"❌ Error sending audio stream end: {e}")
                # Clear any old transcript buffer when user starts speaking
                self.user_transcript_buffer = ""
                # Clear last sent transcript to ensure we don't supplement old messages
                self.last_sent_transcript = ""

            if "activity_end" in msg_data or "activityEnd" in msg_data:
                print(f"🎙️ VAD: User stopped speaking")
                # Send activity end notification to frontend
                await self.client_ws.send_json(
                    {"type": "activity_end", "message": "User stopped speaking"}
                )
                # Always flush user transcript when user stops speaking, with delay to prevent UI flickering
                if self.user_transcript_buffer.strip():
                    print(
                        f'👤 Scheduling delayed user transcript send: "{self.user_transcript_buffer.strip()}"'
                    )
                    # Store the current transcript for delayed sending (cleaned)
                    pending_transcript = self._clean_transcript(
                        self.user_transcript_buffer.strip()
                    )
                    self.user_transcript_buffer = ""
                    self.pending_transcript_content = (
                        pending_transcript  # Store for potential supplementation
                    )

                    # Cancel any existing pending transcript task
                    if (
                        self.pending_transcript_task
                        and not self.pending_transcript_task.done()
                    ):
                        self.pending_transcript_task.cancel()

                    # Schedule delayed send (800ms delay to allow for potential RAG supplementation)
                    self.pending_transcript_task = asyncio.create_task(
                        self._send_delayed_transcript(pending_transcript)
                    )
                else:
                    print(f"👤 No user transcript to flush (buffer was empty)")

            # Handle VAD speech detected event (indicates audio was detected but not necessarily speech)
            if "speechDetected" in msg_data:
                speech_detected = msg_data["speechDetected"]
                print(f"🎙️ VAD: Speech detected = {speech_detected}")
                await self.client_ws.send_json(
                    {"type": "speech_detected", "detected": speech_detected}
                )

            # Removed generic text field search to reduce debug noise

        except Exception as e:
            print(f"❌ Transcript extraction error: {e}")

    async def _send_delayed_transcript(self, transcript_content: str):
        """Send transcript after a delay, unless cancelled by supplementation."""
        try:
            # Wait for 800ms to allow potential RAG supplementation
            await asyncio.sleep(0.8)

            # Check if this transcript is still the latest (not replaced by supplementation)
            if (
                self.pending_transcript_task
                and not self.pending_transcript_task.cancelled()
            ):
                print(f'👤 Sending delayed transcript: "{transcript_content}"')
                await self.client_ws.send_json(
                    {"type": "user_transcript", "content": transcript_content}
                )
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

            for function_call in tool_call.get("functionCalls", []):
                function_name = function_call.get("name")
                function_args = function_call.get("args", {})
                function_id = function_call.get("id")

                print(f"🔍 Calling function: {function_name}")
                print(f"📋 Args: {function_args}")

                if function_name == "search_knowledge_base":
                    # Execute RAG search
                    query = function_args.get("query", "")
                    print(f"🔍 Searching RAG: '{query}'")
                    print(f"🔍 RAG service chain available: {bool(rag_service.chain)}")

                    # Supplement incomplete transcript ONLY if we have a pending transcript from current user input
                    # This ensures we don't accidentally modify old messages
                    if (
                        self.pending_transcript_task
                        and not self.pending_transcript_task.done()
                        and self.pending_transcript_content
                    ):
                        # We have a pending transcript from the current user input - supplement it
                        current_transcript = self.pending_transcript_content
                        print(
                            f"🔍 Found pending transcript task - will supplement before sending"
                        )
                        print(f'   Pending transcript: "{current_transcript}"')
                        print(f'   RAG query: "{query}"')

                        # Cancel the pending task so we can send supplemented version immediately
                        self.pending_transcript_task.cancel()

                        # Check if the query contains information missing from transcript
                        transcript_lower = current_transcript.lower()
                        query_lower = query.lower()

                        # If query has content not in transcript, create a supplemented version
                        missing_words = [
                            word
                            for word in query_lower.split()
                            if word not in transcript_lower
                        ]
                        if missing_words:
                            # Insert the missing words before the punctuation
                            if current_transcript.endswith("?"):
                                supplemented_transcript = f"{current_transcript[:-1]} {' '.join(missing_words)}?"
                            else:
                                supplemented_transcript = (
                                    f"{current_transcript} {' '.join(missing_words)}"
                                )

                            print(f"🔧 Supplementing current incomplete transcript:")
                            print(f'   Original: "{current_transcript}"')
                            print(f"   Missing words: {missing_words}")
                            print(f'   Result: "{supplemented_transcript}"')

                            # Send as initial transcript (replacing the delayed one)
                            await self.client_ws.send_json(
                                {
                                    "type": "user_transcript",
                                    "content": supplemented_transcript,
                                }
                            )

                            self.last_sent_transcript = supplemented_transcript
                            self.pending_transcript_task = (
                                None  # Clear pending task reference
                            )
                            self.pending_transcript_content = (
                                ""  # Clear pending content
                            )
                        else:
                            # No missing words, just send the original transcript
                            print(
                                f"🔍 No supplementation needed, sending original transcript"
                            )
                            await self.client_ws.send_json(
                                {
                                    "type": "user_transcript",
                                    "content": current_transcript,
                                }
                            )
                            self.last_sent_transcript = current_transcript
                            self.pending_transcript_task = None
                            self.pending_transcript_content = ""
                    else:
                        print(
                            f"🔍 No pending transcript to supplement - RAG query processed independently"
                        )

                        # Alternative: If no pending transcript but we have a RAG query,
                        # it means Gemini understood audio that STT might have missed
                        if query and not self.last_sent_transcript:
                            print(
                                f"🔧 STT might have missed audio - Gemini understood: '{query}'"
                            )
                            print(f"🔧 Consider this as evidence of STT quality issues")

                            # Log for analysis
                            print(f"📊 AUDIO PROCESSING DISCREPANCY:")
                            print(
                                f"   STT Result: {self.last_sent_transcript or 'NONE'}"
                            )
                            print(f"   Gemini Understood: {query}")
                            print(
                                f"   This suggests STT quality issues with proper names/complex phrases"
                            )

                    rag_result = await process_rag_query(query)
                    
                    # Get clean answer text directly
                    answer_text = rag_result.get('answer', '').strip()
                    
                    if answer_text and answer_text != "I couldn't process your question.":
                        # Simple cleaning - just the basic text
                        result_text = answer_text.replace("According to", "").replace("Based on", "").strip()
                        print(f"📚 RAG result: '{result_text[:100]}...'")
                    else:
                        result_text = "I don't have specific information about that."
                        print(f"📚 No RAG answer found")
                    
                    # Send ONLY the clean text as function response with explicit instruction
                    function_response = {
                        "tool_response": {
                            "function_responses": [
                                {
                                    "id": function_id,
                                    "name": function_name,
                                    "response": {
                                        "result": {"string_value": result_text}
                                    },
                                }
                            ]
                        }
                    }
                    
                    print(f"📤 Sending clean RAG result to Gemini")
                    await self.gemini_ws.send(json.dumps(function_response))
                    
                    self.last_rag_result = result_text
                    self.last_rag_query = query
                    print(f"✅ Function response sent")
                    
                elif function_name == 'get_current_appointments':
                    # Execute appointment retrieval
                    print(f"📅 Retrieving current appointments from database")
                    
                    try:
                        # Get database session
                        db = next(get_db())
                        
                        # Get all appointments using the CRUD function
                        appointments = AppointmentCRUD.get_all_appointments(db)
                        
                        # Format appointments for Gemini
                        if appointments:
                            appointment_data = []
                            for appointment in appointments:
                                appointment_info = {
                                    "id": appointment.id,
                                    "call_session_id": appointment.call_session_id,
                                    "title": appointment.title,
                                    "start_time": appointment.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "end_time": appointment.end_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "created_at": appointment.created_at.strftime("%Y-%m-%d %H:%M:%S")
                                }
                                appointment_data.append(appointment_info)
                            
                            result_text = f"Current appointments in database: {len(appointments)} appointments found. " + \
                                        "Here are the scheduled appointments: " + \
                                        "; ".join([
                                            f"'{apt['title']}' on {apt['start_time']} to {apt['end_time']}"
                                            for apt in appointment_data
                                        ])
                            print(f"📅 Found {len(appointments)} appointments")
                        else:
                            result_text = "No appointments currently scheduled in the database. All time slots are available for booking."
                            print(f"📅 No appointments found")
                        
                    except Exception as e:
                        result_text = f"Error retrieving appointments: {str(e)}"
                        print(f"❌ Error retrieving appointments: {e}")
                    finally:
                        if 'db' in locals():
                            db.close()
                    
                    # Send appointment data as function response
                    function_response = {
                        "tool_response": {
                            "function_responses": [{
                                "id": function_id,
                                "name": function_name,
                                "response": {"result": {"string_value": result_text}}
                            }]
                        }
                    }
                    
                    print(f"📤 Sending appointment data to Gemini")
                    await self.gemini_ws.send(json.dumps(function_response))
                    print(f"✅ Appointment function response sent")
                    
                elif function_name == 'store_appointment_details':
                    # Store appointment details temporarily during call session
                    appointment_details = function_args.get('appointment_details', '')
                    print(f"📝 Storing appointment details during call session")
                    
                    if appointment_details.strip():
                        # Store the appointment data in the session
                        self.pending_appointment_data = appointment_details.strip()
                        
                        print(f"📝 Appointment stored: {appointment_details}")
                        
                        result_text = f"Appointment details have been successfully recorded for this call session. The appointment will be processed when the call ends."
                        
                        # Send confirmation to Gemini
                        function_response = {
                            "tool_response": {
                                "function_responses": [{
                                    "id": function_id,
                                    "name": function_name,
                                    "response": {"result": {"string_value": result_text}}
                                }]
                            }
                        }
                        
                        print(f"📤 Sending appointment storage confirmation to Gemini")
                        await self.gemini_ws.send(json.dumps(function_response))
                        print(f"✅ Appointment storage confirmation sent")
                        
                    else:
                        result_text = "No appointment details provided. Please provide complete appointment information."
                        
                        function_response = {
                            "tool_response": {
                                "function_responses": [{
                                    "id": function_id,
                                    "name": function_name,
                                    "response": {"result": {"string_value": result_text}}
                                }]
                            }
                        }
                        
                        print(f"⚠️ Empty appointment details provided")
                        await self.gemini_ws.send(json.dumps(function_response))
                    
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
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                await self.client_ws.send_json(
                    {
                        "type": "audio",
                        "data": audio_b64,
                        "config": {
                            "sample_rate": OUTPUT_AUDIO_CONFIG.sample_rate,
                            "channels": OUTPUT_AUDIO_CONFIG.channels,
                        },
                    }
                )
            
            # Handle text responses (avoid duplicate processing)
            if text_response := extract_text_response(msg_data):
                if not self._processing_response:
                    self._processing_response = True
                    try:
                        # Send response directly - no complex cleaning needed
                        await self.client_ws.send_json({
                            "type": "text",
                            "content": text_response
                        })
                    finally:
                        self._processing_response = False
            
            # Handle turn completion
            if "turnComplete" in msg_data.get("serverContent", {}):
                # Flush any remaining transcript buffers
                if self.ai_transcript_buffer.strip():
                    # Clean final transcript before flushing
                    cleaned_final = self._clean_debug_artifacts(self.ai_transcript_buffer.strip())
                    if cleaned_final:
                        print(f"🤖 Flushing final AI transcript: \"{cleaned_final}\"")
                        await self.client_ws.send_json({
                            "type": "text",
                            "content": cleaned_final
                        })
                    else:
                        print(f"🗑️ Discarded final AI transcript after cleaning")
                    self.ai_transcript_buffer = ""

                if self.user_transcript_buffer.strip():
                    print(
                        f'👤 Flushing final user transcript: "{self.user_transcript_buffer.strip()}"'
                    )
                    await self.client_ws.send_json(
                        {
                            "type": "user_transcript",
                            "content": self.user_transcript_buffer.strip(),
                        }
                    )
                    self.user_transcript_buffer = ""

                self.audio_chunks = []
                await self.client_ws.send_json(
                    {
                        "type": "turn_complete",
                        "message": "Turn complete - you can speak now",
                    }
                )

            # Handle interruption
            elif "interrupted" in msg_data.get("serverContent", {}):
                await self.client_ws.send_json(
                    {"type": "interrupted", "message": "Response interrupted"}
                )

        except Exception as e:
            logger.error(f"Error processing Gemini message: {e}")

    def _clean_transcript(self, transcript: str) -> str:
        """Clean and improve transcript quality, especially for proper names."""
        cleaned = transcript.strip()

        # Remove non-English characters and noise words
        import re

        # Remove Thai, Chinese, Japanese characters
        cleaned = re.sub(
            r"[\u0E00-\u0E7F\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]", "", cleaned
        )

        # Remove common noise patterns
        cleaned = re.sub(r"<noise>", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)  # Normalize whitespace

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
                        cleaned = re.sub(
                            pattern, proper_name, cleaned, flags=re.IGNORECASE
                        )
                        print(
                            f"🔧 Restored proper name: '{pattern}' -> '{proper_name}'"
                        )
                        break

        return cleaned.strip()

    def _clean_debug_artifacts(self, text: str) -> str:
        """Remove debug artifacts from Gemini responses."""
        import re
        
        if not text or not text.strip():
            return ""
        
        original_text = text
        cleaned = text.strip()
        
        # Early rejection of responses that are mostly tool artifacts
        if any(pattern in cleaned.lower() for pattern in [
            'tool_outputs', 'function_responses', 'tool_response', '```tool_outputs',
            '"answer":', '{"answer"', '"id":', '"name":', '"response":', "{'answer':",
            'tool_', '```json', '{"', "{'", "answer':"
        ]):
            # Check if there's any meaningful content after removing artifacts
            temp_cleaned = re.sub(r'```[\s\S]*?```', '', cleaned, flags=re.DOTALL)
            temp_cleaned = re.sub(r'\{.*?\}', '', temp_cleaned, flags=re.DOTALL)
            temp_cleaned = temp_cleaned.strip()
            
            # If almost nothing remains, reject the entire response
            if len(temp_cleaned) < 10:
                print(f"🗑️ Rejecting response that's mostly tool artifacts: '{cleaned[:50]}...'")
                return ""
        
        # Remove all code blocks (including tool_outputs)
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned, flags=re.DOTALL)
        
        # Remove tool_outputs patterns more aggressively
        cleaned = re.sub(r'tool_outputs[\s\S]*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'tool[\s_]outputs[\s\S]*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove tool_response and function_responses structures
        cleaned = re.sub(r'tool_response[\s\S]*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r'function_responses[\s\S]*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove JSON structures completely - be more aggressive
        cleaned = re.sub(r'\{[\s\S]*?\}', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\[[\s\S]*?\]', '', cleaned, flags=re.DOTALL)
        
        # Remove any remaining backticks and quotes
        cleaned = re.sub(r'`+', '', cleaned)
        cleaned = re.sub(r'"+', '', cleaned)
        cleaned = re.sub(r"'+", '', cleaned)
        
        # Remove technical terms and patterns
        technical_patterns = [
            r'hits\s*:.*',
            r'string_value\s*:.*',
            r'result\s*:.*',
            r'according to.*',
            r'based on.*',
            r'the documents? show.*',
            r'the information indicates.*',
            r'id\s*:.*',
            r'name\s*:.*',
            r'response\s*:.*'
        ]
        
        for pattern in technical_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up lines that look like JSON keys
        lines = cleaned.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip lines that look like JSON keys or are too short
            if (len(line) > 3 and 
                not re.match(r'^["\']?[\w_]+["\']?\s*:', line) and
                not re.match(r'^[\{\[\}\]]+$', line)):
                clean_lines.append(line)
        
        cleaned = ' '.join(clean_lines)
        
        # Final whitespace cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        # Remove remaining artifacts
        cleaned = re.sub(r'^["\'\{\[\}\]]+|["\'\{\[\}\]]+$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Log cleaning if significant changes were made
        if cleaned != original_text and len(cleaned) > 0:
            print(f"🧹 Cleaned response:")
            print(f"   Original: '{original_text[:80]}{'...' if len(original_text) > 80 else ''}' ({len(original_text)} chars)")
            print(f"   Cleaned:  '{cleaned[:80]}{'...' if len(cleaned) > 80 else ''}' ({len(cleaned)} chars)")
        elif len(cleaned) == 0 and len(original_text) > 0:
            print(f"🗑️ Completely filtered out response: '{original_text[:50]}...'")
        
        return cleaned if cleaned and len(cleaned) > 3 else ""
    
    async def _generate_user_transcript(self):
        """Generate transcript from accumulated user audio buffer."""
        if not self.user_audio_buffer:
            return

        try:
            # Combine all audio chunks
            combined_audio = b"".join(self.user_audio_buffer)
            self.user_audio_buffer = []  # Clear buffer

            if len(combined_audio) < 1000:  # Skip very short audio
                return

            # Use Gemini's own speech recognition by sending the audio as a query
            # This is a workaround since we can't get direct transcripts
            print(
                f"🎯 Generating transcript for {len(combined_audio)} bytes of user audio"
            )

            # For now, just send a placeholder transcript
            # In production, you could integrate Google Speech-to-Text API here
            current_time = time.time()
            if current_time - self.last_user_transcript_time > 2:  # Avoid spam
                await self.client_ws.send_json(
                    {
                        "type": "user_transcript",
                        "content": "[User spoke - transcript not available in audio-only mode]",
                    }
                )
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
                    print(f'💬 User: "{text[:50]}..." (full: {len(text)} chars)')
                    print(f"   Full text: {text}")
                else:
                    print(f'💬 User: "{text}"')

                # Send text directly to Gemini - it will call RAG function if needed
                message = encode_text_input(text)
                await self.gemini_ws.send(json.dumps(message))
                print(f"🚀 Sent to Gemini (RAG available via function calling)")

            except Exception as e:
                logger.error(f"Error sending text to Gemini: {e}")
                print(f"❌ Text send error: {e}")
    
    async def process_pending_appointment(self):
        """Process and save pending appointment data to database when call ends."""
        if not self.pending_appointment_data:
            print("📝 No pending appointment data to process")
            return
            
        try:
            print(f"📝 Processing pending appointment for call session {self.call_session_id}")
            print(f"📝 Appointment data: {self.pending_appointment_data}")
            
            # Parse the appointment string
            appointment_data = parse_appointment_string(self.pending_appointment_data, self.call_session_id)
            
            # Get database session
            db = next(get_db())
            
            try:
                # Create appointment in database
                created_appointment = AppointmentCRUD.create_appointment(db, appointment_data)
                
                print(f"✅ Appointment successfully created in database:")
                print(f"   ID: {created_appointment.id}")
                print(f"   Title: {created_appointment.title}")
                print(f"   Start: {created_appointment.start_time}")
                print(f"   End: {created_appointment.end_time}")
                print(f"   Call Session ID: {created_appointment.call_session_id}")
                
                # Clear the pending data after successful save
                self.pending_appointment_data = ""
                
            except Exception as db_error:
                print(f"❌ Database error creating appointment: {db_error}")
                logger.error(f"Failed to create appointment in database: {db_error}")
                raise db_error
            finally:
                db.close()
                
        except Exception as e:
            print(f"❌ Error processing pending appointment: {e}")
            logger.error(f"Error processing pending appointment for call {self.call_session_id}: {e}")
            # Don't re-raise - we want call cleanup to continue even if appointment fails

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
            message_by=request.message_by,
        )
        return {
            "success": True,
            "suggestion": suggestion,
            "session_id": request.session_id,
        }
    except Exception as e:
        logger.error(f"Admin suggestion error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating admin suggestion: {str(e)}"
        )


@app.post("/api/ai/customer-suggestions")
async def customer_suggestions_endpoint(request: CustomerSuggestionRequest):
    """Generate customer suggestions for post-call actions."""
    try:
        suggestions = generate_caller_suggestions(
            conversation=request.conversation, call_session_id=request.call_session_id
        )
        return {
            "success": True,
            "suggestions": suggestions,
            "call_session_id": request.call_session_id,
        }
    except Exception as e:
        logger.error(f"Customer suggestions error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error generating customer suggestions: {str(e)}"
        )


@app.post("/api/ai/summarize")
async def summarize_conversation_endpoint(request: SummarizationRequest):
    """Summarize a conversation and store it in the database."""
    try:
        summary = summarize_text(
            conversation=request.conversation, call_session_id=request.call_session_id
        )
        return {
            "success": True,
            "summary": summary,
            "call_session_id": request.call_session_id,
        }
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error summarizing conversation: {str(e)}"
        )


@app.post("/api/ai/sentiment-analysis")
async def sentiment_analysis_endpoint(request: SentimentAnalysisRequest):
    """Analyze sentiment of a specific transcript."""
    try:
        result = analyze_sentiment_from_transcript(request.transcript_id)
        result_data = json.loads(result)

        if result_data.get("success"):
            return result_data
        else:
            raise HTTPException(
                status_code=404, detail=result_data.get("error", "Unknown error")
            )
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in sentiment analysis: {e}")
        raise HTTPException(
            status_code=500, detail="Error parsing sentiment analysis result"
        )
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error analyzing sentiment: {str(e)}"
        )


@app.post("/api/ai/session-conversation")
async def session_conversation_endpoint(request: SessionConversationRequest):
    """Retrieve full conversation for a session."""
    try:
        result_data = get_session_conversation(request.session_id)

        if result_data.get("success"):
            return result_data
        else:
            raise HTTPException(
                status_code=404, detail=result_data.get("error", "Unknown error")
            )
    except Exception as e:
        logger.error(f"Session conversation error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving session conversation: {str(e)}"
        )


# Bulk AI processing endpoint
@app.post("/api/ai/process-session")
async def process_session_endpoint(session_id: int):
    """Process a complete session with all AI services (summary, sentiment, suggestions)."""
    try:
        # Get session conversation
        conversation_data = get_session_conversation(session_id)

        if not conversation_data.get("success"):
            raise HTTPException(
                status_code=404, detail="Session not found or has no conversation"
            )

        conversation = conversation_data["conversation"]

        # Format conversation for processing
        formatted_conversation = [
            {"role": msg["message_by"], "content": msg["message"]}
            for msg in conversation
        ]

        # Generate summary
        summary = summarize_text(formatted_conversation, session_id)

        # Generate customer suggestions
        customer_suggestions = generate_caller_suggestions(
            formatted_conversation, session_id
        )

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
            "conversation_count": len(conversation),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session processing error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing session: {str(e)}"
        )


@app.get("/api/ai/health")
async def ai_services_health():
    """Health check for AI services."""
    try:
        # Test AI services by calling them with simple test data
        test_conversation = [
            {"role": "User", "content": "Hello"},
            {"role": "AI", "content": "Hi there"},
        ]

        # Test summarization service
        try:
            summary_test = summarize_text(
                test_conversation, call_session_id=0
            )  # Use 0 for test (won't store)
            summarization_status = (
                "healthy"
                if summary_test and not summary_test.startswith("Error")
                else "error"
            )
        except Exception as e:
            summarization_status = f"error: {str(e)}"

        # Test customer suggestions service
        try:
            suggestions_test = generate_caller_suggestions(
                test_conversation, call_session_id=None
            )
            suggestions_status = (
                "healthy"
                if suggestions_test and not suggestions_test.startswith("Error")
                else "error"
            )
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
            sentiment_status = (
                "healthy" if callable(analyze_sentiment_from_transcript) else "error"
            )
        except Exception as e:
            sentiment_status = f"error: {str(e)}"

        return {
            "status": (
                "healthy"
                if all(
                    [
                        summarization_status == "healthy",
                        suggestions_status == "healthy",
                        db_status == "healthy",
                        sentiment_status == "healthy",
                    ]
                )
                else "partial"
            ),
            "services": {
                "summarization": summarization_status,
                "customer_suggestions": suggestions_status,
                "sentiment_analysis": sentiment_status,
                "database": db_status,
                "transcript_count": transcript_count,
            },
            "api_endpoints": [
                "/api/ai/admin-suggestion",
                "/api/ai/customer-suggestions",
                "/api/ai/summarize",
                "/api/ai/sentiment-analysis",
                "/api/ai/session-conversation",
                "/api/ai/process-session",
                "/api/ai/process-session-manual/{session_id}",
                "/api/call-session/{session_id}/status",
            ],
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
                "database": "unknown",
            },
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
            raise HTTPException(
                status_code=404, detail="Session not found or has no conversation"
            )

        conversation = conversation_data["conversation"]
        if len(conversation) == 0:
            raise HTTPException(
                status_code=400, detail="Session has no messages to process"
            )

        # Start AI processing in background
        asyncio.create_task(process_call_session_ai(session_id))

        return {
            "success": True,
            "message": f"AI processing started for session {session_id}",
            "session_id": session_id,
            "message_count": len(conversation),
            "note": "Processing is running in the background. Check logs for progress.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error manually triggering AI processing for session {session_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Error starting AI processing: {str(e)}"
        )


@app.get("/api/call-session/{session_id}/status")
async def get_call_session_status(session_id: int):
    """
    Get the current status of a call session including AI processing results.
    Shows the automatically updated call session data.
    """
    try:
        db = SessionLocal()
        call_session = (
            db.query(CallSession).filter(CallSession.id == session_id).first()
        )

        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")

        # Get conversation count
        conversation_count = (
            db.query(Transcript).filter(Transcript.session_id == session_id).count()
        )

        # Check if AI processing has been completed (has end_time and summary)
        ai_processed = bool(call_session.end_time and call_session.summarized_content)

        db.close()

        return {
            "success": True,
            "call_session": {
                "id": call_session.id,
                "cust_id": call_session.cust_id,
                "start_time": (
                    call_session.start_time.isoformat()
                    if call_session.start_time
                    else None
                ),
                "end_time": (
                    call_session.end_time.isoformat() if call_session.end_time else None
                ),
                "duration_secs": call_session.duration_secs,
                "duration_formatted": (
                    f"{call_session.duration_secs // 60}m {call_session.duration_secs % 60}s"
                    if call_session.duration_secs
                    else None
                ),
                "sentiment_counts": {
                    "positive": call_session.positive,
                    "neutral": call_session.neutral,
                    "negative": call_session.negative,
                    "total": (call_session.positive or 0)
                    + (call_session.neutral or 0)
                    + (call_session.negative or 0),
                },
                "key_words": call_session.key_words,
                "summarized_content": call_session.summarized_content,
                "customer_suggestions": call_session.customer_suggestions,
                "admin_suggestions": call_session.admin_suggestions,
                "conversation_count": conversation_count,
                "ai_processed": ai_processed,
                "call_status": "completed" if call_session.end_time else "in_progress",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting call session status for {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving call session status: {str(e)}"
        )


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
        call_session = (
            db.query(CallSession).filter(CallSession.id == session_id).first()
        )

        if not call_session:
            db.close()
            raise HTTPException(status_code=404, detail="Call session not found")

        # Check if session has any conversation
        conversation_count = (
            db.query(Transcript).filter(Transcript.session_id == session_id).count()
        )

        if conversation_count == 0:
            db.close()
            raise HTTPException(
                status_code=400, detail="Session has no conversation to process"
            )

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
                "7. Check status with GET /api/call-session/{session_id}/status",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test end call for session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error simulating call end: {str(e)}"
        )


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
        create_call_session = service.create(CallSessionBase(cust_id="0123334444"))
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
        # Process pending appointment before cleanup
        if hasattr(connection, 'pending_appointment_data') and connection.pending_appointment_data:
            try:
                print(f"📝 Processing appointment before call cleanup...")
                await connection.process_pending_appointment()
            except Exception as appointment_error:
                logger.error(f"Error processing appointment during cleanup: {appointment_error}")
                print(f"❌ Appointment processing failed: {appointment_error}")
        
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
                logger.info(
                    f"Starting post-call AI processing for session {call_session_id}"
                )
                print(
                    f"🤖 Starting post-call AI processing for session {call_session_id}"
                )

                # Run AI processing in background to avoid blocking cleanup
                asyncio.create_task(process_call_session_ai(call_session_id))

            except Exception as e:
                logger.error(f"Error starting post-call AI processing: {e}")
                print(f"❌ Error starting post-call AI processing: {e}")


async def handle_client_message(connection: GeminiLiveConnection, data: dict):
    """Handle message from client."""
    try:
        message_type = data.get("type")

        if message_type == "audio":
            # Handle audio data from client
            audio_b64 = data.get("data", "")
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

        elif message_type == "text":
            # Handle text message from client
            text_content = data.get("content", "")
            if text_content:
                print(f'📝 Text received: "{text_content}"')
                await connection.send_text_to_gemini(text_content)
            else:
                print(f"⚠️ Empty text message")

        elif message_type == "ping":
            # Handle ping message for testing
            print(f"🏓 Ping")
            await connection.client_ws.send_json(
                {"type": "pong", "message": "Backend is working!"}
            )

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
            audio_data = audio_data + b"\x00"

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
        ai_services_healthy = all(
            [
                callable(get_suggestion_from_agent),
                callable(generate_caller_suggestions),
                callable(summarize_text),
                callable(analyze_sentiment_from_transcript),
            ]
        )
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
                "/api/call-session/{session_id}/status",
            ],
        },
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
            "status": "processed",
        }
    else:
        return {"received_text": text, "status": "no_rag"}


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
            test_queries = [
                query,
                "Mori Pines",
                "Enso Woods",
                "terrace",
                "property",
                "amenities",
            ]
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
                        "preview": (
                            docs[0].page_content[:200] if docs else "No documents found"
                        ),
                    }

            # Run the original query through RAG
            rag_result = await process_rag_query(query)
            print(f"  RAG result: {rag_result}")

            return {
                "query": query,
                "result": rag_result,
                "test_results": results,
                "status": "success",
            }
        except Exception as e:
            print(f"  RAG error: {e}")
            import traceback

            traceback.print_exc()
            return {"query": query, "error": str(e), "status": "error"}
    else:
        return {
            "query": query,
            "error": "RAG not initialized",
            "status": "not_initialized",
        }


@app.get("/api/test-mongodb")
async def test_mongodb_endpoint():
    """Test MongoDB connection and data."""
    try:
        if rag_service.client:
            # Test connection
            rag_service.client.admin.command("ping")

            # Get collection
            collection = rag_service.client[rag_service.DB_NAME][
                rag_service.COLLECTION_NAME
            ]

            # Get document count
            doc_count = collection.count_documents({})

            # Search for documents containing "Mori Pines" (handling line breaks and spaces)
            mori_pines_docs = list(
                collection.find(
                    {"text": {"$regex": "Mori[\\s\\n]+Pines", "$options": "i"}}
                ).limit(5)
            )

            # Get sample documents
            sample_docs = list(collection.find({}).limit(3))

            # Check if documents have embeddings
            docs_with_embeddings = collection.count_documents(
                {"embedding": {"$exists": True}}
            )

            # Check embedding field structure
            embedding_info = {}
            if sample_docs:
                first_doc = sample_docs[0]
                if "embedding" in first_doc:
                    embedding = first_doc["embedding"]
                    embedding_info = {
                        "embedding_type": str(type(embedding)),
                        "embedding_length": (
                            len(embedding)
                            if hasattr(embedding, "__len__")
                            else "unknown"
                        ),
                        "first_few_values": (
                            embedding[:5]
                            if hasattr(embedding, "__getitem__")
                            else "unknown"
                        ),
                        "is_list": isinstance(embedding, list),
                        "is_array": str(type(embedding)),
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
                        "text_preview": (
                            doc.get("text", "")[:300] + "..."
                            if len(doc.get("text", "")) > 300
                            else doc.get("text", "")
                        ),
                        "has_mori_pines": "Mori Pines" in doc.get("text", ""),
                    }
                    for doc in mori_pines_docs
                ],
                "sample_docs": [
                    {
                        "id": str(doc.get("_id", "")),
                        "content_preview": str(doc).replace(
                            rag_service.COLLECTION_NAME, ""
                        )[:200]
                        + "...",
                        "has_embedding": "embedding" in doc,
                        "doc_keys": list(doc.keys()),
                    }
                    for doc in sample_docs
                ],
                "index_name": rag_service.INDEX_NAME,
            }
        else:
            return {"status": "not_connected", "error": "MongoDB client not available"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/test-gemini-live")
async def test_gemini_live_endpoint(request_data: dict):
    """Test endpoint to send text directly to Gemini Live API."""
    text = request_data.get("text", "Hello Gemini!")
    print(f"\n🧪 TESTING GEMINI LIVE CONNECTION WITH TEXT:")
    print(f"  Text: '{text}'")

    # Create a temporary connection to test Gemini Live
    try:
        from websockets.asyncio.client import connect

        ws_url = f"wss://{GEMINI_HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}"

        async with connect(ws_url) as ws:
            print("✅ Connected to Gemini Live for testing")

            # Send setup
            setup_message = {
                "setup": {
                    "model": GEMINI_MODEL,
                    "generation_config": {"response_modalities": ["TEXT"]},
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
                "gemini_response": response_data,
            }

    except Exception as e:
        print(f"❌ Gemini Live test failed: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
