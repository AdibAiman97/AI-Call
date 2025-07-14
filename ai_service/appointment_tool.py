from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from database.connection import SessionLocal
from database.models.appointment import Appointment
from database.models.call_session import CallSession
from database.models.transcript import Transcript
from services.appointment_crud import AppointmentCRUD
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_
import json

# Constants
DEFAULT_APPOINTMENT_DURATION_HOURS = 1
TITLE_MAX_LENGTH = 50

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Template for automated appointment creation
template = ChatPromptTemplate.from_messages([
    ("system", "You automatically create appointments from transcripts when explicit date and time information is found."),
    ("user", "{input}"),
])

# Template for time extraction
time_extraction_template = ChatPromptTemplate.from_messages([
    ("system", """Extract appointment times from conversation transcripts.
    
    Look for explicit date/time mentions (e.g., "Friday at 2 PM", "tomorrow at 10:30").
    Return in ISO format: YYYY-MM-DDTHH:MM:SS
    If no specific date/time found, return "None"
    
    Return ONLY the ISO datetime string or "None" - no other text."""),
    ("user", "Extract appointment time from: {transcript_text}")
])

@tool
def create_appointment_from_transcript(transcript_id: int) -> str:
    """
    Automatically create an appointment from a transcript with overlap checking.
    """
    db = SessionLocal()
    try:
        # Get transcript
        transcript = db.query(Transcript).filter(Transcript.id == transcript_id).first()
        if not transcript:
            return json.dumps({
                "success": False,
                "error": f"Transcript {transcript_id} not found"
            })
        
        # Extract time from transcript
        start_datetime = _extract_time_from_transcript(transcript.message)
        if not start_datetime:
            return json.dumps({
                "success": False,
                "error": "No explicit appointment time found in transcript"
            })
        
        end_datetime = start_datetime + timedelta(hours=DEFAULT_APPOINTMENT_DURATION_HOURS)
        
        # Check for conflicts
        conflict = _check_overlap(db, start_datetime, end_datetime)
        if conflict:
            return json.dumps({
                "success": False,
                "error": "Scheduling conflict with existing appointment",
                "conflict_id": conflict.id
            })
        
        # Create appointment
        title = _generate_title(transcript)
        appointment_data = {
            "call_session_id": transcript.session_id,
            "title": title,
            "start_time": start_datetime,
            "end_time": end_datetime
        }
        
        appointment = AppointmentCRUD.create_appointment(db, appointment_data)
        
        return json.dumps({
            "success": True,
            "appointment": {
                "id": appointment.id,
                "title": title,
                "start_time": start_datetime.isoformat(),
                "end_time": end_datetime.isoformat()
            }
        })
        
    except SQLAlchemyError as e:
        db.rollback()
        return json.dumps({"success": False, "error": f"Database error: {str(e)}"})
    except Exception as e:
        db.rollback()
        return json.dumps({"success": False, "error": f"Error: {str(e)}"})
    finally:
        db.close()

def _check_overlap(db, start_datetime: datetime, end_datetime: datetime) -> Appointment | None:
    """Check if appointment time overlaps with existing appointments."""
    try:
        return db.query(Appointment).filter(
            and_(
                Appointment.start_time < end_datetime,
                Appointment.end_time > start_datetime
            )
        ).first()
    except SQLAlchemyError:
        return None

def _extract_time_from_transcript(transcript_text: str) -> datetime | None:
    """Extract appointment time from transcript using LLM."""
    if not transcript_text or not transcript_text.strip():
        return None
    
    try:
        chain = time_extraction_template | llm | StrOutputParser()
        result = chain.invoke({"transcript_text": transcript_text}).strip()
        
        if result.lower() == "none" or not result:
            return None
        
        return datetime.fromisoformat(result.replace('Z', '+00:00'))
    except Exception:
        return None

def _generate_title(transcript: Transcript) -> str:
    """Generate appointment title from transcript."""
    # Use summarized content first, then key topics, then fallback
    if transcript.summarized and transcript.summarized.strip():
        title = transcript.summarized[:TITLE_MAX_LENGTH]
    elif transcript.key_topics and transcript.key_topics.strip():
        title = f"Appointment: {transcript.key_topics}"[:TITLE_MAX_LENGTH]
    else:
        title = f"Appointment from Transcript {transcript.id}"
    
    return title + "..." if len(title) == TITLE_MAX_LENGTH else title

# Bind tools to LLM
tools = [create_appointment_from_transcript]
llm_with_tools = llm.bind_tools(tools)
appointment_chain = template | llm_with_tools | StrOutputParser()