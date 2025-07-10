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
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import and_
import json

# Constants
TITLE_MAX_LENGTH = 50
PREVIEW_MAX_LENGTH = 100
MESSAGE_PREVIEW_LENGTH = 100
SUMMARY_PREVIEW_LENGTH = 100
DEFAULT_APPOINTMENT_DURATION_HOURS = 1
TITLE_WORDS_FALLBACK_COUNT = 8

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Template for automated appointment creation
template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that automatically creates appointments from transcripts. 
    You can create appointments using the transcript data automatically, or list available transcripts."""),
    ("user", "{input}"),
])

# Template for time extraction from transcript text
time_extraction_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert at extracting appointment times from conversation transcripts.
    
    Your task is to analyze the conversation and extract any explicit mentions of dates and times for scheduling appointments.
    
    Instructions:
    1. Look for explicit date and time mentions (e.g., "next Friday at 2 PM", "tomorrow at 10:30", "December 15th at 3 PM")
    2. If you find a clear date and time, return it in ISO 8601 format: YYYY-MM-DDTHH:MM:SS
    3. If no specific date/time is mentioned, return exactly the word "None"
    4. Use the current context to resolve relative dates (today, tomorrow, next week, etc.)
    5. Assume the current year if not specified
    6. If only time is mentioned without date, assume it's for today
    
    Return ONLY the ISO format datetime string or "None" - no other text."""),
    ("user", "Extract the appointment date and time from this conversation: {transcript_text}")
])

@tool
def create_appointment_from_transcript(transcript_id: int) -> str:
    """
    Automatically create an appointment from a transcript using the transcript's content and timing.
    
    Args:
        transcript_id: The ID of the transcript
    
    Returns:
        JSON string with appointment details or error information
    """
    db = SessionLocal()
    try:
        # Get transcript
        transcript = db.query(Transcript).filter(Transcript.id == transcript_id).first()
        if not transcript:
            return json.dumps({
                "success": False,
                "error": f"Transcript with ID {transcript_id} not found",
                "transcript_id": transcript_id
            })
        
        # Create title with improved fallback logic
        title = _generate_appointment_title(transcript)
        
        # Get call session for timing information
        call_session = db.query(CallSession).filter(CallSession.id == transcript.session_id).first()
        
        start_datetime, end_datetime = _determine_appointment_times(transcript, call_session, llm)
        
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
                "end_time": end_datetime.isoformat(),
                "call_session_id": transcript.session_id,
                "transcript_id": transcript_id
            },
            "message": f"Appointment created successfully with ID: {appointment.id}"
        })
        
    except SQLAlchemyError as e:
        db.rollback()
        return json.dumps({
            "success": False,
            "error": f"Database error creating appointment: {str(e)}",
            "transcript_id": transcript_id
        })
    except ValueError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid data for appointment creation: {str(e)}",
            "transcript_id": transcript_id
        })
    except Exception as e:
        db.rollback()
        return json.dumps({
            "success": False,
            "error": f"Unexpected error creating appointment: {str(e)}",
            "transcript_id": transcript_id
        })
    finally:
        db.close()

@tool
def list_available_transcripts() -> str:
    """
    List all transcripts that don't have appointments yet using efficient LEFT JOIN.
    
    Returns:
        JSON string with available transcripts
    """
    db = SessionLocal()
    try:
        # Use LEFT JOIN for better performance instead of WHERE NOT IN
        available_transcripts = db.query(Transcript).outerjoin(
            Appointment, Transcript.session_id == Appointment.call_session_id
        ).filter(Appointment.id == None).all()
        
        if not available_transcripts:
            return json.dumps({
                "success": True,
                "transcripts": [],
                "message": "No transcripts available for appointment creation"
            })
        
        transcripts_data = []
        for transcript in available_transcripts:
            message_preview = _truncate_text(transcript.message, MESSAGE_PREVIEW_LENGTH)
            summarized_preview = _truncate_text(transcript.summarized, SUMMARY_PREVIEW_LENGTH) if transcript.summarized else None
            
            transcripts_data.append({
                "id": transcript.id,
                "session_id": transcript.session_id,
                "message_by": transcript.message_by,
                "message_preview": message_preview,
                "summarized": summarized_preview,
                "key_topics": transcript.key_topics,
                "created_at": transcript.created_at.isoformat() if transcript.created_at else None
            })
        
        return json.dumps({
            "success": True,
            "transcripts": transcripts_data,
            "count": len(transcripts_data)
        }, indent=2)
        
    except SQLAlchemyError as e:
        return json.dumps({
            "success": False,
            "error": f"Database error retrieving transcripts: {str(e)}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error retrieving transcripts: {str(e)}"
        })
    finally:
        db.close()

@tool
def get_transcript_summary(session_id: int) -> str:
    """
    Get a summary of all transcripts for a specific session.
    
    Args:
        session_id: The session ID to get transcripts for
    
    Returns:
        JSON string with transcript summary for the session
    """
    db = SessionLocal()
    try:
        transcripts = db.query(Transcript).filter(Transcript.session_id == session_id).all()
        
        if not transcripts:
            return json.dumps({
                "success": True,
                "session_summary": None,
                "message": f"No transcripts found for session {session_id}"
            })
        
        session_summary = {
            "session_id": session_id,
            "total_messages": len(transcripts),
            "messages": []
        }
        
        for transcript in transcripts:
            session_summary["messages"].append({
                "id": transcript.id,
                "message_by": transcript.message_by,
                "message": transcript.message,
                "summarized": transcript.summarized,
                "key_topics": transcript.key_topics,
                "created_at": transcript.created_at.isoformat() if transcript.created_at else None
            })
        
        return json.dumps({
            "success": True,
            "session_summary": session_summary
        }, indent=2)
        
    except SQLAlchemyError as e:
        return json.dumps({
            "success": False,
            "error": f"Database error retrieving transcript summary: {str(e)}",
            "session_id": session_id
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Unexpected error retrieving transcript summary: {str(e)}",
            "session_id": session_id
        })
    finally:
        db.close()

def _extract_appointment_time_from_text(transcript_text: str, llm: ChatGoogleGenerativeAI) -> datetime | None:
    """
    Perform NLU to extract specific appointment time from transcript text.
    
    Args:
        transcript_text: The transcript text to analyze
        llm: The LLM instance for time extraction
        
    Returns:
        datetime object if time is found, None otherwise
    """
    try:
        # Use the time extraction template to analyze the transcript
        chain = time_extraction_template | llm | StrOutputParser()
        result = chain.invoke({"transcript_text": transcript_text})
        
        # Clean up the result
        result = result.strip()
        
        # Check if LLM found no time
        if result.lower() == "none" or not result:
            return None
        
        # Try to parse the ISO format datetime
        try:
            extracted_time = datetime.fromisoformat(result.replace('Z', '+00:00'))
            print(f"✅ Extracted appointment time from transcript: {extracted_time}")
            return extracted_time
        except ValueError as e:
            print(f"⚠️ LLM returned invalid datetime format: {result}")
            return None
            
    except Exception as e:
        print(f"⚠️ Error extracting time from transcript: {str(e)}")
        return None

def _generate_appointment_title(transcript: Transcript) -> str:
    """
    Generate an appropriate title for the appointment with improved fallback logic.
    
    Args:
        transcript: The transcript object
        
    Returns:
        Generated title string
    """
    title = ""
    
    # First priority: use summarized content
    if transcript.summarized and transcript.summarized.strip():
        title = _truncate_text(transcript.summarized, TITLE_MAX_LENGTH)
    # Second priority: use key topics
    elif transcript.key_topics and transcript.key_topics.strip():
        title = f"Appointment: {transcript.key_topics}"
        title = _truncate_text(title, TITLE_MAX_LENGTH)
    # Third priority: use first few words of the message
    elif transcript.message and transcript.message.strip():
        # Extract first few words instead of characters
        words = transcript.message.strip().split()
        if words:
            first_words = " ".join(words[:TITLE_WORDS_FALLBACK_COUNT])
            title = _truncate_text(first_words, TITLE_MAX_LENGTH)
    
    # Final fallback
    if not title.strip():
        title = f"Appointment from Transcript {transcript.id}"
    
    return title

def _determine_appointment_times(transcript: Transcript, call_session: CallSession | None, llm: ChatGoogleGenerativeAI) -> tuple[datetime, datetime]:
    """
    Determine appointment times using tiered approach: transcript NLU -> call_session -> transcript.created_at
    
    Args:
        transcript: The transcript object
        call_session: The call session object (can be None)
        llm: The LLM instance for time extraction
        
    Returns:
        Tuple of (start_datetime, end_datetime)
    """
    # Tier 1: Try to extract time from transcript text using NLU
    if transcript.message and transcript.message.strip():
        extracted_time = _extract_appointment_time_from_text(transcript.message, llm)
        if extracted_time:
            start_datetime = extracted_time
            end_datetime = start_datetime + timedelta(hours=DEFAULT_APPOINTMENT_DURATION_HOURS)
            print(f"✅ Used NLU extracted time: {start_datetime}")
            return start_datetime, end_datetime
    
    # Tier 2: Use call_session timestamps with improved handling
    if call_session:
        try:
            start_datetime = _parse_datetime_flexible(call_session.start_time)
            end_datetime = _parse_datetime_flexible(call_session.end_time)
            
            if start_datetime:
                # If end_time is invalid but start_time is valid, use default duration
                if not end_datetime:
                    end_datetime = start_datetime + timedelta(hours=DEFAULT_APPOINTMENT_DURATION_HOURS)
                
                print(f"✅ Used call_session timestamps: {start_datetime} to {end_datetime}")
                return start_datetime, end_datetime
            
        except (ValueError, TypeError) as e:
            print(f"⚠️ Call session timestamp parsing failed: {str(e)}")
            # Fall through to Tier 3
    
    # Tier 3: Use transcript creation time as last resort
    start_datetime = _parse_datetime_flexible(transcript.created_at)
    if start_datetime:
        end_datetime = start_datetime + timedelta(hours=DEFAULT_APPOINTMENT_DURATION_HOURS)
        print(f"✅ Used transcript creation time as fallback: {start_datetime}")
    else:
        # Final fallback - use current time
        start_datetime = datetime.now()
        end_datetime = start_datetime + timedelta(hours=DEFAULT_APPOINTMENT_DURATION_HOURS)
        print(f"⚠️ Used current time as final fallback: {start_datetime}")
    
    return start_datetime, end_datetime

def _parse_datetime_flexible(time_input: datetime | str | None) -> datetime | None:
    """
    Flexibly parse a datetime that could be a datetime object, an ISO string, or None.
    
    Args:
        time_input: The datetime object, ISO string, or None
        
    Returns:
        A datetime object or None
    """
    if isinstance(time_input, datetime):
        return time_input
    if isinstance(time_input, str) and time_input.strip():
        try:
            return datetime.fromisoformat(time_input.replace('Z', '+00:00'))
        except ValueError:
            return None
    return None

def _truncate_text(text: str | None, max_length: int) -> str:
    """
    Truncate text to specified length with ellipsis if needed.
    
    Args:
        text: The text to truncate
        max_length: Maximum length allowed
        
    Returns:
        Truncated text with ellipsis if necessary
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

# Bind tools to LLM
tools = [create_appointment_from_transcript, list_available_transcripts, get_transcript_summary]
llm_with_tools = llm.bind_tools(tools)

# Chain for processing appointment requests
appointment_chain = template | llm_with_tools | StrOutputParser()