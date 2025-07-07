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
import json

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Template for automated appointment creation
template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that automatically creates appointments from transcripts. 
    You can create appointments using the transcript data automatically, or list available transcripts."""),
    ("user", "{input}"),
])

@tool
def create_appointment_from_transcript(transcript_id: int) -> str:
    """
    Automatically create an appointment from a transcript using the transcript's content and timing.
    
    Args:
        transcript_id: The ID of the transcript
    
    Returns:
        Success or error message with appointment details
    """
    db = SessionLocal()
    try:
        # Get transcript
        transcript = db.query(Transcript).filter(Transcript.id == transcript_id).first()
        if not transcript:
            return f"Error: Transcript with ID {transcript_id} not found"
        
        # Create title from summarized content or key topics
        title = ""
        if transcript.summarized and transcript.summarized.strip():
            title = transcript.summarized[:50] + "..." if len(transcript.summarized) > 50 else transcript.summarized
        elif transcript.key_topics and transcript.key_topics.strip():
            title = f"Appointment: {transcript.key_topics}"
        else:
            # Fall back to first part of the message
            title = transcript.message[:50] + "..." if len(transcript.message) > 50 else transcript.message
        
        if not title.strip():
            title = f"Appointment from Transcript {transcript_id}"
        
        # Get call session for timing information
        call_session = db.query(CallSession).filter(CallSession.id == transcript.session_id).first()
        
        if call_session:
            try:
                # Try to parse the string timestamps from call session
                start_datetime = datetime.fromisoformat(call_session.start_time)
                end_datetime = datetime.fromisoformat(call_session.end_time)
            except (ValueError, AttributeError):
                # If parsing fails, use transcript creation time + 1 hour
                start_datetime = transcript.created_at
                end_datetime = start_datetime + timedelta(hours=1)
        else:
            # If no call session found, use transcript creation time + 1 hour
            start_datetime = transcript.created_at
            end_datetime = start_datetime + timedelta(hours=1)
        
        appointment_data = {
            "call_session_id": transcript.session_id,  # Use session_id for the foreign key
            "title": title,
            "start_time": start_datetime,
            "end_time": end_datetime
        }
        
        appointment = AppointmentCRUD.create_appointment(db, appointment_data)
        
        return f"Appointment created successfully with ID: {appointment.id}, Title: '{title}', Time: {start_datetime} to {end_datetime}"
        
    except Exception as e:
        return f"Error creating appointment: {str(e)}"
    finally:
        db.close()

@tool
def list_available_transcripts() -> str:
    """
    List all transcripts that don't have appointments yet.
    
    Returns:
        JSON string with available transcripts
    """
    db = SessionLocal()
    try:
        # Get all session IDs that already have appointments
        sessions_with_appointments = db.query(Appointment.call_session_id).all()
        existing_session_ids = [session[0] for session in sessions_with_appointments]
        
        # Get transcripts for sessions that don't have appointments
        available_transcripts = db.query(Transcript).filter(
            ~Transcript.session_id.in_(existing_session_ids)
        ).all()
        
        if not available_transcripts:
            return "No transcripts available for appointment creation"
        
        transcripts_data = []
        for transcript in available_transcripts:
            transcripts_data.append({
                "id": transcript.id,
                "session_id": transcript.session_id,
                "message_by": transcript.message_by,
                "message_preview": transcript.message[:100] + "..." if len(transcript.message) > 100 else transcript.message,
                "summarized": transcript.summarized[:100] + "..." if transcript.summarized and len(transcript.summarized) > 100 else transcript.summarized,
                "key_topics": transcript.key_topics,
                "created_at": transcript.created_at.isoformat() if transcript.created_at else None
            })
        
        return json.dumps(transcripts_data, indent=2, default=str)
        
    except Exception as e:
        return f"Error retrieving transcripts: {str(e)}"
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
            return f"No transcripts found for session {session_id}"
        
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
        
        return json.dumps(session_summary, indent=2, default=str)
        
    except Exception as e:
        return f"Error retrieving transcript summary: {str(e)}"
    finally:
        db.close()

# Bind tools to LLM
tools = [create_appointment_from_transcript, list_available_transcripts, get_transcript_summary]
llm_with_tools = llm.bind_tools(tools)

# Chain for processing appointment requests
appointment_chain = template | llm_with_tools | StrOutputParser()