from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
import re
from database.connection import get_db
from database.schemas import CallSessionBase, CallSessionUpdate, CallSessionResponse, CallSessionWithStatus
from services.call_session import CallSessionService

router = APIRouter(prefix="/call_session", tags=["call_session"])


def _fix_timezone_format(dt):
    """Fix timezone format from +08 to +08:00 for FastAPI validation."""
    if dt is None:
        return None
    
    if isinstance(dt, datetime):
        # Convert datetime to string and fix timezone format
        dt_str = dt.isoformat()
        # Fix malformed timezone (e.g., +08 -> +08:00)
        dt_str = re.sub(r'([+-]\d{2})$', r'\1:00', dt_str)
        return dt_str
    
    return dt


@router.get("/", response_model=List[CallSessionBase])
def get_call_sessions(customer_id: str = None, db: Session = Depends(get_db)):
    service = CallSessionService(db)
    if customer_id:
        return service.get_by_customer_id(customer_id)
    return service.get_all()


@router.get("/{call_session_id}", response_model=CallSessionWithStatus)
def get_call_session(call_session_id: int = 1, db: Session = Depends(get_db)):
    service = CallSessionService(db)
    call_session = service.get_by_id(call_session_id)
    if not call_session:
        raise HTTPException(status_code=404, detail="Call session not found")
    
    # Calculate processing status and progress
    processing_status, processing_progress = _calculate_processing_status(call_session)
    
    # Convert to dict and add status fields
    session_dict = {
        "id": call_session.id,
        "cust_id": call_session.cust_id,
        "start_time": call_session.start_time,
        "end_time": _fix_timezone_format(call_session.end_time),
        "duration_secs": call_session.duration_secs,
        "positive": call_session.positive,
        "neutral": call_session.neutral,
        "negative": call_session.negative,
        "key_words": call_session.key_words,
        "summarized_content": call_session.summarized_content,
        "customer_suggestions": call_session.customer_suggestions,
        "admin_suggestions": call_session.admin_suggestions,
        "processing_status": processing_status,
        "processing_progress": processing_progress
    }
    
    return session_dict


def _calculate_processing_status(call_session):
    """Calculate processing status and progress based on completed fields."""
    # Define the key fields that indicate processing completion
    fields_to_check = [
        call_session.summarized_content,
        call_session.customer_suggestions,
        call_session.admin_suggestions,
        call_session.end_time  # This gets set when processing starts
    ]
    
    # Count completed fields (non-null and non-empty strings)
    completed_fields = 0
    for field in fields_to_check[:3]:  # Don't count end_time in progress
        if field and field.strip():
            completed_fields += 1
    
    # Calculate progress percentage
    total_fields = 3  # summary, customer suggestions, admin suggestions
    progress = int((completed_fields / total_fields) * 100)
    
    # Determine status
    if completed_fields == total_fields:
        status = "completed"
        progress = 100
    elif completed_fields > 0 or call_session.end_time:  # Some processing done or call ended
        status = "in_progress"
    else:
        status = "not_started"
        progress = 0
    
    return status, progress


@router.post("/", response_model=CallSessionResponse)
def create_call_session(
    call_session_data: CallSessionBase, db: Session = Depends(get_db)
):
    service = CallSessionService(db)
    return service.create(call_session_data)


@router.put("/{call_session_id}", response_model=CallSessionBase)
def update_call_session(
    call_session_id: int,
    call_session_data: CallSessionUpdate,
    db: Session = Depends(get_db),
):
    service = CallSessionService(db)
    call_session = service.update(call_session_id, call_session_data)
    if not call_session:
        raise HTTPException(status_code=404, detail="Call session not found")
    return call_session


@router.delete("/{call_session_id}")
def delete_call_session(call_session_id: int, db: Session = Depends(get_db)):
    service = CallSessionService(db)
    if not service.delete(call_session_id):
        raise HTTPException(status_code=404, detail="Call session not found")
    return {"message": "Call session deleted successfully"}
