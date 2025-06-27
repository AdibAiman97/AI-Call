from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from database.connection import get_db
from database.schemas import TranscriptCreate, TranscriptUpdate, TranscriptResponse
from services.transcript_crud import (
    create_session_message,
    get_session_messages,
    update_session_summary,
    update_session_key_topics,
)

router = APIRouter(prefix="/transcripts", tags=["transcripts"])


@router.post("/{session_id}/messages/")
def create_message(
    session_id: int,
    message: str,
    message_by: str,
    db: Session = Depends(get_db),
    summarized: str = None,
    key_topics: str = None,
):
    """Creates a new message within a call session."""
    try:
        db_message = create_session_message(
            db,
            session_id=session_id,
            message=message,
            message_by=message_by,
            summarized=summarized,
            key_topics=key_topics,
        )
        return db_message
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{session_id}/messages/")
def read_messages(
    session_id: int, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)
):
    """Retrieves messages for a specific session."""
    messages = get_session_messages(db, session_id=session_id, skip=skip, limit=limit)
    return messages


@router.put("/{session_id}/summary/")
def update_summary(session_id: int, summarized: str, db: Session = Depends(get_db)):
    """Updates the summarized field for all messages in a session."""
    try:
        update_session_summary(db, session_id=session_id, summarized=summarized)
        return {"message": f"Session {session_id} summary updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/{session_id}/key_topics/")
def update_key_topics(session_id: int, key_topics: str, db: Session = Depends(get_db)):
    """Updates the key_topics field for all messages in a session."""
    try:
        update_session_key_topics(db, session_id=session_id, key_topics=key_topics)
        return {"message": f"Session {session_id} key topics updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
