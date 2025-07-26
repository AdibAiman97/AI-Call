from sqlalchemy.orm import Session
from database.models.transcript import Transcript
from database.schemas import TranscriptCreate, TranscriptUpdate
from typing import Optional
from database.models.call_session import CallSession

def create_session_message(
    db: Session,
    session_id: int,
    message: str,
    message_by: str,
    summarized: Optional[str] = None,
    key_topics: Optional[str] = None,
):
    """Creates a new message within a call session."""
    db_transcript = Transcript(
        session_id=session_id,
        message=message,
        message_by=message_by,
        summarized=summarized,
        key_topics=key_topics,
    )
    db.add(db_transcript)
    db.commit()
    db.refresh(db_transcript)
    return db_transcript


def get_session_messages(db: Session, session_id: int, skip: int = 0, limit: int = 100):
    """Retrieves messages for a specific session."""
    return (
        db.query(Transcript)
        .filter(Transcript.session_id == session_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_session_summary(db: Session, session_id: int, summarized: str):
    """Updates the summarized field for all messages in a session."""
    db.query(Transcript).filter(Transcript.session_id == session_id).update(
        {"summarized": summarized}
    )
    db.commit()
    return True


def update_session_key_topics(db: Session, session_id: int, key_topics: str):
    """Updates the key_topics field for all messages in a session."""
    db.query(Transcript).filter(Transcript.session_id == session_id).update(
        {"key_topics": key_topics}
    )
    db.commit()
    return True


def get_message_by_id(db: Session, message_id: int):
    return db.query(Transcript).filter(Transcript.id == message_id).first()
