from sqlalchemy.orm import Session
from database.models.transcript import Transcript
from database.schemas import TranscriptCreate, TranscriptUpdate
from typing import Optional, List


def create_session_message(
    db: Session,
    session_id: int,
    message: str,
    message_by: str,
    summarized: Optional[str] = None,
    key_topics: Optional[str] = None,
) -> Transcript:
    """Creates a new transcript message for a session."""
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


def get_session_messages(
    db: Session, session_id: int, skip: int = 0, limit: int = 100
) -> List[Transcript]:
    """Retrieves transcript messages for a specific session."""
    return (
        db.query(Transcript)
        .filter(Transcript.session_id == session_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def update_session_summary(db: Session, session_id: int, summarized: str) -> int:
    """Updates the summarized field for all messages in a session."""
    updated_count = (
        db.query(Transcript)
        .filter(Transcript.session_id == session_id)
        .update({"summarized": summarized})
    )
    db.commit()
    return updated_count


def update_session_key_topics(db: Session, session_id: int, key_topics: str) -> int:
    """Updates the key_topics field for all messages in a session."""
    updated_count = (
        db.query(Transcript)
        .filter(Transcript.session_id == session_id)
        .update({"key_topics": key_topics})
    )
    db.commit()
    return updated_count


def create_transcript(db: Session, transcript_data: TranscriptCreate) -> Transcript:
    """Creates a new transcript message."""
    db_transcript = Transcript(
        session_id=transcript_data.session_id,
        message=transcript_data.message,
        message_by=transcript_data.message_by,
        summarized=transcript_data.summarized,
        key_topics=transcript_data.key_topics,
    )
    db.add(db_transcript)
    db.commit()
    db.refresh(db_transcript)
    return db_transcript


def get_transcripts_by_session(
    db: Session, session_id: int, skip: int = 0, limit: int = 100
) -> List[Transcript]:
    """Retrieves transcript messages for a specific session."""
    return (
        db.query(Transcript)
        .filter(Transcript.session_id == session_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_transcript_by_id(db: Session, transcript_id: int) -> Optional[Transcript]:
    """Retrieves a specific transcript by ID."""
    return db.query(Transcript).filter(Transcript.id == transcript_id).first()


def update_transcript(
    db: Session, transcript_id: int, transcript_data: TranscriptUpdate
) -> Optional[Transcript]:
    """Updates a specific transcript."""
    transcript = get_transcript_by_id(db, transcript_id)
    if not transcript:
        return None

    update_data = transcript_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        if hasattr(transcript, field):
            setattr(transcript, field, value)

    db.commit()
    db.refresh(transcript)
    return transcript


def update_transcript_summary(
    db: Session, transcript_id: int, summarized: str
) -> Optional[Transcript]:
    """Updates the summarized field for a specific transcript."""
    transcript = get_transcript_by_id(db, transcript_id)
    if not transcript:
        return None
    
    transcript.summarized = summarized
    db.commit()
    db.refresh(transcript)
    return transcript


def update_transcript_key_topics(
    db: Session, transcript_id: int, key_topics: str
) -> Optional[Transcript]:
    """Updates the key_topics field for a specific transcript."""
    transcript = get_transcript_by_id(db, transcript_id)
    if not transcript:
        return None
    
    transcript.key_topics = key_topics
    db.commit()
    db.refresh(transcript)
    return transcript


def delete_transcript(db: Session, transcript_id: int) -> bool:
    """Deletes a specific transcript."""
    transcript = get_transcript_by_id(db, transcript_id)
    if not transcript:
        return False

    db.delete(transcript)
    db.commit()
    return True


def get_all_transcripts(
    db: Session, skip: int = 0, limit: int = 100
) -> List[Transcript]:
    """Retrieves all transcripts with pagination."""
    return db.query(Transcript).offset(skip).limit(limit).all()
