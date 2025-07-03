from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, func
from database.connection import Base


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    message = Column(Text, nullable=False)
    message_by = Column(String(50))
    summarized = Column(Text)
    key_topics = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
