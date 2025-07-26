from sqlalchemy import Column, Integer, String, Double, DateTime, func
from database.connection import Base

class CallSession(Base):
    __tablename__ = "call_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    cust_id = Column(String, nullable=False)
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_secs = Column(Integer, nullable=True)
    positive = Column(Integer, nullable=True)
    neutral = Column(Integer, nullable=True)
    negative = Column(Integer, nullable=True)
    key_words = Column(String, nullable=True)
    summarized_content = Column(String, nullable=True)
    customer_suggestions = Column(String, nullable=True)
    admin_suggestions = Column(String, nullable=True)
    pending_customer_data = Column(String, nullable=True)  # Store customer details during call for processing after call ends