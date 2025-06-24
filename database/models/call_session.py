from sqlalchemy import Column, Integer, String, Double
from database.connection import Base

class CallSession(Base):
    __tablename__ = "call_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    cust_id = Column(String, nullable=False)
    start_time = Column(String, nullable=False)
    end_time = Column(String, nullable=False)
    duration = Column(String, nullable=False)
    positive = Column(Integer, nullable=False)
    neutral = Column(Integer, nullable=False)
    negative = Column(Integer, nullable=False)
    summarized_content = Column(String, nullable=False)
    customer_suggestions = Column(String, nullable=False)
    admin_suggestions = Column(String, nullable=False)