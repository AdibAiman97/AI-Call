from pydantic import BaseModel
from typing import Optional

class CallSessionBase(BaseModel):
    cust_id: str
    start_time: str
    end_time: str
    duration: str
    sentiment: str
    summarized_content: str
    customer_suggestions: str
    admin_suggestions: str

class CallSessionUpdate(CallSessionBase):
    cust_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[str] = None
    sentiment: Optional[str] = None
    summarized_content: Optional[str] = None
    customer_suggestions: Optional[str] = None
    admin_suggestions: Optional[str] = None