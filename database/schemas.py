from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Pydantic models for request/response
class CustomerCreate(BaseModel):
    phone_number: str
    email: str
    first_name: str
    last_name: str
    budget: int
    purchase_purpose: str
    preferred_location: str

class CustomerUpdate(BaseModel):
    phone_number: Optional[str] = None
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    budget: Optional[int] = None
    purchase_purpose: Optional[str] = None
    preferred_location: Optional[str] = None

class CustomerResponse(BaseModel):
    id: int
    phone_number: str
    email: str
    first_name: str
    last_name: str
    budget: int
    purchase_purpose: str
    preferred_location: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 