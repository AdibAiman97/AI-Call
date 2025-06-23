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


class CustomerResponse(BaseModel):
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