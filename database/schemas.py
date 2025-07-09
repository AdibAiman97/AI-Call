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


class TranscriptCreate(BaseModel):
    id: int
    session_id: int
    message: str
    message_by: str
    summarized: str
    key_topics: str


class TranscriptUpdate(BaseModel):
    id: Optional[int] = None
    session_id: Optional[int] = None
    message: Optional[str] = None
    message_by: Optional[str] = None
    summarized: Optional[str] = None
    key_topics: Optional[str] = None


class TranscriptResponse(BaseModel):
    id: int
    session_id: int
    message: str
    message_by: str
    summarized: str
    key_topics: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CallSessionBase(BaseModel):
    cust_id: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_secs: Optional[int] = None
    positive: Optional[int] = None
    neutral: Optional[int] = None
    negative: Optional[int] = None
    key_words: Optional[str] = None
    summarized_content: Optional[str] = None
    customer_suggestions: Optional[str] = None
    admin_suggestions: Optional[str] = None


class CallSessionResponse(BaseModel):
    id: int
    cust_id: str
    start_time: str
    end_time: str
    duration_secs: int
    positive: int
    neutral: int
    negative: int
    key_words: str
    summarized_content: str
    customer_suggestions: str
    admin_suggestions: str


class CallSessionUpdate(CallSessionBase):
    cust_id: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_secs: Optional[int] = None
    positive: Optional[int] = None
    neutral: Optional[int] = None
    negative: Optional[int] = None
    key_words: Optional[str] = None
    summarized_content: Optional[str] = None
    customer_suggestions: Optional[str] = None
    admin_suggestions: Optional[str] = None


class AppointmentCreate(BaseModel):
    call_session_id: int
    title: str
    start_time: datetime
    end_time: datetime


class AppointmentUpdate(BaseModel):
    call_session_id: Optional[int] = None
    title: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class AppointmentResponse(BaseModel):
    id: int
    call_session_id: int
    title: str
    start_time: datetime
    end_time: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PropertyCreate(BaseModel):
    property_name: str
    property_type: str
    price: float
    location: str
    bedrooms: int
    bathrooms: int
    size: float
    availability: bool
    listing_period: int


class PropertyUpdate(BaseModel):
    property_name: Optional[str] = None
    property_type: Optional[str] = None
    price: Optional[float] = None
    location: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    size: Optional[float] = None
    availability: Optional[bool] = None
    listing_period: Optional[int] = None


class PropertyResponse(BaseModel):
    property_id: int
    property_name: str
    property_type: str
    price: float
    location: str
    bedrooms: int
    bathrooms: int
    size: float
    availability: bool
    listing_period: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
