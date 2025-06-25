from sqlalchemy import Column, Integer, String, DateTime, func
from database.connection import Base


class Customer(Base):
    __tablename__ = "customers"

    phone_number = Column(String, primary_key=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    budget = Column(Integer, nullable=False)
    purchase_purpose = Column(String, nullable=False)
    preferred_location = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
