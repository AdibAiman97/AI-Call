from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Float,
    Boolean,
    func,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from database.connection import Base


class Property(Base):
    __tablename__ = "properties"

    property_id = Column(Integer, primary_key=True, index=True)
    property_name = Column(String, nullable=False)
    property_type = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    location = Column(String, nullable=False)
    bedrooms = Column(Integer, nullable=False)
    bathrooms = Column(Integer, nullable=False)
    size = Column(Float, nullable=False)
    availability = Column(Boolean, default=True)
    listing_period = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
