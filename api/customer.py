from sqlalchemy import create_engine, Column, Integer, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# Database configuration
DB_USERNAME = "postgres"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "customer_db"
DATABASE_URL = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Database setup
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database Models
class Customer(Base):
    __tablename__ = "customers"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    phone_number = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    first_name = Column(String, nullable=False)
    last_name = Column(String, nullable=False)
    budget = Column(Integer, nullable=False)
    purchase_purpose = Column(String, nullable=False)
    preferred_location = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

# Create tables
Base.metadata.create_all(bind=engine)

# FastAPI Router
router = APIRouter(prefix="/customers", tags=["customers"])

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

# CRUD operations
@router.post("/", response_model=CustomerResponse)
def create_customer(customer: CustomerCreate, db: Session = Depends(get_db)):
    # Check if customer with phone number or email already exists
    existing_customer = db.query(Customer).filter(
        (Customer.phone_number == customer.phone_number) | 
        (Customer.email == customer.email)
    ).first()
    
    if existing_customer:
        raise HTTPException(
            status_code=400, 
            detail="Customer with this phone number or email already exists"
        )
    
    try:
        db_customer = Customer(**customer.dict())
        db.add(db_customer)
        db.commit()
        db.refresh(db_customer)
        return db_customer
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error creating customer: {str(e)}")

@router.get("/", response_model=List[CustomerResponse])
def get_all_customers(db: Session = Depends(get_db)):
    customers = db.query(Customer).all()
    return customers

@router.get("/phone/{phone_number}", response_model=CustomerResponse)
def get_customer_by_phone(phone_number: str, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.phone_number == phone_number).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@router.get("/{customer_id}", response_model=CustomerResponse)
def get_customer_by_id(customer_id: int, db: Session = Depends(get_db)):
    customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer

@router.put("/{customer_id}", response_model=CustomerResponse)
def update_customer(customer_id: int, customer_update: CustomerUpdate, db: Session = Depends(get_db)):
    db_customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if db_customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # Check for unique constraints if updating phone or email
    update_data = customer_update.dict(exclude_unset=True)
    if "phone_number" in update_data or "email" in update_data:
        existing_customer = db.query(Customer).filter(
            Customer.id != customer_id,
            (Customer.phone_number == update_data.get("phone_number", "")) | 
            (Customer.email == update_data.get("email", ""))
        ).first()
        
        if existing_customer:
            raise HTTPException(
                status_code=400, 
                detail="Another customer with this phone number or email already exists"
            )
    
    try:
        for key, value in update_data.items():
            setattr(db_customer, key, value)
        
        db.commit()
        db.refresh(db_customer)
        return db_customer
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error updating customer: {str(e)}")

@router.delete("/{customer_id}")
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    db_customer = db.query(Customer).filter(Customer.id == customer_id).first()
    if db_customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    try:
        db.delete(db_customer)
        db.commit()
        return {"message": "Customer deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error deleting customer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(router, host="0.0.0.0", port=8000)










