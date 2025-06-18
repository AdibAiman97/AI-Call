from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from database.models import get_db
from database.crud import CustomerCRUD
from database.schemas import CustomerCreate, CustomerUpdate, CustomerResponse

# FastAPI Router
router = APIRouter(prefix="/customers", tags=["customers"])

# API endpoints using the separated CRUD operations
@router.post("/", response_model=CustomerResponse)
def create_customer(customer: CustomerCreate, db: Session = Depends(get_db)):
    return CustomerCRUD.create_customer(db, customer.dict())

@router.get("/", response_model=List[CustomerResponse])
def get_all_customers(db: Session = Depends(get_db)):
    return CustomerCRUD.get_all_customers(db)

@router.get("/phone/{phone_number}", response_model=CustomerResponse)
def get_customer_by_phone(phone_number: str, db: Session = Depends(get_db)):
    return CustomerCRUD.get_customer_by_phone(db, phone_number)

@router.get("/{customer_id}", response_model=CustomerResponse)
def get_customer_by_id(customer_id: int, db: Session = Depends(get_db)):
    return CustomerCRUD.get_customer_by_id(db, customer_id)

@router.put("/{customer_id}", response_model=CustomerResponse)
def update_customer(customer_id: int, customer_update: CustomerUpdate, db: Session = Depends(get_db)):
    update_data = customer_update.dict(exclude_unset=True)
    return CustomerCRUD.update_customer(db, customer_id, update_data)

@router.delete("/{customer_id}")
def delete_customer(customer_id: int, db: Session = Depends(get_db)):
    return CustomerCRUD.delete_customer(db, customer_id) 