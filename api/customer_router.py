from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from database.connection import get_db
from services.customer_crud import CustomerCRUD
from database.schemas import CustomerCreate, CustomerResponse

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


@router.put("/phone/{phone_number}", response_model=CustomerResponse)
def update_customer_by_phone(
    phone_number: str, customer_update: CustomerCreate, db: Session = Depends(get_db)
):
    update_data = customer_update.dict(exclude_unset=True)
    return CustomerCRUD.update_customer_by_phone(db, phone_number, update_data)


@router.delete("/phone/{phone_number}")
def delete_customer_by_phone(phone_number: str, db: Session = Depends(get_db)):
    return CustomerCRUD.delete_customer_by_phone(db, phone_number)
