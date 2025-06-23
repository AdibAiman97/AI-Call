from sqlalchemy.orm import Session
from fastapi import HTTPException
from database.models import Customer
from typing import List, Optional

class CustomerCRUD:
    @staticmethod
    def create_customer(db: Session, customer_data: dict) -> Customer:
        # Check if customer with phone number or email already exists
        existing_customer = db.query(Customer).filter(
            (Customer.phone_number == customer_data["phone_number"]) | 
            (Customer.email == customer_data["email"])
        ).first()
        
        if existing_customer:
            raise HTTPException(
                status_code=400, 
                detail="Customer with this phone number or email already exists"
            )
        
        try:
            db_customer = Customer(**customer_data)
            db.add(db_customer)
            db.commit()
            db.refresh(db_customer)
            return db_customer
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=f"Error creating customer: {str(e)}")

    @staticmethod
    def get_all_customers(db: Session) -> List[Customer]:
        return db.query(Customer).all()

    @staticmethod
    def get_customer_by_phone(db: Session, phone_number: str) -> Customer:
        customer = db.query(Customer).filter(Customer.phone_number == phone_number).first()
        if customer is None:
            raise HTTPException(status_code=404, detail="Customer not found")
        return customer

    @staticmethod
    def update_customer_by_phone(db: Session, phone_number: str, update_data: dict) -> Customer:
        db_customer = db.query(Customer).filter(Customer.phone_number == phone_number).first()
        if db_customer is None:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        # Check for unique constraints if updating email
        if "email" in update_data:
            existing_customer = db.query(Customer).filter(
                Customer.phone_number != phone_number,
                Customer.email == update_data.get("email", "")
            ).first()
            
            if existing_customer:
                raise HTTPException(
                    status_code=400, 
                    detail="Another customer with this email already exists"
                )
        
        # Don't allow updating phone_number as it's the primary key
        if "phone_number" in update_data and update_data["phone_number"] != phone_number:
            raise HTTPException(
                status_code=400, 
                detail="Cannot change phone number as it's the primary key"
            )
        
        try:
            for key, value in update_data.items():
                if key != "phone_number":  # Skip phone_number updates
                    setattr(db_customer, key, value)
            
            db.commit()
            db.refresh(db_customer)
            return db_customer
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=f"Error updating customer: {str(e)}")

    @staticmethod
    def delete_customer_by_phone(db: Session, phone_number: str) -> dict:
        db_customer = db.query(Customer).filter(Customer.phone_number == phone_number).first()
        if db_customer is None:
            raise HTTPException(status_code=404, detail="Customer not found")
        
        try:
            db.delete(db_customer)
            db.commit()
            return {"message": "Customer deleted successfully"}
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=400, detail=f"Error deleting customer: {str(e)}") 