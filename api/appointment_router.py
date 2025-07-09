from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from database.connection import get_db
from services.appointment_crud import AppointmentCRUD
from database.schemas import AppointmentCreate, AppointmentUpdate, AppointmentResponse

router = APIRouter(prefix="/appointments", tags=["appointments"])


@router.post("/", response_model=AppointmentResponse)
def create_appointment(appointment: AppointmentCreate, db: Session = Depends(get_db)):
    return AppointmentCRUD.create_appointment(db, appointment)


@router.get("/", response_model=List[AppointmentResponse])
def get_all_appointments(db: Session = Depends(get_db)):
    return AppointmentCRUD.get_all_appointments(db)


@router.get("/call-session/{call_session_id}", response_model=AppointmentResponse)
def get_appointment_by_call_session(
    call_session_id: str, db: Session = Depends(get_db)
):
    return AppointmentCRUD.get_appointment_by_call_session(db, call_session_id)


@router.get("/date-range", response_model=List[AppointmentResponse])
def get_appointments_by_date_range(
    start_date: datetime = Query(
        ..., description="Start date for filtering appointments"
    ),
    end_date: datetime = Query(..., description="End date for filtering appointments"),
    db: Session = Depends(get_db),
):
    return AppointmentCRUD.get_appointments_by_date_range(db, start_date, end_date)


@router.get("/{appointment_id}", response_model=AppointmentResponse)
def get_appointment(appointment_id: int, db: Session = Depends(get_db)):
    db_appointment = AppointmentCRUD.get_appointment_by_id(db, appointment_id)
    if db_appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return db_appointment


@router.put("/{appointment_id}", response_model=AppointmentResponse)
def update_appointment(
    appointment_id: int, appointment: AppointmentUpdate, db: Session = Depends(get_db)
):
    db_appointment = AppointmentCRUD.update_appointment(db, appointment_id, appointment)
    if db_appointment is None:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return db_appointment


@router.delete("/{appointment_id}", status_code=204)
def delete_appointment(appointment_id: int, db: Session = Depends(get_db)):
    if not AppointmentCRUD.delete_appointment(db, appointment_id):
        raise HTTPException(status_code=404, detail="Appointment not found")
    return {"message": "Appointment deleted successfully"}
