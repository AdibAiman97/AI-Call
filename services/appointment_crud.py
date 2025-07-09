from sqlalchemy.orm import Session
from typing import List, Optional
from database.models.appointment import Appointment
from database.schemas import AppointmentCreate, AppointmentUpdate
from datetime import datetime
from fastapi import HTTPException


class AppointmentCRUD:
    @staticmethod
    def create_appointment(db: Session, appointment_data: dict) -> Appointment:
        existing_appointment = (
            db.query(Appointment)
            .filter(Appointment.call_session_id == appointment_data["call_session_id"])
            .first()
        )

        if existing_appointment:
            raise HTTPException(
                status_code=400,
                detail="Appointment with this call session ID already exists",
            )

        if appointment_data["end_time"] <= appointment_data["start_time"]:
            raise HTTPException(
                status_code=400, detail="End time must be after start time"
            )

        try:
            db_appointment = Appointment(**appointment_data)
            db.add(db_appointment)
            db.commit()
            db.refresh(db_appointment)
            return db_appointment
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Error creating appointment: {str(e)}"
            )

    @staticmethod
    def get_all_appointments(db: Session) -> List[Appointment]:
        return db.query(Appointment).all()

    @staticmethod
    def get_appointment_by_call_session(
        db: Session, call_session_id: str
    ) -> Appointment:
        appointment = (
            db.query(Appointment)
            .filter(Appointment.call_session_id == call_session_id)
            .first()
        )
        if appointment is None:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return appointment

    @staticmethod
    def get_appointment_by_id(db: Session, appointment_id: int) -> Appointment:
        appointment = (
            db.query(Appointment).filter(Appointment.id == appointment_id).first()
        )
        if appointment is None:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return appointment

    @staticmethod
    def get_appointments_by_date_range(
        db: Session, start_date: datetime, end_date: datetime
    ) -> List[Appointment]:
        return (
            db.query(Appointment)
            .filter(
                Appointment.start_time >= start_date, Appointment.end_time <= end_date
            )
            .all()
        )

    @staticmethod
    def update_appointment(
        db: Session, appointment_id: int, update_data: dict
    ) -> Appointment:
        db_appointment = (
            db.query(Appointment).filter(Appointment.id == appointment_id).first()
        )
        if db_appointment is None:
            raise HTTPException(status_code=404, detail="Appointment not found")

        if "call_session_id" in update_data:
            existing_appointment = (
                db.query(Appointment)
                .filter(
                    Appointment.id != appointment_id,
                    Appointment.call_session_id == update_data["call_session_id"],
                )
                .first()
            )

            if existing_appointment:
                raise HTTPException(
                    status_code=400,
                    detail="Another appointment with this call session ID already exists",
                )

        start_time = update_data.get("start_time", db_appointment.start_time)
        end_time = update_data.get("end_time", db_appointment.end_time)

        if end_time <= start_time:
            raise HTTPException(
                status_code=400, detail="End time must be after start time"
            )

        try:
            for key, value in update_data.items():
                setattr(db_appointment, key, value)

            db.commit()
            db.refresh(db_appointment)
            return db_appointment
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Error updating appointment: {str(e)}"
            )

    @staticmethod
    def delete_appointment(db: Session, appointment_id: int) -> dict:
        db_appointment = (
            db.query(Appointment).filter(Appointment.id == appointment_id).first()
        )
        if db_appointment is None:
            raise HTTPException(status_code=404, detail="Appointment not found")

        try:
            db.delete(db_appointment)
            db.commit()
            return {"message": "Appointment deleted successfully"}
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=400, detail=f"Error deleting appointment: {str(e)}"
            )
