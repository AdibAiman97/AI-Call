from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List
from database.connection import get_db
from database.schemas import CallSessionBase, CallSessionUpdate, CallSessionResponse
from services.call_session import CallSessionService

router = APIRouter(prefix="/call_session", tags=["call_session"])


@router.get("/", response_model=List[CallSessionBase])
def get_call_sessions(customer_id: str = None, db: Session = Depends(get_db)):
    service = CallSessionService(db)
    if customer_id:
        return service.get_by_customer_id(customer_id)
    return service.get_all()


@router.get("/{call_session_id}", response_model=CallSessionBase)
def get_call_session(call_session_id: int, db: Session = Depends(get_db)):
    service = CallSessionService(db)
    call_session = service.get_by_id(call_session_id)
    if not call_session:
        raise HTTPException(status_code=404, detail="Call session not found")
    return call_session


@router.post("/", response_model=CallSessionResponse)
def create_call_session(
    call_session_data: CallSessionBase, db: Session = Depends(get_db)
):
    service = CallSessionService(db)
    return service.create(call_session_data)


@router.put("/{call_session_id}", response_model=CallSessionBase)
def update_call_session(
    call_session_id: int,
    call_session_data: CallSessionUpdate,
    db: Session = Depends(get_db),
):
    service = CallSessionService(db)
    call_session = service.update(call_session_id, call_session_data)
    if not call_session:
        raise HTTPException(status_code=404, detail="Call session not found")
    return call_session


@router.delete("/{call_session_id}")
def delete_call_session(call_session_id: int, db: Session = Depends(get_db)):
    service = CallSessionService(db)
    if not service.delete(call_session_id):
        raise HTTPException(status_code=404, detail="Call session not found")
    return {"message": "Call session deleted successfully"}
