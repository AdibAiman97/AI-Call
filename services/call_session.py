from sqlalchemy.orm import Session
from database.models.call_session import CallSession
from fastapi import HTTPException
from database.schemas import CallSessionBase, CallSessionUpdate
from typing import List, Optional


class CallSessionService:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self) -> List[CallSession]:
        return self.db.query(CallSession).all()

    def get_by_id(self, call_session_id: int) -> Optional[CallSession]:
        return (
            self.db.query(CallSession).filter(CallSession.id == call_session_id).first()
        )

    def get_by_customer_id(self, customer_id: str) -> List[CallSession]:
        return (
            self.db.query(CallSession).filter(CallSession.cust_id == customer_id).all()
        )

    def create(self, call_session_data: CallSessionBase) -> CallSession:
        call_session = CallSession(
            cust_id=call_session_data.cust_id,
            # start_time=call_session_data.start_time,
            # end_time=call_session_data.end_time,
            # duration_secs=call_session_data.duration_secs,
            # positive=call_session_data.positive,
            # neutral=call_session_data.neutral,
            # negative=call_session_data.negative,
            # key_words=call_session_data.key_words,
            # summarized_content=call_session_data.summarized_content,
            # customer_suggestions=call_session_data.customer_suggestions,
            # admin_suggestions=call_session_data.admin_suggestions,
        )
        self.db.add(call_session)
        self.db.commit()
        self.db.refresh(call_session)
        return call_session

    def update(
        self, call_session_id: int, call_session_data: CallSessionUpdate
    ) -> Optional[CallSession]:
        call_session = self.get_by_id(call_session_id)
        if not call_session:
            return None

        update_data = call_session_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(call_session, field, value)

        self.db.commit()
        self.db.refresh(call_session)
        return call_session

    def delete(self, call_session_id: int) -> bool:
        call_session = self.get_by_id(call_session_id)
        if not call_session:
            return False

        self.db.delete(call_session)
        self.db.commit()
        return True
