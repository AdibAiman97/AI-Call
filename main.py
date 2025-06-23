from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from stt import router as stt
from api.call_session import router as call_session_router
from database.connection import engine
from database.models.call_session import CallSession

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CallSession.metadata.create_all(bind=engine)

app.include_router(stt)
app.include_router(call_session_router)