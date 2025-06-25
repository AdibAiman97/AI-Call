from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from stt import router as stt
from api.call_session import router as call_session_router
from api.customer_router import router as customer

from database.connection import engine, Base

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# app.include_router(stt)
app.include_router(customer)
app.include_router(call_session_router)
