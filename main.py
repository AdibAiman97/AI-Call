from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.customer_router import router as customer_router
from api.call_session_router import router as call_session_router
from api.transcript_router import router as transcript_router
from api.appointment_router import router as appointment_router
from api.property_router import router as property_router
from database.connection import engine, Base


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# app.include_router(stt)
app.include_router(customer_router)
app.include_router(call_session_router)
app.include_router(transcript_router)
app.include_router(appointment_router)
app.include_router(property_router)
