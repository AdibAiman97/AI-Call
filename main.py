from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.customer_router import router as customer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(customer)