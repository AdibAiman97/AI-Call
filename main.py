from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from stt import router as stt
from tts import router as tts

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stt)
app.include_router(tts)