from pathlib import Path

from IPython.display import Audio, Markdown, display
from google.genai.types import (
    AudioTranscriptionConfig,
    AutomaticActivityDetection,
    Content,
    EndSensitivity,
    GoogleSearch,
    LiveConnectConfig,
    Part,
    PrebuiltVoiceConfig,
    ProactivityConfig,
    RealtimeInputConfig,
    SpeechConfig,
    StartSensitivity,
    Tool,
    ToolCodeExecution,
    VoiceConfig,
)
import numpy as np
from config import GCP_PROJECT_ID, GCP_LOCATION
from google import genai

client = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_LOCATION)

config = LiveConnectConfig(
    response_modalities=["AUDIO"],
    enable_affective_dialog=True,
)

async with client.aio.live.connect(
    model=MODEL_ID,
    config=config,
) as session:
    text_input = "Hello? Gemini are you there? It's really a good day!"
    display(Markdown(f"**Input:** {text_input}"))

    await session.send_client_content(
        turns=Content(role="user", parts=[Part(text=text_input)])
    )

    audio_data = []
    async for message in session.receive():
        if (
            message.server_content.model_turn
            and message.server_content.model_turn.parts
        ):
            for part in message.server_content.model_turn.parts:
                if part.inline_data:
                    audio_data.append(
                        np.frombuffer(part.inline_data.data, dtype=np.int16)
                    )

    if audio_data:
        display(Audio(np.concatenate(audio_data), rate=24000, autoplay=True))