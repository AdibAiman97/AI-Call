import uvicorn
import json
import asyncio
import base64
import re
import threading
from collections import deque
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ai_services import initialize_rag_system, get_rag_system
from google.cloud import texttospeech
from google.cloud import speech


app = FastAPI(title="AI Call Center RAG Engine - Integrated")

# TTS Client
client_tts = texttospeech.TextToSpeechClient()
SENTENCE_ENDINGS = re.compile(r"([.?!])\s+")

# Speech Client for STT
speech_client = speech.SpeechClient()


class AudioBuffer:
    """A thread-safe buffer to hold audio chunks."""

    def __init__(self):
        self.buffer = deque()
        self.finished = False
        self.lock = threading.Lock()

    def add_chunk(self, chunk: bytes):
        with self.lock:
            if not self.finished:
                self.buffer.append(chunk)

    def get_chunk(self) -> Optional[bytes]:
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            return None

    def finish(self):
        with self.lock:
            self.finished = True

    def is_finished(self) -> bool:
        with self.lock:
            return self.finished and len(self.buffer) == 0


# --- Main Application Logic ---
@app.on_event("startup")
async def startup_event():
    """Initializes the RAG system at application startup."""
    await initialize_rag_system()
    print("AI Call Center Backend is Running")
    print("RAG System Initialized.")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return "<h1>AI Call Center Backend is Running (Integrated Model)</h1>"


@app.websocket("/ws/conversation")
async def conversation_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for the entire conversation flow."""
    await websocket.accept()
    print("WebSocket connection accepted.")

    conversation_history = []

    # This task will manage the integrated STT, RAG, and TTS pipeline
    processing_task = asyncio.create_task(
        connection_manager(websocket, conversation_history)
    )
    await processing_task


async def connection_manager(websocket: WebSocket, conversation_history: List[dict]):
    """
    Manages the WebSocket connection and routes messages for STT, Text, RAG, and TTS.
    """
    try:
        while True:
            # We wait for the first message to decide if it's audio or text
            message = await websocket.receive()

            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                if msg_type == "text_message":
                    typed_text = data["data"]
                    print(f"Received text message: '{typed_text}'")
                    conversation_history.append({"role": "user", "text": typed_text})
                    # Trigger the LLM/TTS pipeline directly
                    asyncio.create_task(
                        llm_tts_pipeline(websocket, typed_text, conversation_history)
                    )

            elif "bytes" in message:
                # This is the start of a voice utterance.
                print("Audio stream started.")
                # It will handle receiving subsequent audio chunks internally by use STT processor.
                await stt_processor_hybrid(
                    websocket, message["bytes"], conversation_history
                )
                print("Audio stream processing finished.")

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred in the connection manager: {e}")


async def stt_processor_hybrid(
    websocket: WebSocket, first_chunk: bytes, conversation_history: List[dict]
):
    """
    Integrates the hybrid async/threading STT model.
    Receives audio, processes it in a separate thread, and triggers the RAG pipeline.
    """
    audio_buffer = AudioBuffer()
    audio_buffer.add_chunk(first_chunk)  # Add the first chunk

    stt_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-MY",
        model="telephony_short",
        enable_automatic_punctuation=True,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=stt_config, interim_results=True
    )

    # BLOCKING FUNCTION: This will run in a separate thread via run_in_executor
    def audio_generator():
        """A blocking generator that yields audio chunks from the thread-safe buffer."""
        while not audio_buffer.is_finished():
            chunk = audio_buffer.get_chunk()
            if chunk:
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            else:
                threading.Event().wait(0.1)

    # ASYNC COROUTINE: This runs on the main event loop
    async def audio_receiver():
        """Receives audio chunks from the WebSocket and puts them in the buffer."""
        try:
            while True:
                # This timeout is a simple way to detect end-of-speech
                message = await asyncio.wait_for(websocket.receive(), timeout=3.0)
                if "bytes" in message:
                    audio_buffer.add_chunk(message["bytes"])
                elif "text" in message:
                    # Handle explicit audio end signal from client
                    data = json.loads(message["text"])
                    if data.get("type") == "audio_end":
                        print("🛑 Received audio_end signal.")
                        break
        except asyncio.TimeoutError:
            print("Audio stream timed out. Finalizing utterance.")
        except WebSocketDisconnect:
            print("Client disconnected during audio stream.")
        finally:
            audio_buffer.finish()  # Signal that no more audio will be added

    # Get the current asyncio event loop to bridge the thread back
    loop = asyncio.get_event_loop()
    final_transcript = ""

    # BLOCKING FUNCTION: This will be run in the background thread
    def process_recognition_and_send_messages():
        nonlocal final_transcript
        try:
            responses = speech_client.streaming_recognize(
                config=streaming_config, requests=audio_generator()
            )
            for response in responses:
                if not response.results or not response.results[0].alternatives:
                    continue

                result = response.results[0]
                transcript = result.alternatives[0].transcript

                # Prepare message to send back to client, mimicking original format
                if result.is_final:
                    final_transcript = transcript
                    # Send final transcript update
                    response_data = {
                        "type": "stt_transcript",
                        "data": {"text": transcript},
                    }
                else:
                    # Send interim transcript update
                    response_data = {
                        "type": "stt_transcript",
                        "data": {"text": transcript},
                    }
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json(response_data), loop
                )
        except Exception as e:
            print(f"Error in speech recognition thread: {e}")

    stt_thread = loop.run_in_executor(None, process_recognition_and_send_messages)

    await audio_receiver()

    await stt_thread

    print(f"✅ STT processor finished. Final Transcript: '{final_transcript}'")

    if final_transcript:
        # Get the RAG system instance to access the cleanup method
        rag_system = get_rag_system()

        # Use the LLM to clean the raw transcript
        cleaned_transcript = await rag_system.cleanup_transcript_llm(final_transcript)
        print(f"✨ LLM Cleaned Transcript: '{cleaned_transcript}'")

        # Only proceed if the cleaned transcript is not empty
        if cleaned_transcript:
            await websocket.send_json(
                {"type": "stt_final", "data": {"text": cleaned_transcript}}
            )
            conversation_history.append({"role": "user", "text": cleaned_transcript})
            await llm_tts_pipeline(websocket, cleaned_transcript, conversation_history)


async def llm_tts_pipeline(websocket: WebSocket, user_query: str, history: List[dict]):
    """
    Handles the RAG, LLM, and TTS pipeline.
    """
    print(f"Triggering RAG pipeline for: '{user_query}'")
    rag_system = get_rag_system()
    relevant_docs = await rag_system.retrieve_relevant_docs(user_query)
    context = "\n\n".join([doc["content"] for doc in relevant_docs])

    llm_text_stream = rag_system.generate_response_stream(
        prompt=user_query, chat_history=history, context=context
    )

    sentence_buffer = ""
    full_llm_response = ""
    async for chunk in llm_text_stream:
        full_llm_response += chunk
        sentence_buffer += chunk
        # Send the LLM text chunk to the frontend immediately for display
        await websocket.send_json({"type": "llm_chunk", "data": chunk})

        # Split into sentences for TTS
        parts = SENTENCE_ENDINGS.split(sentence_buffer)
        i = 0
        while i < len(parts) - 1:
            sentence = (parts[i] + parts[i + 1]).strip()
            if sentence:
                await text_to_speech_and_send(websocket, sentence)
            i += 2
        sentence_buffer = parts[-1] if parts else ""

    if sentence_buffer.strip():
        await text_to_speech_and_send(websocket, sentence_buffer.strip())

    history.append({"role": "ai", "text": full_llm_response})
    await websocket.send_json({"type": "pipeline_end"})


async def text_to_speech_and_send(websocket: WebSocket, text: str):
    """
    Synthesizes text to speech and sends it over the WebSocket. (Unchanged from original file)
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Standard-C"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    try:
        # Running the blocking TTS call in a thread to be safe
        tts_response = await asyncio.to_thread(
            client_tts.synthesize_speech,
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        audio_chunk = base64.b64encode(tts_response.audio_content).decode("utf-8")
        await websocket.send_json({"type": "tts_chunk", "data": audio_chunk})
    except Exception as e:
        print(f"TTS Error: {e}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
