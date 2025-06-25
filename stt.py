from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech_v1p1beta1 as speech
import asyncio
import threading
from collections import deque
from typing import Optional
import json

router = APIRouter(prefix="/stt")


class AudioBuffer:
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


class TranscriptManager:
    def __init__(self):
        self.final_transcript = ""  # Accumulated final results
        self.current_interim = ""  # Current interim result
        self.lock = threading.Lock()

    def update_interim(self, interim_text: str):
        """Update the current interim result"""
        with self.lock:
            self.current_interim = interim_text

    def add_final(self, final_text: str):
        """Add final text to the accumulated transcript"""
        with self.lock:
            self.final_transcript += final_text + " "
            self.current_interim = ""  # Clear interim when we get final

    def get_display_text(self):
        """Get the complete text for display (final + interim)"""
        with self.lock:
            return (self.final_transcript + self.current_interim).strip()

    def get_final_only(self):
        """Get only the final transcript"""
        with self.lock:
            return self.final_transcript.strip()


@router.websocket("")
async def stt_route(ws: WebSocket):
    await ws.accept()
    await ws.send_text("✅ WebSocket connected to Google STT")

    # Create Google Speech client.
    speech_client = speech.SpeechClient()
    audio_buffer = AudioBuffer()
    transcript_manager = TranscriptManager()

    # Configuration setup.
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    # BLOCKING FUNCTION: This will run in a separate thread
    def audio_generator():
        """BLOCKING: Synchronous generator for Google Speech API"""
        while not audio_buffer.is_finished():
            chunk = audio_buffer.get_chunk()
            if chunk is not None:
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            else:
                # BLOCKING, stops thread for 100ms.
                threading.Event().wait(0.1)

    # =============================================================

    # ASYNC: Handles WebSocket Communications
    async def audio_receiver():
        """Receive audio chunks from WebSocket"""
        try:
            while True:
                # async/await without blocking
                chunk = await ws.receive_bytes()

                # add to thread-safe buffer (quick op, non-blocking)
                audio_buffer.add_chunk(chunk)
        except WebSocketDisconnect:
            print("WebSocket disconnected by client")
        except Exception as e:
            print(f"Error receiving audio: {e}")
        finally:
            # mark buffer as finished (thread-safe operation)
            audio_buffer.finish()

    # ASYNC: Manages the blocking speech processing
    async def speech_processor():
        """Process speech recognition in background thread"""

        # BLOCKING FUNCTION: This will run in a separate thread
        def process_recognition():
            try:
                responses = speech_client.streaming_recognize(
                    config=streaming_config, requests=audio_generator()
                )

                for response in responses:
                    if response.results:
                        top_result = response.results[0]
                        alternatives = top_result.alternatives[:3]
                        print(f"alternatives: {alternatives}")
                        print(f"chosed: {alternatives[0].transcript}")

                        if top_result.alternatives:
                            transcript = top_result.alternatives[0].transcript
                            is_final = top_result.is_final

                            if is_final:
                                # Add to final transcript
                                transcript_manager.add_final(transcript)
                                print(f"FINAL: {transcript}")

                                # Send structured response
                                response_data = {
                                    "type": "final",
                                    "text": transcript,
                                    "full_transcript": transcript_manager.get_final_only(),
                                }
                            else:
                                # Update interim text
                                transcript_manager.update_interim(transcript)
                                print(f"INTERIM: {transcript}")

                                # Send structured response
                                response_data = {
                                    "type": "interim",
                                    "text": transcript,
                                    "display_text": transcript_manager.get_display_text(),
                                }

                            # BRIDGE: Send from blocking thread back to async world
                            # This is the magic that connects thread to async

                            # Send to frontend
                            asyncio.run_coroutine_threadsafe(
                                ws.send_text(json.dumps(response_data)), loop
                            )

            except Exception as e:
                print(f"Error in speech recognition: {e}")

        # Get the current async event loop
        loop = asyncio.get_event_loop()

        # ASYNC: Run the blocking function in a separate thread
        # This keeps the async event loop free while blocking work happens in thread
        await loop.run_in_executor(None, process_recognition)

    # Run both tasks concurrently
    try:
        # ASYNC: Run both async functions at the same time
        # gather() allows async functions to run concurrently
        await asyncio.gather(
            audio_receiver(), speech_processor(), return_exceptions=True
        )
    except Exception as e:
        print(f"Error in STT processing: {e}")
    else:
        audio_buffer.finish()

        # Send final complete transcript when session ends
        final_complete = transcript_manager.get_final_only()
        if final_complete:
            await ws.send_text(
                json.dumps(
                    {"type": "session_complete", "final_transcript": final_complete}
                )
            )

        print("STT WebSocket session ended")
