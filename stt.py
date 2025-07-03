from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.cloud import speech
from stream_rag import generate_stream
from collections import deque
from typing import Optional
from tts import TTSConfig, TTSStreamProcessor

import json
import asyncio
import threading
import time

router = APIRouter(prefix="/stt")

class TTSStateManager:
    """Manages TTS playback state to coordinate with speech recognition"""
    def __init__(self):
        self.is_tts_active = False
        self.tts_start_time = None
        self.lock = threading.Lock()
    
    def start_tts(self):
        """Mark TTS as active"""
        with self.lock:
            self.is_tts_active = True
            self.tts_start_time = time.time()
            # print("🔊 TTS playback started")
    
    def end_tts(self):
        """Mark TTS as finished"""
        with self.lock:
            self.is_tts_active = False
            self.tts_start_time = None
            # print("🔇 TTS playback ended")
    
    def is_active(self) -> bool:
        """Check if TTS is currently active"""
        with self.lock:
            return self.is_tts_active
    
    def get_duration(self) -> float:
        """Get how long TTS has been active"""
        with self.lock:
            if self.is_tts_active and self.tts_start_time:
                return time.time() - self.tts_start_time
            return 0.0

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

# ===============================================================

# BLOCKING FUNCTION: This will run in a separate thread
def audio_generator(audio_buffer,speech):
    """BLOCKING: Synchronous generator for Google Speech API"""
    while not audio_buffer.is_finished():
        chunk = audio_buffer.get_chunk()
        if chunk is not None:
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
        else:
            # BLOCKING, stops thread for 100ms.
            threading.Event().wait(0.1)

# ASYNC: Handles WebSocket Communications
async def audio_receiver(ws, audio_buffer):
    """Receive audio chunks from WebSocket with improved error handling"""
    try:
        print("🎧 Audio receiver started")
        while True:
            # Check WebSocket state before attempting to receive
            if ws.application_state == 3:  # DISCONNECTED
                print("🔌 WebSocket disconnected - stopping audio receiver")
                break
                
            try:
                # async/await without blocking
                chunk = await ws.receive_bytes()
                
                # add to thread-safe buffer (quick op, non-blocking)
                if not audio_buffer.is_finished():
                    audio_buffer.add_chunk(chunk)
                else:
                    print("🛑 Audio buffer finished - stopping receiver")
                    break
                    
            except asyncio.CancelledError:
                print("🛑 Audio receiver cancelled")
                raise
            except WebSocketDisconnect:
                print("🔌 WebSocket disconnected by client")
                break
            except Exception as chunk_error:
                print(f"❌ Error receiving audio chunk: {chunk_error}")
                # Don't break on single chunk errors, continue receiving
                continue
                
    except asyncio.CancelledError:
        print("🛑 Audio receiver task cancelled")
        raise
    except Exception as e:
        print(f"❌ Fatal error in audio receiver: {e}")
        raise
    finally:
        # mark buffer as finished (thread-safe operation)
        print("🧹 Audio receiver cleanup - marking buffer as finished")
        audio_buffer.finish()

# ASYNC: Manages the blocking speech processing
# Fix for stt.py - Updated speech_processor function
async def speech_processor(
    speech_client, 
    streaming_config, 
    transcript_manager, 
    audio_buffer, 
    speech, 
    ws, 
    rag_sys,
    tts_state_manager=None):
    """Process speech recognition with TTS state awareness"""
    
    loop = asyncio.get_running_loop()
    config = TTSConfig(voice_name="en-US-Standard-C")
    stream_processor = TTSStreamProcessor(config)
    
    # Create TTS state manager if not provided
    if tts_state_manager is None:
        tts_state_manager = TTSStateManager()

    def process_recognition():
        try:
            responses = speech_client.streaming_recognize(
                config=streaming_config,
                requests=audio_generator(audio_buffer, speech)
            )

            for response in responses:
                # ✅ Check WebSocket state before processing
                if ws.application_state == 3:  # DISCONNECTED
                    print("🔌 WebSocket disconnected, stopping speech processing")
                    break
                
                # Also check if audio buffer is finished (indicates session restart)
                if audio_buffer.is_finished():
                    print("🛑 Audio buffer finished, stopping speech processing")
                    break
                    
                if response.results:
                    top_result = response.results[0]
                    
                    if top_result.alternatives:
                        transcript = top_result.alternatives[0].transcript
                        is_final = top_result.is_final
                        
                        if is_final:
                            transcript_manager.add_final(transcript)
                            print(f"FINAL: {transcript}")
                            
                            response_data = {
                                "type": "final",
                                "text": transcript,
                                "full_transcript": transcript_manager.get_final_only(),
                                "is_user_speaking": False,
                            }

                            if not response_data["is_user_speaking"]: 
                                print("🤖 Starting LLM generation...")
                                
                                async def consume_stream():
                                    try:
                                        async for chunk in generate_stream(rag_sys, transcript):
                                            if ws.application_state == 3:
                                                break
                                            yield chunk
                                    except Exception as e:
                                        print(f"Error consuming stream: {e}")

                                async def handle_audio(audio_data, text):
                                    print(f"Generated audio for: {text}")
                                    try:
                                        if ws.application_state != 3:
                                            # Mark TTS as active when first audio is sent
                                            tts_state_manager.start_tts()
                                            
                                            message = {
                                                "type": "tts_audio",
                                                "audio_data": audio_data,
                                                "text": text,
                                                "encoding": "base64"
                                            }
                                            await ws.send_json(message)
                                    except Exception as e:
                                        print(f"Error sending audio: {e}")

                                async def handle_error(error_msg, text):
                                    try:
                                        # End TTS on error
                                        tts_state_manager.end_tts()
                                        
                                        if ws.application_state != 3:
                                            await ws.send_json({
                                                "type": "tts_error",
                                                "error": error_msg,
                                                "text": text
                                            })
                                    except Exception as e:
                                        print(f"Error sending TTS error: {e}")
                                
                                async def stream_tts():
                                    try:
                                        await stream_processor.process_text_stream(
                                            consume_stream(),
                                            on_audio_ready=handle_audio,
                                            on_error=handle_error
                                        )
                                        # Mark TTS as finished when streaming completes
                                        # print("🎵 TTS streaming completed")
                                        tts_state_manager.end_tts()
                                    except Exception as e:
                                        print(f"Error streaming TTS: {e}")
                                        # End TTS on error
                                        tts_state_manager.end_tts()

                                # ✅ Properly handle the coroutine
                                try:
                                    asyncio.run_coroutine_threadsafe(stream_tts(), loop)
                                except Exception as e:
                                    print(f"Error running TTS coroutine: {e}")
                                    # End TTS on error
                                    tts_state_manager.end_tts()
                            else:
                                print("🔇 Skipping LLM - user still speaking")

                        else:
                            transcript_manager.update_interim(transcript)
                            print(f"INTERIM: {transcript}")
                            
                            response_data = {
                                "type": "interim", 
                                "text": transcript,
                                "display_text": transcript_manager.get_display_text(),
                                "is_user_speaking": True,
                            }

                        # ✅ Safe WebSocket send with proper state checking
                        try:
                            if ws.application_state != 3:  # Not disconnected
                                future = asyncio.run_coroutine_threadsafe(
                                    ws.send_text(json.dumps(response_data)),
                                    loop
                                )
                                # Don't wait for the result to avoid blocking
                            else:
                                print("WebSocket disconnected, skipping message send")
                        except Exception as e:
                            print(f"Error sending WebSocket message: {e}")
                        
        except Exception as e:
            error_message = str(e)
            print(f"Error in speech recognition: {error_message}")
            
            # Check for specific timeout errors
            is_timeout_error = ("Audio Timeout Error" in error_message or 
                               "Long duration elapsed without audio" in error_message or
                               "400" in error_message)
            
            if is_timeout_error:
                # Check if timeout occurred during TTS playback
                if tts_state_manager.is_active():
                    tts_duration = tts_state_manager.get_duration()
                    print(f"🔊 Timeout during TTS playback (duration: {tts_duration:.1f}s) - this is expected")
                    print("🔄 Speech timeout during TTS - propagating for retry")
                    # End TTS state since we're restarting
                    tts_state_manager.end_tts()
                else:
                    print("🔄 Speech timeout detected - propagating for retry")
                
                # Re-raise timeout errors so the retry mechanism can handle them
                raise e

    # ✅ Properly await the executor task
    try:
        await loop.run_in_executor(None, process_recognition)

    # Run both tasks concurrently
    try:
        # ASYNC: Run both async functions at the same time
        # gather() allows async functions to run concurrently
        await asyncio.gather(
            audio_receiver(), speech_processor(), return_exceptions=True
        )
    except Exception as e:
        error_message = str(e)
        print(f"Error in speech processor: {error_message}")
        
        # Propagate timeout errors for retry mechanism
        if ("Audio Timeout Error" in error_message or 
            "Long duration elapsed without audio" in error_message or
            "400" in error_message):
            print("🔄 Propagating timeout error for retry")
            raise e
        else:
            print(f"💥 Non-recoverable speech processor error: {error_message}")
# async def speech_processor(
#     speech_client, 
#     streaming_config, 
#     transcript_manager, 
#     audio_buffer, 
#     speech, 
#     ws, 
#     rag_sys
#     ):

#     """Process speech recognition in background thread"""
#     # BLOCKING FUNCTION: This will run in a separate thread
#     loop = asyncio.get_running_loop()
#     config = TTSConfig(voice_name="en-US-Standard-C")
#     stream_processor = TTSStreamProcessor(config)

#     def process_recognition():
#         try:
#             responses = speech_client.streaming_recognize(
#                 config=streaming_config,
#                 requests=audio_generator(audio_buffer,speech)
#             )

#             for response in responses:
#                 if response.results:
#                     top_result = response.results[0]
#                     alternatives = top_result.alternatives[:3]
#                     # print(f"alternatives: {alternatives}")
#                     # print(f"chosed: {alternatives[0].transcript}")

#                     if top_result.alternatives:
#                         transcript = top_result.alternatives[0].transcript
#                         is_final = top_result.is_final
                        
#                         if is_final:
#                             # Add to final transcript
#                             transcript_manager.add_final(transcript)
#                             print(f"FINAL: {transcript}")
                            
#                             # Send structured response
#                             response_data = {
#                                 "type": "final",
#                                 "text": transcript,
#                                 "full_transcript": transcript_manager.get_final_only(),
#                                 "is_user_speaking" : False,
#                             }

#                             if not response_data["is_user_speaking"]: 
#                                 print("🤖 Starting LLM generation...")
                                
#                                 # Create a coroutine that consumes the async generator
#                                 async def consume_stream():
#                                     try:
#                                         async for chunk in generate_stream(rag_sys, transcript):
#                                             # Process each chunk from the stream
#                                             # print(f"{chunk}")
#                                             yield chunk
#                                             # You could send these chunks to WebSocket if needed
#                                             # await ws.send_text(chunk)
#                                     except Exception as e:
#                                         print(f"Error consuming stream: {e}")

#                                 async def handle_audio(audio_data, text):
#                                     # Your custom audio handling logic
#                                     print(f"Generated audio for: {text}")
#                                     try:
#                                         # Send audio data to frontend via WebSocket
#                                         await ws.send_json({
#                                             "type": "tts_audio",
#                                             "audio_data": audio_data,
#                                             "text": text,
#                                             "encoding": "base64"
#                                         })
#                                         # print(f"🎵 Sent audio to frontend for: '{text[:30]}...'")
#                                     except Exception as e:
#                                         print(f"Error sending audio to frontend: {e}")

#                                 async def handle_error(error_msg, text):
#                                     """Send TTS error to frontend"""
#                                     try:
#                                         await ws.send_json({
#                                             "type": "tts_error",
#                                             "error": error_msg,
#                                             "text": text
#                                         })
#                                         print(f"Error: {error_msg}")
#                                     except Exception as e:
#                                         print(f"Error sending TTS error: {e}")
                                    
                                
#                                 async def stream_tts():
#                                     try:
#                                         await stream_processor.process_text_stream(
#                                             consume_stream(),
#                                             on_audio_ready=handle_audio,
#                                             on_error=handle_error
#                                         )
#                                     except Exception as e:
#                                         print(f"Error streaming TTS: {e}")

#                                 # Now run the coroutine
#                                 asyncio.run_coroutine_threadsafe(
#                                     stream_tts(),
#                                     loop
#                                 )
#                             else:
#                                 print("🔇 Skipping LLM - user still speaking")

#                         else:
#                             # Update interim text
#                             transcript_manager.update_interim(transcript)
#                             print(f"INTERIM: {transcript}")
                            
#                             # Send structured response
#                             response_data = {
#                                 "type": "interim", 
#                                 "text": transcript,
#                                 "display_text": transcript_manager.get_display_text(),
#                                 "is_user_speaking" : True,
#                             }

#                         # BRIDGE: Send from blocking thread back to async world
#                         # This is the magic that connects thread to async

#                         # Send to frontend
#                         asyncio.run_coroutine_threadsafe(
#                             ws.send_text(json.dumps(response_data)),
#                             loop
#                         )
                        
#         except Exception as e:
#             print(f"Error in speech recognition: {e}")

#     recognition_task = asyncio.create_task(
#         loop.run_in_executor(None, process_recognition)
#     )

#     await asyncio.gather(
#         recognition_task, 
#         return_exceptions=True
#     )

# @router.websocket("")
# async def stt_route(ws: WebSocket): 
#     await ws.accept()
#     await ws.send_text("✅ WebSocket connected to Google STT")

#     # Create Google Speech client. 
#     speech_client = speech.SpeechClient()
#     audio_buffer = AudioBuffer()
#     transcript_manager = TranscriptManager()

#     # Configuration setup.
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=16000,
#         language_code="en-US",
#     )

#     streaming_config = speech.StreamingRecognitionConfig(
#         config=config,
#         interim_results=True,
#     )

#     # Run both tasks concurrently
#     try:
#         # ASYNC: Run both async functions at the same time
#         # gather() allows async functions to run concurrently
#         await asyncio.gather(
#             audio_receiver(ws, audio_buffer),
#             speech_processor(speech_client, streaming_config, transcript_manager, audio_buffer, speech, ws),
#             return_exceptions=True
#         )
#     except Exception as e:
#         print(f"Error in STT processing: {e}")
#     else:
#         audio_buffer.finish()
        
#         # Send final complete transcript when session ends
#         final_complete = transcript_manager.get_final_only()
#         if final_complete:
#             await ws.send_text(json.dumps({
#                 "type": "session_complete",
#                 "final_transcript": final_complete
#             }))
        
#         print("STT WebSocket session ended")