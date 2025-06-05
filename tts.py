from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import texttospeech
import asyncio
import base64
import json
import httpx # For making asynchronous HTTP requests to Ollama
import re # For regular expressions to detect sentence endings

router = APIRouter(prefix="/tts")

# Initialize Google Cloud Text-to-Speech client
client_tts = texttospeech.TextToSpeechClient()

# Ollama settings
OLLAMA_API_BASE_URL = "http://localhost:11434/api" # Default Ollama API URL
OLLAMA_MODEL_NAME = "llama3" # Or "mistral", "gemma", etc.

async def stream_ollama_response(prompt: str):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True # Crucial for streaming chunks
    }
    url = f"{OLLAMA_API_BASE_URL}/chat"

    async with httpx.AsyncClient() as httpx_client:
        try:
            async with httpx_client.stream("POST", url, headers=headers, json=payload, timeout=None) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    try:
                        decoded_chunk = chunk.decode('utf-8')
                        
                        for line in decoded_chunk.splitlines():
                            if line.strip():
                                json_data = json.loads(line)
                                
                                if "content" in json_data["message"]:
                                    text_content = json_data["message"]["content"]
                                    
                                    if text_content:
                                        yield text_content
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error in Ollama stream: {e}, chunk: {decoded_chunk}")
                        # Consider yielding an error message or stopping
                    except Exception as e:
                        print(f"Error processing Ollama stream chunk: {e}, chunk: {decoded_chunk}")
                        # Consider yielding an error message or stopping
        except httpx.RequestError as e:
            print(f"Ollama request error: {e}")
            yield f"Error connecting to LLM: {e}. Is Ollama running and model '{OLLAMA_MODEL_NAME}' downloaded?"
        except httpx.HTTPStatusError as e:
            print(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
            yield f"LLM server error: {e.response.status_code} - {e.response.text}"

# Define common sentence ending punctuation
SENTENCE_ENDINGS = re.compile(r'[.?!]\s+|\n\n') # Also consider double newlines for paragraph breaks

@router.websocket("")
async def websocket_llm_text_to_speech(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted for LLM-TTS.")

    sentence_buffer = "" # Buffer to accumulate text for sentence segmentation

    try:
        while True:
            message = await websocket.receive_text()
            print(f"Received prompt: {message}")

            try:
                data = json.loads(message)
                user_prompt = data.get("prompt")
                tts_voice_name = data.get("voice", "en-US-Standard-C")
                tts_language_code = data.get("language", "en-US")
                tts_sample_rate_hertz = data.get("sampleRate", 24000)
            except json.JSONDecodeError:
                print("Invalid JSON received, treating as plain text prompt.")
                user_prompt = message
                tts_voice_name = "en-US-Standard-C"
                tts_language_code = "en-US"
                tts_sample_rate_hertz = 24000

            if not user_prompt:
                continue

            llm_response_stream = stream_ollama_response(user_prompt)

            full_llm_response = ""
            async for chunk in llm_response_stream:
                full_llm_response += chunk
                sentence_buffer += chunk

                # Send raw text chunk to frontend for display immediately
                await websocket.send_json({"text_chunk": chunk})

                # Sentence segmentation logic
                match = SENTENCE_ENDINGS.search(sentence_buffer)
                while match:
                    sentence_end_index = match.end()
                    sentence_to_synthesize = sentence_buffer[:sentence_end_index].strip()
                    sentence_buffer = sentence_buffer[sentence_end_index:] # Keep remaining text in buffer

                    if sentence_to_synthesize:
                        print(f"Synthesizing sentence: '{sentence_to_synthesize}'") # Diagnostic print

                        synthesis_input = texttospeech.SynthesisInput(text=sentence_to_synthesize)
                        voice = texttospeech.VoiceSelectionParams(
                            language_code=tts_language_code,
                            name=tts_voice_name,
                        )
                        audio_config = texttospeech.AudioConfig(
                            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                            sample_rate_hertz=tts_sample_rate_hertz,
                        )

                        try:
                            # Use asyncio.to_thread for blocking TTS call
                            tts_response = await asyncio.to_thread(
                                client_tts.synthesize_speech,
                                input=synthesis_input, voice=voice, audio_config=audio_config
                            )

                            encoded_audio_chunk = base64.b64encode(tts_response.audio_content).decode('utf-8')
                            await websocket.send_text(encoded_audio_chunk) # Send audio as plain text (base64)
                            # NO asyncio.sleep(0.001) here. Frontend handles pacing.

                        except Exception as tts_e:
                            print(f"Error during TTS synthesis for sentence '{sentence_to_synthesize[:50]}...': {tts_e}")
                            await websocket.send_json({"error": f"TTS error: {str(tts_e)}"})
                            # Propagate error and stop processing current request
                            raise tts_e # Re-raise to break out of current LLM chunk loop

                    # Check for next sentence ending in the remaining buffer
                    match = SENTENCE_ENDINGS.search(sentence_buffer)

            # After LLM stream ends, synthesize any remaining text in the buffer
            if sentence_buffer.strip():
                print(f"Synthesizing remaining text: '{sentence_buffer.strip()}'") # Diagnostic print
                synthesis_input = texttospeech.SynthesisInput(text=sentence_buffer.strip())
                voice = texttospeech.VoiceSelectionParams(
                    language_code=tts_language_code,
                    name=tts_voice_name,
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=tts_sample_rate_hertz,
                )
                try:
                    tts_response = await asyncio.to_thread(
                        client_tts.synthesize_speech,
                        input=synthesis_input, voice=voice, audio_config=audio_config
                    )
                    encoded_audio_chunk = base64.b64encode(tts_response.audio_content).decode('utf-8')
                    await websocket.send_text(encoded_audio_chunk)
                except Exception as tts_e:
                    print(f"Error during TTS synthesis for remaining text '{sentence_buffer.strip()[:50]}...': {tts_e}")
                    await websocket.send_json({"error": f"TTS error for final segment: {str(tts_e)}"})


            # Signal end of stream from backend
            await websocket.send_json({"stream_end": True})
            print(f"Full LLM response: {full_llm_response}")
            print("Finished processing LLM response and TTS.")

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Send error to frontend if possible
        try:
            await websocket.send_json({"error": f"Backend error: {str(e)}"})
        except RuntimeError:
            pass # WebSocket might already be closed

@router.get("/file")
async def serve_index():
    from fastapi.responses import FileResponse
    return FileResponse("frontend/dist/index.html")