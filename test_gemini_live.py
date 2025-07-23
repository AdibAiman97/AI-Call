#!/usr/bin/env python3
"""
Gemini Live API Test Script
Adapted from the Colab notebook for local execution
"""

import asyncio
import contextlib
import base64
import json
import time
import wave
import numpy as np
import os
from websockets.asyncio.client import connect

# Load API key from environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

# Configuration
HOST = 'generativelanguage.googleapis.com'
MODEL = 'models/gemini-2.0-flash-live-001'
INITIAL_REQUEST_TEXT = "Hello! Can you hear me?"

# Audio configuration
class AudioConfig:
    def __init__(self, sample_rate=24000, channels=1):  # Gemini uses 24kHz
        self.sample_rate = sample_rate
        self.format = 'S16_LE'
        self.channels = channels
    
    @property
    def sample_size(self):
        return 2  # 16-bit
    
    @property
    def frame_size(self):
        return self.channels * self.sample_size
    
    @property
    def numpy_dtype(self):
        return np.dtype(np.int16).newbyteorder('<')

AUDIO_CONFIG = AudioConfig(sample_rate=24000)  # Match Gemini's output rate

def encode_audio_input(data: bytes, config: AudioConfig) -> dict:
    """Build message with user input audio bytes."""
    return {
        'realtimeInput': {
            'mediaChunks': [{
                'mimeType': f'audio/pcm;rate={config.sample_rate}',
                'data': base64.b64encode(data).decode('UTF-8'),
            }],
        },
    }

def encode_text_input(text: str) -> dict:
    """Builds message with user input text."""
    return {
        'clientContent': {
            'turns': [{
                'role': 'USER',
                'parts': [{'text': text}],
            }],
            'turnComplete': True,
        },
    }

def decode_audio_output(input_msg: dict) -> bytes:
    """Returns byte string with model output audio."""
    result = []
    content_input = input_msg.get('serverContent', {})
    content = content_input.get('modelTurn', {})
    for part in content.get('parts', []):
        data = part.get('inlineData', {}).get('data', '')
        if data:
            result.append(base64.b64decode(data))
    return b''.join(result)

class SimpleAudioRecorder:
    """Simple audio recorder using PyAudio for testing."""
    
    def __init__(self, input_rate=16000):  # Use 16kHz for input (more efficient)
        self.input_rate = input_rate
        self.channels = 1
        self.audio = None
        self.stream = None
        self.is_recording = False
        
        try:
            import pyaudio
            self.pyaudio = pyaudio
        except ImportError:
            print("PyAudio not available. Install with: pip install pyaudio")
            self.pyaudio = None
    
    async def __aenter__(self):
        if not self.pyaudio:
            print("Audio recording not available - PyAudio not installed")
            return self
            
        self.audio = self.pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.pyaudio.paInt16,
            channels=self.channels,
            rate=self.input_rate,
            input=True,
            frames_per_buffer=1024
        )
        self.is_recording = True
        print(f"Audio recording started at {self.input_rate}Hz")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
        print("Audio recording stopped")
    
    async def read(self) -> bytes:
        """Read audio data."""
        if not self.stream or not self.is_recording:
            # Return silence if no audio available
            silence_frames = int(0.1 * self.input_rate)  # 100ms of silence
            return b'\x00' * (silence_frames * 2)  # 2 bytes per sample (16-bit)
        
        try:
            # Read audio data (non-blocking)
            data = self.stream.read(1024, exception_on_overflow=False)
            return data
        except Exception as e:
            print(f"Audio read error: {e}")
            return b'\x00' * 2048  # Return silence on error

# Global audio streaming manager
audio_stream = None
audio_queue = None

class ContinuousAudioPlayer:
    """Continuous audio player that queues and plays audio chunks smoothly."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.audio = None
        self.stream = None
        self.queue = asyncio.Queue()
        self.is_playing = False
        self.playback_task = None
        
        try:
            import pyaudio
            self.pyaudio = pyaudio
        except ImportError:
            self.pyaudio = None
    
    async def start(self):
        """Start the continuous audio player."""
        if not self.pyaudio:
            print("PyAudio not available - audio will be saved to files")
            return
        
        self.audio = self.pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.pyaudio.paInt16,
            channels=self.config.channels,
            rate=self.config.sample_rate,
            output=True,
            frames_per_buffer=1024
        )
        
        self.is_playing = True
        self.playback_task = asyncio.create_task(self._playback_loop())
        print(f"Continuous audio player started at {self.config.sample_rate}Hz")
    
    async def stop(self):
        """Stop the continuous audio player."""
        self.is_playing = False
        
        if self.playback_task:
            self.playback_task.cancel()
            try:
                await self.playback_task
            except asyncio.CancelledError:
                pass
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("Continuous audio player stopped")
    
    async def enqueue_audio(self, audio_data: bytes):
        """Add audio data to the playback queue."""
        if audio_data and len(audio_data) > 0:
            await self.queue.put(audio_data)
            print(f"Enqueued audio chunk ({len(audio_data)} bytes)")
    
    async def _playback_loop(self):
        """Continuous playback loop."""
        try:
            while self.is_playing:
                try:
                    # Wait for audio data with timeout
                    audio_data = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                    
                    if self.stream and audio_data:
                        # Play the audio data immediately
                        self.stream.write(audio_data)
                        
                except asyncio.TimeoutError:
                    # No audio data available, continue
                    continue
                except Exception as e:
                    print(f"Playback error: {e}")
                    continue
        
        except asyncio.CancelledError:
            print("Playback loop cancelled")
            raise

# Initialize global audio player
audio_player = None

async def initialize_audio_player(config: AudioConfig):
    """Initialize the global audio player."""
    global audio_player
    if audio_player is None:
        audio_player = ContinuousAudioPlayer(config)
        await audio_player.start()

async def play_audio_data(audio_data: bytes, config: AudioConfig):
    """Add audio data to the continuous playback queue."""
    global audio_player
    
    if not audio_data or len(audio_data) == 0:
        return
    
    try:
        # Initialize player if needed
        if audio_player is None:
            await initialize_audio_player(config)
        
        # Add to playback queue
        if audio_player and audio_player.pyaudio:
            await audio_player.enqueue_audio(audio_data)
        else:
            # Fallback: save to file
            filename = f"response_{int(time.time())}.wav"
            save_audio_to_file(audio_data, config, filename)
            print(f"Audio saved to {filename}")
            
    except Exception as e:
        print(f"Audio playback error: {e}")
        # Fallback: save to file
        filename = f"response_error_{int(time.time())}.wav"
        save_audio_to_file(audio_data, config, filename)
        print(f"Audio saved to {filename} due to playback error")

async def cleanup_audio_player():
    """Clean up the global audio player."""
    global audio_player
    if audio_player:
        await audio_player.stop()
        audio_player = None

def save_audio_to_file(audio_data: bytes, config: AudioConfig, filename: str):
    """Save audio data to WAV file."""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(config.channels)
        wav_file.setsampwidth(config.sample_size)
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(audio_data)

async def main():
    """Main function to run the Gemini Live API test."""
    print(f"Starting Gemini Live API test...")
    print(f"Model: {MODEL}")
    
    # Connect to WebSocket
    ws_url = f'wss://{HOST}/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GOOGLE_API_KEY}'
    print("Connecting to Gemini Live API...")
    
    async with connect(ws_url) as conn:
        print("Connected!")
        
        # Send initial setup
        initial_request = {
            'setup': {
                'model': MODEL,
            },
        }
        await conn.send(json.dumps(initial_request))
        print("Setup message sent")
        
        # Send initial text message
        if INITIAL_REQUEST_TEXT:
            await conn.send(json.dumps(encode_text_input(INITIAL_REQUEST_TEXT)))
            print(f"Sent: {INITIAL_REQUEST_TEXT}")
        
        # Start audio recording (16kHz input)
        input_config = AudioConfig(sample_rate=16000)  # Input at 16kHz
        async with SimpleAudioRecorder(16000) as audio_recorder:
            
            # Task to send audio
            async def send_audio():
                print("Starting audio input...")
                try:
                    while True:
                        data = await audio_recorder.read()
                        if data and len(data) > 0:
                            await conn.send(json.dumps(encode_audio_input(data, input_config)))
                        await asyncio.sleep(0.1)  # 100ms intervals
                except asyncio.CancelledError:
                    print("Audio input stopped")
                    raise
            
            # Start audio sending task
            audio_task = asyncio.create_task(send_audio())
            
            print("\n=== Conversation Started ===")
            print("Speak into your microphone. Press Ctrl+C to stop.\n")
            
            try:
                # Process incoming messages
                enqueued_audio = []
                async for msg in conn:
                    msg_data = json.loads(msg)
                    
                    # Handle audio output
                    if audio_data := decode_audio_output(msg_data):
                        enqueued_audio.append(audio_data)
                        await play_audio_data(audio_data, AUDIO_CONFIG)
                    
                    # Handle interruption
                    elif 'interrupted' in msg_data.get('serverContent', {}):
                        print('<interrupted by user>')
                    
                    # Handle turn completion
                    elif 'turnComplete' in msg_data.get('serverContent', {}):
                        # if enqueued_audio:
                        #     # Save complete response to file
                        #     complete_audio = b''.join(enqueued_audio)
                        #     filename = f"gemini_response_{int(time.time())}.wav"
                        #     save_audio_to_file(complete_audio, AUDIO_CONFIG, filename)
                        #     print(f"Complete response saved to {filename}")
                        
                        # enqueued_audio = []
                        print('<end of turn - you can speak now>')
                    
                    # Handle text responses
                    elif 'serverContent' in msg_data and 'modelTurn' in msg_data['serverContent']:
                        model_turn = msg_data['serverContent']['modelTurn']
                        if 'parts' in model_turn:
                            for part in model_turn['parts']:
                                if 'text' in part:
                                    print(f"Gemini: {part['text']}")
                    
                    # Handle other messages
                    elif msg_data != {'serverContent': {}}:
                        print(f'Other message: {msg_data}')
            
            except KeyboardInterrupt:
                print("\nStopping conversation...")
                audio_task.cancel()
                try:
                    await audio_task
                except asyncio.CancelledError:
                    pass
                finally:
                    await cleanup_audio_player()
            
            except Exception as e:
                print(f"Error during conversation: {e}")
                audio_task.cancel()
                await cleanup_audio_player()
                raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        # Cleanup in case it wasn't done
        try:
            asyncio.run(cleanup_audio_player())
        except:
            pass
    except Exception as e:
        print(f"Error: {e}")
        # Cleanup in case it wasn't done
        try:
            asyncio.run(cleanup_audio_player())
        except:
            pass