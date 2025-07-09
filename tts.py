from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import texttospeech
import asyncio
import base64
import json
import re
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass

# Initialize Google Cloud Text-to-Speech client
client_tts = texttospeech.TextToSpeechClient()

# Define common sentence ending punctuation for natural speech breaks
SENTENCE_ENDINGS = re.compile(r'[.?!]\s+|\n\n')

@dataclass
class TTSConfig:
    """Configuration class for Text-to-Speech settings"""
    voice_name: str = "en-US-Standard-C"
    language_code: str = "en-US"
    sample_rate_hertz: int = 24000
    audio_encoding: texttospeech.AudioEncoding = texttospeech.AudioEncoding.LINEAR16

class TTSProcessor:
    """Core TTS processing class that can be used anywhere"""
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize TTS processor with configuration
        
        Args:
            config: TTS configuration settings, uses defaults if None
        """
        self.config = config or TTSConfig()
        self.sentence_buffer = ""
    
    async def synthesize_text(self, text: str) -> str:
        """
        Convert text to speech and return base64 encoded audio
        
        Args:
            text: The text to convert to speech
            
        Returns:
            Base64 encoded audio data as string
        """
        # Create synthesis input from text
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure voice parameters
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.config.language_code,
            name=self.config.voice_name,
        )
        
        # Configure audio output settings
        audio_config = texttospeech.AudioConfig(
            audio_encoding=self.config.audio_encoding,
            sample_rate_hertz=self.config.sample_rate_hertz,
        )
        
        try:
            # Use asyncio.to_thread to handle blocking TTS call asynchronously
            tts_response = await asyncio.to_thread(
                client_tts.synthesize_speech,
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Encode audio content to base64 for easy transmission
            encoded_audio = base64.b64encode(tts_response.audio_content).decode('utf-8')
            return encoded_audio
            
        except Exception as e:
            print(f"Error during TTS synthesis for text '{text[:50]}...': {e}")
            raise
    
    def process_text_chunk(self, text_chunk: str) -> List[str]:
        """
        Process incoming text chunk and return complete sentences ready for synthesis
        
        Args:
            text_chunk: New text to add to the buffer
            
        Returns:
            List of complete sentences ready for TTS synthesis
        """
        # Add new text to buffer
        self.sentence_buffer += text_chunk
        
        complete_sentences = []
        
        # Find sentence endings in the buffer
        match = SENTENCE_ENDINGS.search(self.sentence_buffer)
        while match:
            # Extract complete sentence
            sentence_end_index = match.end()
            sentence_to_synthesize = self.sentence_buffer[:sentence_end_index].strip()
            
            # Remove processed sentence from buffer
            self.sentence_buffer = self.sentence_buffer[sentence_end_index:]
            
            if sentence_to_synthesize:
                complete_sentences.append(sentence_to_synthesize)
            
            # Look for next sentence ending
            match = SENTENCE_ENDINGS.search(self.sentence_buffer)
        
        return complete_sentences
    
    def get_remaining_text(self) -> Optional[str]:
        """
        Get any remaining text in the buffer that hasn't been processed
        
        Returns:
            Remaining text or None if buffer is empty
        """
        remaining = self.sentence_buffer.strip()
        return remaining if remaining else None
    
    def clear_buffer(self):
        """Clear the sentence buffer"""
        self.sentence_buffer = ""

class TTSStreamProcessor:
    """Handles streaming TTS processing with result callbacks"""
    
    def __init__(self, config: Optional[TTSConfig] = None):
        """
        Initialize TTS stream processor
        
        Args:
            config: TTS configuration settings
        """
        self.processor = TTSProcessor(config)
    
    async def process_text_stream(
        self, 
        text_stream: AsyncGenerator[str, None],
        on_audio_ready=None,
        on_error=None,
        on_complete=None
    ):
        """
        Process streaming text and generate audio output with callbacks
        
        Args:
            text_stream: Async generator yielding text chunks
            on_audio_ready: Callback function for when audio is ready (audio_data, text)
            on_error: Callback function for errors (error_message, text)
            on_complete: Callback function when processing is complete
        """
        try:
            async for text_chunk in text_stream:
                # Process the text chunk to find complete sentences
                complete_sentences = self.processor.process_text_chunk(text_chunk)
                
                # Synthesize each complete sentence
                for sentence in complete_sentences:
                    # print(f"Synthesizing sentence: '{sentence}'")
                    
                    try:
                        # Convert sentence to speech
                        audio_data = await self.processor.synthesize_text(sentence)
                        
                        # Call callback if provided
                        if on_audio_ready:
                            await on_audio_ready(audio_data, sentence)
                        
                    except Exception as e:
                        error_msg = f"TTS synthesis error: {str(e)}"
                        print(f"Error during TTS synthesis for sentence '{sentence[:50]}...': {e}")
                        
                        if on_error:
                            await on_error(error_msg, sentence)
            
            # Process any remaining text in buffer
            remaining_text = self.processor.get_remaining_text()
            if remaining_text:
                print(f"Synthesizing remaining text: '{remaining_text}'")
                
                try:
                    audio_data = await self.processor.synthesize_text(remaining_text)
                    # print("audio data here: ", audio_data)
                    if on_audio_ready:
                        await on_audio_ready(audio_data, remaining_text)
                except Exception as e:
                    error_msg = f"TTS synthesis error for final segment: {str(e)}"
                    print(f"Error during TTS synthesis for remaining text '{remaining_text[:50]}...': {e}")
                    
                    if on_error:
                        await on_error(error_msg, remaining_text)
            
            # Clear buffer after processing
            self.processor.clear_buffer()
            
            # Signal completion
            if on_complete:
                
                await on_complete()
                
        except Exception as e:
            error_msg = f"Stream processing error: {str(e)}"
            print(f"An unexpected error occurred: {e}")
            
            if on_error:
                await on_error(error_msg, "")
