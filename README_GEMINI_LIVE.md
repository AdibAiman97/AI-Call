# Gemini Live Voice-to-Voice RAG Assistant

## Overview

This project has been **completely refactored** to use **Google's Gemini Live 2.5 Flash Preview with Native Audio Dialog** instead of the previous STT/TTS pipeline. This provides a unified, more natural voice conversation experience with better context understanding and seamless integration with your property RAG system.

## 🚀 Key Features

### **Unified Voice Experience**
- **Single WebSocket Connection**: One connection handles both voice input and output
- **Native Audio Dialog**: Gemini Live processes audio directly without transcription overhead
- **Real-time Conversation**: Low-latency voice interactions with natural interruption handling
- **Context Awareness**: Maintains conversation context throughout the session

### **Enhanced RAG Integration**
- **Function Calling**: RAG queries are handled via Gemini Live's tool calling feature
- **Intelligent Query Routing**: Automatically determines when to query the property database
- **Contextual Responses**: Combines RAG knowledge with conversational AI capabilities
- **Memory Management**: Conversation history is maintained by Gemini Live

### **Professional Voice Assistant**
- **Gina Voice Assistant**: Friendly, professional sales consultant persona
- **Property Expertise**: Specialized knowledge about Gamuda Cove properties
- **Appointment Booking**: Guides customers toward scheduling property viewings
- **Multi-language Support**: Enhanced multilingual capabilities

## 🏗️ Architecture Changes

### **Before (STT/TTS Pipeline)**
```
Client Audio → Google Speech STT → RAG System → Gemini Pro → Google TTS → Client Audio
```

### **After (Gemini Live)**
```
Client Audio ↔ Gemini Live 2.5 Flash (with RAG Function Calling) ↔ Client Audio
```

## 🛠️ Technical Implementation

### **Core Components**

1. **`gemini_live_websocket.py`**
   - Handles WebSocket connections to Gemini Live API via **Vertex AI authentication**
   - Manages audio streaming and function calling
   - Integrates RAG system via tool calling
   - Uses Google Cloud Application Default Credentials (ADC)

2. **`main.py`** (Updated)
   - Replaced `/stt/{call_session_id}` endpoint with Gemini Live integration
   - Simplified WebSocket handling
   - Enhanced error handling and session management

3. **`test_client.html`**
   - Complete HTML client for testing voice conversations
   - Real-time audio recording and playback
   - Session management and status monitoring

### **Key Classes**

#### `GeminiLiveConfig`
```python
@dataclass
class GeminiLiveConfig:
    api_key: str
    project_id: str
    location: str = "us-central1"
    model_name: str = "gemini-2.5-flash-preview-native-audio-dialog"
    voice_name: str = "Aoede"  # Professional female voice
    max_output_tokens: int = 1000
    temperature: float = 0.7
```

#### `GeminiLiveSession`
- Manages individual voice conversation sessions
- Handles audio streaming and RAG integration
- Maintains conversation context and session state

#### `GeminiLiveManager`
- Manages multiple concurrent sessions
- Handles session creation, cleanup, and resource management

## 📋 Setup Instructions

### **1. Vertex AI Authentication Setup**

**Option A: Application Default Credentials (Recommended)**
```bash
# Install Google Cloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate with your Google account
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

**Option B: Service Account Key**
```bash
# Download service account key and set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

### **2. Environment Variables**
```bash
# Required
GCP_PROJECT_ID=your_gcp_project_id
GCP_LOCATION=us-central1
MONGO_DB=your_mongodb_connection_string

# Optional
VOICE_NAME=Aoede  # Options: Aoede, Charon, Fenrir, Kore, Puck
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Initialize RAG System**
Ensure your MongoDB Atlas Vector Search is properly configured with property documents.

### **4. Run the Server**
```bash
python main.py
```

### **5. Test the System**
Open `test_client.html` in a web browser to test voice conversations.

## 🎯 Usage Guide

### **Starting a Voice Conversation**

1. **Connect to WebSocket**
   ```javascript
   const ws = new WebSocket('ws://localhost:8000/stt/1');
   ```

2. **Send Audio Data**
   ```javascript
   ws.send(JSON.stringify({
       type: 'audio_chunk',
       audio_data: base64AudioData
   }));
   ```

3. **Receive Audio Response**
   ```javascript
   ws.onmessage = (event) => {
       const data = JSON.parse(event.data);
       if (data.type === 'audio_response') {
           playAudio(data.audio_data);
       }
   };
   ```

### **Conversation Flow**

1. **Greeting**: "Hi, I'm Gina from Gamuda Cove. How can I help you today?"
2. **Property Inquiry**: User asks about properties, pricing, or amenities
3. **RAG Query**: System automatically queries the property database
4. **Response**: Gina provides detailed information based on RAG results
5. **Appointment**: Guides user toward scheduling a viewing

### **Sample Conversation**

```
User: "Hi, what properties do you have available?"
Gina: "Hello! I'm Gina from Gamuda Cove. We have several beautiful properties available. Let me check our current inventory for you."

[RAG Query: "available properties at Gamuda Cove"]

Gina: "We have some fantastic options! Our Areca terrace homes are 1,824 square feet, priced at seven hundred sixty thousand Ringgit. We also have semi-detached and bungalow options. What type of property interests you most?"

User: "Tell me about the Areca homes."
Gina: "Great choice! The Areca is a two-storey terrace home with modern design and excellent amenities. The monthly installment is around three thousand eight hundred Ringgit. Would you like to schedule a viewing to see it in person?"
```

## 🔧 API Endpoints

### **WebSocket Endpoints**

#### `GET /stt/{call_session_id}`
- **Purpose**: Establish Gemini Live voice conversation
- **Protocol**: WebSocket
- **Authentication**: API key via environment variable

### **REST Endpoints**

#### `GET /system-status`
- **Purpose**: Check overall system health
- **Response**: RAG system and Gemini Live status

#### `GET /gemini-live-status`
- **Purpose**: Check Gemini Live system status
- **Response**: Active sessions, model info, voice settings

#### `GET /test-rag`
- **Purpose**: Test RAG system functionality
- **Response**: Sample property query results

## 📊 Audio Format Specifications

### **Input Audio (Client → Gemini Live)**
- **Format**: 16-bit PCM
- **Sample Rate**: 16,000 Hz
- **Channels**: Mono (1 channel)
- **Encoding**: Base64 over WebSocket

### **Output Audio (Gemini Live → Client)**
- **Format**: 16-bit PCM
- **Sample Rate**: 24,000 Hz
- **Channels**: Mono (1 channel)
- **Encoding**: Base64 over WebSocket

## 🎭 Voice Personalities

### **Available Voices**
- **Aoede**: Professional female voice (default)
- **Charon**: Deep male voice
- **Fenrir**: Energetic male voice
- **Kore**: Soft female voice
- **Puck**: Playful neutral voice

### **Voice Configuration**
```python
gemini_config = GeminiLiveConfig(
    voice_name="Aoede",  # Change to preferred voice
    # ... other settings
)
```

## 🚨 Error Handling

### **Common Issues**

1. **Connection Failed**
   - Check Vertex AI authentication (run `gcloud auth application-default login`)
   - Verify GOOGLE_APPLICATION_CREDENTIALS if using service account
   - Verify GCP project and location settings
   - Ensure network connectivity

2. **Audio Not Playing**
   - Check browser audio permissions
   - Verify audio format conversion
   - Test volume settings

3. **RAG Not Responding**
   - Verify MongoDB connection
   - Check vector store configuration
   - Test RAG system independently

### **Debugging Tips**

- Enable debug logging in `gemini_live_websocket.py`
- Use `/system-status` endpoint to check system health
- Test with `test_client.html` for isolated debugging

## 🔒 Security Considerations

### **Production Deployment**
- Use HTTPS/WSS for WebSocket connections
- Implement proper authentication
- Rate limiting for API calls
- Secure environment variable management

### **API Key Security**
- Never expose API keys in client-side code
- Use server-side proxy for production
- Implement token refresh mechanisms
- Monitor API usage and costs

## 📈 Performance Optimization

### **Latency Reduction**
- Direct client-to-Gemini Live connection
- Reduced audio processing overhead
- Efficient RAG query caching
- Optimized WebSocket message handling

### **Resource Management**
- Automatic session cleanup
- Memory-efficient audio streaming
- Connection pooling for high concurrency
- Graceful error recovery

## 🧪 Testing

### **Unit Tests**
```bash
# Test RAG system
curl http://localhost:8000/test-rag

# Test system status
curl http://localhost:8000/system-status

# Test Gemini Live status
curl http://localhost:8000/gemini-live-status
```

### **Integration Tests**
1. Open `test_client.html` in browser
2. Click "Connect" to establish WebSocket connection
3. Start recording and ask property questions
4. Verify voice responses and conversation flow

## 🔮 Future Enhancements

### **Planned Features**
- Video input support for property viewing
- Multi-language conversation support
- Advanced interrupt handling
- Session resumption capabilities
- Enhanced analytics and monitoring

### **Integration Opportunities**
- CRM system integration
- Calendar booking system
- Property management platform
- Customer relationship tracking

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Test with `test_client.html`
5. Submit pull request

### **Code Standards**
- Follow PEP 8 for Python code
- Use type hints for better maintainability
- Add comprehensive error handling
- Include detailed docstrings

## 📞 Support

For technical support or questions:
- Check the `/system-status` endpoint for system health
- Review logs for error messages
- Test individual components (RAG, WebSocket, audio)
- Consult Gemini Live API documentation

---

**Note**: This refactored system provides a significantly improved user experience with more natural voice interactions and better context understanding. The unified architecture reduces complexity while enhancing performance and reliability. 