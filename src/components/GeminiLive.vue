<template>
  <!-- Hidden component - no UI, only WebSocket and audio logic -->
  <div style="display: none;"></div>
</template>

<script lang="ts" setup>
import { ref, onMounted, onUnmounted, watch } from "vue"
import { useCallStore } from '@/stores/call'

// Store integration
const callStore = useCallStore()

// WebSocket and Audio variables
let websocket: WebSocket | null = null
let audioStream: MediaStream | null = null
let audioContext: AudioContext | null = null
let analyser: AnalyserNode | null = null
let continuousAudioPlayer: any = null

// Audio configuration - match Gemini Live API requirements exactly
const INPUT_AUDIO_CONFIG = {
  sampleRate: 16000, // 16kHz input as required by Gemini
  channels: 1,
  bitsPerSample: 16,
}

const OUTPUT_AUDIO_CONFIG = {
  sampleRate: 24000, // 24kHz output from Gemini
  channels: 1,
  bitsPerSample: 16,
}

// Audio buffering for better speech recognition
let audioBuffer: Float32Array[] = []
let bufferSize = 0
const TARGET_BUFFER_SIZE = 8192 // Larger buffer for better quality

// Continuous Audio Player for smooth playback
class ContinuousAudioPlayer {
  private queue: ArrayBuffer[] = []
  private isPlaying = false
  private audioContext: AudioContext | null = null
  private analyser: AnalyserNode | null = null
  private dataArray: Uint8Array | null = null

  private nextChunkTime = 0

  constructor(callStore?: any, parentComponent?: any) {
    this.audioContext = new AudioContext({
      sampleRate: OUTPUT_AUDIO_CONFIG.sampleRate,
      latencyHint: 'interactive', // Optimize for low latency
    })
    this.nextChunkTime = this.audioContext.currentTime
    this.callStore = callStore
    this.parentComponent = parentComponent
    
    // Create analyser for blob visualization
    this.analyser = this.audioContext.createAnalyser()
    this.analyser.fftSize = 256
    this.analyser.smoothingTimeConstant = 0.8
    const bufferLength = this.analyser.frequencyBinCount
    this.dataArray = new Uint8Array(bufferLength)
    
    // Connect analyser to parent component for blob visualization
    if (this.parentComponent && this.parentComponent.setupBlobAnalyser) {
      this.parentComponent.setupBlobAnalyser(this.analyser, this.dataArray)
    }
  }

  private callStore: any = null
  private parentComponent: any = null

  async start() {
    this.isPlaying = true
    this.playbackLoop()
  }

  async stop() {
    this.isPlaying = false
    if (this.audioContext) {
      await this.audioContext.close()
      this.audioContext = null
    }
  }

  enqueueAudio(audioData: ArrayBuffer) {
    this.queue.push(audioData)
    console.log(`🎵 Enqueued audio chunk: ${audioData.byteLength} bytes, queue length: ${this.queue.length}`)
    
    // Log potential audio queue buildup issues
    if (this.queue.length > 10) {
      console.warn(`⚠️ Audio queue is getting large (${this.queue.length} chunks) - possible lag`)
    }
  }

  clearQueue() {
    this.queue = []

    if(this.audioContext) {
      this.nextChunkTime = this.audioContext.currentTime
    }
    
    // Stop playing state when queue is cleared
    if (this.callStore) {
      this.callStore.setPlayingAudio(false)
    }
    
    console.log("Audio queue cleared")
  }

  private async playbackLoop() {
    while (this.isPlaying) {
      if (this.queue.length > 0 && this.audioContext) {
        const audioData = this.queue.shift()!
        await this.playChunk(audioData)
      } else {
        // Use longer intervals when no audio to reduce CPU usage
        await new Promise((resolve) => setTimeout(resolve, 50))
      }
    }
  }

  private async playChunk(audioData: ArrayBuffer): Promise<void> {
    try {
      if (!this.audioContext) return

      const int16Array = new Int16Array(audioData)
      const float32Array = new Float32Array(int16Array.length)

      // Convert Int16 to Float32 with proper scaling and clipping
      for (let i = 0; i < int16Array.length; i++) {
        // Normalize and clamp to prevent distortion
        const normalized = int16Array[i] / 32768.0
        float32Array[i] = Math.max(-1.0, Math.min(1.0, normalized))
      }

      // Create and play audio buffer
      const audioBuffer = this.audioContext.createBuffer(
        1,
        float32Array.length,
        OUTPUT_AUDIO_CONFIG.sampleRate
      )
      audioBuffer.getChannelData(0).set(float32Array)

      const source = this.audioContext.createBufferSource()
      source.buffer = audioBuffer
      
      // Add a low-pass filter to reduce high-frequency noise/buzzing
      const filter = this.audioContext.createBiquadFilter()
      filter.type = 'lowpass'
      filter.frequency.setValueAtTime(8000, this.audioContext.currentTime) // Cut frequencies above 8kHz
      filter.Q.setValueAtTime(1, this.audioContext.currentTime)
      
      // Connect: source -> filter -> analyser -> destination
      source.connect(filter)
      
      if (this.analyser) {
        filter.connect(this.analyser)
        this.analyser.connect(this.audioContext.destination)
      } else {
        filter.connect(this.audioContext.destination)
      }

      // Ensure proper timing - don't let chunks overlap or have gaps
      if (this.nextChunkTime < this.audioContext.currentTime) {
        this.nextChunkTime = this.audioContext.currentTime
      }

      // Start audio at precisely the scheduled time
      source.start(this.nextChunkTime)
      
      // Set audio playing state when audio starts
      if (this.callStore) {
        this.callStore.setPlayingAudio(true)
      }
      
      // Handle when audio finishes playing
      source.onended = () => {
        // Only stop playing state if no more audio is queued
        if (this.queue.length === 0 && this.callStore) {
          this.callStore.setPlayingAudio(false)
        }
      }
      
      // Calculate actual duration and update next chunk time
      const duration = float32Array.length / OUTPUT_AUDIO_CONFIG.sampleRate
      const timingDiff = this.nextChunkTime - this.audioContext.currentTime
      
      console.log(`🔊 Playing audio chunk: ${float32Array.length} samples, duration: ${duration.toFixed(3)}s, timing diff: ${timingDiff.toFixed(3)}s`)
      
      this.nextChunkTime += duration
      
      // No setTimeout needed - Web Audio API handles timing automatically
    } catch (error) {
      console.error("Error playing audio chunk:", error)
    }
  }
}

// Initialize audio context and analyzer with real-time processing
const initializeAudio = async (): Promise<MediaStream> => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: INPUT_AUDIO_CONFIG.sampleRate,
        channelCount: INPUT_AUDIO_CONFIG.channels,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
      },
    })

    // Create AudioContext with optimized settings for low latency
    audioContext = new AudioContext({
      sampleRate: INPUT_AUDIO_CONFIG.sampleRate,
      latencyHint: "interactive", // Prioritize low latency over power consumption
    })

    analyser = audioContext.createAnalyser()
    analyser.fftSize = 256

    const source = audioContext.createMediaStreamSource(stream)
    const processor = audioContext.createScriptProcessor(4096, 1, 1)

    source.connect(analyser)
    source.connect(processor)
    processor.connect(audioContext.destination)

    // Process audio in real-time with buffering
    processor.onaudioprocess = (event) => {
      if (
        websocket &&
        websocket.readyState === WebSocket.OPEN &&
        !callStore.isMuted &&
        callStore.isRecording
      ) {
        const inputData = event.inputBuffer.getChannelData(0)

        // Check if we're getting any audio data
        const maxSample = Math.max(...Array.from(inputData).map(Math.abs))
        if (maxSample > 0.001) {
          console.log(`📊 Audio level: ${maxSample.toFixed(4)}`)
        }

        // Add to buffer
        audioBuffer.push(new Float32Array(inputData))
        bufferSize += inputData.length

        // Send when buffer reaches target size
        if (bufferSize >= TARGET_BUFFER_SIZE) {
          sendBufferedAudioChunk()
        }
      }
    }

    return stream
  } catch (error) {
    throw new Error(`Failed to access microphone: ${error}`)
  }
}

// Send buffered audio chunk to FastAPI backend
const sendBufferedAudioChunk = () => {
  try {
    if (audioBuffer.length === 0) return

    // Combine all buffered audio into one larger chunk
    const combinedData = new Float32Array(bufferSize)
    let offset = 0

    for (const chunk of audioBuffer) {
      combinedData.set(chunk, offset)
      offset += chunk.length
    }

    // Convert Float32Array to Int16Array (PCM 16-bit)
    const pcmData = new Int16Array(combinedData.length)
    let nonZeroCount = 0

    for (let i = 0; i < combinedData.length; i++) {
      const sample = combinedData[i]
      pcmData[i] = Math.max(-32768, Math.min(32767, sample * 32767))
      if (pcmData[i] !== 0) nonZeroCount++
    }

    console.log(`Audio stats: ${nonZeroCount} non-zero samples out of ${combinedData.length} total`)

    // Don't send if all samples are zero
    if (nonZeroCount === 0) {
      console.log("Skipping audio - all samples are zero (silence)")
      audioBuffer = []
      bufferSize = 0
      return
    }

    // Convert to base64 and send to FastAPI backend
    const base64Audio = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)))
    const message = {
      type: "audio",
      data: base64Audio,
    }

    console.log(`Sending buffered audio: ${pcmData.length} samples (${pcmData.length * 2} bytes)`)
    websocket!.send(JSON.stringify(message))

    // Clear buffer
    audioBuffer = []
    bufferSize = 0
  } catch (error) {
    console.error("Error sending buffered audio chunk:", error)
  }
}

// Store reference to parent component for blob analyser connection
let parentComponentRef: any = null

// Handle audio response from FastAPI backend
const handleBackendAudioResponse = async (response: any) => {
  try {
    const audioData = base64ToArrayBuffer(response.data)

    // Initialize continuous audio player if needed
    if (!continuousAudioPlayer) {
      continuousAudioPlayer = new ContinuousAudioPlayer(callStore, parentComponentRef)
      await continuousAudioPlayer.start()
      console.log("Started continuous audio player with internal analyser")
    }

    // Add to continuous playback queue
    continuousAudioPlayer.enqueueAudio(audioData)
  } catch (error) {
    console.error("Error handling backend audio response:", error)
  }
}

// Handle text response from FastAPI backend
const handleBackendTextResponse = (response: any) => {
  try {
    console.log("Received text:", response.content)
    callStore.addAIMessage(response.content)
  } catch (error) {
    console.error("Error handling backend text response:", error)
  }
}

// Handle interruption - clear audio queue and stop playback
const handleInterruption = async () => {
  try {
    console.log("Handling interruption - clearing audio queue and stopping playback")
    
    // Clear the audio queue immediately
    if (continuousAudioPlayer) {
      continuousAudioPlayer.clearQueue()
      console.log("Audio queue cleared")
    }
    
    // Clear any local audio buffer
    audioBuffer = []
    bufferSize = 0
    console.log("Local audio buffer cleared")
    
  } catch (error) {
    console.error("Error handling interruption:", error)
  }
}

// Convert base64 to ArrayBuffer
const base64ToArrayBuffer = (base64: string): ArrayBuffer => {
  const binaryString = atob(base64)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  return bytes.buffer
}

// Setup WebSocket connection to FastAPI backend
const setupWebSocket = (): Promise<void> => {
  return new Promise((resolve, reject) => {
    // Set the call session ID to match the WebSocket connection
    // You can test with different session IDs by changing this value
    const sessionId = 2 // Change this to test different sessions (1, 2, 3, etc.)
    callStore.setCallSessionId(sessionId)
    console.log(`🔧 Using call session ID: ${sessionId}`)
    
    const wsUrl = `ws://localhost:8000/ws/${sessionId}`
    console.log("Connecting to FastAPI WebSocket:", wsUrl)

    websocket = new WebSocket(wsUrl)

    websocket.onopen = () => {
      console.log("WebSocket connected to FastAPI backend")
      callStore.connectCall()


      // No initial greeting needed - backend will trigger AI greeting automatically
      console.log("WebSocket ready - waiting for AI greeting")

      resolve()
    }

    websocket.onmessage = async (event) => {
      try {
        let messageData = event.data

        // Handle binary data (Blob)
        if (event.data instanceof Blob) {
          messageData = await event.data.text()
        }

        // Handle ArrayBuffer
        if (event.data instanceof ArrayBuffer) {
          messageData = new TextDecoder().decode(event.data)
        }

        // Parse JSON
        const response = JSON.parse(messageData)
        console.log("📨 Received WebSocket message:", response.type, response)

        // Handle different message types from FastAPI backend
        switch (response.type) {
          case "audio":
            handleBackendAudioResponse(response)
            break
          case "text":
            handleBackendTextResponse(response)
            break
          case "user_transcript":
            // Handle user speech transcription from backend
            console.log("User said:", response.content)
            callStore.addUserMessage(response.content)
            break
          case "turn_complete":
            console.log("Turn complete - ready for next input")
            break
          case "interrupted":
            console.log("Response interrupted")
            // Clear audio queue and stop playback when interrupted
            await handleInterruption()
            break
          case "error":
            callStore.setError(response.message || "Error occurred")
            break
          case "rag_function_call":
            // Log RAG function calls for debugging
            console.log("RAG function call:", response)
            break
          default:
            console.log("Unknown message type:", response.type)
        }
      } catch (error) {
        console.error("Error processing WebSocket message:", error)
        callStore.setError(`Message processing error: ${error}`)
      }
    }

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error)
      callStore.setError("WebSocket connection error")
      reject(new Error("WebSocket connection failed"))
    }

    websocket.onclose = (event) => {
      console.log("WebSocket closed:", event.code, event.reason)
      callStore.endCall()
    }
  })
}

// Setup real-time audio recording
const setupAudioRecording = async () => {
  audioStream = await initializeAudio()
  callStore.setRecording(true)
  console.log("Real-time audio processing started")
}

// Start conversation (called when component mounts and call is starting)
const startConversation = async () => {
  try {
    callStore.setStatus('connecting')
    
    await setupWebSocket()
    await setupAudioRecording()
    
    console.log("Conversation started successfully")
  } catch (error) {
    console.error("Failed to start conversation:", error)
    callStore.setError(`Failed to start conversation: ${error}`)
  }
}

// Stop conversation
const stopConversation = async () => {
  console.log("Stopping conversation...")

  // Flush any remaining buffered audio
  if (bufferSize > 0) {
    console.log("Flushing remaining audio buffer before stopping...")
    sendBufferedAudioChunk()
  }

  // Stop continuous audio player
  if (continuousAudioPlayer) {
    await continuousAudioPlayer.stop()
    continuousAudioPlayer = null
    console.log("Stopped continuous audio player")
  }

  // Stop audio stream
  if (audioStream) {
    audioStream.getTracks().forEach((track) => {
      track.stop()
      console.log("Stopped audio track")
    })
    audioStream = null
  }

  // Close audio context
  if (audioContext) {
    audioContext
      .close()
      .then(() => {
        console.log("Audio context closed")
      })
      .catch(console.error)
    audioContext = null
    analyser = null
  }

  // Close WebSocket
  if (websocket) {
    websocket.close()
    websocket = null
  }

  // Clear audio buffer
  audioBuffer = []
  bufferSize = 0

  console.log("Conversation stopped")
}

// Watch for call status changes
watch(() => callStore.status, (newStatus, oldStatus) => {
  console.log(`Call status changed: ${oldStatus} -> ${newStatus}`)
  
  if (newStatus === 'connecting' && oldStatus === 'idle') {
    // Start the conversation when status changes to connecting
    startConversation()
  } else if (newStatus === 'idle' && oldStatus !== 'idle') {
    // Stop the conversation when status changes to idle
    stopConversation()
  }
})

// Watch for mute changes
watch(() => callStore.isMuted, (isMuted) => {
  console.log(`Mute status changed: ${isMuted}`)
})

// Expose methods for parent components and accept parent ref
defineExpose({
  startConversation,
  stopConversation,
  setParentComponent: (parent: any) => {
    parentComponentRef = parent
    console.log("✅ Parent component reference set for blob analyser")
  }
})

// Lifecycle
onMounted(async () => {
  console.log("GeminiLive component mounted")
  
  // If call is already in connecting state, start immediately
  if (callStore.status === 'connecting') {
    await startConversation()
  }
  
  // Note: Test messages removed - now relying on real WebSocket messages
})

onUnmounted(() => {
  console.log("GeminiLive component unmounted")
  stopConversation()
})
</script>