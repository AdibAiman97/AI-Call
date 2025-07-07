// stores/call.ts
import { defineStore } from 'pinia'
import { useChatStore } from '@/stores/chat'

// Audio queue item interface
interface AudioQueueItem {
  base64Audio: string;
  text: string;
  id: string;
}

export const useCallStore = defineStore('call', {
  state: () => ({
    // Call state
    isInCall: false,
    isShuttingDown: false, // New state to track shutdown process
    startTime: null as Date | null,
    
    // WebSocket and Audio Context
    socket: null as WebSocket | null,
    audioCtx: null as AudioContext | null,
    processor: null as AudioWorkletNode | null,
    source: null as MediaStreamAudioSourceNode | null,
    stream: null as MediaStream | null,
    
    // Transcript
    transcript: '',
    
    // Audio queue management
    audioQueue: [] as AudioQueueItem[],
    isPlayingAudio: false,
    currentAudio: null as HTMLAudioElement | null,
    currentPlayingText: '',
    
    // Configuration
    url: 'localhost:8000/stt/1',
    // Phone number not needed. Profile is fetched to FE without input on Pipeline.

    sampleRate: 16000,
    
    // Shutdown timeout (in milliseconds)
    shutdownTimeout: 3000, // Wait up to 3 seconds for graceful shutdown
  }),

  actions: {
    // Start call - sets up WebSocket and audio
    async startCall() {
      try {
        this.isInCall = true
        this.isShuttingDown = false
        this.startTime = new Date()
        
        // Setup WebSocket connection
        this.socket = new WebSocket(`ws://${this.url}`)
        
        // Handle WebSocket messages
        this.socket.onmessage = (e: MessageEvent) => {
          // Debug: Log ALL incoming messages
          console.log('🔍 RAW WebSocket message received:', e.data)
          
          try {
            // Try to parse as JSON first
            const data = JSON.parse(e.data)
            console.log('🔍 Parsed JSON message:', data)
            console.log('🔍 Message type:', data.type)
            
            // Handle different message types
            if (data.type === 'tts_audio') {
              // USERS TEXT HERE
              console.log('🎵 Received TTS audio response with text:', data.text)
              console.log('🎵 Audio data length:', data.audio_data?.length || 'undefined')
              this.addToAudioQueue(data.audio_data, data.text)
              
              // Save AI response to chat history
              this.saveToChatHistory('ai', data.text)
            } else if (data.type === 'interim') {
              // Handle interim STT results
              console.log('📝 Received interim transcript:', data.transcript || data.text)
              if (data.transcript) {
                this.transcript = data.transcript
              } else if (data.text) {
                this.transcript = data.text
              }
            } else if (data.type === 'final') {
              // Handle final STT results
              console.log('📝 Received final transcript:', data.transcript || data.text)
              const finalTranscript = data.transcript || data.text
              this.transcript = finalTranscript
              
              // Save user's final transcript to chat history
              this.saveToChatHistory('user', finalTranscript)
            } else if (data.type === 'shutdown_complete') {
              // Backend confirms it's ready to close
              console.log('✅ Backend confirmed shutdown, closing connection')
              this.forceCloseConnection()
            } else if (data.type === 'final_transcript') {
              // Handle final transcript before shutdown
              console.log('📝 Received final transcript before shutdown')
              this.transcript = data.transcript
              
              // Save final transcript to chat history
              this.saveToChatHistory('user', data.transcript)
            } else {
              // Handle other message types
              console.log('❓ Received unknown message type:', data.type, data)
            }
            
          } catch (error) {
            // If it's not JSON, treat as plain text (STT response)
            console.log('📝 Received non-JSON message (treating as transcript):', e.data)
            this.transcript = e.data
            
            // Save transcript to chat history
            this.saveToChatHistory('user', e.data)
          }
        }

        this.socket.onerror = (error) => {
          console.error('🚫 WebSocket error:', error)
        }

        this.socket.onclose = (event) => {
          console.log('🔌 WebSocket connection closed:', event.code, event.reason)
          // Ensure cleanup happens even if connection closes unexpectedly
          if (this.isInCall) {
            this.cleanup()
          }
        }

        // Setup audio stream and context
        this.stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: this.sampleRate,
          },
        })

        this.audioCtx = new AudioContext({ sampleRate: this.sampleRate })
        await this.audioCtx.audioWorklet.addModule("/processor.js")
        
        this.source = this.audioCtx.createMediaStreamSource(this.stream)
        this.processor = new AudioWorkletNode(this.audioCtx, "pcm-processor")

        // Handle audio data from processor
        this.processor.port.onmessage = (e: MessageEvent) => {
          // Only send audio data if we're not shutting down
          if (this.socket && this.socket.readyState === 1 && !this.isShuttingDown) {
            this.socket.send(e.data)
          }
        }

        // Connect audio nodes
        this.source.connect(this.processor).connect(this.audioCtx.destination)
        
        console.log('✅ Call started successfully')
        
      } catch (error) {
        console.error('🚫 Error starting call:', error)
        this.endCall() // Clean up on error
        throw error
      }
    },
    
    // End call with graceful shutdown
    async endCall() {
      if (!this.isInCall || this.isShuttingDown) {
        console.log('⚠️ Call already ended or shutting down')
        return
      }

      console.log('🔄 Starting graceful shutdown...')
      this.isShuttingDown = true

      try {
        // Step 1: Stop sending new audio data by disconnecting audio nodes
        this.disconnectAudio()

        // Step 2: Send shutdown signal to backend if WebSocket is open
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
          console.log('📤 Sending shutdown signal to backend')
          this.socket.send(JSON.stringify({ 
            type: 'shutdown_request',
            message: 'Client requesting graceful shutdown'
          }))

          // Step 3: Wait for backend confirmation or timeout
          await this.waitForShutdownConfirmation()
        } else {
          // WebSocket already closed, proceed with cleanup
          this.forceCloseConnection()
        }

      } catch (error) {
        console.error('🚫 Error during graceful shutdown:', error)
        this.forceCloseConnection()
      }
    },

    // Wait for backend to confirm shutdown or timeout
    async waitForShutdownConfirmation(): Promise<void> {
      return new Promise((resolve) => {
        const timeoutId = setTimeout(() => {
          console.log('⏰ Shutdown timeout reached, forcing close')
          this.forceCloseConnection()
          resolve()
        }, this.shutdownTimeout)

        // Listen for shutdown confirmation (handled in onmessage)
        const originalOnMessage = this.socket?.onmessage
        if (this.socket) {
          this.socket.onmessage = (e: MessageEvent) => {
            // Call original handler first
            if (originalOnMessage && this.socket) {
              originalOnMessage.call(this.socket, e)
            }

            // Check if this is shutdown confirmation
            try {
              const data = JSON.parse(e.data)
              if (data.type === 'shutdown_complete') {
                clearTimeout(timeoutId)
                resolve()
              }
            } catch (error) {
              // Not JSON, ignore for shutdown purposes
            }
          }
        }
      })
    },

    // Force close connection and cleanup
    forceCloseConnection() {
      console.log('🔌 Force closing connection and cleaning up')
      
      // Close WebSocket
      if (this.socket) {
        this.socket.close(1000, 'Client shutdown')
      }

      // Cleanup all resources
      this.cleanup()
    },

    // Disconnect audio nodes but keep other resources
    disconnectAudio() {
      console.log('🔇 Disconnecting audio nodes')
      this.processor?.disconnect()
      this.source?.disconnect()
    },

    // Clean up all resources
    cleanup() {
      console.log('🧹 Cleaning up all resources')
      
      // Clear audio queue and stop current playback
      this.clearAudioQueue()
      
      // Close audio context
      this.audioCtx?.close()
      
      // Stop media stream tracks
      this.stream?.getTracks().forEach((track) => track.stop())
      
      // Reset state
      this.isInCall = false
      this.isShuttingDown = false
      this.startTime = null
      this.socket = null
      this.audioCtx = null
      this.processor = null
      this.source = null
      this.stream = null
      
      console.log('✅ Cleanup completed')
    },

    // Add audio to queue
    addToAudioQueue(base64Audio: string, text: string) {
      const id = Date.now().toString() + Math.random().toString(36).substr(2, 9)
      this.audioQueue.push({
        base64Audio,
        text,
        id
      })
      
      // AI TEXT HERE
      console.log(`🎵 Added to queue: "${text.substring(0, 30)}..." (Queue length: ${this.audioQueue.length})`)
      
      // Start processing queue if not already playing
      if (!this.isPlayingAudio) {
        this.processAudioQueue()
      }
    },

    // Process the audio queue
    async processAudioQueue() {
      if (this.audioQueue.length === 0) {
        console.log('🎵 Audio queue is empty')
        this.isPlayingAudio = false
        return
      }
      
      if (this.isPlayingAudio) {
        console.log('🎵 Already playing audio, waiting...')
        return
      }
      
      this.isPlayingAudio = true
      
      // Get the first item from the queue
      const audioItem = this.audioQueue.shift()
      if (!audioItem) {
        this.isPlayingAudio = false
        return
      }
      
      console.log(`🎵 Processing queue item: "${audioItem.text.substring(0, 30)}..." (${this.audioQueue.length} remaining)`)
      
      // Set current playing text
      this.currentPlayingText = audioItem.text
      
      try {
        await this.playAudioFromBase64(audioItem.base64Audio, audioItem.text)
      } catch (error) {
        console.error('🚫 Error playing queued audio:', error)
      }
      
      // Mark as not playing and process next item
      this.isPlayingAudio = false
      this.currentPlayingText = ''
      
      // Process next item in queue if any
      if (this.audioQueue.length > 0) {
        setTimeout(() => this.processAudioQueue(), 100) // Small delay between audio clips
      }
    },

    // Clear the audio queue
    clearAudioQueue() {
      console.log(`🗑️ Clearing audio queue (${this.audioQueue.length} items)`)
      this.audioQueue = []
      
      // Stop current audio if playing
      if (this.currentAudio) {
        this.currentAudio.pause()
        this.currentAudio.currentTime = 0
        this.currentAudio = null
      }
      
      this.isPlayingAudio = false
      this.currentPlayingText = ''
    },

    // Function to play audio from base64 data
    async playAudioFromBase64(base64Audio: string, text: string) {
      try {
        // Convert base64 to binary data
        const binaryString = atob(base64Audio)
        const bytes = new Uint8Array(binaryString.length)
        
        // Convert each character to byte
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i)
        }
        
        // Create a blob from the binary data
        const audioBlob = new Blob([bytes], { type: 'audio/wav' })
        const audioUrl = URL.createObjectURL(audioBlob)
        
        // Create and play audio element
        const audio = new Audio(audioUrl)
        this.currentAudio = audio
        
        // Return a promise that resolves when audio finishes
        return new Promise<void>((resolve, reject) => {
          // Add event listeners for debugging
          audio.onloadstart = () => console.log('🎵 Audio loading started')
          audio.oncanplaythrough = () => console.log('🎵 Audio can play through')
          audio.onplay = () => console.log(`🎵 Playing TTS for: "${text.substring(0, 30)}..."`)
          
          audio.onended = () => {
            console.log('🎵 Audio playback finished')
            // Clean up the object URL to free memory
            URL.revokeObjectURL(audioUrl)
            this.currentAudio = null
            resolve()
          }
          
          audio.onerror = (e) => {
            console.error('🚫 Audio playback error:', e)
            URL.revokeObjectURL(audioUrl)
            this.currentAudio = null
            reject(e)
          }
          
          // Play the audio
          audio.play().catch(reject)
        })
        
      } catch (error) {
        console.error('🚫 Error playing audio:', error)
        throw error
      }
    },

    // Save to chat history
    saveToChatHistory(role: 'user' | 'ai', content: string) {
      // Get the chat store instance
      const chatStore = useChatStore()
      
      // Only save non-empty content
      if (content && content.trim().length > 0) {
        chatStore.addMessage(role, content.trim())
        console.log(`💬 Saved to chat history [${role}]: "${content.substring(0, 30)}..."`)
      }
    }
  }
})
