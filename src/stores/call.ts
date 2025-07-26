import { defineStore } from 'pinia'
import type { CallState, CallStatus, ConversationMessage } from '@/types/call'

export const useCallStore = defineStore('call', {
  state: (): CallState => ({
    status: 'idle',
    isRecording: false,
    isMuted: false,
    isPlayingAudio: false,
    startTime: null,
    duration: 0,
    messages: [],
    error: undefined,
    callSessionId: null,
  }),

  getters: {
    isActive: (state) => state.status === 'connected',
    isConnecting: (state) => state.status === 'connecting',
    hasError: (state) => state.status === 'error',
    
    formattedDuration: (state) => {
      const hours = Math.floor(state.duration / 3600)
      const minutes = Math.floor((state.duration % 3600) / 60)
      const seconds = state.duration % 60
      
      if (hours > 0) {
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
      }
      return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
    },

    lastMessage: (state) => state.messages[state.messages.length - 1],
    messageCount: (state) => state.messages.length,
  },

  actions: {
    // Call lifecycle actions
    startCall() {
      this.status = 'connecting'
      this.startTime = Date.now()
      this.duration = 0
      this.messages = []
      this.error = undefined
      this.callSessionId = null // Clear any previous session
      this._startTimer()
    },

    connectCall() {
      this.status = 'connected'
      this.isRecording = true
    },

    endCall() {
      this.status = 'idle'
      this.isRecording = false
      this.isMuted = false
      this.isPlayingAudio = false
      // Don't reset callSessionId here - keep it for call summary
      this._stopTimer()
    },

    setError(error: string) {
      this.status = 'error'
      this.error = error
      this.isRecording = false
      this._stopTimer()
    },

    setStatus(status: CallStatus) {
      this.status = status
    },

    setCallSessionId(sessionId: number) {
      this.callSessionId = sessionId
    },

    clearCallSession() {
      this.callSessionId = null
    },

    // Audio controls
    toggleMute() {
      this.isMuted = !this.isMuted
    },

    setRecording(recording: boolean) {
      this.isRecording = recording
    },

    setPlayingAudio(playing: boolean) {
      this.isPlayingAudio = playing
    },

    // Message management
    addMessage(content: string, type: 'user' | 'ai') {
      const message: ConversationMessage = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        type,
        content: content.trim(),
        timestamp: Date.now(),
      }
      
      // Avoid duplicate messages
      if (this.lastMessage?.content !== message.content || this.lastMessage?.type !== message.type) {
        this.messages.push(message)
      }
    },

    addUserMessage(content: string) {
      this.addMessage(content, 'user')
    },

    addAIMessage(content: string) {
      this.addMessage(content, 'ai')
    },

    clearMessages() {
      this.messages = []
    },

    // Timer management
    _timerId: null as ReturnType<typeof setInterval> | null,

    _startTimer() {
      this._stopTimer() // Clear any existing timer
      this._timerId = setInterval(() => {
        if (this.startTime) {
          this.duration = Math.floor((Date.now() - this.startTime) / 1000)
        }
      }, 1000)
    },

    _stopTimer() {
      if (this._timerId) {
        clearInterval(this._timerId)
        this._timerId = null
      }
    },

    // Reset store to initial state
    reset() {
      this.status = 'idle'
      this.isRecording = false
      this.isMuted = false
      this.isPlayingAudio = false
      this.startTime = null
      this.duration = 0
      this.messages = []
      this.error = undefined
      this.callSessionId = null
      this._stopTimer()
    },
  },
})