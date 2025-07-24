export interface ConversationMessage {
  id: string
  type: 'user' | 'ai'
  content: string
  timestamp: number
}

export type CallStatus = 'idle' | 'connecting' | 'connected' | 'disconnected' | 'error'

export interface CallState {
  status: CallStatus
  isRecording: boolean
  isMuted: boolean
  startTime: number | null
  duration: number
  messages: ConversationMessage[]
  error?: string
}