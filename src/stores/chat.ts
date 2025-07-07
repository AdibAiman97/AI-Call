import { defineStore } from 'pinia'

interface ChatMessage {
  role: 'user' | 'ai'
  content: string
}

export const useChatStore = defineStore('chat', {
  state: () => ({
    messages: [] as ChatMessage[],
  }),
  actions: {
    addMessage(role: 'user' | 'ai', content: string) {
      this.messages.push({ role, content })
    },
  },
})