// stores/call.js
import { defineStore } from 'pinia'

export const useCallStore = defineStore('call', {
  state: () => ({
    isInCall: false,
    startTime: null as Date | null,
  }),

  actions: {
    startCall() {
      this.isInCall = true
      this.startTime = new Date()
    },
    
    endCall() {
      this.isInCall = false
      this.startTime = null
    }
  }
})
