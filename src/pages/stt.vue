<template>
  <div>
    <h2>Google STT Live</h2>
    <div class="d-flex ga-2 my-10 align-center">
      <v-btn @click="startCall" :disabled="callStore.isInCall"
        class="d-flex justify-center align-center bg-surface border rounded-pill py-8 px-5">
        <Phone :size="30" stroke="#2EC4B6" />
      </v-btn>
      <v-btn @click="endCall" :disabled="!callStore.isInCall"
        class="d-flex justify-center align-center bg-error border rounded-pill py-8 px-5">
        <PhoneOff :size="30" stroke="red" />
      </v-btn>
      <v-btn @click="clearAudioQueue" :disabled="callStore.audioQueue.length === 0 && !callStore.isPlayingAudio"
        class="d-flex justify-center align-center bg-warning border rounded-pill py-8 px-5" title="Clear Audio Queue">
        Clear Queue
      </v-btn>
    </div>

    <!-- Call Status -->
    <div class="bg-grey pa-2 rounded-lg mb-4">
             <div class="d-flex align-center ga-2 mb-2 text-black">
         <span>Call Status:</span>
         <v-chip color="black" size="small">
           {{ callStore.isShuttingDown ? 'Shutting down...' : (callStore.isInCall ? 'Connected' : 'Disconnected') }}
         </v-chip>
        <span v-if="callStore.isInCall && callStore.startTime" class="text-body-2">
          Started: {{ formatTime(callStore.startTime) }}
        </span>
      </div>
    </div>

    <!-- Audio Queue Status -->
    <div class="bg-grey pa-2 rounded-lg">
      <div class="d-flex align-center ga-2 mb-2 text-black">
        <span>Audio Queue Status:</span>
        <v-chip color="black" size="small">
          {{ callStore.isPlayingAudio ? 'Playing' : 'Idle' }}
        </v-chip>
        <v-chip color="black" size="small">
          {{ callStore.audioQueue.length }} in queue
        </v-chip>
      </div>

      <!-- Currently Playing -->
      <div v-if="callStore.isPlayingAudio" class="playing-now mb-2">
        <span class="text-body-2">🎵 Playing: </span>
        <span class="text-body-2 font-weight-medium">{{ callStore.currentPlayingText }}</span>
      </div>

      <!-- Queue Preview -->
      <div v-if="callStore.audioQueue.length > 0" class="queue-preview">
        <span class="text-body-2">📋 Next in queue:</span>
        <div class=" mt-1">
          <div v-for="(item, index) in callStore.audioQueue.slice(0, 3)" :key="item.id"
            class=" text-body-2 text-grey-darken-1">
            {{ index + 1 }}. {{ item.text.substring(0, 50) }}{{ item.text.length > 50 ? '...' : '' }}
          </div>
          <div v-if="callStore.audioQueue.length > 3" class="text-body-2 text-grey">
            ... and {{ callStore.audioQueue.length - 3 }} more
          </div>
        </div>
      </div>
    </div>

    <!-- Transcript Display -->
    <div class="mt-4">
      <h3>Transcript</h3>
      <div class="h-50 bg-grey pa-4 rounded">
        <p>{{ callStore.transcript || 'Transcript will appear here...' }}</p>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { Phone, PhoneOff } from "lucide-vue-next";
import { useCallStore } from '../stores/call'

// Initialize the call store
const callStore = useCallStore()

// Handle starting the call
async function startCall() {
  try {
    await callStore.startCall()
    console.log('✅ Call started from component')
  } catch (error) {
    console.error('🚫 Failed to start call:', error)
    // You could show a toast notification here
  }
}

// Handle ending the call
function endCall() {
  callStore.endCall()
  console.log('✅ Call ended from component')
}

// Handle clearing audio queue
function clearAudioQueue() {
  callStore.clearAudioQueue()
  console.log('✅ Audio queue cleared from component')
}

// Format time for display
function formatTime(date: Date): string {
  return date.toLocaleTimeString()
}

// Cleanup when component unmounts
onBeforeUnmount(() => {
  if (callStore.isInCall) {
    callStore.endCall()
  }
})
</script>

