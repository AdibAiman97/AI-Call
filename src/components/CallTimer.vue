<template>
  <div class="call-timer">
    <div class="timer-container">
      <v-icon 
        :color="timerColor" 
        size="20" 
        class="mr-2"
      >
        {{ timerIcon }}
      </v-icon>
      <span 
        class="timer-text" 
        :class="{ 'timer-pulse': callStore.isActive }"
      >
        {{ displayTime }}
      </span>
    </div>
    
    <div v-if="callStore.isActive" class="status-indicator">
      <div class="pulse-dot"></div>
      <span class="status-text">Live</span>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { computed } from 'vue'
import { useCallStore } from '@/stores/call'

const callStore = useCallStore()

const displayTime = computed(() => {
  if (callStore.status === 'idle') {
    return '00:00'
  }
  return callStore.formattedDuration
})

const timerColor = computed(() => {
  switch (callStore.status) {
    case 'connected':
      return 'success'
    case 'connecting':
      return 'warning'
    case 'error':
      return 'error'
    default:
      return 'grey'
  }
})

const timerIcon = computed(() => {
  switch (callStore.status) {
    case 'connected':
      return 'mdi-clock-outline'
    case 'connecting':
      return 'mdi-clock-fast'
    case 'error':
      return 'mdi-clock-alert-outline'
    default:
      return 'mdi-clock-outline'
  }
})
</script>

<style scoped>
.call-timer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 12px 20px;
  background: rgba(var(--v-theme-surface), 0.8);
  border-radius: 24px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(var(--v-theme-outline), 0.2);
}

.timer-container {
  display: flex;
  align-items: center;
}

.timer-text {
  font-family: 'Roboto Mono', monospace;
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: 0.5px;
  transition: all 0.3s ease;
}

.timer-pulse {
  animation: pulse 2s infinite;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  background: rgb(var(--v-theme-success));
  border-radius: 50%;
  animation: pulse-dot 2s infinite;
}

.status-text {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgb(var(--v-theme-success));
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

@keyframes pulse-dot {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.2);
  }
}

/* Responsive design */
@media (max-width: 600px) {
  .call-timer {
    padding: 8px 16px;
    gap: 12px;
  }
  
  .timer-text {
    font-size: 1rem;
  }
  
  .status-text {
    font-size: 0.7rem;
  }
}
</style>