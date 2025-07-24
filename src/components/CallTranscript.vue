<template>
  <div class="call-transcript">
    <div class="transcript-header">
      <h3 class="text-subtitle-1 font-weight-medium">
        Conversation Transcript
      </h3>
      <div class="message-counter">
        <v-chip size="small" variant="tonal">
          {{ callStore.messageCount }} messages
        </v-chip>
      </div>
    </div>

    <div 
      ref="messagesContainer"
      class="messages-container"
      :class="{ 'empty-state': callStore.messages.length === 0 }"
    >
      <!-- Empty state -->
      <div v-if="callStore.messages.length === 0" class="empty-transcript">
        <v-icon size="48" color="grey-lighten-1" class="mb-4">
          mdi-message-text-outline
        </v-icon>
        <p class="text-body-2 text-medium-emphasis">
          Your conversation will appear here...
        </p>
      </div>

      <!-- Messages -->
      <div
        v-for="message in callStore.messages"
        :key="message.id"
        class="message-wrapper"
        :class="{ 'user-message': message.type === 'user', 'ai-message': message.type === 'ai' }"
      >
        <div class="message-bubble" :class="`${message.type}-bubble`">
          <div class="message-content">
            <p class="message-text">{{ message.content }}</p>
            <div class="message-meta">
              <span class="message-time">
                {{ formatTime(message.timestamp) }}
              </span>
              <v-icon 
                v-if="message.type === 'user'" 
                size="14" 
                class="ml-1"
                color="white"
              >
                mdi-check
              </v-icon>
            </div>
          </div>
        </div>
      </div>

      <!-- Typing indicator -->
      <div 
        v-if="showTypingIndicator" 
        class="message-wrapper ai-message"
      >
        <div class="message-bubble ai-bubble typing-indicator">
          <div class="typing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup>
import { ref, nextTick, watch, computed } from 'vue'
import { useCallStore } from '@/stores/call'

const callStore = useCallStore()
const messagesContainer = ref<HTMLElement>()

// Props
interface Props {
  showTypingIndicator?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showTypingIndicator: false
})

// Auto-scroll to bottom when new messages arrive
watch(() => callStore.messages.length, async (newLength, oldLength) => {
  console.log(`📝 Messages changed: ${oldLength} -> ${newLength}`, callStore.messages)
  await nextTick()
  scrollToBottom()
})

// Debug: Log messages array changes
watch(() => callStore.messages, (newMessages) => {
  console.log('📝 Messages array updated:', newMessages)
}, { deep: true })

const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const formatTime = (timestamp: number) => {
  const date = new Date(timestamp)
  return date.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit',
    hour12: false 
  })
}

// Expose method to parent components
defineExpose({
  scrollToBottom
})
</script>

<style scoped>
.call-transcript {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.transcript-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid rgba(var(--v-theme-outline), 0.2);
  background: rgba(var(--v-theme-surface), 0.8);
  backdrop-filter: blur(10px);
}

.message-counter {
  opacity: 0.8;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background: rgba(var(--v-theme-background), 0.3);
  scroll-behavior: smooth;
}

.empty-transcript {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
}

.message-wrapper {
  margin-bottom: 16px;
  display: flex;
}

.user-message {
  justify-content: flex-end;
}

.ai-message {
  justify-content: flex-start;
}

.message-bubble {
  max-width: 75%;
  min-width: 120px;
  border-radius: 18px;
  position: relative;
  animation: slideIn 0.3s ease-out;
}

.user-bubble {
  background: rgb(var(--v-theme-primary));
  color: white;
  border-bottom-right-radius: 6px;
}

.ai-bubble {
  background: rgba(var(--v-theme-surface), 0.9);
  border: 1px solid rgba(var(--v-theme-outline), 0.2);
  border-bottom-left-radius: 6px;
}

.message-content {
  padding: 12px 16px;
}

.message-text {
  margin: 0;
  line-height: 1.4;
  word-wrap: break-word;
}

.message-meta {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  margin-top: 4px;
  opacity: 0.8;
}

.user-bubble .message-meta {
  color: rgba(255, 255, 255, 0.8);
}

.ai-bubble .message-meta {
  color: rgba(var(--v-theme-on-surface), 0.6);
}

.message-time {
  font-size: 0.75rem;
  font-weight: 500;
}

/* Typing indicator */
.typing-indicator {
  padding: 16px 20px;
}

.typing-dots {
  display: flex;
  gap: 4px;
}

.dot {
  width: 8px;
  height: 8px;
  background: rgba(var(--v-theme-on-surface), 0.5);
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.dot:nth-child(1) {
  animation-delay: 0s;
}

.dot:nth-child(2) {
  animation-delay: 0.2s;
}

.dot:nth-child(3) {
  animation-delay: 0.4s;
}

/* Animations */
@keyframes slideIn {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes typing {
  0%, 60%, 100% {
    transform: scale(1);
    opacity: 0.5;
  }
  30% {
    transform: scale(1.2);
    opacity: 1;
  }
}

/* Scrollbar styling */
.messages-container::-webkit-scrollbar {
  width: 6px;
}

.messages-container::-webkit-scrollbar-track {
  background: transparent;
}

.messages-container::-webkit-scrollbar-thumb {
  background: rgba(var(--v-theme-outline), 0.3);
  border-radius: 3px;
}

.messages-container::-webkit-scrollbar-thumb:hover {
  background: rgba(var(--v-theme-outline), 0.5);
}

/* Responsive design */
@media (max-width: 600px) {
  .transcript-header {
    padding: 12px 16px;
  }
  
  .messages-container {
    padding: 12px;
  }
  
  .message-bubble {
    max-width: 85%;
  }
  
  .message-content {
    padding: 10px 14px;
  }
}
</style>