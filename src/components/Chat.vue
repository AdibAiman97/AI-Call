<template>
  <div class="chat-container">
    <!-- Chat Header -->
    <div class="chat-header">
      <v-icon class="me-2">mdi-robot</v-icon>
      <span class="font-weight-medium">Voice Assistant</span>
      <v-spacer></v-spacer>
      <v-chip 
        :color="messages.length > 0 ? 'success' : 'grey'" 
        size="small"
        variant="flat"
      >
        {{ messages.length }} messages
      </v-chip>
    </div>

    <!-- Chat Messages Area -->
    <div class="chat-messages" ref="messagesContainer">
      <div v-if="messages.length === 0" class="empty-state">
        <v-icon size="64" color="grey">mdi-chat-outline</v-icon>
        <p class="text-grey mt-4">Start a conversation...</p>
      </div>
      
      <div v-else class="messages-list">
        <div 
          v-for="(message, index) in messages" 
          :key="index"
          class="message-wrapper"
          :class="message.role"
        >
          <!-- User Message -->
          <div v-if="message.role === 'user'" class="message-bubble user-message">
            <div class="message-content">
              {{ message.content }}
            </div>
            <div class="message-time">
              {{ formatTime(new Date()) }}
            </div>
          </div>
          
          <!-- AI Message -->
          <div v-else class="message-bubble ai-message">
            <div class="message-avatar">
              <v-avatar size="32" color="primary">
                <v-icon color="white">mdi-robot</v-icon>
              </v-avatar>
            </div>
            <div class="message-content-wrapper">
              <div class="message-content">
                {{ message.content }}
              </div>
              <div class="message-time">
                {{ formatTime(new Date()) }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, nextTick, watch, onMounted } from 'vue'
import { useChatStore } from '@/stores/chat'

// Store and reactive data
const chatStore = useChatStore()
const messagesContainer = ref(null)
const newMessage = ref('')

// Computed properties
const messages = computed(() => chatStore.messages)

// Helper function to format time
const formatTime = (date) => {
  return date.toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}

// Auto-scroll to bottom when new messages arrive
const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// Watch for new messages and auto-scroll
watch(messages, () => {
  scrollToBottom()
}, { deep: true })

// Scroll to bottom on mount
onMounted(() => {
  scrollToBottom()
})
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 50%;
  max-height: 90vh;
  border: 1px solid rgb(var(--v-theme-primary));
  border-radius: 12px;
  overflow: hidden;
  background-color: rgb(var(--v-theme-surface));
}

.chat-header {
  display: flex;
  align-items: center;
  padding: 16px;
  background-color: rgb(var(--v-theme-primary));
  border-bottom: 1px solid rgb(var(--v-theme-outline));
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  background-color: rgb(var(--v-theme-background));
  max-height: 90vh;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  text-align: center;
}

.messages-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.message-wrapper {
  display: flex;
  width: 100%;
}

.message-wrapper.user {
  justify-content: flex-end;
}

.message-wrapper.ai {
  justify-content: flex-start;
}

.message-bubble {
  max-width: 70%;
  word-wrap: break-word;
}

.user-message {
  background-color: rgb(var(--v-theme-primary));
  color: white;
  border-radius: 18px 18px 4px 18px;
  padding: 12px 16px;
  margin-left: auto;
}

.ai-message {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.message-avatar {
  flex-shrink: 0;
}

.message-content-wrapper {
  background-color: rgb(var(--v-theme-surface));
  border: 1px solid rgb(var(--v-theme-outline));
  border-radius: 18px 18px 18px 4px;
  padding: 12px 16px;
  max-width: 100%;
}

.message-content {
  font-size: 14px;
  line-height: 1.4;
  margin-bottom: 4px;
}

.message-time {
  font-size: 11px;
  opacity: 0.7;
  text-align: right;
}

.ai-message .message-time {
  text-align: left;
}

.chat-input-area {
  background-color: rgb(var(--v-theme-surface));
}

/* Scrollbar styling */
.chat-messages::-webkit-scrollbar {
  width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
  background: transparent;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: rgba(var(--v-theme-on-surface), 0.2);
  border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: rgba(var(--v-theme-on-surface), 0.3);
}

/* Animation for new messages */
.message-wrapper {
  animation: fadeInUp 0.3s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* Responsive design */
@media (max-width: 600px) {
  .message-bubble {
    max-width: 85%;
  }
  
  .chat-messages {
    padding: 12px;
  }
  
  .messages-list {
    gap: 12px;
  }
}
</style>
