<template>
  <div class="chat-container">
    <!-- Chat Header -->
    <div class="chat-header">
      <v-icon class="me-2 text-white">mdi-robot</v-icon>
      <span class="font-weight-medium text-white">Voice Assistant</span>
      <v-spacer />
      <v-chip
        :color="messages.length > 0 ? 'success' : 'grey'"
        size="small"
        variant="flat"
      >
        {{ messages.length }} messages
      </v-chip>
    </div>

    <!-- Chat Messages Area -->
    <div ref="messagesContainer" class="chat-messages">
      <div v-if="messages.length === 0" class="empty-state">
        <v-icon color="grey" size="64">mdi-chat-outline</v-icon>
        <p class="text-grey mt-4">Start a conversation...</p>
      </div>

      <div v-else class="messages-list">
        <div
          v-for="message in messages"
          :key="message.id"
          class="message-wrapper"
          :class="message.type"
        >
          <!-- User Message -->
          <div v-if="message.type === 'user'" class="message-bubble user-message">
            <div class="message-header">
              <v-icon class="mr-1" size="16">mdi-microphone</v-icon>
              <span class="message-label">You said</span>
            </div>
            <div class="message-content">
              {{ message.content }}
            </div>
            <div class="message-time">
              {{ formatTime(new Date(message.timestamp)) }}
            </div>
          </div>

          <!-- AI Message -->
          <div v-else class="message-bubble ai-message">
            <div class="message-avatar">
              <v-avatar color="surface" size="32">
                <v-icon color="white">mdi-robot</v-icon>
              </v-avatar>
            </div>
            <div class="message-content-wrapper">
              <div class="message-header">
                <v-icon class="mr-1" size="16">mdi-robot</v-icon>
                <span class="message-label">AI Assistant</span>
              </div>
              <div class="message-content">
                {{ message.content }}
              </div>
              <div class="message-time">
                {{ formatTime(new Date(message.timestamp)) }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
  import { computed, nextTick, onMounted, ref, watch } from 'vue'
  import { useCallStore } from '@/stores/call'

  // Store and reactive data
  const callStore = useCallStore()
  const messagesContainer = ref<HTMLDivElement | null>(null)

  // Computed properties
  const messages = computed(() => callStore.messages)

  // Helper function to format time
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
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
  border: 1px solid rgb(var(--v-theme-surface));
  border-radius: 12px;
  overflow: hidden;
  background-color: rgb(var(--v-theme-surface));
}

.chat-header {
  display: flex;
  align-items: center;
  padding: 16px;
  background-color: rgb(var(--v-theme-surface));
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
  border: 1px solid rgba(var(--v-theme-primary), 0.3);
}

.user-message .message-header {
  color: rgba(255, 255, 255, 0.8);
}

.user-message .message-label {
  color: rgba(255, 255, 255, 0.9);
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

.message-header {
  display: flex;
  align-items: center;
  margin-bottom: 6px;
  opacity: 0.7;
}

.message-label {
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
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
