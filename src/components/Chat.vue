<template>
  <div class="chat-container">
    <!-- Chat Header -->
    <div class="chat-header">
      <div class="header-content">
        <div class="header-title-section">
          <div class="ai-indicator">
            <v-avatar size="28" color="#6c757d" class="ai-avatar">
              <v-icon size="16" color="white">mdi-robot</v-icon>
            </v-avatar>
            <div class="title-text">
              <span class="header-title">AI Transcript</span>
              <span class="header-subtitle">Live conversation</span>
            </div>
          </div>
        </div>

        <div class="header-stats d-flex justify-center align-center">
          <div class="response-counter" :data-empty="messages.length === 0">
            <v-icon size="12" class="counter-icon">mdi-message-text</v-icon>
            <span class="counter-text">{{ messages.length }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Chat Messages Area -->
    <div ref="messagesContainer" class="chat-messages">
      <div v-if="messages.length === 0" class="empty-state">
        <v-icon color="grey" size="64">mdi-robot-outline</v-icon>
        <p class="text-grey mt-4">Waiting for AI response...</p>
      </div>

      <div v-else class="messages-list">
        <div
          v-for="message in messages"
          :key="message.id"
          class="message-wrapper ai-right"
        >
          <!-- AI Message from Right -->
          <div class="message-bubble ai-message-right">
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
            <div class="message-avatar">
              <v-avatar color="primary" size="32">
                <v-icon color="white">mdi-robot</v-icon>
              </v-avatar>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from "vue";
import { useCallStore } from "@/stores/call";

// Store and reactive data
const callStore = useCallStore();
const messagesContainer = ref<HTMLDivElement | null>(null);

// Computed properties - show only AI messages
const messages = computed(() =>
  callStore.messages.filter((msg) => msg.type === "ai")
);

// Helper function to format time
const formatTime = (date: Date) => {
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
};

// Auto-scroll to bottom when new messages arrive
const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
    }
  });
};

// Watch for new messages and auto-scroll
watch(
  messages,
  () => {
    scrollToBottom();
  },
  { deep: true }
);

// Scroll to bottom on mount
onMounted(() => {
  scrollToBottom();
});
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  width: 100%;
  background-color: transparent;
  overflow: hidden;
}

.chat-header {
  margin: 12px;
  margin-bottom: 8px;
  background-color: rgb(var(--v-theme-surface));
  border: 1px solid transparent;
  border-radius: 12px;
  flex-shrink: 0;
  position: relative;
}

.header-content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: rgba(var(--v-theme-on-surface), 0.05);
  border-radius: 12px;
  padding: 16px 20px;
  margin-top: 10px;
}

.header-title-section {
  display: flex;
  align-items: center;
}

.ai-indicator {
  display: flex;
  align-items: center;
  gap: 12px;
}

.ai-avatar {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.title-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.header-title {
  font-size: 16px;
  font-weight: 600;
  color: rgb(var(--v-theme-on-surface));
  line-height: 1.2;
}

.header-subtitle {
  font-size: 12px;
  color: rgba(var(--v-theme-on-surface), 0.7);
  font-weight: 400;
}

.header-stats {
  /* display: flex; */
  align-items: center;
}

.response-counter {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 12px;
  border-radius: 16px;
  background-color: rgba(var(--v-theme-success), 0.1);
  /* border: 1px solid rgba(var(--v-theme-success), 0.2); */
  min-width: 50px;
  justify-content: center;
}

.response-counter[data-empty="true"] {
  background-color: rgba(var(--v-theme-on-surface), 0.1);
  border-color: rgba(var(--v-theme-on-surface), 0.2);
}

.counter-icon {
  color: rgba(var(--v-theme-success), 0.8);
}

.response-counter[data-empty="true"] .counter-icon {
  color: rgba(var(--v-theme-on-surface), 0.6);
}

.counter-text {
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.5px;
  color: rgba(var(--v-theme-success), 0.9);
}

.response-counter[data-empty="true"] .counter-text {
  color: rgba(var(--v-theme-on-surface), 0.7);
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background-color: rgb(var(--v-theme-surface));
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
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

.message-wrapper.ai-right {
  justify-content: flex-end;
}

.message-bubble {
  max-width: 80%;
  word-wrap: break-word;
}

.ai-message-right {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  flex-direction: row-reverse;
}

.message-avatar {
  flex-shrink: 0;
}

.message-content-wrapper {
  background-color: rgb(var(--v-theme-primary));
  color: black;
  border-radius: 18px 4px 18px 18px;
  padding: 12px 16px;
  max-width: 100%;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.message-header {
  display: flex;
  align-items: center;
  margin-bottom: 6px;
  opacity: 0.9;
}

.message-header .v-icon {
  color: rgba(255, 255, 255, 0.9);
}

.message-label {
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: black;
}

.message-content {
  font-size: 14px;
  line-height: 1.4;
  margin-bottom: 4px;
  color: black;
}

.message-time {
  font-size: 11px;
  opacity: 0.8;
  text-align: right;
  color: black;
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
  animation: fadeInRight 0.3s ease-out;
}

@keyframes fadeInRight {
  from {
    opacity: 0;
    transform: translateX(20px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* Responsive design */
@media (max-width: 768px) {
  .message-bubble {
    max-width: 90%;
  }

  .chat-messages {
    padding: 16px;
  }

  .messages-list {
    gap: 12px;
  }

  .chat-header {
    margin: 8px;
    margin-bottom: 6px;
  }

  .header-content {
    padding: 12px 16px;
  }

  .ai-indicator {
    gap: 10px;
  }

  .ai-avatar {
    width: 24px !important;
    height: 24px !important;
  }

  .header-title {
    font-size: 14px;
  }

  .header-subtitle {
    font-size: 11px;
  }
}
</style>
