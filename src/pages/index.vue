<template>
  <div class="fill-height d-flex">
    <v-row class="text-center mb-12 d-flex align-center">
      <v-col cols="12" class="slide-up">
        <h1 class="text-h2 text-md-h1 font-weight-bold mb-3">
          <span class="text-gradient">AI-Powered</span> Call Assistant
        </h1>
        <p class="text-subtitle-1 text-md-h6 mx-auto" style="max-width: 800px">
          Connect with our intelligent AI agents and speak just like you would
          with a human.
        </p>
        <div class="call-button-container mt-12">
          <button
            class="call-button"
            @click="callStore.startCall(); $router.push('/on-call')"
          >
            <div class="call-button-content">
              <div class="call-icon">
                <svg viewBox="0 0 24 24" width="24" height="24">
                  <path fill="currentColor" d="M6.62,10.79C8.06,13.62 10.38,15.94 13.21,17.38L15.41,15.18C15.69,14.9 16.08,14.82 16.43,14.93C17.55,15.3 18.75,15.5 20,15.5A1,1 0 0,1 21,16.5V20A1,1 0 0,1 20,21A17,17 0 0,1 3,4A1,1 0 0,1 4,3H7.5A1,1 0 0,1 8.5,4C8.5,5.25 8.7,6.45 9.07,7.57C9.18,7.92 9.1,8.31 8.82,8.59L6.62,10.79Z"/>
                </svg>
              </div>
              <span class="call-button-text">Start Call Now</span>
              <div class="call-button-pulse"></div>
            </div>
          </button>
        </div>
      </v-col>
    </v-row>
  </div>
</template>

<script lang="ts" setup>
import { useCallStore } from "@/stores/call";

const callStore = useCallStore();
</script>

<style scoped>
.text-gradient {
  background: linear-gradient(90deg, #64ffda, #0ea5e9);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
}

.call-button-container {
  display: flex;
  justify-content: center;
  align-items: center;
}

.call-button {
  position: relative;
  background: linear-gradient(135deg, #64ffda, #0ea5e9);
  border: none;
  border-radius: 50px;
  padding: 0;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 10px 30px rgba(100, 255, 218, 0.3);
  overflow: hidden;
}

.call-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 15px 40px rgba(100, 255, 218, 0.4);
}

.call-button:active {
  transform: translateY(0);
}

.call-button-content {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px 32px;
  background: rgba(15, 23, 42, 0.9);
  margin: 2px;
  border-radius: 48px;
  color: white;
  font-weight: 600;
  font-size: 1.1rem;
  z-index: 2;
}

.call-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #64ffda, #0ea5e9);
  border-radius: 50%;
  animation: pulse-icon 2s ease-in-out infinite;
}

@keyframes pulse-icon {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}

.call-button-text {
  position: relative;
  z-index: 1;
}

.call-button-pulse {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle, rgba(100, 255, 218, 0.3) 0%, transparent 70%);
  border-radius: 50%;
  transform: translate(-50%, -50%) scale(0);
  animation: pulse-wave 2s ease-out infinite;
  z-index: 0;
}

@keyframes pulse-wave {
  0% {
    transform: translate(-50%, -50%) scale(0);
    opacity: 1;
  }
  100% {
    transform: translate(-50%, -50%) scale(4);
    opacity: 0;
  }
}

@media (max-width: 768px) {
  .call-button-content {
    padding: 14px 24px;
    font-size: 1rem;
  }
  
  .call-icon {
    width: 28px;
    height: 28px;
  }
}
</style>
