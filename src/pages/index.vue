<template>
  <div class="fill-height d-flex hero-container">
    <!-- Cube soundwave background -->
    <div class="soundwave-background">
      <div class="cube-soundwave">
        <div 
          v-for="i in 50" 
          :key="i" 
          class="soundwave-column"
          :style="{ 
            left: `${(i - 1) * 2}%`
          }"
        >
          <div 
            v-for="j in getColumnHeight(i)" 
            :key="j" 
            class="cube-block"
            :style="{ 
              bottom: `${(j - 1) * 22}px`,
              animationDelay: `${i * 0.05}s`
            }"
          ></div>
        </div>
      </div>
    </div>

    <v-row class="text-center mb-12 d-flex align-center position-relative">
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
import { ref, onMounted } from "vue";
import { useCallStore } from "@/stores/call";

const callStore = useCallStore();

// Create dynamic heights for soundwave columns
const columnHeights = ref<number[]>([]);

const getColumnHeight = (index: number) => {
  return columnHeights.value[index] || 1;
};

const updateSoundwave = () => {
  // Create more realistic speech patterns with sudden spikes and quieter moments
  const patterns = [
    // Pattern 1: Conversation start
    [1, 2, 1, 1, 4, 8, 6, 3, 2, 1, 2, 5, 9, 7, 4, 2, 1, 1, 3, 6, 8, 5, 2, 1, 1, 2, 4, 7, 9, 6, 3, 1, 2, 5, 8, 7, 4, 2, 1, 1, 3, 6, 8, 5, 2, 1, 1, 2, 4, 7],
    // Pattern 2: Excited speech
    [2, 1, 3, 7, 10, 8, 5, 3, 6, 9, 7, 4, 2, 1, 3, 8, 11, 9, 6, 4, 2, 5, 8, 10, 7, 3, 1, 2, 6, 9, 8, 5, 3, 1, 4, 7, 9, 6, 3, 2, 1, 5, 8, 7, 4, 2, 1, 3, 6, 8],
    // Pattern 3: Calm explanation
    [1, 1, 2, 4, 6, 5, 4, 3, 2, 3, 5, 6, 4, 3, 2, 1, 2, 4, 5, 4, 3, 2, 1, 2, 4, 6, 5, 3, 2, 1, 3, 5, 4, 3, 2, 1, 2, 4, 5, 4, 3, 2, 1, 2, 4, 6, 5, 3, 2, 1],
    // Pattern 4: Question and pause
    [1, 2, 4, 6, 8, 9, 7, 5, 3, 1, 1, 1, 1, 1, 2, 4, 6, 5, 3, 2, 1, 1, 1, 1, 3, 6, 8, 7, 4, 2, 1, 1, 1, 2, 5, 7, 6, 4, 2, 1, 1, 1, 1, 3, 5, 6, 4, 2, 1, 1]
  ];
  
  const randomPattern = patterns[Math.floor(Math.random() * patterns.length)];
  
  columnHeights.value = randomPattern.map(height => {
    // Add small random variations for more natural feel
    const variation = Math.floor(Math.random() * 2); // 0 or 1
    return Math.max(1, Math.min(12, height + variation));
  });
};

onMounted(() => {
  updateSoundwave();
  // Update more frequently for realistic speech patterns
  setInterval(updateSoundwave, 800);
});
</script>

<style>
/* Global style to remove layout margins specifically for landing page */
.v-main.ma-10 {
  margin: 0 !important;
  padding: 0 !important;
}
</style>

<style scoped>
.hero-container {
  position: relative;
  overflow: hidden;
}

.soundwave-background {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 1;
  pointer-events: none;
}

.cube-soundwave {
  position: relative;
  width: 100%;
  height: 100%;
}

.soundwave-column {
  position: absolute;
  width: 20px;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

.cube-block {
  position: absolute;
  width: 20px;
  height: 20px;
  background: linear-gradient(135deg, rgba(100, 255, 218, 0.5), rgba(14, 165, 233, 0.5));
  border: 1px solid rgba(100, 255, 218, 0.3);
  border-radius: 3px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  animation: cube-glow 1.6s ease-in-out infinite;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  transform-origin: bottom;
}

@keyframes cube-glow {
  0%, 100% {
    opacity: 0.3;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    transform: scaleY(1);
  }
  20% {
    opacity: 0.7;
    box-shadow: 0 4px 8px rgba(100, 255, 218, 0.4);
    transform: scaleY(1.1);
  }
  50% {
    opacity: 0.9;
    box-shadow: 0 6px 12px rgba(100, 255, 218, 0.5);
    transform: scaleY(1.05);
  }
  80% {
    opacity: 0.6;
    box-shadow: 0 4px 8px rgba(100, 255, 218, 0.3);
    transform: scaleY(1);
  }
}

.position-relative {
  position: relative;
  z-index: 2;
}

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
  
  .soundwave-column {
    width: 16px;
  }
  
  .cube-block {
    width: 16px;
    height: 16px;
  }
}
</style>
