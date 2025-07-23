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
            left: `${(i - 1) * 2}%`,
          }"
        >
          <!-- All blocks with smooth animation -->
          <div
            v-for="j in getTotalHeight(i)"
            :key="`block-${j}`"
            class="cube-block"
            :class="{
              'base-block': j <= getBaseHeight(i),
              'animated-block': j > getBaseHeight(i),
            }"
            :style="{
              bottom: `${(j - 1) * 22}px`,
              transitionDelay: `${j * 50}ms`,
              background: getBlockColor(j),
            }"
          ></div>
        </div>
      </div>
    </div>

    <v-row class="text-center mb-12 d-flex align-center position-relative">
      <v-col cols="12" class="slide-up mb-16 pb-16">
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
            @click="initiateCall"
            :disabled="isInitiatingCall"
          >
            <div class="call-button-content">
              <div class="call-icon" :class="{ 'connecting': isInitiatingCall }">
                <svg v-if="!isInitiatingCall" viewBox="0 0 24 24" width="24" height="24">
                  <path
                    fill="currentColor"
                    d="M6.62,10.79C8.06,13.62 10.38,15.94 13.21,17.38L15.41,15.18C15.69,14.9 16.08,14.82 16.43,14.93C17.55,15.3 18.75,15.5 20,15.5A1,1 0 0,1 21,16.5V20A1,1 0 0,1 20,21A17,17 0 0,1 3,4A1,1 0 0,1 4,3H7.5A1,1 0 0,1 8.5,4C8.5,5.25 8.7,6.45 9.07,7.57C9.18,7.92 9.1,8.31 8.82,8.59L6.62,10.79Z"
                  />
                </svg>
                <div v-else class="connecting-dots">
                  <div class="dot"></div>
                  <div class="dot"></div>
                  <div class="dot"></div>
                </div>
              </div>
              <span class="call-button-text">
                {{ isInitiatingCall ? 'Connecting...' : 'Start Call Now' }}
              </span>
              <div class="call-button-pulse"></div>
            </div>
          </button>
        </div>
      </v-col>
    </v-row>
  </div>
</template>

<script lang="ts" setup>
import { ref, onMounted, onUnmounted, nextTick, watch } from "vue";
import { useCallStore } from "@/stores/call";
import { useRouter } from "vue-router";

const callStore = useCallStore();
const router = useRouter();

// State for call initiation
const isInitiatingCall = ref(false);

// Audio context for calling sounds
let callingAudio: HTMLAudioElement | null = null;

// Initialize calling sound
const initializeCallingSound = () => {
  // Create a simple calling tone using Web Audio API
  const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
  
  // Create oscillator for short "du" sound (0.5 seconds)
  const createCallingTone = () => {
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // Set frequency for calling tone (around 440Hz - A note)
    oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
    oscillator.type = 'sine';
    
    // Create envelope for short "du" sound (0.5 seconds)
    gainNode.gain.setValueAtTime(0, audioContext.currentTime);
    gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.05); // Quick fade in
    gainNode.gain.linearRampToValueAtTime(0.1, audioContext.currentTime + 0.4);  // Hold steady
    gainNode.gain.linearRampToValueAtTime(0, audioContext.currentTime + 0.5);   // Quick fade out
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.5);
    
    return new Promise(resolve => {
      oscillator.onended = resolve;
    });
  };
  
  return createCallingTone;
};

// Play calling sound sequence (2-3 seconds total)
const playCallingSound = async () => {
  const createTone = initializeCallingSound();
  
  // Play "du du du" pattern - 3 short tones with pauses
  // Total time: 0.5s + 0.5s + 0.5s + 0.5s + 0.5s = 2.5 seconds
  for (let i = 0; i < 3; i++) {
    if (!isInitiatingCall.value) break; // Stop if call was cancelled
    
    await createTone(); // 0.5 seconds "du"
    
    // Pause between tones (except after last one)
    if (i < 2) {
      await new Promise(resolve => setTimeout(resolve, 500)); // 0.5 second pause
    }
  }
  
  // Pause before repeating the sequence
  await new Promise(resolve => setTimeout(resolve, 1000));
};

// Continuous calling sound loop
const startCallingLoop = async () => {
  while (isInitiatingCall.value) {
    await playCallingSound();
  }
};

// Handle call initiation
const initiateCall = async () => {
  isInitiatingCall.value = true;
  
  try {
    // Start calling sound loop
    const callingSoundPromise = startCallingLoop();
    
    // Start the call setup process
    callStore.startCall();
    
    // Navigate to call page
    await router.push('/on-call');
    
    // The calling sound will stop automatically when isInitiatingCall becomes false
    // This happens when the GeminiLive component establishes connection
    
  } catch (error) {
    console.error('Failed to initiate call:', error);
    isInitiatingCall.value = false;
  }
};

// Watch for call status changes to stop calling sound
watch(() => callStore.status, (newStatus) => {
  if (newStatus === 'connected') {
    // Stop calling sound when connection is established
    isInitiatingCall.value = false;
    console.log('📞 Call connected - stopping calling sound');
  }
});

// Stop calling sound when component unmounts
onUnmounted(() => {
  isInitiatingCall.value = false;
});

// Create dynamic heights for soundwave columns
const baseHeights = ref<number[]>([]);
const currentHeights = ref<number[]>([]);
const targetHeights = ref<number[]>([]);

const getBaseHeight = (index: number) => {
  return baseHeights.value[index - 1] || 1; // Convert 1-based to 0-based index
};

const getTotalHeight = (index: number) => {
  return currentHeights.value[index - 1] || 1; // Convert 1-based to 0-based index
};

const getBlockColor = (blockHeight: number) => {
  // Interpolate between #64ffda (bottom) and #0ea5e9 (top)
  // Assuming max height of 15 blocks
  const progress = Math.min((blockHeight - 1) / 14, 1); // 0 to 1

  // Bottom color: #64ffda (100, 255, 218)
  // Top color: #0ea5e9 (14, 165, 233)
  const r = Math.round(100 + (14 - 100) * progress);
  const g = Math.round(255 + (165 - 255) * progress);
  const b = Math.round(218 + (233 - 218) * progress);

  return `rgb(${r}, ${g}, ${b})`;
};

const animateToTarget = () => {
  currentHeights.value = currentHeights.value.map((current, index) => {
    const target = targetHeights.value[index] || 1;
    if (current < target) {
      return current + 1; // Add one block
    } else if (current > target) {
      return current - 1; // Remove one block
    }
    return current;
  });

  // Continuously generate new targets for columns that reached their goal
  targetHeights.value = targetHeights.value.map((target, index) => {
    const current = currentHeights.value[index];
    // If this column reached its target, give it a new random target
    if (current === target) {
      return Math.floor(Math.random() * 15) + 1; // Increased max height to 15
    }
    return target;
  });
};

onMounted(async () => {
  // Wait for DOM to be ready and apply changes immediately
  await nextTick();
  const vMain = document.querySelector(".v-main") as HTMLElement;
  if (vMain) {
    // Disable transitions temporarily
    vMain.style.transition = "none";
    vMain.classList.add("landing-page-no-margin");
    // Force reflow
    vMain.offsetHeight;
  }

  // Generate base pattern for all 50 columns
  const staticBasePattern = [];
  for (let i = 0; i < 50; i++) {
    // Create a repeating pattern
    const patternValues = [1, 1, 2, 1, 2, 3, 2, 1, 1, 2];
    staticBasePattern.push(patternValues[i % patternValues.length]);
  }
  baseHeights.value = staticBasePattern;

  // Initialize current heights to base heights
  currentHeights.value = [...staticBasePattern];

  // Initialize with random targets to start movement immediately
  const initialTargets = [];
  for (let i = 0; i < 50; i++) {
    initialTargets.push(Math.floor(Math.random() * 15) + 1); // Increased max height to 15
  }
  targetHeights.value = initialTargets;

  // Animate blocks step by step
  setInterval(animateToTarget, 120); // Fast animation step
});

onUnmounted(() => {
  // Restore margin when leaving this page
  const vMain = document.querySelector(".v-main") as HTMLElement;
  if (vMain) {
    // Disable transitions before removing class
    vMain.style.transition = "none";
    vMain.classList.remove("landing-page-no-margin");
    // Force reflow
    vMain.offsetHeight;
    // Re-enable transitions after a brief delay
    setTimeout(() => {
      if (vMain) {
        vMain.style.transition = "";
      }
    }, 50);
  }
});
</script>

<style>
/* Global CSS class for landing page margin removal */
.landing-page-no-margin.v-main.ma-10 {
  margin: 0 !important;
  padding: 0 !important;
  transition: none !important;
}

/* Disable transitions on v-main when changing routes */
.v-main {
  transition: none !important;
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
  border: 1px solid rgba(100, 255, 218, 0.4);
  border-radius: 3px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transform-origin: bottom;
  opacity: 0.8;
  transition: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
  animation: block-appear 0.15s ease-out;
}

.base-block {
  opacity: 0.4;
  border: 1px solid rgba(100, 255, 218, 0.3);
}

.animated-block {
  opacity: 0.9;
  border: 1px solid rgba(100, 255, 218, 0.6);
  box-shadow: 0 3px 6px rgba(100, 255, 218, 0.4);
}

@keyframes block-appear {
  0% {
    opacity: 0;
    transform: translateY(10px) scaleY(0.7);
  }
  100% {
    opacity: inherit;
    transform: translateY(0) scaleY(1);
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

.call-icon.connecting {
  animation: connecting-pulse 1s ease-in-out infinite;
}

.connecting-dots {
  display: flex;
  gap: 3px;
  align-items: center;
  justify-content: center;
}

.connecting-dots .dot {
  width: 4px;
  height: 4px;
  background: white;
  border-radius: 50%;
  animation: dot-pulse 1.4s ease-in-out infinite both;
}

.connecting-dots .dot:nth-child(1) {
  animation-delay: -0.32s;
}

.connecting-dots .dot:nth-child(2) {
  animation-delay: -0.16s;
}

.call-button:disabled {
  cursor: not-allowed;
  opacity: 0.8;
}

.call-button:disabled:hover {
  transform: none;
  box-shadow: 0 10px 30px rgba(100, 255, 218, 0.3);
}

@keyframes pulse-icon {
  0%,
  100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}

@keyframes connecting-pulse {
  0%,
  100% {
    transform: scale(1);
    background: linear-gradient(135deg, #64ffda, #0ea5e9);
  }
  50% {
    transform: scale(1.2);
    background: linear-gradient(135deg, #0ea5e9, #64ffda);
  }
}

@keyframes dot-pulse {
  0%,
  80%,
  100% {
    transform: scale(0);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
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
  background: radial-gradient(
    circle,
    rgba(100, 255, 218, 0.3) 0%,
    transparent 70%
  );
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
