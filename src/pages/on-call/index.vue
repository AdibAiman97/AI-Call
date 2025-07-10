<template>
  <div class="d-flex fill-height ga-2 w-100">
    <div class="d-flex flex-column align-center justify-center fill-height ga-6 w-100">
      <div class="d-flex flex-column align-center justify-center">
        <v-avatar size="120" class="bg-primary">
          <span class="text-h3 text-background font-weight-bold">AI</span>
        </v-avatar>
        <h1 class="d-flex align-center justify-center pt-2">AI Agent</h1>
      </div>

      <div class="d-flex align-center justify-center">
        <div class="soundwave-container">
          <div 
            v-for="(height, index) in waveHeights" 
            :key="index" 
            class="soundwave-bar"
            :style="{ 
              height: callStore.isPlayingAudio ? `${height}px` : '8px',
              animationDelay: `${index * 0.1}s` 
            }"
          />
        </div>
      </div>

      <div class="mb-4">
        <h1 class="d-flex align-center justify-center">{{ formattedTime }}</h1>
        <p>Call Duration</p>
      </div>

      <div class="d-flex align-center justify-center ga-8">
        <v-btn class="bg-foreground" size="70" rounded="circle">
          <v-icon>
            <volume2 />
          </v-icon>
        </v-btn>

        <v-btn @click="endCall()" to="/call-summary" class="bg-error" size="70" rounded="circle">
          <v-icon>
            <phone color="white" />
          </v-icon>
        </v-btn>
      </div>
    </div>

    <Chat />
  </div>
</template>

<script setup lang="ts">
import { AudioLines, Volume2, Phone } from "lucide-vue-next";
import { useCallStore } from "@/stores/call";
import { onMounted, onUnmounted, ref, watch, computed, onBeforeUnmount } from "vue";
import { useRouter } from "vue-router";

const callStore = useCallStore();
const router = useRouter();

const elapsedSeconds = ref(0);
let timer = null as any;

// Soundwave animation data
const waveHeights = ref([30, 50, 70, 50, 30, 60, 40, 80, 35, 55]);
let animationInterval: any = null;

const formattedTime = computed(() => {
  const hours = Math.floor(elapsedSeconds.value / 3600);

  const hh =
    hours > 0
      ? String(Math.floor(elapsedSeconds.value / 3600)).padStart(2, "0") + ":"
      : "";
  const mm = String(Math.floor(elapsedSeconds.value / 60)).padStart(2, "0");
  const ss = String(elapsedSeconds.value % 60).padStart(2, "0");
  return `${hh}${mm}:${ss}`;
});

const handleEndCall = async () => {
  try {
    const savedSession = await callStore.endCall();
    // Navigate to call summary with the saved session ID
    if (savedSession && savedSession.id) {
      await router.push(`/call-summary`);
    } else {
      await router.push("/call-summary");
    }
  } catch (error) {
    console.error("Error ending call:", error);
    // Still navigate to call summary even if save fails
    await router.push("/call-summary");
  }
};

// Watch when the call starts/stops
watch(
  () => callStore.isInCall,
  (inCall) => {
    if (inCall) {
      elapsedSeconds.value = 0;
      timer = setInterval(() => {
        elapsedSeconds.value++;
      }, 1000);
    } else {
      if (timer) clearInterval(timer);
      timer = null;
      elapsedSeconds.value = 0;
    }
  },
  { immediate: true }
);

// Watch for AI speaking state changes
watch(
  () => callStore.isPlayingAudio,
  (isPlaying) => {
    if (isPlaying) {
      startWaveAnimation();
    } else {
      stopWaveAnimation();
    }
  },
  { immediate: true }
);

function startWaveAnimation() {
  if (animationInterval) return;
  
  animationInterval = setInterval(() => {
    waveHeights.value = waveHeights.value.map(() => 
      Math.random() * 60 + 20 // Random height between 20-80px
    );
  }, 200);
}

function stopWaveAnimation() {
  if (animationInterval) {
    clearInterval(animationInterval);
    animationInterval = null;
  }
}

onUnmounted(() => {
  if (timer) clearInterval(timer);
  stopWaveAnimation();
});

onMounted(() => {
  startCall()
})

async function startCall() {
  try {
    await callStore.startCall();
    console.log("✅ Call started from component");
  } catch (error) {
    console.error("🚫 Failed to start call:", error);
    // You could show a toast notification here
  }
}

// Handle ending the call
function endCall() {
  callStore.endCall();
  console.log("✅ Call ended from component");
}

// Handle clearing audio queue
function clearAudioQueue() {
  callStore.clearAudioQueue();
  console.log("✅ Audio queue cleared from component");
}

// Format time for display
function formatTime(date: Date): string {
  return date.toLocaleTimeString();
}

// Cleanup when component unmounts
onBeforeUnmount(() => {
  if (callStore.isInCall) {
    callStore.endCall();
  }
});
</script>

<style scoped>
.soundwave-container {
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 4px;
  height: 100px;
  padding: 20px;
}

.soundwave-bar {
  width: 6px;
  background: linear-gradient(to top, rgb(var(--v-theme-primary)), rgb(var(--v-theme-primary)), rgba(var(--v-theme-primary), 0.7));
  border-radius: 3px;
  transition: height 0.2s ease-in-out;
  box-shadow: 
    0 2px 8px rgba(var(--v-theme-primary), 0.3),
    0 0 20px rgba(var(--v-theme-primary), 0.1);
  animation: pulse 1s ease-in-out infinite alternate;
  transform-origin: bottom;
}

.soundwave-bar:nth-child(even) {
  animation-direction: alternate-reverse;
}

@keyframes pulse {
  0% {
    box-shadow: 
      0 2px 8px rgba(var(--v-theme-primary), 0.3),
      0 0 20px rgba(var(--v-theme-primary), 0.1);
    transform: scaleY(1);
  }
  100% {
    box-shadow: 
      0 4px 16px rgba(var(--v-theme-primary), 0.5),
      0 0 30px rgba(var(--v-theme-primary), 0.3);
    transform: scaleY(1.1);
  }
}

/* Add a subtle glow effect when AI is speaking */
.soundwave-container:has(.soundwave-bar[style*="80px"]) {
  filter: drop-shadow(0 0 10px rgba(var(--v-theme-primary), 0.4));
}
</style>
