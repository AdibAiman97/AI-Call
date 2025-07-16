<template>
  <div class="d-flex fill-height ga-2 w-100">
    <div
      class="d-flex flex-column align-center justify-center fill-height ga-6 w-100"
    >
      <!-- <div class="d-flex flex-column align-center justify-center">
        <v-avatar size="120" class="bg-primary">
          <span class="text-h3 text-background font-weight-bold">AI</span>
        </v-avatar>
      </div> -->

      <div class="d-flex flex-column align-center justify-center">
        <div class="dancing-blob-container">
          <TresCanvas :alpha="true">
            <Suspense>
              <DancingBlob 
                :analyser="analyser" 
                :dataArray="dataArray" 
                :isAudioPlaying="callStore.isPlayingAudio"
              />
            </Suspense>
          </TresCanvas>
        </div>
        <h1 class="d-flex align-center justify-center pt-2">AI Agent</h1>
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

        <v-btn
          @click="endCall()"
          to="/call-summary"
          class="bg-error"
          size="70"
          rounded="circle"
        >
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
import { Volume2, Phone } from "lucide-vue-next";
import { useCallStore } from "@/stores/call";
import {
  onMounted,
  onUnmounted,
  ref,
  watch,
  computed,
  onBeforeUnmount,
} from "vue";
import { useRouter } from "vue-router";
import { TresCanvas } from "@tresjs/core";
import DancingBlob from "@/components/DancingBlob.vue";

const callStore = useCallStore();
const router = useRouter();

const elapsedSeconds = ref(0);
let timer = null as any;

// Audio analysis for dancing blob
const analyser = ref<AnalyserNode | null>(null);
const dataArray = ref<Uint8Array | null>(null);
const audioContext = ref<AudioContext | null>(null);

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

// Setup audio analysis for dancing blob
async function setupAudioAnalysis() {
  try {
    if (!audioContext.value) {
      audioContext.value = new (window.AudioContext ||
        (window as any).webkitAudioContext)();
    }

    if (!analyser.value) {
      analyser.value = audioContext.value.createAnalyser();
      analyser.value.fftSize = 256;
      const bufferLength = analyser.value.frequencyBinCount;
      dataArray.value = new Uint8Array(bufferLength);
    }

    // Connect to audio stream if available
    if (callStore.isPlayingAudio && audioContext.value.state === "suspended") {
      await audioContext.value.resume();
    }
  } catch (error) {
    console.error("Error setting up audio analysis:", error);
  }
}

// Watch for AI speaking state changes
watch(
  () => callStore.isPlayingAudio,
  (isPlaying) => {
    if (isPlaying) {
      setupAudioAnalysis();
    }
  },
  { immediate: true }
);

onUnmounted(() => {
  if (timer) clearInterval(timer);

  // Cleanup audio context
  if (audioContext.value) {
    audioContext.value.close();
  }
});

onMounted(() => {
  startCall();
});

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

// Cleanup when component unmounts
onBeforeUnmount(() => {
  if (callStore.isInCall) {
    callStore.endCall();
  }
});
</script>

<style scoped>
.dancing-blob-container {
  width: 400px;
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 20px;
  background: transparent;
  overflow: hidden;
}

.dancing-blob-container canvas {
  background: transparent !important;
}
</style>
