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
// import { useRouter } from "vue-router";
import { TresCanvas } from "@tresjs/core";
import DancingBlob from "@/components/DancingBlob.vue";
import { useHotkey } from '@/utils/Hotkey'
import { useRouter } from 'vue-router'

const router = useRouter()

useHotkey('g', () => {
  console.log('call-summary')
  router.push('/call-summary')
}, { shift: true, command: true })

const callStore = useCallStore();
// const router = useRouter();

const elapsedSeconds = ref(0);
let timer = null as any;

// Audio analysis for dancing blob
const analyser = ref<AnalyserNode | null>(null);
const dataArray = ref<Uint8Array | null>(null);
const audioContext = ref<AudioContext | null>(null);
const audioSource = ref<MediaElementAudioSourceNode | AudioBufferSourceNode | MediaStreamAudioSourceNode | null>(null);

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
      analyser.value.smoothingTimeConstant = 0.8;
      const bufferLength = analyser.value.frequencyBinCount;
      dataArray.value = new Uint8Array(bufferLength);
    }

    // Resume audio context if suspended
    if (audioContext.value.state === "suspended") {
      await audioContext.value.resume();
    }

    // Try to connect to any existing audio elements
    connectToAudioElements();
  } catch (error) {
    console.error("Error setting up audio analysis:", error);
  }
}

// Connect analyser to actual audio elements being played
function connectToAudioElements() {
  try {
    if (!audioContext.value || !analyser.value) return;

    // Find any audio elements that might be playing TTS
    const audioElements = document.querySelectorAll('audio');
    
    audioElements.forEach((audio) => {
      if (!audio.paused && audio.currentTime > 0) {
        // This audio element is playing, connect it to our analyser
        if (!audioSource.value) {
          audioSource.value = audioContext.value!.createMediaElementSource(audio);
          audioSource.value.connect(analyser.value!);
          audioSource.value.connect(audioContext.value!.destination);
          console.log("✅ Connected audio analysis to playing audio element");
        }
      }
    });

    // Also try to connect to the default audio output
    if (!audioSource.value) {
      // If no audio elements found, try to get user media for analysis
      // This is a fallback for when audio is played through other means
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then((stream) => {
          if (audioContext.value && analyser.value && !audioSource.value) {
            const mediaStreamSource = audioContext.value.createMediaStreamSource(stream);
            audioSource.value = mediaStreamSource;
            mediaStreamSource.connect(analyser.value);
            console.log("✅ Connected audio analysis to media stream");
          }
        })
        .catch((error) => {
          console.log("⚠️ Could not connect to audio stream:", error.message);
        });
    }
  } catch (error) {
    console.error("Error connecting to audio elements:", error);
  }
}

// Handle Base64 PCM audio from Gemini Live API
function processGeminiAudioData(base64Data: string) {
  try {
    if (!audioContext.value || !analyser.value) {
      setupAudioAnalysis();
      return;
    }

    // Decode Base64 to ArrayBuffer
    const binaryString = atob(base64Data);
    const arrayBuffer = new ArrayBuffer(binaryString.length);
    const uint8Array = new Uint8Array(arrayBuffer);
    
    for (let i = 0; i < binaryString.length; i++) {
      uint8Array[i] = binaryString.charCodeAt(i);
    }

    // Convert 16-bit PCM to AudioBuffer
    // Gemini Live: 16-bit PCM, 24kHz, Mono
    const audioBuffer = audioContext.value.createBuffer(1, uint8Array.length / 2, 24000);
    const channelData = audioBuffer.getChannelData(0);
    
    // Convert 16-bit PCM bytes to float32 samples
    for (let i = 0; i < channelData.length; i++) {
      const sample = (uint8Array[i * 2] | (uint8Array[i * 2 + 1] << 8));
      // Convert from 16-bit signed integer to float32 (-1 to 1)
      channelData[i] = sample < 32768 ? sample / 32768 : (sample - 65536) / 32768;
    }

    // Create and play audio buffer
    const bufferSource = audioContext.value.createBufferSource();
    bufferSource.buffer = audioBuffer;
    
    // Connect to analyser for blob animation
    bufferSource.connect(analyser.value);
    bufferSource.connect(audioContext.value.destination);
    
    bufferSource.start();
    console.log("✅ Playing Gemini Live audio with blob analysis");
    
  } catch (error) {
    console.error("Error processing Gemini audio data:", error);
  }
}

// Expose function to call store for audio processing
(window as any).processGeminiAudioData = processGeminiAudioData;

// Watch for AI speaking state changes
watch(
  () => callStore.isPlayingAudio,
  (isPlaying) => {
    if (isPlaying) {
      setupAudioAnalysis();
      // Try to reconnect to audio elements when playback starts
      setTimeout(connectToAudioElements, 100);
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
