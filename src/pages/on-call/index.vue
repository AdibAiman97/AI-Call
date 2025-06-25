<template>
  <div class="d-flex flex-column align-center justify-center fill-height ga-6">
    <div class="d-flex flex-column align-center justify-center">
      <v-avatar size="120" class="bg-primary">
        <span class="text-h3 text-background font-weight-bold">AI</span>
      </v-avatar>
      <h1 class="d-flex align-center justify-center pt-2">AI Agent</h1>
    </div>

    <div class="d-flex align-center justify-center">
      <AudioLines :size="50" />
      <AudioLines :size="70" />
      <AudioLines :size="50" />
      <AudioLines :size="70" />
      <AudioLines :size="50" />
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
        @click="handleEndCall"
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
</template>

<script setup lang="ts">
import { AudioLines, Volume2, Phone } from "lucide-vue-next";
import { useCallStore } from "@/stores/call";
import { onMounted, onUnmounted, ref, watch } from "vue";
import { useRouter } from "vue-router";

const callStore = useCallStore();
const router = useRouter();

const elapsedSeconds = ref(0);
let timer = null as any;

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

onUnmounted(() => {
  if (timer) clearInterval(timer);
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
