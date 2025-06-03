<template>
  <div class="d-flex flex-column align-center justify-center fill-height ga-6">
    <div class="d-flex flex-column align-center justify-center">
      <v-avatar size="120" class="bg-primary">
        <span class="text-h3 text-background font-weight-bold">AI</span>
      </v-avatar>
      <h1 class="d-flex align-center justify-center pt-2">AI Agent</h1>
    </div>

    <div class="d-flex align-center justify-center">
      <AudioLines size="50" />
      <AudioLines size="70" />
      <AudioLines size="50" />
      <AudioLines size="70" />
      <AudioLines size="50" />
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
        @click="callStore.endCall()"
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

<script setup>
import { AudioLines, Volume2, Phone } from "lucide-vue-next";
import { useCallStore } from "@/stores/call";
import { onUnmounted, ref, watch } from "vue";

const callStore = useCallStore();

const elapsedSeconds = ref(0);
let timer = null;

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

onUnmounted(() => {
  if (timer) clearInterval(timer);
});
</script>
