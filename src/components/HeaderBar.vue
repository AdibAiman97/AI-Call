<template>
  <v-app-bar :elevation="2" class="pr-4">
    <template v-slot:prepend>
      <v-app-bar-nav-icon
        @click="toggleDrawer"
        class="text-foreground mx-2"
        :size="40"
      ></v-app-bar-nav-icon>
      <img width="120" src="../assets/voxis.png" alt="" />
    </template>

    <template v-if="onCall" v-slot:append>
      <div class="d-flex align-center ga-3 bg-success px-3 rounded-xl">
        <div class="d-flex align-center">
          <v-avatar size="10" color="#34D399"></v-avatar>
        </div>
        <div class="d-flex ga-2 align-center text-caption">
          <span class="text-foreground">Call in Progress</span>
          <v-avatar size="7" color="foreground"></v-avatar>
          <span class="text-foreground">{{ formattedTime }}</span>
        </div>
      </div>
    </template>
  </v-app-bar>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from "vue";

const props = defineProps({
  toggleDrawer: {
    type: Function,
    required: true,
  },
});

const elapsedSeconds = ref(0);
let timer = null;

const onCall = ref(true);

const formattedTime = computed(() => {
  const hours = Math.floor(elapsedSeconds.value / 3600);
  const minutes = Math.floor((elapsedSeconds.value % 3600) / 60);
  const seconds = elapsedSeconds.value % 60;

  const hh = hours > 0 ? String(hours).padStart(2, "0") + ":" : "";
  const mm = String(minutes).padStart(2, "0");
  const ss = String(seconds).padStart(2, "0");
  return `${hh}${mm}:${ss}`;
});

function startTimer() {
  timer = setInterval(() => {
    elapsedSeconds.value++;
  }, 1000);
}

function endCall() {
  clearInterval(timer);
  // Add any other cleanup or call termination logic here
  alert("Call ended");
}

onMounted(() => {
  startTimer();
});

onUnmounted(() => {
  clearInterval(timer);
});
</script>