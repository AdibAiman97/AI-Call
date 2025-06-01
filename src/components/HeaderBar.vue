<template>
  <v-app-bar :elevation="2" class="px-10">
    <template v-slot:prepend>
      <v-app-bar-nav-icon
        v-if="isAdminRoute"
        @click="toggleDrawer"
        class="text-foreground mr-2 ms-0 ps-0"
        :size="40"
      ></v-app-bar-nav-icon>
      <img width="120" src="../assets/voxis.png" alt="" />
    </template>

    <template v-if="callStore.isInCall" v-slot:append>
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

    <template v-if="!callStore.isInCall" v-slot:append>
      <v-btn
        @click="toggleUserRole"
        class="text-capitalize rounded-lg border-4 border-primary text-primary"
        height="30"
        variant="outlined"
      >
        {{ currentUserRole === "Admin" ? "Customer" : "Admin" }}
      </v-btn>
    </template>
  </v-app-bar>
</template>

<script setup>
import { computed, onUnmounted, ref, watch } from "vue";
import { useCallStore } from "@/stores/call.ts";
import { useRoute, useRouter } from "vue-router";

const props = defineProps({
  toggleDrawer: {
    type: Function,
    required: true,
  },
});

const route = useRoute();

const router = useRouter();

const callStore = useCallStore();

const elapsedSeconds = ref(0);
let timer = null;

// Computed property to check if the current route is an admin route
const isAdminRoute = computed(() => {
  return route.path.startsWith("/admin");
});

// Initialize currentUserRole based on isAdminRoute's initial value
const currentUserRole = ref(isAdminRoute.value ? 'Admin' : 'Customer');

const formattedTime = computed(() => {
  //   const hours = Math.floor(elapsedSeconds.value / 3600);
  //   const minutes = Math.floor((elapsedSeconds.value % 3600) / 60);
  //   const seconds = elapsedSeconds.value % 60;
  //
  //   const hh = hours > 0 ? String(Math.floor(elapsedSeconds.value / 3600)).padStart(2, "0") + ":" : "";
  //   const mm = String(Math.floor(elapsedSeconds.value / 60)).padStart(2, "0");
  //   const ss = String(elapsedSeconds.value % 60).padStart(2, "0");
  //   return `${hh}${mm}:${ss}`;

  const min = String(Math.floor(elapsedSeconds.value / 60)).padStart(2, "0");
  const sec = String(elapsedSeconds.value % 60).padStart(2, "0");
  return `${min}:${sec}`;
});

// Function to toggle the user role
function toggleUserRole() {
  currentUserRole.value = currentUserRole.value === 'Admin' ? 'Customer' : 'Admin';
  console.log('User role switched to:', currentUserRole.value);

  if (currentUserRole.value === 'Admin') {
    router.push('/admin');
  } else {
    router.push('/');
  }
}

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

<style scoped>
/* Target the v-toolbar__prepend class within v-app-bar */
:deep(.v-toolbar__prepend) {
  /* Remove left margin */
  margin-inline-start: 0 !important;
  /* Or for older browsers/specific cases, you might use: */
  /* margin-left: 0 !important; */

  /* Remove right margin */
  margin-inline-end: 0 !important;
  /* Or for older browsers/specific cases, you might use: */
  /* margin-right: 0 !important; */

  /* If there's any padding causing issues, you can reset it too */
  padding-inline-start: 0 !important;
  padding-inline-end: 0 !important;
}

:deep(.v-toolbar__append) {
  margin-right: 0 !important;
}
</style>
