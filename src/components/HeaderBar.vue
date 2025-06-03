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
        </div>
      </div>
    </template>
  </v-app-bar>
</template>

<script setup>
import { computed } from "vue";
import { useRoute } from "vue-router";
import { useCallStore } from "../stores/call";

const callStore = useCallStore();

const props = defineProps({
  toggleDrawer: {
    type: Function,
    required: true,
  },
});

const route = useRoute();

// Computed property to check if the current route is an admin route
const isAdminRoute = computed(() => {
  return route.path.startsWith("/admin");
});
</script>

<style scoped>
/* Target the v-toolbar__prepend class within v-app-bar */
:deep(.v-toolbar__prepend) {
  margin-inline-start: 0 !important;
  margin-inline-end: 0 !important;
  padding-inline-start: 0 !important;
  padding-inline-end: 0 !important;
}

:deep(.v-toolbar__append) {
  margin-right: 0 !important;
}
</style>
