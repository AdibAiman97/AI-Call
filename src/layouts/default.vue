<template>
  <v-app>
    <HeaderBar :toggleDrawer="toggleDrawer" />
    <v-main class="h-fill" :class="{ 'ma-10': !isOnCallPage, 'ma-0': isOnCallPage }">
      <SidePanel v-model:open="isDrawerOpen" />
      <router-view />
    </v-main>
  </v-app>
</template>

<script setup>
import { ref, computed } from "vue";
import { useRoute } from "vue-router";

const route = useRoute();
const isDrawerOpen = ref(false);

// Check if current page is the on-call page
const isOnCallPage = computed(() => {
  return route.path === '/on-call';
});

function toggleDrawer() {
  isDrawerOpen.value = !isDrawerOpen.value;
}
</script>