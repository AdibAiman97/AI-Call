<template>
  <v-app>
    <HeaderBar :toggleDrawer="toggleDrawer" />
    <v-main class="h-fill ma-10">
      <SidePanel :open="isDrawerOpen" />
      <router-view />
    </v-main>
  </v-app>
</template>

<script setup>
import { ref } from "vue";

const isDrawerOpen = ref(false);

function toggleDrawer() {
  isDrawerOpen.value = !isDrawerOpen.value;
}
</script>