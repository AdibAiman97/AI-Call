<template>
  <v-app>
    <HeaderBar :toggleDrawer="toggleDrawer" />
    <SidePanel :open="isDrawerOpen" />
    
    <v-main class="min-h-screen ma-10">
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