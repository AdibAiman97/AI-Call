<template>
  <v-app>
    <HeaderComp :toggleDrawer="toggleDrawer" />
    <v-main>
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