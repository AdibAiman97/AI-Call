<template>
  <v-app-bar :elevation="2" class="pr-3">
    <template v-slot:prepend>
      <v-app-bar-nav-icon @click="toggleDrawer" class="text-foreground"></v-app-bar-nav-icon>
      <img width="120" src="../assets/voxis.png" alt=""/>
    </template>

    <template v-slot:append>
      <div class="d-flex align-center ga-3 bg-success px-3 rounded-xl">
        <div class="d-flex align-center">
          <v-avatar size="10" color="#34D399"></v-avatar>
        </div>
        <div class="d-flex ga-2 align-center text-caption">
          <span class="text-foreground">Call in Progress</span>
          <v-avatar size="7" color="foreground"></v-avatar>
          <span class="text-foreground">12:45</span>
        </div>
      </div>
    </template>
  </v-app-bar>
</template>

<script setup>
import { mdiMenu } from "@mdi/js";

const path = mdiMenu;

const props = defineProps({
  toggleDrawer: {
    type: Function,
    required: true,
  },
});
</script>
