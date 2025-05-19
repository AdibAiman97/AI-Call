<template>
  <v-app-bar class="px-5" app>
    <v-col cols="1">
      <v-btn @click="toggleDrawer">
        <svg-icon type="mdi" :path="path"></svg-icon>
      </v-btn>
    </v-col>

    <v-col cols="5">
      <v-row>
        <v-col class="d-flex">
          <img width="120" src="../assets/voxis.png" alt=""/>
        </v-col>
      </v-row>
    </v-col>

    <v-col cols="6">
      <v-row class="d-flex justify-end">
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
      </v-row>
    </v-col>
  </v-app-bar>
</template>

<script setup>
import SvgIcon from "@jamescoyle/vue-icon";
import { mdiMenu } from "@mdi/js";

const path = mdiMenu;

const props = defineProps({
  toggleDrawer: {
    type: Function,
    required: true,
  },
});
</script>
