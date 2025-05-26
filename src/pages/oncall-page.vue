<template>
  <div class="d-flex flex-column align-center justify-center fill-height  
  ga-10">
    <div class="d-flex flex-column align-center justify-center">
    <v-avatar size="100" class="bg-icon">
      <span class="text-h4 text-white">AI</span>
      </v-avatar>
      <h2 class="d-flex align-center justify-center pt-2">AI Agent</h2>
    </div>

    <div class="d-flex align-center justify-center">
      <AudioLines size="50" />
      <AudioLines size="70" />
      <AudioLines size="50" />
      <AudioLines size="70" />
      <AudioLines size="50" />
      <AudioLines size="40" />
    </div>

    <div class="d-flex align-center justify-center ga-8">
      <v-btn class="bg-white" size="70" rounded="circle">
        <v-icon>
          <volume2 />
        </v-icon>
      </v-btn>

      <v-btn class="bg-red" size="70" rounded="circle">
        <v-icon>
          <phone color="black" />
        </v-icon>
      </v-btn>
    </div>
  </div>
</template>

<script setup>
import { AudioLines, Volume2, Phone } from "lucide-vue-next";
</script>
