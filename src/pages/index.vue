<template>
  <div class="fill-height d-flex">
      <v-row class="text-center mb-12 d-flex align-center">
        <v-col cols="12" class="slide-up">
          <h1 class="text-h2 text-md-h1 font-weight-bold mb-3">
            <span class="text-gradient">AI-Powered</span> Call Assistant
          </h1>
          <p class="text-subtitle-1 text-md-h6 mx-auto" style="max-width: 800px;">
            Connect with our intelligent AI agents and speak just like you would with a human.
          </p>
            <v-btn 
              size="large" 
              color="primary"
              class="mt-12 text-primary gradient-wave-btn"
              to="/on-call"
              rounded
              @click="callStore.startCall()"
            >
              Start Call Now
            </v-btn>
        </v-col>
      </v-row>
  </div>
</template>

<script lang="ts" setup>
import { useCallStore } from '@/stores/call'
import { useHotkey } from '@/utils/Hotkey'
import { useRouter } from 'vue-router'

const callStore = useCallStore()
const router = useRouter()

useHotkey('Enter', () => {
  callStore.startCall()
  router.push('/on-call')
}, { ctrl: true })

</script>

<style scoped>
.text-gradient {
  background: linear-gradient(90deg, #64ffda, #0ea5e9);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  color: transparent;
}

.gradient-wave-btn {
  position: relative;
  padding: 12px 24px;
  font-weight: 600;
  /* Your solid background */
  background-color: #0f172a; 
  border: 2px solid transparent;
  border-radius: 8px;
  cursor: pointer;
  color: transparent;
  background-clip: padding-box;
  z-index: 1;

  /* Gradient border */
  --gradient: linear-gradient(90deg, #64ffda, #0ea5e9, #64ffda);
  background-image: var(--gradient);
  background-origin: border-box;
  background-clip: padding-box, border-box;
  box-shadow: 0 0 0 2px transparent;

  /* Gradient text */
  -webkit-background-clip: text;
  background-clip: text;
  background-size: 200%;
  animation: wave 3s linear infinite;
}

.gradient-wave-btn::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 10px;
  padding: 2px;
  background: linear-gradient(90deg, #64ffda, #0ea5e9, #64ffda);
  background-size: 200%;
  animation: wave 3s linear infinite;
  mask: 
    linear-gradient(#fff 0 0) content-box, 
    linear-gradient(#fff 0 0);
  mask-composite: exclude;
  -webkit-mask-composite: destination-out;
  z-index: -1;
}

@keyframes wave {
  0% {
    background-position: 0% center;
  }
  100% {
    background-position: 200% center;
  }
}
</style>