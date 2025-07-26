<!-- Github Luckystriike: https://github.com/luckystriike22/TresJsPlayground/ -->
<script lang="ts" setup>
import { ref, computed, watch, shallowRef } from 'vue';
import { Vector2, Vector3 } from "three";
import vertexShader from "./shaders/vertex.glsl?raw";
import fragmentShader from "./shaders/fragment.glsl?raw";
import { useLoop } from "@tresjs/core";
import { OrbitControls } from "@tresjs/cientos";
import { useTheme } from 'vuetify';

const props = defineProps<{
  analyser: any;
  dataArray: any;
  isAudioPlaying?: boolean;
}>();

// ===== COLOR CONFIGURATION =====
// 🎨 EASY COLOR SWITCHING - Change the line below:
// 
// blobColor = "primary"  -> Uses your theme's primary color (current)
// blobColor = "default"  -> Uses original orange color (fallback)
//
// 👥 Team members: Just change "primary" to "default" if you prefer the original!
const blobColor = "default" as "primary" | "default"; // <-- Change this line to switch colors
// ================================

// composables
const { onBeforeRender } = useLoop();
const theme = useTheme();

// refs
const blobRef = shallowRef<any>(null);

onBeforeRender(({ elapsed }) => {
  if (blobRef.value) {
    // Always update time for gradient animations
    uniforms.value.u_time.value = elapsed;
    
    // Handle audio-based behavior
    if (props.analyser && props.dataArray && props.isAudioPlaying) {
      // AI is speaking - use audio data for blob deformation
      props.analyser?.getByteFrequencyData(props.dataArray);

      // calc average frequency
      let sum = 0;
      for (let i = 0; i < props.dataArray?.length; i++) {
        sum += props.dataArray[i];
      }

      uniforms.value.u_frequency.value =
        sum > 0 ? sum / props.dataArray?.length : 0;
      
      // Moderate rotation when AI is speaking
      blobRef.value.rotation.x += 0.01;
      blobRef.value.rotation.y += 0.008;
    } else {
      // AI is not speaking - gentle rotation and idle state
      uniforms.value.u_frequency.value = 0;
      
      // Slower, more elegant rotation when idle
      blobRef.value.rotation.x += 0.005;
      blobRef.value.rotation.y += 0.008;
      blobRef.value.rotation.z += 0.003;
    }
  }
});

// Convert theme primary color to Vector3
function hexToRgb(hex: string): Vector3 {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (result) {
    return new Vector3(
      parseInt(result[1], 16) / 255,
      parseInt(result[2], 16) / 255,
      parseInt(result[3], 16) / 255
    );
  }
  // Default to a nice blue if theme color can't be parsed
  return new Vector3(0.2, 0.6, 1.0);
}

// Color configurations
const colorConfigs = {
  primary: computed(() => {
    const themeColor = theme.current.value.colors.primary;
    return hexToRgb(themeColor);
  }),
  default: new Vector3(1.0, 0.5, 0.2) // Original orange color
};

// Get selected color based on configuration
const selectedColor = computed(() => {
  if (blobColor === "primary") {
    return colorConfigs.primary.value; // Access computed ref value directly
  } else {
    return colorConfigs.default; // Vector3 direct value
  }
});

// shader
// set props to pass into the shader
const uniforms = ref({
  u_resolution: {
    type: "V2",
    value: new Vector2(window.innerWidth, window.innerHeight),
  },
  u_time: { type: "f", value: 0.0 },
  u_frequency: { type: "f", value: 0.0 },
  u_amplitude: { type: "f", value: 0.45 },
  u_primaryColor: { type: "v3", value: selectedColor.value },
});

// Watch for color changes and update the shader
watch(selectedColor, (newColor) => {
  uniforms.value.u_primaryColor.value = newColor;
}, { deep: true });
</script>

<template>
  <TresPerspectiveCamera :position="[11, 0, 0]" />
  <OrbitControls :enable-zoom="false" :enable-pan="false" />
  <TresMesh ref="blobRef">
    <TresIcosahedronGeometry :args="[4, 80]" />
    <TresShaderMaterial
      wireframe
      :uniforms="uniforms"
      :fragment-shader="fragmentShader"
      :vertex-shader="vertexShader"
    />
  </TresMesh>
  <TresDirectionalLight :position="[1, 1, 1]" />
  <TresAmbientLight :intensity="1" />
</template>

<style scoped>
.gitBtn {
  margin-bottom: 10px;
  margin-right: 10px;
  z-index: 10;
  color: white;
}

.blobPermissionDialog {
  height: 100vh;
  justify-content: center;
  display: flex;
  background-color: #0c1a30;
  width: 100vw;
  color: white;
  font-size: x-large;
}

.blobPermissionDialog p {
  width: 700px;
}
</style>
