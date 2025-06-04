<template>
  <v-app>
    <v-main>
      <v-container class="fill-height">
        <v-responsive class="align-center text-center fill-height">
          <h1 class="text-h2 font-weight-bold mb-5">AI Voice Agent</h1>
          <v-textarea
            v-model="inputText"
            label="Enter your message"
            variant="outlined"
            rows="5"
            clearable
            auto-grow
            class="mb-4"
            :disabled="isSynthesizing"
          ></v-textarea>

          <v-select
            v-model="selectedVoice"
            :items="voices"
            item-title="name"
            item-value="value"
            label="Select Voice"
            variant="outlined"
            class="mb-4"
            :disabled="isSynthesizing"
          ></v-select>

          <v-btn
            color="primary"
            size="large"
            variant="elevated"
            @click="synthesizeAndPlay"
            :loading="isSynthesizing"
            :disabled="!inputText || isSynthesizing"
          >
            <v-icon left>mdi-microphone</v-icon>
            Ask LLM & Speak
          </v-btn>

          <v-card v-if="llmResponseText" class="mt-8 pa-4 text-left">
            <v-card-title>LLM Response:</v-card-title>
            <v-card-text>
              <pre style="white-space: pre-wrap; font-family: inherit;">{{ llmResponseText }}</pre>
              <v-progress-linear
                v-if="isSynthesizing && !isPlaying && audioQueue.length === 0"
                indeterminate
                color="primary"
                class="mt-4"
              ></v-progress-linear>
              </v-card-text>
          </v-card>

          <v-alert
            v-if="error"
            type="error"
            class="mt-4"
            icon="mdi-alert-circle-outline"
            closable
            @click:close="error = null"
          >
            {{ error }}
          </v-alert>
        </v-responsive>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue';

const inputText = ref('');
const isSynthesizing = ref(false); // Indicates if LLM is generating and TTS is processing
const error = ref(null);

const ws = ref(null);

const audioContext = ref(null);
const audioQueue = ref([]);

const isPlaying = ref(false); // Indicates if audio is actively playing
const llmResponseText = ref(''); // To display LLM's streaming text response

const voices = ref([
  { name: 'US English Male (Standard)', value: 'en-US-Standard-C' },
  { name: 'US English Female (Standard)', value: 'en-US-Standard-A' },
  { name: 'US English Male (Wavenet)', value: 'en-US-Wavenet-F' },
  { name: 'US English Female (Wavenet)', value: 'en-US-Wavenet-E' },
  { name: 'British English Male (Wavenet)', value: 'en-GB-Wavenet-B' },
  { name: 'British English Female (Wavenet)', value: 'en-GB-Wavenet-A' },
]);
const selectedVoice = ref('en-US-Standard-C');

// Initialize Web Audio API
const initAudioContext = () => {
  if (!audioContext.value) {
    audioContext.value = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.value.state === 'suspended') {
      audioContext.value.resume().then(() => {
        console.log('AudioContext resumed successfully');
      }).catch(e => console.error("Error resuming AudioContext:", e));
    }
  }
};

// Function to process and play audio chunks from the queue
const processAudioQueue = async () => {
  if (isPlaying.value || audioQueue.value.length === 0 || !audioContext.value) {
    return;
  }

  isPlaying.value = true;
  const audioBytes = audioQueue.value.shift(); // This is a Uint8Array

  try {
    if (audioContext.value.state === 'suspended') {
        await audioContext.value.resume();
    }

    const sampleRate = audioContext.value.sampleRate; // Use the actual AudioContext sample rate
    const numberOfChannels = 1; // LINEAR16 from Google Cloud TTS is typically mono

    const pcmSamples = new Float32Array(audioBytes.length / 2);
    const dataView = new DataView(audioBytes.buffer);

    for (let i = 0; i < pcmSamples.length; i++) {
      const s = dataView.getInt16(i * 2, true);
      pcmSamples[i] = s / 32768.0;
    }

    const audioBuffer = audioContext.value.createBuffer(
      numberOfChannels,
      pcmSamples.length,
      sampleRate
    );
    audioBuffer.copyToChannel(pcmSamples, 0);

    const source = audioContext.value.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.value.destination);

    source.onended = () => {
      isPlaying.value = false;
      if (audioQueue.value.length > 0) {
        processAudioQueue(); // Play the next chunk if available
      } else {
        // If queue is empty AND backend has signaled stream_end, then it's fully done.
        // If isSynthesizing is still true here, it means LLM/TTS is still processing
        // and more audio is expected.
        if (!isSynthesizing.value) { // This check should happen *after* onmessage has processed stream_end
            console.log("All audio played and LLM response considered complete.");
        }
      }
    };

    source.start(0); // Play immediately
  } catch (e) {
    console.error("Error processing audio data:", e);
    error.value = "Failed to play audio. Please try again.";
    isPlaying.value = false;
    isSynthesizing.value = false; // Set to false on playback error
    audioQueue.value = [];
  }
};

// Connect to WebSocket
const connectWebSocket = () => {
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    return;
  }
  const wsUrl = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000/ws/llm-tts'; // Ensure correct WS endpoint
  ws.value = new WebSocket(wsUrl);

  ws.value.onopen = () => {
    console.log('WebSocket connected.');
    error.value = null;
  };

  ws.value.onmessage = async (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.error) {
        error.value = `Backend Error: ${data.error}`;
        isSynthesizing.value = false;
        isPlaying.value = false;
        audioQueue.value = [];
        console.error("Backend Error:", data.error);
      } else if (data.text_chunk) { // This is for LLM streaming text (not currently sent by main.py)
        llmResponseText.value += data.text_chunk;
      } else if (data.stream_end) { // Signal from backend that TTS stream is finished
        isSynthesizing.value = false; // TTS is finished
        console.log("TTS stream ended from backend.");
        // If queue is empty and not playing, then it's fully done.
        if (audioQueue.value.length === 0 && !isPlaying.value) {
            console.log("All audio played and TTS stream officially complete.");
        }
      } else {
          // If it's JSON, but not a known key (like error, text_chunk, stream_end), log it
          console.warn("Unknown JSON message from WebSocket:", data);
      }
    } catch (e) {
      // If it's not JSON, assume it's base64 audio
      const base64Audio = event.data;
      try {
        const binaryString = window.atob(base64Audio);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        audioQueue.value.push(bytes);
        processAudioQueue();
      } catch (innerError) {
        console.error("Error processing audio data:", innerError);
        error.value = "Error receiving audio data.";
        isSynthesizing.value = false;
        isPlaying.value = false;
        audioQueue.value = [];
      }
    }
  };

  ws.value.onclose = () => {
    console.log('WebSocket disconnected.');
    isSynthesizing.value = false;
    isPlaying.value = false;
    audioQueue.value = [];
    llmResponseText.value = '';
  };

  ws.value.onerror = (err) => {
    console.error('WebSocket error:', err);
    error.value = 'WebSocket connection error. Please ensure backend is running.';
    ws.value.close();
    isSynthesizing.value = false;
    isPlaying.value = false;
    audioQueue.value = [];
    llmResponseText.value = '';
  };
};

const synthesizeAndPlay = async () => {
  if (!inputText.value.trim()) {
    error.value = 'Please enter a message for the LLM.';
    return;
  }

  isSynthesizing.value = true;
  error.value = null;
  llmResponseText.value = ''; // Clear previous LLM response text
  audioQueue.value = [];
  isPlaying.value = false;

  initAudioContext();

  if (ws.value.readyState !== WebSocket.OPEN) {
    console.log("WebSocket not open, attempting to reconnect...");
    await new Promise(resolve => {
      ws.value.onopen = () => {
        resolve();
        console.log("WebSocket reconnected before sending message.");
      };
      connectWebSocket(); // Ensure connectWebSocket is called to trigger onopen
    });
  }

  try {
    const messagePayload = JSON.stringify({
      prompt: inputText.value,
      voice: selectedVoice.value,
      language: selectedVoice.value.substring(0, 5),
      encoding: "LINEAR16",
      sampleRate: audioContext.value.sampleRate
    });
    ws.value.send(messagePayload);
    console.log("Sent message to WS:", messagePayload);
  } catch (e) {
    console.error("Error sending message via WebSocket:", e);
    error.value = "Failed to send message to the server.";
    isSynthesizing.value = false;
  }
};

onMounted(() => {
  connectWebSocket();
});

onBeforeUnmount(() => {
  if (ws.value) {
    ws.value.close();
  }
  if (audioContext.value) {
    audioContext.value.close();
  }
});
</script>

<style>
.v-container {
  max-width: 800px;
}
pre {
    word-wrap: break-word;
    white-space: pre-wrap;
}
</style>