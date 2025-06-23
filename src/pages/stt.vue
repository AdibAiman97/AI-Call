<template>
  <div>
    <h2>Google STT Live</h2>
    <div class="d-flex ga-2 my-10 align-center">
      <v-btn
        @click="start"
        class="d-flex justify-center align-center bg-surface border rounded-pill py-8 px-5"
      >
        <Phone :size="30" stroke="#2EC4B6" />
      </v-btn>
      <v-btn
        @click="end"
        class="d-flex justify-center align-center bg-error border rounded-pill py-8 px-5"
      >
        <PhoneOff :size="30" stroke="red" />
      </v-btn>
      <v-btn
        @click="clearAudioQueue"
        :disabled="audioQueue.length === 0 && !isPlayingAudio"
        class="d-flex justify-center align-center bg-warning border rounded-pill py-8 px-5"
        title="Clear Audio Queue"
      >
        Clear Queue
      </v-btn>
    </div>
    
    <!-- Audio Queue Status -->
    <div class="audio-queue-status mb-4">
      <div class="d-flex align-center ga-2 mb-2">
        <span class="text-subtitle-2">Audio Queue Status:</span>
        <v-chip 
          :color="isPlayingAudio ? 'success' : 'grey'" 
          size="small"
        >
          {{ isPlayingAudio ? 'Playing' : 'Idle' }}
        </v-chip>
        <v-chip 
          :color="audioQueue.length > 0 ? 'info' : 'grey'" 
          size="small"
        >
          {{ audioQueue.length }} in queue
        </v-chip>
      </div>
      
      <!-- Currently Playing -->
      <div v-if="isPlayingAudio" class="playing-now mb-2">
        <span class="text-body-2">🎵 Playing: </span>
        <span class="text-body-2 font-weight-medium">{{ currentPlayingText }}</span>
      </div>
      
      <!-- Queue Preview -->
      <div v-if="audioQueue.length > 0" class="queue-preview">
        <span class="text-body-2">📋 Next in queue:</span>
        <div class="queue-items mt-1">
          <div 
            v-for="(item, index) in audioQueue.slice(0, 3)" 
            :key="item.id"
            class="queue-item text-body-2 text-grey-darken-1"
          >
            {{ index + 1 }}. {{ item.text.substring(0, 50) }}{{ item.text.length > 50 ? '...' : '' }}
          </div>
          <div v-if="audioQueue.length > 3" class="text-body-2 text-grey">
            ... and {{ audioQueue.length - 3 }} more
          </div>
        </div>
      </div>
    </div>
    
    <p>{{ transcript }}</p>
  </div>
</template>

<script lang="ts" setup>
import { Phone, PhoneOff } from "lucide-vue-next";

let socket: WebSocket;
let audioCtx: AudioContext;
let processor: AudioWorkletNode;
let source: MediaStreamAudioSourceNode;
let stream: MediaStream;

const transcript = ref("Transcript here");
const url = "localhost:8000/stt";

const samepleRate = 16000;

// Audio queue management
interface AudioQueueItem {
  base64Audio: string;
  text: string;
  id: string;
}

const audioQueue = ref<AudioQueueItem[]>([]);
const isPlayingAudio = ref(false);
const currentAudio = ref<HTMLAudioElement | null>(null);
const currentPlayingText = ref<string>('');

// Add audio to queue
function addToAudioQueue(base64Audio: string, text: string) {
  const id = Date.now().toString() + Math.random().toString(36).substr(2, 9);
  audioQueue.value.push({
    base64Audio,
    text,
    id
  });
  
  console.log(`🎵 Added to queue: "${text.substring(0, 30)}..." (Queue length: ${audioQueue.value.length})`);
  
  // Start processing queue if not already playing
  if (!isPlayingAudio.value) {
    processAudioQueue();
  }
}

// Process the audio queue
async function processAudioQueue() {
  if (audioQueue.value.length === 0) {
    console.log('🎵 Audio queue is empty');
    isPlayingAudio.value = false;
    return;
  }
  
  if (isPlayingAudio.value) {
    console.log('🎵 Already playing audio, waiting...');
    return;
  }
  
  isPlayingAudio.value = true;
  
  // Get the first item from the queue
  const audioItem = audioQueue.value.shift();
  if (!audioItem) {
    isPlayingAudio.value = false;
    return;
  }
  
  console.log(`🎵 Processing queue item: "${audioItem.text.substring(0, 30)}..." (${audioQueue.value.length} remaining)`);
  
  // Set current playing text
  currentPlayingText.value = audioItem.text;
  
  try {
    await playAudioFromBase64(audioItem.base64Audio, audioItem.text);
  } catch (error) {
    console.error('🚫 Error playing queued audio:', error);
  }
  
  // Mark as not playing and process next item
  isPlayingAudio.value = false;
  currentPlayingText.value = '';
  
  // Process next item in queue if any
  if (audioQueue.value.length > 0) {
    setTimeout(() => processAudioQueue(), 100); // Small delay between audio clips
  }
}

// Clear the audio queue
function clearAudioQueue() {
  console.log(`🗑️ Clearing audio queue (${audioQueue.value.length} items)`);
  audioQueue.value = [];
  
  // Stop current audio if playing
  if (currentAudio.value) {
    currentAudio.value.pause();
    currentAudio.value.currentTime = 0;
    currentAudio.value = null;
  }
  
  isPlayingAudio.value = false;
  currentPlayingText.value = '';
}

// Function to play audio from base64 data
async function playAudioFromBase64(base64Audio: string, text: string) {
  try {
    // Convert base64 to binary data
    const binaryString = atob(base64Audio);
    const bytes = new Uint8Array(binaryString.length);
    
    // Convert each character to byte
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    
    // Create a blob from the binary data
    const audioBlob = new Blob([bytes], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    
    // Create and play audio element
    const audio = new Audio(audioUrl);
    currentAudio.value = audio;
    
    // Return a promise that resolves when audio finishes
    return new Promise<void>((resolve, reject) => {
      // Add event listeners for debugging
      audio.onloadstart = () => console.log('🎵 Audio loading started');
      audio.oncanplaythrough = () => console.log('🎵 Audio can play through');
      audio.onplay = () => console.log(`🎵 Playing TTS for: "${text.substring(0, 30)}..."`);
      
      audio.onended = () => {
        console.log('🎵 Audio playback finished');
        // Clean up the object URL to free memory
        URL.revokeObjectURL(audioUrl);
        currentAudio.value = null;
        resolve();
      };
      
      audio.onerror = (e) => {
        console.error('🚫 Audio playback error:', e);
        URL.revokeObjectURL(audioUrl);
        currentAudio.value = null;
        reject(e);
      };
      
      // Play the audio
      audio.play().catch(reject);
    });
    
  } catch (error) {
    console.error('🚫 Error playing audio:', error);
    throw error;
  }
}

async function start() {
  socket = new WebSocket(`ws://${url}`);
  
  socket.onmessage = (e: MessageEvent) => {
    try {
      // Try to parse as JSON first
      const data = JSON.parse(e.data);
      
      // Handle different message types
      if (data.type === 'tts_audio') {
        // Handle TTS audio response - add to queue instead of playing immediately
        console.log('🎵 Received TTS audio response');
        addToAudioQueue(data.audio_data, data.text);
      } else {
        // Handle other message types or fallback
        console.log('📝 Received other message:', data);
      }
      
    } catch (error) {
      // If it's not JSON, treat as plain text (like your original STT response)
      transcript.value = e.data;
      console.log('📝 Received text transcript:', e.data);
    }
  };

  socket.onerror = (error) => {
    console.error('🚫 WebSocket error:', error);
  };

  socket.onclose = () => {
    console.log('🔌 WebSocket connection closed');
  };

  stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      sampleRate: samepleRate,
    },
  });

  audioCtx = new AudioContext({ sampleRate: samepleRate });
  await audioCtx.audioWorklet.addModule("/processor.js");
  // audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: samepleRate })
  source = audioCtx.createMediaStreamSource(stream);
  processor = new AudioWorkletNode(audioCtx, "pcm-processor");

  processor.port.onmessage = (e: MessageEvent) => {
    if (socket.readyState === 1) socket.send(e.data);
  };

  source.connect(processor).connect(audioCtx.destination);
}

function end() {
  // Clear audio queue and stop current playback
  clearAudioQueue();
  
  processor?.disconnect();
  source?.disconnect();
  audioCtx?.close();
  socket?.close();
  stream?.getTracks().forEach((track) => track.stop());
}

onBeforeUnmount(() => {
  end();
});
</script>
