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

async function start() {
  socket = new WebSocket(`ws://${url}`);
  socket.onmessage = (e: MessageEvent) => {
    transcript.value = e.data;
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
