<template>
  <div>
    <h2> Google STT Live</h2>
    <div class="d-flex ga-2 my-10 align-center">
      <v-btn @click="start" class="d-flex justify-center align-center bg-surface border rounded-pill py-8 px-5">
        <Phone :size="30" stroke="#2EC4B6" />
      </v-btn>
      <v-btn @click="end" class="d-flex justify-center align-center bg-error border rounded-pill py-8 px-5">
        <PhoneOff :size="30" stroke="red" />
      </v-btn>
    </div>
    <p>{{ transcript }}</p>
  </div>
</template>

<script lang="ts" setup>
import { Phone, PhoneOff } from 'lucide-vue-next'

let socket: WebSocket
let audioCtx: AudioContext
let processor: ScriptProcessorNode
let source: MediaStreamAudioSourceNode

const transcript = ref("Transcript here")
const url = "localhost:8000"

function convertToInt16(float32: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(float32.length * 2)
  const view = new DataView(buffer)

  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(
      -1, Math.min(1, float32[i])
    );

    view.setInt16(i * 2,
      s < 0 ?
        s * 0x8000 :
        s * 0x7FFF, true);
  }
  return buffer
}

async function start() {
  socket = new WebSocket(`ws://${url}`)
  socket.onmessage = (e: MessageEvent) => {
    transcript.value = e.data
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: false,
      sampleRate: 44100
    }
  })

  audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 44100 })

  processor = audioCtx.createScriptProcessor(4096, 1, 1)
  source = audioCtx.createMediaStreamSource(stream)

  processor.onaudioprocess = e => {
    const float32 = e.inputBuffer.getChannelData(0)
    const int16 = convertToInt16(float32)
    if (socket.readyState === 1) socket.send(int16)
  }

  source.connect(processor)
  processor.connect(audioCtx.destination)
}

function end() {
  processor?.disconnect()
  source?.disconnect()
  audioCtx?.close()
  socket?.close()
}

onBeforeUnmount(() => {
  end()
})
</script>
