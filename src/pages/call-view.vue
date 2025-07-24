<script setup lang="ts">
import { ref, nextTick, onMounted, computed, watch, onUnmounted, onBeforeUnmount } from 'vue';
import { Volume2, Phone } from "lucide-vue-next";
import { useCallStore } from '@/stores/call';

interface Message {
  sender: 'user' | 'ai';
  text: string;
}

const callStore = useCallStore();
const messages = ref<Message[]>([]);
const interimTranscript = ref(""); // To hold the live user transcript
const messagesContainer = ref<HTMLElement | null>(null);
const websocket = ref<WebSocket | null>(null);

// --- Call State ---
const isCallActive = ref(false);
const isAiResponding = ref(false); // This will be true when waiting for first TTS chunk
const isUserSpeaking = ref(false);
const elapsedSeconds = ref(0);
let timer: NodeJS.Timeout | null = null;

// --- Web Audio API variables ---
let audioContext: AudioContext | null = null;
let stream: MediaStream | null = null;
let processor: AudioWorkletNode | null = null;
let source: MediaStreamAudioSourceNode | null = null;
const audioPlayer = new Audio();
const audioQueue = ref<string[]>([]);
const isPlayingAudio = ref(false);

// --- AI Audio Visualizer variables ---
let aiAudioSource: MediaElementAudioSourceNode | null = null;
let aiAudioAnalyserNode: AnalyserNode | null = null;
const barCount = 32; // Number of bars in the visualizer
const audioBars = ref(Array(barCount).fill({ height: 0 }));

// --- Constants ---
const WEBSOCKET_URL = 'ws://localhost:8000/ws/conversation'; // Correct protocol and endpoint
const SAMPLE_RATE = 16000;

// --- Computed Properties ---
const isAiSpeaking = computed(() => isPlayingAudio.value);

const formattedTime = computed(() => {
  const hours = Math.floor(elapsedSeconds.value / 3600);
  const hh = hours > 0 ? String(hours).padStart(2, "0") + ":" : "";
  const mm = String(Math.floor((elapsedSeconds.value % 3600) / 60)).padStart(2, "0");
  const ss = String(elapsedSeconds.value % 60).padStart(2, "0");
  return `${hh}${mm}:${ss}`;
});

// --- Lifecycle Hooks ---
onMounted(() => {
  connectWebSocket();
  setupAiAudioVisualizer();
});

onUnmounted(() => {
  if (timer) clearInterval(timer);
  endCall();
});

onBeforeUnmount(() => {
    endCall();
});

watch(isCallActive, (isActive) => {
    if (isActive) {
        elapsedSeconds.value = 0;
        timer = setInterval(() => {
            elapsedSeconds.value++;
        }, 1000);
    } else {
        if (timer) clearInterval(timer);
        timer = null;
    }
});

// --- Setup for AI Audio Visualization ---
function setupAiAudioVisualizer() {
  audioPlayer.crossOrigin = "anonymous";
  const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
  audioContext = new AudioContext();

  aiAudioSource = audioContext.createMediaElementSource(audioPlayer);
  aiAudioAnalyserNode = audioContext.createAnalyser();
  aiAudioAnalyserNode.fftSize = 256;

  aiAudioSource.connect(aiAudioAnalyserNode);
  aiAudioAnalyserNode.connect(audioContext.destination);

  updateVisualizer();
}

// --- Animation loop for the visualizer ---
function updateVisualizer() {
  if (isAiSpeaking.value && aiAudioAnalyserNode) {
    const bufferLength = aiAudioAnalyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    aiAudioAnalyserNode.getByteFrequencyData(dataArray);

    const newBars = [];
    const step = Math.floor(bufferLength / barCount);
    for (let i = 0; i < barCount; i++) {
      let barHeight = dataArray[i * step];
      newBars.push({ height: Math.max(2, (barHeight / 255) * 50) });
    }
    audioBars.value = newBars;
  } else if (audioBars.value[0]?.height > 2) {
    // Reset bars to minimum height when AI is not speaking
    audioBars.value = Array(barCount).fill({ height: 2 });
  }
  
  requestAnimationFrame(updateVisualizer);
}

// --- WebSocket Logic ---
function connectWebSocket() {
    websocket.value = new WebSocket(WEBSOCKET_URL);
    websocket.value.onopen = () => console.log("WebSocket connection established.");
    websocket.value.onmessage = (event) => handleWebSocketMessage(JSON.parse(event.data));
    websocket.value.onerror = (error) => console.error("WebSocket Error:", error);
    websocket.value.onclose = () => {
        console.log("WebSocket connection closed. Reconnecting...");
        setTimeout(connectWebSocket, 3000);
    };
}

function handleWebSocketMessage(data: any) {
    switch(data.type) {
        case 'interim':
            isUserSpeaking.value = true;
            interimTranscript.value = data.display_text;
            break;
        case 'final':
            isUserSpeaking.value = false;
            // Finalize the user message in the array
            const lastMsg = messages.value[messages.value.length - 1];
            if (lastMsg && lastMsg.sender === 'user') {
                lastMsg.text = data.full_transcript;
            }
            interimTranscript.value = ""; // Clear interim text
            isAiResponding.value = true; // AI is now processing
            break;
        case 'tts_audio':
            // When the first audio chunk arrives, the AI is no longer "thinking"
            if (isAiResponding.value) {
                isAiResponding.value = false;
                messages.value.push({ sender: 'ai', text: '' }); // Add empty AI message bubble
            }
            // Append the text part of the message to the last AI bubble
            const lastAiMsg = messages.value[messages.value.length - 1];
            if (lastAiMsg && lastAiMsg.sender === 'ai') {
                lastAiMsg.text += data.text;
            }
            // Queue the audio data for playback
            audioQueue.value.push(data.audio_data);
            if (!isPlayingAudio.value) {
                playNextAudioChunk();
            }
            break;
        case 'error':
            messages.value.push({ sender: 'ai', text: `An error occurred: ${data.text}` });
            isAiResponding.value = false;
            break;
    }
    scrollToBottom();
}

// Watcher to add a new user message bubble when speaking starts
watch(isUserSpeaking, (speaking) => {
    if (speaking && messages.value[messages.value.length - 1]?.sender !== 'user') {
        messages.value.push({ sender: 'user', text: '' });
    }
});

// Computed property to display the live transcript
const liveTranscript = computed(() => {
    const lastMessage = messages.value[messages.value.length - 1];
    if (isUserSpeaking.value && lastMessage?.sender === 'user') {
        // Combine the final part from the last message with the new interim part
        const baseText = lastMessage.text ? lastMessage.text + ' ' : '';
        return baseText + interimTranscript.value;
    }
    return lastMessage ? lastMessage.text : '';
});

function playNextAudioChunk() {
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
    if (audioQueue.value.length === 0) {
        isPlayingAudio.value = false;
        return;
    }
    isPlayingAudio.value = true;
    const audioBase64 = audioQueue.value.shift();
    if (!audioBase64) {
        isPlayingAudio.value = false;
        return;
    }
    const audioUrl = `data:audio/mpeg;base64,${audioBase64}`;
    audioPlayer.src = audioUrl;
    audioPlayer.play().catch(e => {
        console.error("Audio playback error:", e);
        isPlayingAudio.value = false;
        playNextAudioChunk();
    });
    audioPlayer.onended = () => {
        playNextAudioChunk();
    };
}

// --- Call & VAD Logic ---
function toggleCall() {
    if (isCallActive.value) {
        endCall();
    } else {
        startCall();
    }
}

async function startCall() {
  if (isCallActive.value) return;
  console.log("Starting call...");
  isCallActive.value = true;

  try {
    const audioConstraints = {
      audio: {
        sampleRate: SAMPLE_RATE,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    };
    stream = await navigator.mediaDevices.getUserMedia(audioConstraints);

    // Use the existing audio context if it's there and open
    if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
          sampleRate: SAMPLE_RATE,
        });
    }
    
    await audioContext.audioWorklet.addModule('/processor.js');

    source = audioContext.createMediaStreamSource(stream);
    processor = new AudioWorkletNode(audioContext, 'pcm-processor');

    processor.port.onmessage = (e: MessageEvent) => {
      if (websocket.value && websocket.value.readyState === WebSocket.OPEN) {
        websocket.value.send(e.data);
      }
    };

    source.connect(processor); // Don't connect to destination to avoid hearing self

  } catch (error) {
    console.error("Error starting call:", error);
    endCall();
  }
}

function endCall() {
  console.log("Ending call and resetting the application state.");
  
  // Stop microphone stream
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  
  // Disconnect audio nodes
  if (processor) {
    processor.disconnect();
    processor = null;
  }
  if (source) {
    source.disconnect();
    source = null;
  }

  // Stop audio playback
  if (audioPlayer) {
      audioPlayer.pause();
      audioPlayer.src = '';
  }

  // Clear queues and reset UI state
  audioQueue.value = [];
  messages.value = [];
  interimTranscript.value = "";
  isCallActive.value = false;
  isUserSpeaking.value = false;
  isAiResponding.value = false;
}

// --- Utility Functions ---
function renderMessageContent(text: string) {
    // When user is speaking, show the live transcript
    const lastMessage = messages.value[messages.value.length - 1];
    if (isUserSpeaking.value && lastMessage?.sender === 'user' && text === lastMessage.text) {
        const baseText = lastMessage.text ? lastMessage.text + ' ' : '';
        return (baseText + interimTranscript.value).replace(/\n/g, '<br>');
    }

    if (!text) return '';
    let html = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    const lines = html.split('\n');
    let inList = false;
    let processedHtml = '';
    lines.forEach(line => {
        const trimmedLine = line.trim();
        if (trimmedLine.startsWith('* ') || trimmedLine.startsWith('- ')) {
            if (!inList) { processedHtml += '<ul>'; inList = true; }
            processedHtml += `<li>${trimmedLine.substring(2)}</li>`;
        } else {
            if (inList) { processedHtml += '</ul>'; inList = false; }
            if (line) processedHtml += `<p>${line}</p>`;
        }
    });
    if (inList) processedHtml += '</ul>';
    return processedHtml.replace(/<p><\/p>/g, '');
}

function scrollToBottom() {
    nextTick(() => {
        if (messagesContainer.value) messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight;
    });
}
</script> 