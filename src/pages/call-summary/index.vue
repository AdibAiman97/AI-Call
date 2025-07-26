<template>
  <div class="d-flex justify-space-between align-center mb-2">
    <h1>Call Summary</h1>
  </div>

  <p class="text-foreground pb-2 mb-4" v-if="!loading && callSessionData">
    {{ formatDuration(callSessionData.duration_secs) }} •
    {{ formatDateTime(callSessionData.start_time) }}
  </p>
  <p class="text-foreground pb-2 mb-4" v-else-if="loading">
    Loading call information...
  </p>


  <!-- Summary Content -->
  <div class="d-flex flex-column">
    <v-card class="mb-6 flex-grow-1 rounded-lg mr-md-4 w-100 elevation-2">
      <v-card-title class="text-h6 text-foreground">
        Summarized Context
      </v-card-title>
      <v-card-text
        class="d-flex flex-column ga-2 text-body-1 text-secForeground"
      >
        <div v-for="item in summaryList" class="d-flex ga-2">
          <p>•</p>
          <p>{{ item }}</p>
        </div>
      </v-card-text>
      <v-card-actions class="pa-2">
        <v-btn
          text
          color="primary"
          class="text-none px-2 text-capitalize"
          @click="fullTranscript"
          >View full transcript >
        </v-btn>
      </v-card-actions>
    </v-card>

    <!-- Suggestions -->
    <v-card class="mb-6 rounded-lg mr-md-4 w-100 elevation-2">
      <v-card-title class="text-h6 text-foreground"> Suggestions </v-card-title>
      <v-card-text
        class="d-flex flex-column ga-2 text-body-1 text-secForeground"
      >
        <div v-for="item in customerNextSteps" class="d-flex ga-2">
          <p>•</p>
          <p>{{ item }}</p>
        </div>
      </v-card-text>
      <v-card-actions class="pa-2">
        <v-btn
          text
          color="primary"
          class="text-none px-2 text-capitalize"
          @click="fullTranscript"
          >View full transcript >
        </v-btn>
      </v-card-actions>
    </v-card>
  </div>

  <!-- Action Buttons -->
  <div class="d-flex ga-5 mb-6">
    <v-btn outlined large to="/" class="px-6 text-capitalize">
      <!-- <v-icon left>mdi-home</v-icon> -->
      Back to Home Page
    </v-btn>

    <!-- <v-btn
      color="primary"
      large
      @click="scheduleAppointment"
      class="px-6 text-capitalize"
    >
      Schedule Appointment
    </v-btn> -->
  </div>
  <!-- </v-container> -->
</template>

<script setup>
import { useHotkey } from '@/utils/Hotkey'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
useHotkey('a', () => {
  console.log('go to admin page')
  router.push('/admin')
 }, { shift: false, command: true })


 useHotkey('b', () => {
  console.log('go to home page')
  router.push('/')
 }, { shift: false, command: true })
 
import { ref, onMounted } from "vue";
import { useCallStore } from "../../stores/call";

const summaryList = ref([]);
const customerNextSteps = ref([]);
const callSessionData = ref(null);
const loading = ref(true);

const callStore = useCallStore();

// Local fallback formatTime function
// const formatDuration = (timeInSecs) => {
//   if (!timeInSecs || isNaN(timeInSecs)) return 'Duration not available';
  
//   const minutes = Math.floor(timeInSecs / 60);
//   const seconds = timeInSecs % 60;

//   if (minutes === 0) {
//     return `${seconds} second${seconds !== 1 ? "s" : ""}`;
//   } else if (seconds === 0) {
//     return `${minutes} minute${minutes !== 1 ? "s" : ""}`;
//   } else {
//     return `${minutes} minute${minutes !== 1 ? "s" : ""} ${seconds} second${
//       seconds !== 1 ? "s" : ""
//     }`;
//   }
// };

const fetchCallSessionData = async () => {
  try {
    loading.value = true;
    
    console.log('📊 Call Summary - Debug Info:');
    console.log('  callStore.callSessionId:', callStore.callSessionId);
    console.log('  typeof callSessionId:', typeof callStore.callSessionId);
    console.log('  callStore state:', callStore.$state);
    
    if (!callStore.callSessionId) {
      // Try to get from URL params as fallback  
      const urlParams = new URLSearchParams(window.location.search);
      const sessionId = urlParams.get('sessionId') || urlParams.get('id');
      
      // Also try Vue router query params
      const route = useRoute();
      const routeSessionId = route.query.sessionId || route.query.id;
      
      const finalSessionId = sessionId || routeSessionId;
      
      if (finalSessionId) {
        console.log('📊 Using session ID from params:', finalSessionId);
        callStore.setCallSessionId(parseInt(finalSessionId.toString()));
      } else {
        console.error('❌ No call session ID available in store, URL params, or route params');
        throw new Error("No call session ID available");
      }
    }
    
    console.log(`📊 Fetching call session data for ID: ${callStore.callSessionId}`);
    
    const response = await fetch(
      `${import.meta.env.VITE_API_BASE_URL}/call_session/${callStore.callSessionId}`
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch call session data: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log("📊 Raw call session data received:", data);
    callSessionData.value = data;

    // Parse summarized content with detailed logging
    console.log("📝 Raw summarized_content:", data.summarized_content);
    if (data.summarized_content && data.summarized_content.trim()) {
      const rawSummary = data.summarized_content;
      console.log("📝 Processing summary of length:", rawSummary.length);
      
      // Split by lines and process each line
      const summaryLines = rawSummary
        .split(/\r?\n/)
        .map((line, index) => {
          const trimmed = line.trim();
          console.log(`📝 Line ${index}: "${trimmed}"`);
          return trimmed;
        })
        .filter((line) => {
          // Keep lines that have meaningful content
          const hasContent = line.length > 2 && 
                           !line.match(/^(Summary:|Key Topics:|Call Summary:|CALL ANALYSIS|---)/i);
          console.log(`📝 Line "${line.substring(0, 30)}..." kept: ${hasContent}`);
          return hasContent;
        })
        .map((line) => {
          // Clean up formatting but preserve content
          return line.replace(/^[•\-\*\d+\.\)\s]+/, "").trim();
        })
        .filter((line) => line.length > 0);
      
      summaryList.value = summaryLines;
      console.log("📝 Final summary list:", summaryLines);
    }
    
    // Parse customer suggestions with detailed logging  
    console.log("💡 Raw customer_suggestions:", data.customer_suggestions);
    if (data.customer_suggestions && data.customer_suggestions.trim()) {
      const rawSuggestions = data.customer_suggestions;
      console.log("💡 Processing suggestions of length:", rawSuggestions.length);
      
      // Split by lines and process each line
      const suggestionLines = rawSuggestions
        .split(/\r?\n/)
        .map((line, index) => {
          const trimmed = line.trim();
          console.log(`💡 Line ${index}: "${trimmed}"`);
          return trimmed;
        })
        .filter((line) => {
          // Keep lines that have meaningful content
          const hasContent = line.length > 2 && 
                           !line.match(/^(Suggestions:|Based on|Okay,|Here are|---)/i);
          console.log(`💡 Line "${line.substring(0, 30)}..." kept: ${hasContent}`);
          return hasContent;
        })
        .map((line) => {
          // Clean up formatting but preserve content
          return line.replace(/^[•\-\*\d+\.\)\s]+/, "").trim();
        })
        .filter((line) => line.length > 0);
      
      customerNextSteps.value = suggestionLines;
      console.log("💡 Final suggestions list:", suggestionLines);
    }
    // Set fallback content if data is empty or wasn't processed correctly
    if (summaryList.value.length === 0) {
      if (data.summarized_content) {
        console.log("⚠️ Summary content exists but couldn't be parsed properly");
        console.log("Raw summary content:", data.summarized_content);
        
        // As a last resort, try to show the raw content directly (first 500 chars)
        const rawPreview = data.summarized_content.substring(0, 500);
        summaryList.value = [
          "Call summary data found but formatting needs adjustment:",
          rawPreview + (data.summarized_content.length > 500 ? "..." : "")
        ];
      } else {
        console.log("ℹ️ No summary content available yet");
        summaryList.value = [
          "Call summary is being processed by AI.",
          "Please refresh in a few moments or check that the call ended properly."
        ];
      }
    }
    
    if (customerNextSteps.value.length === 0) {
      if (data.customer_suggestions) {
        console.log("⚠️ Customer suggestions exist but couldn't be parsed properly");
        console.log("Raw customer suggestions:", data.customer_suggestions);
        
        // As a last resort, try to show the raw content directly (first 500 chars)
        const rawPreview = data.customer_suggestions.substring(0, 500);
        customerNextSteps.value = [
          "Customer suggestions data found but formatting needs adjustment:",
          rawPreview + (data.customer_suggestions.length > 500 ? "..." : "")
        ];
      } else {
        console.log("ℹ️ No customer suggestions available yet");
        customerNextSteps.value = [
          "Customer suggestions are being generated by AI.",
          "Please refresh in a few moments or check that the call ended properly."
        ];
      }
    }
    
  } catch (error) {
    console.error("Error fetching call session data:", error);
    summaryList.value = [
      "Unable to load call summary. Please try again later.",
    ];
    customerNextSteps.value = ["Please contact support for assistance."];
  } finally {
    loading.value = false;
  }
};

const formatDateTime = (dateTimeString) => {
  if (!dateTimeString) return "";
  
  // Handle both ISO string and datetime object formats
  let date;
  if (typeof dateTimeString === 'string') {
    date = new Date(dateTimeString);
  } else if (dateTimeString instanceof Date) {
    date = dateTimeString;
  } else {
    console.warn('Unknown date format:', dateTimeString);
    return "Invalid date";
  }
  
  // Check if date is valid
  if (isNaN(date.getTime())) {
    console.warn('Invalid date:', dateTimeString);
    return "Invalid date";
  }
  
  const options = {
    weekday: "long",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  };
  return date.toLocaleString("en-US", options);
};

const formatDuration = (durationSecs) => {
  if (!durationSecs) return "00:00";
  const hours = Math.floor(durationSecs / 3600);
  const minutes = Math.floor((durationSecs % 3600) / 60);
  const seconds = durationSecs % 60;
  
  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  }
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
};

onMounted(() => {
  // Debug: Check what's available on the call store
  console.log('🔍 Call store object:', callStore);
  console.log('🔍 Call store methods:', Object.getOwnPropertyNames(callStore));
  console.log('🔍 formatTime function:', callStore.formatTime);
  console.log('🔍 typeof formatTime:', typeof callStore.formatTime);

  fetchCallSessionData();
});
</script>