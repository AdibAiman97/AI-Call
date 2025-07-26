<template>
  <!-- Admin Call Summary -->
  <div>
    <div class="d-flex justify-space-between align-center mb-2">
      <h1>{{ pageTitle }}</h1>

      <!-- Call Summary Tab Buttons -->
      <div v-if="tab === 'sum'" class="d-flex ga-3">
        <v-btn 
          class="text-capitalize text-foreground"
          @click="refreshData"
          :loading="loading"
        >
          <v-icon class="mr-2">mdi-refresh</v-icon>
          Refresh
        </v-btn>
        <v-btn 
          class="bg-primary text-background text-capitalize"
          @click="updateCallSession"
          :loading="updating"
        >
          <v-icon class="mr-2">mdi-content-save</v-icon>
          Update Data
        </v-btn>
      </div>

      <!-- Appointment Tab Buttons -->
      <div v-else-if="tab === 'app'" class="d-flex ga-3">
        <v-btn class="text-capitalize text-foreground">
          <ListFilter :size="20" class="mr-2" />
          Filter
        </v-btn>
        <v-btn class="bg-primary text-background text-capitalize">
          <Plus :size="20" class="mr-2" />
          Add Appointment
        </v-btn>
      </div>
    </div>
    <p class="text-foreground pb-2 mb-4" v-if="tab === 'sum'">
      <span v-if="!loading && callSessionData && customerData">
        {{ customerData.first_name }} {{ customerData.last_name }} • {{ customerInfo.duration }} •
        {{ customerInfo.time }}
      </span>
      <span v-else-if="loading"> Loading call information... </span>
      <span v-else-if="error"> Error loading call data: {{ error }} </span>
    </p>
    <p class="text-foreground pb-2 mb-4" v-else-if="tab === 'app'">
      Schedule and manage customer appointments
    </p>

    <v-tabs v-model="tab" class="text-primary mb-5">
      <v-tab value="sum" class="text-capitalize text-h6 p-0 m-0">Summary (⌘K)</v-tab>
      <v-tab value="app" class="text-capitalize text-h6 p-0 m-0"
        >Appointment (⌘L)</v-tab
      >
    </v-tabs>

    <v-tabs-window v-model="tab" class="">
      <v-tabs-window-item value="sum">
        <div class="admin-layout">
          <!-- Row 1: Summarized Content & Customer -->
          <div class="row-1">
            <!-- Summarized Content -->
            <v-card class="rounded-lg elevation-2 d-flex flex-column">
              <v-card-title class="text-h6 text-foreground flex-shrink-0">
                Summarized Context
              </v-card-title>
              <v-card-text
                class="d-flex flex-column ga-2 text-body-1 text-secForeground flex-grow-1 overflow-y-auto"
                style="min-height: 200px;"
              >
                <div v-if="!loading && summaryList.length > 0">
                  <div v-for="item in summaryList" class="d-flex ga-2 mb-2">
                    <p>•</p>
                    <p v-html="highlightKeywordInText(item)"></p>
                  </div>
                </div>
                <div v-else-if="loading" class="text-center pa-4">
                  <v-progress-circular
                    indeterminate
                    size="24"
                  ></v-progress-circular>
                  <p class="mt-2">Loading summary...</p>
                </div>
                <div v-else class="text-center pa-4 text-grey">
                  No summary available
                </div>
              </v-card-text>
              <v-card-actions class="pa-2 flex-shrink-0">
                <v-btn
                  text
                  color="primary"
                  class="text-capitalize px-2"
                  @click="viewFullTranscript"
                  >View Full Transcript >
                </v-btn>
              </v-card-actions>
            </v-card>

            <!-- Customer Card -->
            <v-card class="rounded-lg elevation-2 d-flex flex-column">
              <v-card-title class="text-h6 flex-shrink-0"
                >Customer Details</v-card-title
              >
              <v-card-text class="flex-grow-1 d-flex flex-column">
                <div
                  class="d-flex align-center mb-4"
                  v-if="!loading && customerData"
                >
                  <v-avatar
                    size="40"
                    class="mr-4"
                    color="#BFC6C0"
                    rounded="circle"
                  >
                    {{ getCustomerInitials(customerData.first_name + ' ' + customerData.last_name) }}
                  </v-avatar>
                  <div class="d-flex flex-column ga-2">
                    <p class="font-weight-bold mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-account</v-icon>
                      {{ customerData.first_name }} {{ customerData.last_name }}
                    </p>
                    <p class="mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-email</v-icon>
                      {{ customerData.email }}
                    </p>
                    <p class="mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-phone</v-icon>
                      {{ customerData.phone_number }}
                    </p>
                    <p class="mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-cash</v-icon>
                      Budget: RM{{ customerData.budget?.toLocaleString() }}
                    </p>
                    <p class="mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-map-marker</v-icon>
                      {{ customerData.preferred_location }}
                    </p>
                    <p class="mb-1 text-foreground text-caption">
                      <v-icon small class="mr-2">mdi-target</v-icon>
                      {{ customerData.purchase_purpose }}
                    </p>
                  </div>
                </div>
                <div v-else-if="loading" class="text-center pa-4 flex-grow-1">
                  <v-progress-circular
                    indeterminate
                    size="24"
                  ></v-progress-circular>
                  <p class="mt-2">Loading customer info...</p>
                </div>
                <div v-else-if="customerError" class="text-center pa-4 text-error flex-grow-1">
                  <p>{{ customerError }}</p>
                </div>

                <!-- Spacer to push button to bottom -->
                <div class="flex-grow-1"></div>

                <v-btn
                  text
                  color="primary"
                  class="text-capitalize text-background"
                  style="width: 100%"
                  @click="viewCustomerProfile"
                >
                  View Full Profile
                </v-btn>
              </v-card-text>
            </v-card>
          </div>

          <!-- Row 2: Key Topics + Suggestions & Sentiment Analysis -->
          <div class="row-2">
            <!-- Left side: Key Topics + Suggestions -->
            <div class="topics-suggestions">
              <!-- Key Topics -->
              <v-card class="rounded-lg elevation-2">
                <v-card-title class="text-h6 text-capitalize flex-shrink-0"
                  >Key Words</v-card-title
                >
                <v-card-text class="px-4">
                  <v-chip-group v-if="!loading && keywordsList.length > 0">
                    <v-chip
                      v-for="keyword in keywordsList"
                      :key="keyword"
                      :class="[
                        'ma-0 mr-3 cursor-pointer',
                        isKeywordSelected(keyword) ? 'bg-primary' : 'bg-info',
                      ]"
                      text-color="white"
                      @click="selectKeyword(keyword)"
                    >
                      {{ keyword }}
                    </v-chip>
                  </v-chip-group>
                  <div v-else-if="loading" class="text-center pa-4">
                    <v-progress-circular
                      indeterminate
                      size="24"
                    ></v-progress-circular>
                    <p class="mt-2">Loading keywords...</p>
                  </div>
                  <div v-else class="text-center pa-4 text-grey">
                    No keywords available
                  </div>
                </v-card-text>
              </v-card>

              <!-- Suggestions -->
              <v-card
                class="rounded-lg elevation-2 d-flex flex-column"
                style="flex: 1"
              >
                <v-card-title class="text-h6 text-foreground flex-shrink-0"
                  >Admin Suggestions</v-card-title
                >
                <v-card-text
                  class="d-flex flex-column ga-2 text-body-1 text-secForeground flex-grow-1 overflow-y-auto"
                >
                  <div v-if="!loading && agentNextSteps.length > 0">
                    <div
                      v-for="item in agentNextSteps"
                      class="d-flex ga-2 mb-2"
                    >
                      <p>•</p>
                      <p>{{ item }}</p>
                    </div>
                  </div>
                  <div v-else-if="loading" class="text-center pa-4">
                    <v-progress-circular
                      indeterminate
                      size="24"
                    ></v-progress-circular>
                    <p class="mt-2">Loading suggestions...</p>
                  </div>
                  <div v-else class="text-center pa-4 text-grey">
                    No suggestions available
                  </div>
                </v-card-text>
                <v-card-actions class="pa-2 flex-shrink-0">
                  <v-btn
                    text
                    color="primary"
                    class="text-capitalize px-2"
                    @click="viewFullTranscript"
                    >View full transcript >
                  </v-btn>
                </v-card-actions>
              </v-card>
            </div>

            <!-- Sentiment Analysis Card -->
            <v-card class="rounded-lg elevation-2 d-flex flex-column">
              <v-card-title class="text-h6 text-foreground flex-shrink-0"
                >Sentiment Analysis</v-card-title
              >
              <v-card-text
                class="pa-4 d-flex flex-column ga-3 flex-grow-1"
                v-if="!loading"
              >
                <!-- Dynamic Sentiment Items -->
                <div
                  v-for="sentiment in sentimentConfig"
                  :key="sentiment.key"
                  class="d-flex align-center pa-3 rounded-lg"
                  :class="getSentimentStyle(sentiment.key).bgClass"
                >
                  <v-icon
                    size="28"
                    :color="getSentimentStyle(sentiment.key).color"
                    class="mr-3"
                  >
                    {{ sentiment.icon }}
                  </v-icon>
                  <div
                    class="d-flex justify-space-between align-center flex-grow-1"
                  >
                    <span
                      class="font-weight-medium text-body-1"
                      :style="{ color: getSentimentStyle(sentiment.key).color }"
                    >
                      {{ sentiment.label }}
                    </span>
                    <span
                      class="font-weight-bold text-body-1"
                      :style="{ color: getSentimentStyle(sentiment.key).color }"
                    >
                      {{ sentimentData[sentiment.key] }}%
                    </span>
                  </div>
                </div>

                <div
                  class="mt-2 text-secForeground text-body-1"
                  v-if="callSessionData"
                >
                  {{ getSentimentDescription() }}
                </div>
              </v-card-text>
              <v-card-text v-else class="text-center pa-4">
                <v-progress-circular
                  indeterminate
                  size="24"
                ></v-progress-circular>
                <p class="mt-2">Loading sentiment analysis...</p>
              </v-card-text>
            </v-card>
          </div>
        </div>
      </v-tabs-window-item>

      <v-tabs-window-item value="app">
        <Appointment />
      </v-tabs-window-item>
    </v-tabs-window>

    <!-- Success/Error Snackbars -->
    <v-snackbar
      v-model="showUpdateSuccess"
      color="success"
      timeout="3000"
    >
      Call session data updated successfully!
    </v-snackbar>

    <v-snackbar
      v-model="showUpdateError"
      color="error"
      timeout="5000"
    >
      Error updating call session: {{ updateErrorMessage }}
    </v-snackbar>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import { useRouter, useRoute } from "vue-router";
import axios from "axios";
import { useCallStore } from "@/stores/call_prev";
import { Plus, ListFilter } from "lucide-vue-next";
import { useHotkey } from '@/utils/Hotkey'

const tab = ref("sum");
const router = useRouter();
const route = useRoute();

useHotkey('b', () => {
  console.log('go to call summary page')
  router.push({ path: '/call-summary', query: { id: callStore.callSessionId } })
}, { shift: false, command: true })

useHotkey(
  "k",
  () => {
    tab.value = "sum";
  },
  { shift: false, command: true }
);

useHotkey(
  "l",
  () => {
    tab.value = "app";
  },
  { shift: false, command: true }
);

const callStore = useCallStore();

// Reactive data properties
const callSessionData = ref(null);
const customerData = ref(null);
const loading = ref(true);
const updating = ref(false);
const error = ref(null);
const customerError = ref(null);
const selectedKeyword = ref(null);
const showUpdateSuccess = ref(false);
const showUpdateError = ref(false);
const updateErrorMessage = ref("");

// Computed properties for parsed data
const summaryList = computed(() => {
  if (!callSessionData.value?.summarized_content) return [];
  return callSessionData.value.summarized_content
    .split("\n")
    .filter((item) => item.trim());
});

const agentNextSteps = computed(() => {
  if (!callSessionData.value?.admin_suggestions) return [];
  return callSessionData.value.admin_suggestions
    .split("\n")
    .filter((item) => item.trim());
});

const keywordsList = computed(() => {
  if (!callSessionData.value?.key_words) return [];
  return callSessionData.value.key_words
    .split(",")
    .map((keyword) => keyword.trim())
    .filter((keyword) => keyword);
});

const sentimentData = computed(() => {
  if (!callSessionData.value) return { positive: 0, neutral: 0, negative: 0 };
  return {
    positive: callSessionData.value.positive || 0,
    neutral: callSessionData.value.neutral || 0,
    negative: callSessionData.value.negative || 0,
  };
});

const highestSentiment = computed(() => {
  const { positive, neutral, negative } = sentimentData.value;
  if (positive >= neutral && positive >= negative) return "positive";
  if (neutral >= positive && neutral >= negative) return "neutral";
  return "negative";
});

const customerInfo = computed(() => {
  if (!callSessionData.value) return { name: "", duration: "", time: "" };
  
  // Safely format duration with fallback
  let duration = "0 min";
  if (callSessionData.value.duration_secs && typeof callStore.formatTime === 'function') {
    try {
      duration = callStore.formatTime(callSessionData.value.duration_secs);
    } catch (error) {
      console.warn('Error formatting time:', error);
      duration = `${callSessionData.value.duration_secs} sec`;
    }
  } else if (callSessionData.value.duration_secs) {
    // Fallback formatting if store method is not available
    const secs = callSessionData.value.duration_secs;
    const minutes = Math.floor(secs / 60);
    const seconds = secs % 60;
    
    if (minutes === 0) {
      duration = `${seconds} second${seconds !== 1 ? "s" : ""}`;
    } else if (seconds === 0) {
      duration = `${minutes} minute${minutes !== 1 ? "s" : ""}`;
    } else {
      duration = `${minutes} minute${minutes !== 1 ? "s" : ""} ${seconds} second${seconds !== 1 ? "s" : ""}`;
    }
  }
  
  return {
    name: callSessionData.value.cust_id || "Unknown Customer",
    duration: duration,
    time: formatDateTime(callSessionData.value.start_time),
  };
});

const pageTitle = computed(() => {
  return tab.value === "sum" ? "Admin Dashboard" : "Appointment";
});

// API functions
const fetchCallSessionData = async (sessionId = null) => {
  try {
    loading.value = true;
    error.value = null;

    // Try multiple sources for call session ID
    const callSessionId =
      sessionId || callStore.callSessionId || route.query.id || route.params.id;

    if (!callSessionId) {
      // If no ID is available, get the latest call session
      await fetchLatestCallSession();
      return;
    }

    console.log(`🔍 Admin - Fetching call session data for ID: ${callSessionId}`);
    
    const response = await axios.get(
      `${import.meta.env.VITE_API_BASE_URL}/call_session/${callSessionId}`
    );
    
    callSessionData.value = response.data;
    console.log("📊 Admin - Call session data received:", response.data);
    
    // Fetch customer data using cust_id from call session
    if (response.data.cust_id) {
      await fetchCustomerData(response.data.cust_id);
    }
    
  } catch (err) {
    console.error("Error fetching call session data:", err);
    error.value = err.message;
  } finally {
    loading.value = false;
  }
};

const fetchCustomerData = async (custId) => {
  try {
    customerError.value = null;
    console.log(`👤 Admin - Fetching customer data for phone: ${custId}`);
    
    const response = await axios.get(
      `${import.meta.env.VITE_API_BASE_URL}/customers/phone/${custId}`
    );
    
    customerData.value = response.data;
    console.log("👤 Admin - Customer data received:", response.data);
    
  } catch (err) {
    console.error("Error fetching customer data:", err);
    customerError.value = `Customer details not found for ${custId}`;
    
    // Create a fallback customer object for display
    customerData.value = {
      first_name: "Unknown",
      last_name: "Customer",
      email: "No email available",
      phone_number: custId,
      budget: 0,
      preferred_location: "Not specified",
      purchase_purpose: "Not specified"
    };
  }
};

// Fallback to get the latest call session
const fetchLatestCallSession = async () => {
  try {
    console.log("🔍 Admin - Fetching latest call session...");
    const response = await axios.get(
      `${import.meta.env.VITE_API_BASE_URL}/call_session`
    );
    console.log("📊 Admin - All sessions response:", response.data);

    if (response.data && response.data.length > 0) {
      // Get the most recent call session (assuming they're ordered by ID)
      const latestSession = response.data[response.data.length - 1];
      console.log("📊 Admin - Latest session found:", latestSession);
      callSessionData.value = latestSession;
      callStore.callSessionId = latestSession.id;
      
      // Fetch customer data for the latest session
      if (latestSession.cust_id) {
        await fetchCustomerData(latestSession.cust_id);
      }
    } else {
      console.log("📊 Admin - No call sessions found, using demo data");
      // No call sessions found, use demo data
      callSessionData.value = getDemoData();
      customerData.value = getDemoCustomerData();
    }
  } catch (err) {
    console.error("Error fetching latest call session:", err);
    console.log("📊 Admin - Falling back to demo data due to error");
    // Fallback to demo data
    callSessionData.value = getDemoData();
    customerData.value = getDemoCustomerData();
  }
};

// Update call session data
const updateCallSession = async () => {
  if (!callSessionData.value?.id) {
    showUpdateError.value = true;
    updateErrorMessage.value = "No call session data to update";
    return;
  }

  try {
    updating.value = true;
    
    console.log("💾 Admin - Updating call session:", callSessionData.value.id);
    
    const updateData = {
      cust_id: callSessionData.value.cust_id,
      start_time: callSessionData.value.start_time,
      end_time: callSessionData.value.end_time,
      duration_secs: callSessionData.value.duration_secs,
      positive: callSessionData.value.positive,
      neutral: callSessionData.value.neutral,
      negative: callSessionData.value.negative,
      key_words: callSessionData.value.key_words,
      summarized_content: callSessionData.value.summarized_content,
      customer_suggestions: callSessionData.value.customer_suggestions,
      admin_suggestions: callSessionData.value.admin_suggestions,
    };

    const response = await axios.put(
      `${import.meta.env.VITE_API_BASE_URL}/call_session/${callSessionData.value.id}`,
      updateData
    );

    console.log("✅ Admin - Call session updated successfully:", response.data);
    callSessionData.value = response.data;
    showUpdateSuccess.value = true;
    
  } catch (err) {
    console.error("❌ Admin - Error updating call session:", err);
    showUpdateError.value = true;
    updateErrorMessage.value = err.response?.data?.detail || err.message;
  } finally {
    updating.value = false;
  }
};

// Refresh all data
const refreshData = async () => {
  await fetchCallSessionData(callSessionData.value?.id);
};

// Demo data fallbacks
const getDemoData = () => ({
  id: "demo",
  cust_id: "+60123456789",
  duration: "5 min",
  start_time: new Date().toISOString(),
  positive: 70,
  neutral: 20,
  negative: 10,
  key_words: "demo, example, test, features, overview",
  summarized_content:
    "This is a demo call session.\nShowing sample data for the admin dashboard.\nUse this to test the interface.",
  customer_suggestions:
    "Complete a real call to see actual data.\nThe system will automatically populate this section.\nDemo data is shown when no real sessions exist.",
  admin_suggestions:
    "Set up actual call sessions to replace this demo.\nThe admin dashboard will show real call analytics.\nThis demo helps visualize the interface layout.",
});

const getDemoCustomerData = () => ({
  first_name: "Demo",
  last_name: "Customer",
  email: "demo.customer@example.com",
  phone_number: "+60123456789",
  budget: 500000,
  preferred_location: "Kuala Lumpur",
  purchase_purpose: "Investment property"
});

// UI interaction functions
const viewFullTranscript = () => {
  // Navigate to transcript view or open modal
  console.log("View full transcript clicked");
};

const viewCustomerProfile = () => {
  // Navigate to customer profile or open modal
  console.log("View customer profile clicked");
};

// Utility functions
const formatDateTime = (dateString) => {
  if (!dateString) return "";
  const date = new Date(dateString);
  const options = {
    weekday: "long",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  };
  return date.toLocaleString("en-US", options);
};

// Sentiment configuration
const sentimentConfig = computed(() => [
  {
    key: "positive",
    label: "Positive",
    icon: "mdi-emoticon-happy-outline",
    bgClass: "bg-success",
    inactiveColor: "#9E9E9E",
  },
  {
    key: "neutral",
    label: "Neutral",
    icon: "mdi-emoticon-neutral-outline",
    bgClass: "bg-warning",
    inactiveColor: "#9E9E9E",
  },
  {
    key: "negative",
    label: "Negative",
    icon: "mdi-emoticon-sad-outline",
    bgClass: "bg-error",
    inactiveColor: "#9E9E9E",
  },
]);

const getSentimentStyle = (sentimentKey) => {
  const isHighest = highestSentiment.value === sentimentKey;
  const config = sentimentConfig.value.find((s) => s.key === sentimentKey);

  return {
    bgClass: isHighest ? config.bgClass : "bg-secondary",
    color: isHighest ? "white" : config.inactiveColor,
  };
};

const getSentimentDescription = () => {
  const { positive, neutral, negative } = sentimentData.value;
  const highest = highestSentiment.value;

  if (highest === "positive") {
    return `Customer shows a strong interest with ${positive}% of positive score.`;
  } else if (highest === "neutral") {
    return `Customer maintains a neutral attitude with ${neutral}% of neutral score.`;
  } else {
    return `Customer expresses concerns with ${negative}% of negative score.`;
  }
};

const getCustomerInitials = (name) => {
  if (!name) return "CU";
  return name
    .split(" ")
    .map((n) => n[0])
    .join("")
    .toUpperCase()
    .slice(0, 2);
};

// Keyword highlighting functions
const selectKeyword = (keyword) => {
  selectedKeyword.value = selectedKeyword.value === keyword ? null : keyword;
};

const highlightKeywordInText = (text) => {
  if (!selectedKeyword.value || !text) return text;

  const keyword = selectedKeyword.value.toLowerCase();
  const regex = new RegExp(`(${keyword})`, "gi");

  return text.replace(regex, '<mark class="highlighted-keyword">$1</mark>');
};

const isKeywordSelected = (keyword) => {
  return selectedKeyword.value === keyword;
};

// Lifecycle
onMounted(() => {
  fetchCallSessionData();
});
</script>

<style scoped>
.highlighted-keyword {
  background-color: #ffd54f;
  color: #333;
  padding: 2px 4px;
  border-radius: 4px;
  font-weight: bold;
}

.cursor-pointer {
  cursor: pointer;
}

.cursor-pointer:hover {
  opacity: 0.8;
  transform: scale(1.02);
  transition: all 0.2s ease;
}

.admin-layout {
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 20px;
  height: 100%;
}

.row-1 {
  display: grid;
  grid-template-columns: 3fr 1fr;
  gap: 20px;
  align-items: stretch;
}

.row-2 {
  display: grid;
  grid-template-columns: 3fr 1fr;
  gap: 20px;
  align-items: stretch;
}

.topics-suggestions {
  display: grid;
  grid-template-rows: auto 1fr;
  gap: 20px;
}

/* Ensure cards in each row have equal heights */
.row-1 > .v-card,
.row-2 > .v-card,
.row-2 > .topics-suggestions {
  height: 100%;
}
</style>