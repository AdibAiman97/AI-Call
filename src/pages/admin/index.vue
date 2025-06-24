<template>
  <!-- Admin Call Summary -->
  <div>
    <div class="d-flex justify-space-between align-center mb-2">
      <h1>Call Summary</h1>
      <v-btn class="text-capitalize text-foreground">
        <v-icon class="mr-2"> mdi-export </v-icon> Export
      </v-btn>
    </div>
    <p class="text-foreground pb-2 mb-4" v-if="!loading && callSessionData">
      {{ customerInfo.name }} • {{ customerInfo.duration }} •
      {{ customerInfo.time }}
    </p>
    <p class="text-foreground pb-2 mb-4" v-else-if="loading">
      Loading call information...
    </p>
    <p class="text-foreground pb-2 mb-4" v-else-if="error">
      Error loading call data: {{ error }}
    </p>

    <v-tabs v-model="tab" class="text-primary mb-5">
      <v-tab value="sum" class="text-capitalize text-h6 p-0 m-0">Summary</v-tab>
      <v-tab value="app" class="text-capitalize text-h6 p-0 m-0"
        >Appointment</v-tab
      >
    </v-tabs>

    <v-tabs-window v-model="tab" class="">
      <v-tabs-window-item value="sum">
        <div class="d-flex ga-5" style="height: 100%">
          <!-- Left Column: Main Content -->
          <v-col cols="9" class="pa-0 d-flex flex-column ga-5">
            <!-- Row 1: Summarized Content (matches Customer height) -->
            <div
              class="d-flex flex-column flex-md-row"
              style="height: 280px"
            >
              <v-card
                class="rounded-lg elevation-2 d-flex flex-column"
                style="flex: 1"
              >
                <v-card-title class="text-h6 text-foreground flex-shrink-0">
                  Summarized Context
                </v-card-title>
                <v-card-text
                  class="d-flex flex-column ga-2 text-body-1 text-secForeground flex-grow-1 overflow-y-auto"
                >
                  <div v-if="!loading && summaryList.length > 0">
                    <div v-for="item in summaryList" class="d-flex ga-2 mb-2">
                      <p>•</p>
                      <p>{{ item }}</p>
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
                    @click=""
                    >View Full Transcript >
                  </v-btn>
                </v-card-actions>
              </v-card>
            </div>

            <!-- Row 2: Key Topics & Suggestions (matches Sentiment Analysis height) -->
            <div class="d-flex flex-column ga-5" style="flex: 1">
              <!-- Key Topics -->
              <v-card class="rounded-lg elevation-2">
                <v-card-title class="text-h6 text-capitalize flex-shrink-0"
                  >Key Topics</v-card-title
                >
                <v-card-text
                  class="px-4"
                  style=""
                >
                  <v-chip-group v-if="!loading && keywordsList.length > 0">
                    <v-chip
                      v-for="keyword in keywordsList"
                      :key="keyword"
                      class="bg-info ma-0 mr-3"
                      text-color="white"
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
                  >Suggestions</v-card-title
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
                    @click=""
                    >View full transcript >
                  </v-btn>
                </v-card-actions>
              </v-card>
            </div>
          </v-col>

          <!-- Right Column: Customer & Sentiment -->
          <v-col class="pa-0 d-flex flex-column ga-5">
            <!-- Customer Card (fixed height) -->
            <v-card
              class="rounded-lg w-100 elevation-2"
              style="height: 280px"
            >
              <v-card-title class="text-h6 flex-shrink-0"
                >Customer</v-card-title
              >
              <v-card-text
                class="d-flex flex-column justify-space-between"
                style="height: calc(100% - 64px)"
              >
                <div
                  class="d-flex align-center mb-5"
                  v-if="!loading && callSessionData"
                >
                  <v-avatar
                    size="40"
                    class="mr-4"
                    color="#BFC6C0"
                    rounded="circle"
                  >
                    {{ getCustomerInitials(customerInfo.name) }}
                  </v-avatar>
                  <div class="d-flex flex-column ga-3">
                    <p class="font-weight-bold mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-account</v-icon
                      >{{ customerInfo.name }}
                    </p>
                    <p class="mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-email</v-icon
                      >{{
                        customerInfo.name.toLowerCase().replace(" ", ".")
                      }}@customer.com
                    </p>
                    <p class="mb-1 text-foreground">
                      <v-icon small class="mr-2">mdi-phone</v-icon>(+60) 12-345
                      6789
                    </p>
                  </div>
                </div>
                <div v-else-if="loading" class="text-center pa-4">
                  <v-progress-circular
                    indeterminate
                    size="24"
                  ></v-progress-circular>
                  <p class="mt-2">Loading customer info...</p>
                </div>
                <v-btn
                  text
                  color="primary"
                  class="text-capitalize text-background mt-auto"
                  style="width: 100%"
                >
                  View Full Profile
                </v-btn>
              </v-card-text>
            </v-card>

            <!-- Sentiment Analysis Card (flexible height) -->
            <v-card
              class="rounded-lg elevation-2 d-flex flex-column"
              style="flex: 1"
            >
              <v-card-title class="text-h6 text-foreground flex-shrink-0"
                >Sentiment Analysis</v-card-title
              >
              <v-card-text
                class="pa-4 d-flex flex-column ga-3 flex-grow-1"
                v-if="!loading"
              >
                <!-- Positive Sentiment -->
                <div
                  class="d-flex align-center pa-3 rounded-lg"
                  :class="
                    highestSentiment === 'positive'
                      ? 'bg-success'
                      : 'bg-grey-lighten-3'
                  "
                >
                  <v-icon
                    size="28"
                    :color="
                      highestSentiment === 'positive' ? '#4CAF50' : '#9E9E9E'
                    "
                    class="mr-3"
                  >
                    mdi-emoticon-happy-outline
                  </v-icon>
                  <div
                    class="d-flex justify-space-between align-center flex-grow-1"
                  >
                    <span
                      class="font-weight-medium text-body-1"
                      :style="{
                        color:
                          highestSentiment === 'positive'
                            ? '#4caf50'
                            : '#9e9e9e',
                      }"
                    >
                      Positive
                    </span>
                    <span
                      class="font-weight-bold text-body-1"
                      :style="{
                        color:
                          highestSentiment === 'positive'
                            ? '#4caf50'
                            : '#9e9e9e',
                      }"
                    >
                      {{ sentimentData.positive }}%
                    </span>
                  </div>
                </div>

                <!-- Neutral Sentiment -->
                <div
                  class="d-flex align-center pa-3 rounded-lg"
                  :class="
                    highestSentiment === 'neutral'
                      ? 'bg-warning'
                      : 'bg-grey-lighten-3'
                  "
                >
                  <v-icon
                    size="28"
                    :color="
                      highestSentiment === 'neutral' ? '#FF9800' : '#9E9E9E'
                    "
                    class="mr-3"
                  >
                    mdi-emoticon-neutral-outline
                  </v-icon>
                  <div
                    class="d-flex justify-space-between align-center flex-grow-1"
                  >
                    <span
                      class="font-weight-medium text-body-1"
                      :style="{
                        color:
                          highestSentiment === 'neutral'
                            ? '#FF9800'
                            : '#9e9e9e',
                      }"
                    >
                      Neutral
                    </span>
                    <span
                      class="font-weight-bold text-body-1"
                      :style="{
                        color:
                          highestSentiment === 'neutral'
                            ? '#FF9800'
                            : '#9e9e9e',
                      }"
                    >
                      {{ sentimentData.neutral }}%
                    </span>
                  </div>
                </div>

                <!-- Negative Sentiment -->
                <div
                  class="d-flex align-center pa-3 rounded-lg"
                  :class="
                    highestSentiment === 'negative'
                      ? 'bg-error'
                      : 'bg-grey-lighten-3'
                  "
                >
                  <v-icon
                    size="28"
                    :color="
                      highestSentiment === 'negative' ? '#F44336' : '#9E9E9E'
                    "
                    class="mr-3"
                  >
                    mdi-emoticon-sad-outline
                  </v-icon>
                  <div
                    class="d-flex justify-space-between align-center flex-grow-1"
                  >
                    <span
                      class="font-weight-medium text-body-1"
                      :style="{
                        color:
                          highestSentiment === 'negative'
                            ? '#f44336'
                            : '#9e9e9e',
                      }"
                    >
                      Negative
                    </span>
                    <span
                      class="font-weight-bold text-body-1"
                      :style="{
                        color:
                          highestSentiment === 'negative'
                            ? '#f44336'
                            : '#9e9e9e',
                      }"
                    >
                      {{ sentimentData.negative }}%
                    </span>
                  </div>
                </div>

                <div class="mt-2 text-foreground" v-if="callSessionData">
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
          </v-col>
        </div>
      </v-tabs-window-item>

      <v-tabs-window-item value="app">
        <Appointment />
      </v-tabs-window-item>
    </v-tabs-window>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import { useRoute } from "vue-router";
import axios from "axios";
import { useCallStore } from "@/stores/call";

const tab = ref("sum");
const route = useRoute();
const callStore = useCallStore();

// Reactive data properties
const callSessionData = ref(null);
const loading = ref(true);
const error = ref(null);

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
  return {
    name: callSessionData.value.cust_id || "Unknown Customer",
    duration: callSessionData.value.duration || "0 min",
    time: formatDateTime(callSessionData.value.start_time),
  };
});

// API function
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

    const response = await axios.get(
      `${import.meta.env.VITE_API_BASE_URL}/call_session/${callSessionId}`
    );
    callSessionData.value = response.data;
  } catch (err) {
    console.error("Error fetching call session data:", err);
    error.value = err.message;
  } finally {
    loading.value = false;
  }
};

// Fallback to get the latest call session
const fetchLatestCallSession = async () => {
  try {
    console.log("Fetching latest call session...");
    const response = await axios.get(
      `${import.meta.env.VITE_API_BASE_URL}/call_session`
    );
    console.log("All sessions response:", response.data);

    if (response.data && response.data.length > 0) {
      // Get the most recent call session (assuming they're ordered by ID)
      const latestSession = response.data[response.data.length - 1];
      console.log("Latest session found:", latestSession);
      callSessionData.value = latestSession;
      callStore.callSessionId = latestSession.id;
    } else {
      console.log("No call sessions found, using demo data");
      // No call sessions found, use demo data
      callSessionData.value = getDemoData();
    }
  } catch (err) {
    console.error("Error fetching latest call session:", err);
    console.log("Falling back to demo data due to error");
    // Fallback to demo data
    callSessionData.value = getDemoData();
  }
};

// Demo data fallback
const getDemoData = () => ({
  id: "demo",
  cust_id: "Demo Customer",
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

const getSentimentDescription = () => {
  const { positive, neutral, negative } = sentimentData.value;
  const highest = highestSentiment.value;

  if (highest === "positive") {
    return `Customer showed strong positive sentiment (${positive}%) during the call with ${neutral}% neutral and ${negative}% negative responses.`;
  } else if (highest === "neutral") {
    return `Customer maintained a neutral tone (${neutral}%) throughout the call with ${positive}% positive and ${negative}% negative responses.`;
  } else {
    return `Customer expressed concerns with ${negative}% negative sentiment, balanced by ${positive}% positive and ${neutral}% neutral responses.`;
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

// Lifecycle
onMounted(() => {
  fetchCallSessionData();
});
</script>
