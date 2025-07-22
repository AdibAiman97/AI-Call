<template>
  <!-- Admin Call Summary -->
  <div>
    <div class="d-flex justify-space-between align-center mb-2">
      <h1>{{ pageTitle }}</h1>

      <!-- Call Summary Tab Buttons -->
      <div v-if="tab === 'sum'">
        <v-btn class="text-capitalize text-foreground">
          <v-icon class="mr-2"> mdi-export </v-icon> Export
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
      <span v-if="!loading && callSessionData">
        {{ customerInfo.name }} • {{ customerInfo.duration }} •
        {{ customerInfo.time }}
      </span>
      <span v-else-if="loading"> Loading call information... </span>
      <span v-else-if="error"> Error loading call data: {{ error }} </span>
    </p>
    <p class="text-foreground pb-2 mb-4" v-else-if="tab === 'app'">
      Schedule and manage customer appointments
    </p>

    <v-tabs v-model="tab" class="text-primary mb-5">
      <v-tab value="sum" class="text-capitalize text-h6 p-0 m-0">Summary</v-tab>
      <v-tab value="app" class="text-capitalize text-h6 p-0 m-0"
        >Appointment</v-tab
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
                  @click=""
                  >View Full Transcript >
                </v-btn>
              </v-card-actions>
            </v-card>

            <!-- Customer Card -->
            <v-card class="rounded-lg elevation-2 d-flex flex-column">
              <v-card-title class="text-h6 flex-shrink-0"
                >Customer</v-card-title
              >
              <v-card-text class="flex-grow-1 d-flex flex-column">
                <div
                  class="d-flex align-center mb-4"
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
                <div v-else-if="loading" class="text-center pa-4 flex-grow-1">
                  <v-progress-circular
                    indeterminate
                    size="24"
                  ></v-progress-circular>
                  <p class="mt-2">Loading customer info...</p>
                </div>

                <!-- Spacer to push button to bottom -->
                <div class="flex-grow-1"></div>

                <v-btn
                  text
                  color="primary"
                  class="text-capitalize text-background"
                  style="width: 100%"
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
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from "vue";
import { useRoute } from "vue-router";
import axios from "axios";
import { useCallStore } from "@/stores/call";
import { Plus, ListFilter } from "lucide-vue-next";

const tab = ref("sum");

useHotkey(
  "b",
  () => {
    tab.value = "sum";
  },
  { shift: true, command: true }
);

useHotkey(
  "g",
  () => {
    tab.value = "app";
  },
  { shift: true, command: true }
);
const route = useRoute();
const callStore = useCallStore();

// Reactive data properties
const callSessionData = ref(null);
const loading = ref(true);
const error = ref(null);
const selectedKeyword = ref(null);

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
    duration:
      callStore.formatTime(callSessionData.value.duration_secs) || "0 min",
    time: formatDateTime(callSessionData.value.start_time),
  };
});

const pageTitle = computed(() => {
  return tab.value === "sum" ? "Call Summary" : "Appointment";
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
