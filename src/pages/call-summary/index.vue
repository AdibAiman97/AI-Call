<template>
  <div class="d-flex justify-space-between align-center mb-2">
    <h1>Call Summary</h1>
    <v-btn class="text-capitalize text-foreground">
      <v-icon class="mr-2"> mdi-export </v-icon> Export
    </v-btn>
  </div>

  <p class="text-foreground pb-2 mb-4" v-if="!loading && callSessionData">
    {{ callSessionData.duration }} •
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
import { ref, onMounted } from "vue";
import { useRoute } from "vue-router";
import { useCallStore } from "../../stores/call";

const route = useRoute();
const summaryList = ref([]);
const customerNextSteps = ref([]);
const callSessionData = ref(null);
const loading = ref(true);

const callStore = useCallStore();

const fetchCallSessionData = async () => {
  try {
    const response = await fetch(
      `http://localhost:8000/call_session/${callStore.callSessionId}`
    );

    console.log("From Call Summary", response.data);
    if (!response.ok) {
      throw new Error("Failed to fetch call session data");
    }
    const data = await response.json();
    callSessionData.value = data;

    if (data.summarized_content) {
      summaryList.value = data.summarized_content
        .split("\n")
        .filter((item) => item.trim());
    }
    if (data.customer_suggestions) {
      customerNextSteps.value = data.customer_suggestions
        .split("\n")
        .filter((item) => item.trim());
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

const formatDateTime = (startTime) => {
  if (!startTime) return "";
  const date = new Date(startTime);
  const options = {
    weekday: "long",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  };
  return date.toLocaleString("en-US", options);
};

onMounted(() => {
  // const callSessionId = route.query.id || route.params.id
  // if (callSessionId) {
  //   fetchCallSessionData(callSessionId)
  // } else {
  //   summaryList.value = ["No call session ID provided."]
  //   customerNextSteps.value = ["Please return to the previous page and try again."]
  //   loading.value = false
  // }

  fetchCallSessionData();
});
</script>
