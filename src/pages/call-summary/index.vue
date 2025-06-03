<template>
  <div class="d-flex justify-space-between align-center mb-2">
    <h1>Call Summary</h1>
    <v-btn class="text-capitalize text-foreground">
      <v-icon class="mr-2"> mdi-export </v-icon> Export
    </v-btn>
  </div>

  <p class="text-foreground pb-2 mb-4">
    Customer Name • 12 min • Today, 2:30 PM
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
const summaryList = ref([
  "The customer expressed interest in purchasing a family home and outlined their preferred neighborhood, budget range, and essential features like 3 bedrooms and a backyard. The agent acknowledged their needs and suggested a few initial listings for consideration.",
  "A prospective buyer called to inquire about a specific property listing they saw online, asking for details on its square footage, age, and recent renovations. The agent provided the requested information and offered to schedule a virtual tour or an in-person viewing.",
  "The conversation revolved around a customer who was looking to sell their current property and buy a new one. The agent discussed market conditions in their area, provided an initial valuation estimate, and explained the process of listing their home while also helping them search for a new one.",
  "A customer wanted to understand the financing options available for a property they were interested in. The agent provided an overview of loan types, connected them with a recommended mortgage broker, and explained the importance of pre-approval.",
]);

const customerNextSteps = ref([
  "Review provided listings: Carefully go through any property listings or development brochures the agent sent. Make notes on what you like and dislike about each.",
  "Get mortgage pre-approval: If you haven't already, actively pursue pre-approval for a mortgage. This clarifies your exact budget and shows agents you're a serious buyer.",
  "Prepare for viewings: For properties that interest you, create a list of specific questions (e.g., utility costs, neighborhood specifics, reason for selling) to ask during a viewing.",
  "Confirm viewing schedules: Promptly confirm your availability for any property viewings the agent proposes or request specific times that work for you.",
]);
</script>
