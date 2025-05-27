<template>
  <div fluid class="mb-2">
    <!-- Call Summary Header -->
    <div
      class="d-flex align-center justify-space-between mb-4"
      style="min-height: 100px"
    >
      <div class="py-3 text-foreground"><h1>Call Summary</h1>
      <p class="text-foreground">Customer Name • 12 min • Today, 2:30 PM</p>
      </div>
      
      
      <div class="d-flex">
        <v-btn @click=""><v-icon> mdi-export </v-icon> Export</v-btn>
      </div>
    </div>

    <!-- Summary Content -->
    <div class="d-flex flex-column flex-md-row">
      <!-- Left Column -->
      <div class="flex-grow-1 mr-md-4">
        <!-- Summarized Context -->
        <v-card class="mb-6 rounded-lg elevation-2">
          <v-card-title class="text-h6 text-foreground">
            Summarized Context
          </v-card-title>
          <v-card-text class="pa-2">
            <div class="text-body-1"></div>
          </v-card-text>
          <v-card-actions class="pa-2">
            <v-btn text color="primary" class="text-none px-2" @click=""
              >View full transcript >
            </v-btn>
          </v-card-actions>
        </v-card>

        <!-- Key Topics -->
        <v-card class="mb-6 flex-grow-1 rounded-lg mr-md-4 w-100 elevation-2">
          <v-card-title class="text-h6 text-capitalize"
            >Key Topics</v-card-title
          >
          <v-card-text class="pa-2">
            <v-chip-group>
              <v-chip class="ma-1" color="primary" text-color="white"
                >Pricing</v-chip
              >
              <v-chip class="ma-1" color="primary" text-color="white"
                >Enterprise</v-chip
              >
              <v-chip class="ma-1" color="primary" text-color="white"
                >Demo</v-chip
              >
              <v-chip class="ma-1" color="primary" text-color="white"
                >Follow-up</v-chip
              >
              <v-chip class="ma-1" color="primary" text-color="white"
                >Implementation</v-chip
              >
              <v-chip class="ma-1" color="primary" text-color="white"
                >Onboarding</v-chip
              >
            </v-chip-group>
          </v-card-text>
        </v-card>

        <!-- Suggestions -->
        <v-card class="mb-6 flex-grow-1 rounded-lg mr-md-4 w-100 elevation-2">
          <v-card-title class="text-h6 text-foreground"
            >Suggestions</v-card-title
          >
          <v-card-text class="pa-2">
            <div class="text-body-1"></div>
          </v-card-text>
          <v-card-actions class="pa-2">
            <v-btn text color="primary" class="text-none px-2" @click=""
              >View full transcript >
            </v-btn>
          </v-card-actions>
        </v-card>

        <!-- Action Buttons -->
        <div class="d-flex ga-5 mb-6">
          <v-btn outlined large @click="" class="px-6">
            Back to Home Page
          </v-btn>
        </div>
      </div>

      <!-- Right Column -->
      <div style="width: 300px">
        <!-- Customer Info -->
        <v-card class="mb-6 rounded-lg w-100 elevation-2">
          <v-card-title class="text-h6">Customer</v-card-title>
          <v-card-text>
            <div class="d-flex align-center mb-2">
              <v-avatar size="40" class="me-3" color="#BFC6C0" rounded="circle"
                >JD</v-avatar
              >
              <div>
                <p class="font-weight-bold mb-1 text-foreground">
                  <v-icon small class="mr-2">mdi-account</v-icon>John Doe
                </p>
                <p class="mb-1 text-foreground">
                  <v-icon small class="mr-2">mdi-email</v-icon>john.doe@acme.com
                </p>
                <p class="mb-1 text-foreground">
                  <v-icon small class="mr-2">mdi-phone</v-icon>(+60) 12-345 6789
                </p>
              </div>
            </div>
            <v-btn
              text
              color="primary"
              class="text-capitalize"
              style="width: 100%"
            >
              View Full Profile
            </v-btn>
          </v-card-text>
        </v-card>

        <!-- Sentiment Analysis -->
        <v-card class="mb-6 rounded-lg elevation-2">
          <v-card-title class="text-h6 text-foreground"
            >Sentiment Analysis</v-card-title
          >
          <v-card-text class="pa-4 d-flex flex-column ga-3">
            <!-- Positive Block -->
            <div
              class="d-flex align-center pa-3 rounded-lg"
              style="background-color: #064e3b"
            >
              <v-icon size="28" color="#4CAF50" class="mr-3"
                >mdi-emoticon-happy-outline</v-icon
              >
              <div
                class="d-flex justify-space-between align-center flex-grow-1"
              >
                <span
                  class="font-weight-medium text-body-1"
                  style="color: #2e7d32"
                  >Positive</span
                >
                <span
                  class="font-weight-bold text-body-1"
                  style="color: #2e7d32"
                  >65%</span
                >
              </div>
            </div>

            <!-- Neutral Block -->
            <div
              class="d-flex align-center pa-3 rounded-lg"
              style="background-color: #2d3748"
            >
              <v-icon size="28" color="#9E9E9E" class="mr-3"
                >mdi-emoticon-neutral-outline</v-icon
              >
              <div
                class="d-flex justify-space-between align-center flex-grow-1"
              >
                <span
                  class="font-weight-medium text-body-1"
                  style="color: #616161"
                  >Neutral</span
                >
                <span
                  class="font-weight-bold text-body-1"
                  style="color: #616161"
                  >30%</span
                >
              </div>
            </div>

            <!-- Negative Block -->
            <div
              class="d-flex align-center pa-3 rounded-lg"
              style="background-color: #7f1d1d"
            >
              <v-icon size="28" color="#F44336" class="mr-3"
                >mdi-emoticon-sad-outline</v-icon
              >
              <div
                class="d-flex justify-space-between align-center flex-grow-1"
              >
                <span
                  class="font-weight-medium text-body-1"
                  style="color: #c62828"
                  >Negative</span
                >
                <span
                  class="font-weight-bold text-body-1"
                  style="color: #c62828"
                  >5%</span
                >
              </div>
            </div>

            <!-- Summary Text -->
            <div class="text-body-2 mt-2 text-foreground">
              Customer showed strong interest in our product with some concerns
              about implementation timeline.
            </div>
          </v-card-text>
        </v-card>
      </div>
    </div>
  </div>
</template>
