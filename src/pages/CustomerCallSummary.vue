<template>

  <v-container fluid class="mb-2 ms-4">
    <!-- Call Summary -->

    <v-row align="center" justify="space-between" class="elevation-2 pa-4 mb-4">
      <v-col cols="6" class="pa-4 text-foreground"><h1>Call Summary</h1></v-col>
      <v-btn @click="export"><v-icon> mdi-export </v-icon> Export</v-btn>
    </v-row>

    <!-- Summary Content -->
    <div class="d-flex flex-column">
      <v-card
        class="mb-6 flex-grow-1 rounded-lg mr-md-4 w-100 elevation-2"
      >
        <v-card-title class="text-h6 text-foreground">
          Summarized Context
        </v-card-title>
        <v-card-text class="pa-2">
          <div class="text-body-1"></div>
        </v-card-text>
        <v-card-actions class="pa-2">
          <v-btn 
            text
            color="primary"
            class="text-none px-2"
            @click="fullTranscript">View full transcript >
          </v-btn>
        </v-card-actions>
      </v-card>

      <!-- Suggestions -->
      <v-card
        class="mb-6 rounded-lg mr-md-4 w-100 elevation-2"
      >
        <v-card-title class="text-h6 text-foreground"> Suggestions </v-card-title>
        <v-card-text class="pa-2"> </v-card-text>
        <v-card-actions class="pa-2">
          <v-btn 
            text
            color="primary"
            class="text-none px-2"
            @click="fullTranscript">View full transcript >
          </v-btn>
        </v-card-actions>
      </v-card>
    </div>

    <!-- Action Buttons -->
    <div class="d-flex ga-5 mb-6">
      <v-btn outlined large @click="goToHomePage" class="px-6">
        <!-- <v-icon left>mdi-home</v-icon> -->
        Back to Home Page
      </v-btn>

      <v-btn color="primary" large @click="scheduleAppointment" class="px-6">
        <!-- <v-icon left>mdi-calendar-plus</v-icon> -->
        Schedule Appointment
      </v-btn>
    </div>
  </v-container>
</template>

<script setup>
</script>
