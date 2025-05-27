<template>
  <!-- Header Section -->

  <div class="d-flex justify-space-between align-center">
    <h1>Import Note to Database</h1>
    <v-btn>
      <svg-icon
        class="text-foreground"
        type="mdi"
        :path="mdiEyeOutline"
      ></svg-icon>

      <p class="ml-2 text-capitalize">View FAQ Database</p>
    </v-btn>
  </div>

  <p class="text-foreground pb-2">Add new information to the database</p>

  <!-- Main Card Content -->
  <div class="">
    <v-card
      class="pa-6"
      elevation="2"
      width="100%"
      variant="outlined"
      color="#1F2937"
      border-color="grey-lighten-2"
    >
      <!-- Context Section -->
      <p class="text-h6 text-foreground">Context:</p>
      <br />
      <v-textarea
        rows="10"
        auto-grow
        label="Input Text.."
        style="color: #fff"
        input-class="text-white"
      ></v-textarea>
      <!-- Buttons -->
      <v-row class="mt-2" justify="start" align="center" no-gutters>
        <v-col cols="auto">
          <v-btn class="mr-2" style="min-width: 120px">
            <span class="text-capitalize">Cancel</span>
          </v-btn>
        </v-col>
        <v-col cols="auto">
          <v-btn color="primary" variant="flat" style="min-width: 120px">
            <span class="text-capitalize">Submit</span>
          </v-btn>
        </v-col>
      </v-row>
    </v-card>
  </div>
</template>

<script setup>
import SvgIcon from "@jamescoyle/vue-icon";
import { mdiEyeOutline } from "@mdi/js";
</script>
