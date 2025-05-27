<template>
  <v-app>
    <v-container fluid class="pa-0">
      <!-- Header Section -->
      <div class="pb-6">
        <div class="d-flex justify-space-between py-6 align-center">
          <h1 class="text-h3 font-weight-bold">Import Note to Database</h1>
          <v-btn variant="outlined" color="grey-lighten-1" style="min-width: 120px">
            <v-icon size="x-large" class="mr-1">mdi-eye-circle</v-icon>
            <p class="text-capitalize ma-0">View FAQ Database</p>
          </v-btn>
        </div>
        <p>Add new information to the database</p>
      </div>

      <!-- Main Card Content -->
      <div class="">
        <v-card class="pa-6" elevation="2" width="100%" variant="outlined" color="#1F2937" border-color="grey-lighten-2">
          <!-- Context Section -->
          <p class="text-h6 text-foreground">Context:</p>
          <br />
          <v-textarea rows="10" auto-grow label="Input Text.." style="color: #fff" input-class="text-white"></v-textarea>
          <!-- Buttons -->
          <v-row class="mt-2" justify="start" align="center" no-gutters>
            <v-col cols="auto">
              <v-btn variant="outlined" color="grey-lighten-1" class="mr-2" style="min-width: 120px">
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
    </v-container>
  </v-app>
</template>
