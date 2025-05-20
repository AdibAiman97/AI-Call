<template>
  <v-app>
    <v-container fluid class="pa-0">
      <!-- Header Section -->
      <v-row class="pa-6" justify="space-between" align="center">
        <h1 class="text-h3 font-weight-bold">Import Note to Database</h1>
        <v-btn variant="outlined" size="small" color="grey-lighten-1">
          <v-icon size="x-large" class="mr-1">mdi-eye-circle</v-icon>
          <p class="text-capitalize ma-0">View FAQ Database</p>
        </v-btn>
      </v-row>
      <!-- Subtext -->
      <v-row class="px-6">
        <p>Add new information to the database</p>
      </v-row>
      <br>
      <br>
      <!-- Main Card Content -->
      <v-row class="px-6" justify="center">
        <v-col cols="12">
          <v-card class="pa-6" elevation="2" width="100%">
            <!-- Context Section -->
            <p class="text-h6 pa-2 text-Foreground">Context:</p>
            <v-textarea
              rows="10"
              auto-grow
              variant="outlined"
              class="text-forefround pa-2"
              label="Input Text.."
              style="width: 100%;"
            ></v-textarea>
            <!-- Buttons -->
            <v-row justify="end">
              <v-btn variant="outlined" class="mr-2" color="grey">
                <p class="text-capitalize ma-0">Cancel</p>
              </v-btn>
              <v-btn color="teal-accent-3" variant="flat">
                <p class="text-capitalize ma-0">Submit</p>
              </v-btn>
            </v-row>
          </v-card>
        </v-col>
      </v-row>
    </v-container>
  </v-app>
</template>