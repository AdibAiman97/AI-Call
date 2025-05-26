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
        <p class="text-grey-darken-1 mb-0">Add new information to the database</p>
      </v-row>
      <br>
      <br>
 <!-- Main Card Content -->
  <v-row justify="center" align="start" >
  <v-col cols="12" md="12" lg="12" xl="11">
    <v-card class="pa-6" elevation="2" width="100%" variant="outlined" color="#1F2937" border-color="grey-lighten-2">
      <!-- Context Section -->
      <p class="text-h6 pa-2 text-foreground">Context:</p>
      <v-textarea
        rows="10"
        auto-grow
        class="pa-2"
        label="Input Text.."
        style=" color: #fff;"
        input-class="text-white"
      ></v-textarea>
      <!-- Buttons -->
      <v-row class="mt-2" justify="start" align="center" no-gutters>
        <v-col cols="auto">
          <v-btn variant="outlined" color="grey" class="mr-2" style="min-width:120px;">
            <span class="text-capitalize">Cancel</span>
          </v-btn>
        </v-col>
        <v-col cols="auto">
          <v-btn color="primary" variant="flat" style="min-width:120px;">
            <span class="text-capitalize">Submit</span>
          </v-btn>
        </v-col>
      </v-row>
    </v-card>
  </v-col>
</v-row>
</v-container>
</v-app>
</template>

