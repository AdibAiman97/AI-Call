<template>
  <v-app>
    <v-row no-gutters>

      <!-- Main Content -->
      <v-col cols="0" class="d-flex align-center justify-center" style="height: 50vh;">
        <v-card class="pa-6" width="600" elevation="2">
          <v-card-title class="text-h6 font-weight-bold">Call Manager</v-card-title>
          <v-divider></v-divider>
          <v-card-text>
            <div class="text-h6 font-weight-bold mb-2">Import Note to Database</div>
            <div class="mb-2">Add new information to the database</div>
            <v-row class="mb-4" align="center" justify="space-between">
              <span class="font-weight-medium">Context:</span>
              <v-btn variant="outlined" size="small" color="grey-lighten-1">View FAQ Database</v-btn>
            </v-row>
            <v-textarea
              label="Input Text"
              rows="6"
              variant="outlined"
              class="mb-6"
            ></v-textarea>
            <v-row justify="end" class="mt-2">
              <v-btn variant="outlined" class="mr-2" color="grey">Cancel</v-btn>
              <v-btn color="teal-accent-3" variant="flat">Submit</v-btn>
            </v-row>
          </v-card-text>
        </v-card>
      </v-col>
    </v-row>
  </v-app>
</template>