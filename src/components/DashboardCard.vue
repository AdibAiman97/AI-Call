<template>
  <v-card class="pa-4 custom-stat-card rounded-lg">
    <div class="d-flex align-center">
      <v-icon size="40" :color="iconColor" class="mr-3">{{ icon }}</v-icon>
      <div>
        <div class="text-caption text-medium-emphasis">{{ title }}</div>
        <div class="text-h5 font-weight-bold">{{ value }}</div>
        <div :class="change.includes('-') ? 'text-red-accent-3' : 'text-green-accent-3'" class="text-caption">{{ change }}</div>
      </div>
    </div>
  </v-card>
</template>

<script setup>
defineProps({
  icon: {
    type: String,
    default: "mdi-phone-dial-outline",
  },
  title: {
    type: String,
    default: "Title",
  },
  value: {
    type: String,
    default: "Value",
  },
  change: {
    type: String,
    default: "Statistics",
  },
    iconColor: {
        type: String,
        default: "blue-lighten-2",
    },
});
</script>
