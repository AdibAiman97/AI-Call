<template>
  <v-navigation-drawer v-model="drawerOpen">
    <!-- <v-list-item title="Contents" class="pa-5"></v-list-item>
    <v-divider></v-divider> -->
    <v-list-item-group class="pa-5 text-center">
      <v-list-item
        v-for="item in menuItems"
        :key="item.id"
        @click="navigate(item.route)"
      >
        <div class="d-flex align-center px-8 py-3 ga-3">
          <svg-icon type="mdi" class="text-foreground" :path="item.path"></svg-icon>
          <v-list-item-title>{{ item.text }}</v-list-item-title>
        </div>
      </v-list-item>
    </v-list-item-group>
  </v-navigation-drawer>
</template>

<script setup>
import { computed } from "vue";
import { useRouter } from "vue-router";

import SvgIcon from "@jamescoyle/vue-icon";
import { mdiViewDashboard } from "@mdi/js";
import { mdiCalendarMonth } from '@mdi/js';
import { mdiFrequentlyAskedQuestions } from '@mdi/js';

const router = useRouter();

const props = defineProps({
  open: {
    type: Boolean,
    required: true,
  },
});

const menuItems = [
  { id: 1, text: "Home", path: mdiViewDashboard, route: "/admin" },
  { id: 2, text: "FAQ Database", path: mdiFrequentlyAskedQuestions, route: "/admin/faq-database" },
];

function navigate(route) {
  router.push(route);
}

const drawerOpen = computed({
  get: () => props.open,
  set: (value) => {
    // Emit an event if you want to notify the parent about changes
    // This is optional and depends on your use case
    emit("update:open", value);
  },
});
</script>
