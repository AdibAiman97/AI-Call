<template>
  <v-navigation-drawer 
    v-model="drawerOpen" 
    :temporary="true"
    :disable-resize-watcher="true"
    v-if="isAdminRoute"
  >
    <v-list-item-group class="pa-5 text-center">
      <v-list-item
        v-for="item in menuItems"
        :key="item.id"
        :class="{ 'active-item': isActiveRoute(item.route) }"
        @click="navigate(item.route)"
      >
        <div class="menu-item-content d-flex align-center px-4 py-3 ga-3">
          <svg-icon
            type="mdi"
            class="text-foreground"
            :path="item.path"
          ></svg-icon>
          <v-list-item-title>{{ item.text }}</v-list-item-title>
        </div>
      </v-list-item>
    </v-list-item-group>
  </v-navigation-drawer>
</template>

<script setup>
import { computed } from "vue";
import { useRouter, useRoute } from "vue-router";

import SvgIcon from "@jamescoyle/vue-icon";
import { mdiViewDashboard } from "@mdi/js";
import { mdiCalendarMonth } from "@mdi/js";
import { mdiFrequentlyAskedQuestions } from "@mdi/js";

const emit = defineEmits(["update:open"]);

const router = useRouter();
const route = useRoute();

// Only show sidepanel on admin routes
const isAdminRoute = computed(() => {
  return route.path.startsWith("/admin");
});

const props = defineProps({
  open: {
    type: Boolean,
    required: true,
  },
});

const menuItems = [
  { id: 1, text: "Home", path: mdiViewDashboard, route: "/admin" },
  {
    id: 2,
    text: "FAQ Database",
    path: mdiFrequentlyAskedQuestions,
    route: "/admin/faq-database", }, ];

function navigate(route) {
  router.push(route);
  drawerOpen.value = false;
}

function isActiveRoute(route) {
  return router.currentRoute.value.path === route;
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

<style scoped>
/* Customize Vuetify hover to use foreground color */
.v-list-item {
  --v-theme-on-surface: rgb(var(--v-theme-foreground));
  border-radius: 12px !important;
  margin: 4px 16px;
  overflow: hidden;
}

.v-list-item .v-list-item__content {
  border-radius: 12px !important;
}

.v-list-item::before {
  border-radius: 12px !important;
}

.v-list-item:hover {
  background-color: rgba(var(--v-theme-foreground), 0.08) !important;
  border-radius: 12px !important;
}

.v-list-item:focus {
  background-color: rgba(var(--v-theme-foreground), 0.12) !important;
  border-radius: 12px !important;
}

.v-list-item:active {
  background-color: rgba(var(--v-theme-foreground), 0.16) !important;
  border-radius: 12px !important;
}

/* Style the content area */
.menu-item-content {
  transition: all 0.2s ease;
  width: 100%;
  border-radius: 12px;
}

.v-list-item:hover .menu-item-content {
  transform: scale(1.02);
}

.v-list-item:active .menu-item-content {
  transform: scale(0.98);
}

/* Active route highlighting */
.v-list-item.active-item {
  background-color: rgba(var(--v-theme-primary), 0.12) !important;
}

.v-list-item.active-item .menu-item-content {
  background-color: transparent !important;
  color: rgb(var(--v-theme-primary));
  font-weight: 600;
}

.v-list-item.active-item .text-foreground {
  color: rgb(var(--v-theme-primary)) !important;
}

.v-list-item.active-item:hover {
  background-color: rgba(var(--v-theme-primary), 0.16) !important;
}

.v-list-item.active-item:hover .menu-item-content {
  background-color: transparent !important;
}
</style>
