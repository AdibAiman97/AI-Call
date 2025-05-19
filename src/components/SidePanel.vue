<template>
  <v-navigation-drawer v-model="drawerOpen">
    <v-list-item title="Contents" class="pa-5"></v-list-item>
    <v-divider></v-divider>
    <v-list-item-group class="">
      <v-list-item
        v-for="item in menuItems"
        :key="item.id"
        @click="navigate(item.route)"
      >
        <v-list-item-title>{{ item.text }}</v-list-item-title>
      </v-list-item>
    </v-list-item-group>
  </v-navigation-drawer>
</template>

<script setup>
import { computed } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter();

const props = defineProps({
  open: {
    type: Boolean,
    required: true
  }
});

const menuItems = [
  { id: 1, text: "Home", route: "/" },
  { id: 2, text: "About", route: "/about" },
];

function navigate(route) {
  router.push(route);
}

const drawerOpen = computed({
  get: () => props.open,
  set: (value) => {
    // Emit an event if you want to notify the parent about changes
    // This is optional and depends on your use case
    emit('update:open', value);
  }
});
</script>