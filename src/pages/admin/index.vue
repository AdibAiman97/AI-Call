<template>
  <v-app>
    <v-main>
      <div>
        <h1 class="d-flex justify-space-between text-foreground align-center">
          Customer Calls Dashboard
          <!-- <div class="d-flex ga-2">
            <v-btn
              class="text-capitalize text-foreground border border-foreground"
            >
              <v-icon>mdi-filter-variant</v-icon>
              Filter
            </v-btn>
            <v-btn class="text-capitalize text-background bg-primary">
              <v-icon>mdi-export</v-icon>
              Export
            </v-btn>
          </div> -->
        </h1>
        <p class="text-foreground">Customers who have interacted with our AI agent</p>

        <v-row class="mb-3 mt-1">
          <v-col v-for="stat in statistics" cols="12" sm="6" md="3">
            <DashboardCard
              :key="stat.title"
              :icon="stat.icon"
              :title="stat.title"
              :value="stat.value"
              :change="stat.change"
              :iconColor="stat.iconColor"
            />
          </v-col>
        </v-row>

        <v-card class="rounded-lg">
          <v-table class="">
            <thead class="bg-secondary">
              <tr>
                <th class="text-left" style="width: 30%">Customer Name</th>
                <th class="text-center" style="width: 20%">Call Date</th>
                <th class="text-center" style="width: 20%">Duration</th>
                <th class="text-center" style="width: 20%">Sentiment</th>
                <th class="text-center" style="width: fit-content">Action</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="item in customers" :key="item.name">
                <td>
                  <div class="d-flex align-center py-4">
                    <v-avatar :color="item.avatarColor" size="40" class="mr-3">
                      <span class="text-h6">{{ item.initials }}</span>
                    </v-avatar>
                    <div>
                      <div class="font-weight-bold">{{ item.name }}</div>
                      <div class="text-caption text-medium-emphasis">
                        {{ item.email }}
                      </div>
                    </div>
                  </div>
                </td>
                <td align="center">
                  <div>{{ item.callDate }}</div>
                  <div class="text-caption text-medium-emphasis">
                    {{ item.callTime }}
                  </div>
                </td>
                <td align="center" class="text-center">{{ item.duration }}</td>
                <td align="center">
                  <div
                    :class="`bg-${getSentimentColor(item.sentiment)}`"
                    class="w-50 text-center rounded-xl"
                  >
                    {{ item.sentiment }}
                  </div>
                </td>
                <td class="text-center">
                  <v-btn to="/admin/call-summary" icon variant="text" size="small">
                    <svg-icon class="text-foreground" type="mdi" :path="mdiEyeOutline"></svg-icon>
                  </v-btn>
                </td>
              </tr>
            </tbody>
          </v-table>

          <v-row class="px-4 mt-4 mb-1 align-center">
            <v-col cols="12" sm="6" class="text-caption text-medium-emphasis">
              Showing 1-3 of 89 customers
            </v-col>
            <v-col
              cols="12"
              sm="6"
              class="d-flex justify-sm-end justify-center"
            >
              <v-pagination
                v-model="page"
                :length="totalPages"
                :total-visible="5"
                density="compact"
              ></v-pagination>
            </v-col>
          </v-row>
        </v-card>
      </div>
    </v-main>
  </v-app>
</template>

<script setup>
import { ref, computed } from "vue";
import DashboardCard from "@/components/DashboardCard.vue";
import SvgIcon from '@jamescoyle/vue-icon';
import { mdiEyeOutline } from '@mdi/js';

const page = ref(1); // Current page for pagination
const itemsPerPage = ref(5); // Items per page for pagination

const statistics = ref([
  {
    icon: "mdi-phone-dial-outline",
    title: "Total Calls",
    value: "247",
    change: "+12% from last week",
    iconColor: "blue-lighten-2",
  },
  {
    icon: "mdi-account-group-outline",
    title: "Active Customers",
    value: "89",
    change: "+8% from last week",
    iconColor: "cyan-lighten-2",
  },
  {
    icon: "mdi-clock-outline",
    title: "Avg Duration",
    value: "8:42",
    change: "-2% from last week",
    iconColor: "purple-lighten-2",
  },
  {
    icon: "mdi-check-circle-outline",
    title: "Success Rate",
    value: "94.2%",
    change: "+3% from last week",
    iconColor: "light-green-lighten-2",
  },
]);

// Dummy data for the table
const customers = ref([
  {
    name: "John Doe",
    email: "john.doe@acme.com",
    initials: "JD",
    avatarColor: "blue",
    company: "Acme Inc.",
    position: "Product Manager",
    callDate: "Today",
    callTime: "2:30 PM",
    duration: "12:34",
    status: "Completed",
    sentiment: "Positive",
  },
  {
    name: "Sarah Johnson",
    email: "sarah.j@techcorp.com",
    initials: "SJ",
    avatarColor: "red-lighten-2",
    company: "TechCorp",
    position: "CTO",
    callDate: "Yesterday",
    callTime: "3:45 PM",
    duration: "8:22",
    status: "Completed",
    sentiment: "Neutral",
  },
  {
    name: "Michael Smith",
    email: "m.smith@globalsol.com",
    initials: "MS",
    avatarColor: "purple",
    company: "Global Solutions",
    position: "VP Sales",
    callDate: "May 14",
    callTime: "11:20 AM",
    duration: "15:47",
    status: "Follow-up",
    sentiment: "Positive",
  },
  {
    name: "Emily Davis",
    email: "emily.d@innovate.com",
    initials: "ED",
    avatarColor: "deep-purple-lighten-2",
    company: "Innovate LLC",
    position: "CEO",
    callDate: "May 13",
    callTime: "4:15 PM",
    duration: "6:33",
    status: "Missed",
    sentiment: "Negative",
  },
  {
    name: "Robert Wilson",
    email: "r.wilson@startup.io",
    initials: "RW",
    avatarColor: "green",
    company: "Startup.io",
    position: "Founder",
    callDate: "May 12",
    callTime: "9:30 AM",
    duration: "22:15",
    status: "Completed",
    sentiment: "Positive",
  },
  // Add more dummy data as needed
]);

// Computed property for total pages (for pagination)
const totalPages = computed(() => Math.ceil(89 / itemsPerPage.value)); // Assuming 89 total customers from the image

// Function to determine chip color based on sentiment
const getSentimentColor = (sentiment) => {
  switch (sentiment.toLowerCase()) {
    case "positive":
      return "success";
    case "negative":
      return "error";
    case "neutral":
      return "secondary";
  }
};
</script>
