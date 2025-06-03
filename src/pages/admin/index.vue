<template>
  <!-- Admin Call Summary -->
  <div class="mb-2">
    <div class="d-flex justify-space-between align-center mb-2">
      <h1>Call Summary</h1>
      <v-btn class="text-capitalize text-foreground">
        <v-icon class="mr-2"> mdi-export </v-icon> Export
      </v-btn>
    </div>
    <p class="text-foreground pb-2 mb-4">
      Customer Name • 12 min • Today, 2:30 PM
    </p>

    <!-- Summary Content -->
    <div class="d-flex flex-column flex-md-row">
      <!-- Left Column -->
      <v-col cols="9" class="mr-md-4 pa-0">
        <!-- Summarized Context -->
        <v-card class="mb-6 rounded-lg elevation-2">
          <v-card-title class="text-h6 text-foreground">
            Summarized Context
          </v-card-title>
          <v-card-text
            class="d-flex flex-column ga-2 text-body-1 text-secForeground"
          >
            <div v-for="item in summaryList" class="d-flex ga-2">
              <p>•</p>
              <p>{{ item }}</p>
            </div>
          </v-card-text>
          <v-card-actions class="pa-2">
            <v-btn text color="primary" class="text-capitalize px-2" @click=""
              >View Full Transcript >
            </v-btn>
          </v-card-actions>
        </v-card>

        <!-- Key Topics -->
        <v-card class="mb-6 flex-grow-1 rounded-lg mr-md-4 w-100 elevation-2">
          <v-card-title class="text-h6 text-capitalize"
            >Key Topics</v-card-title
          >
          <v-card-text class="pa-2">
            <v-chip-group>
              <v-chip class="ma-1 bg-info" text-color="white">Pricing</v-chip>
              <v-chip class="ma-1 bg-info" text-color="white"
                >Enterprise</v-chip
              >
              <v-chip class="ma-1 bg-info" text-color="white">Demo</v-chip>
              <v-chip class="ma-1 bg-info" text-color="white">Follow-up</v-chip>
              <v-chip class="ma-1 bg-info" text-color="white"
                >Implementation</v-chip
              >
              <v-chip class="ma-1 bg-info" text-color="white"
                >Onboarding</v-chip
              >
            </v-chip-group>
          </v-card-text>
        </v-card>

        <!-- Suggestions -->
        <v-card class="mb-6 flex-grow-1 rounded-lg mr-md-4 w-100 elevation-2">
          <v-card-title class="text-h6 text-foreground"
            >Suggestions</v-card-title
          >
          <v-card-text
            class="d-flex flex-column ga-2 text-body-1 text-secForeground"
          >
            <div v-for="item in agentNextSteps" class="d-flex ga-2">
              <p>•</p>
              <p>{{ item }}</p>
            </div>
          </v-card-text>
          <v-card-actions class="pa-2">
            <v-btn text color="primary" class="text-capitalize px-2" @click=""
              >View full transcript >
            </v-btn>
          </v-card-actions>
        </v-card>
      </v-col>

      <!-- Right Column -->
      <v-col class="pa-0">
        <!-- Customer Info -->
        <v-card class="mb-6 rounded-lg w-100 elevation-2">
          <v-card-title class="text-h6">Customer</v-card-title>
          <v-card-text>
            <div class="d-flex align-center mb-5">
              <v-avatar size="40" class="mr-4" color="#BFC6C0" rounded="circle"
                >JD</v-avatar
              >
              <div class="d-flex flex-column ga-3">
                <p class="font-weight-bold mb-1 text-foreground">
                  <v-icon small class="mr-2">mdi-account</v-icon>John Doe
                </p>
                <p class="mb-1 text-foreground">
                  <v-icon small class="mr-2">mdi-email</v-icon>john.doe@acme.com
                </p>
                <p class="mb-1 text-foreground">
                  <v-icon small class="mr-2">mdi-phone</v-icon>(+60) 12-345 6789
                </p>
              </div>
            </div>
            <v-btn
              text
              color="primary"
              class="text-capitalize text-background"
              style="width: 100%"
            >
              View Full Profile
            </v-btn>
          </v-card-text>
        </v-card>

        <!-- Sentiment Analysis -->
        <v-card class="mb-6 rounded-lg elevation-2">
          <v-card-title class="text-h6 text-foreground"
            >Sentiment Analysis</v-card-title
          >
          <v-card-text class="pa-4 d-flex flex-column ga-3">
            <!-- Positive Block -->
            <div class="d-flex align-center pa-3 rounded-lg bg-success">
              <v-icon size="28" color="#4CAF50" class="mr-3"
                >mdi-emoticon-happy-outline</v-icon
              >
              <div
                class="d-flex justify-space-between align-center flex-grow-1"
              >
                <span
                  class="font-weight-medium text-body-1"
                  style="color: #4caf50"
                  >Positive</span
                >
                <span
                  class="font-weight-bold text-body-1"
                  style="color: #4caf50"
                  >65%</span
                >
              </div>
            </div>

            <!-- Neutral Block -->
            <div class="d-flex align-center pa-3 rounded-lg bg-secondary">
              <v-icon size="28" color="#9E9E9E" class="mr-3"
                >mdi-emoticon-neutral-outline</v-icon
              >
              <div
                class="d-flex justify-space-between align-center flex-grow-1"
              >
                <span
                  class="font-weight-medium text-body-1"
                  style="color: #9e9e9e"
                  >Neutral</span
                >
                <span
                  class="font-weight-bold text-body-1"
                  style="color: #9e9e9e"
                  >30%</span
                >
              </div>
            </div>

            <!-- Negative Block -->
            <div class="d-flex align-center pa-3 rounded-lg bg-error">
              <v-icon size="28" color="#F44336" class="mr-3"
                >mdi-emoticon-sad-outline</v-icon
              >
              <div
                class="d-flex justify-space-between align-center flex-grow-1"
              >
                <span
                  class="font-weight-medium text-body-1"
                  style="color: #f44336"
                  >Negative</span
                >
                <span
                  class="font-weight-bold text-body-1"
                  style="color: #f44336"
                  >5%</span
                >
              </div>
            </div>

            <!-- Summary Text -->
            <div class="mt-2 text-foreground">
              Customer showed strong interest in our product with some concerns
              about implementation timeline.
            </div>
          </v-card-text>
        </v-card>
      </v-col>
    </div>
  </div>

  <v-divider class="my-6"></v-divider>

  <!-- Calendar -->
  <div>
    <h1 class="d-flex justify-space-between text-foreground align-center mb-2">
      Appointment
      <div class="d-flex ga-2">
        <v-btn class="text-capitalize text-foreground">
          <ListFilter :size="20" class="mr-2" />
          Filter
        </v-btn>
        <v-btn class="bg-primary text-background text-capitalize">
          <Plus :size="20" class="mr-2" />
          Add Appointment
        </v-btn>
      </div>
    </h1>
    <p class="text-foreground mb-4">Manage your upcoming customer meetings</p>
    <v-calendar
      :interval-minutes="30"
      :interval-height="48"
      ref="calendar"
      v-model="value"
      :events="events"
      :view-mode="type"
      :weekdays="days"
    >
    </v-calendar>
  </div>
</template>

<script setup>
import { useDate } from "vuetify";
import { Plus, ListFilter } from "lucide-vue-next";

// Values
const type = ref("week");
const days = ref([0, 1, 2, 3, 4, 5, 6]);
const value = ref(new Date());
const events = ref([
  // {
  //   title: 'Time',
  //   start: new Date(1747969200000),
  //   end: new Date(1747969200000 + 3600000),
  //   color: 'primary'
  // }
]);

const summaryList = ref([
  "The discussion centered on a customer's concerns about property taxes and maintenance costs for a potential investment property. The agent offered insights into typical expenses in the area and provided resources for estimating ongoing costs.",
  "The customer was a first-time homebuyer seeking guidance on the entire process, from identifying suitable properties to making an offer and closing the deal. The agent patiently walked them through each step, addressing their questions about market trends and legal procedures.",
  "The customer and agent discussed a new development project, with the customer inquiring about floor plans, amenities, and the estimated completion date. The agent provided brochures and highlighted the unique selling points of the development.",
  "A customer had seen a property that was slightly outside their budget but was very keen on it. The agent discussed negotiation strategies and explored potential concessions or alternative financing solutions to try and make the property more attainable.",
]);

const agentNextSteps = ref([
  "Curate and send relevant listings: Based on the customer's expressed needs (neighborhood, budget, features), send a targeted list of available properties or new development information.",
  "Initiate mortgage broker connection: If the customer needs financing assistance, proactively connect them with a trusted mortgage broker (with the customer's consent).",
  "Schedule property viewings: Propose and arrange property visits or virtual tours for the listings the customer shows interest in.",
  "Prepare detailed property information: For scheduled viewings, have comprehensive information packets ready (e.g., floor plans, property history, recent comparable sales, tax info).",
]);

console.log(events.value.start);

// Options
const types = ["week", "day"];
const weekdays = [
  { title: "Mon-Sun", value: [1, 2, 3, 4, 5, 6, 0] },
  { title: "Mon-Fri", value: [1, 2, 3, 4, 5] },
];

const colors = [
  "calendarGreen",
  "calendarRed",
  "calendarBlue",
  "calendarYellow",
];
const titles = ["Meeting", "Holiday", "Workshop", "Appointment"];

function rand(a, b) {
  return Math.floor((b - a + 1) * Math.random()) + a;
}

// Random Duration
function randomDuration() {
  const duration = ["60", "120", "480"]; // in minutes
  const ms = 60 * 1000; // min to ms
  const randDur = rand(0, duration.length - 1);
  const eventDuration = parseInt(duration[randDur]) * ms;

  return eventDuration;
}

// Random Day 9am - 7pm
function randomDay(min, max) {
  const randDay = rand(min, max);
  const startDay = new Date(randDay);
  startDay.setHours(9, 0, 0, 0);
  const endDay = new Date(randDay);
  endDay.setHours(19, 0, 0, 0);

  return { startDay, endDay };
}

function randomEvent(startDay, endDay, eventDuration) {
  const randStart = rand(startDay.getTime(), endDay.getTime() - eventDuration);
  const start = new Date(randStart - (randStart % 3600000));
  const end = new Date(start.getTime() + eventDuration);

  return { start, end };
}

function checkCalendar(result, max, min, eventDuration) {
  const { startDay, endDay } = randomDay(min, max);
  const { start, end } = randomEvent(startDay, endDay, eventDuration);

  const hasOverlap = result.some(
    (event) => start < event.end && end > event.start
  );

  if (!hasOverlap) {
    return { start, end, status: false };
  } else {
    return { status: true };
  }
}

function getEvents({ startWeek, endWeek }) {
  const result = [];
  // SoW & EoW
  const min = startWeek.getTime();
  const max = endWeek.getTime();
  const eventCount = 20;
  for (let i = 0; i < eventCount; i++) {
    const eventDuration = randomDuration();
    // const { startDay, endDay } = randomDay(min, max)
    // const { start, end } = randomEvent(startDay, endDay, eventDuration)
    const isBooked = checkCalendar(result, max, min, eventDuration);

    if (isBooked.status === false) {
      const { start, end } = isBooked;
      result.push({
        title: titles[rand(0, titles.length - 1)],
        start,
        end,
        color: colors[rand(0, colors.length - 1)],
      });
    }
  }

  events.value = result;
  console.clear();
  console.log(result);
  result.map((each) => {
    console.log(
      "start",
      each.start.toLocaleString([], {
        weekday: "short", // e.g., Mon, Tue
        hour: "2-digit",
        minute: "2-digit",
      })
    );

    // console.log('start',each.start.getTime())

    console.log(
      "end",
      each.end.toLocaleString([], {
        weekday: "short",
        hour: "2-digit",
        minute: "2-digit",
      })
    );
  });
}

onMounted(() => {
  const adapter = useDate();
  const today = new Date();
  const startWeek = adapter.startOfWeek(today);
  const endWeek = adapter.endOfWeek(today);

  getEvents({ startWeek, endWeek });
});
</script>

<style scoped>
/* Specific styles for internal events */
:deep(.v-calendar-internal-event) {
  margin-top: 0px !important;
  height: 48px !important;
  border-color: #374151 !important;
}

/* Outer container border and rounded corners */
.v-calendar :deep(.v-calendar__container) {
  border-color: #374151 !important;
  border-radius: 6px !important; /* Added for rounded corners */
  overflow: hidden; /* Crucial for rounded corners to work correctly */
}

/* Grouped border-drawing elements within v-calendar (EXCLUDING v-calendar__container now) */
.v-calendar :deep(.v-calendar-interval__line),
.v-calendar :deep(.v-calendar-day__row-content),
.v-calendar :deep(.v-calendar-day__row-hairline),
.v-calendar :deep(.v-calendar-day__row-hairline::after), /* Pseudo-element needs background-color too */
.v-calendar :deep(.v-calendar-day__container), /* This refers to the day content container, not the main calendar container */
.v-calendar :deep(.v-calendar-day__row-without-label),
.v-calendar :deep(.v-calendar-day__row-with-label),
.v-calendar :deep(.v-calendar-weekly__day-grid),
.v-calendar :deep(.v-calendar-weekly__day),
.v-calendar :deep(.v-calendar-weekly__head-weekday),
.v-calendar :deep(.v-calendar-daily__day-name),
.v-calendar :deep(.v-calendar-daily__intervals-body),
.v-calendar :deep(.v-calendar-daily__time),
.v-calendar :deep(.v-calendar-header),
.v-calendar :deep(.v-event-more),
.v-calendar :deep(.v-event) {
  border-color: #374151 !important;
}

/* Styling buttons in v-calendar-header with direct hex color --- */
.v-calendar :deep(.v-calendar-header .v-calendar-header__today) {
  text-transform: capitalize !important; /* text-capitalize */
  color: #f9fafb !important; /* Directly set to your foreground hex color */
  /* border: 0.5px solid #4b5563 !important; */
  border: none !important;
  background-color: #1f2937;
}

/* For specific Vuetify button variants, you might need to override their background */
.v-calendar :deep(.v-calendar-header .v-btn.v-btn--variant-text) {
  background-color: transparent !important;
}
.v-calendar :deep(.v-calendar-header .v-btn.v-btn--variant-tonal) {
  background-color: transparent !important;
}
</style>
