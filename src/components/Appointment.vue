<template>
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
    <v-calendar :interval-minutes="30" :interval-height="48" ref="calendar" v-model="value" :events="events"
      :view-mode="type" :weekdays="days">
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
const events = ref(
  [
    {
      "title": "Workshop",
      "start": new Date("2025-06-09T02:00:00.000Z"),
      "end": new Date("2025-06-09T10:00:00.000Z"),
      "color": "calendarYellow"
    },
    {
      "title": "Workshop",
      "start": new Date("2025-06-13T01:00:00.000Z"),
      "end": new Date("2025-06-13T02:00:00.000Z"),
      "color": "calendarBlue"
    },
    {
      "title": "Workshop",
      "start": new Date("2025-06-10T06:00:00.000Z"),
      "end": new Date("2025-06-10T08:00:00.000Z"),
      "color": "calendarRed"
    },
    {
      "title": "Workshop",
      "start": new Date("2025-06-08T01:00:00.000Z"),
      "end": new Date("2025-06-08T09:00:00.000Z"),
      "color": "calendarYellow"
    },
    {
      "title": "Holiday",
      "start": new Date("2025-06-12T02:00:00.000Z"),
      "end": new Date("2025-06-12T10:00:00.000Z"),
      "color": "calendarBlue"
    },
    {
      "title": "Workshop",
      "start": new Date("2025-06-13T03:00:00.000Z"),
      "end": new Date("2025-06-13T04:00:00.000Z"),
      "color": "calendarYellow"
    },
    {
      "title": "Holiday",
      "start": new Date("2025-06-11T02:00:00.000Z"),
      "end": new Date("2025-06-11T10:00:00.000Z"),
      "color": "calendarBlue"
    }
  ]
);

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

      if (start instanceof Date && end instanceof Date) {
        result.push({
          title: titles[rand(0, titles.length - 1)],
          start,
          end,
          color: colors[rand(0, colors.length - 1)],
        });
      }
    }
  }

  events.value = result;

  console.log('results', result);
  console.log(new Object(result[0].start))
  console.log(typeof new Object(result[0].start))

  // result.map((each) => {
  //   console.log(
  //     "start",
  //     each.start.toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit", })
  //   );
  //   console.log(
  //     "end",
  //     each.end.toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit", })
  //   );
  // });
}

onMounted(() => {
  const adapter = useDate();
  const today = new Date();
  const startWeek = adapter.startOfWeek(today);
  const endWeek = adapter.endOfWeek(today);

  // getEvents({ startWeek, endWeek });
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
  border-radius: 6px !important;
  /* Added for rounded corners */
  overflow: hidden;
  /* Crucial for rounded corners to work correctly */
}

/* Grouped border-drawing elements within v-calendar (EXCLUDING v-calendar__container now) */
.v-calendar :deep(.v-calendar-interval__line),
.v-calendar :deep(.v-calendar-day__row-content),
.v-calendar :deep(.v-calendar-day__row-hairline),
.v-calendar :deep(.v-calendar-day__row-hairline::after),
/* Pseudo-element needs background-color too */
.v-calendar :deep(.v-calendar-day__container),
/* This refers to the day content container, not the main calendar container */
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
  text-transform: capitalize !important;
  /* text-capitalize */
  color: #f9fafb !important;
  /* Directly set to your foreground hex color */
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
