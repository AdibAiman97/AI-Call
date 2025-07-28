<template>
  <!-- Calendar -->
  <div>
    <!-- Loading state -->
    <v-progress-linear v-if="loading" indeterminate color="primary" class="mb-4"></v-progress-linear>

    <!-- Error state -->
    <v-alert v-if="error" type="warning" class="mb-4" dismissible @click:close="error = null">
      <div class="d-flex align-center">
        <v-icon class="mr-2">mdi-alert-circle</v-icon>
        <div>
          <strong>Appointment Service Issue</strong>
          <br>
          <small>{{ error }}</small>
        </div>
      </div>
    </v-alert>
    
    <v-calendar :interval-minutes="30" :interval-height="48" ref="calendar" v-model="value" :events="events"
      :view-mode="type" :weekdays="days">
    </v-calendar>
  </div>
</template>

<script setup>
import { useDate } from "vuetify";
import { Plus, ListFilter } from "lucide-vue-next";

// API Configuration - Use environment variable if available
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Values
const loading = ref(false);
const error = ref(null);
const type = ref("week");
const days = ref([0, 1, 2, 3, 4, 5, 6]);
const value = ref(new Date());
// const events = ref([]);
const events = ref(
  [
    {
      "title": "WS2",
      "start": new Date("2025-06-29T02:00:00.000Z"),
      "end": new Date("2025-06-09T10:00:00.000Z"),
      "color": "calendarYellow"
    },
    {
      "title": "Workshop",
      "start": new Date("2025-07-04T01:00:00.000Z"),
      "end": new Date("2025-06-13T02:00:00.000Z"),
      "color": "calendarBlue"
    },
    {
      "title": "Workshop",
      "start": new Date("2025-07-05T03:00:00.000Z"),
      "end": new Date("2025-07-05T07:00:00.000Z"),
      "color": "calendarRed"
    },
    {
      "title": "WS1",
      "start": new Date("2025-06-30T01:00:00.000Z"),
      "end": new Date("2025-06-08T09:00:00.000Z"),
      "color": "calendarYellow"
    },
    {
      "title": "Holiday",
      "start": new Date("2025-07-02T02:00:00.000Z"),
      "end": new Date("2025-06-12T10:00:00.000Z"),
      "color": "calendarBlue"
    },
    {
      "title": "Workshop",
      "start": new Date("2025-07-01T03:00:00.000Z"),
      "end": new Date("2025-06-13T04:00:00.000Z"),
      "color": "calendarYellow"
    },
    {
      "title": "Holiday",
      "start": new Date("2025-07-02T02:00:00.000Z"),
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

// Color mapping for appointment titles
const titleColorMap = {
  "Meeting": "calendarBlue",
  "Holiday": "calendarRed", 
  "Workshop": "calendarGreen",
  "Appointment": "calendarYellow"
};

// Function to get color for appointment
function getAppointmentColor(appointment) {
  // Try to map by title first
  if (titleColorMap[appointment.title]) {
    return titleColorMap[appointment.title];
  }
  // Fall back to using appointment ID for consistent color
  const colorIndex = appointment.id % colors.length;
  return colors[colorIndex];
}

// Test API connection
async function testApiConnection() {
  try {
    console.log('🔍 Testing API connection to:', API_BASE_URL);
    const response = await fetch(`${API_BASE_URL}/appointments/`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    console.log('🔍 API Test Response status:', response.status);
    
    if (response.ok) {
      const data = await response.json();
      console.log('🔍 API Test successful. Sample data:', data);
      return true;
    } else {
      console.warn('🔍 API Test failed with status:', response.status);
      return false;
    }
  } catch (error) {
    console.error('🔍 API Test connection failed:', error);
    return false;
  }
}

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

  console.log("results", result);
  console.log(new Object(result[0].start));
  console.log(typeof new Object(result[0].start));

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

async function fetchAppointments(startDate, endDate) {
  loading.value = true;
  error.value = null;
  
  try {
    const startTime = new Date(startDate).toISOString();
    const endTime = new Date(endDate).toISOString();

    console.log('📅 Fetching appointments for date range:', {
      startTime,
      endTime,
      apiUrl: API_BASE_URL
    });
    
    const url = `${API_BASE_URL}/appointments/date-range?start_date=${encodeURIComponent(startTime)}&end_date=${encodeURIComponent(endTime)}`;
    console.log('📅 API URL:', url);
    
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    console.log('📅 API Response status:', response.status);

    if (!response.ok) {
      const errorText = await response.text();
      console.error('📅 API Error Response:', errorText);
      throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
    }

    const appointments = await response.json();
    console.log('📅 Raw appointments data:', appointments);
    
    // Handle empty response
    if (!Array.isArray(appointments)) {
      console.warn('📅 Appointments response is not an array:', appointments);
      events.value = [];
      return;
    }

    if (appointments.length === 0) {
      console.log('📅 No appointments found for the specified date range');
      events.value = [];
      return;
    }
    
    // Transform API data to calendar event format
    events.value = appointments.map(appointment => {
      console.log('📅 Processing appointment:', appointment);
      
      return {
        id: appointment.id,
        title: appointment.title,
        start: new Date(appointment.start_time),
        end: new Date(appointment.end_time),
        color: getAppointmentColor(appointment),
        callSessionId: appointment.call_session_id,
        createdAt: appointment.created_at,
        updatedAt: appointment.updated_at
      };
    });

    console.log('📅 Transformed calendar events:', events.value);
    
  } catch (err) {
    console.error('📅 Error fetching appointments:', err);
    error.value = `Failed to load appointments: ${err.message}`;
    
    // Fall back to empty events array
    events.value = [];
  } finally {
    loading.value = false;
  }
}

onMounted(async () => {
  const adapter = useDate();
  const today = new Date().setHours(0,0,0,0);
  
  const startWeek = adapter.startOfWeek(today).setHours(0,0,0);
  const endWeek = adapter.endOfWeek(today).setHours(23,59,59);

  console.log('📅 Calendar component mounted. Testing API connection...');
  
  // Test API connection first
  const apiConnected = await testApiConnection();
  
  if (apiConnected) {
    console.log('📅 API connection successful. Fetching appointments...');
    // Initial load of appointments for current week
    await fetchAppointments(startWeek, endWeek);
  } else {
    console.warn('📅 API connection failed. Using sample data...');
    error.value = 'Unable to connect to appointment service. Using sample data.';
    // Keep the existing sample events as fallback
    // events.value remains with the hardcoded sample data
  }
});

// Expose functions for potential external use
defineExpose({
  fetchAppointments,
  // fetchAllAppointments,
  refreshCalendar: () => {
    const adapter = useDate();
    const startWeek = adapter.startOfWeek(value.value);
    const endWeek = adapter.endOfWeek(value.value);
    fetchAppointments(startWeek, endWeek);
  }
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
