<template>
  <div>

    <v-calendar :interval-minutes="30" :interval-height="48" ref="calendar" v-model="value" :events="events"
      :view-mode="type" :weekdays="days">
    </v-calendar>

  </div>
</template>

<script setup>
import { useDate } from 'vuetify'

// Values
const type = ref('week');
const days = ref([0, 1, 2, 3, 4, 5, 6]);
const value = ref(new Date())
const events = ref([
  // {
  //   title: 'Time',
  //   start: new Date(1747969200000),
  //   end: new Date(1747969200000 + 3600000),
  //   color: 'primary'
  // }
])

console.log(events.value.start)

// Options
const types = ['week', 'day'];
const weekdays = [
  { title: 'Mon-Sun', value: [1, 2, 3, 4, 5, 6, 0] },
  { title: 'Mon-Fri', value: [1, 2, 3, 4, 5] }
];

const colors = ['primary', 'secondary', 'info']
const titles = ['Meeting', 'Holiday', 'Workshop', 'Appointment']

function rand(a, b) {
  return Math.floor((b - a + 1) * Math.random()) + a
}

// Random Duration
function randomDuration() {
  const duration = ['60', '120', '480']; // in minutes
  const ms = 60 * 1000; // min to ms
  const randDur = rand(0, duration.length - 1);
  const eventDuration = parseInt(duration[randDur]) * ms;

  return eventDuration
}

// Random Day 9am - 7pm 
function randomDay(min, max) {
  const randDay = rand(min, max)
  const startDay = new Date(randDay)
  startDay.setHours(9, 0, 0, 0)
  const endDay = new Date(randDay)
  endDay.setHours(19, 0, 0, 0)

  return { startDay, endDay }
}

function randomEvent(startDay, endDay, eventDuration) {
  const randStart = rand(startDay.getTime(), endDay.getTime() - eventDuration)
  const start = new Date(randStart - (randStart % 3600000))
  const end = new Date(start.getTime() + eventDuration)

  return { start, end }
}

function checkCalendar(result, max, min, eventDuration) {
  const { startDay, endDay } = randomDay(min, max)
  const { start, end } = randomEvent(startDay, endDay, eventDuration)

  const hasOverlap = result.some(event => start < event.end && end > event.start)

  if (!hasOverlap) {
    return { start, end, status: false }
  } else {
    return { status: true }
  }
}

function getEvents({ startWeek, endWeek }) {
  const result = []
  // SoW & EoW
  const min = startWeek.getTime()
  const max = endWeek.getTime()
  const eventCount = 20
  for (let i = 0; i < eventCount; i++) {

    const eventDuration = randomDuration()
    // const { startDay, endDay } = randomDay(min, max)
    // const { start, end } = randomEvent(startDay, endDay, eventDuration)
    const isBooked = checkCalendar(result, max, min, eventDuration)

    if (isBooked.status === false) {

      const { start, end } = isBooked
      result.push({
        title: titles[rand(0, titles.length - 1)],
        start,
        end,
        color: colors[rand(0, colors.length - 1)],
      })
    }
  }

  events.value = result
  console.clear()
  console.log(result)
  result.map((each) => {
    console.log('start', each.start.toLocaleString([], {
      weekday: 'short',  // e.g., Mon, Tue
      hour: '2-digit',
      minute: '2-digit'
    }))

    // console.log('start',each.start.getTime())

    console.log('end', each.end.toLocaleString([], {
      weekday: 'short',
      hour: '2-digit',
      minute: '2-digit'
    }))
  })
}

onMounted(() => {
  const adapter = useDate()
  const today = new Date();
  const startWeek = adapter.startOfWeek(today)
  const endWeek = adapter.endOfWeek(today)

  getEvents({ startWeek, endWeek })
})
</script>

<style scoped>
:deep(.v-calendar-internal-event) {
  margin-top: 0px !important;
  height: 48px !important;
}
</style>