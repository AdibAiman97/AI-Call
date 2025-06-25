<template>
    <div>
      <h1 class="d-flex justify-space-between text-foreground">
        Appointment
        <div class="d-flex ga-2">
          <v-btn class="text-capitalize text-foreground">
            <ListFilter :size="20" />
            Filter
          </v-btn>
          <v-btn class="bg-primary text-capitalize">
            <Plus :size="20" />
            Add Appointment
          </v-btn>
        </div>
      </h1>
      <p class="text-foreground text-body-2">
        Manage your upcoming customer meetings
      </p>
      
      <v-progress-circular v-if="loading" indeterminate color="primary" class="ma-4" />
      
      <v-calendar 
        v-else
        :interval-minutes="30" 
        :interval-height="48" 
        ref="calendar" 
        v-model="value" 
        :events="events"
        :view-mode="type" 
        :weekdays="days">
      </v-calendar>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue'
  import { useDate } from 'vuetify'
  import { Plus, ListFilter } from 'lucide-vue-next'
  
  // Values
  const type = ref('week')
  const days = ref([0, 1, 2, 3, 4, 5, 6])
  const value = ref(new Date())
  const events = ref([])
  const loading = ref(true)
  
  // Options
  const types = ['week', 'day']
  const weekdays = [
    { title: 'Mon-Sun', value: [1, 2, 3, 4, 5, 6, 0] },
    { title: 'Mon-Fri', value: [1, 2, 3, 4, 5] }
  ]
  
  const colors = ['primary', 'secondary', 'info']
  
  // Fetch appointments from backend database
  async function fetchAppointments(startWeek, endWeek) {
    try {
      loading.value = true
      const startParam = startWeek.toISOString()
      const endParam = endWeek.toISOString()
      
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/appointments/date-range?start_date=${startParam}&end_date=${endParam}`
      )
  
      if (!response.ok) {
        throw new Error('Failed to fetch appointments')
      }
  
      const data = await response.json()
      
      // Transform backend data to calendar format
      events.value = data.map(appointment => ({
        title: appointment.title,
        start: new Date(appointment.start_time),
        end: new Date(appointment.end_time),
        color: colors[Math.floor(Math.random() * colors.length)], // Random color assignment
        id: appointment.id,
        callSessionId: appointment.call_session_id
      }))
  
      console.clear()
      console.log('Fetched appointments from database:', events.value)
      
      // Log appointment details
      events.value.forEach((appointment) => {
        console.log('start', appointment.start.toLocaleString([], {
          weekday: 'short',
          hour: '2-digit',
          minute: '2-digit'
        }))
        console.log('end', appointment.end.toLocaleString([], {
          weekday: 'short',
          hour: '2-digit',
          minute: '2-digit'
        }))
      })
  
    } catch (error) {
      console.error('Error fetching appointments:', error)
      events.value = []
      // You could show a user-friendly error message here
    } finally {
      loading.value = false
    }
  }
  
  // Alternative function to fetch all appointments (if you want to show all instead of date range)
  async function fetchAllAppointments() {
    try {
      loading.value = true
      const response = await fetch(
        `${import.meta.env.VITE_API_BASE_URL}/appointments/`
      )
  
      if (!response.ok) {
        throw new Error('Failed to fetch appointments')
      }
  
      const data = await response.json()
      
      events.value = data.map(appointment => ({
        title: appointment.title,
        start: new Date(appointment.start_time),
        end: new Date(appointment.end_time),
        color: colors[Math.floor(Math.random() * colors.length)],
        id: appointment.id,
        callSessionId: appointment.call_session_id
      }))
  
      console.log('Fetched all appointments:', events.value)
  
    } catch (error) {
      console.error('Error fetching appointments:', error)
      events.value = []
    } finally {
      loading.value = false
    }
  }
  
  onMounted(() => {
    const adapter = useDate()
    const today = new Date()
    const startWeek = adapter.startOfWeek(today)
    const endWeek = adapter.endOfWeek(today)
  
    // Fetch appointments for current week
    fetchAppointments(startWeek, endWeek)
    
    // Alternative: Uncomment below to fetch all appointments instead
    // fetchAllAppointments()
  })
  </script>
  
  <style scoped>
  :deep(.v-calendar-internal-event) {
    margin-top: 0px !important;
    height: 48px !important;
  }
  </style>