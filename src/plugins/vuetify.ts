/**
 * plugins/vuetify.ts
 *
 * Framework documentation: https://vuetifyjs.com`
 */

// Styles
import '@mdi/font/css/materialdesignicons.css'
import 'vuetify/styles'

// Composables
import { createVuetify } from 'vuetify'

// https://vuetifyjs.com/en/introduction/why-vuetify/#feature-guides
export default createVuetify({
  theme: {
    defaultTheme: 'dark',
    themes: {
      dark: {
          colors: {
            background: '#111827',
            foreground: '#D1D5DB',
            surface: '#1F2937',
            primary: '#2EC4B6',
            secondary: '#2D3748',
            accent: '#5E5E5E',
            info: '#023E7D',
            success: '#064E3B',
            warning: '#FFA726',
            error: '#7F1D1D',
          },
      },
    },
  },
})
