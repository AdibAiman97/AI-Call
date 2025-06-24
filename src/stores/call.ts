// stores/call.js
import { defineStore } from "pinia";
import axios from "axios";

export const useCallStore = defineStore("call", {
  state: () => ({
    isInCall: false,
    startTime: null as Date | null,
    endTime: null as Date | null,
    callSessionId: null as number | null,
    callId: null as number | null,
  }),

  actions: {
    startCall() {
      this.isInCall = true;
      this.startTime = new Date();
    },

    async endCall(duration: string) {
      this.isInCall = false;
      this.endTime = new Date();

      const callSessionData = {
        cust_id: "Chee Tat",
        start_time: this.startTime?.toISOString() || new Date().toISOString(),
        end_time: this.endTime?.toISOString() || new Date().toISOString(),
        duration: duration,
        positive: 75,
        neutral: 20,
        negative: 5,
        key_words:
          "property, budget, location, bedrooms, mortgage, viewing, investment, amenities",
        summarized_content:
          "Customer inquired about a 3-bedroom property in the downtown location with a budget of $500,000.\nDiscussed mortgage options and financing requirements for the investment property.\nCustomer showed strong interest in scheduling a viewing and exploring nearby amenities.\nReviewed property features including modern kitchen, parking space, and proximity to schools.",
        customer_suggestions:
          "Arrange property viewing for next weekend.\nProvide mortgage pre-approval documentation.\nSend detailed information about local amenities and schools.\nPrepare investment property analysis with rental yield projections.",
        admin_suggestions:
          "Schedule property viewing appointment within 48 hours.\nConnect customer with preferred mortgage broker for financing options.\nPrepare comprehensive property report including market analysis.\nFollow up with similar properties in the same location and budget range.",
      };

      try {
        const response = await axios.post(
          `${import.meta.env.VITE_API_BASE_URL}/call_session`,
          callSessionData
        );
        this.callSessionId = response.data.id;
        // console.log('Call session saved successfully:', response.data)
        return response.data;
      } catch (error) {
        console.error("Error saving call session:", error);
        throw error;
      }
    },
  },
});
