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
        positive: 65,
        neutral: 25,
        negative: 10,
        key_words:
          "pricing, enterprise, demo, follow-up, implementation, onboarding",
        summarized_content:
          "Inquiry about product features and pricing.\nDiscussed available options and provided detailed information.\nCustomer showed interest in premium package.",
        customer_suggestions:
          "Follow up with detailed pricing information.\nSchedule demo session for next week.\nSend product comparison chart.",
        admin_suggestions:
          "Review customer's specific use case requirements.\nPrepare customized proposal.\nSchedule follow-up call within 3 business days.",
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
