# AI Call

A comprehensive, web-based call management system designed for businesses. It leverages AI to provide real-time call transcription, sentiment analysis, and agent assistance.

# 1) Key Features

* **Real-Time Transcription:** Get a live transcription of calls as they happen.
* **Sentiment Analysis:** Gauge customer sentiment in real-time to better understand their needs.
* **AI-Powered Agent Assistance:** Provide agents with real-time suggestions and information to help them resolve customer issues more effectively.
* **Call Recording and Playback:** Record calls for quality assurance and training purposes.
* **Call Summary:** Generate a summary of the call after it has ended.
* **Multiple User Roles:** The system is built with a clear separation of roles and permissions:
    * **Admin:** Manages the entire system, including creating and managing agents and viewing call analytics.
    * **Agent:** Handles calls and has access to real-time transcription, sentiment analysis, and AI-powered assistance.

# 2) Technology Stack

* **Backend:** Python, FastAPI
* **Frontend:** Vue.js, Vuetify, Vite
* **Database:** MongoDB
* **Key Libraries:**
    * **Google Generative AI:** For real-time transcription, sentiment analysis, and agent assistance.
    * **LangChain:** For building and managing the RAG (Retrieval-Augmented Generation) system.
    * **WebSockets:** For real-time communication between the frontend and backend.

# 3) Project Structure

The application is organized into two main parts:

* **/voxis-ai:** The frontend of the application, built with Vue.js and Vuetify.
* **/voxis-ai-be:** The backend of the application, built with Python and FastAPI.

# 4) Project Preview

(You can add screenshots or a GIF of your project here)