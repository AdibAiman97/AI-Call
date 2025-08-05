# Voxis AI

A web-based application designed for AI-powered call management and communication. It provides real-time transcription, AI assistance during calls, and tools for managing appointments and customer interactions.

# 1) Key Features

*   **Real-time Call Management:** Facilitates live call handling, including audio/video streaming and call state management.
*   **AI-Powered Transcription:** Provides real-time speech-to-text transcription of calls.
*   **Gemini AI Integration:** Leverages Google's Gemini AI for advanced functionalities, likely including natural language understanding, response generation, and RAG (Retrieval Augmented Generation) capabilities.
*   **Appointment Management:** Tools for scheduling, viewing, and managing appointments.
*   **Chat Functionality:** Integrated chat features for communication.
*   **Interactive User Interface:** A modern and responsive user interface built with Vue.js and Vuetify.
*   **Backend API:** A robust backend API for handling data, AI services, and real-time communication via WebSockets.

# 2) Technology Stack

*   **Frontend:** Vue.js, Vuetify, Vite, TypeScript, Pinia, Axios, Three.js, Lucide Vue Next, @mdi/font.
*   **Backend:** Python, FastAPI, Uvicorn, WebSockets, Google Generative AI, NumPy, PyAudio, Langchain, Langchain-MongoDB, PyMongo, python-dotenv.
*   **Database:** MongoDB (inferred from PyMongo and Langchain-MongoDB usage).

# 3) Project Structure

The project is divided into two main parts: the frontend (`voxis-ai`) and the backend (`voxis-ai-be`).

*   `voxis-ai/`: Frontend application
    *   `src/components/`: Reusable Vue components.
    *   `src/pages/`: Vue views/pages for different routes.
    *   `src/stores/`: Pinia stores for state management.
    *   `src/router/`: Vue Router configuration for navigation.
    *   `src/utils/`: Utility functions and helpers.
    *   `public/`: Static assets and client-side processing scripts (e.g., `processor.js`).
*   `voxis-ai-be/`: Backend application
    *   `api/`: FastAPI routers defining API endpoints.
    *   `ai_services/`: Modules containing AI-related services and tools.
    *   `database/`: Database connection, schemas, and models.
    *   `services/`: Business logic and CRUD operations.
    *   `VertexRagSystem/`: Components related to the Retrieval Augmented Generation (RAG) system.
    *   `rag_prev_version/`: Contains previous iterations or versions of the RAG implementation.

# 4) Project Preview

(No preview available yet.)