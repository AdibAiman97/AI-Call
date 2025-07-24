import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration (Legacy - kept for backward compatibility)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # No longer needed with Vertex AI auth
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key="

# MongoDB Configuration
# Create a .env file with: MONGO_DB="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
MONGO_DB_CONNECTION_STRING = os.getenv("MONGO_DB", "")  # MongoDB Atlas connection string
DB_NAME = "test_db"
COLLECTION_NAME = "voxis_ai"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "voxis_ai_vector_index"

# GCP Configuration
# Set GOOGLE_APPLICATION_CREDENTIALS environment variable to path of your service account JSON file
GCP_PROJECT_ID = "voxis-ai"
GCP_LOCATION = "us-central1"

# MongoDB Atlas Configuration
# MONGODB_ATLAS_CLUSTER_URI = ""  # MongoDB Atlas connection string
# MONGODB_DB_NAME = "test_db"
# MONGODB_COLLECTION_NAME = "test_collection"
# MONGODB_VECTOR_INDEX_NAME = "test-index-1"
