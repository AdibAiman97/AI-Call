import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from config import GCP_PROJECT_ID, GCP_LOCATION

llm_model = None


def initialize_llm_model():
    """Initializes the Vertex AI GenerativeModel (Gemini)."""
    global llm_model
    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)

        model_name = "gemini-2.0-flash-001"
        llm_model = GenerativeModel(model_name)
        print(f"Vertex AI and Gemini model '{model_name}' initialized successfully.")
    except Exception as e:
        print(f"ERROR: Could not initialize Vertex AI GenerativeModel: {e}")
        print(
            "Please ensure 'gcloud auth application-default login' is run and your GCP_PROJECT_ID/GCP_LOCATION are correct in config.py."
        )
        llm_model = None


async def call_gemini_api(conversation_content: list) -> str:
    """
    Calls the Gemini model on Vertex AI to generate a response with conversation history.
    """
    if llm_model is None:
        return "I'm sorry, the AI model is not initialized. Please check backend logs for initialization errors."

    formatted_conversation = []
    for item in conversation_content:
        role = item["role"]
        text_part = Part.from_text(item["parts"][0]["text"])
        formatted_conversation.append(
            vertexai.generative_models.Content(role=role, parts=[text_part])
        )

    try:
        print(f"Calling Vertex AI Gemini model '{llm_model._model_name}'...")
        print(
            f"Conversation content (truncated): {json.dumps(conversation_content, indent=2)[:500]}..."
        )

        response = await llm_model.generate_content_async(
            formatted_conversation,
            generation_config={"temperature": 0.5, "max_output_tokens": 500},
        )

        if (
            response.candidates
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            return response.candidates[0].content.parts[0].text
        else:
            print(
                f"Error: Unexpected Vertex AI response structure. Response: {response}"
            )
            return "I'm sorry, I couldn't generate a response from the AI."

    except Exception as e:
        print(f"Error calling Vertex AI Gemini API: {e}")
        return f"I'm sorry, an error occurred while processing your request with Vertex AI: {e}"
