from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

# Shared model instance
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

def summarize_text(conversation: List[Dict[str, str]], call_session_id: int) -> str:
    """Summarize the given conversation.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        call_session_id: ID of the call session (used for logging)
    """
    try:
        # Format conversation for summarization
        formatted_conversation = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation
        ])
        summary_prompt = f"Please provide a concise summary of the following conversation:\n\n{formatted_conversation}"
        response = model.invoke(summary_prompt)
        summary = response.content
        return summary
    except Exception as e:
        print(f"Error summarizing text for session {call_session_id}: {str(e)}")
        return f"Error summarizing text: {str(e)}"