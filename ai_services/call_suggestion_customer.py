from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict, Optional

load_dotenv()
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

def generate_caller_suggestions(conversation: List[Dict[str, str]], call_session_id: Optional[int] = None) -> str:
    """Generate suggestions for what the caller can do next after the call ends.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        call_session_id: ID of the call session (used for logging)
    """
    try:
        formatted_conversation = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation
        ])
        suggestion_prompt = f"Based on the following call transcript, provide helpful suggestions for what the caller (customer) can do next after this call ends:\n\n{formatted_conversation}"
        response = model.invoke(suggestion_prompt)
        suggestions = response.content
        return suggestions
    except Exception as e:
        print(f"Error generating suggestions for session {call_session_id}: {str(e)}")
        return f"Error generating suggestions: {str(e)}"