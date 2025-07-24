from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict, Optional

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

def generate_caller_suggestions(conversation: List[Dict[str, str]], call_session_id: Optional[int] = None) -> str:
    """Generate suggestions for what the caller can do next after the call ends.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
    """
    try:
        formatted_conversation = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation
        ])
        suggestion_prompt = f"Based on the following call transcript, provide helpful suggestions for what the caller (customer) can do next after this call ends:\n\n{formatted_conversation}"
        response = model.invoke(suggestion_prompt)
        suggestions = response.content

        if call_session_id:
            try:
                update_data = {"customer_suggestions": suggestions}
                api_response = requests.put(
                    f"http://localhost:8000/call_session/{call_session_id}",
                    json=update_data
                )
                if api_response.status_code == 200:
                    print(f"Customer suggestions stored successfully for call session {call_session_id}")
                else:
                    print(f"Failed to store customer suggestions: {api_response.status_code}")
            except Exception as db_error:
                print(f"Database update error: {str(db_error)}")
        
        return suggestions
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"