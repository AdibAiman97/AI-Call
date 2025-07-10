from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import requests
from typing import List, Dict

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

# Shared model instance
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GEMINI_API_KEY)

def summarize_text(conversation: List[Dict[str, str]], call_session_id: int) -> str:
    """Summarize the given conversation and optionally store it in the database.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        call_session_id: ID of the call session to update in database
    """
    try:
        # Format conversation for summarization
        formatted_conversation = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in conversation
        ])
        summary_prompt = f"Please provide a concise summary of the following conversation:\n\n{formatted_conversation}"
        response = model.invoke(summary_prompt)
        summary = response.content
        
        # Store summary in database if call_session_id is provided
        if call_session_id:
            try:
                update_data = {"summarized_content": summary}
                api_response = requests.put(
                    f"http://localhost:8000/call_session/{call_session_id}",
                    json=update_data
                )
                if api_response.status_code == 200:
                    print(f"Summary stored successfully for call session {call_session_id}")
                else:
                    print(f"Failed to store summary: {api_response.status_code}")
            except Exception as db_error:
                print(f"Database update error: {str(db_error)}")
        
        return summary
    except Exception as e:
        return f"Error summarizing text: {str(e)}"