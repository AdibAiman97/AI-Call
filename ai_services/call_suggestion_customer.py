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
        suggestion_prompt = f"""Based on the following real estate call transcript, provide 3-4 SHORT and PRECISE suggestions for what the customer should do next after this call ends.

Requirements:
- Focus on immediate next steps the customer can take
- Keep each suggestion to 1 line (max 15 words)
- Be specific and actionable
- Related to property viewing, documentation, decision-making, or follow-up
- NO bullet symbols (•) - output one suggestion per line
- Frontend will format as unordered list

Examples of good suggestions:
Schedule a property viewing for this weekend
Review the property brochure and floor plans sent
Prepare mortgage pre-approval documents
Visit the show gallery during weekday hours

Call Transcript:
{formatted_conversation}

Customer Next Steps (one per line, no bullets):"""
        response = model.invoke(suggestion_prompt)
        suggestions = response.content
        return suggestions
    except Exception as e:
        print(f"Error generating suggestions for session {call_session_id}: {str(e)}")
        return f"Error generating suggestions: {str(e)}"