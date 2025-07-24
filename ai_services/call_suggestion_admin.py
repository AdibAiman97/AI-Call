import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session
from api.models import CallSession
from api.crud import create_session_message, update_session_summary, update_session_key_topics, get_message_by_id
from api.models import get_db

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

db_generator = get_db()

@tool
def create_session_message_tool(session_id: int, message: str, message_by: str) -> str:
    """Creates a new message within a call session."""
    try:
        db = next(db_generator)
        create_session_message(db, session_id, message, message_by)
        return "Message created successfully."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_session_messages_tool(session_id: int, skip: int = 0, limit: int = 100) -> str:
    """Retrieves messages for a specific session."""
    try:
        db = next(db_generator)
        messages = db.query(CallSession).filter(CallSession.session_id == session_id).offset(skip).limit(limit).all()
        return str(messages)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def update_session_summary_tool(session_id: int, summarized: str) -> str:
    """Updates the summarized field for all messages in a session."""
    try:
        db = next(db_generator)
        update_session_summary(db, session_id, summarized)
        return "Summary updated successfully."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def update_session_key_topics_tool(session_id: int, key_topics: str) -> str:
    """Updates the key_topics field for all messages in a session."""
    try:
        db = next(db_generator)
        update_session_key_topics(db, session_id, key_topics)
        return "Key topics updated successfully."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_message_by_id_tool(message_id: int) -> str:
    """Retrieves a message by its ID."""
    try:
        db = next(db_generator)
        message = get_message_by_id(db, message_id)
        return str(message)
    except Exception as e:
        return f"Error: {str(e)}"

def get_suggestion_from_agent(session_id: int, query: str, message_by: str):
    """Orchestrates creating a message, getting a suggestion, and saving it."""
    db = next(db_generator)
    
    # 1. Create and save the original user message
    create_session_message(db, session_id, query, message_by)
    
    # 2. Get a suggestion from the AI agent if the message is not from the agent
    if message_by.lower() != "agent":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GEMINI_API_KEY)
        tools = [
            create_session_message_tool,
            get_session_messages_tool,
            update_session_summary_tool,
            update_session_key_topics_tool,
            get_message_by_id_tool,
        ]

        agent = create_tool_calling_agent(model, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10
        )

        response = agent_executor.invoke({
            "input": query
        })
        
        suggestion = response["output"]
        
        # 3. Create and save the AI's response as a new message
        create_session_message(db, session_id, suggestion, "Agent")
        
        return suggestion
    
    return None