import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.orm import Session
from database.models.call_session import CallSession
from database.models.transcript import Transcript
from services.transcript_crud import (
    create_session_message,
    update_session_summary,
    update_session_key_topics,
    get_message_by_id,
    get_session_messages,
)
from database.connection import get_db, SessionLocal
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
        messages = get_session_messages(db, session_id, skip, limit)
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


def analyze_call_and_generate_suggestions(session_id: int) -> dict:
    """Analyzes a call session and generates comprehensive suggestions for admin follow-up."""
    db = SessionLocal()
    try:
        # Get call session details
        call_session = (
            db.query(CallSession).filter(CallSession.id == session_id).first()
        )
        if not call_session:
            return {"error": f"Call session {session_id} not found"}

        # Get all transcripts for the session
        transcripts = (
            db.query(Transcript)
            .filter(Transcript.session_id == session_id)
            .order_by(Transcript.created_at)
            .all()
        )

        if not transcripts:
            return {"error": f"No transcripts found for session {session_id}"}

        # Format conversation for analysis
        conversation = []
        for transcript in transcripts:
            conversation.append(
                {
                    "speaker": transcript.message_by,
                    "message": transcript.message,
                    "timestamp": transcript.created_at.isoformat(),
                }
            )

        # Create comprehensive conversation text
        conversation_text = "\n".join(
            [f"{msg['speaker']}: {msg['message']}" for msg in conversation]
        )

        # Generate analysis using Gemini
        model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

        analysis_prompt = f"""
        Analyze the following real estate sales call transcript and provide comprehensive insights focused on CUSTOMER ACQUISITION and RELATIONSHIP WARMING:

        CALL TRANSCRIPT:
        {conversation_text}

        Please provide a detailed analysis in the following JSON format:
        {{
            "call_summary": "Brief summary of the call",
            "key_topics": ["topic1", "topic2", "topic3"],
            "customer_sentiment": "Positive/Negative/Neutral",
            "customer_needs": ["need1", "need2"],
            "admin_suggestions": [
                "Short, specific action to warm the customer and move toward sale",
                "Follow-up action to maintain engagement", 
                "Next step to convert interest into commitment"
            ],
            "follow_up_required": true/false,
            "priority_level": "High/Medium/Low",
            "next_steps": "Recommended next steps for the admin"
        }}

        ADMIN SUGGESTIONS REQUIREMENTS:
        - Focus on SALES ACQUISITION and customer relationship warming
        - Keep each suggestion to 1 line (max 15 words)
        - Be specific and immediately actionable
        - NO bullet symbols (•) - output one suggestion per line
        - Prioritize follow-up timing, personalized outreach, and closing opportunities
        - Examples: "Send property brochure within 2 hours", "Schedule callback in 3 days", "Invite to weekend show gallery"

        Focus on actionable insights that will help convert this lead into a sale.
        """

        try:
            response = model.invoke(analysis_prompt)
            analysis_text = response.content

            # Extract JSON from response
            try:
                # Try to find JSON in the response
                import re

                json_match = re.search(r"\{.*\}", analysis_text, re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    # Fallback if no JSON found
                    analysis_data = {
                        "call_summary": "Analysis generated successfully",
                        "key_topics": ["Customer inquiry"],
                        "customer_sentiment": "Neutral",
                        "customer_needs": ["Information request"],
                        "admin_suggestions": [
                            "Follow up with customer",
                            "Review call quality",
                        ],
                        "follow_up_required": True,
                        "priority_level": "Medium",
                        "next_steps": "Review analysis and plan follow-up",
                    }
            except json.JSONDecodeError:
                # Create default analysis if JSON parsing fails
                analysis_data = {
                    "call_summary": f"Call analysis for session {session_id}",
                    "key_topics": ["General inquiry"],
                    "customer_sentiment": "Neutral",
                    "customer_needs": ["Support request"],
                    "admin_suggestions": [
                        "Review call transcript",
                        "Follow up if needed",
                    ],
                    "follow_up_required": True,
                    "priority_level": "Medium",
                    "next_steps": "Manual review recommended",
                }

            # Update database with analysis results
            try:
                # Update call session summary
                update_session_summary(
                    db, session_id, analysis_data.get("call_summary", "")
                )

                # Update key topics
                key_topics_str = ", ".join(analysis_data.get("key_topics", []))
                update_session_key_topics(db, session_id, key_topics_str)

                # Create admin suggestion message
                suggestions_text = "\n".join(
                    [
                        f"• {suggestion}"
                        for suggestion in analysis_data.get("admin_suggestions", [])
                    ]
                )
                full_suggestion = f"""
ADMIN ANALYSIS & SUGGESTIONS:

Summary: {analysis_data.get('call_summary', '')}

Key Topics: {key_topics_str}

Customer Sentiment: {analysis_data.get('customer_sentiment', 'Unknown')}

Admin Action Items:
{suggestions_text}

Priority: {analysis_data.get('priority_level', 'Medium')}
Follow-up Required: {analysis_data.get('follow_up_required', 'Unknown')}

Next Steps: {analysis_data.get('next_steps', '')}
"""

                create_session_message(
                    db, session_id, full_suggestion, "Admin_Analysis"
                )

                return {
                    "success": True,
                    "session_id": session_id,
                    "analysis": analysis_data,
                }

            except Exception as db_error:
                print(f"Database update error: {db_error}")
                return {
                    "success": False,
                    "error": f"Failed to update database: {str(db_error)}",
                    "analysis": analysis_data,
                }

        except Exception as ai_error:
            print(f"AI analysis error: {ai_error}")
            return {"success": False, "error": f"AI analysis failed: {str(ai_error)}"}

    except Exception as e:
        print(f"General error in analysis: {e}")
        return {"success": False, "error": f"Analysis failed: {str(e)}"}
    finally:
        db.close()


def get_suggestion_from_agent(session_id: int, query: str, message_by: str):
    """Orchestrates creating a message, getting a suggestion, and saving it."""
    db = next(db_generator)

    # 1. Create and save the original user message
    create_session_message(db, session_id, query, message_by)

    # 2. Get a suggestion from the AI agent if the message is not from the agent
    if message_by.lower() != "agent":
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant for a customer service call center. 
            Provide practical, actionable suggestions based on the user's query. 
            Focus on customer service best practices and specific actions that can improve customer satisfaction.""",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
        tools = [
            create_session_message_tool,
            get_session_messages_tool,
            update_session_summary_tool,
            update_session_key_topics_tool,
            get_message_by_id_tool,
        ]

        agent = create_tool_calling_agent(model, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, max_iterations=10
        )

        response = agent_executor.invoke(
            {
                "input": f"Based on this customer service context: {query}, provide specific suggestions for improving the customer experience and next steps for the admin team."
            }
        )

        suggestion = response["output"]

        # 3. Create and save the AI's response as a new message
        create_session_message(db, session_id, suggestion, "Agent")

        return suggestion

    return None
