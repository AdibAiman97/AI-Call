from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from database.connection import SessionLocal
from database.models.transcript import Transcript
from sqlalchemy.exc import SQLAlchemyError
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=GOOGLE_API_KEY)

# Template for sentiment analysis with more explicit JSON formatting requirements
sentiment_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in sentiment analysis. Your task is to analyze the sentiment of a given text and classify it as 'Positive', 'Negative', or 'Neutral'.
            Provide a confidence score between 0 and 1 for your classification.
            Also, provide a brief explanation for your sentiment classification.
            
            IMPORTANT: You must return ONLY a valid JSON object with the following structure:
            {{
                "sentiment": "Positive" | "Negative" | "Neutral",
                "confidence": 0.95,
                "explanation": "Brief explanation here"
            }}
            
            Do not include any other text before or after the JSON. Only return the JSON object.
            """,
        ),
        ("user", "Analyze the sentiment of this text: {text}"),
    ]
)


def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response, handling various formatting issues."""
    try:
        # First try direct parsing
        return json.loads(response.strip())
    except json.JSONDecodeError:
        try:
            # Try to find JSON block in the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # If all else fails, create a default response
        return {
            "sentiment": "Neutral",
            "confidence": 0.5,
            "explanation": "Could not parse sentiment analysis response",
        }


@tool
def analyze_sentiment_from_transcript(transcript_id: int) -> str:
    """
    Analyzes the sentiment of a specific transcript.

    Args:
        transcript_id: The ID of the transcript to analyze.

    Returns:
        JSON string with sentiment analysis details or error information.
    """
    db = SessionLocal()
    try:
        transcript = db.query(Transcript).filter(Transcript.id == transcript_id).first()
        if not transcript:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Transcript with ID {transcript_id} not found",
                }
            )

        # Check if transcript message is empty or too short
        if not transcript.message or len(transcript.message.strip()) < 3:
            return json.dumps(
                {
                    "success": True,
                    "transcript_id": transcript_id,
                    "sentiment_analysis": {
                        "sentiment": "Neutral",
                        "confidence": 0.5,
                        "explanation": "Text too short for meaningful sentiment analysis",
                    },
                }
            )

        chain = sentiment_template | llm | StrOutputParser()
        result = chain.invoke({"text": transcript.message})

        # Use the robust JSON extraction function
        sentiment_data = extract_json_from_response(result)

        return json.dumps(
            {
                "success": True,
                "transcript_id": transcript_id,
                "sentiment_analysis": sentiment_data,
            }
        )

    except SQLAlchemyError as e:
        return json.dumps({"success": False, "error": f"Database error: {str(e)}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"})
    finally:
        db.close()


@tool
def get_session_conversation(session_id: int) -> str:
    """
    Retrieves the full conversation for a given session ID.

    Args:
        session_id: The ID of the session.

    Returns:
        JSON string with the full conversation or error information.
    """
    db = SessionLocal()
    try:
        transcripts = (
            db.query(Transcript)
            .filter(Transcript.session_id == session_id)
            .order_by(Transcript.created_at)
            .all()
        )

        if not transcripts:
            return json.dumps(
                {
                    "success": False,
                    "error": f"No transcripts found for session ID {session_id}",
                }
            )

        conversation = [
            {
                "message_by": t.message_by,
                "message": t.message,
                "created_at": t.created_at.isoformat(),
            }
            for t in transcripts
        ]

        return json.dumps(
            {"success": True, "session_id": session_id, "conversation": conversation}
        )

    except SQLAlchemyError as e:
        return json.dumps({"success": False, "error": f"Database error: {str(e)}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"})
    finally:
        db.close()


def analyze_overall_conversation_sentiment(session_id: int) -> str:
    """
    Analyzes the overall sentiment of an entire conversation for a session.
    This is a regular function (not a LangChain tool) for direct use.
    
    Args:
        session_id: The ID of the session to analyze.
        
    Returns:
        JSON string with overall conversation sentiment analysis and breakdown.
    """
    db = SessionLocal()
    try:
        transcripts = (
            db.query(Transcript)
            .filter(Transcript.session_id == session_id)
            .order_by(Transcript.created_at)
            .all()
        )
        
        if not transcripts:
            return json.dumps({
                "success": False,
                "error": f"No transcripts found for session ID {session_id}"
            })
        
        # Combine all messages into a conversation flow
        conversation_text = "\n".join([
            f"{t.message_by}: {t.message}" for t in transcripts if t.message and t.message.strip()
        ])
        
        if not conversation_text.strip():
            return json.dumps({
                "success": True,
                "session_id": session_id,
                "overall_sentiment": {
                    "sentiment": "Neutral",
                    "confidence": 0.5,
                    "explanation": "No meaningful conversation content to analyze"
                },
                "sentiment_breakdown": {"positive": 0, "neutral": 1, "negative": 0}
            })
        
        # Analyze overall conversation sentiment
        overall_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in sentiment analysis for real estate customer conversations. 
                Analyze the OVERALL sentiment of the entire conversation and provide balanced analysis.
                
                IMPORTANT: Ensure percentages are realistic and balanced:
                - Most conversations have mixed sentiment (not 100% in one category)
                - Typical distributions: 60-70% dominant, 20-30% neutral, 10-20% other
                - Even positive conversations may have 10-20% neutral/concerned moments
                - Professional inquiries often have 40-60% neutral content
                
                Return ONLY a valid JSON object:
                {{
                    "sentiment": "Positive" | "Negative" | "Neutral",
                    "confidence": 0.85,
                    "explanation": "Brief explanation of overall conversation tone",
                    "sentiment_breakdown": {{
                        "positive_percentage": 60,
                        "neutral_percentage": 30, 
                        "negative_percentage": 10
                    }},
                    "key_drivers": ["Key factors that influenced the sentiment"]
                }}
                
                Provide realistic, balanced sentiment percentages that reflect real conversation dynamics.
                """
            ),
            ("user", "Analyze the overall sentiment of this customer service conversation:\n\n{conversation}")
        ])
        
        chain = overall_prompt | llm | StrOutputParser()
        result = chain.invoke({"conversation": conversation_text})
        
        # Parse the response
        analysis_data = extract_json_from_response(result)
        
        # Normalize percentages to ensure they total 100%
        breakdown = analysis_data.get("sentiment_breakdown", {"positive_percentage": 34, "neutral_percentage": 33, "negative_percentage": 33})
        
        positive_pct = max(0, breakdown.get("positive_percentage", 34))
        neutral_pct = max(0, breakdown.get("neutral_percentage", 33))
        negative_pct = max(0, breakdown.get("negative_percentage", 33))
        
        # Normalize to 100%
        total_pct = positive_pct + neutral_pct + negative_pct
        if total_pct > 0:
            positive_pct = round(positive_pct * 100 / total_pct)
            neutral_pct = round(neutral_pct * 100 / total_pct)
            negative_pct = 100 - positive_pct - neutral_pct  # Ensure exact 100% total
        else:
            positive_pct, neutral_pct, negative_pct = 34, 33, 33
        
        # Convert to counts for database storage (maintaining percentage logic)
        positive_count = positive_pct
        neutral_count = neutral_pct
        negative_count = negative_pct
        
        # Count total meaningful messages for reference
        total_messages = len([t for t in transcripts if t.message and t.message.strip()])
        
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "total_messages": total_messages,
            "overall_sentiment": {
                "sentiment": analysis_data.get("sentiment", "Neutral"),
                "confidence": analysis_data.get("confidence", 0.5),
                "explanation": analysis_data.get("explanation", "Overall conversation analysis")
            },
            "sentiment_breakdown": {
                "positive": positive_count,
                "neutral": neutral_count, 
                "negative": negative_count
            },
            "key_drivers": analysis_data.get("key_drivers", ["General conversation tone"])
        })
        
    except SQLAlchemyError as e:
        return json.dumps({"success": False, "error": f"Database error: {str(e)}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"})
    finally:
        db.close()


@tool
def analyze_conversation_sentiment(session_id: int) -> str:
    """
    Analyzes the overall sentiment of an entire conversation for a session.
    
    Args:
        session_id: The ID of the session to analyze.
        
    Returns:
        JSON string with overall conversation sentiment analysis and breakdown.
    """
    db = SessionLocal()
    try:
        transcripts = (
            db.query(Transcript)
            .filter(Transcript.session_id == session_id)
            .order_by(Transcript.created_at)
            .all()
        )
        
        if not transcripts:
            return json.dumps({
                "success": False,
                "error": f"No transcripts found for session ID {session_id}"
            })
        
        # Combine all messages into a conversation flow
        conversation_text = "\n".join([
            f"{t.message_by}: {t.message}" for t in transcripts if t.message and t.message.strip()
        ])
        
        if not conversation_text.strip():
            return json.dumps({
                "success": True,
                "session_id": session_id,
                "overall_sentiment": {
                    "sentiment": "Neutral",
                    "confidence": 0.5,
                    "explanation": "No meaningful conversation content to analyze"
                },
                "sentiment_breakdown": {"positive": 0, "neutral": 1, "negative": 0}
            })
        
        # Analyze overall conversation sentiment
        overall_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in sentiment analysis for real estate customer conversations. 
                Analyze the OVERALL sentiment of the entire conversation and provide balanced analysis.
                
                IMPORTANT: Ensure percentages are realistic and balanced:
                - Most conversations have mixed sentiment (not 100% in one category)
                - Typical distributions: 60-70% dominant, 20-30% neutral, 10-20% other
                - Even positive conversations may have 10-20% neutral/concerned moments
                - Professional inquiries often have 40-60% neutral content
                
                Return ONLY a valid JSON object:
                {{
                    "sentiment": "Positive" | "Negative" | "Neutral",
                    "confidence": 0.85,
                    "explanation": "Brief explanation of overall conversation tone",
                    "sentiment_breakdown": {{
                        "positive_percentage": 60,
                        "neutral_percentage": 30, 
                        "negative_percentage": 10
                    }},
                    "key_drivers": ["Key factors that influenced the sentiment"]
                }}
                
                Provide realistic, balanced sentiment percentages that reflect real conversation dynamics.
                """
            ),
            ("user", "Analyze the overall sentiment of this customer service conversation:\n\n{conversation}")
        ])
        
        chain = overall_prompt | llm | StrOutputParser()
        result = chain.invoke({"conversation": conversation_text})
        
        # Parse the response
        analysis_data = extract_json_from_response(result)
        
        # Normalize percentages to ensure they total 100%
        breakdown = analysis_data.get("sentiment_breakdown", {"positive_percentage": 34, "neutral_percentage": 33, "negative_percentage": 33})
        
        positive_pct = max(0, breakdown.get("positive_percentage", 34))
        neutral_pct = max(0, breakdown.get("neutral_percentage", 33))
        negative_pct = max(0, breakdown.get("negative_percentage", 33))
        
        # Normalize to 100%
        total_pct = positive_pct + neutral_pct + negative_pct
        if total_pct > 0:
            positive_pct = round(positive_pct * 100 / total_pct)
            neutral_pct = round(neutral_pct * 100 / total_pct)
            negative_pct = 100 - positive_pct - neutral_pct  # Ensure exact 100% total
        else:
            positive_pct, neutral_pct, negative_pct = 34, 33, 33
        
        # Convert to counts for database storage (maintaining percentage logic)
        positive_count = positive_pct
        neutral_count = neutral_pct
        negative_count = negative_pct
        
        # Count total meaningful messages for reference
        total_messages = len([t for t in transcripts if t.message and t.message.strip()])
        
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "total_messages": total_messages,
            "overall_sentiment": {
                "sentiment": analysis_data.get("sentiment", "Neutral"),
                "confidence": analysis_data.get("confidence", 0.5),
                "explanation": analysis_data.get("explanation", "Overall conversation analysis")
            },
            "sentiment_breakdown": {
                "positive": positive_count,
                "neutral": neutral_count, 
                "negative": negative_count
            },
            "key_drivers": analysis_data.get("key_drivers", ["General conversation tone"])
        })
        
    except SQLAlchemyError as e:
        return json.dumps({"success": False, "error": f"Database error: {str(e)}"})
    except Exception as e:
        return json.dumps({"success": False, "error": f"Unexpected error: {str(e)}"})
    finally:
        db.close()


# Bind tools to LLM
tools = [
    analyze_sentiment_from_transcript,
    get_session_conversation,
    analyze_conversation_sentiment,
]
llm_with_tools = llm.bind_tools(tools)

# Chain for processing sentiment analysis requests
sentiment_chain = sentiment_template | llm_with_tools | StrOutputParser()
