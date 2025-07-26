from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from database.connection import SessionLocal
from database.models.transcript import Transcript
from sqlalchemy.exc import SQLAlchemyError
import json

# LLM Configuration
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Template for sentiment analysis
sentiment_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in sentiment analysis. Your task is to analyze the sentiment of a given text and classify it as 'Positive', 'Negative', or 'Neutral'.
            Provide a confidence score between 0 and 1 for your classification.
            Also, provide a brief explanation for your sentiment classification.
            Return the result in JSON format with the following keys: 'sentiment', 'confidence', and 'explanation'.
            """,
        ),
        ("user", "Analyze the sentiment of this text: {text}"),
    ]
)


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

        chain = sentiment_template | llm | StrOutputParser()
        result = chain.invoke({"text": transcript.message})

        return json.dumps(
            {
                "success": True,
                "transcript_id": transcript_id,
                "sentiment_analysis": json.loads(result),
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


# Bind tools to LLM
tools = [
    analyze_sentiment_from_transcript,
    get_session_conversation,
]
llm_with_tools = llm.bind_tools(tools)

# Chain for processing sentiment analysis requests
sentiment_chain = sentiment_template | llm_with_tools | StrOutputParser()
