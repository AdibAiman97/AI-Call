from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from conversation_manager import process_query_with_rag_llm
from rag_functions import initialize_embedding_model, initialize_embeddings
from llm_service import initialize_llm_model

app = FastAPI(
    title="AI Call Center Backend",
    description="FastAPI backend for AI Call Center RAG and LLM components.",
    version="0.1.0",
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """
    This function runs when the FastAPI application starts up.
    We'll use it to initialize both LLM and RAG embedding models.
    """
    print("Application startup: Initializing Vertex AI models for RAG and LLM...")

    initialize_embedding_model()
    initialize_embeddings()
    initialize_llm_model()

    print("Application startup: All models initialization complete (if successful).")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <title>FastAPI Backend Running</title>
        </head>
        <body>
            <h1>FastAPI Backend for AI Call Center is Running!</h1>
            <p>This is your AI Call Center backend, handling RAG and LLM processes.</p>
            <p>Please use your Vue.js frontend (<code>index.html</code>) to interact with the AI assistant.</p>
            <p>For API documentation, visit <a href="/docs">/docs</a> or <a href="/redoc">/redoc</a>.</p>
        </body>
    </html>
    """


@app.post("/query")
async def handle_chat_query(request: Request):
    try:
        data = await request.json()
        user_query = data.get("query")
        conversation_history = data.get("history", [])

        if not user_query:
            raise HTTPException(status_code=400, detail="'query' parameter is missing.")

        print(f"Received query from frontend: '{user_query}'")
        print(
            f"Received history (first 500 chars): {json.dumps(conversation_history, indent=2)[:500]}..."
        )

        ai_response_text = await process_query_with_rag_llm(
            user_query, conversation_history
        )

        return JSONResponse(content={"response": ai_response_text})

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON format in request body."
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unhandled error occurred during query processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
