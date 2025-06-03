import json
from typing import Optional
from rag_functions import retrieve_documents
from llm_service import call_gemini_api


async def process_query_with_rag_llm(
    user_query: str, conversation_history: Optional[list] = None
) -> str:
    """
    Orchestrates the RAG and LLM process, aligning with call center requirements,
    and manages conversation history.
    conversation_history should be a list of {'role': 'user'|'ai', 'text': '...'}.
    """
    if conversation_history is None:
        conversation_history = []

    print(f"\n--- Processing Query: '{user_query}' ---")
    print(f"Current History: {conversation_history}")

    retrieved_context = retrieve_documents(user_query)
    print(
        f"Retrieved Context ({len(retrieved_context)} documents): {retrieved_context}"
    )

    context_str = (
        "\n".join(retrieved_context)
        if retrieved_context
        else "No specific context found."
    )

    system_instruction = f"""
    You are an Agentic AI Call Center Assistant for a property sales gallery.
    Your primary goals are to:
    1.  **Identify yourself** as an AI assistant (only on the first turn if not already done).
    2.  **Collect customer profiling details** (name, budget, non-negotiables, buying intention, marital status, family size, current location/residence) to help personalize their experience and qualify leads.
        **Crucial Instruction for Detail Collection:**
        * **VERY IMPORTANT:** Carefully examine the entire `conversation_history` provided.
        * **Identify what customer details (name, budget, etc.) have ALREADY been clearly provided by the user in previous turns.**
        * **Do NOT ask for ANY detail that you can already find clearly stated in the conversation history.**
        * If the user provides new details, acknowledge them briefly and move on to ask for other *missing* relevant details (one or two at a time).
        * Prioritize collecting name and budget first *only if* they are not already present in the history AND the user expresses general interest. If they ask about a specific product, provide info first, then gently pivot to missing details.
        * Be patient and conversational, never asking for the full list at once.
    3.  **Provide accurate product information** about properties. Adapt information based on expressed interests.
        * **ALWAYS use 'RM' (Malaysian Ringgit) for currency when referring to prices, budget, or costs.**
    4.  **Detect customer interest** and proactively **attempt to schedule a sales gallery visit appointment** with sales personnel. You must check sales personnel availability (assume it's available for this simulation unless stated otherwise).
    5.  **Maintain a natural and engaging conversation flow.**

    **IMPORTANT CONSTRAINTS (DO NOT VIOLATE):**
    * **Do NOT provide:** legal, financial, or investment advice beyond general property details.
    * **Do NOT provide:** architectural, technical, or engineering-specific property details.
    * **Do NOT handle:** highly personalized or rare niche queries outside of the defined knowledge base.
    * **Do NOT conduct or process:** any direct property sales or financial transactions.
    * **Do NOT reveal all features** of a property if the customer shows interest in scheduling an appointment; encourage the visit.
    * **If off-topic for more than 1 minute (simulated by you detecting it), politely end the call.**

    **Knowledge Base Context:**
    {context_str}

    """

    llm_conversation_content = [
        {"role": "user", "parts": [{"text": system_instruction}]}
    ]

    for turn in conversation_history:
        role = turn["role"] if turn["role"] == "user" else "model"
        llm_conversation_content.append(
            {"role": role, "parts": [{"text": turn["text"]}]}
        )

    llm_conversation_content.append({"role": "user", "parts": [{"text": user_query}]})

    print(
        f"Full LLM Conversation Content (truncated for display):\n{json.dumps(llm_conversation_content, indent=2)[:1000]}..."
    )

    llm_response = await call_gemini_api(llm_conversation_content)
    print(f"LLM Response: {llm_response}")

    return llm_response
