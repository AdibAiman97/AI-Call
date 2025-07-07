"""
Example usage of the simplified automated appointment tool
"""

from appointment_tool import (
    create_appointment_auto,
    list_available_call_sessions,
    llm_with_tools,
    appointment_chain
)

def example_usage():
    """Demonstrate how to use the automated appointment tools"""
    
    print("=== Automated Appointment Tool Usage Examples ===\n")
    
    # Example 1: List available call sessions
    print("1. Listing available call sessions:")
    result = list_available_call_sessions.invoke({})
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Example 2: Auto-create appointment from call session
    print("2. Auto-creating appointment from call session:")
    result = create_appointment_auto.invoke({
        "call_session_id": 1
    })
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Example 3: Using the LLM chain for natural language processing
    print("3. Using natural language with LLM:")
    user_input = "Create an appointment automatically for call session 2"
    result = appointment_chain.invoke({"input": user_input})
    print(f"User: {user_input}")
    print(f"Assistant: {result}")
    print("\n" + "="*50 + "\n")
    
    # Example 4: Batch processing multiple call sessions
    print("4. Example of batch processing:")
    call_session_ids = [3, 4, 5]  # Example IDs
    for session_id in call_session_ids:
        result = create_appointment_auto.invoke({"call_session_id": session_id})
        print(f"Session {session_id}: {result}")

if __name__ == "__main__":
    example_usage() 