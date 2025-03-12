import json
import os
import google.generativeai as genai
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    st.error("Error: API key not found. Set GEMINI_API_KEY in a .env file.")
    st.stop()


genai.configure(api_key=API_KEY)

#Loads the structured conversation tree from a JSON file.
try:
    with open("conversation_tree.json", "r") as file:
        conversation_tree = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    st.error("Error: Invalid or missing conversation_tree.json.")#If the file is missing or invalid, the application stops.
    st.stop()

#Fetches a specific conversation node by its nodeId from the JSON tree
def get_node(node_id):
    return next((node for node in conversation_tree if node["nodeId"] == node_id), None)

# Validate user response using Gemini LLM
def validate_response(user_input, expected_condition):
    """Uses LLM to check if user_input agrees with the expected condition"""
    prompt = f"""
    You are an AI assistant validating user input for a structured chatbot. 
    User response: "{user_input}" 
    Expected condition: "{expected_condition}"

    Does the user's response logically align with the expected condition?
    Reply with only 'yes' or 'no'.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip().lower() == "yes"
    except Exception as e:
        st.error(f"LLM processing error: {e}")
        return False

# Determine the next node based on user input
def classify_response(user_input, edges):
    if not edges:
        return None
    for edge in edges: #Fetches a specific conversation node by its nodeId from the JSON tree
        if validate_response(user_input, edge["condition"]):  # Use LLM to validate
            return edge["targetNodeId"]
    return None


st.set_page_config(page_title="Flow-Based Chatbot", layout="centered")
st.title("Flow-Based Chatbot")
st.write("Chatbot that follows a structured conversation flow using LLM.")

# Initialize session state-->Initialize Chatbot State
if "current_node" not in st.session_state:#Tracks the active conversation node.
    st.session_state.current_node = get_node("node1")
    st.session_state.chat_history = [] #Stores past messages.
    st.session_state.conversation_ended = False  # Track end of conversation

# Display previous chat messages  (both user and LLM responses).
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# Show chatbot message (only if it's a new message)
if st.session_state.current_node:
    bot_message = st.session_state.current_node["prompt"]
    
    # Prevent duplicate bot messages
    if not st.session_state.chat_history or st.session_state.chat_history[-1] != ("assistant", bot_message):
        st.session_state.chat_history.append(("assistant", bot_message))
        with st.chat_message("assistant"):
            st.write(bot_message)

# Disable chat input if conversation has ended
if not st.session_state.conversation_ended:
    user_input = st.chat_input("Your response")
else:
    st.write(" The conversation has ended. Thank you!")
    user_input = None

# Process user input (without UI flickering)
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    
    next_node_id = classify_response(user_input, st.session_state.current_node["edges"])
    
    if next_node_id:
        st.session_state.current_node = get_node(next_node_id)
    else:
        # End conversation if no valid next node exists
        st.session_state.chat_history.append(("assistant", " The conversation has ended. Thank you!"))
        st.session_state.current_node = None
        st.session_state.conversation_ended = True

    st.rerun()
