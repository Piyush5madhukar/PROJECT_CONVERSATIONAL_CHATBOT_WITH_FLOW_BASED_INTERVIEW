from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
import json
import streamlit as st

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")


gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=gemini_api_key)


with open("conversation_tree.json", "r") as file:
    conversation_tree = json.load(file)


validate_prompt = PromptTemplate(
    input_variables=["user_input", "expected_condition"],
    template="""
    You are an AI validating user responses for a structured chatbot.
    User response: "{user_input}" 
    Expected condition: "{expected_condition}"

    Does the user's response logically match the expected condition? Reply with only 'yes' or 'no'.
    """
)


def validate_response(user_input, expected_condition, llm):
    """Uses LLM to check if user_input agrees with the expected condition"""
    chain = LLMChain(llm=llm, prompt=validate_prompt)
    response = chain.run({"user_input": user_input, "expected_condition": expected_condition})
    return response.strip().lower() == "yes" 


def classify_response(user_input, edges, llm):
    """Finds the next conversation node based on user input"""
    for edge in edges:
        if validate_response(user_input, edge["condition"], llm):
            return edge["targetNodeId"]
    return None  

st.set_page_config(page_title="Conversational Chatbot with Flow-Based Interview ", layout="centered")
st.title(" Conversational Chatbot with Flow-Based Interview ")


if "current_node" not in st.session_state:
    st.session_state.current_node = conversation_tree[0] 
    st.session_state.chat_history = []
    st.session_state.conversation_ended = False


for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)


if st.session_state.current_node:
    bot_message = st.session_state.current_node["prompt"]
    if not st.session_state.chat_history or st.session_state.chat_history[-1] != ("assistant", bot_message):
        st.session_state.chat_history.append(("assistant", bot_message))
        with st.chat_message("assistant"):
            st.write(bot_message)


if not st.session_state.conversation_ended:
    user_input = st.chat_input("Your response")
else:
    st.write("The conversation has ended. Thank you!")
    user_input = None


if user_input:
    st.session_state.chat_history.append(("user", user_input))
    next_node_id = classify_response(user_input, st.session_state.current_node["edges"], gemini_llm)  # Choose LLM

    if next_node_id:
        st.session_state.current_node = next(
            (node for node in conversation_tree if node["nodeId"] == next_node_id), None
        )
    else:
        st.session_state.chat_history.append(("assistant", "The conversation has ended. Thank you!"))
        st.session_state.current_node = None
        st.session_state.conversation_ended = True

    st.rerun()
