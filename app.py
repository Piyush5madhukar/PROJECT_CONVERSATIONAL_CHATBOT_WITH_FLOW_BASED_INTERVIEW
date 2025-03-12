import json
import os
import google.generativeai as genai
from dotenv import load_dotenv 


load_dotenv()


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: API key not found. Set GEMINI_API_KEY in a .env file.")
    exit(1)

genai.configure(api_key=API_KEY)


try:
    with open("conversation_tree.json", "r") as file:
        conversation_tree = json.load(file)
except FileNotFoundError:
    print("Error: conversation_tree.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON format in conversation_tree.json.")
    exit(1)

def get_node(node_id):
    """Retrieve node by ID from the conversation tree."""
    return next((node for node in conversation_tree if node["nodeId"] == node_id), None)

def validate_response(user_input, expected_conditions):
    """Use LLM to determine if the user's response correctly matches an expected condition."""
    prompt = f"Given the user response: '{user_input}', determine if the user is affirming they are John. Respond with only 'yes' or 'no'."
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        decision = response.text.strip().lower()
        return decision == "yes"
    except Exception as e:
        print(f"Error in LLM processing: {e}")
        return False

def classify_response(user_input, edges):
    """Check if the user's response correctly matches a defined condition using the LLM."""
    if not edges:
        return None
    
    for edge in edges:
        if edge["condition"].lower() == "user is john":
            if validate_response(user_input, [edge["condition"]]):
                return edge["targetNodeId"]
        else:
            if validate_response(user_input, [edge["condition"]]):
                return edge["targetNodeId"]
    
    return None

def chatbot():
    """Main chatbot loop."""
    current_node = get_node("node1")  
    
    if not current_node:
        print("Error: Root node not found in conversation tree.")
        return
    
    while current_node:
        print(f"AI: {current_node['prompt']}")
        user_input = input("You: ").strip()
        
        if not user_input:
            print("AI: I didn't catch that. Could you please repeat?")
            continue
        
        next_node_id = classify_response(user_input, current_node["edges"])
        if next_node_id:
            current_node = get_node(next_node_id)
        else:
            print("AI: That response does not match any valid condition. Ending chat.")
            break

if __name__ == "__main__":
    chatbot()