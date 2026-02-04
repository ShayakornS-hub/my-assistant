
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
from typing import List, Dict, Any

# load the environment variables 
load_dotenv()

# App Configurations
st.set_page_config(
    page_title="My Assistant", 
    page_icon=":robot_face:",
    layout="wide"
    )
    
# Add a description of the app
st.markdown("""The Chatbot assistant that helps you analyze the financial data of a company""")
st.divider()

# Add a collapsible section 
with st.expander("About me",expanded= False):
    st.markdown("""
    **Model** gpt-5-nano \n
    **RAG**: File search tool using your pre-build Vector Store \n
    **Features**: multi-turn chat, images input, clear conversation \n
    **Secrets** : reads OPEN_AI_API_KEY and VECTOR_STORE_ID from .env file
    """)

# Retrieve the credentials 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Warn if OpenAI or vector store is not initialized
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set")
if not VECTOR_STORE_ID:
    st.warning("VECTOR_STORE_ID is not set")

# Configure of the system prompt
system_prompt = """
You are a very helpful fundmanager assistant that can help with the financial data of a company.
You are given a file that contains the financial data of a company.
You are also given a question about the financial data of the company.
You need to answer the question based on the financial data of the company.
You need to use the file to answer the question.
"""

# Store the previous response id 
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

#Display the conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a sidebar with user controls 
with st.sidebar:
    st.header("User Controls")
    st.divider()
    #Clear the conversation - reset chat history and context 
    if st.button("Clear Conversation", use_container_width = True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        st.success("Conversation cleared")
        #reset the page 
        st.rerun()
# Helper functions 
def build_input_parts(text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build the imput parts array for the OpenAI from text and images.

     Arg: 
     text: The text to be sent to the OpenAI 
     images: The images to be sent to the OpenAI

     Returns: 
     A list of inpout parts compatible with the OpenAI responses API
    """ 
    content =[]
    if text and text.strip(): 
        content.append({
            "type" : "input_text",
            "text": text.strip()
        })
    for img in images:
        content.append({
            "type": "input_image",
            "image_url": img["data_url"]  # API expects URL string, not object
        })
    return [{"type": "message", "role": "user", "content": content}] if content else []




# Function to generate a response from the OpenAI responses API 
def call_responses_api(parts: List[Dict[str, Any]], previous_response_id: str | None = None) -> Any:
    """
    Call the open at response with the input part 
    """
    tools = [
        {
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID],
            "max_num_results": 20,
        }
    ]
    response = client.responses.create(
        model="gpt-5-nano",
        input=parts,
        instructions=system_prompt,
        tools=tools,
        previous_response_id=previous_response_id,
    )
    return response

# Function to get the text output
def get_text_output(response: Any) -> str:
    """
    Get the text output from the OpenAI responses API.
    """
    return response.output_text


# Render all previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        content = m["content"]
        if isinstance(content, str):
            st.markdown(content)
        else:
            # content may be: [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hey"}]}]
            items = content if isinstance(content, list) else []
            if items and isinstance(items[0], dict) and items[0].get("type") == "message":
                items = items[0].get("content", [])
            for part in items:
                if isinstance(part, dict):
                    if part.get("type") == "input_text":
                        st.markdown(part.get("text", ""))
                    elif part.get("type") == "input_image" and "image_url" in part:
                        url = part["image_url"]
                        st.image(url if isinstance(url, str) else url.get("url"))



# User interface upload images (key resets after each send so images only go with that prompt)
uploaded = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key=f"upload_{len(st.session_state.messages)}",
)

# User interface - chat imput 
prompt = st.chat_input("Type your message here...")

if prompt is not None: 
    # Process the images into an API-compatible format 
    images = [
        {
            "mime": f.type or "image/png",
            "data_url": f"data:{f.type or 'image/png'};base64,{base64.b64encode(f.read()).decode('utf-8')}",
        }
        for f in (uploaded or [])
    ]

    # Build the input part for responses API 
    parts = build_input_parts(prompt,images)

    # Store the user message (plain text for display in history)
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display the users's message
    with st.chat_message('user'):
        for p in parts:
            if p['type'] == 'message':
                for content_item in p.get('content',[]) : 
                    if content_item['type'] == 'input_text':
                        st.markdown(content_item['text'])
                    elif content_item['type'] == "input_image" : 
                        url = content_item['image_url']
                        st.image(url if isinstance(url, str) else url.get('url'))
                    else:
                        st.error(f"Unknown content type: {content_item['type']}")
    #Generate the AI response
    with st.chat_message("assistant"): 
        with st.spinner("Thinking"):
            try:
                response = call_responses_api(parts, st.session_state.previous_response_id)
                output_text = get_text_output(response)

                #Display the AI's response 
                st.markdown(output_text)
                st.session_state.messages.append({'role':'assistant','content':output_text})

                # Retrieve the ID if available
                if hasattr(response, 'id') : 
                    st.session_state.previous_response_id = response.id

            except Exception as e : 
                st.error(f"Error generating response: {e}")
