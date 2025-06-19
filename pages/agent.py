import os
import streamlit as st
import requests
import json
from dotenv import load_dotenv

load_dotenv()
backend_url = os.getenv("server1_url")

st.set_page_config(page_title="AI Agent Chat", page_icon="💬", layout="wide")
st.title("Agent Search: Web, Wikipedia, and Arxiv")

query = st.chat_input("Please enter your question")

if query:
    st.session_state.messages = []
    st.session_state.full_response = ""

    st.chat_message("user").write(query)
    assistant_msg = st.chat_message("assistant")
    response_container = assistant_msg.container()

    def get_streaming_response(query: str):
        payload = {"query": query}
        headers = {"Content-Type": "application/json"}
        with requests.post(
            backend_url + "/stream_chat",
            json=payload,
            headers=headers,
            stream=True,
            timeout=180
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk:
                    try:
                        data = json.loads(chunk)
                        if data["type"] == "agent_action":
                            print(f"[DEBUG] Tool: {data['tool']} | Input: {data['tool_input']}")
                            continue
                        elif data["type"] == "final_answer":
                            yield f"\n\n**Answer:**\n\n{data['output']}"
                        elif data["type"] == "error":
                            yield f"❌ Error: {data['error']}"
                    except json.JSONDecodeError:
                        continue

    with st.spinner("Thinking..."):
        for response_chunk in get_streaming_response(query):
            st.session_state.full_response += str(response_chunk)
            st.session_state.messages.append({"role": "assistant", "content": response_chunk})
            response_container.write(st.session_state.full_response)
