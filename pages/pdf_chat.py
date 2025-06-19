import requests
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

backend_url=os.getenv("server2_url")

def get_groq_response(input_text):
    json_body={"input": input_text}
    headers = {'Content-Type': 'application/json'}  # Add Content-Type header
    response=requests.post(backend_url+"/invoke", json=json_body, headers=headers)
    # print(response.json())

    return response.json()

## Streamlit app
st.title("Q&A Chatbot")
st.write("This chatbot uses the LangChain RAG model to answer questions related to the document users upload.")


if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "question" not in st.session_state:
    st.session_state.question = ""

uploaded_file = st.file_uploader("Upload a document", type="pdf")

if uploaded_file is not None:
    if st.button("Process Document"):
        st.info("Processing document...")
        url = backend_url+"/process_pdf/"
        files = {"file": uploaded_file.getvalue()}

        try:
            response = requests.post(url, files=files)
            response.raise_for_status()
            result = response.json()
            st.success(result["message"])
            # Indicate file has been processed
            st.session_state.file_processed = True  
            st.session_state.question = "" 

        except requests.exceptions.RequestException as e:
            st.error(f"Error processing document: {e}")
            st.session_state.file_processed = False 


# Ask a question
input_text = st.text_input("Please enter your question below and click 'Get Response'")


if input_text and st.button("Get Response"):
    if not st.session_state.file_processed:
        st.error("Please upload and process a document before asking a question.")
    else:
        st.session_state.question = input_text #update the question in session state.
        with st.spinner("Generating answer..."):
            answer_data = get_groq_response(st.session_state.question )

        st.subheader("Answer:")
        st.write(answer_data["answer"])

        st.subheader("Sources:")
        st.write("Chatbot used the below page content as context from retriever to answer your question:")

        if "sources" in answer_data and answer_data["sources"]:
            for i, source in enumerate(answer_data["sources"], start=1):
                with st.expander(f"Source {i}"):
                    st.write(source)
        else:
            st.write("No sources found")

