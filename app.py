import streamlit as st
from rag_chatbot import ask_question, ingest_document

st.set_page_config(page_title="AI Knowledge Chatbot", layout="wide")

st.title("📚 AI Knowledge Base Chatbot")

st.sidebar.header("Upload Knowledge")

uploaded_file = st.sidebar.file_uploader(
    "Upload TXT or PDF file",
    type=["txt", "pdf"]
)

if uploaded_file:
    with st.spinner("Processing document..."):
        ingest_document(uploaded_file)
    st.sidebar.success("Document added to knowledge base!")

st.write("Ask questions from the uploaded knowledge.")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask a question")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        answer, sources = ask_question(user_input)

    st.session_state.messages.append({"role": "assistant", "content": answer})

for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

            if "sources" in message:
                st.caption("Sources:")
                for s in message["sources"]:
                    st.write("-", s)