import streamlit as st
import tempfile
import os
from ingest import ingest_documents
from query import ask

st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Knowledge Base")
st.caption("Upload your documents and ask questions about them")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_ingested" not in st.session_state:
    st.session_state.doc_ingested = False

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"]
    )

    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Reading and processing document..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                ingest_documents(tmp_path)
                os.unlink(tmp_path)

            st.session_state.doc_ingested = True
            st.session_state.chat_history = []
            st.success("Document processed! Ask your questions.")

    if st.session_state.doc_ingested:
        st.success("Document is ready")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for source in message["sources"]:
                    st.caption(source)

if st.session_state.doc_ingested:
    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask(question)
            st.write(answer)
            with st.expander("View sources"):
                for source in sources:
                    st.caption(source)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
else:
    st.info("Please upload and process a PDF document from the sidebar to get started.")