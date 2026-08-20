import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest_documents(pdf_path: str):
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    print(f"Creating embeddings for {len(chunks)} chunks...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Saving to vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")

    print(f"Done! Ingested {len(chunks)} chunks.")