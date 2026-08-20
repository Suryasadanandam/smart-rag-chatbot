import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(question: str) -> tuple[str, list]:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find the answer in the uploaded documents."

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    answer = response.choices[0].message.content
    sources = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        sources.append(f"Page {page + 1}: {doc.page_content[:100]}...")
    return answer, sources