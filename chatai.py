
import os
import uuid
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
import requests

QDRANT_URL = "https://8bc6e12d-1b56-4cd3-9864-f2d491cddb81.us-east4-0.gcp.cloud.qdrant.io:6333"    # 🔑 Replace with your Qdrant cluster URL
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.yJd0ywvQfh4pJfbWfjN0BBLtsz7dcF6vw21y2odZwzI"          # 🔑 Replace with your Qdrant key
COLLECTION_NAME = "rag_demo"


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions


OPENROUTER_API_KEY =  "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"

LLM_MODEL = "openai/gpt-4o-mini"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Ensure collection exists
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance="Cosine")
    )


def load_text_from_file(filepath: str) -> str:
    text_content = ""

    if filepath.lower().endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text_content = f.read()

    elif filepath.lower().endswith(".pdf"):
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"

    elif filepath.lower().endswith(".docx"):
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text_content += para.text + "\n"

    else:
        raise ValueError("⚠️ Unsupported file type. Use .txt, .pdf, or .docx")

    return text_content.strip()

def insert_document(filepath: str):
    text = load_text_from_file(filepath)
    if not text:
        raise ValueError("⚠️ No text extracted from the file.")

    embedding = embedding_model.encode(text)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={"text": text}
    )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
    print("✅ Document inserted into Qdrant!")

def retrieve_context(query: str, top_k: int = 3):
    vector = embedding_model.encode(query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )
    return [r.payload["text"] for r in results]

def ask_rag(question: str):
    context_docs = retrieve_context(question, top_k=3)
    context = "\n".join(context_docs)

    prompt = f"""
You are a helpful assistant.
Use the following context to answer the question:

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        }
    )

    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        return f"⚠️ Error {resp.status_code}: {resp.text}"


if __name__ == "__main__":
    print("🤖 RAG Assistant with Qdrant + SentenceTransformer + OpenRouter")

    while True:
        print("\n--- Menu ---")
        print("1. Upload & insert document")
        print("2. Ask a question (RAG)")
        print("3. Exit")

        choice = input("Choose (1-3): ")

        if choice == "1":
            filepath = input("Enter file path (e.g., C:/Users/YourName/Documents/sample.pdf): ")
            try:
                insert_document(filepath)
                
                user_q = input("📄 Document added! What do you want to know from this document? ")
                answer = ask_rag(user_q)
                print(f"\n🤖 Answer: {answer}\n")
            except Exception as e:
                print("⚠️ Error:", e)

        elif choice == "2":
            query = input("Your question: ")
            answer = ask_rag(query)
            print(f"\n🤖 Answer: {answer}\n")

        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("⚠️ Invalid choice. Try again.")
