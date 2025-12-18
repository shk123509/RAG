from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from dotenv import load_dotenv
import os

# ================== SETUP ==================
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6334",
    collection_name="RAG_DOC",
    embedding=embeddings
)

# ================== FASTAPI ==================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== REQUEST MODEL ==================
class ChatRequest(BaseModel):
    query: str

# ================== CHAT ENDPOINT ==================
@app.post("/chat")
def chat(data: ChatRequest):
    query = data.query

    results = vector_store.similarity_search(query=query, k=5)

    if not results:
        return {
            "answer": "The answer is not available in the provided sources.",
            "sources": []
        }

    # 🔥 Primary + Supporting Context
    primary = results[0]
    supporting = results[1:]

    context = f"""
PRIMARY CONTEXT (Most Relevant):
Title: {primary.metadata.get("title")}
Page: {primary.metadata.get("page", "N/A")}
Source: {primary.metadata.get("source")}

Content:
{primary.page_content}

SUPPORTING CONTEXT:
"""

    for i, doc in enumerate(supporting, start=1):
        context += f"""
--- Supporting Doc {i} ---
Title: {doc.metadata.get("title")}
Page: {doc.metadata.get("page", "N/A")}
Source: {doc.metadata.get("source")}

Content:
{doc.page_content}
"""

    SYSTEM_PROMPT = f"""
You are a helpful AI assistant.

Your task is to answer the user's question strictly based on the provided context
retrieved from the knowledge base.

Rules you MUST follow:
1. Use ONLY the given context to answer. Do NOT add outside knowledge.
2. If the answer is not present in the context, respond with:
   "The answer is not available in the provided sources."
3. Provide a clear, concise, and well-structured explanation.
4. Mention the document title and the exact page number(s) where the information appears.
5. Always include sources in a separate "Sources" section.
6. Guide the user to open the relevant page number(s) to learn more.
7. Do NOT fabricate or assume page numbers, document titles, or sources.
8. If multiple documents are used, list each separately.

Use the following answer format ONLY:

Answer:
<clear explanation based on context>

Learn more:
- Document: <Title>


Sources:
- <Source URL or Document Name>

Context:
{context}

"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[
            {"role": "user", "parts": [{"text": SYSTEM_PROMPT}]},
            {"role": "user", "parts": [{"text": query}]}
        ]
    )

    return {
        "response": response.text,
        "sources": list({doc.metadata.get("source") for doc in results})
    }
