from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
import os
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/gemini-embedding-001"
# )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6334",
    collection_name="RAG",
    embedding=embeddings
)

query = input("> ")

search_result = vector_store.similarity_search(
    query=query,
    k=8
)
context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage title: {result.metadata['title']}\Page description: {result.metadata['description']}\n content source : {result.metadata['source']}" for result in search_result])




SYSTEM_PROMPTS = f"""

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

message = [
    {"role" : "user", "parts": [{"text": SYSTEM_PROMPTS}]}
]

message.append({"role" : "user", "parts": [{"text": query}]})

while True:
    response = client.models.generate_content(
    model="gemini-flash-latest",
        contents=message
    )

    message.append({"role" : "model", "parts" : [{"text" : response.text}]})

    print(f"🤖 : {response.text}")


    if not response.text:
      print("🤖 : (empty response, stopping)")
    break
  