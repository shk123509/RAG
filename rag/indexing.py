from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
import os
from dotenv import load_dotenv


load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# URI 

urls = [

    "https://docs.chaicode.com/youtube/chai-aur-devops/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-vpc/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/setup-nginx/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-rate-limiting/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/nginx-ssl-setup/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-nginx-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-docker/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/postgresql-vps/",
    "https://docs.chaicode.com/youtube/chai-aur-devops/node-logger/",
]


loader_multiple_pages = WebBaseLoader(urls)

doc = loader_multiple_pages.load()

print("LEN : " , len(doc))

# Chunking of text

text_split = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500
)

split_doc = text_split.split_documents(documents=doc)

# Vector  Embedding 

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
# Using [embedding_model] create embeddings of [split_docs] and store in DB

vector_store = QdrantVectorStore.from_documents(
    documents=split_doc,
    url="http://localhost:6334",
    collection_name="RAG_DOC",
    embedding=embeddings
)

print("Indexing creating successfully...")