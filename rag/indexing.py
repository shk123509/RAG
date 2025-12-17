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
    # ===== Getting Started =====
    "https://docs.chaicode.com/youtube/getting-started/",

    # ===== Chai aur HTML =====
    "https://docs.chaicode.com/youtube/chai-aur-html/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-html/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-html/emmit-crash-course/",
    "https://docs.chaicode.com/youtube/chai-aur-html/html-tags/",

    # ===== Chai aur Git =====
    "https://docs.chaicode.com/youtube/chai-aur-git/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-git/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-git/terminology/",
    "https://docs.chaicode.com/youtube/chai-aur-git/behind-the-scenes/",
    "https://docs.chaicode.com/youtube/chai-aur-git/branches/",
    "https://docs.chaicode.com/youtube/chai-aur-git/diff-stash-tags/",
    "https://docs.chaicode.com/youtube/chai-aur-git/managing-history/",
    "https://docs.chaicode.com/youtube/chai-aur-git/github/",

    # ===== Chai aur C =====
    "https://docs.chaicode.com/youtube/chai-aur-c/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-c/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-c/hello-world/",
    "https://docs.chaicode.com/youtube/chai-aur-c/variables-and-constants/",
    "https://docs.chaicode.com/youtube/chai-aur-c/data-types/",
    "https://docs.chaicode.com/youtube/chai-aur-c/operators/",
    "https://docs.chaicode.com/youtube/chai-aur-c/control-flow/",
    "https://docs.chaicode.com/youtube/chai-aur-c/loops/",
    "https://docs.chaicode.com/youtube/chai-aur-c/functions/",

    # ===== Chai aur Django =====
    "https://docs.chaicode.com/youtube/chai-aur-django/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-django/getting-started/",
    "https://docs.chaicode.com/youtube/chai-aur-django/jinja-templates/",
    "https://docs.chaicode.com/youtube/chai-aur-django/tailwind/",
    "https://docs.chaicode.com/youtube/chai-aur-django/models/",

    # ===== Chai aur SQL =====
    "https://docs.chaicode.com/youtube/chai-aur-sql/welcome/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/introduction/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/postgres/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/normalization/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/database-design-exercise/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-and-keys/",
    "https://docs.chaicode.com/youtube/chai-aur-sql/joins-exercise/",

    # ===== Chai aur DevOps =====
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