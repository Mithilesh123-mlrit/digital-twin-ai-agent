import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    api_key=api_key,
    vertexai=False,
    output_dimensionality=768
)


# Existing Chroma database
db = Chroma(
    collection_name="digital_twin_resume",
    persist_directory="chroma_db",
    embedding_function=embeddings
)


retriever = db.as_retriever(
    search_kwargs={"k": 3}
)