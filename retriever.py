import os

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load API key
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# Create the same embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=api_key,
    output_dimensionality=768
)


# Load existing Chroma database
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# Create retriever
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# Test question
query = input("Ask something about the resume: ")

results = retriever.invoke(query)


print("\n===== RELEVANT RESUME INFORMATION =====\n")

for i, document in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(document.page_content)
    print()