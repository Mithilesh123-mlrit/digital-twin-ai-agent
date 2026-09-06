import os
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# 1. Load .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

print("Gemini API key loaded successfully.")


# 2. Resume path
pdf_path = os.path.join("data", "Resume1.pdf")

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"Resume not found: {pdf_path}")

if os.path.getsize(pdf_path) == 0:
    raise ValueError("Resume PDF is empty.")

print(f"Loading PDF: {pdf_path}")


# 3. Load PDF
loader = PyPDFLoader(pdf_path)

docs = loader.load()

print(f"Loaded {len(docs)} page(s).")


# 4. Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = splitter.split_documents(docs)

print(f"Created {len(documents)} chunks.")


# 5. Gemini embeddings
print("Creating Gemini embeddings...")

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    google_api_key=api_key,
    vertexai=False,
    output_dimensionality=768
)

print("Gemini embeddings ready.")


# 6. Create Chroma database
chroma_path = "chroma_db"

if os.path.exists(chroma_path):
    print("Removing old Chroma database...")
    shutil.rmtree(chroma_path)

print("Creating Chroma database...")

db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="digital_twin_resume",
    persist_directory=chroma_path
)

print("Vector database created successfully!")


# 7. Test retrieval
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

query = "What projects are mentioned in the resume?"

results = retriever.invoke(query)

print("\n===== RETRIEVED INFORMATION =====\n")

for i, document in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(document.page_content)
    print()