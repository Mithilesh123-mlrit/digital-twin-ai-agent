import os

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma


# -----------------------------------
# 1. Load API key
# -----------------------------------

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# -----------------------------------
# 2. Load Gemini embeddings
# -----------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=api_key,
    output_dimensionality=768
)


# -----------------------------------
# 3. Load Chroma database
# -----------------------------------

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# -----------------------------------
# 4. Create retriever
# -----------------------------------

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------------
# 5. Create Gemini LLM
# -----------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# -----------------------------------
# 6. Ask questions
# -----------------------------------

while True:

    question = input("\nYou: ").strip()

    if question.lower() in ["exit", "quit", "bye"]:
        print("Digital Twin: Goodbye!")
        break

    # Retrieve relevant resume information
    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Prompt Gemini using retrieved information
    prompt = f"""
You are the digital twin of the person described in the resume.

Answer the user's question using ONLY the resume information
provided below.

If the answer is not present in the resume, say:
"I don't have that information in my resume."

Be natural, professional, confident, and concise.

Resume information:
-------------------
{context}
-------------------

User question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nDigital Twin:", response.content)