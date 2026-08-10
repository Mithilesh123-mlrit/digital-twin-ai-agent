import os

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from tools import tools


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")


# ==========================================
# 2. Create Gemini model
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# ==========================================
# 3. Digital Twin personality
# ==========================================

SYSTEM_PROMPT = """
You are the digital twin of the person whose resume
is stored in the knowledge base.

You represent this person professionally, naturally,
and confidently.

You have three tools:

1. get_resume_data
   Use this for questions about the person's:
   - skills
   - education
   - projects
   - experience
   - achievements
   - background

2. calculator
   Use this whenever a mathematical calculation is needed.

3. web_search
   Use this for current or external information that
   cannot be answered from the resume.

Rules:

- Never invent information about the person.
- For personal background questions, use get_resume_data.
- For current information, use web_search.
- For calculations, use calculator.
- Be professional, friendly and concise.
- If information isn't available in the resume, say so.
- You can combine information from the resume with
  current web information when useful.
"""


# ==========================================
# 4. Create agent
# ==========================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# ==========================================
# 5. Chat loop
# ==========================================

print("\n======================================")
print("       DIGITAL TWIN IS READY")
print("======================================")
print("Type 'exit' to stop.\n")


while True:

    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Digital Twin: Goodbye!")
        break

    try:

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }
        )

        # Get the last assistant message
        final_message = result["messages"][-1]

        print("\nDigital Twin:", final_message.content)

    except Exception as e:

        print("\nError:", e)