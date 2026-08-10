from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from retriever import retriever


# ==================================================
# TOOL 1: Resume / RAG search
# ==================================================

@tool
def get_resume_data(query: str) -> str:
    """
    Search the personal resume for skills, education,
    experience, projects, achievements, and background.
    """

    try:
        documents = retriever.invoke(query)

        if not documents:
            return "No relevant information was found in the resume."

        return "\n\n".join(
            document.page_content
            for document in documents
        )

    except Exception as e:
        return f"Resume search error: {e}"


# ==================================================
# TOOL 2: Calculator
# ==================================================

@tool
def calculator(expression: str) -> str:
    """
    Perform a mathematical calculation.
    Example: 125 * 48
    """

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "I could not calculate that expression."


# ==================================================
# TOOL 3: Web Search
# ==================================================

tavily_search = TavilySearch(
    max_results=5
)


@tool
def web_search(query: str) -> str:
    """
    Search the internet for current information.
    Use this for news, current events, recent technologies,
    companies, products, or information not contained
    in the resume.
    """

    try:
        result = tavily_search.invoke({
            "query": query
        })

        return str(result)

    except Exception as e:
        return f"Web search error: {e}"


# ==================================================
# All tools
# ==================================================

tools = [
    get_resume_data,
    calculator,
    web_search
]


# ==================================================
# Test tools
# ==================================================

if __name__ == "__main__":

    print("================================")
    print("Testing Calculator")
    print("================================")

    print(calculator.invoke("125 * 48"))


    print("\n================================")
    print("Testing Resume Search")
    print("================================")

    print(
        get_resume_data.invoke(
            "What are my technical skills?"
        )
    )


    print("\n================================")
    print("Testing Web Search")
    print("================================")

    print(
        web_search.invoke(
            "What are the latest developments in artificial intelligence?"
        )
    )


    print("\n================================")
    print("All tools are working!")
    print("================================")