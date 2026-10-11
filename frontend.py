import uuid
import requests
import streamlit as st


# ==========================================================
# Configuration
# ==========================================================

st.set_page_config(
    page_title="Digital Twin AI",
    page_icon="🤖",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000"


# ==========================================================
# Session setup
# ==========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🤖 Digital Twin")

st.sidebar.write(
    "Personal AI assistant powered by Gemini, "
    "LangChain, RAG and useful tools."
)

st.sidebar.divider()

st.sidebar.write("Session ID")

st.sidebar.code(st.session_state.session_id)


# Check backend
try:
    response = requests.get(
        f"{API_URL}/health",
        timeout=5
    )

    if response.status_code == 200:
        st.sidebar.success("FastAPI Server Online")

    else:
        st.sidebar.error("FastAPI Server Error")

except requests.exceptions.RequestException:
    st.sidebar.error("FastAPI Server Offline")


# Reset conversation
if st.sidebar.button(
    "🔄 Reset Conversation",
    use_container_width=True
):
    try:

        requests.put(
            f"{API_URL}/sessions/{st.session_state.session_id}",
            timeout=10
        )

        st.session_state.messages = []

        st.success("Conversation reset.")

        st.rerun()

    except requests.exceptions.RequestException:
        st.error("Could not connect to FastAPI.")


# New session
if st.sidebar.button(
    "➕ New Session",
    use_container_width=True
):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []

    st.rerun()


# Delete session
if st.sidebar.button(
    "🗑 Delete Session",
    use_container_width=True
):
    try:

        response = requests.delete(
            f"{API_URL}/sessions/{st.session_state.session_id}",
            timeout=10
        )

        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []

        st.rerun()

    except requests.exceptions.RequestException:
        st.error("Could not connect to FastAPI.")


# ==========================================================
# Main UI
# ==========================================================

st.title("🤖 Digital Twin")

st.caption(
    "Ask about my skills, education, projects and experience, "
    "perform calculations, or search for current information."
)


# ==========================================================
# Tabs
# ==========================================================

chat_tab, resume_tab, tools_tab, api_tab = st.tabs(
    [
        "💬 Chat",
        "📄 Resume Search",
        "🛠 Tools",
        "⚙️ API Status"
    ]
)


# ==========================================================
# CHAT TAB
# ==========================================================

with chat_tab:

    st.subheader("Chat with the Digital Twin")

    # Show previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Chat input
    prompt = st.chat_input(
        "Ask me something..."
    )


    if prompt:

        # Show user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)


        # Call FastAPI
        with st.chat_message("assistant"):

            with st.spinner(
                "Digital Twin is thinking..."
            ):

                try:

                    response = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "message": prompt,
                            "session_id":
                                st.session_state.session_id
                        },
                        timeout=120
                    )


                    if response.status_code == 200:

                        data = response.json()

                        answer = data["response"]

                        st.markdown(answer)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer
                            }
                        )

                    else:

                        st.error(
                            f"Server error: {response.text}"
                        )


                except requests.exceptions.RequestException as e:

                    st.error(
                        f"Could not connect to FastAPI: {e}"
                    )


# ==========================================================
# RESUME SEARCH TAB
# ==========================================================

with resume_tab:

    st.subheader("📄 Resume RAG Search")

    st.write(
        "Search the vector database directly to see "
        "what information RAG retrieves from the resume."
    )


    resume_query = st.text_input(
        "Search Resume",
        placeholder="Example: What projects have I worked on?"
    )


    if st.button(
        "Search Resume",
        type="primary"
    ):

        if resume_query.strip():

            try:

                with st.spinner(
                    "Searching resume..."
                ):

                    response = requests.post(
                        f"{API_URL}/resume/search",
                        json={
                            "query": resume_query
                        },
                        timeout=60
                    )


                if response.status_code == 200:

                    data = response.json()

                    results = data.get(
                        "results",
                        []
                    )


                    if not results:

                        st.warning(
                            "No relevant information found."
                        )


                    for index, result in enumerate(
                        results,
                        start=1
                    ):

                        with st.expander(
                            f"Retrieved Chunk {index}"
                        ):

                            st.write(
                                result["content"]
                            )

                            if result.get("metadata"):

                                st.caption(
                                    f"Metadata: "
                                    f"{result['metadata']}"
                                )


                else:

                    st.error(response.text)


            except requests.exceptions.RequestException as e:

                st.error(
                    f"FastAPI connection error: {e}"
                )


# ==========================================================
# TOOLS TAB
# ==========================================================

with tools_tab:

    st.subheader("🛠 Digital Twin Tools")

    try:

        response = requests.get(
            f"{API_URL}/tools",
            timeout=10
        )


        if response.status_code == 200:

            tools = response.json()["tools"]


            for tool in tools:

                with st.container(border=True):

                    st.markdown(
                        f"### {tool['name']}"
                    )

                    st.write(
                        tool["description"]
                    )


        else:

            st.error(
                "Unable to load tools."
            )


    except requests.exceptions.RequestException:

        st.error(
            "FastAPI server is not running."
        )


    st.divider()

    st.subheader("Tool Examples")

    st.code(
        "Calculate 125 * 48"
    )

    st.code(
        "What are my technical skills?"
    )

    st.code(
        "What are the latest developments in AI?"
    )


# ==========================================================
# API STATUS TAB
# ==========================================================

with api_tab:

    st.subheader("⚙️ FastAPI Backend")

    st.write(
        f"Backend URL: `{API_URL}`"
    )


    if st.button(
        "Check API Health"
    ):

        try:

            response = requests.get(
                f"{API_URL}/health",
                timeout=10
            )


            if response.status_code == 200:

                st.success(
                    "API is healthy."
                )

                st.json(
                    response.json()
                )

            else:

                st.error(
                    "API returned an error."
                )


        except requests.exceptions.RequestException:

            st.error(
                "FastAPI server is offline."
            )


    st.divider()

    st.write(
        "FastAPI Swagger documentation:"
    )

    st.code(
        "http://127.0.0.1:8000/docs"
    )