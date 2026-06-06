import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Sai Kumar AI Persona",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Sai Kumar AI Persona")

# -----------------------------------
# LOAD PROMPT
# -----------------------------------

with open(
    "prompts/persona_prompt.txt",
    "r",
    encoding="utf-8"
) as f:

    persona_prompt = f.read()

# -----------------------------------
# CACHE
# -----------------------------------

@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )


@st.cache_resource
def load_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )


llm = load_llm()
db = load_db()

# -----------------------------------
# CHAT HISTORY
# -----------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------------
# INPUT
# -----------------------------------

question = st.chat_input(
    "Ask me anything about Sai Kumar..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if question.lower().strip() in greetings:

        answer = """
Hello 👋

I'm Sai Kumar's AI representative.

You can ask me about:

- Education
- Skills
- Internship Experience
- Roots of Change
- LangGraph AI Chatbot
- Hand Gesture Control System
- AI/ML Experience
- Career Goals
"""

    else:

        try:

            question_lower = question.lower()

            # -------------------------
            # ROUTING
            # -------------------------

            if any(word in question_lower for word in [
                "skill",
                "education",
                "cgpa",
                "resume",
                "background",
                "internship",
                "experience"
            ]):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={"source": "resume"}
                )

            elif "roots" in question_lower:

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source":
                        "rootsofchange_modernizing-Agriculture"
                    }
                )

            elif (
                "hand gesture" in question_lower
                or "gesture" in question_lower
            ):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source":
                        "HandGesture-Project"
                    }
                )

            elif (
                "langgraph" in question_lower
                or "chatbot" in question_lower
            ):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source":
                        "chatbot-in-langgraph"
                    }
                )

            else:

                docs = db.similarity_search(
                    question,
                    k=5
                )

            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            # -------------------------
            # CHAT HISTORY
            # -------------------------

            history = ""

            for msg in st.session_state.messages[-6:]:

                history += (
                    f"{msg['role']}: "
                    f"{msg['content']}\n"
                )

            # -------------------------
            # PROMPT
            # -------------------------

            prompt = f"""
{persona_prompt}

Conversation History:
{history}

Retrieved Context:
{context}

User Question:
{question}

Answer:
"""

            response = llm.invoke(
                prompt
            )

            answer = response.content

        except Exception as e:

            answer = f"""
⚠️ Error

{str(e)}
"""

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):
        st.markdown(answer)