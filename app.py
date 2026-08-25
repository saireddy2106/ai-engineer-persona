import streamlit as st

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


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

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        collection_name="sai_kumar_persona",
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

    # -----------------------------------
    # STORE USER MESSAGE
    # -----------------------------------

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


    # ===================================
    # GREETING
    # ===================================

    if question.lower().strip() in greetings:

        answer = """
Hello 👋

I'm Sai Kumar's AI Persona — an AI-powered representation of his
professional journey, technical work, AI expertise, and career vision.

Sai Kumar is an AI-focused Software Engineer with hands-on experience
building AI/ML applications, LLM-powered systems, RAG pipelines,
LangGraph workflows, and intelligent software solutions.

He is passionate about turning AI concepts into practical products,
exploring Agentic AI, and continuously improving his ability to build
production-oriented AI systems.

Beyond engineering, he has a strong builder and founder mindset, with
a long-term ambition to create AI-driven products and technology
ventures that solve meaningful real-world problems.

You can explore:

• 🤖 AI Engineering & Technical Expertise
• 🧠 LLMs, RAG & Agentic AI
• 💻 LangGraph AI Chatbot
• ✋ Hand Gesture Control System
• 🌱 Roots of Change
• 🏢 Professional & Internship Experience
• 🚀 Startup & Product Ideas
• 🎯 Career Vision & Goals
• 👨‍💻 Engineering & Founder Mindset

Ask me anything about Sai Kumar.
"""


    # ===================================
    # RAG QUESTION
    # ===================================

    else:

        try:

            question_lower = question.lower()


            # -----------------------------------
            # PERSONAL / AI JOURNEY
            # -----------------------------------

            if any(word in question_lower for word in [
                "passion",
                "passionate",
                "career goal",
                "career goals",
                "career vision",
                "future",
                "founder",
                "founder mindset",
                "entrepreneur",
                "entrepreneurship",
                "startup",
                "startup idea",
                "startup ideas",
                "vision",
                "motivation",
                "ai journey",
                "about me",
                "long term",
                "long-term",
                "ambition",
                "why ai",
                "why artificial intelligence",
                "ai philosophy"
            ]):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source": "about_me"
                    }
                )


            # -----------------------------------
            # RESUME
            # -----------------------------------

            elif any(word in question_lower for word in [
                "skill",
                "skills",
                "education",
                "cgpa",
                "resume",
                "academic",
                "degree",
                "university",
                "internship",
                "experience",
                "professional experience",
                "work experience",
                "qualification"
            ]):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source": "resume"
                    }
                )


            # -----------------------------------
            # ROOTS OF CHANGE
            # -----------------------------------

            elif (
                "roots of change" in question_lower
                or "roots" in question_lower
            ):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source": "roots_of_change"
                    }
                )


            # -----------------------------------
            # HAND GESTURE
            # -----------------------------------

            elif (
                "hand gesture" in question_lower
                or "gesture" in question_lower
            ):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source": "hand_gesture"
                    }
                )


            # -----------------------------------
            # LANGGRAPH CHATBOT
            # -----------------------------------

            elif (
                "langgraph" in question_lower
                or "chatbot" in question_lower
            ):

                docs = db.similarity_search(
                    question,
                    k=5,
                    filter={
                        "source": "chatbot_langgraph"
                    }
                )


            # -----------------------------------
            # GENERAL SEMANTIC SEARCH
            # -----------------------------------

            else:

                docs = db.similarity_search(
                    question,
                    k=5
                )


            # -----------------------------------
            # BUILD CONTEXT
            # -----------------------------------

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )


            # -----------------------------------
            # EMPTY RETRIEVAL
            # -----------------------------------

            if not context.strip():

                answer = (
                    "I don't have enough information to answer that."
                )


            else:

                # -----------------------------------
                # CHAT HISTORY
                # -----------------------------------

                history = ""

                for msg in st.session_state.messages[-6:]:

                    history += (
                        f"{msg['role']}: "
                        f"{msg['content']}\n"
                    )


                # -----------------------------------
                # PROMPT
                # -----------------------------------

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


                # -----------------------------------
                # LLM
                # -----------------------------------

                response = llm.invoke(prompt)

                answer = response.content


        except Exception as e:

            answer = f"""
⚠️ Error

{str(e)}
"""


    # -----------------------------------
    # STORE ASSISTANT RESPONSE
    # -----------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    with st.chat_message("assistant"):

        st.markdown(answer)