from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()



llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


class Question(BaseModel):
    question: str



@app.post("/ask")
def ask_question(data: Question):

    docs = db.similarity_search(
        data.question,
        k=5
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are Sai Kumar's AI representative.

Answer only from context.

Context:
{context}

Question:
{data.question}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }