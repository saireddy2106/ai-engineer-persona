from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

with open(
    "prompts/persona_prompt.txt",
    "r",
    encoding="utf-8"
) as f:
    persona_prompt = f.read()
    
while True:

    question = input("\nAsk a Question: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(
        question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = f"""
{persona_prompt}

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(final_prompt)

    print("\nAI Persona:\n")
    print(response.content)