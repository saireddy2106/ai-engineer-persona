from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import os

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------

documents = []

# -----------------------------
# RESUME
# -----------------------------

pdf_path = "data/resume/sabbidi_saikumar_resume.pdf"

loader = PyPDFLoader(pdf_path)

resume_docs = loader.load()

for doc in resume_docs:
    doc.metadata = {
        "source": "resume"
    }

documents.extend(resume_docs)

print(f"Resume Pages Loaded: {len(resume_docs)}")

# -----------------------------
# GITHUB REPOSITORIES
# -----------------------------

repos_path = r"data/github/repos"

for repo in os.listdir(repos_path):

    repo_path = os.path.join(repos_path, repo)

    readme_path = os.path.join(
        repo_path,
        "README.md"
    )

    if os.path.exists(readme_path):

        with open(
            readme_path,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": repo
                }
            )
        )

        print(f"Loaded README from: {repo}")

print(f"\nTotal Documents Loaded: {len(documents)}")

# -----------------------------
# CHUNKING
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(
    documents
)

print(f"\nTotal Chunks: {len(chunks)}")

# -----------------------------
# EMBEDDINGS
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# CHROMA DB
# -----------------------------

print("\nCreating ChromaDB...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("\nVector Database Created Successfully!")