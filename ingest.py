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
        "source": "resume",
        "type": "resume"
    }

documents.extend(resume_docs)

print(f"Resume Pages Loaded: {len(resume_docs)}")

# -----------------------------
# GITHUB REPOSITORIES
# -----------------------------

readmes_path = r"data/github/readmes"

for file in os.listdir(readmes_path):

    if file.endswith(".md"):

        file_path = os.path.join(
            readmes_path,
            file
        )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read()

        project_name = file.replace(".md", "")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": project_name,
                    "type": "project"
                }
            )
        )

        print(f"Loaded: {file}")
print(f"\nTotal Documents Loaded: {len(documents)}")

# -----------------------------
# CHUNKING
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(
    documents
)

print(f"\nTotal Chunks: {len(chunks)}")
print("\nChunk Sources:\n")

for chunk in chunks[:10]:
    print(chunk.metadata)
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