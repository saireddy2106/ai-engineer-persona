from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import os
import shutil


# -----------------------------
# CONFIGURATION
# -----------------------------

CHROMA_PATH = "chroma_db"

# NEW RESUME
RESUME_PATH = "data/resume/Sabbidi_SaikumarReddy_Resume (1) (1).pdf"

# PERSONAL / AI DOCUMENTS
ABOUT_ME_PATHS = [
    "data/resume/ai_document.pdf"
]

GITHUB_READMES_PATH = "data/github/readmes"


# -----------------------------
# LOAD DOCUMENTS
# -----------------------------

documents = []


# -----------------------------
# RESUME
# -----------------------------

print("\nLoading new resume...")

if not os.path.exists(RESUME_PATH):
    raise FileNotFoundError(
        f"Resume not found: {RESUME_PATH}"
    )

loader = PyPDFLoader(RESUME_PATH)

resume_docs = loader.load()

for doc in resume_docs:

    doc.metadata = {
        "source": "resume",
        "type": "resume",
        "filename": os.path.basename(RESUME_PATH)
    }

documents.extend(resume_docs)

print(
    f"Resume Pages Loaded: {len(resume_docs)}"
)


# -----------------------------
# ABOUT ME / PERSONAL PROFILE
# -----------------------------

print("\nLoading personal profile documents...")

for about_me_path in ABOUT_ME_PATHS:

    if not os.path.exists(about_me_path):

        raise FileNotFoundError(
            f"About-me PDF not found: {about_me_path}"
        )

    loader = PyPDFLoader(about_me_path)

    about_me_docs = loader.load()

    for doc in about_me_docs:

        doc.metadata = {
            "source": "about_me",
            "type": "personal_profile",
            "filename": os.path.basename(about_me_path)
        }

    documents.extend(about_me_docs)

    print(
        f"Loaded: {os.path.basename(about_me_path)} "
        f"({len(about_me_docs)} pages)"
    )


# -----------------------------
# GITHUB REPOSITORIES
# -----------------------------

print("\nLoading GitHub repositories...")

if os.path.exists(GITHUB_READMES_PATH):

    for file in os.listdir(GITHUB_READMES_PATH):

        if file.endswith(".md"):

            file_path = os.path.join(
                GITHUB_READMES_PATH,
                file
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as f:

                content = f.read()

            project_name = file.replace(
                ".md",
                ""
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": project_name,
                        "type": "project",
                        "filename": file
                    }
                )
            )

            print(
                f"Loaded project: {project_name}"
            )


# -----------------------------
# TOTAL DOCUMENTS
# -----------------------------

print(
    f"\nTotal Documents Loaded: {len(documents)}"
)


# -----------------------------
# CHUNKING
# -----------------------------

print("\nCreating chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(
    documents
)

print(
    f"Total Chunks: {len(chunks)}"
)


# -----------------------------
# CHUNK SOURCE SUMMARY
# -----------------------------

print("\nChunk Sources:")

source_counts = {}

for chunk in chunks:

    source = chunk.metadata.get(
        "source",
        "unknown"
    )

    source_counts[source] = (
        source_counts.get(source, 0) + 1
    )

for source, count in source_counts.items():

    print(
        f"  {source}: {count} chunks"
    )


# -----------------------------
# EMBEDDINGS
# -----------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# DELETE OLD CHROMA DATABASE
# -----------------------------

if os.path.exists(CHROMA_PATH):

    print(
        f"\nRemoving old Chroma database: {CHROMA_PATH}"
    )

    shutil.rmtree(CHROMA_PATH)

    print(
        "Old Chroma database removed."
    )


# -----------------------------
# CREATE NEW CHROMA DATABASE
# -----------------------------

print("\nCreating new Chroma database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH,
    collection_name="sai_kumar_persona"
)


# -----------------------------
# COMPLETE
# -----------------------------

print("\n===================================")
print("Vector Database Created Successfully!")
print("===================================")

print("\nIndexed sources:")

for source, count in source_counts.items():

    print(
        f"  {source}: {count} chunks"
    )