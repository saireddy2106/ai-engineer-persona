from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


#lading database
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


#asking questions
query = "What are Sai Kumar's technical skills?"


#retrive

results = db.similarity_search(
    query,
    k=2
)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("-" * 50)
    print(doc.page_content)