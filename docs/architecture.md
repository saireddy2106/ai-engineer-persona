# Architecture Overview

User
↓
Streamlit Chat Interface
↓
Question Processing
↓
Chroma Vector Database
↓
Relevant Context Retrieval
↓
Gemini 2.5 Flash
↓
Persona Prompt
↓
Response Generation
↓
User

Data Sources

1. Resume PDF
2. Roots of Change Repository
3. LangGraph Chatbot Repository
4. Hand Gesture Repository

Core Components

* Streamlit
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Gemini 2.5 Flash
* Retrieval-Augmented Generation (RAG)
