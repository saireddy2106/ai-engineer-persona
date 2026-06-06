# AI Persona Architecture

## Overview

This project is an AI-powered persona assistant that represents Sai Kumar during interviews. The system uses Retrieval-Augmented Generation (RAG) to answer questions based on Sai Kumar's resume, projects, and professional experience.

## System Architecture

User
↓
Streamlit Chat Interface
↓
Query Routing Layer
↓
Chroma Vector Database
↓
Retrieved Context
↓
Groq LLM (Llama 3.3 70B)
↓
Generated Response

## Components

### 1. Streamlit Interface

Provides a conversational chat interface where recruiters can ask questions about Sai Kumar's background, skills, projects, and experience.

### 2. Document Ingestion Pipeline

The ingestion pipeline loads:

* Resume PDF
* Project README files

Documents are chunked and embedded before being stored in ChromaDB.

### 3. Embedding Model

Sentence Transformers:

all-MiniLM-L6-v2

Used to generate vector embeddings for semantic retrieval.

### 4. Vector Database

ChromaDB stores document embeddings and metadata.

Metadata includes:

* resume
* roots_of_change
* hand_gesture
* chatbot_langgraph

### 5. Retrieval Layer

The system retrieves the most relevant chunks based on the user's question.

Project-specific routing is used to improve retrieval accuracy.

### 6. Large Language Model

Groq-hosted Llama 3.3 70B is used to generate final responses using retrieved context.

### 7. Prompt Layer

The persona prompt ensures:

* No hallucinations
* Accurate representation
* Clear distinction between implemented features and future plans
* Professional interview responses

## Current Capabilities

* Resume Q&A
* Project discussions
* Skill-based questions
* Career background questions
* Context-aware responses

## Future Extensions

* Voice Agent
* Calendar Scheduling
* Google Calendar Integration
* Public Deployment
* Automated Evaluation Pipeline
