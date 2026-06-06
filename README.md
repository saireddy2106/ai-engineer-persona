# AI Persona – Sai Kumar

## Overview

This project implements an AI-powered digital persona of Sai Kumar that can:

* Answer questions about education, skills, projects, and experience
* Retrieve information from resume and GitHub repositories using RAG
* Conduct voice conversations through Vapi
* Check Google Calendar availability
* Schedule interviews automatically

## Architecture

Voice Agent:
Vapi → LLM → Google Calendar

Chat Agent:
Streamlit → ChromaDB → Resume + GitHub Data → Groq LLM

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* HuggingFace Embeddings
* Groq LLM
* Vapi
* Google Calendar API

## Features

### Voice Agent

* Real-time voice conversations
* Calendar availability checking
* Automated interview scheduling

### Chat Agent

* Resume Question Answering
* GitHub Repository Question Answering
* Retrieval Augmented Generation (RAG)

## Setup

1. Clone repository
2. Install requirements

pip install -r requirements.txt

3. Add API keys in .env

4. Run ingestion

python ingest.py

5. Start application

streamlit run app.py

## Cost Breakdown

Voice:

* Vapi
* Deepgram
* LLM

Approximate cost:
$0.05–0.10 per minute

Chat:

* Groq API
* Local ChromaDB
* Local Embeddings

Approximate cost:
Less than $0.01 per session

## Future Improvements

* Multi-agent architecture
* Better retrieval evaluation
* Resume versioning
* Email notifications
* Meeting reminders
