# 📚 AI Knowledge Base Chatbot

## 🚀 Overview
This project is an AI-powered chatbot built using Retrieval-Augmented Generation (RAG).  
It allows users to upload documents and ask questions based on their content.

## 🛠️ Tech Stack
- Python
- Streamlit
- LangChain
- Qdrant (Vector Database)
- HuggingFace Transformers
- Docker

## ⚙️ Features
- Upload PDF/TXT files
- Semantic search using embeddings
- Context-based question answering
- Interactive UI using Streamlit
- Docker deployment

## 🧠 How It Works
1. Upload document
2. Split into chunks
3. Convert to embeddings
4. Store in Qdrant
5. Retrieve relevant chunks
6. Generate answer using LLM

## ▶️ Run Project

```bash
docker-compose up --build
