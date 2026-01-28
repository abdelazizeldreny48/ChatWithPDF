# ChatWithPDF
📄 ChatWithPDF – RAG-based PDF Question Answering API

ChatWithPDF is a Retrieval-Augmented Generation (RAG) API built with FastAPI that enables users to upload PDF documents and ask natural language questions about their content.

The system extracts text from PDFs, splits it into meaningful chunks, generates embeddings, stores them in a FAISS vector database, and retrieves the most relevant context to generate accurate answers using either OpenAI models or local transformer models.

🚀 Features

Upload and ingest PDF files

Automatic text extraction and chunking

Semantic search using FAISS

Question answering over PDF content

Supports OpenAI API or local models

RESTful API built with FastAPI

Modular and production-ready structure

🛠️ Tech Stack

Python

FastAPI

FAISS

Sentence Transformers

PyPDF2

OpenAI API / HuggingFace Transformers

📌 Use Cases

Chat with documents

AI-powered document search

Knowledge base systems

RAG-based applications
