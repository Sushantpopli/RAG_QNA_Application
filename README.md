# Document RAG Q&A App

A Streamlit application that lets users upload documents or provide URLs, creates embeddings, stores them in FAISS, and answers questions using retrieval-augmented generation.

## Features

- Supports PDF, DOCX, TXT, pasted text, and webpage URLs.
- Splits documents into searchable chunks.
- Uses Hugging Face sentence-transformer embeddings.
- Stores vectors with FAISS for semantic retrieval.
- Answers with source citations from retrieved chunks.
- Includes a local extractive fallback when the hosted LLM API is unavailable.

## Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face
- FAISS
- PyPDF2
- python-docx

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Create a `.env` file:

```bash
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
USE_HF_LLM=false
```

Set `USE_HF_LLM=true` only when you have a working Hugging Face inference token/model. The app still answers from retrieved context when this is false.

Run:

```bash
python -m streamlit run app.py
```

## How It Works

1. Extract text from the uploaded source.
2. Split text into overlapping chunks.
3. Generate embeddings with `sentence-transformers/all-mpnet-base-v2`.
4. Store chunks and metadata in FAISS.
5. Retrieve the most relevant chunks for a question.
6. Generate an answer using Hugging Face, with a local fallback if the API fails.

## Resume Description

Built a document-based RAG Q&A app using Streamlit, LangChain, Hugging Face embeddings, and FAISS, supporting PDF, DOCX, TXT, and URL inputs with semantic retrieval, source citations, and API-failure fallback handling.
