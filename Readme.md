# Multimodal RAG with Persistent Memory

A **production-grade Multimodal Retrieval-Augmented Generation (RAG)** system built using **LangChain, LangGraph, FAISS, OCR, and Ollama**, with a **Streamlit UI** and **thread-based persistent memory**.

---

##  Features

* **PDF text extraction** (PyMuPDF)
* **Image & table OCR** (Tesseract)
* **Semantic search** with FAISS
* **Persistent chat memory** using LangGraph + SQLite
* **Thread-based conversations**
* **Local LLM inference** via Ollama (CPU-safe)
* **ChatGPT-like UI** using Streamlit
* **No hallucinations** (strict context grounding)

---

## Architecture

```
User (Streamlit UI)
        |
        v
LangGraph (Thread Memory + SQLite)
        |
        v
Multimodal Retriever (FAISS)
        |
        v
LLM (Ollama - llama2:7b)
```

### Memory Priority

```
Chat Memory  >  Document Context  >  Refusal
```

---

## Project Structure

```
MultiModel-RAG/
│
├── app.py                  # Streamlit UI
├── backend.py              # Multimodal RAG + LangGraph
├── chatbot.db              # SQLite memory (auto-created)
├── requirements.txt
├── .gitignore
│
├── data/
│   └── pdfs/               # Place your PDFs here
│
└── faiss_db/               # Vector store (auto-created)
```

---

## Setup Instructions (Windows)

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install System Dependencies

* **Tesseract OCR**
  [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

* **Poppler (for PDF images)**
  [https://github.com/oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows)

Update paths in `backend.py`:

```python
TESSERACT_DIR = r"C:\Program Files\Tesseract-OCR"
POPPLER_BIN = r"C:\poppler\Library\bin"
```

---

### Start Ollama

```bash
ollama serve
ollama pull llama2:7b
```

---

### Run Application

```bash
streamlit run app.py
```

---

## Example Usage

```
User: my name is ritesh
Assistant: Hello Ritesh! 

User: what is my name?
Assistant: Your name is Ritesh.
```
---

## Key Design Notes

* The **LLM has no internal memory**
* Memory is implemented via **LangGraph + SQLite**
* Multimodal understanding is achieved via **OCR → text embeddings**
* Designed to be **CPU-safe and local-first**

---

## Deployment Note

This project uses:

* Ollama (local LLM)
* Tesseract / Poppler (system binaries)

**Not directly deployable on Streamlit Cloud**

### Recommended Deployment:

* Backend on **local / EC2 / VPS**
* Frontend on **Streamlit Cloud**
* Communicate via REST API

---

## Tech Stack

* Python
* Streamlit
* LangChain
* LangGraph
* FAISS
* SentenceTransformers
* Tesseract OCR
* Ollama

---

## License

MIT License
