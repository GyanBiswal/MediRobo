# 🩺 Medical Information RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers general health
questions grounded in trusted medical reference documents. Built as a portfolio project
demonstrating a complete, understandable RAG pipeline — no black-box abstractions.

> **⚠️ Disclaimer:** This is an educational tool only. It is **not** a diagnostic system
> and does **not** replace professional medical advice. Always consult a qualified
> healthcare provider for concerns about your health.

## Live Demo

[Add your Streamlit Cloud URL here once deployed]

## Architecture

    Medical PDFs
        ↓
    LangChain PyPDFLoader
        ↓
    LangChain RecursiveCharacterTextSplitter
        ↓
    Hugging Face Sentence-Transformer Embeddings (local, free)
        ↓
    FAISS Vector Store
        ↓
    LangChain Retriever (top-k similarity search)
        ↓
    LangChain Prompt (medical grounding + safety rules)
        ↓
    Groq LLM (openai/gpt-oss-20b)
        ↓
    Answer + Source Citations
        ↓
    Streamlit Chat UI

## Key Features

- **Grounded answers**: responses are generated only from retrieved document context, not the LLM's general training knowledge.
- **Source citations**: every answer links back to the specific PDF and page it came from.
- **Safety-first prompting**: refuses to answer when the knowledge base doesn't cover a topic, avoids diagnosis or personalized treatment advice, and always includes a professional-consultation disclaimer.
- **Simple, transparent pipeline**: every RAG step (loading, splitting, embedding, retrieval, prompting, generation) is a readable, standalone function — no hidden agent frameworks.
- **Free to run**: local embeddings (no API cost), Groq's free-tier LLM, free vector store (FAISS runs in-process), free deployment (Streamlit Community Cloud).
- **Basic evaluation suite**: a repeatable script tests grounding, refusal behavior, and disclaimer presence across a fixed question set.

## Tech Stack

| Component | Technology |
|---|---|
| RAG framework | LangChain |
| LLM | Groq (`openai/gpt-oss-20b`) |
| Embeddings | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` (local) |
| Vector store | FAISS |
| UI | Streamlit |
| PDF parsing | `pypdf` via LangChain's `PyPDFLoader` |

## Project Structure

    medical-rag-chatbot/
    ├── app.py                    # Streamlit UI entry point
    ├── requirements.txt
    ├── src/
    │   ├── config.py              # Centralized settings (models, chunk sizes, paths)
    │   ├── ingest.py               # PDF loading + text splitting
    │   ├── embeddings.py           # Hugging Face embedding model
    │   ├── vectorstore.py          # FAISS index build/load + retriever
    │   ├── prompts.py               # RAG prompt with medical safety rules
    │   └── rag.py                   # Retriever + prompt + Groq chain
    ├── evaluation/
    │   ├── questions.json           # Test question set
    │   └── evaluate.py               # Automated grounding/safety checks
    ├── data/medical_documents/     # Source medical PDFs
    └── vectorstore/                 # Generated FAISS index (gitignored)

## Running Locally

### 1. Clone and set up environment

    git clone https://github.com/\<your-username\>/medical-rag-chatbot.git
    cd medical-rag-chatbot
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### 2. Configure your Groq API key

    cp .env.example .env
    # Edit .env and add your free key from https://console.groq.com

### 3. Add medical documents

Place PDF files in `data/medical_documents/`. Sources used in this project: [MedlinePlus](https://medlineplus.gov), [CDC fact sheets](https://www.cdc.gov), [WHO fact sheets](https://www.who.int/news-room/fact-sheets).

### 4. Build the vector index

    python -m src.vectorstore

### 5. Run the app

    streamlit run app.py

### 6. (Optional) Run the evaluation suite

    python -m evaluation.evaluate

## Medical Safety Design

This project treats safety as a first-class design constraint, not an afterthought:

- Answers are grounded strictly in retrieved context via explicit prompt instructions.
- The model is instructed to ignore its own training knowledge and say "I don't have enough information" when the retrieved context doesn't cover the question.
- The system avoids personalized diagnosis or treatment recommendations, redirecting instead to qualified healthcare professionals.
- Every response includes a clear educational-use disclaimer.
- An automated evaluation script checks these behaviors are consistently upheld.

## Future Improvements

- Add a similarity-score threshold to skip generation entirely on very weak retrieval matches.
- Expand the evaluation set and add basic hallucination-rate tracking over time.
- Support additional document formats (HTML fact sheets, plain text).

## License

This project is for educational and portfolio purposes.
