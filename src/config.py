"""
Central configuration for the Medical RAG Chatbot.
All tunable settings live here so the rest of the codebase
doesn't have magic strings/numbers scattered around.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env into the environment
load_dotenv()

# --- Secrets ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Did you create a .env file from .env.example "
        "and add your real Groq API key?"
    )

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data" / "medical_documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# --- Groq LLM settings ---
GROQ_MODEL = "openai/gpt-oss-20b"  # OpenAI's open-weight model, served on Groq
LLM_TEMPERATURE = 0.1  # low = more factual/deterministic, good for medical grounding

# --- Embedding model settings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free, runs locally

# --- Text splitting settings ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retrieval settings ---
RETRIEVER_TOP_K = 4  # how many chunks to retrieve per question