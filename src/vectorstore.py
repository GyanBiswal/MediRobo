"""
FAISS vector store setup for the Medical RAG Chatbot.

Pipeline so far:
    PDFs -> chunks (src.ingest) -> embeddings (src.embeddings)
    -> FAISS index (this file) -> retriever

Building the index is a one-time step (rerun only when source PDFs change).
Querying it is fast and happens on every user question.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.config import VECTORSTORE_DIR, RETRIEVER_TOP_K
from src.ingest import load_and_split
from src.embeddings import get_embedding_model


def build_vectorstore():
    """
    Build a FAISS index from scratch: load PDFs, split them, embed them,
    and save the resulting index to disk.

    Returns:
        FAISS: the in-memory vector store (also saved to VECTORSTORE_DIR).
    """
    chunks = load_and_split()
    embeddings = get_embedding_model()

    print("Building FAISS index (embedding all chunks)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"FAISS index saved to {VECTORSTORE_DIR}")

    return vectorstore


def load_vectorstore():
    """
    Load a previously-built FAISS index from disk, instead of rebuilding it.
    Much faster than build_vectorstore() — use this at app startup.

    Returns:
        FAISS: the loaded vector store.
    """
    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # safe here: we created this file ourselves
    )
    return vectorstore


def get_retriever(vectorstore, k: int = RETRIEVER_TOP_K):
    """
    Wrap a FAISS vector store as a LangChain retriever.

    Args:
        vectorstore: a FAISS instance (from build_vectorstore or load_vectorstore)
        k: how many chunks to return per query

    Returns:
        A LangChain retriever object with .invoke(question) -> list[Document]
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    index_path = Path(VECTORSTORE_DIR) / "index.faiss"

    if index_path.exists():
        print("Existing FAISS index found — loading it.")
        vectorstore = load_vectorstore()
    else:
        print("No existing index found — building a new one.")
        vectorstore = build_vectorstore()

    retriever = get_retriever(vectorstore)

    # Quick sanity check: ask a question and see what comes back
    question = "What are the symptoms of the flu?"
    results = retriever.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} (source: {doc.metadata.get('source')}, page: {doc.metadata.get('page')}) ---")
        print(doc.page_content[:200], "...\n")