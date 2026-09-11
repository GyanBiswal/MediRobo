"""
Embedding model setup for the Medical RAG Chatbot.

Uses a small, free, local Hugging Face sentence-transformer model
to convert text chunks into numeric vectors for similarity search.
No API key needed — this model runs on your own machine.
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBEDDING_MODEL_NAME


def get_embedding_model():
    """
    Load the Hugging Face embedding model.

    Returns:
        HuggingFaceEmbeddings: a LangChain-compatible embeddings object
        with .embed_query() and .embed_documents() methods, used later
        by FAISS to build and search the vector store.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},       # M1 Macs: CPU is simplest and reliable
        encode_kwargs={"normalize_embeddings": True},  # normalized vectors -> cosine similarity works cleanly
    )
    return embeddings


if __name__ == "__main__":
    embeddings = get_embedding_model()

    sample_text = "Type 2 diabetes happens when your body does not use insulin well."
    vector = embeddings.embed_query(sample_text)

    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Vector length (dimensions): {len(vector)}")
    print(f"First 5 values: {vector[:5]}")