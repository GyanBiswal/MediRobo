"""
PDF ingestion for the Medical RAG Chatbot.
Loads all PDFs from the data directory using LangChain's PyPDFLoader
and returns a list of LangChain Document objects (one per page).
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

from src.config import DATA_DIR


def load_pdfs(data_dir: Path = DATA_DIR):
    """
    Load every PDF in `data_dir` using LangChain's PyPDFLoader.

    Returns:
        list[Document]: one LangChain Document per PDF page, each with
        `page_content` (the text) and `metadata` (includes 'source' filename
        and 'page' number) already attached by the loader.
    """
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in {data_dir}. "
            "Add some medical PDFs there before running ingestion."
        )

    all_documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded {len(documents)} pages from {pdf_path.name}")

    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents


if __name__ == "__main__":
    docs = load_pdfs()

    # Quick sanity check: show the first document's content and metadata
    if docs:
        print("\n--- Sample document ---")
        print("Metadata:", docs[0].metadata)
        print("Content preview:", docs[0].page_content[:300])