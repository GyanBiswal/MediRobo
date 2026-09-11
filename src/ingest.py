"""
PDF ingestion and text splitting for the Medical RAG Chatbot.

Pipeline so far:
    PDFs -> LangChain PyPDFLoader -> page-level Documents -> text splitter -> chunks
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


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

    print(f"Total pages loaded: {len(all_documents)}")
    return all_documents


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Split page-level Documents into smaller overlapping chunks.

    Args:
        documents: list[Document] from load_pdfs()
        chunk_size: max characters per chunk
        chunk_overlap: characters shared between consecutive chunks,
            so a sentence/idea split across a chunk boundary isn't lost.

    Returns:
        list[Document]: smaller chunks, each still carrying the original
        'source' and 'page' metadata (LangChain propagates it automatically).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph breaks first, then sentences, then words
    )

    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} pages into {len(chunks)} chunks")
    return chunks


def load_and_split():
    """Convenience function: load PDFs and split them in one call."""
    documents = load_pdfs()
    chunks = split_documents(documents)
    return chunks


if __name__ == "__main__":
    chunks = load_and_split()

    print("\n--- Sample chunk ---")
    print("Metadata:", chunks[0].metadata)
    print("Content:", chunks[0].page_content)
    print(f"\nChunk length (characters): {len(chunks[0].page_content)}")