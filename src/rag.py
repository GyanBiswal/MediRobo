"""
Core RAG chain for the Medical RAG Chatbot.

Wires together:
    retriever (src.vectorstore) -> prompt (src.prompts) -> Groq LLM -> answer

This is the "brain" of the app. Streamlit (Phase 9) will just call
`answer_question()` from this file.
"""

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE
from src.vectorstore import load_vectorstore, get_retriever
from src.prompts import get_rag_prompt, format_docs


def get_llm():
    """
    Create the Groq chat model client.

    Returns:
        ChatGroq: a LangChain-compatible LLM object.
    """
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def build_rag_chain():
    """
    Assemble the retriever, prompt, and LLM into ready-to-use pieces.

    We don't use LangChain's piped `|` LCEL chain syntax for the whole
    thing here, because we need access to the raw retrieved Documents
    (for source citations) alongside the final answer — piping everything
    together would hide that intermediate result. Instead we return the
    pieces and combine them explicitly in `answer_question()`, which is
    more readable for a beginner-friendly project anyway.

    Returns:
        dict with 'retriever', 'prompt', and 'llm' keys.
    """
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)
    prompt = get_rag_prompt()
    llm = get_llm()

    return {"retriever": retriever, "prompt": prompt, "llm": llm}


def answer_question(question: str, chain: dict = None):
    """
    Answer a single question using the full RAG pipeline.

    Args:
        question: the user's question (str)
        chain: optional pre-built chain dict from build_rag_chain().
            If not provided, builds one on the fly (slower — mainly
            useful for quick one-off testing).

    Returns:
        dict with:
            'answer': str, the LLM's generated answer
            'source_documents': list[Document], the chunks used as context
    """
    if chain is None:
        chain = build_rag_chain()

    retriever = chain["retriever"]
    prompt = chain["prompt"]
    llm = chain["llm"]

    # 1. Retrieve relevant chunks
    docs = retriever.invoke(question)

    # 2. Format them into a context string
    context = format_docs(docs)

    # 3. Fill the prompt template
    filled_prompt = prompt.invoke({"context": context, "question": question})

    # 4. Call the Groq LLM
    response = llm.invoke(filled_prompt)

    return {
        "answer": response.content,
        "source_documents": docs,
    }


if __name__ == "__main__":
    print("Building RAG chain (loading vector store + LLM)...")
    chain = build_rag_chain()

    test_questions = [
        "What are the symptoms of the flu?",
        "How is type 2 diabetes managed?",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        result = answer_question(question, chain)

        print(f"\nA: {result['answer']}")

        print("\nSources:")
        seen = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "unknown").split("/")[-1]
            page = doc.metadata.get("page", "unknown")
            key = (source, page)
            if key not in seen:
                print(f"  - {source}, page {page}")
                seen.add(key)