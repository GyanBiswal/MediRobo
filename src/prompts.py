"""
Prompt template for the Medical RAG Chatbot.

This is where medical safety and grounding rules are enforced at the
prompt level: the LLM is instructed to answer only from retrieved
context, admit when it doesn't know, cite sources, and always include
a disclaimer that this is not a substitute for professional medical advice.
"""

from langchain_core.prompts import ChatPromptTemplate


SYSTEM_INSTRUCTIONS = """You are a medical information assistant. Your job is to answer \
health-related questions using ONLY the context provided below, which comes from \
trusted medical reference documents.

Critical rule: The context below is your ONLY source of medical knowledge for this \
answer. Even if you recognize the topic and know things about it from your training, \
you must IGNORE that outside knowledge and rely solely on the context provided. If the \
context does not clearly and directly answer the question, you must say so — do not \
fill gaps with anything not explicitly stated in the context below.

Rules you must follow:
1. Base your answer strictly on the provided context. Do not use outside knowledge \
or make up information that isn't in the context, even if you believe it to be true.
2. Before answering, check: does the context actually contain information that \
answers this specific question? If not, respond only with: "I don't have enough \
information in my knowledge base to answer that." Do not add a partial answer \
from outside knowledge alongside this statement.
3. Do not diagnose the user or recommend a specific personalized treatment. You may \
explain general medical information (e.g. what a condition is, common symptoms, \
general prevention) ONLY when that information is present in the context. Decisions \
about an individual's care belong to a qualified healthcare professional.
4. Keep your tone clear, calm, and factual — avoid alarming language.
5. Always end your answer with this exact disclaimer on its own line:
"This information is for educational purposes only and is not a substitute for \
professional medical advice, diagnosis, or treatment. Please consult a qualified \
healthcare provider for concerns about your health."

Context from medical documents:
{context}
"""

USER_TEMPLATE = "{question}"


def get_rag_prompt():
    """
    Build the chat prompt template used by the RAG chain.

    Returns:
        ChatPromptTemplate: a LangChain prompt with {context} and {question}
        placeholders, ready to be filled in and sent to the Groq LLM.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_INSTRUCTIONS),
        ("human", USER_TEMPLATE),
    ])
    return prompt


def format_docs(docs):
    """
    Turn a list of retrieved Documents into a single context string,
    with each chunk labeled by its source so the model (and later, we)
    can trace claims back to a specific document.

    Args:
        docs: list[Document] from the retriever

    Returns:
        str: formatted context block to insert into the prompt
    """
    formatted_chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        source_name = source.split("/")[-1]
        page = doc.metadata.get("page", "unknown")
        formatted_chunks.append(
            f"[Source: {source_name}, page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)


if __name__ == "__main__":
    from src.vectorstore import load_vectorstore, get_retriever
    from pathlib import Path
    from src.config import VECTORSTORE_DIR

    index_path = Path(VECTORSTORE_DIR) / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(
            "No FAISS index found. Run `python -m src.vectorstore` first "
            "(Phase 6) to build it."
        )

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore)

    question = "What are the symptoms of the flu?"
    docs = retriever.invoke(question)
    context = format_docs(docs)

    prompt = get_rag_prompt()
    filled_prompt = prompt.invoke({"context": context, "question": question})

    print("--- Formatted context ---")
    print(context[:500], "...\n")

    print("--- Final prompt messages sent to the LLM ---")
    for message in filled_prompt.to_messages():
        print(f"\n[{message.type}]")
        print(message.content[:800])