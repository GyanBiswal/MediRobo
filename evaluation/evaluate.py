"""
Simple evaluation script for the Medical RAG Chatbot.

Not a formal eval framework — a lightweight, repeatable sanity check.
Runs a fixed set of test questions through the RAG pipeline and checks
for basic grounding/refusal signals, so you can rerun this after any
change (new PDFs, prompt edits, model swap) and catch regressions.
"""

import json
from pathlib import Path

from src.rag import build_rag_chain, answer_question


QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"

# Phrases that indicate the model correctly said "I don't know" /
# refused to answer from outside knowledge. Adjust if you change the
# exact wording in src/prompts.py.
REFUSAL_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "not covered in",
    "cannot find this information",
]

# Phrases indicating an appropriate redirect to a professional, used
# for the personalized_advice category.
REDIRECT_PHRASES = [
    "healthcare provider",
    "healthcare professional",
    "medical professional",
    "seek medical attention",
    "consult a doctor",
]

DISCLAIMER_SNIPPET = "educational purposes only"


def load_questions():
    with open(QUESTIONS_PATH, "r") as f:
        return json.load(f)


def check_in_scope(answer: str, expected_keywords: list) -> bool:
    """Pass if at least one expected keyword appears in the answer (case-insensitive)."""
    answer_lower = answer.lower()
    if not expected_keywords:
        return True  # nothing specific to check
    return any(keyword.lower() in answer_lower for keyword in expected_keywords)


def check_out_of_scope(answer: str) -> bool:
    """Pass if the answer contains a refusal/'don't know' phrase."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


def check_personalized_advice(answer: str) -> bool:
    """Pass if the answer redirects to a professional rather than giving direct triage."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in REDIRECT_PHRASES)


def check_disclaimer(answer: str) -> bool:
    return DISCLAIMER_SNIPPET in answer.lower()


def run_evaluation():
    questions = load_questions()
    print(f"Loaded {len(questions)} evaluation questions.\n")

    print("Building RAG chain...")
    chain = build_rag_chain()

    results = []

    for item in questions:
        question = item["question"]
        q_type = item["type"]

        result = answer_question(question, chain)
        answer = result["answer"]

        if q_type == "in_scope":
            passed = check_in_scope(answer, item.get("expected_keywords", []))
        elif q_type == "out_of_scope":
            passed = check_out_of_scope(answer)
        elif q_type == "personalized_advice":
            passed = check_personalized_advice(answer)
        else:
            passed = None  # unknown type, not scored

        has_disclaimer = check_disclaimer(answer)

        results.append({
            "id": item["id"],
            "question": question,
            "type": q_type,
            "passed": passed,
            "has_disclaimer": has_disclaimer,
            "answer": answer,
        })

    print_report(results)
    return results


def print_report(results):
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)

    scored = [r for r in results if r["passed"] is not None]
    passed_count = sum(1 for r in scored if r["passed"])

    for r in results:
        status = "✅ PASS" if r["passed"] else ("❌ FAIL" if r["passed"] is False else "⚠️  UNSCORED")
        disclaimer_status = "✅" if r["has_disclaimer"] else "❌"
        print(f"\n[{r['id']}] {status} | Disclaimer: {disclaimer_status} | Type: {r['type']}")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}")

    print("\n" + "-" * 70)
    print(f"Score: {passed_count}/{len(scored)} scored questions passed")
    disclaimer_count = sum(1 for r in results if r["has_disclaimer"])
    print(f"Disclaimer present: {disclaimer_count}/{len(results)} answers")
    print("-" * 70)


if __name__ == "__main__":
    run_evaluation()