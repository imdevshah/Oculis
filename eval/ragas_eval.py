import json
import time
from datetime import datetime
from groq import Groq
from retrieval.retriever import retrieve
from agent.agent import answer
from guardrails.faithfulness import score_faithfulness
from config import GROQ_API_KEY, CHECK_MODEL

_client = Groq(api_key=GROQ_API_KEY)


# ── Test dataset ─────────────────────────────────────────────────────────────
# Each entry has:
#   question       : what the user asks
#   ground_truth   : the correct answer (you write this manually)
#   relevant_chunk : a key phrase that MUST appear in retrieved context
#                    if the retriever is working correctly
#
# Why write these manually?
# There is no automated way to know what the "correct" answer is —
# that requires human judgment. This is called a "golden dataset."
# In production you'd have 50-100 entries. For a dev project, 5-8
# is enough to get meaningful signal.


EVAL_DATASET = [
    {
        "question":      "How many encoder layers does the transformer use?",
        "ground_truth":  "The transformer uses 6 encoder layers.",
        "relevant_chunk": "6"
    },
    {
        "question":      "What BLEU score did the Transformer (big) achieve on WMT 2014 English-to-German?",
        "ground_truth":  "The Transformer (big) achieved a BLEU score of 28.4 on WMT 2014 English-to-German translation.",
        "relevant_chunk": "28.4"
    },
    {
        "question":      "The big model trained for 300,000 steps and each step took 1.0 second. How many hours is that?",
        "ground_truth":  "The big model took 83.33 hours to train.",
        "relevant_chunk": "83.33"
    },
    {
        "question":      "Who proposed scaled dot-product attention?",
        "ground_truth":  "Noam proposed scaled dot-product attention",
        "relevant_chunk": "Noam"
    },
    {
        "question":      "Which 6 encoder layers are there in this pdf?",
        "ground_truth":  "The encoder is composed of a stack of N = 6 identical layers. Each layer has twosub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [11] around each ofthe two sub-layers, followed by layer normalization [12].",
        "relevant_chunk": "6 identical layers"
    },
]


# ── Metric 1: Context Recall ─────────────────────────────────────────────────

def score_context_recall(relevant_chunk: str, retrieved_chunks: list) -> float:
    """
    Did the retriever find the chunk that contains the answer?

    Why check for a key phrase instead of exact match?
    The relevant chunk is a short distinctive phrase from the correct
    source chunk. If ANY retrieved chunk contains
    that phrase, the retriever found the right material.

    Returns:
        1.0 = correct chunk was retrieved
        0.0 = retriever missed it entirely
    """
    combined = " ".join(c["text"] for c in retrieved_chunks).lower()
    return 1.0 if relevant_chunk.lower() in combined else 0.0


# ── Metric 2: Answer Relevancy ───────────────────────────────────────────────

def score_answer_relevancy(question: str, answer_text: str) -> float:
    """
    Does the answer actually address what was asked?

    How it works:
    We ask the model to generate what question this answer would be
    responding to. If the generated question is semantically similar
    to the original question, the answer is relevant.

    This is the standard RAGAS approach — instead of directly scoring
    relevancy (which is subjective), we reverse-engineer the question
    from the answer. A relevant answer should produce a question close
    to the original.

    Returns:
        float 0.0 to 1.0
    """
    # Step 1: generate the implied question from the answer
    prompt_gen = f"""Given this answer, what question was it responding to?
Write only the question, nothing else.

Answer: {answer_text}
Question:"""

    r = _client.chat.completions.create(
        model=CHECK_MODEL,
        messages=[{"role": "user", "content": prompt_gen}],
        max_tokens=60,
        temperature=0
    )
    implied_question = r.choices[0].message.content.strip()

    # Step 2: ask the model if the implied question matches the original
    prompt_match = f"""Do these two questions ask for the same information?
Reply with only a number between 0.0 and 1.0.
1.0 = identical meaning, 0.0 = completely different topic.

Question A: {question}
Question B: {implied_question}
Score:"""

    r = _client.chat.completions.create(
        model=CHECK_MODEL,
        messages=[{"role": "user", "content": prompt_match}],
        max_tokens=5,
        temperature=0
    )

    try:
        score = float(r.choices[0].message.content.strip())
        return round(min(max(score, 0.0), 1.0), 2)
    except ValueError:
        return 0.5  # neutral if parsing fails


# ── Metric 3: Faithfulness ───────────────────────────────────────────────────
# Already implemented in guardrails/faithfulness.py — we reuse it directly.
# No duplication: eval and guardrails share the same scorer.


# ── Per-question evaluation ──────────────────────────────────────────────────

def evaluate_one(entry: dict) -> dict:
    """
    Runs all three metrics for a single question.

    Returns a result dict with every score and the raw
    agent answer — useful for debugging individual failures.
    """
    question       = entry["question"]
    relevant_chunk = entry["relevant_chunk"]

    print(f"\n  Q: {question}")

    # Step 1: retrieve context (what the retriever found)
    chunks  = retrieve(question)
    context = "\n".join(c["text"] for c in chunks)

    # Step 2: get the agent's answer + confidence from guardrails
    result      = answer(question)
    answer_text = result["answer"]
    confidence  = result["confidence"]

    print(f"  A: {answer_text[:80]}{'...' if len(answer_text) > 80 else ''}")

    # Step 3: score all three metrics
    recall    = score_context_recall(relevant_chunk, chunks)
    relevancy = score_answer_relevancy(question, answer_text)
    faith     = score_faithfulness(answer_text, context)

    print(f"  recall={recall:.2f}  relevancy={relevancy:.2f}  "
          f"faithfulness={faith:.2f}  guardrail_conf={confidence:.2f}")

    return {
        "question":      question,
        "answer":        answer_text,
        "ground_truth":  entry["ground_truth"],
        "confidence":    confidence,
        "context_recall":     recall,
        "answer_relevancy":   relevancy,
        "faithfulness":       faith,
        # Overall per-question score: average of all three metrics
        "score": round((recall + relevancy + faith) / 3, 2)
    }


# ── Full eval run ────────────────────────────────────────────────────────────

def run_eval(dataset: list = None, save_report: bool = True) -> dict:
    """
    Runs evaluation over the full dataset and returns a summary report.

    Args:
        dataset     : list of eval entries (defaults to EVAL_DATASET)
        save_report : if True, saves a JSON report to eval/reports/

    Returns:
        {
            "avg_context_recall":   float,
            "avg_answer_relevancy": float,
            "avg_faithfulness":     float,
            "avg_score":            float,
            "results":              list of per-question dicts,
            "timestamp":            str
        }
    """
    dataset = dataset or EVAL_DATASET
    print(f"\n{'='*55}")
    print(f"  Oculis Eval — {len(dataset)} questions")
    print(f"{'='*55}")

    results = []
    for i, entry in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}]", end="")
        try:
            r = evaluate_one(entry)
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            results.append({
                "question": entry["question"],
                "error":    str(e),
                "context_recall": 0.0,
                "answer_relevancy": 0.0,
                "faithfulness": 0.0,
                "score": 0.0
            })

    # Aggregate scores
    def avg(key):
        vals = [r[key] for r in results if key in r]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    summary = {
        "avg_context_recall":   avg("context_recall"),
        "avg_answer_relevancy": avg("answer_relevancy"),
        "avg_faithfulness":     avg("faithfulness"),
        "avg_score":            avg("score"),
        "total_questions":      len(dataset),
        "timestamp":            datetime.now().isoformat(),
        "results":              results
    }

    # Print summary table
    print(f"\n\n{'='*55}")
    print(f"  RESULTS")
    print(f"{'='*55}")
    print(f"  Context Recall      {summary['avg_context_recall']:.0%}")
    print(f"  Answer Relevancy    {summary['avg_answer_relevancy']:.0%}")
    print(f"  Faithfulness        {summary['avg_faithfulness']:.0%}")
    print(f"  {'─'*40}")
    print(f"  Overall Score       {summary['avg_score']:.0%}")
    print(f"{'='*55}\n")

    # Flag weak areas automatically
    if summary["avg_context_recall"] < 0.7:
        print("  ⚠  Low context recall — improve chunking or TOP_K in config.py")
    if summary["avg_answer_relevancy"] < 0.7:
        print("  ⚠  Low answer relevancy — tighten the system prompt in agent.py")
    if summary["avg_faithfulness"] < 0.7:
        print("  ⚠  Low faithfulness — guardrails catching issues, check retrieval quality")
    if summary["avg_score"] >= 0.8:
        print("  Overall score above 80% — system is performing well")

    # Save JSON report
    if save_report:
        import os
        os.makedirs("eval/reports", exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"eval/reports/report_{ts}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Report saved → {path}\n")

    return summary


if __name__ == "__main__":
    summary = run_eval()
