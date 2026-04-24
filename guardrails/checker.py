import re
from groq import Groq
from guardrails.nli_scorer   import score_nli
from guardrails.faithfulness import score_faithfulness
from config import (GROQ_API_KEY, CHECK_MODEL, SELFCHECK_SAMPLES,
                    CONFIDENCE_THRESHOLD, MAX_CONTEXT_CHARS)

_client = Groq(api_key=GROQ_API_KEY)

def _clean_reasoning_model_output(content: str) -> str:
    """
    DeepSeek-R1 and other reasoning models include <think> blocks.
    This removes the thinking and returns only the final answer.
    """
    # Remove everything inside <think>...</think> tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()

def _sample_answer(question: str, context: str) -> str:
    prompt = f"""You are a research assistant. Answer the question using ONLY 
the context below. Be concise — 2 sentences maximum.
If the context partially supports an answer, give that partial answer.
Only say "I don't know" if the context contains ZERO relevant information.

Context:
{context}

Question: {question}
Answer:"""

    try:
        r = _client.chat.completions.create(
            model=CHECK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, # Increased slightly for reasoning models
            temperature=0.7
        )
        return _clean_reasoning_model_output(r.choices[0].message.content)
    except Exception as e:
        print(f"[Checker] _sample_answer failed: {e}")
        return "I don't know."

def _score_consistency(main_answer: str, samples: list) -> float:
    if not samples:
        return 0.5

    total = 0.0
    for sample in samples:
        sample_lower = sample.lower()

        if any(phrase in sample_lower for phrase in
               ["i don't know", "not found", "no information",
                "cannot determine", "context does not", "not mentioned"]):
            total += 0.5
            continue

        prompt = f"""Compare these two answers to the same question.
Does Answer B CONTRADICT any specific fact stated in Answer A?

Answer A: {main_answer[:500]}
Answer B: {sample}

Reply with ONLY one word at the very end of your response:
CONTRADICTS — if B states a fact that directly conflicts with A
CONFIRMS    — if B agrees with or supports A's key claims  
NEUTRAL     — if B neither confirms nor contradicts"""

        try:
            r = _client.chat.completions.create(
                model=CHECK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300, # Increased to allow space for <think> block
                temperature=0
            )
            # CLEAN THE OUTPUT
            verdict = _clean_reasoning_model_output(r.choices[0].message.content.strip().upper())
            # USE "IN" INSTEAD OF "STARTSWITH"
            if "CONTRADICT" in verdict:
                total += 0.0
            elif "CONFIRM" in verdict or "AGREE" in verdict:
                total += 1.0
            else:
                total += 0.5 

        except Exception as e:
            print(f"[Checker] _score_consistency call failed: {e}")
            total += 0.5

    return round(total / len(samples), 3)

def check(question: str, answer: str, context: str) -> dict:
    # Use the full character limit from config
    context = context[:MAX_CONTEXT_CHARS]

    if not context.strip():
        return {
            "answer":     answer,
            "confidence": 0.3,
            "flagged":     False,
            "warning":    "No document context retrieved."
        }

    # Run checks
    try:
        samples = [_sample_answer(question, context) for _ in range(SELFCHECK_SAMPLES)]
        consistency = _score_consistency(answer, samples)
    except Exception as e:
        consistency = 0.5

    try:
        nli = score_nli(answer, context)
    except Exception as e:
        nli = 0.5

    try:
        faith = score_faithfulness(answer, context)
    except Exception as e:
        faith = 0.5

    # Master's Level Weighting: NLI is the gold standard for Table data
    confidence = (consistency * 0.2) + (nli * 0.5) + (faith * 0.3)
    flagged = confidence < CONFIDENCE_THRESHOLD

    return {
        "answer":     answer,
        "confidence": round(confidence, 2),
        "flagged":     flagged,
        "warning": (
            f"Low confidence ({confidence:.0%}). "
            f"Answer may not be fully supported. "
            f"[consistency={consistency:.2f}, nli={nli:.2f}, faithfulness={faith:.2f}]"
        ) if flagged else ""
    }