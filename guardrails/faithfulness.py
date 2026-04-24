from groq import Groq
from config import GROQ_API_KEY, CHECK_MODEL, MAX_CONTEXT_CHARS

_client = Groq(api_key=GROQ_API_KEY)


def score_faithfulness(answer: str, context: str) -> float:
    context = context[:MAX_CONTEXT_CHARS]
    if not context.strip():
        return 0.5

    # guardrails/faithfulness.py — replace the prompt inside score_faithfulness()

    prompt = f"""You are a strict fact-checking assistant.

Your job: check if EVERY single claim in the ANSWER is explicitly stated in the CONTEXT.
You must NOT use any outside knowledge.
If even ONE claim in the answer cannot be found word-for-word or by direct implication in the context, reply UNSUPPORTED.

CONTEXT:
{context}

ANSWER:
{answer}

Think step by step:
1. List each factual claim in the answer.
2. For each claim, check if the context explicitly supports it.
3. If ALL claims are supported → reply SUPPORTED
4. If ANY claim is not in the context → reply UNSUPPORTED

Final reply — ONLY one word: SUPPORTED or UNSUPPORTED
"""

    try:
        r = _client.chat.completions.create(
            model=CHECK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0,
            top_p=0.1
        )
        reply = r.choices[0].message.content.strip().upper()
        if "UNSUPPORTED" in reply: # Check negative first to be safe
            return 0.0
        return 1.0 if "SUPPORTED" in reply else 0.5
    except Exception as e:
        print(f"Faithfulness scorer error: {e}")
        return 0.5
