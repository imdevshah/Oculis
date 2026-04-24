from groq import Groq
from config import GROQ_API_KEY, CHECK_MODEL, MAX_CONTEXT_CHARS

_client = Groq(api_key=GROQ_API_KEY)

def score_nli(answer: str, context: str) -> float:
    context = context[:MAX_CONTEXT_CHARS]
    if not context.strip():
        # No context retrieved — cannot check, return neutral score
        return 0.5

    prompt = f"""You are a fact-checking assistant.

Given the CONTEXT below, does the ANSWER contradict any facts in the context?

CONTEXT:
{context}

ANSWER:
{answer}

Reply with ONLY one word: CONTRADICTION or CONSISTENT"""

    try:
        r = _client.chat.completions.create(
            model=CHECK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0
        )
        reply = r.choices[0].message.content.strip().upper()
        # 1.0 = consistent (good), 0.0 = contradiction (bad)
        if "CONTRADICTION" in reply:
            return 0.0
        if "CONSISTENT" in reply:
            return 1.0
        return 0.5
    except Exception as e:
        print(f"NLI scorer error: {e}")
        return 0.5  # neutral on failure
