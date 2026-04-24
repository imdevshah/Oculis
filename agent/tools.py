from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from retrieval.retriever import retrieve  
from config import MAX_CHUNK_CHARS

@tool
def rag_search(query: str) -> str:
    """Search the local PDF documents for information. Use this for ANY question about the files."""
    chunks = retrieve(query)
    # ADD DEBUG PRINTS RIGHT HERE
    print("\n=== RETRIEVED CHUNKS ===")
    for c in chunks:
        print(f"[Page {c['page']}] {c['text'][:150]}")

    if not chunks:
        return "No results found."

    context_parts = []
    for chunk in chunks:
        text = chunk["text"][:MAX_CHUNK_CHARS]
        page = chunk["page"]
        context_parts.append(f"[Page {page}] {text}")

    return "\n\n".join(context_parts)

@tool
def web_search(query: str) -> str:
    """Search the web for current events or recent news. Use only if rag_search fails."""
    try:
        results = DuckDuckGoSearchRun().run(query)
        return results
    except Exception as e:
        return f"Web search failed: {e}"
    
@tool
def calculate(expression: str) -> str:
    """Evaluate math expressions. Input must be a string like '2 + 2' or '0.15 * 4.2'."""    
    try:
        # allowed = {"abs": abs, "round": round}
        expression = expression.strip().strip("'\"")
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Calculation failed: {e}"
    