import chromadb
#Here the tool is defined as a wrapper around the retrieve() function from retriever.py, @toolspecifically converts a plain python function into something the langgraph agent can discover, call, and read results from. The agent will automatically convert the input and output to and from JSON when calling the tool, so you can use regular Python data structures like lists and dictionaries in your function. The agent will also handle any exceptions that occur within the tool and return an error message instead of crashing the entire agent.
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from retrieval.retriever import retrieve  

@tool
def rag_search(query: str) -> str:
    """Search documents for information."""
    # Keep your comments here if you want, but keep the triple-quote string SHORT.
    chunks = retrieve(query)
    if not chunks:
        return "No results found."

    output = "Context:\n"
    for i, chunk in enumerate(chunks):
        # Use a standard hyphen '-' NOT a long dash '—'
        output += f"Source {i+1} (Page {chunk['page']}): {chunk['text']}\n\n"
    return output

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
        allowed = {"abs": abs, "round": round}
        result = eval(expression, {"__builtins__": None}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation failed: {e}"
    

#Now test block:

if __name__ == "__main__":
       # Test 1: rag_search
    print("Testing rag_search...")
    result = rag_search.invoke("how much revenue did the company make?")
    print(result)
    print()

    #Test 2: rag_search with no results
    print("Testing rag_search with no results...")
    result = rag_search.invoke("what is the meaning of life?")
    print(result)
    print()

    # Test 3: calculate
    print("Testing calculate...")
    result = calculate.invoke("2 + 2 * 3")
    print(result)
    print()

    #Test 4: dangerous input for calculate
    print("Testing calculate with dangerous input...")
    result = calculate.invoke("__import__('os').system('echo hello')")
    print("Safety Result:", result)
    print()

    #test 5: web_search
    print("Testing web_search...")
    result = web_search.invoke("What is the capital of France?")
    print(result[:300])  # Print only the first 300 characters of the web search results for brevity
    print()

    print("All tests completed.")