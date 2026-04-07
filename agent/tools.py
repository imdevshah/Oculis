import chromadb
#Here the tool is defined as a wrapper around the retrieve() function from retriever.py, @toolspecifically converts a plain python function into something the langgraph agent can discover, call, and read results from. The agent will automatically convert the input and output to and from JSON when calling the tool, so you can use regular Python data structures like lists and dictionaries in your function. The agent will also handle any exceptions that occur within the tool and return an error message instead of crashing the entire agent.
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from retrieval.retriever import retrieve  

@tool
def rag_search(query: str) -> str:
    """
    A wrapper around the retrieve() function from retriever.py, designed to be used as a tool within the agent.

    Args:
        query : the user's question in plain English
    Returns:
        list of dicts, each containing: 
        - text       : the actual chunk content
        - source     : which file it came from
        - page       : which page number
        - type       : "text" or "image_caption"
        - similarity : 0.0 to 1.0 (1.0 = identical, 0.0 = unrelated)
    """
    chunks = retrieve(query)
     # If ChromaDB found nothing similar, chunks will be an empty list []
    # We must return a STRING — not [], not None, not an empty string
    # This string tells the agent "nothing found, try something else
    # The agent reads this and decides to try web_search instead

    if not chunks:
        return [{"text": "No relevant information found.", "source": "", "page": "", "type": "", "similarity": 0.0}]    
     # We label it clearly so the agent knows this is document context not web results or its own memory
    output = "DOCUMENT CONTEXT:\n\n"

    # Loop through each chunk the retriever returned enumerate() gives us i (0,1,2...) and chunk (the dict) together
    for i, chunk in enumerate(chunks):
        # Build a header line for each chunk showing: - which result number this is (i+1 because humans count from 1) - which file it came from (source) - which page number - how similar it was to the query (0.0 to 1.0) - whether it was text or an image caption. This header helps the agent cite its sources properly
        output += (
            f"[Source {i+1} — {chunk['source']}, "
            f"page {chunk['page']} | "
            f"similarity: {chunk['similarity']} | "
            f"type: {chunk['type']}]\n"
            f"{chunk['text']}\n\n"
        )
    return output
@tool

def web_search(query: str) -> str:
     # Docstring tells agent: use this ONLY when documents don't have the answer "time-sensitive" is a hint that agent learns to use this for news, prices etc
    """Search the web for current information not available in the documents. Use this for recent events, live prices, or anything time-sensitive. Only use this if rag_search did not return useful results."""
    try:
        results = DuckDuckGoSearchRun().run(query)
        return results
    except Exception as e:
        # If the search fails we catch the error and return it as a string. Never let a tool crash silently — always tell the agent what went wrong so it can try a different approach instead of just giving up
        return f"Web search failed: {e}"
    
@tool
def calculate(expression: str) -> str:
        # Docstring tells agent: use this for any math calculations, from simple arithmetic to complex equations. The agent will pass the expression as a string, and you should return the result as a string. Be sure to handle errors gracefully and return an error message if the calculation fails.
    try:
        # eval() executes a string as Python code
        # For example: eval("2 + 2") returns the integer 4
        # eval("(4.2 - 3.5) / 3.5 * 100") returns 20.0

        # THE SECURITY TRICK: eval() takes a second argument called globals
        # Normally eval() has access to ALL Python built-ins: import, open, os, sys — everything dangerous
        # By passing {"__builtins__": {}} we replace builtins with empty dict
        # Now eval() can ONLY do pure math — no imports, no file access
        # Try: eval("__import__('os')", {"__builtins__": {}})
        # Result: NameError — __import__ doesn't exist anymore. Safe.
        result = eval(expression, {"__builtins__": {}})
        #eval() returns a Python number (int or float)
        # But the tool MUST return a string — so we wrap with str()
        # str(20.0) → "20.0"
        # str(4)    → "4"
        return str(result)
    except Exception as e:
        # If the expression is invalid math: "hello + world" or if someone tries injection: "__import__('os')" eval() raises an exception — we catch it here.
        # Return the error as a string so the agent knows what failed
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