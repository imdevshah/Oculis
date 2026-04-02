import sys
import os

# Get the absolute path to the current directory (Oculis root)
module_path = os.path.abspath(os.getcwd())

# Add it to the Python path if it's not already there
if module_path not in sys.path:
    sys.path.append(module_path)

print(f"Added to path: {module_path}")

from langgraph.graph import StateGraph, END
from agent.state import AgentState
print("Import Successful! Oculis logic is now connected.")

# --- Nodes ---

def retrieve_node(state: AgentState):
    print("--- RETRIEVING CONTEXT (VLM/VECTOR) ---")
    # For now, we simulate a retrieval
    return {"context": ["Simulated context from Oculis VLM engine."]}

def generate_node(state: AgentState):
    print("--- GENERATING ANSWER ---")
    # Logic to call LLM would go here
    return {"prediction": "The system is functioning. Trust Oculis."}

def guardrail_node(state: AgentState):
    print("--- CHECKING FOR HALLUCINATIONS ---")
    # Simulate a check
    return {"is_hallucination": False}

# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add our nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("guardrail", guardrail_node)

# Define the edges (The flow)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "guardrail")
workflow.add_edge("guardrail", END)

# Compile the graph
app = workflow.compile()