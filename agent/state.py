from typing import Annotated, List, TypedDict, Optional
from operator import add

class AgentState(TypedDict):
    # The user's original question was: "What is the state of the agent in the context of a conversation or interaction?"
    input: str
    # Chat history for context
    chat_history: Annotated[List[str], add]
    # Chunks retrieved from the vector store (text + visual descriptions)
    context: Annotated[List[str], add]
    #The current generated answer
    predictions: str
    #Boolean flag from the guardrail node
    is_hallucination: bool
    confidence_score: float
    #List of citation used.
    sources: List[str]