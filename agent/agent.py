from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.prompts import PromptTemplate
from agent.tools import rag_search, web_search, calculate
from config import ANSWER_MODEL, GROQ_API_KEY

_executor = None

def build_agent():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=ANSWER_MODEL,
        temperature=0
    )

    tools = [rag_search, web_search, calculate]

    template = """
You are Oculis, a research assistant.

You have access to the following tools:
{tools}

Previous conversation:
{chat_history}

Use the following format:

Question: {input}
Thought: think step-by-step
Action: one of [{tool_names}]
Action Input: input for the tool
Observation: result
... repeat if needed ...
Thought: I now know the final answer
Final Answer: final answer

Question: {input}
Thought:{agent_scratchpad}
"""

    prompt = PromptTemplate.from_template(template)

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )
    

def get_executor():
    global _executor
    if _executor is None:
        _executor = build_agent()
    return _executor

def answer(question: str, history: list = None) -> str:
    chat_history = history or []
    response = get_executor().invoke({"input": question, "chat_history": chat_history})
    return response["output"]

#Now testing block
if __name__ == "__main__":
    print("Welcome to the Research Assistant! Ask me anything about the documents or recent events.")

    print("Example questions:")
    print("-" * 40)
    response = answer("What was the Q3 revenue?")
    print("\nFinal answer:", response)
    print()

    # Test 2: Math question — should trigger calculate tool
    print("Test 2: Calculation question")
    print("-" * 40)
    response = answer("What is 18 percent of 4.2 million?")
    print("\nFinal answer:", response)
    print()

    # Test 3:   Multi-turn — second question references first answer
    # history shows how context carries forward across turns
    print("Test 3: Follow-up question with history")
    print("-" * 40)
    from langchain_core.messages import HumanMessage, AIMessage
    history = [
        HumanMessage(content="What was Q3 revenue?"),
        AIMessage(content="Q3 revenue was $4.2 million.")
    ]
    response = answer("How does that compare to Q2?", history)
    print("\nFinal answer:", response)
    print()

    print("=== Done ===")

