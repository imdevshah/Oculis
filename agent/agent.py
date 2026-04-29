from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from agent.tools import rag_search, web_search, calculate
from config import ANSWER_MODEL, GROQ_API_KEY

_executor = None


def build_agent():
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=ANSWER_MODEL,
        temperature=0,
        streaming=True,
        max_retries=3,
    )

    tools = [rag_search, web_search, calculate]

    template = """You are Oculis, a high-precision research agent. Your goal is to provide evidence-based answers using the provided tools.

You have access to the following tools:
{tools}

Previous conversation:
{chat_history}

OPERATIONAL PROTOCOLS:
1. SEARCH STRATEGY: Always start with 'rag_search'. Use 'web_search' ONLY if the information is missing from the documents or requires real-time data.
2. STRICT GROUNDING:
- You MUST answer ONLY using the Observation text
- DO NOT use prior knowledge
- DO NOT guess or infer beyond the text
- If partial information exists, answer using it.
Do NOT default to "I don't know" if useful context is present.

3. ANSWER SCOPE:
- Answer ONLY what was asked — do not add unrequested information.
- One question = one focused answer. Do not volunteer extra facts.

4. CITATIONS:
- Every sentence MUST include [Source X, Page Y]
- Do NOT invent page numbers

5. FAILURE MODE:
- If Observation is unclear or incomplete → say exactly:
  "I could not find this in the provided documents."

You MUST use EXACTLY this format. Never deviate from it:

Question: {input}
Thought: [your reasoning]
Action: [MUST be one of: {tool_names}]
Action Input: [MUST always follow Action — never omit this line]
Observation: [tool result — do not write this yourself]
Thought: [reasoning after seeing the result]
Action: [next tool if needed]
Action Input: [input for next tool]
Observation: [tool result]
Thought: I now have enough information to answer.
Final Answer: [your answer with page citations]

CRITICAL RULES FOR FORMAT:
- Action and Action Input MUST always appear together as a pair.
- Never write "Action: None" — if no tool is needed go straight to Final Answer.
- Never skip Action Input after writing Action.
- Never write Observation yourself — it is filled by the tool.

Begin.

Question: {input}
Thought:{agent_scratchpad}"""

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
        max_iterations=10,
        handle_parsing_errors=True,
        max_execution_time=90,
        return_intermediate_steps=True,
    )


def get_executor():
    global _executor
    if _executor is None:
        _executor = build_agent()
    return _executor


def _extract_context(intermediate_steps: list) -> str:
    """
    Pulls the raw text returned by rag_search out of the agent's
    intermediate steps so the guardrails checker has real document
    context to verify the answer against.

    intermediate_steps is a list of (AgentAction, tool_output) tuples.
    We only care about rag_search outputs — web_search and calculate
    results are not document context.
    """
    parts = []
    for action, observation in intermediate_steps:
        if hasattr(action, "tool") and action.tool == "rag_search":
            if isinstance(observation, str) and observation.strip():
                parts.append(observation)
    return "\n\n".join(parts)


def _run_guardrails(question: str, raw_answer: str, context: str) -> dict:
    """
    Passes the answer through the hallucination checker.
    If context is empty (no documents retrieved), skips the check
    and returns a neutral score — nothing to verify against.
    If the checker itself crashes, still returns the answer safely
    with a warning rather than killing the whole response.
    """
    if not context.strip():
        return {
            "answer":     raw_answer,
            "confidence": 0.5,
            "flagged":    False,
            "warning":    "No document context retrieved — confidence unverified."
        }

    try:
        from guardrails.checker import check
        result = check(
            question=question,
            answer=raw_answer,
            context=context
        )

        # Defensive check — ensure checker returned the right shape
        if not isinstance(result, dict) or "answer" not in result:
            raise ValueError(f"Unexpected checker output: {type(result)}")

        return result

    except Exception as e:
        print(f"[Guardrails] check() failed: {e}")
        return {
            "answer":     raw_answer,
            "confidence": 0.5,
            "flagged":    False,
            "warning":    f"Guardrails check failed: {str(e)}"
        }


def answer(question: str, history: list = None) -> dict:
    """
    Full pipeline:
      1. Run the ReAct agent (rag_search → web_search / calculate as needed)
      2. Extract rag_search context from intermediate steps
      3. Pass answer + context through hallucination guardrails
      4. Always return a dict with answer, confidence, flagged, warning
    """
    chat_history = history or []

    try:
        response = get_executor().invoke({
            "input":        question,
            "chat_history": chat_history
        })
    except Exception as e:
        print(f"[Agent] Executor failed: {e}")
        return {
            "answer":     f"The agent encountered an error: {str(e)}",
            "confidence": 0.0,
            "flagged":    True,
            "warning":    "Agent execution failed — answer may be unreliable."
        }

    raw_answer         = response.get("output", "No answer produced.")
    intermediate_steps = response.get("intermediate_steps", [])
    context            = _extract_context(intermediate_steps)

    return _run_guardrails(question, raw_answer, context)

async def answer_streaming(question: str, history: list = None):
    """
    Asynchronous generator that yields tokens in real-time.
    """
    chat_history = history or []
    executor = get_executor()

    # .astream_events is a powerful LangChain method that captures 
    # every internal step of the agent as it happens.
    async for event in executor.astream_events(
        {
            "input": question,
            "chat_history": chat_history
        },
        version="v1"
    ):
        kind = event["event"]

        # 'on_chat_model_stream' triggers whenever a new token is 
        # generated by the LLM in the Final Answer phase.
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                yield content
