from src.agents.state import AgentState
from src.services.groq_llm import get_llm
from src.cores.logger import get_logger

logger = get_logger(__name__)

_DECIDE_PROMPT = """You are a routing assistant for a personal portfolio chatbot.

Given the user's question, decide if it requires retrieving information from the portfolio knowledge base (projects, CV, experience, skills, education).

Answer with ONLY "yes" or "no".
- "yes" if the question is about the portfolio owner's projects, skills, experience, education, tech stack, or professional background
- "no" if the question is a greeting, small talk, or general question not related to the portfolio

Question: {question}
Answer:"""


def decide_node(state: AgentState) -> AgentState:
    question = state["question"]
    prompt = _DECIDE_PROMPT.format(question=question)
    response = get_llm().invoke(prompt)
    decision = response.content.strip().lower()
    needs_retrieval = decision.startswith("yes")
    logger.info(
        f"Session {state['session_id']}: routing decision='{decision}' needs_retrieval={needs_retrieval}"
    )
    return {**state, "needs_retrieval": needs_retrieval}
