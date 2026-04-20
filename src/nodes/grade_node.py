from src.agents.state import AgentState
from src.services.groq_llm import get_llm
from src.cores.logger import get_logger

logger = get_logger(__name__)

_GRADE_PROMPT = """You are a relevance grader for a portfolio assistant.

Given a user question and retrieved document chunks, decide if the documents contain information relevant to answering the question.

Answer with ONLY "yes" or "no".
- "yes" if at least one chunk contains information relevant to the question
- "no" if the chunks are unrelated to the question

Question: {question}

Document chunks:
{documents}

Answer:"""


def grade_node(state: AgentState) -> AgentState:
    question = state["question"]
    documents = state.get("documents", [])
    docs_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    prompt = _GRADE_PROMPT.format(question=question, documents=docs_text)
    response = get_llm().invoke(prompt)
    decision = response.content.strip().lower()
    docs_relevant = decision.startswith("yes")
    logger.info(
        f"Session {state['session_id']}: grading decision='{decision}' docs_relevant={docs_relevant}"
    )
    return {**state, "docs_relevant": docs_relevant}
