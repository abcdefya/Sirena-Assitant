from src.agents.state import AgentState
from src.services.vectorstore import get_retriever
from src.cores.logger import get_logger

logger = get_logger(__name__)


def retrieve_node(state: AgentState) -> AgentState:
    question = state["question"]
    retriever = get_retriever(k=4)
    documents = retriever.invoke(question)
    logger.info(f"Session {state['session_id']}: retrieved {len(documents)} documents")
    return {**state, "documents": documents}
