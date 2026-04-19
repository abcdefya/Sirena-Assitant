from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.nodes.decide_node import decide_node
from src.nodes.retrieve_node import retrieve_node
from src.nodes.grade_node import grade_node
from src.cores.logger import get_logger

logger = get_logger(__name__)

_app = None


def _route_after_decide(state: AgentState) -> str:
    return "retrieve" if state["needs_retrieval"] else END


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("decide", decide_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)

    workflow.set_entry_point("decide")
    workflow.add_conditional_edges(
        "decide",
        _route_after_decide,
        {"retrieve": "retrieve", END: END},
    )
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", END)

    app = workflow.compile()
    logger.info("LangGraph pipeline compiled successfully")
    return app


def get_pipeline():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
