from typing import get_type_hints

def test_agent_state_has_all_required_fields():
    from src.agents.state import AgentState
    hints = get_type_hints(AgentState)
    required = {"session_id", "question", "chat_history", "documents", "answer", "needs_retrieval", "docs_relevant"}
    assert required.issubset(set(hints.keys()))

def test_agent_state_chat_history_is_list():
    from src.agents.state import AgentState
    hints = get_type_hints(AgentState)
    origin = getattr(hints["chat_history"], "__origin__", None)
    assert origin is list

def test_agent_state_needs_retrieval_is_bool():
    from src.agents.state import AgentState
    hints = get_type_hints(AgentState)
    assert hints["needs_retrieval"] is bool

def test_agent_state_docs_relevant_is_bool():
    from src.agents.state import AgentState
    hints = get_type_hints(AgentState)
    assert hints["docs_relevant"] is bool
