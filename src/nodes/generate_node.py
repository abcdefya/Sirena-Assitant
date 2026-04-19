from pathlib import Path
from typing import AsyncIterator
from src.agents.state import AgentState
from src.services.groq_llm import get_llm
from src.cores.logger import get_logger

logger = get_logger(__name__)


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent / "prompts" / "system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return (
        "You are Sirena, a friendly and professional portfolio assistant. "
        "Answer questions about the portfolio owner based on the provided context and chat history. "
        "Be concise and accurate. If you don't know something, say so."
    )

_SYSTEM_PROMPT = _load_system_prompt()


def _build_messages(state: AgentState, context: str = "") -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if context:
        messages.append({
            "role": "system",
            "content": f"Portfolio context:\n{context}",
        })
    for turn in state.get("chat_history", []):
        messages.append(turn)
    messages.append({"role": "user", "content": state["question"]})
    return messages


async def stream_answer(state: AgentState) -> AsyncIterator[str]:
    documents = state.get("documents", [])
    docs_relevant = state.get("docs_relevant", False)
    context = ""
    if documents and docs_relevant:
        context = "\n\n---\n\n".join([doc.page_content for doc in documents])
    messages = _build_messages(state, context=context)
    logger.info(
        f"Session {state['session_id']}: streaming answer "
        f"(rag={'yes' if context else 'no'})"
    )
    async for chunk in get_llm().astream(messages):
        if chunk.content:
            yield chunk.content
