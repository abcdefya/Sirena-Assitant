from typing import TypedDict, List
from langchain_core.documents import Document


class AgentState(TypedDict):
    session_id: str
    question: str
    chat_history: List[dict]    # [{"role": "user/assistant", "content": "..."}]
    documents: List[Document]   # retrieved chunks from ChromaDB
    answer: str
    needs_retrieval: bool
    docs_relevant: bool