from typing import Dict, List
from src.cores.logger import get_logger

logger = get_logger(__name__)

_sessions: Dict[str, List[dict]] = {}


def get_history(session_id: str) -> List[dict]:
    return _sessions.get(session_id, [])


def append_turn(session_id: str, role: str, content: str) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append({"role": role, "content": content})
    logger.debug(f"Session {session_id}: appended {role} turn")


def clear_session(session_id: str) -> None:
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Session {session_id}: cleared")
