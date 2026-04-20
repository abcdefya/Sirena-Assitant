import pytest

@pytest.fixture(autouse=True)
def clear_sessions():
    from src.cores import session
    session._sessions.clear()
    yield
    session._sessions.clear()

def test_get_history_returns_empty_for_unknown_session():
    from src.cores.session import get_history
    assert get_history("nonexistent-session") == []

def test_append_turn_creates_session():
    from src.cores import session
    session.append_turn("sess-1", "user", "Hello")
    assert session.get_history("sess-1") == [{"role": "user", "content": "Hello"}]

def test_append_turn_adds_multiple_turns():
    from src.cores import session
    session.append_turn("sess-2", "user", "Hi")
    session.append_turn("sess-2", "assistant", "Hello!")
    history = session.get_history("sess-2")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "Hi"}
    assert history[1] == {"role": "assistant", "content": "Hello!"}

def test_clear_session_removes_history():
    from src.cores import session
    session.append_turn("sess-3", "user", "Hi")
    session.clear_session("sess-3")
    assert session.get_history("sess-3") == []

def test_clear_nonexistent_session_does_not_raise():
    from src.cores.session import clear_session
    clear_session("does-not-exist")  # should not raise
