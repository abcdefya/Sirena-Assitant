import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.agents.pipeline import get_pipeline
from src.agents.state import AgentState
from src.cores.session import get_history, append_turn, clear_session
from src.nodes.generate_node import stream_answer
from src.cores.logger import get_logger
from src.api.limiter import limiter

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)


@router.post("/chat")
@limiter.limit("10/minute;50/hour")
async def chat(request: Request, body: ChatRequest):
    session_id = body.session_id
    question = body.message
    chat_history = get_history(session_id)
    logger.info(f"Session {session_id}: received message")

    initial_state: AgentState = {
        "session_id": session_id,
        "question": question,
        "chat_history": chat_history,
        "documents": [],
        "answer": "",
        "needs_retrieval": False,
        "docs_relevant": False,
    }

    pipeline = get_pipeline()
    result_state: AgentState = pipeline.invoke(initial_state)

    async def token_stream():
        full_answer = ""
        try:
            async for token in stream_answer(result_state):
                full_answer += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"Session {session_id}: streaming error — {e}")
            yield f"data: {json.dumps({'error': 'Something went wrong. Please try again.'})}\n\n"
        else:
            append_turn(session_id, "user", question)
            append_turn(session_id, "assistant", full_answer)
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


@router.delete("/chat/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
