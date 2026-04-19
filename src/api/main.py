from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.api.limiter import limiter
from src.services.vectorstore import ingest_documents
from src.cores.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Sirena Assistant — running document ingestion...")
    ingest_documents()
    logger.info("Sirena Assistant ready to serve requests")
    yield
    logger.info("Sirena Assistant shutting down")


app = FastAPI(title="Sirena Assistant API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


from src.api.routes.chat import router as chat_router  # noqa: E402
app.include_router(chat_router)
