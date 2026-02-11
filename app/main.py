from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .config import settings
from .reranker import reranker

logging.basicConfig(level=settings.LOG_LEVEL)
trace_logger = logging.getLogger("n8n-assistant.trace")
if settings.TRACE_LOG_ENABLED:
    trace_logger.setLevel(settings.TRACE_LOG_LEVEL)
    trace_logger.propagate = False
    trace_path = settings.TRACE_LOG_FILE
    if trace_path:
        trace_dir = os.path.dirname(trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        handler_exists = any(
            getattr(handler, "_trace_handler", False) for handler in trace_logger.handlers
        )
        if not handler_exists:
            file_handler = RotatingFileHandler(
                trace_path,
                maxBytes=settings.TRACE_LOG_MAX_BYTES,
                backupCount=settings.TRACE_LOG_BACKUPS,
                encoding="utf-8",
            )
            file_handler.setLevel(settings.TRACE_LOG_LEVEL)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            file_handler._trace_handler = True  # type: ignore[attr-defined]
            trace_logger.addHandler(file_handler)
    trace_logger.info(
        "TRACE SESSION START %s pid=%s env=%s",
        datetime.now().isoformat(timespec="seconds"),
        os.getpid(),
        settings.APP_ENV,
    )
else:
    trace_logger.setLevel(logging.CRITICAL + 10)

app = FastAPI(title="n8n Workflow Assistant", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)


@app.on_event("startup")
def _warm_reranker() -> None:
    if settings.ENABLE_RERANK:
        reranker.load()
