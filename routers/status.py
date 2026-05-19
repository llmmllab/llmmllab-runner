"""Runner status / identity endpoint.

Exposes a cheap liveness probe that returns this process's startup epoch.
The api side polls this to detect runner restarts: if `startup_epoch` ever
changes between calls, the api invalidates all cached server handles and
re-acquires them. The endpoint must be cheap (no DB / GPU / llama.cpp work)
because it may be called on every request.
"""

import time

from fastapi import APIRouter
from pydantic import BaseModel

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="StatusRouter")

# Computed ONCE at module import time. Each worker process has its own value,
# fixed for the life of that process. The api uses this to detect restarts.
_STARTUP_EPOCH_MS: int = int(time.time() * 1000)

logger.info("Runner startup epoch", startup_epoch_ms=_STARTUP_EPOCH_MS)

router = APIRouter()


class StatusResponse(BaseModel):
    startup_epoch: int
    now: int


@router.get("/v1/status", response_model=StatusResponse)
def status() -> StatusResponse:
    """Return this process's startup epoch (ms) and current time (ms).

    `startup_epoch` is fixed per process; the api uses changes to detect
    runner restarts. `now` is provided for clock-skew detection.
    """
    return StatusResponse(
        startup_epoch=_STARTUP_EPOCH_MS,
        now=int(time.time() * 1000),
    )
