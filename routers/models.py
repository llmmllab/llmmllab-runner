"""Models router - list available models."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from models import Model, ModelTask
from utils.model_loader import ModelLoader

router = APIRouter()
model_loader = ModelLoader()


@router.get("/v1/models", response_model=List[Model])
def list_models(task: Optional[str] = Query(default=None)):
    """List all available models, optionally filtered by task."""
    all_models = model_loader.get_available_models()
    result = []

    for model in all_models.values():
        if task and model.task.value != task:
            continue
        result.append(model)

    return result


@router.get("/v1/models/{model_id}")
def get_model(model_id: str) -> Dict[str, Any]:
    """Get a specific model by ID.

    Returns 404 with structured error if the model is not configured
    on this runner, including a list of available model IDs.
    """
    model = model_loader.get_model_by_id(model_id)
    if not model:
        available = sorted(model_loader.get_available_models().keys())
        raise HTTPException(
            status_code=404,
            detail={
                "reason": "model_not_configured",
                "message": f"Model '{model_id}' is not configured on this runner",
                "requested_model": model_id,
                "available_models": available if len(available) <= 20 else None,
            },
        )
    return {
        "model_id": model.id,
        "name": model.name,
        "provider": model.provider,
        "task": model.task.value,
        "details": model.details.model_dump() if model.details else None,
    }
