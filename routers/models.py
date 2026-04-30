"""Models router - list available models."""

from typing import List, Optional

from fastapi import APIRouter, Query

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
