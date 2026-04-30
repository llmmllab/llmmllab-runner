"""Simplified user config for runner server argument building."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class UserConfig(BaseModel):
    """Simplified user config for runner server argument building."""

    model_config = ConfigDict(extra="ignore", protected_namespaces=())
