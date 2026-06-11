"""Tests for the infer_repo pod->repo mapping in scripts/check_errors.py.

The infer_repo function maps Loki pod/app labels to GitHub repo names
so log-monitor auto-fix files issues in the correct repository.
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock

# Mock the gh module before importing check_errors (gh.py lives in the
# skill directory, not this repo).
sys.modules["gh"] = MagicMock()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from check_errors import infer_repo


class TestInferRepo:
    """Verify pod/app labels map to the correct GitHub repo."""

    def test_runner_pod(self):
        assert infer_repo("llmmllab-runner") == "llmmllab-runner"

    def test_runner_small_pod(self):
        assert infer_repo("llmmllab-runner-small") == "llmmllab-runner"

    def test_gateway_pod(self):
        assert infer_repo("llmmllab-gateway") == "llmmllab-gateway"

    def test_ui_pod(self):
        assert infer_repo("llmmllab-ui") == "llmmllab-ui"

    def test_openclaw_pod(self):
        assert infer_repo("openclaw") == "openclaw-k8s"

    def test_api_pod(self):
        assert infer_repo("llmmllab-api") == "llmmllab-api"

    def test_unknown_pod_defaults_to_api(self):
        assert infer_repo("unknown-service") == "llmmllab-api"

    def test_case_insensitive(self):
        assert infer_repo("LLMMLLAB-GATEWAY") == "llmmllab-gateway"
        assert infer_repo("OpenClaw") == "openclaw-k8s"
        assert infer_repo("LLMMLLAB-RUNNER-SMALL") == "llmmllab-runner"

    def test_empty_pod_defaults_to_api(self):
        assert infer_repo("") == "llmmllab-api"
