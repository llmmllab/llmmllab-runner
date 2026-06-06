# llmmllab-runner - standalone llama.cpp server manager and request proxy

PYTHON   ?= python3
UV       ?= uv
APP      ?= app:app
HOST     ?= 0.0.0.0
PORT     ?= 9000
IMAGE    ?= llmmllab-runner
TAG      ?= latest
REGISTRY ?= 192.168.0.71:31500

# Pinned llama.cpp commit — the SOURCE OF TRUTH for which llama.cpp the runner
# builds against. MUST be a commit that has gemma-4 vision (the gemma4uv mtmd
# model; the deploy guard enforces this). `git commit -a` from a drifted local
# checkout keeps silently reverting the submodule gitlink to an old SHA and
# blocking the deploy — so do NOT let the gitlink be the source of truth.
# To roll llama.cpp forward: bump THIS line, then `make vendor-sync && git commit`.
LLAMA_CPP_SHA ?= 308f61c31f083251ce8150f10b9ef97679b500b5

.PHONY: help install start start-reload validate clean test docker-build docker-run vendor-sync vendor-status vendor-check vendor-install-py install-hooks

## Help
help: ## Show this help
	@grep -E '^[a-zA-Z]' $(MAKEFILE_LIST) | awk -F: '{print $$1}' | grep -v help

## Dependencies
install: ## Install dependencies
	$(UV) sync

## Running
start: ## Start runner server (no reload)
	$(UV) run python -m uvicorn $(APP) --host $(HOST) --port $(PORT) --reload --timeout-graceful-shutdown 30

## Code quality
validate: ## Python syntax check
	@for f in $$(find . -name '*.py' -not -path './__pycache__/*' -not -path './vendors/*' -not -path './.venv/*'); do $(UV) run $(PYTHON) -m py_compile "$$f" || exit 1; done
	@echo "Syntax OK"

clean: ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

## Testing
test: ## Run all tests
	$(UV) run pytest -v

test-unit: ## Run unit tests only
	$(UV) run pytest tests/unit -v

## Vendors (git submodules under vendors/)
##   vendors/llama.cpp             text inference (ggml-org/llama.cpp)
##   vendors/stable-diffusion.cpp  image inference (leejet/stable-diffusion.cpp)
##   vendors/Hunyuan3D-2           image-to-3D pipeline (Tencent/Hunyuan3D-2)
vendor-sync: ## Sync vendor submodules; FORCE llama.cpp to $(LLAMA_CPP_SHA) and stage the gitlink
	git submodule sync --recursive
	-git submodule update --init --recursive
	@echo "→ pinning vendors/llama.cpp to $(LLAMA_CPP_SHA)"
	-@git -C vendors/llama.cpp fetch -q origin $(LLAMA_CPP_SHA) 2>/dev/null || git -C vendors/llama.cpp fetch -q --all
	git -C vendors/llama.cpp checkout -q --detach $(LLAMA_CPP_SHA)
	git add vendors/llama.cpp
	@grep -rq gemma4uv vendors/llama.cpp/tools/mtmd 2>/dev/null \
	  && echo "✓ llama.cpp pinned ($(LLAMA_CPP_SHA), gemma4uv ✓) + gitlink staged — now: git commit" \
	  || { echo "✗ $(LLAMA_CPP_SHA) lacks gemma4uv (same check the deploy guard runs) — fix LLAMA_CPP_SHA"; exit 1; }

vendor-check: ## Verify the STAGED llama.cpp gitlink matches the pin (run by the pre-commit hook + CI)
	@got=`git ls-files -s vendors/llama.cpp | awk '{print $$2}'`; \
	if [ "$$got" = "$(LLAMA_CPP_SHA)" ]; then \
	  echo "✓ llama.cpp gitlink matches pin ($(LLAMA_CPP_SHA))"; \
	else \
	  echo "✗ llama.cpp gitlink ($$got) != pin ($(LLAMA_CPP_SHA))"; \
	  echo "  A stale checkout reverted the gitlink. Fix: make vendor-sync && git commit"; \
	  exit 1; \
	fi

install-hooks: ## Install the repo's git hooks (pre-commit guards the llama.cpp pin). Run once after cloning.
	git config core.hooksPath scripts/git-hooks
	-chmod +x scripts/git-hooks/* 2>/dev/null
	@echo "✓ hooks installed (core.hooksPath=scripts/git-hooks). commit -a can no longer revert the llama.cpp pin."

vendor-status: ## Show pinned commit / current state of each vendor submodule
	@echo "pin (LLAMA_CPP_SHA): $(LLAMA_CPP_SHA)"
	@git submodule status

vendor-install-py: vendor-sync ## Editable-install the Python-side vendors locally for type hints + IDE jump-to-def (Hunyuan3D's hy3dgen package). Requires the runner's CUDA-bound deps (torch, kornia, timm) already in your venv if you want to actually run anything; without them you still get pyright/symbol resolution.
	$(UV) pip install --no-deps -e vendors/Hunyuan3D-2

## Docker
docker-build: vendor-sync ## Build Docker image (auto-syncs vendor submodules first)
	docker build -t $(REGISTRY)/$(IMAGE):$(TAG) .

docker-push: ## Push Docker image to registry
	docker push $(REGISTRY)/$(IMAGE):$(TAG)

docker-build-push: docker-build docker-push ## Build and push Docker image

## Kubernetes
# RUNNER_DEPLOY: all (default), main, small
RUNNER_DEPLOY ?= all

deploy: docker-build-push k8s-deploy ## Build, push, and deploy to k8s
	@echo "Deployment complete!"

k8s-deploy: ## Apply k8s manifests (use RUNNER_DEPLOY=main|small|all)
	@chmod +x k8s/apply.sh
	@if [ "$(RUNNER_DEPLOY)" = "small" ]; then \
		./k8s/apply.sh --no-build --small; \
	elif [ "$(RUNNER_DEPLOY)" = "main" ]; then \
		./k8s/apply.sh --no-build --main; \
	else \
		./k8s/apply.sh --no-build; \
	fi

k8s-logs: ## Stream runner logs from k8s
	@kubectl logs -n llmmllab -l app=llmmllab-runner -f --tail=50

k8s-restart: ## Restart runner deployment(s)
ifeq ($(RUNNER_DEPLOY),small)
	@kubectl rollout restart -n llmmllab deployment/llmmllab-runner-small
else ifeq ($(RUNNER_DEPLOY),main)
	@kubectl rollout restart -n llmmllab deployment/llmmllab-runner
else
	@kubectl rollout restart -n llmmllab deployment/llmmllab-runner deployment/llmmllab-runner-small 2>/dev/null || kubectl rollout restart -n llmmllab deployment/llmmllab-runner
endif

docker-run: ## Run Docker container (with GPU)
	docker run --gpus all -p $(PORT):$(PORT) \
		-v $(PWD)/.models.yaml:/app/.models.yaml:ro \
		-e MODELS_FILE_PATH=/app/.models.yaml \
		-e RUNNER_PORT=$(PORT) \
		$(REGISTRY)/$(IMAGE):$(TAG)
