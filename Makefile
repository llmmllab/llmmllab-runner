# llmmllab-runner - standalone llama.cpp server manager and request proxy

PYTHON   ?= python3
UV       ?= uv
APP      ?= app:app
HOST     ?= 0.0.0.0
PORT     ?= 9000
IMAGE    ?= llmmllab-runner
TAG      ?= latest
REGISTRY ?= 192.168.0.71:31500

.PHONY: help install start start-reload validate clean test docker-build docker-run

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
	@for f in $$(find . -name '*.py' -not -path './__pycache__/*'); do $(UV) run $(PYTHON) -m py_compile "$$f" || exit 1; done
	@echo "Syntax OK"

clean: ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true

## Testing
test: ## Run all tests
	$(UV) run pytest -v

test-unit: ## Run unit tests only
	$(UV) run pytest tests/unit -v

## Docker
docker-build: ## Build Docker image
	docker build -t $(REGISTRY)/$(IMAGE):$(TAG) .

docker-push: ## Push Docker image to registry
	docker push $(REGISTRY)/$(IMAGE):$(TAG)

docker-build-push: docker-build docker-push ## Build and push Docker image

## Kubernetes
# RUNNER_DEPLOY: all (default), main, small
RUNNER_DEPLOY ?= main

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
