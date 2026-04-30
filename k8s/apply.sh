#!/bin/bash
# Deploy llmmllab-runner: build, push, apply k8s manifests.
#
# Usage:
#   ./k8s/apply.sh              # Build, push, deploy both runners
#   ./k8s/apply.sh --no-build   # Skip build/push, only apply manifests
#   ./k8s/apply.sh --small      # Only deploy the small runner
#   ./k8s/apply.sh --main       # Only deploy the main runner
#
# Environment variables:
#   DOCKER_TAG   - Image tag (default: latest)
#   REGISTRY     - Docker registry (default: 192.168.0.71:31500)

set -euo pipefail

REGISTRY="${REGISTRY:-192.168.0.71:31500}"
DOCKER_TAG="${DOCKER_TAG:-latest}"
IMAGE="${REGISTRY}/llmmllab-runner:${DOCKER_TAG}"
NAMESPACE="llmmllab"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
K8S_DIR="${PROJECT_ROOT}/k8s"

NO_BUILD=false
DEPLOY_MODE="main"  # all | main | small (default: main, small not ready yet)

for arg in "$@"; do
  case "$arg" in
    --no-build) NO_BUILD=true ;;
    --small)    DEPLOY_MODE="small" ;;
    --main)     DEPLOY_MODE="main" ;;
    *)          echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Build & push ─────────────────────────────────────────────────────────
if [[ "$NO_BUILD" != "true" ]]; then
  echo "Building Docker image: ${IMAGE}"
  docker build -t "$IMAGE" "$PROJECT_ROOT"
  echo "Pushing image to registry..."
  docker push "$IMAGE"
fi

# ── Apply manifests ──────────────────────────────────────────────────────
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo "Applying ReferenceGrant..."
kubectl apply -f "$K8S_DIR/referencegrant.yaml"

echo "Applying Service..."
kubectl apply -f "$K8S_DIR/service.yaml" -n "$NAMESPACE"

case "$DEPLOY_MODE" in
  small)
    echo "Applying small runner deployment (tag: ${DOCKER_TAG})..."
    kubectl set image deployment/llmmllab-runner-small \
      llmmllab-runner="$IMAGE" -n "$NAMESPACE" --record=true \
      || kubectl apply -f "$K8S_DIR/deployment.yaml" -n "$NAMESPACE" \
        --selector='size=small'
    ;;
  main)
    echo "Applying main runner deployment (tag: ${DOCKER_TAG})..."
    kubectl set image deployment/llmmllab-runner \
      llmmllab-runner="$IMAGE" -n "$NAMESPACE" --record=true \
      || kubectl apply -f "$K8S_DIR/deployment.yaml" -n "$NAMESPACE" \
        --selector='size!=small'
    ;;
  all)
    echo "Applying runner deployments (tag: ${DOCKER_TAG})..."
    kubectl set image deployment/llmmllab-runner \
      llmmllab-runner="$IMAGE" -n "$NAMESPACE" --record=true \
      || true
    kubectl set image deployment/llmmllab-runner-small \
      llmmllab-runner="$IMAGE" -n "$NAMESPACE" --record=true \
      || true
    # Fallback: if set image failed (deployments don't exist yet), apply fresh
    kubectl get deployment -n "$NAMESPACE" -l app=llmmllab-runner >/dev/null 2>&1 \
      || kubectl apply -f "$K8S_DIR/deployment.yaml" -n "$NAMESPACE"
    ;;
esac

echo ""
echo "Deployment complete!"
echo "  Image: ${IMAGE}"
echo "  Service: llmmllab-runner.${NAMESPACE}.svc.cluster.local:8000"
echo "  Health: http://llmmllab-runner.${NAMESPACE}.svc.cluster.local:8000/health"
