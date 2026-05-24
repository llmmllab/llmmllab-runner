#!/usr/bin/env bash
#
# runner_shutdown.sh — force-evict llama.cpp / sd-server processes from the
# runner so VRAM gets freed immediately, instead of waiting for the idle
# timeout (default 60 min).
#
# Hits the runner's POST /v1/server/{server_id}/evict (the explicit
# force-evict path — the same one /v1/server/{server_id}/release calls
# eventually, but skipping the use-count → idle-timer dance) and
# POST /v1/pipelines/{name}/unload for the in-process pipeline.
#
# The runner is ClusterIP-only — if you're outside the cluster, port-forward
# first:
#
#     kubectl port-forward -n llmmllab svc/llmmllab-runner 8000:8000
#
# then either default to ``RUNNER_URL=http://localhost:8000`` or set it
# explicitly.
#
# Usage:
#
#   ./scripts/runner_shutdown.sh                 # interactive: list, ask, kill
#   ./scripts/runner_shutdown.sh --list          # print stats, do nothing
#   ./scripts/runner_shutdown.sh --all           # kill every server + unload every pipeline
#   ./scripts/runner_shutdown.sh --server <id>   # kill one server by id
#   ./scripts/runner_shutdown.sh --model <id>    # kill every server backing the named model
#   ./scripts/runner_shutdown.sh --pipelines     # also unload in-process pipelines
#   ./scripts/runner_shutdown.sh --yes           # skip the interactive confirm
#
# Env:
#
#   RUNNER_URL  default http://localhost:8000
#   RUNNER_NS   default llmmllab          (only used by the printed port-forward hint)

set -euo pipefail

RUNNER_URL="${RUNNER_URL:-http://localhost:8000}"
RUNNER_NS="${RUNNER_NS:-llmmllab}"

MODE=interactive            # interactive | list | all
TARGET_SERVER=""
TARGET_MODEL=""
ALSO_PIPELINES=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)        MODE=list; shift ;;
        --all)         MODE=all;  ALSO_PIPELINES=1; shift ;;
        --server)      TARGET_SERVER="$2"; shift 2 ;;
        --model)       TARGET_MODEL="$2";  shift 2 ;;
        --pipelines)   ALSO_PIPELINES=1;   shift ;;
        --yes|-y)      ASSUME_YES=1;       shift ;;
        --help|-h)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            echo "see --help" >&2
            exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

api() {
    local method=$1 path=$2
    curl -sS -X "$method" "$RUNNER_URL$path" \
        -H "Content-Type: application/json" \
        --max-time 30 \
        --fail-with-body
}

# Pretty-print free VRAM per GPU.  /health returns:
#   { "gpu": { "0": {"name": ..., "free_mb": ..., "total_mb": ...}, ... }, ... }
print_gpu_stats() {
    local label="$1"
    echo "== $label =="
    api GET /health 2>/dev/null | jq -r '
        .gpu | to_entries[] |
        "  GPU \(.key)  \(.value.name)  free=\(.value.free_mb|floor)MB / \(.value.total_mb|floor)MB"
    ' || echo "  (could not read /health)"
}

# Print one summary row per active server.  Stats endpoint returns
# {"active_servers": int, "servers": [{server_id, model_id, port, use_count, ...}, ...]}
list_servers() {
    api GET /health 2>/dev/null | jq -r '
        if (.active_servers // 0) == 0
        then "  (no active servers)"
        else
          "  active=\(.active_servers)"
        end
    '
    # /health summary doesn't include the per-server details — hit the
    # collection endpoint for that.
    local servers_json
    servers_json=$(api GET /v1/servers 2>/dev/null || echo '{"servers":[]}')
    echo "$servers_json" | jq -r '
        if (.servers // []) | length == 0
        then "  (no per-server detail available)"
        else
          (.servers[] |
            "  - \(.server_id)  model=\(.model_id)  port=\(.port)  use=\(.use_count)  healthy=\(.healthy)")
        end
    ' 2>/dev/null || true
}

list_pipelines() {
    api GET /v1/pipelines 2>/dev/null | jq -r '
        if (.pipelines // []) | length == 0
        then "  (no pipelines registered)"
        else
          (.pipelines[] |
            "  - \(.name)  task=\(.task)  loaded=\(.loaded)")
        end
    ' || echo "  (could not read /v1/pipelines)"
}

confirm() {
    local prompt="$1"
    if [[ "$ASSUME_YES" == "1" ]]; then
        echo "$prompt  [yes via --yes]"
        return 0
    fi
    read -r -p "$prompt [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

# ---------------------------------------------------------------------------
# reachability check
# ---------------------------------------------------------------------------

if ! api GET /health >/dev/null 2>&1; then
    cat >&2 <<EOF
✘ Couldn't reach the runner at $RUNNER_URL.

If you're outside the cluster, port-forward first:

    kubectl port-forward -n $RUNNER_NS svc/llmmllab-runner 8000:8000

Then re-run with the default RUNNER_URL=http://localhost:8000.
EOF
    exit 1
fi

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

print_gpu_stats "BEFORE"
echo ""
echo "== Active servers =="
list_servers
echo ""
echo "== Pipelines =="
list_pipelines
echo ""

if [[ "$MODE" == "list" ]]; then
    exit 0
fi

# Collect target server ids
target_ids=()
if [[ -n "$TARGET_SERVER" ]]; then
    target_ids=("$TARGET_SERVER")
elif [[ -n "$TARGET_MODEL" ]]; then
    mapfile -t target_ids < <(
        api GET /v1/servers 2>/dev/null \
            | jq -r --arg m "$TARGET_MODEL" '.servers[] | select(.model_id == $m) | .server_id'
    )
    if [[ ${#target_ids[@]} -eq 0 ]]; then
        echo "✘ no active servers matched model=$TARGET_MODEL" >&2
        exit 3
    fi
elif [[ "$MODE" == "all" || "$MODE" == "interactive" ]]; then
    mapfile -t target_ids < <(
        api GET /v1/servers 2>/dev/null \
            | jq -r '.servers[].server_id' 2>/dev/null
    )
fi

if [[ ${#target_ids[@]} -eq 0 && "$ALSO_PIPELINES" -eq 0 ]]; then
    echo "Nothing to do."
    exit 0
fi

if [[ ${#target_ids[@]} -gt 0 ]]; then
    echo "Will evict ${#target_ids[@]} server(s):"
    for sid in "${target_ids[@]}"; do echo "  - $sid"; done
fi
if [[ "$ALSO_PIPELINES" == "1" ]]; then
    echo "Will unload any loaded pipelines."
fi

if [[ "$MODE" == "interactive" ]]; then
    confirm "Proceed?" || { echo "aborted"; exit 0; }
fi

# Evict servers
for sid in "${target_ids[@]}"; do
    echo "→ POST /v1/server/$sid/evict"
    api POST "/v1/server/$sid/evict" || {
        echo "  ! evict $sid failed; continuing"
    }
done

# Unload pipelines
if [[ "$ALSO_PIPELINES" == "1" ]]; then
    mapfile -t pipeline_names < <(
        api GET /v1/pipelines 2>/dev/null \
            | jq -r '.pipelines[] | select(.loaded == true) | .name'
    )
    for name in "${pipeline_names[@]}"; do
        echo "→ POST /v1/pipelines/$name/unload"
        api POST "/v1/pipelines/$name/unload" || {
            echo "  ! unload $name failed; continuing"
        }
    done
fi

# Allow a moment for CUDA contexts to release before re-reading stats.
sleep 2

echo ""
print_gpu_stats "AFTER"
