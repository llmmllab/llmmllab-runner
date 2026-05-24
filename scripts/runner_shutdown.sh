#!/usr/bin/env bash
#
# runner_shutdown.sh — force-evict llama.cpp / sd-server processes from
# the runner so VRAM gets freed immediately, instead of waiting for the
# idle timeout (default 60 min).
#
# Wired through the api's /v1/runner/* admin endpoints, so you don't
# need a port-forward to reach the runner directly.  Auth is the same
# bearer-token flow the other test scripts use; the api requires the
# token to belong to an admin user.
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
#   API_BASE   default http://192.168.0.71:9999  (the cluster gateway)
#   API_KEY    admin bearer token (required to reach /v1/runner/*)

set -euo pipefail

API_BASE="${API_BASE:-http://192.168.0.71:9999}"

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
            sed -n '2,28p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            echo "see --help" >&2
            exit 2 ;;
    esac
done

AUTH_HEADER=()
if [[ -n "${API_KEY:-}" ]]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_KEY")
fi

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

api() {
    local method=$1 path=$2
    curl -sS -X "$method" "$API_BASE$path" \
        -H "Content-Type: application/json" \
        "${AUTH_HEADER[@]+"${AUTH_HEADER[@]}"}" \
        --max-time 30 \
        --fail-with-body
}

# Free VRAM per GPU, one block per runner endpoint.
print_gpu_stats() {
    local label="$1"
    echo "== $label =="
    api GET /v1/runner/health 2>/dev/null | jq -r '
        if (.runners // []) | length == 0
        then "  (no runners reachable)"
        else
          (.runners[] |
            ("  " + .endpoint +
              (if .error
               then "  [\(.error)]"
               else
                 "  active_servers=\(.active_servers)\n" +
                 ((.gpu // {}) | to_entries |
                   map("    GPU \(.key)  \(.value.name)  free=\(.value.free_mb|floor)MB / \(.value.total_mb|floor)MB") |
                   join("\n"))
               end)))
        end
    ' || echo "  (could not read /v1/runner/health)"
}

# One row per active server across every runner.
list_servers() {
    api GET /v1/runner/servers 2>/dev/null | jq -r '
        if (.runners // []) | length == 0
        then "  (no runners reachable)"
        else
          [.runners[] | (.endpoint as $e | (.servers // [])[] |
            "  - \(.server_id)  model=\(.model_id)  port=\(.port)  use=\(.use_count)  healthy=\(.healthy)  @ \($e)")] |
          if length == 0 then "  (no active servers)" else .[] end
        end
    ' || echo "  (could not read /v1/runner/servers)"
}

list_pipelines() {
    api GET /v1/runner/pipelines 2>/dev/null | jq -r '
        if (.pipelines // []) | length == 0
        then "  (no pipelines registered)"
        else
          (.pipelines[] |
            "  - \(.name)  task=\(.task)  loaded=\(.loaded)  @ \(.endpoint)")
        end
    ' || echo "  (could not read /v1/runner/pipelines)"
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

if ! api GET /v1/runner/health >/dev/null 2>&1; then
    cat >&2 <<EOF
✘ Couldn't reach the api admin endpoints at $API_BASE.

Check:
  - API_BASE points at the api gateway (default: http://192.168.0.71:9999)
  - API_KEY is set to an admin bearer token (anything else gets 403)
  - The api is healthy: curl $API_BASE/health
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

# Build the list of server ids to evict.  --server / --model are
# precise; bare --all picks up every active server.
target_ids=()
if [[ -n "$TARGET_SERVER" ]]; then
    target_ids=("$TARGET_SERVER")
elif [[ -n "$TARGET_MODEL" ]]; then
    mapfile -t target_ids < <(
        api GET /v1/runner/servers 2>/dev/null \
            | jq -r --arg m "$TARGET_MODEL" \
                '[.runners[].servers[] | select(.model_id == $m) | .server_id] | .[]'
    )
    if [[ ${#target_ids[@]} -eq 0 ]]; then
        echo "✘ no active servers matched model=$TARGET_MODEL" >&2
        exit 3
    fi
elif [[ "$MODE" == "all" || "$MODE" == "interactive" ]]; then
    mapfile -t target_ids < <(
        api GET /v1/runner/servers 2>/dev/null \
            | jq -r '[.runners[].servers[]?.server_id] | .[]'
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

# Evict.
#
# For --model and --server we hit the specific evict path so the api
# returns 404 if the id has already gone away (race with the idle
# eviction).  For --all we use the batch endpoint which is one HTTP
# call regardless of fleet size.
if [[ -n "$TARGET_SERVER" || -n "$TARGET_MODEL" ]]; then
    for sid in "${target_ids[@]}"; do
        echo "→ POST /v1/runner/servers/$sid/evict"
        api POST "/v1/runner/servers/$sid/evict" || {
            echo "  ! evict $sid failed; continuing"
        }
    done
elif [[ ${#target_ids[@]} -gt 0 ]]; then
    echo "→ POST /v1/runner/servers/evict-all"
    api POST "/v1/runner/servers/evict-all" >/dev/null || {
        echo "  ! evict-all failed; continuing"
    }
fi

# Unload pipelines (one call per pipeline name).
if [[ "$ALSO_PIPELINES" == "1" ]]; then
    mapfile -t pipeline_names < <(
        api GET /v1/runner/pipelines 2>/dev/null \
            | jq -r '[.pipelines[] | select(.loaded == true) | .name] | unique | .[]'
    )
    for name in "${pipeline_names[@]}"; do
        echo "→ POST /v1/runner/pipelines/$name/unload"
        api POST "/v1/runner/pipelines/$name/unload" || {
            echo "  ! unload $name failed; continuing"
        }
    done
fi

# Allow a moment for CUDA contexts to release before re-reading stats.
sleep 2

echo ""
print_gpu_stats "AFTER"
