#!/usr/bin/env python3
"""check-errors.py - Deduplicates Loki log data against open GitHub PRs.

Part of github-collaborate skill.
Reads a Loki JSON result on stdin, cross-references open PRs,
reports whether errors are already covered. Outputs structured JSON.
If no unaddressed errors found, prints {"action": "none"} and exits 0.

Fingerprints are normalized to strip variable parts (timestamps, UUIDs,
request IDs) so the same error with different transient values deduplicates.

Usage: python3 ~/.openclaw/skills/log-monitor/scripts/log-query.sh 60 | \\
        python3 ~/.openclaw/skills/github-collaborate/scripts/check-errors.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# Ensure sibling modules in the same directory are importable.
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])

import gh  # noqa: E402

# Patterns that match variable parts in log lines
_VARYING_PATTERNS = [
    re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?'),  # ISO timestamps
    re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'),  # UUIDs
    re.compile(r'"?request[_-]?id"?\s*[:=]\s*"?[0-9a-f-]+"?'),  # request IDs
    re.compile(r'\b\d{4,}\b'),  # large numbers (PIDs, ports, memory sizes)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check log errors against open PRs")
    parser.add_argument("--repos", type=str, default=None, help="Comma-separated list of repos")
    parser.add_argument("--user", type=str, default=None, help="Only PRs by this user")
    parser.add_argument("--branch-prefix", type=str, default=None, help="Only PRs with this branch prefix")
    return parser.parse_args()


# Pod/app name → GitHub repo mapping.
# Extended to cover all managed llmmllab pods so errors are filed in the
# correct repo instead of defaulting to llmmllab-api.
_POD_TO_REPO = [
    ("runner", "llmmllab-runner"),
    ("gateway", "llmmllab-gateway"),
    ("ui", "llmmllab-ui"),
    ("openclaw", "openclaw-k8s"),
    ("api", "llmmllab-api"),
]


def infer_repo(pod: str) -> str:
    """Infer the GitHub repo name from a Loki pod/app label.

    Checks each known substring pattern in order; first match wins.
    Falls back to ``llmmllab-api`` for unknown pods (least surprising
    default for the monolith that handles most traffic).
    """
    pod_lower = pod.lower()
    for substring, repo in _POD_TO_REPO:
        if substring in pod_lower:
            return repo
    return "llmmllab-api"


def normalize_fingerprint(line: str) -> str:
    """Normalize a log line for stable fingerprinting.

    Strips variable parts like timestamps, UUIDs, request IDs, and large numbers
    so the same error with different transient values produces the same fingerprint.
    """
    fp = line[:200].strip()
    for pattern in _VARYING_PATTERNS:
        fp = pattern.sub("_", fp)
    # Collapse multiple underscores/spaces into one, then strip leading/trailing
    fp = re.sub(r'_[\s_]*', '_', fp)
    return fp.strip('_ ')


def main() -> None:
    args = parse_args()

    # Resolve repos: CLI arg > env > default
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    else:
        repos = gh.resolve_repos()

    # Read Loki JSON from stdin
    try:
        log_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        print(json.dumps({"error": "invalid JSON on stdin"}), file=sys.stderr)
        sys.exit(1)

    errors = log_data.get("errors", [])
    if not errors:
        print(json.dumps({"action": "none", "reason": "no errors found"}))
        sys.exit(0)

    # Fetch open PRs across repos
    all_prs = gh.fetch_open_prs(repos=repos, user=args.user, branch_prefix=args.branch_prefix)

    # Build set of fingerprints already covered by open PRs
    covered: set[str] = set()
    for pr in all_prs:
        body = pr.get("body", "") or ""
        for line in body.split("\n"):
            if "LOGMONITOR-FINGERPRINT:" in line:
                fp = line.split("LOGMONITOR-FINGERPRINT:")[1].strip()
                covered.add(normalize_fingerprint(fp))

    # Filter out covered errors
    unaddressed: list[dict] = []
    for e in errors:
        raw_line = e.get("line", "").strip()
        fp = normalize_fingerprint(raw_line)
        if fp and fp not in covered:
            pod = e.get("pod", "")
            unaddressed.append({**e, "repo": infer_repo(pod), "fingerprint": fp})

    if not unaddressed:
        print(json.dumps({
            "action": "none",
            "reason": "all errors covered by open PRs",
            "error_count": len(errors),
        }))
        sys.exit(0)

    # Deduplicate by normalized fingerprint
    seen: set[str] = set()
    deduped: list[dict] = []
    for e in unaddressed:
        fp = e["fingerprint"]
        if fp not in seen:
            seen.add(fp)
            deduped.append(e)

    print(json.dumps({
        "action": "unaddressed_errors_found",
        "total_unaddressed": len(deduped),
        "total_errors": len(errors),
        "covered_by_prs": len(errors) - len(unaddressed),
        "errors": deduped[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
