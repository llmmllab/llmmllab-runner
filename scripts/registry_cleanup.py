#!/usr/bin/env python3
"""Trim a Docker Distribution v2 registry to the N most recent tags per repo.

Designed for the LAN registry at ``192.168.0.71:31500`` which fills up
with commit-SHA-named tags from every CI build.  Keeps ``--keep`` newest
tags per repository (default 5), plus any tag matching ``--protect``
(default: ``latest``, ``latest-*``).  All other tags get their manifests
deleted.

Important: deleting a manifest in Distribution v2 only **dereferences**
the layers — disk space isn't reclaimed until you run

    registry garbage-collect /etc/docker/registry/config.yml -m

against the registry binary (the runbook in the README has the exact
``docker exec`` invocation).  This script doesn't do that step itself
because it doesn't have shell access to the registry container.

The script uses only the Python stdlib (``urllib``, ``json``,
``base64``) so it runs unchanged in any minimal container image.

Auth resolution order (first wins):

  1. ``--user`` / ``--password`` CLI flags
  2. ``REGISTRY_USER`` / ``REGISTRY_PASSWORD`` env vars
  3. ``DOCKER_CONFIG_JSON`` env var (a Kubernetes dockerconfigjson
     secret mounted directly as an env value), parsed for the entry
     matching ``--registry``
  4. ``DOCKER_CONFIG_JSON_FILE`` env var (path to a file containing
     the same JSON; this is what a CronJob would mount from a
     ``kubernetes.io/dockerconfigjson`` secret)

Dry-run by default; pass ``--apply`` to actually delete.
"""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple


_LOG_FMT = "%(asctime)s %(levelname)-7s %(message)s"
logger = logging.getLogger("registry-cleanup")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


_MANIFEST_ACCEPT = ", ".join([
    # Docker v2 schema-2 manifest (most images CI pushes are this)
    "application/vnd.docker.distribution.manifest.v2+json",
    # OCI image manifest (newer / buildx)
    "application/vnd.oci.image.manifest.v1+json",
    # Manifest list (multi-arch) — we still want the digest so it's
    # deletable; the layers it references get cleaned up by registry GC.
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
])


def _request(
    url: str,
    method: str = "GET",
    auth_header: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    timeout: float = 30.0,
) -> Tuple[int, dict, bytes]:
    """One HTTP round-trip.  Returns ``(status_code, response_headers, body)``."""
    req = urllib.request.Request(url, method=method)
    if auth_header:
        req.add_header("Authorization", auth_header)
    for k, v in (extra_headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items() if e.headers else {}), e.read() or b""


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def _resolve_auth_header(
    registry: str,
    cli_user: Optional[str],
    cli_password: Optional[str],
) -> Optional[str]:
    """Build the ``Authorization: Basic ...`` header from the first source
    that has credentials."""

    user, password = cli_user, cli_password

    if not (user and password):
        user = user or os.environ.get("REGISTRY_USER")
        password = password or os.environ.get("REGISTRY_PASSWORD")

    if not (user and password):
        # k8s dockerconfigjson — either inline or as a mounted file.
        raw = os.environ.get("DOCKER_CONFIG_JSON")
        if not raw:
            path = os.environ.get("DOCKER_CONFIG_JSON_FILE")
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    raw = fh.read()
        if raw:
            try:
                cfg = json.loads(raw)
                auths = cfg.get("auths", {})
                # Match exact host or any entry containing the host (cluster
                # registries are sometimes keyed without scheme).
                entry = auths.get(registry)
                if entry is None:
                    for host, info in auths.items():
                        if registry in host or host in registry:
                            entry = info
                            break
                if entry:
                    if entry.get("auth"):
                        return f"Basic {entry['auth']}"
                    user = user or entry.get("username")
                    password = password or entry.get("password")
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not parse dockerconfigjson: %s", e)

    if user and password:
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return f"Basic {token}"
    return None


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


@dataclass
class TagInfo:
    repo: str
    tag: str
    digest: str
    created: Optional[datetime]
    size_bytes: int = 0


@dataclass
class CleanupPlan:
    keep: List[TagInfo] = field(default_factory=list)
    delete: List[TagInfo] = field(default_factory=list)
    protected: List[TagInfo] = field(default_factory=list)
    skipped: List[Tuple[str, str, str]] = field(default_factory=list)  # (repo, tag, reason)


class Registry:
    """Thin Distribution v2 client — only the calls the cleanup needs."""

    def __init__(self, base: str, auth_header: Optional[str]):
        self.base = base.rstrip("/")
        self.auth = auth_header

    def _url(self, path: str) -> str:
        return f"{self.base}{path}"

    def list_repos(self) -> List[str]:
        """Iterate ``/_catalog`` with pagination — registries cap n per page."""
        repos: List[str] = []
        next_url = self._url("/v2/_catalog?n=200")
        while next_url:
            status, headers, body = _request(next_url, auth_header=self.auth)
            if status != 200:
                raise RuntimeError(
                    f"GET {next_url} -> {status}: {body[:200].decode(errors='replace')}"
                )
            data = json.loads(body or b"{}")
            repos.extend(data.get("repositories", []) or [])
            # Follow ``Link`` header if present (rel="next")
            link_header: Optional[str] = headers.get("Link") or headers.get("link")
            next_url = None
            if link_header and 'rel="next"' in link_header:
                # ``<...>; rel="next"`` — extract the URL.
                bracket = link_header.split(">", 1)[0]
                if bracket.startswith("<"):
                    path_part = bracket[1:]
                    next_url = (
                        path_part
                        if path_part.startswith("http")
                        else self._url(path_part)
                    )
        return repos

    def list_tags(self, repo: str) -> List[str]:
        status, _, body = _request(self._url(f"/v2/{repo}/tags/list"), auth_header=self.auth)
        if status == 404:
            return []
        if status != 200:
            raise RuntimeError(
                f"GET tags/list({repo}) -> {status}: {body[:200].decode(errors='replace')}"
            )
        data = json.loads(body or b"{}")
        return data.get("tags", []) or []

    def get_tag_metadata(self, repo: str, tag: str) -> Optional[TagInfo]:
        """Fetch the manifest + config blob to read ``created`` + size.

        Returns ``None`` when the tag has been deleted out from under us
        (manifest 404) — the caller logs + skips.  Manifest-list
        responses also short-circuit with the digest but no creation
        date; the caller can still decide whether to delete those.
        """
        # Step 1 — manifest, including the Docker-Content-Digest header
        # (we need that exact digest to issue the DELETE later).
        status, headers, body = _request(
            self._url(f"/v2/{repo}/manifests/{tag}"),
            auth_header=self.auth,
            extra_headers={"Accept": _MANIFEST_ACCEPT},
        )
        if status == 404:
            return None
        if status != 200:
            raise RuntimeError(
                f"GET manifests({repo}:{tag}) -> {status}: "
                f"{body[:200].decode(errors='replace')}"
            )
        digest = headers.get("Docker-Content-Digest") or headers.get("docker-content-digest")
        if not digest:
            raise RuntimeError(f"manifests({repo}:{tag}) missing Docker-Content-Digest")
        manifest = json.loads(body or b"{}")

        # Manifest list / index — there's no single ``config`` blob, so
        # we can't read ``created``.  Use the registry's modified-at
        # timestamp from the header if available, else mark as unknown.
        if "manifests" in manifest:
            return TagInfo(repo=repo, tag=tag, digest=digest, created=None)

        config_ref = manifest.get("config") or {}
        config_digest = config_ref.get("digest")
        size_bytes = sum(int(layer.get("size", 0) or 0) for layer in manifest.get("layers", []))
        if config_digest:
            size_bytes += int(config_ref.get("size", 0) or 0)

        created: Optional[datetime] = None
        if config_digest:
            cstatus, _, cbody = _request(
                self._url(f"/v2/{repo}/blobs/{config_digest}"),
                auth_header=self.auth,
            )
            if cstatus == 200:
                try:
                    cfg = json.loads(cbody or b"{}")
                    raw = cfg.get("created")
                    if isinstance(raw, str):
                        # ISO-8601 like ``2026-05-24T01:23:45.678901Z``
                        created = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except Exception as e:  # noqa: BLE001
                    logger.debug("config blob parse failed for %s:%s: %s", repo, tag, e)

        return TagInfo(
            repo=repo,
            tag=tag,
            digest=digest,
            created=created,
            size_bytes=size_bytes,
        )

    def delete_manifest(self, repo: str, digest: str) -> int:
        status, _, body = _request(
            self._url(f"/v2/{repo}/manifests/{digest}"),
            method="DELETE",
            auth_header=self.auth,
        )
        if status == 404:
            # Already gone — fine, log and move on.
            logger.info("  %s@%s already deleted", repo, digest[:16])
            return status
        if status not in (200, 202):
            raise RuntimeError(
                f"DELETE manifests({repo}@{digest}) -> {status}: "
                f"{body[:200].decode(errors='replace')}"
            )
        return status


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def _is_protected(tag: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(tag, p) for p in patterns)


def plan_repo_cleanup(
    registry: Registry,
    repo: str,
    keep: int,
    protect_patterns: List[str],
) -> CleanupPlan:
    """For one repo, decide which tags to keep vs delete."""
    plan = CleanupPlan()

    tags = registry.list_tags(repo)
    if not tags:
        return plan

    infos: List[TagInfo] = []
    for tag in tags:
        if _is_protected(tag, protect_patterns):
            try:
                info = registry.get_tag_metadata(repo, tag)
            except Exception as e:  # noqa: BLE001
                logger.warning("  metadata failure on protected tag %s:%s: %s", repo, tag, e)
                plan.skipped.append((repo, tag, str(e)))
                continue
            if info is not None:
                plan.protected.append(info)
            continue

        try:
            info = registry.get_tag_metadata(repo, tag)
        except Exception as e:  # noqa: BLE001
            logger.warning("  metadata failure on %s:%s: %s — skipping", repo, tag, e)
            plan.skipped.append((repo, tag, str(e)))
            continue
        if info is None:
            plan.skipped.append((repo, tag, "manifest 404"))
            continue
        infos.append(info)

    # Most recent first; tags with no creation date sort to the back so
    # we keep dated tags when possible.  Two-key sort: (has_date,
    # date) descending — entries with ``created is None`` collapse to
    # ``(False, EPOCH)`` and end up after any dated entry.
    _EPOCH = datetime.fromtimestamp(0, tz=timezone.utc)
    infos.sort(
        key=lambda i: (i.created is not None, i.created or _EPOCH),
        reverse=True,
    )
    plan.keep = infos[:keep]
    plan.delete = infos[keep:]

    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 1)[0]
    )
    p.add_argument(
        "--registry",
        default=os.environ.get("REGISTRY_URL", "http://192.168.0.71:31500"),
        help="Registry base URL (default: $REGISTRY_URL or http://192.168.0.71:31500)",
    )
    p.add_argument("--user", help="Registry username (or set $REGISTRY_USER)")
    p.add_argument("--password", help="Registry password (or set $REGISTRY_PASSWORD)")
    p.add_argument(
        "--keep",
        type=int,
        default=int(os.environ.get("REGISTRY_KEEP", "5")),
        help="Number of most-recent tags to keep per repository (default: 5)",
    )
    p.add_argument(
        "--protect",
        action="append",
        default=None,
        help=(
            "fnmatch pattern that's never deleted (repeatable). "
            "Default: ``latest`` and ``latest-*``."
        ),
    )
    p.add_argument(
        "--repo",
        action="append",
        help="Only clean up these repos (repeatable). Default: every repo.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the manifests.  Without this, the script only prints the plan.",
    )
    p.add_argument(
        "--verbose", "-v", action="count", default=0, help="Repeat for more logging"
    )
    args = p.parse_args(argv)

    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(format=_LOG_FMT, level=max(level, logging.DEBUG))

    protect = args.protect or ["latest", "latest-*"]

    # ``--registry`` may or may not include the scheme; the host shape used
    # in the secret matches what was pushed (e.g. ``192.168.0.71:31500``).
    base = args.registry
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    host_only = base.split("://", 1)[1].rstrip("/")

    auth = _resolve_auth_header(host_only, args.user, args.password)
    if auth is None:
        logger.warning(
            "No credentials resolved; proceeding anonymously (the LAN registry "
            "likely requires Basic auth and you'll see 401)"
        )

    reg = Registry(base, auth)

    if args.repo:
        repos = list(args.repo)
    else:
        repos = reg.list_repos()
    logger.info("Found %d repositories", len(repos))

    grand_total_deleted = 0
    grand_total_bytes = 0
    for repo in sorted(repos):
        plan = plan_repo_cleanup(reg, repo, args.keep, protect)
        if not (plan.keep or plan.delete or plan.protected):
            logger.debug("  %s: no tags", repo)
            continue
        bytes_to_free = sum(t.size_bytes for t in plan.delete)
        logger.warning(
            "%s: %d protected, %d keep (newest), %d will be deleted (~%.2f GiB)",
            repo,
            len(plan.protected),
            len(plan.keep),
            len(plan.delete),
            bytes_to_free / (1024 ** 3),
        )
        for t in plan.delete:
            created_label = t.created.isoformat() if t.created else "?"
            logger.info("  DELETE %s:%s (%s, %d bytes)", t.repo, t.tag, created_label, t.size_bytes)
            if args.apply:
                try:
                    reg.delete_manifest(t.repo, t.digest)
                except Exception as e:  # noqa: BLE001
                    logger.error("  delete failed for %s:%s: %s", t.repo, t.tag, e)
                    continue
        grand_total_deleted += len(plan.delete)
        grand_total_bytes += bytes_to_free

    mode = "applied" if args.apply else "dry-run"
    logger.warning(
        "[%s] would delete %d manifests across %d repos (~%.2f GiB before GC).",
        mode,
        grand_total_deleted,
        len(repos),
        grand_total_bytes / (1024 ** 3),
    )
    if args.apply and grand_total_deleted:
        logger.warning(
            "Disk space is not reclaimed until you run "
            "``registry garbage-collect /etc/docker/registry/config.yml -m`` "
            "against the registry binary.  See scripts/README.md."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
