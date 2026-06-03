# TODO: Upstream llama.cpp patch — slot save/restore + cache-reuse for multimodal models

**Status:** open · **Priority:** high · **Owner:** unassigned
**Workaround in place:** `SLOT_SAVE_DIR=""` on both runners (disables persistence
entirely) — see `k8s/deployment.yaml` and the 2026-06-03 runner commit.

## Problem

Every model we serve is multimodal (each has an `mmproj`/`clip_model_path`:
Qwen3.6-27B, 35B-A3B, 9B, and Qwen3.5-4B). llama.cpp's server refuses slot
save/restore **and** prefix cache-reuse for multimodal contexts:

- `POST /slots/{id}?action=save` → **501**, `error: "This feature is not
  supported by multimodal"` (observed continuously in the 27B runner log).
- `slot update_slots ... cache reuse is not supported - ignoring n_cache_reuse`.

Because the proxy attempts a save **after every turn** (`_schedule_post_turn_save`)
**and on every LRU eviction** (`proxy/router.py`), this produced a per-turn /
per-eviction storm of 501s. More importantly, with persistence unavailable an
LRU-evicted session loses its KV and must **full-re-prefill its entire prompt**
(40-80k tokens, tens of seconds) when it returns — the dominant cause of the
multi-session slowdown we hit on 2026-06-03.

The current mitigation (`SLOT_SAVE_DIR=""`) stops the storm but gives up KV
persistence entirely, including across restarts and evictions.

## Why llama.cpp blocks it (to confirm)

The server gates save/restore (and prefix reuse) behind a "not multimodal"
check. The likely reason: the KV/sequence state serialization
(`llama_state_seq_get_data` / `_set_data` and the server's `slot_save`/
`slot_restore`) was written for text-only contexts and does not account for the
**image/clip embedding state** (the mtmd/clip side), so restoring a saved slot
could desync the multimodal embeddings from the KV positions.

**Investigate (pin exact refs against our vendored llama.cpp):**
- `tools/server/server.cpp` — the `handle_slots_action` / `slot_save` /
  `slot_restore` paths and the `is_multimodal` / `mtmd` guard that returns 501.
- The `n_cache_reuse` path in `update_slots` that logs "cache reuse is not
  supported" for multimodal.
- `llama_state_seq_*` in `src/llama.cpp` — what state is (not) serialized.
- The mtmd/clip context (`tools/mtmd`) — whether image embedding state is
  per-request (recomputed) or carried in the KV (which would need serializing).

## Hypothesis / approach

For an **agentic text workload that only occasionally sends an image**, the KV
after the multimodal tokens are processed is just ordinary KV — the image
embeddings have already been projected into token positions. If no live image
embedding state needs to persist, save/restore of the **text KV** should be
safe.

Candidate upstream change (smallest viable):
1. Allow `slot save/restore` for multimodal **when the slot's context contains
   no pending/unflushed image embeddings** (i.e. all mtmd chunks already
   consumed into KV). Gate on that condition instead of a blanket multimodal
   refusal.
2. Same for `n_cache_reuse` (prefix reuse) — permit reuse of the text-only
   prefix region; only refuse reuse across an image-chunk boundary.
3. If image embedding state genuinely must persist, extend
   `llama_state_seq_get_data`/`set_data` to (de)serialize the mtmd/clip slot
   state alongside the KV, and bump the state-file version.

## Plan

- [ ] Reproduce minimally: `llama-server --mmproj … --slot-save-path /tmp/s`,
      run a text-only turn, `POST /slots/0?action=save` → capture the 501 + the
      exact guard in `server.cpp`.
- [ ] Determine whether text-only multimodal slots carry any non-KV state that
      blocks serialization (read mtmd/clip slot lifecycle).
- [ ] Implement the conditional gate (1)+(2); add a server test
      (`tools/server/tests`) that saves+restores a multimodal-model slot after a
      text-only turn and asserts the continuation matches.
- [ ] Open the PR upstream (ggml-org/llama.cpp); reference the desync risk +
      the gating condition. Link this doc.
- [ ] Once merged + vendored: re-enable `SLOT_SAVE_DIR=/slots` on the runners
      and re-enable `cache_reuse` for the multimodal models; verify no 501s and
      that evicted sessions restore instead of re-prefilling.

## Impact when fixed

- Evicted sessions restore from disk instead of re-prefilling 40-80k tokens →
  removes the multi-session slowdown without needing `kv_unified`/extra slots as
  the only lever.
- KV survives runner restarts (faster recovery after deploys).
- `cache_reuse` re-enabled → fewer re-prefilled tokens on prefix divergence.
