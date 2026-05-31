# OpenRouter Remote LLM — Go-Forward Plan (LOCKED)

**Branch `v2.0-alpha` · execution plan · supersedes the architecture-options doc**

## Decision

Build **Option A** with **controlled T1, fail-closed**. Two virtual catalog rows bound by environment variables; no new nodes, no new writer widgets, no workflow JSON changes, no model picker outside the writer. Remote technical (JSON-strict) calls are permitted **only** through schema-enforced output with hard validation and a fail-closed gate, so malformed remote output can never enter the ledger. Cloud is opt-in and default-off; the offline baseline stays byte-identical.

## Architecture — Option A

Add exactly two virtual rows to the catalog:

- `openrouter:slot-a`
- `openrouter:slot-b`

Both carry `loader_backend="openrouter_http"` and `vram_fit_tier="PASS"`. They appear in both writer dropdowns (creative and technical) automatically, and **only** when OpenRouter is enabled.

Binding lives in env (mirrors the existing key pattern):

- `OPENROUTER_API_KEY` — credential
- `OPENROUTER_MODEL_A` / `OPENROUTER_MODEL_B` — the real slugs A/B resolve to
- `OTR_ENABLE_OPENROUTER=1` — gate; unset ⇒ rows absent, no remote call possible

No new nodes. No writer config widgets. No real model slugs listed directly in the dropdown. No graph surgery. The writer's two dropdowns, the slot scheduler, the `technical_model` broadcast, and all consumers stay untouched — remote selection changes the backend behind a slot, not the slot surface.

## Technical JSON rule — controlled T1, fail-closed

Remote technical calls are allowed, but never casually. For every technical-slot call routed to OpenRouter:

1. Send with `response_format={type: json_schema}`.
2. Validate the returned JSON against the existing validators.
3. Run existing repair as a **bounded** backstop only.
4. If the selected model lacks schema support, or validation still fails after bounded repair, **fail clearly**. Never emit broken ledger JSON.

This gates remote technical output behind the same integrity guarantee the local grammar path provides, without weakening the pipeline.

## Operating posture

The architecture permits remote on either slot; the value is in the discipline.

- **Creative slot is the primary remote target.** The narrative passes (outline, cast, dialogue, polish) are the token and latency bulk and where a strong remote model buys the most quality.
- **Technical slot defaults to local.** The local grammar path enforces JSON at the token level and already handles the structured passes (validators, reviewer verdicts, critic, news_interpreter) reliably.
- **Remote technical is opt-in for a verified model only** — one confirmed to support schema output and to earn its place on the structured passes. Anything less stays local, and the fail-closed gate enforces that even if a slot is mis-set.

Leaving the technical slot local is a first-class configuration, not a fallback.

## Error & cost contract (frozen in S0)

- **Remote call failure** (network / rate-limit): bounded retries; on exhaustion, **abort the run with a clear error**. No auto-fall-back mid-episode — no episode is ever half-remote / half-local, so every run records exactly one provenance.
- **Invalid JSON** (technical slot): fail-closed per the rule above.
- **Cost**: hard, configurable per-run token/spend ceiling, conservative default. Abort **before** exceeding; log spend per call. No unbounded paid calls.

## Implementation

### S0 — Baseline lock
Run and green the full local baseline before any code: full pytest, audio-byte-identical, workflow JSON audits, Bug Bible regression. Freeze the backend surface (= existing `LoaderBackend`), the env names, the virtual-row schema, the default-off gate, and the error/cost contract above. Commit the clean baseline.

### S1 — OpenRouter backend
Create `nodes/_otr_openrouter_backend.py` implementing `load()`, `generate()`, `unload()`. Requirements: key from env only; no secrets in logs (use "placeholder"/"stub", never "dummy"); request timeout; bounded retries; hard cost guard; clear error messages. Tests are mocked HTTP, no network: happy path, timeout, retry exhaustion → clean abort, and **cost-ceiling abort proven with a mocked token counter** (do not defer the cost proof to the live smoke). Register `"openrouter_http": OpenRouterBackend()`.

### S2 — Catalog rows
Add `"openrouter_http"` to the `loader_backend` literal. Inject the two virtual rows only when enabled. Add a `validate_model_id` admit-path for `openrouter:*` that bypasses file/VRAM checks. Do not add real slugs to the dropdown. Tests: rows present when enabled, absent when disabled; catalog scan green.

### S3 — Loader bypasses
For `openrouter_http` rows, skip local snapshot lookup, auto-download, CUDA warmup, VRAM-fit check, and CUDA teardown; set `context_cap` from the row. Remote rows consume zero local VRAM, so the 14.5 GB ceiling can only move down when remote is active. Tests: no CUDA calls on the remote path; cache-key / mismatch tests clean.

### S4 — Technical JSON enforcement
Wire the fail-closed gate for technical-slot remote calls: schema response format → validate → bounded repair → fail-closed. Tests: valid schema output passes; output still invalid after repair **fails clean**; a model with no schema support **fails clean**. Malformed remote output must never reach the ledger.

### S5 — Metadata stamp
Stamp every remote run (mirror the existing `creative_model` meta stamp): `provider: openrouter`, the selected slot (`openrouter:slot-a|b`), the resolved model slug, basic generation params, and whether schema mode was used. Runs become reproducible by record.

### S6 — Smoke proofs
Two operator runs:

1. **Disabled** — rows absent, no remote call possible, audio-byte-identical still green.
2. **Enabled** — A/B rows appear; creative remote call works; technical schema call works; cost-ceiling abort fires; a non-schema technical model fails closed; a forced mid-run remote error aborts cleanly.

Final full regress + Bug Bible.

## Non-negotiable

- **Audio is king** — remote is default-off; the byte-identical baseline is always computed on the unchanged local path.
- **VRAM ceiling 14.5 GB** — remote uses zero local VRAM; never force offload between LLM phases.
- **Offline-first** — no remote call unless `OPENROUTER_API_KEY` + `OTR_ENABLE_OPENROUTER=1` are set.
- **Two-model tag** — remote changes the backend behind a slot, not the slot; no `model_id` widget anywhere; every call keeps its `# LLM slot: creative|technical` tag.
- **No secrets** in code or logs; key from env only.
- **No** OpenRouter profile nodes. **No** writer config widgets (yet). **No** dropdown full of slugs. **No** remote technical output bypassing validation. **No** default-on cloud.
