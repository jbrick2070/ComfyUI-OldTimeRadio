# OpenRouter Remote LLM — Go-Forward Plan (LOCKED)

**2026-05-31 · branch `v2.0-alpha` · execution doc — drives code → wire → regress → commit until all sprints green. Optimized for autonomous Cowork execution: subagent waves, no operator-approval gates.**

Supersedes `docs/2026-05-31-openrouter-remote-llm-architecture-options.md`. Incorporates the round-robin decision (preserved verbatim at `docs/2026-05-31-openrouter-remote-llm__round-robin-locked.md`) and the corrections from the live-path code review (the dispatch table is dormant; no `validate_model_id` surgery is needed; the single-resident cache must not be thrashed). Where this plan refines the round-robin, the change is called out inline as **[code-review refinement]**.

---

## Decision

Build **Option A** with **controlled T1, fail-closed**. Two virtual catalog rows bound by environment variables — no new nodes, no new writer widgets, no workflow JSON changes, no model picker outside the writer. Remote technical (JSON-strict) calls are permitted **only** through schema-enforced output with hard validation and a fail-closed gate, so malformed remote output can never enter the ledger. Cloud is opt-in and default-off; the offline baseline stays byte-identical.

The architecture permits remote on **either** slot. The value is in the discipline: creative is the primary remote target, technical defaults local, and remote technical is opt-in for a verified schema-capable model only.

---

## Hard constraints (every sprint respects these — non-negotiable)

- **C1 — Audio is king.** Remote is default-off; the byte-identical baseline is always computed on the unchanged local path. `tests/test_audio_byte_identical.py` stays green at every gate.
- **C2 — VRAM ceiling 14.5 GB + single-resident cache.** Remote uses zero local VRAM. `request_slot` holds at most one resident model and tears it down on every model-id change (`_otr_model_loader.py:796`). **[code-review refinement]** The remote branch must **not** evict the resident local model — a remote call leaves `LLM_CACHE` untouched. Without this, the common config (creative=remote, technical=local) would evict and reload the local model across slot transitions. Never `force_vram_offload()` between LLM phases.
- **C3 — Offline-first.** No remote call unless `OPENROUTER_API_KEY` **and** `OTR_ENABLE_OPENROUTER=1` are set. Unset ⇒ rows absent, no remote path reachable.
- **C4 — Fail-closed JSON.** Remote technical output that fails schema validation after bounded repair aborts the call with a clear error. Malformed remote JSON never reaches the ledger.
- **C5 — No half-remote episodes.** On remote failure (network / rate-limit / cost) there is **no** mid-episode fall-back to local. Bounded retries, then abort the whole run. Every run records exactly one provenance.
- **C6 — Hard cost guard.** Configurable per-run token/spend ceiling, conservative default. Abort **before** exceeding; log spend per call. No unbounded paid calls.
- **C7 — Two-model tag (PD6).** Remote changes the backend behind a slot, not the slot. No `model_id` widget anywhere; remote ids ride the existing two dropdowns; every LLM call keeps its `# LLM slot: creative|technical` tag.
- **C8 — Guardrails green by construction.** No new widgets, no new nodes, no forbidden symbols. B6 / B7 / workflow-JSON audits unchanged.
- **C9 — No secrets, SFW.** Key from env only; never logged, never committed. No profanity; never the word "dummy" — use "placeholder"/"stub".

---

## Frozen contracts (defined once in S0, never renegotiated)

These are what make the sprints — and any subagent parallelism — safe. Frozen at S0; later sprints consume them, they are not redefined.

**FC1 — Backend surface = the existing `LoaderBackend` protocol** (`nodes/_otr_loader_backends.py:47`): `load(repo_id, row) -> dict`, `generate(model, messages, **kwargs) -> str`, `unload(model) -> None`. `OpenRouterBackend` implements exactly this; the returned cache-entry dict matches the legacy `load_llm` shape and carries `cache_entry["provider"] = "openrouter"`.

**FC2 — The two live seams.** **[code-review refinement]** `get_backend_for_row` / `BACKENDS_BY_KEY` are **dormant** — nothing in production calls them; `request_slot` calls `load_llm` directly (`_otr_model_loader.py:812`). Remote is wired in at exactly two places, named here so no sprint widens the surface:
  1. `request_slot` (`_otr_model_loader.py:712`) — remote branch after validate + cache-hit.
  2. The generate-fn factory — `_build_truncating_generate_fn` (`OTR_LedgerScriptWriter.py:586`), `make_generate_fn` / `make_polish_generate_fn` (`_otr_model_loader.py:864` / `:939`) — remote branch on `cache_entry["provider"]`.

**FC3 — Env names.** `OPENROUTER_API_KEY` (credential), `OPENROUTER_MODEL_A` / `OPENROUTER_MODEL_B` (real slugs A/B resolve to), `OTR_ENABLE_OPENROUTER` (gate). Optional per-slot params (`OPENROUTER_A_TEMP`, `OPENROUTER_A_MAXTOK`, …) read with safe defaults.

**FC4 — Virtual-row schema.** Two `CuratedModel` rows, `repo_id` `openrouter:slot-a` / `openrouter:slot-b`, `loader_backend="openrouter_http"`, `vram_fit_tier="PASS"`, `approx_safetensors_gb=0.0`, `context_window=8192`, plus a new `provider` field (default `"local"`, `"openrouter"` here). **[code-review refinement]** The rows join the curated set **only when enabled**, so `validate_model_id` Path 1 (`_otr_model_catalog.py:493`) admits them with **no validator surgery** — `openrouter:slot-a` already passes `_structural_reject` (`:407`). The round-robin's "add an `openrouter:*` admit-path" step is dropped as unnecessary.

**FC5 — Error & cost contract.** Per C4–C6 above: bounded retries → clean abort; fail-closed JSON; hard cost ceiling enforced before the call. Frozen at S0.

---

## Operator activation (Windows env vars)

Concrete instantiation of FC3 — set once via `setx` (User scope, persists across reboots), then **restart ComfyUI in a fresh terminal** so the process reads them. No secrets live in this repo, only the variable names. `OPENROUTER_API_KEY` is **already set** (2026-05-31); its value is never committed.

To enable remote and pick the A/B models when ready to go live:

```
setx OTR_ENABLE_OPENROUTER 1
setx OPENROUTER_MODEL_A "anthropic/claude-3.5-sonnet"
setx OPENROUTER_MODEL_B "openai/gpt-4o"
```

The A/B slugs are the operator's choice — swappable anytime by re-running `setx`; confirm the exact current slug at openrouter.ai/models (they version). To fully disable remote, unset `OTR_ENABLE_OPENROUTER` or set it to `0`: the rows vanish from the dropdowns and the offline baseline is untouched (C3). None of S0–S3 or the mocked tests need these set; only the enabled smoke run (S6 / W4) does. End-user walkthrough (account → key → enable → use): `docs/openrouter-setup.md`.

---

## Architecture — Option A

Add exactly two virtual rows to the catalog: `openrouter:slot-a`, `openrouter:slot-b` (schema per FC4). They appear in both writer dropdowns (creative and technical) automatically via `dropdown_choices()` (`OTR_LedgerScriptWriter.py:1532` / `:1546`), and **only** when OpenRouter is enabled.

Binding lives in env (FC3); the dropdown shows the named handles "OpenRouter A / B", never raw slugs. No new nodes, no writer config widgets, no graph surgery. The writer's two dropdowns, the slot scheduler, the `technical_model` broadcast, and all consumers stay untouched — remote selection changes the backend behind a slot, not the slot surface.

**Reproducibility** is preserved by stamping the resolved slug into meta at run time (S5), so the env-side binding is always recorded in the run.

---

## Technical JSON rule — controlled T1, fail-closed

**[code-review refinement: grounded on the real structured-call infrastructure — verified to exist.]** Technical-slot calls already route through `structured_call(prompt, schema, slot_fn=generate_fn, ...)` (`nodes/_otr_structured_call.py:293`), which parses + validates via `_parse_and_validate` (`:259`, using `parse_first_json_object` at `nodes/_otr_json.py:76` + Pydantic `model_validate`) and runs a **bounded 3-attempt repair ladder** (base temp → lower temp → `repair_temp=0.1` via `make_dispatching_repair_factory`, `nodes/_otr_repair_prompts.py`), raising `StructuredCallFailedError` on exhaustion. Remote reuses this exact path:

1. Build the remote generate fn from the call's Pydantic schema — mirror `make_constrained_generate_fn(cache_entry, schema_model)` (`nodes/_otr_constrained_generate.py:161`) with a remote equivalent that maps `schema_model.model_json_schema()` → OpenRouter `response_format={type: json_schema, json_schema: {schema: …}}`.
2. Pass that fn as `slot_fn` into the **existing** `structured_call`, so validation + bounded repair are byte-for-byte the same logic local uses.
3. If the model lacks schema support, or `structured_call` exhausts the ladder (`StructuredCallFailedError`), **fail closed** (C4) — abort the call; never write the ledger.

This gives remote technical output the same integrity guarantee the local grammar path (`nodes/_otr_constrained_generate.py`, `nodes/_otr_lmfe_compat.py`) provides at the token level, with **zero new validation logic** — it rides the path the technical slot already uses.

---

## Operating posture

- **Creative slot is the primary remote target.** Narrative passes (outline, cast, dialogue, polish) are the token/latency bulk and where a strong remote model buys the most quality.
- **Technical slot defaults to local.** `DEFAULT_LLM` (`_otr_model_catalog.py:32`) stays Mistral-Nemo for both slots; the local grammar path already handles validators, reviewer verdicts, critic, and news_interpreter reliably.
- **Remote technical is opt-in for a verified schema-capable model only.** Anything less stays local, and the C4 fail-closed gate enforces integrity even if a slot is mis-set. Leaving technical local is a first-class configuration, not a fallback.

---

## Build tracking protocol

Two documents, zero overlap, linked by a pointer never a copy: **this file** is the source of truth for build PROGRESS; **`BUG_LOG.md`** is the source of truth for BUGS. A bug found mid-build is logged in `BUG_LOG.md` first, then pointed to here.

A `- [ ]` item flips to `- [x]` only when: done **and** the regression suite is green **and** (if a node surface was touched) the workflow JSON is re-wired to match. Completion gate: all items `[x]`, all regressions green, every `BUG-LOCAL-NNN` either `[FIXED]` or explicitly parked.

### Sprint status board

| Sprint | Status | Bug pointers | Notes |
|--------|--------|--------------|-------|
| S0 — Baseline lock + freeze contracts | DONE (2026-05-31) | BUG-LOCAL-293 | baseline was RED (5 known pre-existing fails) — fixed to green per Jeffrey; full OTR 3204 pass / 12 skip / 0 fail; Bug Bible 23/1skip/2xfail; FC1–FC5 frozen; checkpoint commit |
| S1 — OpenRouter backend (mocked) | NOT STARTED | — | `_otr_openrouter_backend.py`; cost-abort proven with mocked counter |
| S2 — Catalog rows (enabled-gated) | NOT STARTED | — | rows join curated set when enabled; no validator surgery |
| S3 — Wire remote branch into live path | NOT STARTED | — | `request_slot` + generate-fn branches; no-evict (C2) |
| S4 — Technical JSON fail-closed gate | NOT STARTED | — | schema → validate → bounded repair → fail-closed (C4) |
| S5 — Metadata stamp | NOT STARTED | — | provider + slot + resolved slug + params + schema-mode |
| S6 — Smoke proofs (disabled + enabled) | NOT STARTED | — | byte-identical (off); end-to-end + abort proofs (on) |

---

## Autonomous execution (Cowork) + subagent waves

This plan runs **autonomously with no operator-approval gates**. The executing agent decides and proceeds — it does **not** stop to ask Jeffrey between sprints. It halts only on (a) a red regression it cannot fix after reasonable attempts, (b) a genuine ambiguity this plan does not resolve, or (c) any breach of C1–C9. The frozen contracts **FC1–FC5** make the parallelism safe: each agent codes against a fixed interface, so disjoint-file work never collides.

**Quality gates are automated, not human.** "No stopping gate" means no waiting on approval — it does **not** mean skipping tests. Every wave ends by running the regression set (Bug Bible + core + the suites the sprint touched) and, if a node surface changed, re-wiring + re-auditing the workflow JSON. A wave merges only when green; a red gate is the one thing that pauses the drive-through.

```text
W0  [solo]        S0  baseline lock + freeze FC1–FC5
                      │  (nothing proceeds until baseline green + contracts frozen)
                      ▼
W1  [2 agents ∥]  A: S1  _otr_openrouter_backend.py  (new file)        ┐ disjoint
                  B: S2  _otr_model_catalog.py  (rows + provider)      ┘ files
                      │  merge → full regress → gate
                      ▼
W2  [solo]        S3  wire remote branch into request_slot +
                      generate-fn factory  (live loader; single-threaded
                      to avoid conflicts; enforces C2 no-evict)
                      │  full regress → gate
                      ▼
W3  [2 agents ∥]  A: S4  technical JSON fail-closed gate (structured_call) ┐ mostly
                  B: S5  meta stamp (writer meta block ~:3732)             ┘ disjoint
                      │  merge → full regress → gate
                      ▼
W4  [solo]        S6  smoke proofs (disabled + enabled) + final
                      full regress + Bug Bible → done
```

Git per CLAUDE.md: one push attempt via Desktop Commander cmd shell; commit via `.git\COMMIT_EDITMSG` + `-F` (never `-m`); after each push verify HEAD match, no 0-byte files, no BOM, AST parse. Log bugs to `BUG_LOG.md` the moment they're found.

---

## Sprints

### S0 — Baseline lock + freeze contracts
Run and green the full local baseline before any code: full pytest, `test_audio_byte_identical.py`, workflow JSON audits (`scripts/_audit_workflow_json.py`, `tools/audit_workflow_schema.py`), and the Bug Bible regression. Freeze FC1–FC5 (backend surface, two seams, env names, virtual-row schema, error/cost contract). Commit the clean baseline as the rollback point.

- [ ] Full regression suite green (record the command set + result).
- [ ] `test_audio_byte_identical.py` green — captured as the rollback baseline hash.
- [ ] FC1–FC5 written into this doc and confirmed unchanged for the rest of the build.
- [ ] Checkpoint commit on `v2.0-alpha`; hash recorded in the build log.

### S1 — OpenRouter backend (mocked, no network)
Create `nodes/_otr_openrouter_backend.py` implementing FC1. Key from env only; no secrets in logs; request timeout; bounded retries; hard cost guard; clear error messages. Register `"openrouter_http": OpenRouterBackend()` in `BACKENDS_BY_KEY` (necessary but not sufficient — S3 wires the live call).

- [ ] `load()` returns a provider-tagged cache-entry (no weights, no tokenizer); `unload()` is a no-op.
- [ ] `generate()` posts chat messages, applies the cost guard, returns the decoded string.
- [ ] Mocked-HTTP tests: happy path; timeout; retry-exhaustion → clean abort; **cost-ceiling abort proven with a mocked token counter** (do not defer the cost proof to live smoke).
- [ ] No network in CI; no secret ever printed. Regress green.

### S2 — Catalog rows (enabled-gated)
Add `"openrouter_http"` to the `loader_backend` Literal (`_otr_model_catalog.py:71`) and the `provider` field to `CuratedModel`. **[code-review refinement]** Inject the two virtual rows into the curated set **only when `OTR_ENABLE_OPENROUTER=1`** (via a helper feeding `_by_repo_id()` + `build_dropdown_choices`), so Path 1 validation admits them with no `validate_model_id` change. Do not list real slugs in the dropdown.

- [ ] Rows present in both dropdowns when enabled; absent when disabled.
- [ ] `validate_model_id("openrouter:slot-a")` returns it unchanged when enabled; raises cleanly when disabled. No new admit-path added.
- [ ] `test_model_catalog_scan` + dropdown tests green; new test for the enabled/disabled toggle. Regress green.

### S3 — Wire the remote branch into the live path (load-bearing)
This is where remote becomes reachable (the dispatch table is otherwise dormant — FC2). In `request_slot` (`_otr_model_loader.py:712`), after validate + cache-hit, branch on `row.loader_backend == "openrouter_http"`: route to `get_backend_for_row(row).load(...)`, **skip** steps 3–8 (`resolve_context_cap`, `check_vram_fit`, `auto_download_if_missing`, the resident-model teardown, `load_llm`), and **leave any resident local model in `LLM_CACHE` untouched** (C2). Branch the generate-fn factory (FC2 seam 2) on `cache_entry["provider"] == "openrouter"`.

- [ ] Remote path makes zero CUDA / snapshot / download calls (asserted).
- [ ] A resident local model survives a remote call (no-evict, C2) — explicit test.
- [ ] Generate-fn factory returns a working remote callable for a provider-tagged entry.
- [ ] `test_llm_cache_mismatch_diagnostics` + `test_cache_key_mutations` clean. Regress green.

### S4 — Technical JSON fail-closed gate
Add the remote analog of `make_constrained_generate_fn` (maps the call's Pydantic schema → OpenRouter `response_format`) and route remote technical calls through the **existing** `structured_call` validate + bounded-repair ladder (`nodes/_otr_structured_call.py:293`). On `StructuredCallFailedError` or a no-schema model, fail closed (C4). No new validation logic — reuse `_parse_and_validate`.

- [ ] Remote technical generate fn sets `response_format` from `schema_model.model_json_schema()`.
- [ ] Valid schema output passes `_parse_and_validate` unchanged.
- [ ] Output still invalid after the bounded ladder raises `StructuredCallFailedError` → **fails clean** (no ledger write).
- [ ] A model with no schema support **fails clean** with an actionable error.
- [ ] Regress green; a malformed-remote fixture proves nothing reaches the ledger.

### S5 — Metadata stamp
Mirror the existing `creative_model` meta stamp (`OTR_LedgerScriptWriter.py` ~3724–3741): stamp `provider: openrouter`, the selected slot (`openrouter:slot-a|b`), the resolved model slug, basic generation params, and whether schema mode was used.

- [ ] Every remote run records provider + slot + resolved slug + params + schema-mode in meta.
- [ ] No raw `[NOT DOWNLOADED]` or secret material in any stamp (B6 stamp guards stay green). Regress green.

### S6 — Smoke proofs
Two operator runs, then close out.

1. **Disabled** — rows absent, no remote call possible, `test_audio_byte_identical.py` still green.
2. **Enabled** — A/B rows appear; a creative remote call works; a technical schema call works; the cost-ceiling abort fires; a non-schema technical model fails closed; a forced mid-run remote error aborts cleanly (no half-remote episode).

- [ ] Both runs captured (params + result) in the build log.
- [ ] **User docs shipped:** `docs/openrouter-setup.md` verified current; the README pointer promoted from "experimental" to a real entry.
- [ ] **README refresh (required, not optional):** the README is stale (still v1.7 badge/framing) and is the main on-ramp. Bring it current to v2.0-alpha and reframe it for the real audience — **ComfyUI newbies who drive AI coding assistants ("vibe coders")**: low-jargon, copy-paste-first, "do this, then this" so a beginner gets a run done with zero prior context. The OpenRouter section is one part of that refresh, not the whole job.
- [ ] **In-app hint:** selecting an OpenRouter A/B row while `OPENROUTER_API_KEY` / `OTR_ENABLE_OPENROUTER` is unset raises a clear error that points to `docs/openrouter-setup.md`; the writer's two model dropdowns carry a tooltip naming the env vars.
- [ ] Final full regress + Bug Bible green. Promote any `Bible candidate: yes` fixes via the Three-File Contract.

---

## Deferred / explicitly out of scope

- **Option B (writer config widgets for the A/B slugs).** Clean future upgrade if the binding should live in the workflow JSON rather than env. **[code-review refinement]** It needs **no** B6 change (the writer is exempt; widget names like `openrouter_a_model` are off the reserved list). Not in this build.
- **OpenRouter profile nodes (Option C), a dropdown full of real slugs (Option D), default-on cloud, mid-episode local fall-back, streaming.** All out.
- **Period-model interplay.** The dormant GPTQ-int4 backend and `otr_1940s_v1` profile are untouched by this work.

---

## Appendix — verified code anchors (as of 2026-05-31)

| Claim | Anchor |
|---|---|
| Backend protocol (`load`/`generate`/`unload`) | `nodes/_otr_loader_backends.py:47` |
| Dispatch table + lookup (**dormant** — only callers are this module + tests) | `nodes/_otr_model_runtime.py:152` (`BACKENDS_BY_KEY`), `:159` (`get_backend_for_row`) |
| Live load path (calls `load_llm` directly) | `nodes/_otr_model_loader.py:712` (`request_slot`), `:796` (resident teardown), `:812` (`load_llm` call) |
| Generate-fn factory (needs remote branch) | `OTR_LedgerScriptWriter.py:586`; `_otr_model_loader.py:864` / `:939` |
| Catalog: dataclass / literal / curated set / id map | `_otr_model_catalog.py:53` / `:71` / `:96` / `:195` |
| Validator + structural reject (verified `openrouter:` safe; curated Path 1) | `_otr_model_catalog.py:452` / `:407` / `:493` |
| Skipped-for-remote helpers | `_otr_model_catalog.py:579` (`resolve_context_cap`), `:699` (`check_vram_fit`), `:874` (`auto_download_if_missing`) |
| Default model (byte-identical baseline) | `_otr_model_catalog.py:32` (`DEFAULT_LLM` = Mistral-Nemo) |
| Writer dropdowns + resolved ids + broadcast | `OTR_LedgerScriptWriter.py:1532` / `:1546`; resolved `:1347-1348`; `technical_model` RETURN index 4 |
| B6 scope (3 reserved keys; writer exempt; `NON_LLM_MODEL_WIDGET_OK` opt-out) | `tests/test_b6_wiring_guardrails.py:43` / `:265` |
| Local grammar mechanism | `nodes/_otr_constrained_generate.py`, `nodes/_otr_lmfe_compat.py` |
| Structured-call validate + bounded repair (S4 reuses this) | `nodes/_otr_structured_call.py:293` (`structured_call`), `:259` (`_parse_and_validate`); `nodes/_otr_json.py:76` (`parse_first_json_object`); `nodes/_otr_repair_prompts.py` (`make_dispatching_repair_factory`) |
| Grammar entry the remote path mirrors | `nodes/_otr_constrained_generate.py:161` (`make_constrained_generate_fn(cache_entry, schema_model)`) |
| Reusable technical schemas (Pydantic, `.model_json_schema()`) | `_otr_ledger_reviewer.py` (PreAuditReport, ScriptDoctorDiagnosis/Report), `_otr_story_critic.py` (StoryCriticReport), `_otr_stage1_plan.py` (Stage1Plan), `_otr_outline.py` (Outline) |
| Canonical workflow | `workflows/otr_scifi_16gb_full.json` (writer node id 1) |
| External Bug Bible regression | `C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py` |

---

## Build progress log — append one dated entry per session

### 2026-05-31 · Session 0 (plan locked + verified)
- Round-robin synthesized into this go-forward plan; raw round-robin preserved at `docs/2026-05-31-openrouter-remote-llm__round-robin-locked.md`.
- Code-review refinements folded in: dormant dispatch table (FC2 two-seam wiring), no `validate_model_id` surgery (FC4 curated-set admit), single-resident no-evict rule (C2).
- Live-code accuracy review (2 subagents): all structural claims **PASS**. S4 re-grounded on the real `structured_call`/`_parse_and_validate` ladder (no new validation logic; reuses existing Pydantic schemas).
- Restructured for autonomous Cowork execution: subagent waves W0–W4, no operator-approval gates.
- Status: verified and ready. Awaiting go to start W0/S0.

### 2026-05-31 · Session 1 (W0 / S0 — baseline lock)
- Reviewed every live seam named in the appendix — all anchors confirmed accurate (the writer file is `nodes/OTR_LedgerScriptWriter.py`; line numbers match: `_build_truncating_generate_fn:586`, resolved ids `:1348`, dropdowns `:1546`, creative meta stamp `:3732`; `request_slot:712`, generate-fn factories `make_generate_fn:864`/`make_polish_generate_fn:939`; catalog dataclass/literal/curated/`validate_model_id`; `structured_call:293`/`_parse_and_validate:259`; `make_constrained_generate_fn:161`; dispatch table dormant in `_otr_model_runtime.py`). FC1–FC5 confirmed unchanged.
- **S0 surfaced a RED baseline at HEAD `688263b`**: the full `tests/` walk failed on the 5 long-carried "known pre-existing failures" — none related to OpenRouter. Per S0 ("no feature code until baseline green") I halted and reported; Jeffrey chose **fix-to-green-first**. Logged + fixed as **BUG-LOCAL-293** (test/tooling/JSON-widget only): stale 29-node floor; node-11 `bypass_freeze_halt` restored to safe `false`; missing `# LLM slot: technical` tag at `_otr_slot_drama_contract.py:384`; forbidden-sweep `OSError(22)` from a OneDrive-locked multi-MB temp diff → tooling now writes to the OS temp dir.
- **Baseline now GREEN:** full OTR `tests/` 3204 passed / 12 skipped / 0 failed; Bug Bible 23 passed / 1 skipped / 2 xfailed; `tools/audit_workflow_schema.py` OK. (`scripts/_audit_workflow_json.py` `unregistered_types` is a known environmental artifact — needs the ComfyUI runtime to register all node + built-in types; identical pre/post and not a wiring break.)
- Checkpoint commit on `v2.0-alpha` is the rollback point. Next: W1 (S1 backend ∥ S2 catalog rows).
