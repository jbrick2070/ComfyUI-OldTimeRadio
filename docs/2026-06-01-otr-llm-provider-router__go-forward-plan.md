# OTR LLM Provider Router — Go-Forward Plan (rev 2, reconciled to shipped code)

**Date:** 2026-06-01 · **Branch:** v2.0-alpha · **Status:** GO-FORWARD (pivot confirmed)
**Reconciled against live code at HEAD `3ff7692`** (Desktop Commander; sandbox `bash` git is stale — use DC/file-tools for git).

---

## North star

> **Two visible choices. Three backend providers. Creative gets the strongest available model. Technical stays cheap, stable, and local unless changed.**

One model writes the story; one handles structure/repair/ledger. Don't expose a provider matrix.

## 0. Decision

Keep **two visible dropdowns** — `Creative LLM`, `Technical LLM`. Each resolves to one of **three providers** (`local` / `openrouter` / `comfy_credits`). Creative defaults to **Auto / Best Available**; Technical defaults to **Local**. This is a **migration of the already-shipped 4-dropdown router** (see §4) — *reuse the backend, swap the UI, add comfy_credits + Auto + script-lock + fail-fallback*.

## 1. User-facing architecture

- **`creative_writing_model`** values: `Auto / Best Available` *(default)*, `Comfy Credits`, `OpenRouter`, then local model ids.
- **`technical_model`** values: `Local` *(default)*, `Auto / Best Available`, `Comfy Credits`, `OpenRouter`.

Two widgets only → **PD6 returns to two model picks, no amendment** (the shipped 4-widget amendment is reverted; see §6/§9). The specific remote *slug* is no longer a visible picker — it resolves behind the scenes via the existing `resolve_slug` chain (env / recommended default).

## 2. Backend resolution

Routing is already data-driven: a slot goes to a provider's generate-fn from the resolved `cache_entry["provider"]` tag (`OTR_LedgerScriptWriter.py` `_build_truncating_generate_fn` ~L622). Each dropdown value → a provider **at execution time**:

- `Local` / local id → local generate-fn.
- `OpenRouter` → own-key lane (shipped); slug from `resolve_slug` chain.
- `Comfy Credits` → **new** `comfy_credits` lane; slug from recommended default.
- `Auto / Best Available` → **run-time** probe order: Comfy Credits (if free probe passes) → OpenRouter (`openrouter_enabled()`) → Local.

`Auto` must resolve at **execution**, never at `INPUT_TYPES`: Comfy login/credits is undetectable at load (only `openrouter_enabled()` — own-key env — is load-visible). The widget just says "Auto"; `INPUT_TYPES` stays zero-network.

## 3. Defaults + why

- **Creative = Auto / Best Available** — frontier quality where the story is written.
- **Technical = Local** — it makes *dozens* of small JSON/ledger/validation calls per episode; defaulting it to paid burns credits for no narrative gain. Never auto-flip technical to remote.

## 4. Current state — the 4-dropdown router is SHIPPED (this is a migration)

HEAD `3ff7692`, `local == origin`, full `tests/` **3366 pass / 0 fail**, forbidden sweep green. Shipped sprints: S0 `8239f3d` (cache), S1 `2c96c3b` (slot catalog builder + recommended constants), S2 `d5fe8c8` (slot A/B widgets + conditional default), S3 `579571a` (resolve_slug preservation), S4 `d43c9b0` (audit pin 2→4, refresh script, setup doc, PD6 amend). The pivot **reuses the backend and unwinds the 4-dropdown UI**.

## 5. Already built — REUSE, do not rebuild (verified refs)

- **Cache + refresh** (S0): `models/openrouter_models.json`; `scripts/otr_openrouter_refresh.py`. Zero-network `INPUT_TYPES`.
- **`resolve_slug` 3-tier preservation** — `_otr_openrouter_backend.py` ~L168-214 (bound slug verbatim + warn-if-stale, never swap; unbound → `OTR_OPENROUTER_SLOT_x_DEFAULT` → `OPENROUTER_MODEL_x` → `recommended_slug_for_slot()` → error).
- **Recommended slugs** (answers old open-Q): `OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT="anthropic/claude-opus-4.8"`, `..._TECHNICAL_DEFAULT="deepseek/deepseek-v4-pro"` (~L63); `recommended_slug_for_slot()` ~L365.
- **Provider seam** — `_otr_model_runtime.py` `BACKENDS_BY_KEY` L153 + `get_backend_for_row()` (dispatches on `row.loader_backend`); `_otr_loader_backends.py`; provider tag set in `OpenRouterBackend.load()` ~L669. `comfy_credits` registers here as a new `loader_backend` key.
- `openrouter_enabled()` ~L143; fail-closed `OpenRouterCallFailedError` ~L108; `provider.sort` routing L218-308; cost guards (run 300k / per-call 32768); conditional creative default `_creative_default` (writer ~L1527/L1575).
- `set_slot_bindings()` ~L350 (kept; just no longer fed by widgets — slot_bindings stay unset → fallback chain provides the slug).

## 6. Migration sprints (each ends with the §7 loop)

| # | Type | Change |
|---|------|--------|
| **M1** | UI swap + unwind | `creative_writing_model`/`technical_model` values → provider modes (§1). **Remove** slot widgets `openrouter_slot_a_model`/`b` (writer L1219-1220, 1390-1391, 1897-1914, 2033-2034). Keep the conditional creative default. **Unwind 4-dropdown residue:** workflow `otr_scifi_16gb_full.json` node-1 wv **21→19**; `tests/test_b6_wiring_guardrails.py` `_MODEL_WIDGET_KEYS` **4→2** (L50); length gates (companions ~L418, guardrails ~L666) + `_writer_schemas`/`_writer_node_fixture` 21→19; revert CLAUDE.md PD6 **4→2**; rewrite `docs/openrouter-setup.md` for the 2-dropdown. |
| **M2** | Add (real new work) | `comfy_credits` lane: new `BACKENDS_BY_KEY` key (`loader_backend="comfy_credits_http"`) + a `ComfyCreditsBackend` + the `~L622` provider branch + a generate-fn over ComfyUI hidden-auth (`auth_token_comfy_org`/`api_key_comfy_org`) + `comfy_api` client. **Isolate behind the seam** (fragile internal API → a break degrades Auto to openrouter/local, never crashes). |
| **M3** | Add | `Auto / Best Available` execution-time resolution (comfy_credits→openrouter→local). Defaults: creative=Auto, technical=Local. |
| **M4** | Add | Failure handling: early (probe/first validated call) → **whole-run local fallback + clear warning**; deep/mid-episode → **fail loud, never splice**; never remote→remote swap. First paid-run warning: "Creative LLM will use paid remote calls. Technical stays local." |
| **M5** | Add | Script-lock: `IS_CHANGED` returns `time.time()` today (writer ~L1947, FreezeCascade ~L266) → every run re-calls/rebills. Add a reuse/lock path so re-rendering FLUX/HuMo/LTX reuses cached `script_json` (no re-call, no re-randomize). |
| **M6** | Gates/docs | Forbidden sweep + b6 green at 2 widgets; setup doc done; full regression; workflow JSON re-wired + validated. |

## 7. The loop (every migration sprint)

**code → wire into the workflow JSON → run Bug Bible + core + dropdown + forbidden sweep → commit.** Nothing is "done" until all gates are green AND the workflow JSON matches the node surface. Commit via Desktop Commander **cmd** (`.git\COMMIT_EDITMSG` + `git commit -F`), one atomic commit per sprint. **Use DC/file-tools for git — sandbox bash git is stale.** Do not commit until the sprint's gates pass.

## 8. Hard rules (carried)

- `INPUT_TYPES()` zero network; `Auto` probes only at execution.
- Local always works offline/free — now an **option, not a commandment** (2026-06-01 relaxation), but it stays the universal fallback (only path that runs with no account/credits/network).
- No surprise charges: first paid-run warning + prepaid + cost guard. No silent remote→remote swap; no mid-episode splice.
- PD1 audio untouched; PD2 VRAM; PD5 no "dummy"; **PD6 reverts to two writer model widgets** (no amendment).

## 9. Open questions

1. **Codify the local-first relaxation in CLAUDE.md?** It's currently a verbal decision; the global Platform rule still says "100% local / no paid." Global (all your projects) vs OTR-only — unresolved.
2. **`comfy_credits` probe must be free** (account/credits check, not a billed generation); confirm + internal-API stability across ComfyUI updates.
3. **Recommended slugs** already set (`claude-opus-4.8` / `deepseek-v4-pro`) — confirm still live on openrouter.ai/models.
4. **Script-lock UX** — explicit "lock" toggle vs automatic `IS_CHANGED` input-diff.
5. **Provider knobs** — decided NOT to add (`require_parameters`/`max_price`/etc.); keep lean. `require_parameters` is a future one-line lever only if remote-technical JSON ever flakes.

## 10. Out of scope

The 2-dropdown router + cache + preservation, text-only writer. No per-model dynamic inputs/vision slots, image-gen, chat mode. Topology items unchanged: **VEO3 swap rejected** (breaches local/offline/no-API + reproducibility); **audio-gated parallelism** real but VRAM-capped (PD1: don't break audio-timing deps); **upscale→blend reorder likely intentional** (`56 RTXUpscale → 58 PostUpscaleProcgenBlend`, captions burn at 58 — A/B, not a bug fix).

---

*Go-forward plan rev 2 — uncommitted, v2.0-alpha. Reconciled to the shipped 4-dropdown (HEAD 3ff7692); this is the migration to the 2-dropdown provider router. Pairs with `session_handoff.md`.*
