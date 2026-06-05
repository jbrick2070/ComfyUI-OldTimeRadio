# OpenRouter Model Router — Sprint Plan (lean, v5)

> ## ⚠ SUPERSEDED (2026-06-01) — UI replaced by the 2-dropdown provider router
> The **four-dropdown** surface in this doc (creative/technical slot handles + `openrouter_slot_a/b_model` slug pickers) is **superseded** by `2026-06-01-otr-llm-provider-router__go-forward-plan.md`: **two** dropdowns (`Creative LLM`, `Technical LLM`), each resolving to a provider (`local`/`openrouter`/`comfy_credits`). **Build the 2-dropdown design, not this one.** The *internals* below (disk cache, preservation of explicit picks, cost guard, no-network-in-INPUT_TYPES) are **retained** and still apply.

**Branch:** `v2.0-alpha` · **Date:** 2026-06-01 · **Status:** SUPERSEDED (internals retained; UI replaced)

**Scope:** the writer model router expands from two dropdowns to four:
`creative_writing_model`, `technical_model`, `openrouter_slot_a_model`, `openrouter_slot_b_model`.
**Intentional node-surface change limited to `OTR_LedgerScriptWriter`** — amends Prime Directive 6 (§8).

**Lineage:** v2 dynamic catalog · v3 conditional default · v4 4-dropdown router · **v5 preservation hardening** (no silent model swap under a saved workflow).

---

## 0. Goal

- `creative_writing_model` / `technical_model` choose an **execution slot**: a local model, `openrouter:slot-a`, or `openrouter:slot-b`. (Short list — no catalog.)
- `openrouter_slot_a_model` / `openrouter_slot_b_model` choose the **actual OpenRouter slug** from the cached catalog.

A fresh node boots safely: `creative_writing_model` defaults to `openrouter:slot-a` when remote is enabled, else local Mistral-Nemo. `INPUT_TYPES()` does zero network.

**Equivalence claim (corrected):** offline / remote-disabled **execution behavior stays equivalent**; the node surface changes by appending two inert OpenRouter slot widgets. Separately, the **audio output stays byte-identical** (PD1) — this sprint never touches the audio path.

---

## 1. Where it plugs in

- `nodes/_otr_openrouter_backend.py` — fetch/cache catalog; resolve slot handles from explicit per-run slot bindings; env fallback preserved for old workflows / headless.
- `nodes/_otr_model_catalog.py` — keep the local dropdown builder for creative/technical; add `openrouter_catalog_dropdown_choices()` for slot A/B; keep `openrouter:slot-a/b` virtual rows in the creative/technical selectors when remote enabled.
- `nodes/OTR_LedgerScriptWriter.py` — append two optional widgets at the END; resolve + stamp all four; pass slot bindings to backend/meta.
- `workflows/otr_scifi_16gb_full.json` — gains two trailing widget values on node 1 (surface change; §8).

---

## 2. Hard rules (every sprint)

- **`INPUT_TYPES()` never touches the network.** All four dropdowns build from the disk cache only.
- **Append, never insert.** New widgets go at the END of the optional block so existing indices `[0..18]` are unchanged ("Order is load-bearing — saved workflows bind by widget index"). Old workflows load with defaults for the new slots.
- **Only empty/unset slot values fall back to defaults.** An **explicit saved slug is preserved and attempted**, warned if absent from the cache. A **failed selected call is a hard error — never a silent swap** to another remote model. (This is the central rule; see §5.)
- **Missing/corrupt cache governs discovery only.** It must never mutate a saved slot slug value, and re-saving a workflow with a cold cache must not rewrite those values.
- **Placeholder choices are UI-only sentinels.** `(enable OpenRouter)` and friends are rejected before backend resolution and can never resolve as a slug; they are treated as empty/disabled.
- **Slot picks are passive bindings.** `openrouter_slot_a/b_model` do **not** activate a remote call by themselves — they only bind a handle if `creative_writing_model` / `technical_model` selects `openrouter:slot-a/b`.
- **Conditional default; local is the backstop.** `creative_writing_model`: remote OFF → local `DEFAULT_LLM`; remote ON → `openrouter:slot-a`. `technical_model`: local `DEFAULT_LLM` unless explicitly overridden — never auto-flipped to OpenRouter. `choices[0]` on both is always local.
- **Cache staleness visible.** Log on refresh + at dropdown build: `OpenRouter catalog: N models, refreshed <ISO ts>, source=live|cache|stale`; stamp into run meta.
- **PD6 amended, intent preserved.** Writer pickers go 2 → 4; no non-writer node gains a model pick; creative/technical remain the two routing tags. Forbidden sweep + CLAUDE.md PD6 update in the same commit (§8).
- **PD1 / PD2 / PD5:** audio path untouched; remote rows zero local VRAM; no "dummy" naming. Run Bug Bible + core + dropdown tests after every change.

---

## 3. Current state vs delta

**Today:** two writer model widgets; `openrouter_enabled()` appends two virtual rows (`slot-a/b`) bound to `OPENROUTER_MODEL_A/B` via env; default `DEFAULT_LLM` unconditional; no catalog fetch.

**Delta:** add two slot-slug picker widgets; move catalog into them; make creative/technical pure slot selectors; resolve slots from the new widgets (env = fallback); conditional default; hardened cache; **preservation rules** so a saved workflow's chosen model is never silently swapped. Node surface + shipped JSON change.

---

## 4. Dropdown contract

**`creative_writing_model`** — 1) `DEFAULT_LLM` (index-0 backstop) · 2) other local models · 3) `openrouter:slot-a` · 4) `openrouter:slot-b`
**`technical_model`** — 1) `DEFAULT_LLM` · 2) local technical models · 3) `openrouter:slot-a` · 4) `openrouter:slot-b`
**`openrouter_slot_a_model`** — 1) recommended creative default · 2) favorites · 3) recent · 4) full cached catalog (filtered)
**`openrouter_slot_b_model`** — 1) recommended technical default · 2) favorites · 3) recent · 4) full cached catalog (filtered)

The catalog never appears in the creative/technical dropdowns. Remote disabled → slot A/B show a `(enable OpenRouter)` sentinel (pure-Python; true hide/disable = optional `web/` JS).

**Default policy (new node only — saved `widgets_values` always win; defaults apply only to empty/unset widgets):**

- `creative_writing_model`: remote disabled → `DEFAULT_LLM`; remote enabled → `openrouter:slot-a`.
- `technical_model`: `DEFAULT_LLM` unless explicitly overridden — no auto-flip.
- `openrouter_slot_a_model`: `OTR_OPENROUTER_SLOT_A_DEFAULT` if set + present, else `OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT`.
- `openrouter_slot_b_model`: `OTR_OPENROUTER_SLOT_B_DEFAULT` if set + present, else `OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT`, else the general recommended default.

**Optional filters (slot A/B only; filters, never a cage):** `OTR_OPENROUTER_MODEL_ALLOWLIST`, `OTR_OPENROUTER_MODEL_DENYLIST`, `OTR_OPENROUTER_PROVIDER_FILTER`, and **per-slot** `OTR_OPENROUTER_SLOT_A_REQUIRE_JSON` / `OTR_OPENROUTER_SLOT_B_REQUIRE_JSON` (structured-output filter — defaults off for A, the slot you'd set is B/technical; never global, so a creative model is never hidden from A because B needs JSON).

---

## 5. Backend slot resolution (the preservation core)

`openrouter:slot-a` → `openrouter_slot_a_model`; `openrouter:slot-b` → `openrouter_slot_b_model`. Resolution runs at execution on the **stored widget string**, independent of the current dropdown choices. Three distinct cases:

1. **Empty / unset / placeholder sentinel** → fallback chain: `OTR_OPENROUTER_SLOT_x_DEFAULT` → `OPENROUTER_MODEL_x` (env fallback) → recommended default → clear config error.
2. **Explicit saved slug present** → **use it as-is**; if absent from the current cache, **warn and still attempt it** (cache is stale ≠ model gone). No fallback substitution.
3. **Selected call fails** → **hard error** recommending slot-a/b. No silent remote→remote swap.

A placeholder sentinel is never a slug. Missing/corrupt cache never rewrites a saved slug.

---

## 6. Sprints

| # | Change | Files | Gate |
|---|--------|-------|------|
| **S0** | Cache + refresh primitive: hardened `models/openrouter_models.json` (`schema_version, fetched_at, source, count, models[]`), atomic write, corrupt/offline → safe empty. `INPUT_TYPES` never fetches. | `_otr_openrouter_backend.py` *(new)* | self-test: fresh/stale/missing/corrupt/offline → safe, never raises/blocks |
| **S1** | Split dropdown builders: creative/technical = local + `slot-a/b` only; `openrouter_catalog_dropdown_choices()` for slot A/B (catalog + favorites + recent + filters + `(enable OpenRouter)` sentinel when disabled). | `_otr_model_catalog.py` | extend `tests/test_openrouter_catalog_rows.py`: catalog absent from creative/technical; per-slot filters; sentinel present when disabled |
| **S2** | Four-widget surface: append slot A/B widgets at END; update `_resolve_inputs`; conditional defaults (§4); update shipped JSON; **migration test**. | `OTR_LedgerScriptWriter.py` + `workflows/*.json` | self-test: 4 widgets resolve; defaults per policy; **load old node-1 `widgets_values` (no slot entries) → defaults supplied, NO existing value shifted** |
| **S3** | Resolution + preservation (§5): 3-case logic; placeholder rejected; saved slug preserved + warned; failed call hard-errors; env demoted to fallback; stamp resolved slugs + cache meta. | `_otr_openrouter_backend.py` `resolve_slug`, writer `_resolve_inputs` | self-test: empty→fallback; **saved slug absent from cache → preserved + warned + attempted, not swapped**; placeholder never resolves; failed call → error |
| **S4** | Tests / docs / sweep: update `docs/_s28_forbidden_sweep.py` to allow exactly the two new slot widgets; update CLAUDE.md PD6; JSON check for four model widgets; update `docs/openrouter-setup.md`. | sweep + CLAUDE.md + docs | forbidden sweep green (4 allowed picks); Bug Bible + core + dropdown pass |

---

## 7. Test matrix (invariants)

**remote disabled:** creative/technical show local only; slot A/B show the sentinel; creative + technical default `DEFAULT_LLM`; 0 network from `INPUT_TYPES`.

**remote enabled + fresh cache:** creative/technical include `slot-a/b`; slot A/B show the catalog; slot A default recommended-creative, slot B recommended-technical; 0 network.

**preservation (the point of v5):**
- saved slug **present** in cache → used as-is
- saved slug **absent** from cache → **preserved + warned + attempted**, never swapped
- `(enable OpenRouter)` sentinel → rejected before resolution, never a slug
- missing/corrupt cache + re-save → saved slug values **unchanged**
- *selected* remote call fails → hard error recommending slot-a/b; no remote→remote swap

**migration:** old node-1 `widgets_values` (no slot A/B) → defaults supplied, existing values unshifted; indices `[0..18]` intact.

**defaults / routing:** `technical_model` stays local unless chosen; slot A/B slug picks do nothing unless creative/technical points at the handle; saved widget values win over defaults.

**filters:** per-slot `REQUIRE_JSON` narrows only its slot; allow/deny/provider narrow slot lists; local + slot-a/b never removed.

**meta stamped:** four model selections + `slot_a_resolved_slug` + `slot_b_resolved_slug` + catalog `source/fetched_at/staleness`. Forbidden sweep green.

---

## 8. Wiring / JSON + PD6 (surface change)

- **Surface change:** two optional widgets appended to the writer. Update `workflows/otr_scifi_16gb_full.json` node-1 `widgets_values` with two trailing values; verify `[3]/[4]` unchanged and `[0..18]` still bind. Shipped workflow keeps local pins; conditional default affects only freshly-dropped nodes.
- **PD6 amendment (same commit):** `docs/_s28_forbidden_sweep.py` allowlist expands 2 → 4 named writer model widgets; CLAUDE.md PD6 text updated to match, rationale recorded (the two new pickers are writer-only opt-in slug bindings; no non-writer node gains a pick). Auditable via the wiring test pin.

---

## 9. Acceptance

- Headless gates green (full `tests/`, self-tests, forbidden sweep with 4 allowed picks).
- Offline / remote disabled: no hang; creative + technical default local; pipeline works with no key.
- **Remote enabled:** freshly-dropped writer can default `creative_writing_model` to `openrouter:slot-a`; `openrouter_slot_a_model` **visibly shows the actual slug**; `openrouter_slot_b_model` shows the secondary slug; `technical_model` stays local unless chosen.
- A saved workflow's chosen slug is **never silently changed** by a stale cache; a genuinely failed call errors clearly.
- `docs/openrouter-setup.md` documents the router, conditional default, preservation rules, cache, filters, refresh script.

---

## 10. Open questions

1. **COMBO out-of-list display (new risk).** Resolution preserves a saved slug robustly (§5, on the stored string), but does the ComfyUI **frontend** keep an out-of-list saved combo value at load, or reset it before the backend sees it? Confirm the version behavior; if it resets, a small `web/` JS shim (or storing the slug in a way the loader can't drop) is needed to keep preservation airtight at the UI layer. The migration + preservation tests must cover this.
2. **Recommended-default slugs** — values for `OPENROUTER_RECOMMENDED_CREATIVE_DEFAULT` (Opus?) and `OPENROUTER_RECOMMENDED_TECHNICAL_DEFAULT` (cheaper, reliably-JSON), and refresh ownership. The only drift-prone constants.
3. **Favorites / recent tier** — `OTR_OPENROUTER_FAVORITES` ∪ newest-by-`created`, or a shipped set? *Lean: env + recency.*
4. **Cache location** — in-repo `models/` (git-ignored) vs `C:\ComfyUI-Models`. *Lean: in-repo.*
5. **`REQUIRE_JSON` source** — confirm OpenRouter exposes a reliable per-model structured-output flag for `supports_json`.

---

## 11. Out of scope

**This sprint:** the 4-dropdown router + cache + preservation, text-only writer. No per-model dynamic inputs / vision slots, image-gen, or chat mode. (A true hide/disable for slot dropdowns, and any UI shim for out-of-list preservation, are the only things that would touch `web/` JS.)

**Topology criticism — NOT this sprint (verdicts for the record):**

- **Replace LTX/HuMo with VEO3 — reject.** Different product architecture; cloud cost, reproducibility/control loss, dependency risk, breach of 100%-local / offline / no-API directives.
- **Audio-gated parallelism — real but capped** by the 14.5 GB VRAM ceiling, not just dependencies; PD1 forbids breaking deps needing final audio timing (HuMo lip-sync).
- **Upscale → blend reorder — likely intentional.** `56 RTXUpscale → 58 PostUpscaleProcgenBlend`, captions burn at 58; grain/overlays composited at final res on purpose. A quality A/B, not a bug fix.

---

*Draft for review — uncommitted, `v2.0-alpha`. v5 hardens the router so a saved workflow's chosen model is preserved and never silently swapped: only empty/unset slots fall back; explicit saved slugs are kept + warned; failed calls fail loud.*
