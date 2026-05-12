# OTR Roadmap

**Branch:** `v2.0-alpha` | **Owner:** Jeffrey A. Brick | **Stack head:** `5d7e887` | **Last refactored:** 2026-05-09/10 (LPL sprint in flight)

This file is the **canonical going-forward plan**. Forward-only. Historical session logs and "what shipped" archives are in `docs/ROADMAP_HISTORY.md`.

**Format codename:** the new structured-ledger contract is **L3** (matches `schema_version: "l3-2026-05-08"` already on the wire). All consumer rewrites target L3-native reads + L3-native write-back via `patch_line_fields(led, line_id, {...})`. When the schema bumps next, the codename evolves cleanly to L4.

---

## STANDING DIRECTIVE — NO LEGACY BACK-COMPAT (Jeffrey, 2026-05-11)

OTR v2.0 is **greenfield**. The project is being written front-to-back as a single rigorous workflow. We are NOT preserving compatibility with legacy nodes, legacy class names, legacy field names, or legacy on-disk shapes from any pre-v2.0 / pre-LFC state.

**Rule for every contributor (human + AI):**

- When a node is renamed, **delete** the old name. Do not ship a re-export shim.
- When `_RENAME_ALIASES` (or equivalent) gets a new entry, **delete the entry** along with the rename. The old workflow JSON is expected to be updated to the new class name.
- When a meta field gets a new canonical name, **delete the old key**. Do not stamp both names "to keep legacy consumers working".
- When a phase / pass is replaced, **delete the old function** and every test that pinned the old contract. The new contract is the only contract.
- When an output socket renames, **don't carry the old name in the JSON shape**. Update every consumer.

**Audit hits as of HEAD `302b839`** that need to be pruned in a follow-up commit (call it `commit 12.3 — legacy prune`):

- `nodes/OTR_LedgerScriptReviewer.py` re-export shim → DELETE the file.
- `__init__.py` `_RENAME_ALIASES["OTR_LedgerScriptReviewer"]` entry → REMOVE the line.
- `__init__.py` `_RENAME_ALIASES["OTR_Gemma4Director"]` entry → REMOVE the line (the legacy Gemma director name is dead).
- `nodes/OTR_LedgerFreezeCascade.py::_no_ledger_error_json` stamps `"reviewer_verdict"` alongside `"freeze_verdict"` with comment "Legacy field kept so consumers still keyed on the old name see the same signal" → DROP the legacy field; downstream nodes should read `freeze_verdict` only.
- Any test asserting that `OTR_LedgerScriptReviewer` (old name) still resolves to a class → DELETE the assertion.
- Workflow JSON `widgets_values` defaults that exist purely to "match the pre-rename legacy" → review; the JSON should be the single canonical surface, written from scratch if needed.

**Acceptance criteria for `commit 12.3 — legacy prune`:**

1. `grep -rn "OTR_LedgerScriptReviewer" nodes/ __init__.py` returns ZERO hits outside the legacy-rename log entry in BUG_LOG.md and the ADR text.
2. `grep -rn "Gemma4" nodes/ __init__.py` returns zero hits.
3. `grep -rn "reviewer_verdict" nodes/` returns zero hits.
4. The workflow JSON loads in ComfyUI Desktop with NO missing-node warnings or back-compat aliases firing.
5. Bug Bible regression holds 23/1/2xf.

The legacy-prune is its own commit so the diff stays small + auditable. Defer it to a fresh session — context-heavy sessions tend to put the legacy shims back if they aren't pruned in a clean pass.

---

## CURRENT WORK — news_interpreter sprint (all 5 commits SHIPPED 2026-05-10) — COMPLETE

**State:** Sprint complete on `v2.0-alpha`. Commits: `6f3218d` (ADR + canary tests), `70d25eb` (agnostic module + GBNF grammar), `f518fb3` (writer wiring + cast + outline + schema bump `l3-2026-05-14` + canary case 12 flipped), `9f82685` (announcer closing-line override + post-assembly key_terms audit + 13 new wiring tests), `4f45c7c` (era literals stripped + 5 text-scan canaries flipped, originally shipped at `92e58e5` with wrong subject and force-amended). Module is strictly LLM-agnostic — `generate_fn(messages, *, temperature, max_new_tokens) -> str` only, no model branches. End-to-end pipeline: RSS → full article dict → `build_news_briefs` (one LLM call, 4 outputs) → `meta.news` → cast prompt + outline prompt + announcer closing line + post-assembly key_terms audit. Bug Bible 15p/2x/1s baseline held across every commit. Two canaries remain armed as out-of-scope future-ADR work (RADIO portrait + MusicGen cues per ADR section 1). The downstream prompt audit (`outputs/downstream_prompt_audit.html` artifact) identified 5 hardcoded era-literal violations across `script_critic.py` + `story_orchestrator.py` and a structural gap where downstream consumers never see the news article body. Round-robin synthesis (ChatGPT gpt-5.5 + Gemini 3.1 Pro + NVIDIA) converged on a unified 4-output news_interpreter LLM stage inserted between style-resolve (D.2) and cast-lock (D.3) in `OTR_LedgerScriptWriter`. Canonical ADR at `docs/news_interpreter_adr.md`.

### Architecture (locked)

- **One unified LLM call** emits `casting_brief` (≤200ch), `script_brief` (≤350ch), `news_close_brief` (≤250ch), `key_terms` (2-6 entries, ≤40ch each).
- **Input cap:** `headline + " " + summary + first 1500 chars` of body; on bodies >2500 chars also append last 500 chars with explicit `[BODY_GAP truncated N chars]` marker (inverted-pyramid front + closing-graf tail).
- **Source wrapper** marks article body as inert via `[SOURCE_BEGIN]` / `[SOURCE_END]` with `INERT SOURCE MATERIAL` preamble (prompt-injection defense).
- **GBNF grammar** required at commit 2 (small-model JSON reliability — Mistral-Nemo + Gemma both support `--grammar-file`). Structural enforcement; pydantic + validators handle semantic checks.
- **Validators (source-context allowance):** V1 word-boundary `key_terms` match against `headline + summary + cleaned_body`. V2 rejects period literals only when absent from source. V3 rejects formulaic style phrasing (`in a noir style`, `noir-style`, `make this into a noir`) not bare style-word occurrence.
- **Cache key:** `sha256(source_hash + style + prompt_version + schema_version + model_id + decoder_profile + seed)`. Stored at `ledger.meta.news.cache_key`. Any change to any field → cache miss → regenerate.
- **Determinism contract narrowed:** byte-identity is a fixture-test claim only. Live model calls assert schema validity + contract preservation, not byte identity. Documented in ADR section 3.5.
- **Python stamps** `source_hash`, `model_id`, `attempts`, `attempt_failures` on `meta.news`. LLM does not author its own metadata.
- **Post-assembly key_terms check** runs after line composer at `min_required=2`. Zero terms landed → hard fail + repair pass. Some missing (≥2 landed) → warn and proceed.

### Commit order — safety net first

| # | Commit | State | Hash | What |
|---|---|---|---|---|
| 1 | ADR + xfail-strict canary tests | **SHIPPED** | `6f3218d` | `docs/news_interpreter_adr.md`, `tests/test_news_interpreter.py` (12 unit tests, importorskip dormant), `tests/test_downstream_prompt_contract.py` (8 xfail-strict canaries: 5 text-scan against existing era literals + 3 integration placeholders for commits 3-4). Locks the API surface before any code that satisfies it. |
| 2 | news_interpreter module | **SHIPPED** | `70d25eb` | `nodes/news_interpreter.py` (~700 LOC): NewsBriefs pydantic v2, V0/V1/V2/V3 validators with source-context allowance, build_source_wrapper, compute_cache_key, extract_json_block, build_news_briefs with 3-attempt T=0.7/0.8/repair@0.3 ladder. `grammars/news_interpreter.gbnf` (~30 lines) shipped loader-side, not passed by module (agnostic surface). 12/12 unit tests pass. Production 2-6 key_terms bound enforced at orchestration layer (V0); schema accepts 1-6 so V1/V2/V3 isolate cleanly. Schema bump to production_ledger.py deferred to commit 3 alongside writer wiring. |
| 3 | wire into writer/cast/outline | **SHIPPED** | `f518fb3` | `OTR_LedgerScriptWriter._fetch_rss_seed_or_die` returns full article dict (was string); `full_text` no longer discarded. New D.2.5 between style-resolve and cast-lock calls `build_news_briefs()` and stamps `meta["news"]`. Graceful degrade (warn + fall back to raw news_seed) on build_news_briefs failure. `_otr_casting` + `_otr_outline` gain additive optional kwargs (`casting_brief`, `script_brief`, `key_terms`) defaulting to empty so existing fixtures preserve behavior. Schema bumped `l3-2026-05-08` → `l3-2026-05-14`. Canary case 12 flipped from xfail-strict to PASSED in lockstep. |
| 4 | wire announcer + post-assembly | **SHIPPED** | `9f82685` | New `nodes/_otr_news_wiring.py` with `override_announcer_close` + `post_assembly_keyterm_check` helpers. Writer I.5 section runs after per-beat loop: stamps `news_close_brief` onto LAST announcer line; word-boundary audits each `key_term` across voiced lines; stamps `meta["post_assembly_key_terms"]` diagnostic. 13 new wiring tests. ADR deviation tracked: zero-terms-landed ships warn-only; targeted repair pass deferred to follow-up. RADIO portrait + MusicGen canaries reason-text updated to point to future ADR (out of sprint scope per ADR section 1). |
| 5 | strip era literals | **SHIPPED** | `4f45c7c` (amend of `92e58e5`) | `script_critic.py:330,339-340,556` stripped of "1940s setting" / "1940s-style" / "You are revising a 1940s ..." literals. `story_orchestrator.py:_LTX_STYLE_BRIEF_PROMPT` (lines 3394-3411) fully rewritten per ADR section 7.4 Option A — three style-spanning examples (near-future newsroom / deep-space vessel / rust-belt industrial decay) replace the three baked vacuum-tube anchors. 5 xfail-strict text-scan canaries flipped to PASSED with markers removed in lockstep per the canary mechanic. Originally shipped with wrong subject (cmd-chain stale COMMIT_EDITMSG anti-pattern from CLAUDE.md); force-amended with Jeffrey's OK. |

### A/B sanity check (before merging v2.0-alpha to `main`)

Run 10 episodes through old path + 10 through new path with the same seeds. Eyeball cast diversity (gender balance, role-fit, archetype spread). ~30 min subjective scoring. Catches the category of regression unit tests won't.

### Deferred follow-ups (post-sprint)

Tracked here per project rule — deferrals live in ROADMAP, not in sidecar docs. No separate punch-list document to delete later.

| # | Item | Why deferred | Tracking signal |
|---|---|---|---|
| D1 | **Targeted repair pass when zero `key_terms` land in dialogue.** ADR section 4.4 canonical policy is hard-fail + re-compose the line whose intent is closest to the missing term's topic. | Commit 4 shipped warn-only — alpha-branch pragmatism, episodes still ship. | `meta.post_assembly_key_terms.repair_pass == "deferred"` in every produced ledger. Flip to `"v1"` (or whatever scheme) when the pass lands. |
| D2 | **Future ADR — audio-plane (MusicGen cues).** `nodes/musicgen_theme.py:52-74` still hardcodes "1940s old time radio" as the opening / closing / interstitial cue defaults. Should read `ledger.meta.gen_params_initial.style` (and optionally `meta.news.script_brief` for mood signal). | ADR section 1 explicitly OUT OF SCOPE — "MusicGen cues land in their own ADR once narrative plane is stable." Narrative plane is now stable. | xfail-strict canary `test_musicgen_does_not_default_to_period_cues` in `tests/test_downstream_prompt_contract.py`. Flips to XPASS the moment the fix lands. |
| D3 | **Future ADR — FLUX character portraits (RADIO portrait fallback).** `scripts/render_flux_batch.py:266` falls back to a hardcoded `"vintage 1940s console radio"` string when `cast["RADIO"].character_description` is empty. Should hard-fail (the cast contract guarantees the field is populated) or read style. | ADR section 1 OUT OF SCOPE — "FLUX character portraits land in their own ADR." | xfail-strict canary `test_radio_portrait_empty_char_desc_hard_fails`. Flips on fix. |

### Round-robin transcripts

- Question brief: `outputs/news_interpreter_question.md`
- Synthesis ADR: `docs/news_interpreter_adr.md`

### Hard rules (locked, never violated this sprint)

- **LLM-agnostic control plane.** No Mistral / Gemma / Qwen branches in news_interpreter. Proxy-test against gemma-2-2b-it first.
- **Lean prompts.** Prompt body ≤250 tokens. `max_new_tokens=400` (not 250) to leave safety margin on full payloads.
- **No hardcoded period literals.** Anywhere. Code, comments, prompt strings, test fixtures.
- **C7 byte-identity** within fixture tests (mocked `generate_fn`). Live runs assert contract preservation only.
- **14.5 GB VRAM ceiling.** Validator + reroll is the safety net, not prompt cleverness.
- **UTF-8 no BOM.** No edits to `_otr_outline.py` or `_otr_canon.py` until commit 3 (already-locked v2.0 modules).

---

## PREVIOUS SPRINT — Ledger Consumer Rewrite sprint (shipped green 2026-05-09/10)

**State:** **7 of 7 consumers shipped green.** Patterns doc folded into ROADMAP under "L3 contract — patterns lock-in" below; standalone `docs/2026-05-09-ledger-consumer-rewrite-patterns.md` archived to `docs/ROADMAP_HISTORY.md` and deleted. Bug Bible 23/1/2/0 baseline held across every consumer ship. EpisodeAssembler audited clean (no rewrite needed). No commits, no pushes — working-tree only until soak proves out. Next: video pipeline recon (Flux/HuMo/LTX/VideoComposite, post-consumer #7) + B4 LLM prompt audit + fresh workflow JSON wiring + dry-run gates → STOP, hand to Jeffrey for soak ramp.

### Phase 3 writer extraction — SHIPPED

| What | Where |
|---|---|
| `LegacyLLMScriptWriter` extracted to dedicated module | `nodes/_otr_legacy_writer.py` (~6 KLOC, class span 5,877 lines, 28 methods) |
| `story_orchestrator.py` class span deleted, replaced with PEP 562 lazy `__getattr__` shim | `nodes/story_orchestrator.py` |
| `OTR_LedgerScriptWriter` registered alongside; old `OTR_LLMScriptWriter` repointed at extracted module with display name `Story Writer (legacy)` | `__init__.py` |
| 5/5 gates green: schema validation, AST parse, workflow binding (15 saved widgets vs 17 current — trailing-default drift acceptable), Bug Bible regression baseline match, legacy self-test 5/5 | `tests/_phase3_schema_gate.py` + `tests/_phase3_workflow_gate.py` |

### L3 helper module + patterns doc — SHIPPED

| Artifact | Purpose |
|---|---|
| `nodes/_otr_ledger_consumers.py` | Read-side helper: `load_ledger`, `iter_lines`, `cast_lookup`, `speaker_name`, `voice_preset`, `production_plan_or_empty`. ~140 LOC including type hints + docstrings. Strict by design — `load_ledger` raises named `ValueError` on legacy parser-list shape. |
| `nodes/_otr_ledger.py` (existing, no edits) | Write-side surface: `in_flight_ledger_path`, `patch_line_fields`, `save_ledger_safe`, `stamp_per_line_audio_meta`, `audio_gate_record`, `record_phase_ms`, `set_meta`. Unchanged; consumers wire to it for write-back. |
| `_otr_ledger.patch_line_text(led, line_id, text)` helper | Atomic update of `text` + `char_count` + `word_count` to prevent metric drift on REVISE passes. Mandatory at every text-mutation site. |
| `tests/fixtures/__init__.py`, `tests/fixtures/ledger_stub.py` | `make_stub_ledger(...)` + `make_legacy_list()` shared fixtures for the 7 per-consumer self-tests. Cast modeled as list-of-dicts to match `production_ledger.set_cast` output. |
| `docs/2026-05-09-ledger-consumer-rewrite-patterns.md` | ~7 KB, 7 patterns + anti-patterns + cross-consumer status table. The canonical reference for consumers 4-7 (and for any future schema-evolving work). |

### Consumer rewrite progress — 7 of 7 SHIPPED GREEN

| # | Consumer | Status | Self-test | Bug Bible | Notes |
|---|---|---|---|---|---|
| 1 | `script_critic.py` | **SHIPPED** | 4/4 PASS | 23/1/2/0 | Meta-stamper pattern. Legacy-list ValueError → PASS passthrough w/ `meta.critic_skipped_reason="legacy_list_input"` (Critic is non-blocking by policy). `meta.critic_verdict` augment alongside append-only `script_gates[]`/`script_revisions[]` history. |
| 2 | `batch_bark_generator.py` | **SHIPPED** | 4/4 PASS | 23/1/2/0 | Per-line stamper pattern. `roles={"character"}` only — announcer stays on Kokoro bus per file-comment design rationale ("ums and ahs out of bookends"). Duplicate-text canary test confirms line_id-based stamping where text-match would collide. |
| 3 | `kokoro_announcer.py` | **SHIPPED** | 4/4 PASS | 23/1/2/0 | `roles={"announcer"}` only. Pattern 5 N/A (Kokoro doesn't take `production_plan_json`). Test harness lesson: mock pipelines need `time.sleep(0.005)` so `render_ms = int(elapsed * 1000) > 0`. |
| 4 | `scene_sequencer.py` | **SHIPPED** | 4/4 PASS | 23/1/2/0 | No role filter on iter; `music_*` lines pass through unstamped (forensic log line). SFX are first-class lines in v2 ledger so the legacy `ledger.sfx[]` parallel-walk degrades to a no-op for v2 producers. line_id stamping for both dialogue and sfx. |
| 4b | `EpisodeAssembler` (in scene_sequencer.py) | **AUDITED CLEAN — no rewrite needed** | — | 23/1/2/0 | Uses in-flight ledger directly on disk via `load_ledger_safe` + `save_ledger_safe`. No wire `script_json` input. `start_s_space` shift from `"scene_audio"` to `"master_mix"` already structured. Out of consumer-rewrite scope; flagged here for inventory completeness. |
| 5 | `batch_audiogen_generator.py` | **SHIPPED** | 4/4 PASS | 23/1/2/0 | `roles={"sfx"}`. Cue text = `line["text"]` (no regex). Two-track write-back: legacy `ledger.sfx[]` parallel walk preserved (no-ops on v2 producers); NEW `ledger.lines[]` line_id stamping with `sfx_wav_path` + `sfx_engine="audiogen"` (sfx-specific names disambiguate from dialogue's `tts_engine`/`bark_wav_path` on the same `lines[]` array). Cache-hit path verified. |
| 6 | `batch_procedural_sfx.py` | **SHIPPED (Option 2)** | 4/4 PASS | 23/1/2/0 | Read+write rewrite. NEW write-back per architect Option 2 decision: wavs persisted to `<episode>/audio/sfx/proc_<sfx_type>_<line_id>.wav`, paths stamped on `ledger.lines[]` per line_id alongside `sfx_engine="procedural"`, `sfx_type`, `dur_s`. Matches AudioGen's contract for ledger audio-file inventory completeness. No cache layer (procedural is cheap + deterministic). Disk-write failure is best-effort — falls through to `sfx_wav_path=None`, AUDIO batch continues. Decision history: shipped Option 2 after Option 1 (read-only port) was reverted per scope flip. |
| 7 | `video_engine.py` (`SignalLostVideoRenderer`) | **SHIPPED** | 4/4 PASS | 23/1/2/0 | Meta-stamper pattern. Title chain (Path B): `led.meta.episode_title` → `led.meta.title` → `led.title` → widget → TIMESTAMP_LASTRESORT. `news_used` and `meta.news_seed.headline` intentionally NOT in chain (both surface news/outline content, not finished-script titles). HUD + treatment helpers refactored to take parsed `led` dict + `plan` dict. HUD becomes line-count-fidelity (single pseudo-scene); treatment becomes flat list (no `── SCENE` headers — v2 schema has no `scene_break`/`environment`/`pause` markers). `meta.procgen_path` stamp via `set_meta` after render. Title chain primary slot (`meta.episode_title`) not stamped today — see Post-soak follow-ups B1+B2 below. |

### Hard rules (locked, never violated this sprint)

- **Do not edit shipped v2.0 modules:** `_otr_outline.py`, `_otr_canon.py`, `_otr_line_composer.py`, `_otr_model_loader.py`, `OTR_LedgerScriptWriter.py`.
- **Do not touch:** `_load_llm`, `_unload_llm`, `_LLM_CACHE`, `_MODEL_CONTEXT_CAPS` in `story_orchestrator.py`.
- **Bug Bible 23/1/2/0 must hold** after each consumer ship.
- **UTF-8 no BOM.** No commits, no pushes, no branch switches.
- **Per-consumer scope:** parsing block + stamping block ONLY. INPUT_TYPES untouched except production_plan_json demoted required→optional. No widget renames, no reorderings (saved workflows bind by position). No new optional widgets. Existing field names preserved exactly on stamps.

### Sprint exit criteria

1. All 7 consumers shipped with 4/4 self-tests + Bug Bible 23/1/2/0 holding after each.
2. `tests/test_otr_ledger_consumers.py` covering the helper API w/ stub ledgers.
3. **B4 LLM prompt audit pass** (see release blockers below) — gated on items 1-2.
4. Fresh workflow JSON wired `OTR_LedgerScriptWriter → Critic → fan-out to 6 audio/video consumers → Flux → HuMo → LTX → VideoComposite → RTXUpscale → PostUpscaleProcgenBlend`.
5. Dry-run gates: workflow instantiation + binding resolution, NO GPU.
6. **STOP. Hand to Jeffrey for manual soak ramp** (30 → 100 → 200 → 340 words, full pipeline including video review of Flux/LTX/HuMo). Soak is NOT in dev scope.
7. After soak proves out: delete legacy writer (`_otr_legacy_writer.py`, `__getattr__` shim, `OTR_LLMScriptWriter` registration). Archive saved workflow as `workflows/legacy_archive/`. Re-run Bug Bible.

### Video pipeline recon (read-only confirmation, post-consumer #7)

After all 7 audio/critic consumers ship, recon the 4 video files (`batch_flux_render.py`, `batch_humo_render.py`, `batch_ltx_render.py`, `video_composite.py`). All read ledger from disk (not wire `script_json`), so they should "just work" with the L3 format. Confirm. If recon surfaces text-matching or list-index access on `ledger.lines[]`, write a per-file mini-spec; otherwise mark "AUDITED CLEAN, no rewrite needed" in the patterns doc cross-consumer status table (matches the EpisodeAssembler precedent).

### QA strategy (post-sprint, pre-soak)

Three tiers when consumers + audit complete:

1. **Mechanical (automated):** Hypothesis property-based testing + Pydantic schema validation at consumer boundaries. Catches edge cases the canonical 4-test pattern misses (empty cast, single-line episodes, unicode in text, lines with `start_s` already populated, etc.). Add as new test classes.
2. **Cross-cutting (AI):** Fresh-context Claude/Opus reads the 7 consumers + writer + patterns doc, gives a code review using the patterns doc as the roadmap. Captures the "obvious in hindsight" bugs nobody on the team can see anymore.
3. **Subjective (Jeffrey):** Soak ramps 30/100/200/340 words. No tool replaces this for audio drama vibes.

### L3 contract — patterns lock-in

This subsection folds in the 7 patterns + anti-patterns from the standalone `docs/2026-05-09-ledger-consumer-rewrite-patterns.md` (now archived to `docs/ROADMAP_HISTORY.md` and deleted). It's the canonical reference for any future schema-evolving work or new consumer added to the OTR pipeline. **Read this before opening any consumer.**

#### TL;DR — the four-line rewrite

For every consumer, the visible diff is confined to:

1. **Parsing block at the top of the function** — `json.loads(script_json) → list iteration → regex on content` replaced with `iter_lines(load_ledger(script_json), roles={...})` plus structured field reads.
2. **Stamping block where ledger writes happen** — `text_to_idx` text-matching replaced with `patch_line_fields(led_disk, line["line_id"], {...})`. Same field names as before.
3. **`production_plan_json` demotion** — required → optional with `default="{}"` in `INPUT_TYPES`, plus a default value on the function signature.
4. **One `save_ledger_safe` at the end** — single atomic write per consumer call.

Everything else stays bit-for-bit identical. No widget renames, no reorderings, no new optional widgets, no model dropdown changes, no output path changes, no audio-setting changes.

#### Pattern 1 — `load_ledger` placement and posture

```python
from . import _otr_ledger_consumers as _OTRLC
led = _OTRLC.load_ledger(script_json)
plan = _OTRLC.production_plan_or_empty(production_plan_json)
```

`load_ledger` raises `ValueError` on the legacy parser-list shape. Two postures, picked per consumer's role in the workflow:

- **Loud (Bark, Kokoro, Sequencer, AudioGen, ProcSFX, Video):** let `ValueError` propagate. Bad wiring fails the run early instead of silently producing half-degraded audio mid-soak. Test asserts the raise.
- **Non-blocking (Critic only):** wrap in `try/except ValueError`, log loud, stamp `meta.critic_skipped_reason`, return passthrough+PASS. Critic's "never a blocker" rule wins for this one node — it's observe-only by design.

#### Pattern 2 — Iterating ledger lines by role

```python
for line in _OTRLC.iter_lines(led, roles={"character"}):
    text     = (line.get("text") or "").strip()
    line_id  = line.get("line_id")
    name     = _OTRLC.speaker_name(led, line)         # cast lookup, "UNKNOWN" on miss
    traits   = (line.get("traits") or "")
    preset   = _OTRLC.voice_preset(led, line)         # cast.voice_preset, None on miss
```

Field reads are direct dict access — no regex, no `[VOICE: NAME, traits] text` parsing. The structured ledger gives you the speaker, text, and metadata without parsing.

**Role-filter judgment rule.** Architect specs sometimes widen role filters in ways that contradict in-file design comments. **Never widen a role set beyond the consumer's current behavior without checking the in-file rationale.**

Concrete case (Bark): architect spec said `roles={"character","announcer"}`. The file comment in `batch_bark_generator.py` line 493-496 said:

> ANNOUNCER lines are intentionally skipped — they are rendered by the dedicated KokoroAnnouncer node on a separate bus. Keeping them out of the Bark pool eliminates Bark's "ums" and "ahs" from the broadcast-ready opening and closing bookends.

Widening to include announcer would double-render every announcer line (Bark + Kokoro both producing audio for the same `line_id`). Filtered to `{"character"}` only. Architect confirmed.

**Rule:** architect specs reflect intent at the cross-consumer level; file-level comments document why current behavior diverges from the apparent default. When in doubt, the latest scope clarification ("Behavior on those stays bit-for-bit identical") wins. Surface the discrepancy in the STEP 3 report and proceed with bit-for-bit-identical filtering. The architect approves or corrects.

#### Pattern 3 — Voice preset resolution with graceful fallback

```python
preset_from_cast = _OTRLC.voice_preset(led, line)
if preset_from_cast and str(preset_from_cast).startswith("v2/"):
    preset = preset_from_cast
else:
    preset = _voice_preset_for_character(name, voice_map, traits)
```

Prefer cast.voice_preset (v2 cast contract) when present and well-formed; fall back to the consumer's existing deterministic resolver (gender-aware hash for Bark, seeded grab-bag for Kokoro) when missing. The fallback is the existing function — don't replace it, just *prepend* the cast lookup.

`voice_map` comes from `production_plan_or_empty(production_plan_json).get("voice_assignments", {})`. With Director unwired (the v2 default), `voice_map` is `{}` and the existing fallback handles it.

#### Pattern 4 — Write-back contract

Single shape, applied to every consumer that stamps per-line fields:

```python
ledger_path = _OTRL_PATHS.in_flight_ledger_path()       # _otr_ledger module
if ledger_path is not None:
    led_disk = _OTRL_PATHS.load_ledger_safe(ledger_path)
    if led_disk is not None:
        for item in dialogue_items:
            line_id = item.get("line_id")               # carried from iter_lines
            if not line_id:
                continue
            fields = {                                   # same names as before
                "dur_s": dur,
                "start_s": cumulative_start,
                "tts_engine": "bark",
                # ... preserve every existing stamp field
            }
            if _OTRL_PATHS.patch_line_fields(led_disk, line_id, fields):
                cumulative_start += dur
                updated += 1
        if updated:
            # phase_ms, git_commit, audio_gate stamps still go on led_disk
            _OTRL_PATHS.save_ledger_safe(ledger_path, led_disk)
```

Key invariants:

- **Stamp by `line_id`, not by text.** `patch_line_fields` is the existing helper in `_otr_ledger.py`. It walks `ledger["lines"]` and matches on `line_id`, which is unique. Never collides on duplicate text (`"Okay."` × 2 was the failure mode of the old text-match block).
- **One `save_ledger_safe` at the end.** Atomic via tempfile + `os.replace`. Don't intersperse multiple saves; mutate `led_disk` in place across all per-line patches and meta stamps, then save once.
- **Preserve every existing stamp field name.** Bark: `bark_wav_dur_s`, `bark_render_ms`, `text_for_tts`, `tts_engine`, `voice_preset`, `render_ms`, `generated_dur_s`, `audio_sample_hash`, `dur_s`, `start_s`. Kokoro: similar plus `kokoro_wav_path` (if applicable). Sequencer: `start_s`, `dur_s`, `boundary`, `start_s_space`. AudioGen: `sfx_wav_path`, `sfx_engine`, `sfx_type`, `dur_s` on `lines[]`; legacy `wav_path`/`tts_engine` preserved on `sfx[]`. ProcSFX: `sfx_wav_path` (may be None on disk-write failure), `sfx_engine="procedural"`, `sfx_type`, `dur_s`. Video: `meta.procgen_path` (no per-line stamps). **Don't introduce new field names or remove existing ones during the rewrite** — that's a separate breaking change and not in scope.
- **Meta stamps still go on `led_disk`.** `record_phase_ms`, `set_meta(led_disk, "git_commit", ...)`, `audio_gate_record` + `append_audio_gate` — all still apply. Same call sites, same field names.

#### Pattern 4b — Asset replacement contract

When a node produces a new version of an existing asset (upscale pass, denoise pass, re-render with different settings), the new path **REPLACES** the old path on the same `line_id` via `patch_line_fields`. The ledger field always holds the latest version on disk; older versions are not tracked.

**Invariant: duration must match.** If a replacement asset has a different duration than what's on the ledger row, the replacement is wrong (timeline drift). Add a duration-check guard at the replacement site:

```python
old_dur = line.get("dur_s")
if old_dur is not None and abs(new_dur_s - old_dur) > 0.05:
    raise ValueError(
        f"asset replacement on {line_id} changed duration: "
        f"{old_dur:.3f}s -> {new_dur_s:.3f}s"
    )
patch_line_fields(led, line_id, {field: new_path, "dur_s": new_dur_s})
```

**Why no `_history` array:** downscaled or older versions are recoverable by re-running the source generator. Audit trail of "what's currently on disk" is the load-bearing question; "what used to be on disk" is not.

**Opt-out:** if a node legitimately needs to change duration on a replacement (voice retake at different pace, etc.), pass `allow_dur_change=True` explicitly. Default is strict — silent timeline drift is a bug, not a feature.

#### Pattern 5 — `production_plan_json` demotion

INPUT_TYPES change:

```python
"required": {
    "script_json": (...,),
    # production_plan_json USED to be here; demote ↓
},
"optional": {
    "production_plan_json": ("STRING", {
        "multiline": True, "default": "{}",
        "tooltip": "Production plan JSON from LLMDirector (optional under v2 ledger flow; empty {} degrades gracefully)",
    }),
    # ... existing optionals stay where they are, unchanged
},
```

Function signature change:

```python
def generate_batch(self, script_json, production_plan_json="{}", temperature=0.7):
```

Why both: the `INPUT_TYPES` move makes the socket optional in the ComfyUI graph (saved workflows still bind by name; the slot is preserved). The signature default makes an unwired socket safe. `production_plan_or_empty(plan_json)` handles `""`, `None`, malformed JSON, and non-dict shapes uniformly — one helper, one degradation path.

#### Pattern 6 — Hermetic test fixture for GPU-bearing consumers

The `patched_<x>_env` fixture (canonical example: `tests/test_bark_ledger.py`) does five things:

1. Patches `force_vram_offload` and `_runtime_log` to no-op (module-level).
2. Patches the inner generator (e.g. `_generate_single_line` for Bark) to return a deterministic numpy buffer + sample rate. Encodes input length into output length so different inputs produce different `dur_s` values — lets tests assert per-line distinct timings.
3. Patches `_load_<engine>` (e.g. `_load_bark` from `bark_tts`) to return mock model + processor.
4. Patches `_unload_llm` to no-op (story_orchestrator module).
5. Patches `_otr_ledger.in_flight_ledger_path` + `load_ledger_safe` + `save_ledger_safe` against a `state` dict so the test can:
   - Pre-seed the in-flight ledger view (`state["led_disk"] = json.loads(json.dumps(led_in))`).
   - Assert the merged ledger after the consumer ran (`state["led_disk"]` post-call).
   - Avoid touching the real disk.

Pre-seeding is **deep-copied** (`json.loads(json.dumps(led_in))`) because `save_ledger_safe` mutates the dict in place. Without the copy, the `state` dict and `led_in` alias the same object and assertions get confused.

**Mock-pipeline lessons learned:**

- **Sub-millisecond mocks lose `render_ms` stamps.** When a consumer measures `render_ms = int(elapsed * 1000)` and stamps it conditionally on `> 0`, a truly instantaneous mock makes the field disappear and confuses test assertions. Add `time.sleep(0.005)` inside the mock pipeline so the integer-millisecond floor clears. (Surfaced during Kokoro #3.)
- **Don't patch lazy-loaded transformers attributes.** When a consumer does `from transformers import AudiogenForConditionalGeneration` lazily inside a function, patching `transformers.AudiogenForConditionalGeneration` at module level can fail if transformers exposes the class via a lazy `__getattr__` rather than as a top-level attribute (different transformers versions differ). Prefer **pre-seed-the-cache** patterns: write a valid WAV at the canonical cache path, run the consumer, assert via the consumer's own `"CACHE HIT"` / `"MISS"` log line which path executed. Let the unreachable code stay unreachable. (Surfaced during AudioGen #5.)

#### Pattern 7 — Per-consumer test plan (4 cases)

Every audio consumer test file should cover:

1. **`test_<consumer>_iter_<role>_lines_only`** — non-target roles must NOT be stamped. Assert by checking `tts_engine` (or equivalent) is *absent* on filtered-out line_ids.
2. **`test_<consumer>_stamps_by_line_id_with_duplicate_text`** — two lines with identical `text` but distinct `line_id`. Both stamped. start_s monotonic. The single best test that proves the rewrite is correct (the old text-match block fails this).
3. **`test_<consumer>_voice_preset_fallback`** (TTS only) or `test_<consumer>_<consumer-specific-default-path>` — missing cast.voice_preset / unwired Director / empty production plan → existing deterministic fallback fires and produces a valid output. ProcSFX adapted this to `test_procsfx_disk_write_failure_graceful` (mock wav writer to raise; verify `sfx_wav_path=None` stamp + AUDIO batch still ships).
4. **`test_<consumer>_legacy_list_input_raises`** — `pytest.raises(ValueError)`. Message must contain `"legacy parser-list"` or `"OTR_LedgerScriptWriter"` so log triage points at the right wiring. Critic exception: this becomes a passthrough-with-PASS test (Critic is non-blocking).

For non-TTS consumers (Sequencer, ProcSFX), test 3 substitutes a "consumer-specific default behavior with no production plan" check.

#### Anti-patterns (do not do these)

- **Don't catch `ValueError` from `load_ledger` to "be safe"** — except in the Critic. Loud failure is the goal.
- **Don't use `_otr_ledger.in_flight_ledger_path()` and then mutate the wire-input ledger** in the same consumer. Two ledger views drift. Pick one source of truth per consumer: wire input for Critic (it's an output node), in-flight on-disk for Bark/Kokoro/Sequencer/AudioGen/ProcSFX (they have no ledger output socket). Video uses both — wire input for parse + helpers, singleton for the final `meta.procgen_path` stamp + episode rename.
- **Don't introduce new ledger field names during the rewrite.** A rewrite is a port, not a schema bump. New fields land in a separate commit with a `BUG_LOG.md` entry. (ProcSFX Option 2 is the **one** documented exception this sprint — adding `sfx_wav_path` was an architect-approved scope expansion, not a stealth schema bump.)
- **Don't reorder `INPUT_TYPES`.** Saved workflows bind by name in the API but position in the saved JSON. Demotion (required → optional) is OK; reordering within a section is not.
- **Don't widen role filters past the consumer's current behavior** without checking file-level comments. Bark + Kokoro is the canonical case (announcer goes to Kokoro only).
- **Don't write multiple `save_ledger_safe` calls in one consumer.** Mutate `led_disk` in place; save once at the end. Multiple saves = race condition + redundant disk writes.

#### Quick reference — files in scope

| Concern | Module | Notes |
|---|---|---|
| Read-side helpers | `nodes/_otr_ledger_consumers.py` | `load_ledger`, `iter_lines`, `cast_lookup`, `speaker_name`, `voice_preset`, `production_plan_or_empty` |
| Write-side helpers | `nodes/_otr_ledger.py` (existing) | `in_flight_ledger_path`, `load_ledger_safe`, `save_ledger_safe`, `patch_line_fields`, `set_meta`, `record_phase_ms`, `audio_gate_record`, `append_audio_gate`, `lookup_git_commit` |
| Test fixtures | `tests/fixtures/ledger_stub.py` | `make_stub_ledger(*, with_sfx=True, with_music=True, ...)`, `make_legacy_list()` |

### Post-soak follow-ups

#### B1 + B2 — coupled post-script title generation pass

`OTR_LedgerScriptWriter` today generates `outline.title` during the outline-planning phase, BEFORE the script is written. Recon during consumer #7 surfaced this: the title reflects the LLM's *plan*, not the LLM's finished output. Jeffrey's design intent is a dedicated title pass that reads the full finished script and produces a punchy title reflecting the actual completed plot.

Until that lands, Video's title chain primary slot (`led.meta.episode_title`) resolves empty under the new-writer flow, and the chain falls through to widget or TIMESTAMP_LASTRESORT.

- **B1** — Add post-script title-gen pass to `OTR_LedgerScriptWriter`. New LLM call after script is finalized but before save. Inputs: full `script_text` + `outline.premise` (for context). Output: punchy 3-8 word title. Approximate scope: ~50 LOC. Locked file this sprint; do during post-soak deletion sprint when writer is editable again.
- **B2** — Stamp result at `led["meta"]["episode_title"]`. One-line addition at the title-gen call site. Couples directly to B1.

Together: Video's title chain primary slot resolves cleanly under the new writer flow. Until then, chain falls through to widget or TIMESTAMP_LASTRESORT under new-writer flow.

### Visual chain recon — AUDITED CLEAN, 2026-05-10

Recon pass over every active downstream visual + post-process + cast/portrait + utility node, validating ROADMAP's prediction (line 67) that "video files all read ledger from disk (not wire `script_json`), so they should 'just work' with the L3 format." Run as STEP 0 of the post-consumer-7 continuation sprint. **No rewrites needed.** Every node already uses L3-native field names (`line_id`, `char_id`, `speaker_role`, `start_s`, `dur_s`, `text`, `word_count`, `shot_id`, `cast[].char_id`, `cast[].name`, `cast[].voice_preset`, `cast[].portrait_path`, `meta.gen_params_initial.style`, `meta.radio_bookend_path`, `episode_id`), reads ledger from disk via `_OTRL.in_flight_ledger_path()` / `production_ledger.get_ledger()` singleton + `load_ledger_safe`, and degrades gracefully with `.get(...)` defaults on missing fields.

| File | Verdict | Rationale |
|---|---|---|
| `visual/batch_flux_render.py` | AUDITED CLEAN | DEAD path `_parse_env_prompts(script_json)` looks for legacy `[{"type":"environment", "description":...}]`; on L3 dict input falls back to `[fallback]` (no crash). Default widget `skip_env_stills=True` bypasses entirely. LIVE radio bookend pass reads ledger via singleton + `load_ledger_safe`, uses `meta.gen_params_initial.style` (L3-correct) with `meta.gen_params.style` back-compat. `led.get("scenes")` tier-4 fallback is L3-orphaned but degrades safely (no `scenes` array in L3 → returns []). Stamps top-level `radio_bookend_path` + `meta.radio_bookend_path` — no per-line writes. |
| `visual/batch_flux_portrait_render.py` | AUDITED CLEAN | Reads ledger via `_OTRL.in_flight_ledger_path()` + `load_ledger_safe`. Walks `cast[]` for `char_id`, `name`, `voice_preset`, `portrait_path`. BUG-094 cast filter uses `iter_lines` semantically by walking `lines[]` and grouping by `char_id` + `resolve_speaker_role(ln)`. All L3-native. |
| `nodes/batch_humo_render.py` | AUDITED CLEAN | Reads ledger via `_load_ledger_with_path`. Uses L3-native fields exclusively. Orphan-rescue speaker fallback chain `ln.get("speaker") or ln.get("name") or ln.get("character_name")` (`L691`) only fires when `char_id` misses `cast[]`; on clean L3 data it's dormant. The fallback is intentionally defensive against future writer drift. |
| `nodes/batch_ltx_render.py` | AUDITED CLEAN | Reads ledger via `_OTRL.load_ledger_safe`. Uses `line_id`, `speaker_role`, `dur_s`. `_build_ltx_role_prompt(role, line, ledger)` returns a static prompt by role (no field interpolation). |
| `nodes/video_composite.py` | AUDITED CLEAN | Reads ledger via `_load_ledger_with_path`. Uses `line_id`, `speaker_role` (default "character" on missing), `start_s`, `dur_s`. BUG-LOCAL-129a static-radio fill + BUG-135 motion-loop fill paths intact. |
| `nodes/rtx_upscale.py` | AUDITED CLEAN | Path-in/path-out wrapper. Only ledger read is for spacesaver cleanup: reads `meta.perfect_run_spacesaver` flag + `episode_id`. Both top-level/meta — fully L3 compatible. |
| `nodes/otr_post_upscale_procgen_blend.py` | AUDITED CLEAN | Path-in/path-out wrapper. Uses `_OTRL.in_flight_ledger_path()` for episode_id discovery. No `lines[]` reads. |
| `nodes/otr_save_to_episode_workspace.py` | AUDITED CLEAN | IMAGE save sink. Uses `_OTRL.in_flight_ledger_path()` + `episode_id` only. No `lines[]` reads. |
| `nodes/otr_video_plan.py` + `otr_shot_duration_calculator.py` | AUDITED CLEAN | Pre-FLUX adapters. Take `production_plan_json` (Director output) and emit shot/compose plans. Don't touch `script_json`/ledger directly. |
| `nodes/otr_save_copy.py` + `otr_video_concat.py` | AUDITED CLEAN | Pure path-in/path-out helpers. No ledger reads. |
| `nodes/post_audio_video_pipeline.py` | RETIRED | Per `__init__.py` comment: "RETIRED in favour of in-graph batch nodes". NOT in `workflows/otr_scifi_16gb_full.json`. Backward-compat-only registration. Audit moot. |
| `nodes/_otr_cast_repair.py` | AUDITED CLEAN | Helper module (no INPUT_TYPES). Used by writer; orphan classification + plateau-bounded repair. No legacy parser-list assumptions. |
| `nodes/_otr_voice_resolver.py` | AUDITED CLEAN | Helper module (no INPUT_TYPES). `VoiceSpec` dataclass for engine:preset parsing. Field-agnostic. |
| `nodes/voice_render.py` | NOT REGISTERED | `OTR_VoiceRender` class exists with `RETURN_TYPES = ("AUDIO",)` but is NOT in `__init__.py:_NODE_MODULES`. Not in active workflow. |
| `nodes/_voice_backends/{bark,kokoro}.py` | AUDITED CLEAN | Voice backend driver implementations. Take `VoiceSpec` + raw text. No direct ledger interaction. |
| `nodes/_otr_period_prompts.py` | AUDITED CLEAN | Period exemplar dataclass + `render_prompt(user_instruction, ...)`. Field-agnostic; doesn't read ledger. |

The recon collapses STEPS 1-8 of the planned post-consumer-7 sprint into a single recon-verdict deliverable. The remaining work — helper API tests (SHIPPED 48/48 PASS), B4 prompt audit (SHIPPED, see below), workflow JSON edit, dry-run gates, final report — proceeds against this AUDITED CLEAN baseline.

### Helper API tests — SHIPPED 2026-05-10 (48/48 PASS)

`tests/test_otr_ledger_consumers.py` — 48 tests across six classes mirroring the helper module API:

- `TestLoadLedger` (5) — dict input → dict; legacy list → ValueError with "OTR_LedgerScriptWriter" in message; non-dict-non-list root → ValueError; invalid JSON propagates; empty dict input returns empty dict.
- `TestIterLines` (9) — no filter yields every line in original order; role-filter narrows the walk; empty role set yields nothing; lines with missing/unknown roles skipped under filter, yielded under no-filter; missing `lines` key + None value both yield empty.
- `TestCastLookup` (8) — known char_id resolves; second char_id resolves (no short-circuit); unknown / empty / None char_id → `{}`; missing `cast` key → `{}`; non-dict cast entries skipped safely; int char_id coerces to string.
- `TestSpeakerName` (7) — character line → cast name; announcer/sfx → "UNKNOWN" (role tag != cast member); missing/None/empty line → "UNKNOWN"; cast entry missing `name` → "UNKNOWN".
- `TestVoicePreset` (6) — known char_id → preset; announcer/unknown → None; cast entry missing `voice_preset` → None.
- `TestProductionPlanOrEmpty` (9) — valid dict plan returns plan; "" / None / "{}" / invalid JSON / list root / non-dict roots all → `{}` (graceful Pattern 5 demotion).
- `TestComposition` (3) — full Pattern 2 walk shape (`load_ledger → iter_lines → speaker_name → voice_preset`) for character + announcer roles; legacy list short-circuits at `load_ledger`.

Bug Bible regression: 23/1/2/0 baseline confirmed pre-test; held post-test.

### LLM prompt audit — 2026-05-10

B4 audit (per release-blocker B4 in this file): contract-verification pass over every LLM prompt construction site that interpolates ledger fields, performed AFTER the consumer-rewrite sprint shipped + visual chain recon completed. Goal: confirm no prompt site reads stale field names that would silently render with wrong data on L3 input.

Methodology: grep across `nodes/` for `_build_*prompt`, `def *prompt`, prompt f-strings interpolating `led.` / `ledger.` / `cast` / `line` / `speaker`. Each site read end-to-end, fields inventoried against the L3 schema, verdict assigned.

| Prompt site | File:line | Inputs (interpolated) | Verdict | Notes |
|---|---|---|---|---|
| `_build_user_prompt` (outline) | `_otr_outline.py:253` | `req.news_seed`, `req.style`, `req.cast_size`, `req.target_words` | **CLEAN** | Decoupled from ledger via `OutlineRequest` dataclass. Writer (`OTR_LedgerScriptWriter._validate_inputs`) builds the request from widget args + `gen_params_initial`, never reads `lines[]`/`cast[]`. Locked file (Phase 3 LPL writer); audit-only. Field renamed `style_hint` → `style` 2026-05-10 to match user-visible widget name; `target_seconds` removed earlier (words-only contract). |
| `_REPAIR_PROMPT_TEMPLATE` (outline) | `_otr_outline.py:265` | `prev_response`, `validation_error` | **CLEAN** | Pure JSON-schema-validation feedback loop. No ledger fields. Locked file. |
| `_build_user_prompt` (line composer) | `_otr_line_composer.py:175` | `req.canon_header`, `req.last_lines`, `req.speaker`, `req.intent`, `req.mood`, `req.target_words` | **CLEAN** | Decoupled via `LineRequest` dataclass. Writer feeds `req.speaker` from cast `name`, `req.intent`/`req.mood` from beat fields. No raw `lines[]` reads. Locked file. |
| `_format_last_lines` (line composer) | `_otr_line_composer.py:168` | `(spk, txt)` tuples | **CLEAN** | Caller passes already-resolved `(speaker_name, text)` pairs. Field-agnostic. Locked file. |
| `_SYSTEM_PROMPT` (line composer) | `_otr_line_composer.py:139` | (none — static string) | **CLEAN** | Static system prompt. No interpolation. |
| `OTR_PERIOD_SYSTEM_PROMPT` + `render_prompt` | `_otr_period_prompts.py:186` | `user_instruction`, `exemplars` (`PeriodExemplar` dataclass list) | **CLEAN** | Static system prompt + few-shot block prepended to caller's `user_instruction`. No ledger reads. |
| `_build_critic_prompt` | `script_critic.py:306` | `script_text`, `style`, `anti_slop` | **CLEAN** | `style` resolved from `meta.gen_params_initial.style` (L3-correct) at L843-852 with cleanup_model_id / model_id chain. `anti_slop` from `OTR-ANTI-SLOP.md` filtered by `_coerce_params(meta.gen_params_initial)`. Both flow from L3-correct meta block. |
| `_build_revision_prompt` | `script_critic.py:541` | `script`, `issues` (list[str]), `style` | **CLEAN** | Same `style` resolution as `_build_critic_prompt`. Issues list comes from critic's parsed structured response. No raw `lines[]` interpolation. |
| `_anti_slop` rubric template (critic) | `OTR-ANTI-SLOP.md` via `_filter_rubric` | `target_length`, `target_words`, `num_characters`, `style`, `scene_count`, `scene_word_budget` (all from `meta.gen_params_initial`) | **CLEAN** | Gate evaluator (`_evaluate_gate`) reads `meta.gen_params_initial` (L3-correct) via `_coerce_params`. Missing fields fail-open (rule still ships) — safe direction. |
| `_build_director_json_repair_prompt` | `story_orchestrator.py:4570` | `raw_output`, `script_text` | **CLEAN** | Director JSON repair feedback. No ledger field reads. Director is unwired in v2 ledger flow per Pattern 5; this prompt fires only when Director is actively wired. |
| `_build_normalize_prompt` (legacy normalize) | `_otr_legacy_writer.py:4330` | `script_text`, `is_segment` | **DEAD CODE** | Legacy writer FORMAT_NORM phase. Field-agnostic prompt (text formatting only). Active only when `OTR_LLMScriptWriter` (legacy node) runs; new `OTR_LedgerScriptWriter` (v2) does NOT call this. Will be deleted post-soak per ROADMAP sprint exit criterion 7. |
| `_radio_bookend_prompt` widget override (FLUX) | `visual/batch_flux_render.py:406` | (user widget — verbatim) | **CLEAN** | Widget passes through verbatim when non-empty. No ledger interpolation. |
| `_build_dynamic_radio_prompt` (FLUX) | `visual/batch_flux_render.py:73` | `meta.gen_params_initial.style`, `meta.gen_params.style` (back-compat), `meta.gen_params_initial.style_custom`, `scenes[0].env`/`description` (L3-orphaned), `episode_id` slug, `_RADIO_FALLBACK_PROMPT` | **CLEAN** | Six-tier fallback chain. Tier 4 (`scenes[0]`) is L3-orphaned (no `scenes` array in L3) but degrades safely to next tier. Tier 1 (`meta.gen_params_initial.style`) is the live L3 path. |
| `_PROMPT_BY_ROLE` (LTX `_build_ltx_role_prompt`) | `nodes/batch_ltx_render.py:404` | `role` (`speaker_role` value) | **CLEAN** | Returns a static prompt indexed by role. No `line` / `ledger` field interpolation despite the signature carrying them (preserved for future per-line overrides per BUG-LOCAL-112 comment block). |
| `_normalize_target_length`/`_evaluate_gate` (critic rubric gates) | `script_critic.py:102, 125` | `target_length`, `target_words`, `num_characters`, `style`, `scene_count`, `scene_word_budget` | **CLEAN** | All from `meta.gen_params_initial` via `_coerce_params`. Sandboxed eval (`__builtins__={}`). Fail-open on parse failure — safe. |
| MusicGen `cue_entry["generation_prompt"]` | `story_orchestrator.py:5566`, `musicgen_theme.py:266` | `entry.get("generation_prompt")` from Director's music_plan | **CLEAN** | Director-emitted prompts; consumed by MusicGen. No ledger field interpolation; the prompt is verbatim user/Director text. Field-agnostic on the consumer side. |

**Audit summary:** **15 CLEAN / 1 DEAD CODE / 0 NEEDS UPDATE.** No prompt site reads stale field names that would silently render with wrong data on L3 input. The single DEAD CODE entry (`_build_normalize_prompt`) is on the legacy writer path; it's already documented for deletion post-soak per ROADMAP sprint exit criterion 7. **No prompt rewrites needed.** Audit verdict locked into Bug Bible 23/1/2/0 regression baseline.

---

## PRIOR WORK — pre-FULL acceptance soak (handoff-ready as of 2026-05-07 PM)

**State:** code complete on `v2.0-alpha`, 0 known faultlines blocking. Awaits a single acceptance FULL run on the RTX 5080.

### What landed today (commits 4198d72 + 5d7e887)

| ID | What | Where |
|---|---|---|
| **BUG-LOCAL-117d** | ffmpeg boomerang post-process (default ON via `OTR_LTX_LOOP_VIA_REVERSE`) — each non-character chunk renders HALF audio-target dur, then `[a]` + `[b].reverse.trim(start_frame=1).concat` doubles back. Sample wall time halved; chunk-boundary snap eliminated (both ends are radio_bookend). | `nodes/batch_ltx_render.py` `_make_boomerang_via_ffmpeg` |
| **BUG-LOCAL-117e** | Music chunk cap 7s -> 22s (validated against 25s @ 832×480 mega-test). `LTX_MAX_FRAMES` 353 -> 705. `clip_length` widget default 7.0 -> 22.0. | `nodes/scene_sequencer.py` `_MUSIC_MAX_CHUNK_DUR_S`, `nodes/batch_ltx_render.py`, `workflows/otr_scifi_16gb_full.json` |
| **BUG-LOCAL-117f** | Duration-aware anti-clobber. ffprobes `<line_id>.mp4`; if actual < expected − 0.25s, unlink + re-render. Heals half-duration clips left by crashed runs. `STALE-LOCKED` report path on unlink failure. | `nodes/batch_ltx_render.py` execute() pre_existing block |
| **117d hardening (Patch A)** | Boomerang pins `-video_track_timescale 12800`. | `nodes/batch_ltx_render.py` `_make_boomerang_via_ffmpeg` |
| **117d hardening (Patch B)** | All 5 silent-encode sites in VideoComposite pin `_STATIC_SEGMENT_TIMEBASE` (`_layered_per_clip_silent` layered + scale-fit, `_pillarbox_humo_silent`, `_make_gap_segment`, `_normalize_humo_segment`). Uniform timebase across HuMo+LTX+gap-fill+boomerang -> no `Non-monotonous DTS` at any seam. | `nodes/video_composite.py` |
| **Audit script (Patch D)** | `scripts/audit_otr_full_run.py` — post-run acceptance audit. ffprobes each `videos/*.mp4` vs `ledger.lines[].dur_s`, greps comfyui.log for failure patterns, exit 0 = all bullets pass. | `scripts/audit_otr_full_run.py` |
| **Tests** | 33/33 in `tests/test_batch_ltx_render.py` pass on Windows venv. New pins: `LTX_MAX_FRAMES==705`, `clip_length default==22.0`, `clip_length max==28.16`, boomerang default-on + truthy set + helper-exists + missing-input-raises + filter-graph + timebase, anti-clobber probe-call + 0.25s tolerance + unlink + STALE-LOCKED + fall-through guard + TESTCHAR fixture name. | `tests/test_batch_ltx_render.py` |

### What's left to test — pre-FULL handoff checklist

These are the only items between now and a green v2.0 cut. Every one is a **runtime-only** check; nothing remaining is code work.

**Quick start:** `.\scripts\prep_full_run.ps1` runs steps 2-3 as a single read-only report (env vars + active log + newest pending episode + videos/ contents + composited/ status + C: free space). Pass `-Wipe` to actually delete stale clips. Pass `-Episode <ep_id>` to target a specific folder.

1. **Restart ComfyUI Desktop.** All four touched modules (`batch_ltx_render.py`, `scene_sequencer.py`, `video_composite.py`, plus the test file) are cached in `sys.modules` of the running ComfyUI process. They will not hot-reload. Confirm by checking the ComfyUI version banner regenerates on the splash.
2. **Verify env vars are set in HKCU\Environment.** ComfyUI Desktop inherits User-scope env vars at process launch only — not Machine, not session. Run from PowerShell:
   ```
   [Environment]::GetEnvironmentVariable("OTR_LTX_ENGINE","User")
   [Environment]::GetEnvironmentVariable("OTR_LTX_LOOP_VIA_REVERSE","User")
   ```
   Expected: `v2_3` (or unset to fall through to `v0_9` default), and `on` (or unset — same default).
3. **Wipe stale clips for the target episode.** Even with BUG-LOCAL-117f duration-aware healing, a sampler/LoRA-strength change between runs leaves valid-duration but stale-content clips that the duration check can't catch. `Remove-Item` `output\otr\episodes\<ep_id>\videos\*.mp4` before queueing.
4. **Queue `otr_scifi_16gb_full.json`.** Watch banner at run start for these load-bearing lines:
   - `[BatchLTXRender] BUG-LOCAL-117 engine=v2_3` (or `v0_9` — confirms env var picked up)
   - `[BatchLTXRender]   boomerang: ON (BUG-LOCAL-117d) -- render HALF chunk_dur_s, ffmpeg-reverse-and-concat doubles back to full audio target`
   - `[EpisodeAssembler] music mirror: appended=N, chunked_cues=M ... post-BUG-117e: music chunks <= 22.0s`
5. **Watch the log for these failure strings.** Any one of these is a STOP signal — paste the surrounding context into the next session and we triage:
   - `Non-monotonous DTS` (Patch A/B failed; means a silent encode site missed the timebase pin)
   - `boomerang FAILED` (post-process crashed; chunk has half-duration content under full-duration audio)
   - `duration contract VIOLATED` (VideoComposite final-mux duration check; audio overran video)
   - `[BatchLTXRender] <line_id> failed:` (per-clip exception inside the LTX loop)
   - `STALE-LOCKED` (anti-clobber wanted to heal a half-duration clip but couldn't unlink — means a Windows process is holding the file open)
   - `derived ledger from .mp4 not found` (BUG-082 regression)
   - `audio may be truncated` (BUG-084 tail-pad fallback fired; episode is fine but flag for follow-up)
6. **Run the audit script after the run completes.** `--log` is optional; when omitted the script auto-discovers the most-recently-modified active `comfyui_<port>.log` under `C:\Users\jeffr\Documents\ComfyUI\user\` (rotated `.prev*.log` files are ignored).
   ```
   & C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
       scripts\audit_otr_full_run.py `
       --episode "C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\<ep_id>"
   ```
   To pin a specific log instead, pass `--log "C:\Users\jeffr\Documents\ComfyUI\user\comfyui_8000.log"`. Exit code `0` = all S3.x acceptance bullets pass. Exit `1` = full report printed; copy and paste into next session.

   **Live tail during the run:** `.\scripts\tail_otr_run.ps1` from the repo root auto-discovers the active port log and color-codes load-bearing lines green, STOP signals red, pipeline markers cyan. Pass `-Port 8001` to pin a port, or `-Tail 50` for last-N-lines context before live tail begins.
7. **Acceptance bullets that count as "GREEN":**
   - All non-character `ledger.lines[]` entries have `clip_meta.source_kind == "ltx"` (no `static` fallbacks except for cues with no audio).
   - Every `videos/<line_id>.mp4` duration is within 0.25s of `ledger.lines[].dur_s`.
   - Pre-upscale episode mp4 dims = (832, 480); post-upscale dims = (1920, 1080).
   - 0 occurrences of any failure string from step 5 in `comfyui.log`.
   - `silent_combined.mp4` concat completed without `-c copy` rejection or `Non-monotonous DTS` warnings.

### If any acceptance bullet fails

Open a new session with the failing log section + the audit script's report. The likely fixes by symptom:

| Symptom | Likely fix |
|---|---|
| `Non-monotonous DTS` at silent_combined concat | A silent-encode site missed a timebase pin — grep `video_composite.py` for `libx264.*yuv420p` not followed by `_STATIC_SEGMENT_TIMEBASE` |
| `boomerang FAILED` | Look at the ffmpeg stderr in the warning; usually means input mp4 was 0-byte (render crashed before _save_video_mp4 finished) |
| Duration drift > 0.25s | BUG-LOCAL-091 chunking math edge case OR BUG-LOCAL-117f anti-clobber kept a stale clip — wipe `videos/*.mp4` and re-run |
| First-run `STALE-LOCKED` | Probably a Windows Explorer preview holding the file open. Close any video preview pane and re-run. |
| Engine=v0_9 in banner when you expected v2_3 | Env var not set, OR ComfyUI Desktop was launched before the env var was set. Restart ComfyUI Desktop from a fresh PowerShell after setting the var. |

### After GREEN

1. Cut tag `v2.0-alpha` -> `v2.0-rc1` on Jeffrey's machine via Desktop Commander cmd shell (per CLAUDE.md, only Jeffrey tags releases).
2. Promote BUG-LOCAL-117a/117b/117c/117d/117e/117f entries to the Bible (sister repo `comfyui-custom-node-survival-guide` -> add to `BUG_BIBLE.yaml`, add regression tests to `tests/bug_bible_regression.py`, run three-file contract test).
3. README pass — document `OTR_LTX_ENGINE` and `OTR_LTX_LOOP_VIA_REVERSE` env vars + 22s clip_length default.
4. Then move to the v2.0 ecosystem review queued in this file (ComfyUI Core, ComfyUI-GGUF, ComfyUI-Ollama, Gemma 2026 Challenge — see "v2.0 pre-ship ecosystem review" section below if it exists, otherwise create).

---

## Phase 0+ candidates (post-v2.0-alpha)

### Status snapshot — 2026-05-08 OVERNIGHT autonomous sprint

**Pushed to `origin/v2.0-alpha` overnight (head: `4eeda0e`):**

| Commit | What landed |
|---|---|
| `b8c26f4` | Predecessor baseline -- §1+§2+§3 skeletons + audit calibration + BUG-118 widget fix |
| `dfe26e6` | §1 helpers: `build_contract_from_director_plan` + `detect_aliases` |
| `c10bf16` | BUG-LOCAL-120 update |
| `8e07a1a` | BUG-LOCAL-121 (round-robin Element 4): KeyError on padded `voice_assignments` keys |
| `f7a06e1` | BUG-LOCAL-122 (round-robin Element 2): `lock_to_episode` read-and-compare-version with `CastContractMismatch` |
| `eba1f5e` | §4+§5 skeleton: `nodes/_otr_cast_repair.py` -- `OrphanClass` 5-bucket enum + `apply_classifications` + plateau-bounded `repair_orphans` + `CastContractUnreparable` |
| `dfa9b07` | Voice Backend Abstraction skeleton (NEW FILES ONLY): `nodes/_voice_backends/{__init__,_protocol,bark,kokoro}.py` registry/protocol + bark/kokoro stubs + `nodes/voice_render.py` `OTR_VoiceRender` (UNREGISTERED) |
| `6b05fd0` | Old-timey LLM module: `nodes/_otr_period_prompts.py` 1940s system prompt + 3 period exemplars + `render_prompt` |
| `4eeda0e` | `scripts/soak_watch.ps1` polls episode dir + auto-audits when soak quiets |

**Test floor:** 106/106 cast-contract suite + 33/33 LTX regression + AST clean.

**What did NOT land overnight (waiting on FULL acceptance soak finish):**

- Story orchestrator hooks at L6423 / L~640 / L920 (locked file)
- `production_ledger.py` `cast_contract_version` merge guard (locked file)
- Migration of Bark / Kokoro logic into `_voice_backends/{bark,kokoro}.py` (touches `batch_bark_generator.py` + `kokoro_announcer.py`, both locked)
- Registration of `OTR_VoiceRender` in `__init__.py`
- LLM wire-up for `repair_orphans` (the `Classifier` callable; today it's a deterministic stub)
- Period-prompt integration into the existing LLM call site

These are all "edit-locked-file" tasks. They can land in a single follow-up session once the soak is verifiably done -- the new helpers + 106-test green baseline are the substrate that follow-up session will build on.

**Round-robin code review:** transcripts at `docs/2026-05-08-cast-contract-shipped-code-review__01_chatgpt.md` (gpt-5.5, 80.2s) + `__02_gemini.md` (gemini-3.1-pro-preview-customtools, 65.4s) + `__04_synthesis.md`. Two real bugs caught (BUG-121 padded-key KeyError + BUG-122 lock blind-refusal) and both fixed in the same autonomous loop.

---

### Cast Contract Extensions

**Extends:** existing "Character Identity as a Data Contract" RFC.
**Source patterns:** [NousResearch/autonovel](https://github.com/NousResearch/autonovel) — `state.json` versioning, `characters/canon` split, `adversarial_edit.py`, plateau revision loop.

**Verdict on existing RFC:** ~80% correct. Keep all of it. Five gaps below.

- Data-contract thesis: confirmed (autonovel = `audiobook_voices.json`)
- Insertion point at `story_orchestrator.py:6423`: correct
- Anti-brick-by-brick: correct
- Already have alias plumbing: `_consolidate_similar_cast_rows_with_aliases` line 440

**Five additions (leverage order):**

1. **Propagation debt — stamp version everywhere.** Every dialogue line carries `cast_contract_version: "sha:a3f9c2e1"` plus `character_id`. `production_ledger.py` rejects merges where version mismatches. Pattern: autonovel `state.json`.

2. **Lock contract per episode.** After `_bark_health_check_for_cast` (~line 640), freeze to `episodes/<ep_id>/cast_contract.locked.json`. Immutable for episode lifetime. Kills SceneSequencer ↔ BatchBark drift.

3. **Character canon layer.** Per-episode `character_canon.md`:
   ```markdown
   ## c02 — AEGEUS
   - Voice: v2/en_speaker_5
   - Tics: clipped, no contractions, marine metaphors
   - Forbidden: military slang (c01's register)
   - Phrase pattern: "[Noun] is [verb-ing] back through [place]"
   ```
   Inject into ScriptWriter prompt. Feed to existing `_check_voice_consistency` (line 920) as rubric. Hard contract = routing; canon = identity.

4. **Adversarial classification before repair.** Tiny LLM classifies orphan tags into 5 buckets:

   | Class | Action |
   |---|---|
   | `TYPO_OF_EXISTING` | Auto-canonicalize |
   | `ALIAS_OF_EXISTING` | Add to alias map, bump version |
   | `GENUINELY_NEW` | Hard fail, reroll cast |
   | `NARRATIVE_LEAK` | Demote to narration, HuMo bypass |
   | `DISCARD` | Drop |

   Pattern: autonovel `adversarial_edit.py`.

5. **Plateau-bounded repair loop.**
   ```python
   prev = None
   for _ in range(3):
       orphans = validate(script, contract)["unknown"]
       if not orphans: break
       if orphans == prev: raise CastContractUnreparable(orphans)
       script = repair_orphans(script, orphans, contract)
       prev = orphans
   ```
   Two-pass identity → escalate. No third LLM call.

**Rejected from autonovel (not OTR-fit):** voice fingerprinting (GPU cost vs rare failure mode), Opus dual-persona review (novel-scale, wrong size for radio episode), reader panel (same).

**File touches:**

*New:*
- `nodes/_otr_cast_contract.py` *(existing RFC)*
- `nodes/_otr_voice_resolver.py` *(existing RFC)*
- `nodes/_otr_cast_repair.py` *(§4, §5)*
- `nodes/_otr_canon.py` *(§3)*

*Edit:*
- `nodes/story_orchestrator.py` L6423 — gate after `_parse_script` *(existing RFC)*
- `nodes/story_orchestrator.py` L~640 — lock contract to disk *(§2)*
- `nodes/story_orchestrator.py` L920 — canon as consistency rubric *(§3)*
- `nodes/production_ledger.py` — version field + merge guard *(§1)*
- `nodes/scene_sequencer.py`, `nodes/batch_bark_generator.py` — strip dup resolvers *(existing RFC)*

**Acceptance criteria (delta only):**

- [ ] Every dialogue line carries `cast_contract_version` + `character_id`
- [ ] `cast_contract.locked.json` byte-identical from BatchBark start → episode end
- [ ] `character_canon.md` injected into ScriptWriter, used by `_check_voice_consistency`
- [ ] Orphans classified into 5 categories before any repair LLM call
- [ ] Repair loop terminates on plateau, raises structured error, no third call

**Open questions:**

1. Version: content-addressed sha vs monotonic v1/v2? → leaning sha
2. `NARRATIVE_LEAK` → ANNOUNCER role, or new narrator role?
3. Canon prompt slot: system / fewshot / cast-roster append?

---

### Voice Model Agnostic Nodes (Voice Backend Abstraction)

**Pairs with:** Cast Contract Extensions §3 — character canon entries carry fully-qualified voice specs (`bark:v2/en_speaker_5`, `cosyvoice:robotic_calm`, `kokoro:bm_fable`) once both pieces ship.
**Source patterns:** existing TTS upgrade backlog at `project_tts_upgrade_candidates_2026-04-23.md` (CosyVoice 2/3 Apache-2.0 first pick); applies `feedback_use_community_nodes_not_custom` (wrap community nodes, don't vendor model code) and `feedback_otr_stays_mit` (license bar per backend).

**Problem:** `OTR_BatchBark` is hard-bound to Bark, `OTR_KokoroAnnouncer` is hard-bound to Kokoro. Adding a new TTS engine (CosyVoice, XTTS, Piper, Fish Speech, Qwen3-TTS) means a new node class + workflow JSON edits + parallel BatchBark-equivalent batching code. Per-character voice-model assignment is impossible: AEGEUS can't get a synthetic-timbre engine while MONTY uses warmth-tuned Bark.

**Five additions (leverage order):**

1. **Single canonical node — `OTR_VoiceRender`.** Widgets: `voice_model` enum (bark / kokoro / cosyvoice / xtts / piper), `voice_preset` STRING (model-specific), `text` input, standard knobs (temperature, hallucination guard) routed only when the backend supports them.

2. **`nodes/_voice_backends/` driver module.** One file per engine implementing a small interface: `load(preset)`, `generate(text, **kw) -> wav`, `unload()`. Initial drivers wrap existing Bark + Kokoro impls; subsequent drivers wrap community ComfyUI TTS nodes (verify each license against MIT bar before adopting).

3. **Voice spec format in cast contract.** Cast canon entries become `Voice: bark:v2/en_speaker_5` rather than implicit engine binding. `nodes/_otr_voice_resolver.py` (already in Cast Contract RFC file list) parses to `(engine, preset)` pairs.

4. **Per-character routing.** A single batch node walks the dialogue ledger, looks up each line's `character_id` in `cast_contract.locked.json`, routes to the resolved backend. Eliminates the "No Director mapping for MONTGOMERY → pool fallback" path observed 2026-05-07 in `signal_lost_silent_countdown` run.

5. **Back-compat shims.** `OTR_BatchBark` and `OTR_KokoroAnnouncer` stay registered as thin wrappers that delegate to `OTR_VoiceRender` with `voice_model` pre-pinned. Existing workflow JSONs validate and run unchanged.

**Rejected (defer or out of scope):** voice cloning / fingerprinting (GPU cost, not OTR-fit at radio-episode scale); in-engine streaming (current batch model fits 30-second-cue scope).

**Migration path (non-destructive):**

1. Add `_voice_backends/bark.py` + `kokoro.py` wrapping current code (no behavior change, just relocation).
2. Add `nodes/voice_render.py` registering `OTR_VoiceRender`. Register in `__init__.py`.
3. Existing `BatchBark` + `KokoroAnnouncer` become thin shims OR stay full impls during transition (decide per stability of new path).
4. Workflow JSONs unchanged short term; new workflows opt into `OTR_VoiceRender` directly.
5. Add `cosyvoice.py` once Bark + Kokoro path is proven — first real cross-engine episode validates the contract end-to-end.

**File touches:**

*New:*
- `nodes/voice_render.py` (registers `OTR_VoiceRender`)
- `nodes/_voice_backends/__init__.py` (driver registry)
- `nodes/_voice_backends/bark.py` (wraps current Bark impl from `batch_bark_generator.py`)
- `nodes/_voice_backends/kokoro.py` (wraps current Kokoro impl from `kokoro_announcer.py`)
- Future drivers: `cosyvoice.py`, `xtts.py`, `piper.py` (added as adopted)

*Edit:*
- `nodes/batch_bark_generator.py` — relocate impl into backend driver; remaining file becomes shim
- `nodes/kokoro_announcer.py` — same
- `nodes/_otr_voice_resolver.py` (from Cast Contract RFC) — parse `engine:preset` voice specs
- `__init__.py` — register `OTR_VoiceRender`

**Acceptance criteria:**

- [ ] `OTR_VoiceRender` registered, accepts `voice_model` enum across at least Bark + Kokoro
- [ ] Cast contract `Voice:` entries use `engine:preset` form, parsed by `_otr_voice_resolver.py`
- [ ] Per-character routing verified in a single episode: AEGEUS uses one engine, MONTY uses another, both render correctly
- [ ] Existing `OTR_BatchBark` workflows still validate and run (back-compat)
- [ ] At least one TTS upgrade candidate (CosyVoice 2/3 preferred) has a working backend driver

**Open questions:**

1. Single batch-aware node vs. one-line-at-a-time? → leaning one-line for v1 (simpler contract), batch optimization in follow-up
2. Voice preset namespace: flat `engine:preset` strings vs. structured dict? → leaning flat (workflow widget compat)
3. Where does VoiceHealth lazy-check live? Central or per-engine? → leaning per-engine (each backend has different validation needs)

---

## Status snapshot — 2026-05-03 EVENING (post BUG-027 + BUG-028 soak fixes)

**Code work for the v2.0-alpha cycle is now 19 entries deep.** All 19 BUG-LOCAL entries below are `[FIXED]` in code and pushed to `origin/v2.0-alpha`. The 2026-05-03 EVENING soak surfaced two new failure modes (BUG-027 dialogue wipe + BUG-028 FLUX legacy save paths); both were fixed in the same autonomous session per direct user directive ("yes ofrget rop8u7hnd robins just fix fix fix"). Round-robin consult was SKIPPED for both fixes per the same directive — extra verification in lieu (AST + format-safety + targeted regression + Bug Bible regression all green pre-commit). The remaining work is **a single real-run acceptance soak** to confirm the live behavior on Jeffrey's RTX 5080.

**Committed and pushed (in chronological order):**

| Bug | Phase | Commit | What it fixed |
|---|---|---|---|
| 003 | Sprint 1 | (pre-QA-pass mega-commit) | `scripts/run_comfyui.cmd` reads HF_HOME from HKCU\Environment |
| 004 | Sprint 1 | (same) | LLM script-writer OOM — `_flush_vram_keep_llm()` + `MAX_PARSE_RETRIES=2` |
| 005 | Sprint 1 | (same) | 30-word preset CHARACTER:/SCENE: enforcement + ULTRA_SMOKE strict-VOICE parse |
| 006 | Sprint 1 | (same) | `tests/conftest.py` CUDA mask; later promoted from `[PARTIAL]` to `[FIXED]` after re-verification |
| 014 | A | `d2c2df8` | Spacesaver wrong-episode wipe via global mtime ledger scan |
| 015 | B | `29295c9` | production_ledger treatment rename gap + os.replace silent split state |
| 016 | C | `3e1d995` | Filename pattern audit — slug-reconstruction regression guard |
| 017 | D | `e43695d` | MusicGen + AudioGen cache miss every run — `_cache_key` returned fresh ts |
| 018 | E | `7c84ee8` | Ledger schema bump l3-2026-05-02 + meta.paths block |
| 019 | (cleanup) | `ca85a01` | Sprint 1 full-suite acceptance — pre-existing test rot fixed |
| 020 | G | `1fabd5c` | video_engine.py procgen mp4 written to legacy `output/otr/audio/` (SOAK BLOCKER from 2026-05-02 23:00 run) |
| 021 | G | `1fabd5c` | Audio-side nodes used global mtime walker (latent BUG-LOCAL-014 wrong-episode shape in 7 sites) |
| 022 | G | `1fabd5c` | BatchHumoRender stem-swap broken when `safe_title[:40]` truncates the title |
| 023 | H | `5075b9e` | ANNOUNCER portrait wasted FLUX context + skewed scene composition |
| 024 | H | `5075b9e` | Radio bookend FLUX prompt fell back to generic when style missing OR ledger stale |
| 025 | H | `5075b9e` | LTX role prompts ignore story style + scene context (every episode looked the same) |
| 026 | G/H hotfix | `03dfbfa` | DIRECTOR_PROMPT.format crash from Phase H unescaped curly braces (caused soak crash 23:46) |
| **027** | **soak fix** | **`f1467a2`** | **Critique/revision pass strips all CHARACTER dialogue (parser regex didn't accept `[N] CHARNAME:` format + acceptance gate had no total-collapse check + revision LLM under temp=0.95 would happily produce SCENE/ENV/SFX-only output). 3-part fix: regex + total-collapse hard gate + ABSOLUTE REQUIREMENT prompt clause.** |
| **028** | **soak fix** | **`f1467a2`** | **FLUX env stills + radio bookend save to legacy flat dirs (`_legacy_stills/` + flat `otr/stills/` shared global counter) instead of per-episode workspace — VideoComposite + BatchHumo + BatchLTX all looked in the wrong places after Phase B reorg. 4-site write+read alignment fix.** |
| **078** | **portraits** | **(BUG_LOG)** | **Per-cast portrait pass (`OTR_BatchFluxPortraitRender`) — renders one clean head-and-shoulders FLUX portrait per cast member to `<ep>/portraits/<char_id>_portrait.png`, stamps `cast[i].portrait_path` into the ledger so HuMo's tier-1 lookup hits instead of falling through to env-still tier-4 stopgap.** |
| **081** | **workflow-wiring** | **`413ef3a`** | **Portrait node never executed in workflow — Node 59 `ledger_json` socket was wired to Node 12 `video_path` (a `.mp4` filesystem path) so `_load_ledger` raised `RuntimeError`; AND the Node 12 dependency forced portraits to run AT THE END of the workflow, after HuMo had already needed them. Fix (workflow JSON only): drop link 100, set `ledger_json` widget to empty for in-flight auto-pickup, re-route link 45 from `(23 → 24)` to `(59 → 24)` so chain is FLUX env stills → Portraits → UnloadAll → HuMo. Portraits confirmed live in run `signal_lost_skindeep_microneedle_..._222516` — `c01/c02/c03_portrait.png` all rendered.** |
| **082** | **filename-derivation** | **`b34d272`** | **VideoComposite missing the BUG-118 underscore-mismatch fallback. SignalLostVideo writes procgen mp4 with `__` (double underscore) before the timestamp; ledger writer uses `_` (single). VideoComposite's naive `mp4 → _ledger.json` derivation got the wrong path and crashed `derived ledger from .mp4 not found`. BatchLTXRender already had the fallback; ported it to VideoComposite (when `__` in stem, also try single-underscore variant before raising).** |
| **083** | **kwarg-signature** | **`e601ee8`** | **`probe_duration_s(...)` called with `ffmpeg=ffprobe` kwarg but the function signature names it `ffprobe`. Caught by smoke harness on first run after BUG-082 landed — TypeError silenced by strict_c7 exception handler. Fix: rename kwarg at both call sites in `video_composite.py` (lines 1033 + 1135).** |
| **084** | **composite-sync** | **`7f2d03f`** | **VideoComposite per-clip-mux concatenated 6 line clips back-to-back at t=0 with no gap-fill — audio timeline has 9.5s pre-roll music + 0.6s inter-line silences + post-roll, video timeline had none. Cumulative 9.5s+ drift made wrong-mouth-on-wrong-voice; trailing audio truncated by `-shortest`. 4-site fix: (1) LTX clip stamps real `start_s` + ffprobed `dur_s` into ledger.clips, (2) per-clip BUG-031 duration matching (already wired), (3) NEW gap-fill pass walks sorted timeline + inserts static-radio segments for gaps >0.1s + trailing tail-fill, (4) NEW duration-contract assertion before mux with tail-pad fallback if audio overruns.** |
| **085** | **hf-cache** | **`56cf493`** | **Mistral-Nemo OOM at SDPA prefill with 24 GiB allocated on 16 GiB GPU. Cause: ComfyUI Desktop's Electron parent process didn't inherit `HF_HOME` from `HKCU\Environment`, so OTR's `_load_llm` fell through to `~/.cache/huggingface` default. With wrong cache_dir + `local_files_only=True` + sharded-safetensors layout on Windows, transformers misresolved the model location, fell back to fp16 silently despite `BitsAndBytesConfig(load_in_4bit=True)` being passed. Fix: NEW `nodes/_otr_hf_env.py` (winreg HF_HOME resolver + canonical snapshot directory resolver) wired into `_load_llm` so the loader passes the absolute snapshot path (bypasses transformers' Hub-resolution). Standalone check confirms NF4 working at 7.79 GiB allocated, 280/281 modules quantized.** |

**Cumulative regression test count (post-027/028):** 155 passed in 3.27s (targeted set: production_ledger + radio_still_resolver + filename_pattern_audit + cache_key_mutations + meta_paths + ledger_rename + critique_dialogue_preservation + save_to_episode_workspace + prompt_format_safety) PLUS Bug Bible regression 24 passed / 1 skipped / 1 xfailed in 1.24s. Full `tests/` directory NOT re-run (BUG-LOCAL-006 dropdown_guardrails hang resurfaced under live ComfyUI; pre-existing, not caused by these fixes; documented as known regression in cohabit mode).

**Promotion to Bug Bible:** All 19 entries are Bible candidates. Promotion happens after the next real-run soak confirms behavior end-to-end.

### What still needs Jeffrey's hands

1. **Restart ComfyUI Desktop** so the new code is loaded (custom node `.py` files are cached in `sys.modules`; mid-process changes don't hot-reload). Especially important after BUG-028 because a NEW node class (`OTR_SaveToEpisodeWorkspace`) was registered in `__init__.py` and the workflow JSON now references it.
2. **Re-queue any episode** — the BUG-027 + BUG-028 fixes are general-purpose, no special title needed.
3. **Tail the run** and confirm the new acceptance signatures:
   - `CRITIQUE: Character line counts - draft={'CHAR1': N, ...} revised={...}` with NON-EMPTY draft dict (BUG-027 parser fix)
   - If revision wipes dialogue: `CRITIQUE: CRITIQUE_REJECTED - total character lines collapsed from N to M` (BUG-027 hard gate fires)
   - `[BatchBark] Found >=1 dialogue lines in Canonical 1.0 format` (downstream confirms dialogue survived)
   - `output/otr/episodes/<ep>/stills/full_env_NNNNN_.png` files exist with counter starting at 1 (BUG-028 writer fix)
   - `output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png` exists (BUG-028 writer fix)
   - `[BatchHumoRender] cast-still binding: N/M cast members matched to fresh stills` reports N>0 (BUG-028 reader fix)
4. **On a green soak,** promote all 19 BUG-LOCAL entries to the Bug Bible together.

### Known remaining suspects (NOT blocking the soak — Phase H+ candidates)

- `nodes/scene_sequencer.py:147` `DEFAULT_OUT = output/otr/audio` legacy default. Only matters if it's ever the actual write target.
- `nodes/batch_humo_render.py:1773` uses `otr_legacy_audio_dir()` in the auto-pick fallback. Only fires when `ledger_json` input is empty.
- `nodes/batch_ltx_render.py:300/846` use `otr_stills_dir()` / `otr_audio_dir()` with NO episode_id (returns legacy dirs).
- `nodes/video_composite.py:282` legacy audio dir scan.
- `nodes/story_orchestrator.py:6276` hardcoded `output/otr/audio/` path.
- `nodes/post_audio_video_pipeline.py:126` empty-input fallback uses mtime walker (intentional for headless mode).

These are documented in the Phase G consult (`docs/2026-05-03-phase-g-path-reorg-blast-radius__01_chatgpt.md` Section 3) and queued for a future pass.

---

## Original P0/P1/P2 sections below are NOW HISTORICAL — Sprint 1 is DONE

**Canonical narrative hierarchy** — every ledger, workflow, and doc in this repo follows this:

```
Scene  >  Shot  >  Beat  >  Clip
```

- **Scene** — high-level narrative location (`AstroTech Research Facility`, `Control Room`, ...). One per `scene_id`.
- **Shot** — continuous visual unit. Same framing, same lighting. May contain multiple speakers.
- **Beat** — single-speaker continuous turn within a shot. The unit at which the 7 s clip-fill rule applies — beats never cross speakers, so HuMo audio windows align to one voice.
- **Clip** — one HuMo render call. Length must be `4n + 1` frames (Wan VAE temporal compression of 4) and ≤ 177 (verified ceiling on 16 GB).

Every consumer of `ledger.json` must understand all four levels.

---

## Platform Pins

Lock these. Any work item that contradicts this list is wrong.

- RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, single GPU, no cloud.
- Windows, Python 3.12, torch 2.10.0, CUDA 13.0.

**Canonical stack (do not downgrade):**
- CUDA 13.x / cu130
- PyTorch cu130 matching the ComfyUI environment
- SDPA as guaranteed fallback
- SageAttention only when the cu130 wheel/source build matches Python + Torch exactly
- FlashAttention not required for shipped OTR

CUDA 13 is non-negotiable because (1) Blackwell sm_120 support is the point, (2) NVFP4 / FP4 model support in ComfyUI requires `comfy-kitchen` which requires CUDA 13+, (3) Task #2 SeedVR2 v2.5 NVFP4 path needs cu130 downstream. The cu128 SageAttention path exists in the wild and is the easier wheel target, but it belongs in a SEPARATE experimental ComfyUI folder if needed for sandbox work — never in the production OTR pipeline.

**Attention backend policy:**
- Default: PyTorch SDPA (boring, safe, in-tree).
- Preferred acceleration: SageAttention via KJNodes "Patch Sage Attention" node, tested per-workflow only.
- Do NOT use global `--use-sage-attention` unless a specific model/workflow has passed smoke testing — Triton route can produce black outputs with some models.
- FlashAttention 2/3: out of scope on Windows Blackwell. Do not chase community wheels for the shipped pipeline.
- FlashAttention 4: real and worth tracking (`pip install flash-attn-4`, exposes `flash_attn.cute` namespace), but NOT a ComfyUI production dependency yet. Older FA2-style custom nodes hard-coding the top-level import won't see it. FA4 is the future-looking transformer/training answer; SageAttention is the practical diffusion/ComfyUI answer today.
- Any third-party attention wheel must pass before shipping: import test → one FLUX smoke → one Wan/HuMo smoke → no black frames → no VRAM regression → no audio-path impact. Then it's blessed.
- Note on SageAttention wheel sourcing: `mobcat40/sageattention-blackwell` is the leading prebuilt wheel repo for sm_120, but its primary build line is PyTorch 2.11 nightly + CUDA 12.8. A cu130 build exists in that repo, but verify with smoke workflow on our pinned torch 2.10.0 / CUDA 13.0 stack before blessing.
- 100% local, offline-first, open source, no API keys for the shipped pipeline. Cloud LLMs (OpenAI / Gemini / NVIDIA NIM) are for **internal QA round-robins only**, never shipped output.
- VRAM ceiling: **14.5 GB audio** / **15.5 GB video** (lifted 2026-04-17 for the video stack only — audio stays at 14.5 GB).
- Audio is king (rule **C7**). Full narrative output must never break, shorten, or degrade. If video breaks audio, revert immediately. Audio output must remain byte-identical to v1.5 baseline at every gate.

---

## P0 — Sprint 1: make smoke green (code work, blocks everything else)

Tonight's smoke (2026-05-02, prompt_id `e6b87239-16d4-4318-bfde-134468d32904`) failed end-to-end. Six new entries in `docs/BUG_LOG.md`. The four fixes below unblock the entire BUG-128/129 acceptance verification — that work is already shipped in code, but cannot be observed because the pipeline cannot reach the audio path on the 30-word smoke.

### BUG-LOCAL-005 — 30-word ultra-smoke ScriptWriter output unparseable

**Fix:** port the BUG-007 `CHARACTER:` / `SCENE:` enforcement clause from the "short (3 acts)" prompt into the "30 words (smoke, 1 act)" preset prompt in `nodes/story_orchestrator.py`. Add a unit test in `tests/test_dropdown_guardrails.py` (or a new `tests/test_30word_preset.py`) that asserts the compiled prompt contains the literal substrings `CHARACTER:` and `SCENE:` whenever `target_length.lower().startswith("30 words")`.

**Verify:** re-queue the 30-word smoke, expect ≥3 dialogue lines parsed, ≥1 scene, 2 named characters in `ledger.cast`.

### BUG-LOCAL-004 — OOM in script writer after parse-retry loop (peak 29.5 GB on 16 GB device)

**Fix:** in `nodes/story_orchestrator.py::write_script`, add (a) explicit `_LLM_CACHE` cleanup between the OpenClose synthesizer and the main script-writer call, (b) a hard parse-retry cap (`MAX_PARSE_RETRIES = 2`) so a runaway 0-line parse fails with a clear `MAX_PARSE_RETRIES_EXCEEDED` instead of OOMing on the fourth forward pass. Audit `_generate_with_llm`'s finally block: `torch.cuda.empty_cache()` is in place but the model's internal `past_key_values` may need an explicit `del` before it fires. Log `prompt_token_count` alongside `vram_snapshot("llm_generate_entry")` so future OOMs can be bisected.

**Verify:** re-queue 30-word smoke, expect peak_gb < 14.5 across the LLM ladder; if parse keeps failing, expect `MAX_PARSE_RETRIES_EXCEEDED` not `torch.OutOfMemoryError`.

### BUG-LOCAL-006 — `pytest tests/` hangs at session-start when ComfyUI is on the GPU

**Fix:** add `tests/conftest.py` with an autouse fixture that sets `CUDA_VISIBLE_DEVICES=""` for unit tests so collection never tries to bind to GPU. Optionally also lazy-import the heavy OTR modules from `__init__.py` so collection imports don't pull torch on path-only tests.

**Verify:** `python -m pytest tests/ -q` runs to completion in <60 s with ComfyUI Desktop up on `:8000`.

### BUG-LOCAL-003 — ComfyUI Desktop launch `HF_HOME` inheritance

**Fix:** add `scripts/run_comfyui.cmd` that reads `HF_HOME` + `HUGGINGFACE_HUB_CACHE` from `HKCU\Environment` via PowerShell + `[Environment]::GetEnvironmentVariable(...,'User')` and exports them into the launch shell before `start "" "...\ComfyUI.exe"`. Document in `README.md` under "Running ComfyUI Desktop" section. Source patch into Electron is out of scope (third-party).

**Verify:** kill ComfyUI, run `scripts/run_comfyui.cmd`, queue any episode that touches an HF model — expect `LLM tokenizer loaded from cache (no HTTP checks)` log line, no `local_files_only=True failed` errors.

### Sprint 1 acceptance

All four bugs marked `[FIXED]` in `docs/BUG_LOG.md`. `python -m pytest tests/` runs to completion. 30-word smoke produces a parseable script, reaches `master_mix_per_clip_mux`, ledger.json on disk.

---

## P0 — Live-test verification (already coded, awaits clean smoke + your manual cycle)

The work below is **observation against shipped code**, not new development. Items can be checked off only after Sprint 1 lands and a clean smoke completes.

### BUG-128/129 acceptance list (locked 2026-05-01)

1. No HuMo render job ever receives the radio still (assertion in dispatch — already in `nodes/batch_humo_render.py`).
2. ANNOUNCER clips l001 and l021 in a regression episode resolve to the same announcer portrait family — no generic-blonde drift.
3. `music_*` / standalone-`sfx` segments render through the static-video path (`ledger.clips[].source_kind == "static_ffmpeg"` vs `"humo"`).
4. Final mp4's extracted audio packet-hash matches procgen's audio stream byte-for-byte.
5. Peak VRAM stays below 14.5 GB.
6. Final video duration ≈ master mix duration (no `-shortest` truncation).
7. `tests/test_dropdown_guardrails.py`, `tests/test_core.py`, and the Bug Bible regression all pass.

### Live-test verification of the radio-coverage + bit-perfect-audio architecture

Confirmation items, not new design work:

- `ledger.lines[]` carries a `speaker_role` on every entry. No nulls, no missing rows. Roles: `character` / `announcer` / `music_open` / `music_close` / `music_inter` / `sfx`.
- `ledger.meta.audio_path_selected = "master_mix_per_clip_mux"` and `audio_path_reason = "ok (zero audio re-encodes downstream of SignalLostVideo)"`.
- BUG-129 routing (locked 2026-05-02 — see Architecture Truth section below):
  - `character` lines ONLY: `BatchHumoRender` dispatches HuMo with the cast portrait. Log line: `ref=full_env_NNNNN_.png source=ledger-cast-fresh` (or composite/portrait fallback).
  - `announcer` / `music_*` / standalone `sfx` lines: `BatchHumoRender` log line shows `SKIP HuMo (role=<role>, covered by VideoComposite static-radio fill)`. `is_never_humo_role()` short-circuits before any portrait lookup. No HuMo render fires for these.
  - NO log line should ever show `source=radio-still (...)` -- if one does, BUG-129 has regressed (`_RADIO_ROLES` was re-populated in `_otr_speaker_role.py`).
- `ledger.meta.radio_bookend_prompt_source` populated with the dynamic-build branch tag (e.g. `"dynamic (style='space opera epic')"`).
- BUG-129a static-fill fires for any line with no clip on disk. VideoComposite report includes `[<n_humo> humo + <n_static> static]` summary; expect static count > 0 if any music_*/sfx lines exist.
- BUG-128 tail-pad: VideoComposite report shows `tail-pad: +0.500s on <line_id>` after the pillarbox loop completes. The line_id matches the actual surviving last clip, not necessarily the last in the original timeline.
- Music tracks > 7s show up as multiple chunked entries (`music_open_001`, `music_open_002`, ...) — chunking math fired.
- ffprobe on the final mp4: video + audio streams both present; final mp4 audio `codec_name == aac` (passthrough from procgen); duration ≈ master mix duration.
- No `[VideoComposite] master_mix_per_clip_mux FAILED` in the log. With `strict_c7=True` (default), any failure would have raised.

### P1 audio pipeline — live-test verification (7 items, code-shipped on `v2.0-alpha`)

| Item | Confirmed in code | Awaits real-run observation |
|---|---|---|
| `min_line_count_per_character` self-critique guard | `nodes/story_orchestrator.py:6624` (default=2) | CRITIQUE_REJECTED log line on a real run where revision drops a character below 2 lines |
| Director JSON schema + validator | `_DIRECTOR_SCHEMA` at `:9239`, `_validate_director_plan` at `:10332` | DIRECTOR_SCHEMA_REPAIR log line on a malformed director plan |
| Length-sorted Bark batching | `nodes/batch_bark_generator.py:478` `@vram_sentinel` decorator | Throughput improvement vs unsorted baseline (10-15% expected) |
| VRAM-Sentinel decorator | `nodes/_vram_log.py::vram_sentinel`, used in 4 nodes | VRAM_SENTINEL_ENTRY/EXIT lines bracketing every decorated phase |
| High-creativity soak profile | "maximum chaos" in CREATIVITIES dropdown, temp 0.95 | One soak run on this tier; expect format-resilient output (no SFX loops, no [ACT N] injection) |
| Per-LLM-call VRAM snapshots | `vram_snapshot("llm_generate_entry/exit")` at every `_generate_with_llm` boundary | Snapshot lines visible in runtime log; peak summable across phases |
| ScriptCritic + Reviser advisory gate | `nodes/script_critic.py`, `block_on_reject` defaults False | 3-5 successful runs; flip to True after critic rate stabilizes |

### Open follow-ups (P2/P3-flavored, not blocking smoke green)

- **Audio codec ffprobe pre-flight** (P2) — confirm procgen audio stream is AAC before `-c:a copy` mux. One-line subprocess.run + assertion. Trivial; deferred until first run confirms procgen output codec.
- **Post-mux audio stream identity validation** (P3) — extract per-stream packet hash on procgen vs final mp4, fail tier on mismatch. Concrete proof of bit-identity. Ship as a separate validation node since the ffmpeg incantation needs care on Windows.
- **Low-motion observability for radio HuMo clips** (P3) — frame-difference metric on non-dialogue clips so "static" failures (Whisper OOD producing flat frames) surface as warnings instead of going unnoticed. No behavior change.
- **HuMo continuity layer for >7s narrative beats** (v2.0-beta) — hybrid blending across HuMo windows so 30s narrative beats don't show 7s jump-cuts. Decoupled from the audio path; gates "production unattended."
- **Per-scene environment FLUX still + LTX/zoompan animated background** (v2.0-beta) — bottom layer under the HuMo center pillarbox in dialogue windows.
- **Procgen-CRT lighten layer on top** (v2.0-beta) — audio-reactive scanlines + flicker as the SIGNAL LOST signature.
- **Drifted-filename smoke for BUG-LOCAL-118** — force an underscore-drifted .mp4 stem to verify the fallback chain fires before relying on it in a long soak.
- **Reconcile `16294df` ROADMAP-vs-git-log mismatch** — git log says "BUG-LOCAL-112 news-history reset"; prior narrative had it as "Wire ScriptCritic." Likely a rebase artifact. Decide canonical message before the next QA pass walks the history.

#### Hardware floor (locked 2026-04-25, do not relitigate)

- HuMo 14B fp8 e4m3fn scaled (Kijai) — `Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors`. Stock `UNETLoader`. Tuned by Kijai for 16 GB cards.
- Fallback ladder (kept on disk, do NOT delete): `humo_17B_fp8_e4m3fn.safetensors` (highest quality, slower ~6 min/clip), `Wan2_1-HuMo-17B_Q5_K_M.gguf` (speed-tuned).
- Stable shape: `length=97` (3.88 s @ 25 fps), 480x832, batch=1. Or `length=177` at 640x640 (7 s, OOD but verified working).
- Frame count must be `4n + 1`. Helper `humo_length_for_dur(dur_s)` snaps. Cap mirrored to `7.0s` in EpisodeAssembler music chunking.
- Per-step: 42 s. Per-clip: ~4:30 native, ~6:15 in TEST_humo. Cold load: ~50 s.

---

## Sprint 2 — harness + test-rot cleanup

Pre-existing test infrastructure rot blocking the regression contract from being measurable.

### BUG-LOCAL-001 — 8 stale test collectors importing `otr_v2.visual`

**Fix:** delete the 8 orphan test files (`tests/test_anchor_gen.py`, `test_camera_path_determinism.py`, `test_character_regression.py`, `test_cold_open_canary.py`, `test_episode_dry_run.py`, `test_lhm_monitor.py`, `test_three_minute_continuous.py`, `test_visual_phase_a.py`) OR rewrite them against the active video-stack code path. `otr_v2/visual/` was deleted in commit `7706660`; the test files were never updated. Triage during the cleanup: any test still asserting current behavior gets ported, the rest get deleted.

**Verify:** `python -m pytest tests/ --collect-only -q` reports zero collection errors.

### BUG-LOCAL-002 — `scripts/soak_operator.py` + `scripts/supersoaker.py` widget indices stale

**Fix:** delete both scripts. Replace with `scripts/otr_api.py` containing: (a) `patch_widget(workflow, node_id, widget_name, value)` that reads `/object_info` for the node's input order and writes by name (no fragile `WV_*` positional indices), (b) `workflow_to_api_prompt(workflow, schemas)` ported from soak_operator's working converter, (c) `submit_prompt(api_prompt) -> prompt_id` and `poll_history(prompt_id, timeout_s) -> status` helpers. Rewire `scripts/queue_smoke.py` onto `otr_api.py`.

**Verify:** running `scripts/queue_smoke.py` against `otr_scifi_16gb_full.json` produces a `/history` entry with `current_inputs` matching the patched values exactly (`target_words=30`, `num_characters=2`, `target_length="30 words (smoke, 1 act)"`).

### Triage 14 `tests/test_backend_dispatch.py` failures (logged, root cause not yet bisected)

Investigate during Sprint 2 — may be tied to the `otr_v2.visual` rot or to backend-dispatch refactors. Captured at baseline 2026-05-02: pytest -q output showed `FFFFFFFFFFFFFF` (14 failures) for this file. After Sprint 1's conftest CUDA-mask fixture is in place, re-run with `--tb=short` to capture exception types; fix or mark `xfail` with reason.

---

## Sprint 3 — MEGA-SPRINT: status (2026-05-02)

**Wiring SHIPPED on `v2.0-alpha`. Live acceptance BLOCKED on BUG-LOCAL-010 (pre-existing LLM-phase OOM regression).**

The Sprint 3 mega-sprint code is in place: LTX wiring (LowVRAMCheckpointLoader + OTR_BatchLTXRender), RTX VSR upscale (OTR_RTXUpscale), VideoComposite rewired downstream of LTX, anti-clobber + pipe-deadlock + cache-buster fixes from the round-robin consult. AST-clean, regression-clean (225 tests pass), workflow JSON valid, all three new nodes register, ComfyUI accepts the patched workflow at /prompt. The smoke OOM'd at OTR_LLMScriptWriter (BUG-LOCAL-010 in `docs/BUG_LOG.md`) -- the wiring code never executed because the LLM phase couldn't progress.

Once BUG-LOCAL-010 is fixed in a separate bisect window, re-queue the same workflow JSON and the S3.x acceptance bullets become directly observable. The full shipped scope and consult transcripts live in `docs/ROADMAP_HISTORY.md` under the 2026-05-02 mega-sprint entry; the Architecture Truth (locked 2026-05-02) is preserved there too.

**Locked-but-not-yet-verified S3.x acceptance bullets** (move to Done after a clean post-LLM-fix smoke):

- `ledger.clips[].source_kind == "ltx"` on announcer / music / sfx rows.
- VideoComposite report logs `[N humo + N ltx + N static]`.
- Pre-upscale ffprobe: width=832 height=480.
- Post-upscale ffprobe: width=1920 height=1080.
- Bypass path produces 832x480 unchanged.
- Audio byte-identical between pre- and post-upscale (stream MD5 match).
- Peak VRAM stays below 14.5 GB audio / 15.5 GB video.

### Architecture Truth (locked 2026-05-02 — do not relitigate)

The decisions below are settled. Any future session that tries to "improve" them must show a real-run failure first, not theory.

**Resolution policy — native 832x480 end-to-end:**
- `SignalLostVideo` procgen: 832x480 (canonical OTR landscape).
- `OTR_BatchLTXRender`: 832x480 (matches procgen + canvas; no upscale at composite time).
- `VideoComposite` canvas: 832x480 default (was 1920x1080 — corrected to native).
- `BatchHumoRender`: stays portrait pillarbox (480x832 internal, 832x480 letterboxed on canvas).
- `BatchFluxRender` cast portraits: 1024x1024 (FLUX-native square; HuMo `ref_image` is face-centered conditioning, not first-frame I2V).
- `BatchFluxRender` radio bookend: renders at **1248x720** then Lanczos-downscales to 832x480 in-node. Pixel budget locked — do NOT switch to 1344x768 or 1280x720.

**Role routing — `_NEVER_HUMO_ROLES` is the single source of truth:**
- Defined in `nodes/_otr_speaker_role.py` as a frozenset including `announcer`, `music_open`, `music_close`, `music_inter`, `sfx`. `_RADIO_ROLES` is empty (defense-in-depth).
- `BatchHumoRender` short-circuits via `is_never_humo_role()` BEFORE any portrait lookup. HuMo's `ref_image` is face-locked conditioning — it cannot animate the radio still as a non-face reference (verified in `comfy_extras/nodes_wan.py:1070-1108`).
- Coverage for non-character lines: `OTR_BatchLTXRender` (motion radio loops) takes precedence; `VideoComposite` static-radio fallback (BUG-129a) covers any line LTX skipped.

**LTX seamless-loop architecture — radio still as both start AND end keyframe:**
- `OTR_BatchLTXRender` uses `LTXVAddGuide` twice in the conditioning chain: `frame_idx=0` with strength 0.75 (start), `frame_idx=-1` with strength 0.6 (end). Both reference the same radio still PNG so the clip loops cleanly back to the bookend frame — no visible cut at loop boundary.
- Frame-count rule: `8n + 1` (LTX VAE temporal compression of 8). `LTX_MAX_FRAMES = 177` to match HuMo's verified ceiling on 16 GB; do NOT raise to 257 without a fresh VRAM smoke.
- Tiling: `LTX_TILE_SIZE=512`, `OVERLAP=64`, `TEMPORAL_SIZE=4096`, `TEMPORAL_OVERLAP=8` (Goofer-proven Blackwell params; see Jeffrey's `ComfyUI-Goofer` project).
- Strict teardown after the per-line loop: `unload_all_models()` + `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()` in `finally`. LTX must fully release VRAM before the next pipeline stage.

**Loader policy — UNETLoader chain, NO C2 carve-out:**
- LTX 2B fp16 wires through `UNETLoader` + `CLIPLoader` (T5) + `VAELoader`. NOT `CheckpointLoaderSimple`.
- Reason: C2 stays intact (no carve-out drift); split-load lets ComfyUI offload T5 / VAE independently; bundled-load on a hot HuMo cache is the OOM shape C2 was written to prevent.

**DAG sequencing — `humo_clips_dir` optional dependency edge:**
- `OTR_BatchLTXRender` accepts an optional `humo_clips_dir` STRING input. When present, LTX waits for HuMo to finish writing its clips before starting — this is a pure dependency edge, not data flow. Sequential model load: HuMo loads → renders character clips → unloads → LTX loads → renders radio loops → unloads.
- LTX clips stamp `ledger.clips[].source_kind == "ltx"` (NOT `"humo"`). One-line clip-emit fix in `batch_ltx_render.py`; ship in the same commit as the wiring.

**Round-robin ladders (locked 2026-05-02):**
- OpenAI: `gpt-5.5` via `/v1/responses`. Gemini: `gemini-3.1-pro-preview-customtools`. NVIDIA: `nvidia/llama-3.3-nemotron-super-49b-v1.5`.
- See `scripts/_consult_round_robin.py` + `scripts/_consult_nvidia.py`. Typed error logging (404/400/403/429 fall through; 401/transport re-raise).
- Internal QA only — never shipped output.

### S3.1 — Wire `OTR_BatchLTXRender` into `workflows/otr_scifi_16gb_full.json`

Node already built (`nodes/batch_ltx_render.py`, registered `__init__.py:155`). This is JSON wiring, not Python.

**Scope:**
1. Add `UNETLoader` + `CLIPLoader` (T5) + `VAELoader` triplet for LTX 2B fp16. Distinct `_meta.title` per loader.
2. `EpisodeAssembler.ledger_json` → `OTR_BatchLTXRender.ledger_json`.
3. `BatchHumoRender.clips_dir` → `OTR_BatchLTXRender.humo_clips_dir` (optional STRING dependency edge; sequencing only).
4. `OTR_BatchLTXRender.clips_dir` → `VideoComposite` as sibling source to HuMo's `clips_dir`. VideoComposite already merges by `line_id`.
5. Add `humo_clips_dir` optional STRING to `INPUT_TYPES` if missing.
6. Confirm clip-emit stamps `source_kind="ltx"`.

**Acceptance:**
- `ledger.clips[].source_kind == "ltx"` on announcer / music / sfx rows.
- Final mp4 shows LTX motion on those windows, looping seamlessly back to bookend.
- VideoComposite report logs `[N humo + N ltx + N static]`.
- Peak VRAM < 14.5 GB.
- Audio byte-identical to no-LTX baseline.

### S3.2 — FLUX radio bookend visual confirmation

Already coded. Observation only on next smoke.

**Acceptance:**
- Saved radio bookend PNG is exactly 832x480.
- Image is sharp (Lanczos downscale, not box / nearest).
- Same PNG hash feeds VideoComposite static fallback AND LTX start/end keyframes.

### S3.3 — 832x480 native end-to-end audit

**Acceptance:**
- `ffprobe` on the final composited mp4 (pre-upscale): `width=832 height=480` exactly.
- All segments (procgen / LTX / HuMo-pillarboxed / static-radio) composite onto 832x480 with no scale ops.

### S3.4 — RTX VSR ULTRA upscale to 1080p

Wire NVIDIA's RTX Video Super Resolution ULTRA ComfyUI node as the final stage after VideoComposite. ~0 GB VRAM (HW-accelerated via RTX driver), near-real-time. Output is the saved deliverable.

**Scope:**
1. Add RTX VSR ULTRA node to `workflows/otr_scifi_16gb_full.json` after VideoComposite's mp4 output.
2. Target resolution: 1920x1080 (16:9 from 832x480 source — the upscaler's standard 1080p mode).
3. Workflow toggle (Ctrl+B bypassable) so the user can disable per-run for raw 832x480 output.
4. Saved deliverable: `output/episodes_for_obs/<ep>/<ep>_1080p.mp4` when upscale on; `<ep>.mp4` when bypassed.

**Acceptance:**
- `ffprobe` on the upscaled mp4: `width=1920 height=1080`.
- Audio stream byte-identical to pre-upscale mp4 (RTX VSR is video-only; passthrough audio).
- Wall-clock for upscale stage: target near-real-time (≤ episode duration on a 5 min episode).
- Bypass path produces the original 832x480 mp4 unchanged.

**Deferred (NOT this sprint):** SeedVR2 v2.5 NVFP4 quality upscale lane — adds as second toggle once the RTX VSR fast path is validated. Wall-clock for SeedVR2 is ~2-3 h per 5 min episode, so it needs its own session and a dedicated VRAM smoke.

### B1 — Workflow JSON path scrub — VERIFIED SHIPPED 2026-05-02

Re-audit on 2026-05-02 found zero hardcoded user paths in `workflows/otr_scifi_16gb_full.json`, `workflows/otr_humo_smoke.json`, `workflows/otr_flux_smoke.json`, or `workflows/otr_humo_radio_experiment.json`. The "Resonance Chamber" `LoadAudio` widget on the smoke workflow already has an empty default. The portability concern is closed; everything goes through `OTR_OUTPUT_DIR` / `folder_paths.get_output_directory()` as designed.

The only remaining B1 work is documentation: `README.md` should explicitly state the env override pattern (`OTR_OUTPUT_DIR=/path/to/out`) for cloud / non-Windows installs.

---

## P2 — Continuity layer

Blocked on video-stack maturity. Design begins once stack empirics exist from the live-test cycle.

| Item | Summary |
|---|---|
| Scene-Geometry-Vault | Series-scale persistent geometry vault so Act 3's bridge matches Act 1's bridge across episodes. Seeded by FLUX anchor outputs |
| Style-Anchor cache | Reuse engine over the vault. Same geometry, N relight passes. `style_anchor_hash` in Director schema keys the split |
| Head-Start async pre-bake (Phase B.5) | Kick off VisualBridge on `outline_json` while ScriptWriter + Director run. Wall-clock win. Blocked on vault stability |
| ASCII sanitizer in prompt_compiler | Strip non-ASCII before Tencent text encoders. Preserve case. Collapse whitespace |
| Diff 3 — spine ledger-stamping + schema bump l3 → l4 | New ledger fields (`outline`, `beats[]`, `spine_meta`) + bundled metadata (`episode_title`, `meta.gen_params`, `meta.news_seed`, `meta.bug_109_retries`, `meta.word_ratio_pct`, `meta.title_source`, `meta.episode_breakdown_s`). See `docs/2026-04-29-spine-ledger-stamping-ticket.md`. **Unblocked by:** 2-3 real-episode runs of `voice_warnings[]` + Mistral-Nemo + Gemma 4 E4B both PASSing the LLM edge-case matrix + v2.0-alpha video stack feature-complete |

---

## P3 — Experiments & polish

| Item | Summary |
|---|---|
| `torch.compile` on Bark sub-models | `mode="reduce-overhead"` on semantic, coarse, fine acoustic. Needs isolated A/B timing; variable-length loops may fight the compiler |
| Skip/shorten Bark fine acoustic pass | Fine pass detail that AudioEnhance destroys via tape emu / LPF / Haas. Needs listening test, not spectrogram |
| `episode_title` socket on `OTR_SignalLostVideo` | Replace implicit `script_json` title-token read with explicit socket. v2.1 cleanup |
| News-history fuzzy dedup for syndication edge case | URL dedup catches direct repeats; same content with different URLs needs a fuzzy headline match |
| Empty-section pruning in filtered rubric | 1-character runs keep `### Ensemble-voice collapse` heading after all 3 rules filter out. Wastes tokens, doesn't break anything |
| VideoComposite cleanup deletion logic | Widget shipped (`cleanup_clips_after_assembly`), no-op for now. Wire actual deletion when stable enough to trust |
| Auto-update `OTR-CANON.md` from passing critic verdicts | `_canon_update()` helper exists in `script_critic.py` but is intentionally not called yet. Wire in once 3-5 runs of critic data accumulate |
| Tune `_MODEL_CONTEXT_CAPS` from real `OTR_VRAMContextTest` data | Currently conservative defaults |
| Update stale dropdown-guardrail tests in same commit as widget changes | Lesson from 2026-04-30: when widget mins/defaults change, update `tests/test_dropdown_guardrails.py` in the same commit so the test suite never drifts behind production |

---

## v2.0 release blockers

### B0 — Portrait pass polish (post BUG-LOCAL-081 verification)

**Status:** queued 2026-05-03 LATE EVENING. Discovered live in run `signal_lost_skindeep_microneedle_..._222516` after BUG-081's wiring fix landed and portraits actually rendered for the first time. Two cosmetic-but-real issues:

**B0.1 — Portraits duplicated into `stills/` as `full_env_NNNNN_.png`.** When I re-routed link 45 from `(Node 23 → Node 24 UnloadAll)` to `(Node 59 → Node 24 UnloadAll)`, the downstream `OTR_SaveToEpisodeWorkspace` (Node 25) inherited the new IMAGE source. It now writes the portrait_batch tensors out as `stills/full_env_00001-3_.png` thinking they're env stills. Real portraits are still correctly at `portraits/c0X_portrait.png`, so HuMo's tier-1 lookup is unaffected, but it's ~6 MB of duplicate data per episode with misleading filenames. **Fix options:** (a) detect the source node in SaveToEpisodeWorkspace and route portrait_batch tensors to `portraits/` instead of `stills/`, OR (b) leave SaveToEpisodeWorkspace wired only to genuine env-still sources and let the portrait node manage its own saves (it already does — `<ep>/portraits/<char_id>_portrait.png`). Option (b) is cleaner: just unwire link 46 from UnloadAll → Node 25 when env stills are skipped.

**B0.2 — `skip_announcer=True` widget never fires.** Cast field `cast[i].speaker_role` is empty in the ledger (`role=` for all entries — confirmed via PowerShell on the 222516 run). The portrait node's announcer-skip logic has nothing to match against, so it renders a portrait for ANNOUNCER (c01) too. Cost: ~10s extra FLUX time + one unused 1024x1024 PNG per episode. **Fix:** either (a) populate `speaker_role` field on cast at LLMDirector time (canonical fix; benefits any future role-aware logic), OR (b) fall back to `name.upper() == "ANNOUNCER"` substring match in the portrait node when `speaker_role` is empty (cheap defensive fix). Probably both — populate the field upstream AND keep the substring fallback as defense-in-depth.

**Why release blocker:** v2.0 ships when the per-episode workspace is clean. Phantom env stills + unused announcer portrait are both visible to anyone who opens the workspace folder, and both make the JSON layout harder to reason about during debugging. Cheap to fix once HuMo soak completes.

### B1 — Generic / relative paths (no Windows-hardcoded absolutes)

**Status:** Step 0 paths refactor shipped 2026-04-28 (`70f4a5c`) — `nodes/_otr_paths.py` helper module with resolution order: `OTR_OUTPUT_DIR` env → `folder_paths.get_output_directory()` → walk-up to ComfyUI root → cwd fallback. ~12-15 hardcoded `r"C:\Users\jeffr\..."` strings replaced.

**Remaining:** see Sprint 3 above.

**Why it's a release blocker:** every Windows-absolute path is a portability blocker for any non-Jeffrey user (Linux/Mac/RunPod/cloud) and a portability blocker for the 8GB-tier work. v2.0 cannot ship while paths are user-and-OS-specific.

### B2 — 8GB-VRAM-class user experience

**Stance:** v2.0 doesn't release until 8GB-class users get an enhanced visual output too.

**Architecture (Locked 2026-04-30):** Single master JSON with bypassable video-stack groups. Shared audio chain → procgen, then multiple side-by-side render groups — each group bypassable via Ctrl+B. Final VideoComposite takes whichever group is active.

**Stance:** 8 GB tier does NOT get "full animated backgrounds" or generative character video. They get an **enhanced visual mode** optimized for their VRAM limits: still + parallax + interpolation for motion, with optional Wan 2.2 5B B-roll for users who want to gamble on render time.

**Do NOT offer:** HuMo, LTX-2, LTX-2.3, or 14B Wan to 8 GB users. The support burden and OOM risk are too high.

**Locked picks (2026-04-30, after evaluating LTX 2.3, LTX-2 19B, ERNIE Image, NVIDIA CES 2026 NVFP4, and round-robin consult on background models):**

| Component | 16 GB tier | 8 GB tier | Why |
|---|---|---|---|
| **Stills** | **NVFP4 FLUX.2** (RTX 50 Series, ~5 GB; falls back to FLUX-fp8 ~12 GB if NVFP4 unavailable) | **FLUX.1-dev Q4_K_S** (city96 GGUF, ~5-6 GB) | FLUX is the visual anchor for both tiers. NVFP4 is the new official quantization NVIDIA announced at CES 2026 — 3x faster, 60% less VRAM than fp8 on RTX 50 Series. Q4_K_S is the safe 8GB GGUF option. |
| **Motion** | **HuMo 14B fp8** + master_mix_per_clip_mux + LTXV background layer | **Still + Parallax + Interpolation** (deterministic Ken-Burns + frame interp on FLUX stills) | HuMo for 16 GB character lip-sync. 8 GB gets safest, fastest, most deterministic motion — high quality, zero VRAM spikes, no diffusion-per-beat. |
| **Optional B-roll** | n/a (HuMo covers all character beats; LTXV covers backgrounds) | **Wan 2.2 5B TI2V** (native ComfyUI template, optional toggle) | Strictly optional B-roll lane for 8 GB users who want generative motion on non-dialogue beats. Slow, not guaranteed; document expectation upfront. |
| **Upscale — Speed option** | **RTX Video Super Resolution ULTRA** (~0 GB, HW-accelerated, target 4K, real-time) | **RTX VSR ULTRA** (same node, same zero VRAM cost) | Default. NVIDIA CES 2026 ComfyUI node. Whole-episode upscale, near-real-time, ships with RTX driver. Use this when speed matters more than maximum diffusion-based detail. |
| **Upscale — Quality option** | **SeedVR2 v2.5 NVFP4** (7B, ~6 GB on RTX 50 NVFP4, ~78 s per 65-frame 720p→1080p clip — full episode ~2-3 h on a 5-min run) | not viable on 8 GB | Whole-episode upscale via the diffusion upscaler. Quality king for AI-generated content. SeedVR2 v2.5 NVFP4 support landed via [PR #486](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/pull/486). On RTX 50 NVFP4: 3x faster + 60% less VRAM vs fp16 baseline. |

Both upscale options run on the WHOLE episode (every clip), exposed as a workflow toggle so the user picks per-run. Default = RTX VSR (fast). Quality run = SeedVR2 v2.5 (slow but state-of-the-art on AI content). Either can be toggled off entirely for raw 480x832 / 720p output.
| **TTS / Audio** | Bark + Kokoro + MusicGen + AudioGen → master mix (canonical) | Same | OTR's TTS pipeline is the project. NEVER replaced by model-internal A/V generation (LTX-2's prompt-driven audio is a paradigm mismatch). |

**Picks REJECTED after evaluation:**
- **LTX 2.3 22B distilled** — smallest GGUF (Q5_K_M ~14 GB) doesn't fit 8GB; "distilled" = step-distilled NOT param-distilled; 22B is the only param size Lightricks publishes.
- **LTX-2 19B distilled (Kijai)** — Q4_K_M ~12 GB still over 8GB.
- **LTX-2's built-in audio for character dialogue** — model GENERATES speech from text prompt, doesn't accept input audio. Replacing OTR's TTS would lose Bark/Kokoro voice control + script→voice mapping. Unless an audio-input ControlNet/LoRA ships, LTX-2 is visuals-only for OTR.
- **Wan 2.2 14B GGUF Q3/Q4** — RAM-thrashes on Windows under aggressive offload; support-ticket bait.
- **FLUX.1-dev Q5_K_S** — over 8GB budget once T5 + VAE + OS overhead added.
- **Z-Image Turbo / PixArt-Sigma** — weaker prompt-adherence than FLUX for radio-drama series consistency.
- **ERNIE Image 8B** — parked pending model card review (Jeffrey to provide spec link).

**Acceptance for 8GB path:**
- Full audio pipeline (LLM + Bark + AudioGen + MusicGen + SceneSequencer + EpisodeAssembler) — same as 16 GB.
- SignalLostVideo procgen base — same.
- Stills via FLUX.1-dev Q4_K_S; video via Wan 2.2 5B (atmospheric B-roll / scene loops).
- Final mp4 lands in `output/episodes_for_obs/<ep>/<ep>.mp4` same as 16 GB.
- Wall-clock expectation: FLUX still ~45-90 s/still, Wan 5B clip ~4-8 min/clip (significantly slower than 16GB tier; document upfront).

**Distribution requirements before tagging v2.0:**
- Pin exact ComfyUI version + GGUF model versions in README; include checksums for the GGUF files.
- README must set time expectations explicitly so 8GB users don't think the run hung.
- Both tier workflows live in the same JSON; the README screenshot shows the "8GB mode" group toggles to enable.

**Related:** flip default `optimization_profile` to `Pro (Ultra Quality)` once 16 GB FULL has shipped clean — Jeffrey: *"I almost feel we should default to Pro Ultra"*.

### B3 — v2.0-alpha deferred cosmetic cleanup tail (non-blocking)

**Status:** opened 2026-05-09. Catch-bucket for cosmetic / stylistic items surfaced during the LPL sprint that were deliberately deferred rather than fixed mid-flight. Non-blocking for the FULL acceptance soak; clean these up before the v2.0 cut at Jeffrey's discretion.

**B3.1 — `nodes/_otr_model_loader.py` `make_generate_fn` torch import idiom.** Inside the inner `generate_fn` closure, the `with` statement uses `with __import__("torch").no_grad():` after a preceding `import torch  # noqa: F401`. Both forms work and refer to the same cached `sys.modules["torch"]`, but the `__import__` form is unusual stylistically — the local `torch` name is already bound from the explicit import two lines above and could be used directly: `with torch.no_grad():`. Decision was to preserve the spec verbatim during Phase 1+2 so the diff stayed reviewable. Cleanup: replace the `__import__("torch").no_grad()` call with `torch.no_grad()` and drop the `# noqa: F401` once the local `torch` is actually referenced. Sized: ~2 line edit + re-run self-test (7 tests). **Why non-blocking:** functionally equivalent; only affects readability.

**B3.2 — `_unload_llm` cache-schema asymmetry (story_orchestrator.py line 3019 vs line 3075).** The module-level `_LLM_CACHE` declaration includes `budget_profile` and `VERSION` keys; the re-assignment inside `_unload_llm` drops those two keys. Consumers in `_load_llm` use `.get(...)` so the asymmetry is tolerated, but it always logs a delta on the next reload because the cache reads "missing keys" as drift. Latent issue identified during Task 4 Step 4a recon (2026-05-09). Cleanup: align the post-unload schema with the declaration so the cache-mismatch diagnostics don't fire spurious deltas. **Why non-blocking:** consumers tolerate it; cosmetic log noise only.

**Why this section exists:** the LPL sprint surfaces these mid-flight; logging here keeps them out of the in-flight code review and out of mid-soak edits, while ensuring they don't get lost before v2.0 cut.

---

### B4 — LLM-prompt audit pass (contract verification, post-consumer-green)

**Status:** queued 2026-05-09. Gated behind "all 7 consumers ship green AND the patterns doc is final." The patterns doc (`docs/2026-05-09-ledger-consumer-rewrite-patterns.md`) and L3 helper module (`nodes/_otr_ledger_consumers.py`) are SHIPPED as of 2026-05-09; gate now reduces to "all 7 consumers ship green" (3/7 done, 1 in flight, 3 remaining as of 2026-05-10). Audit-only pass when triggered; no edits in this round. A second pass applies fixes after the audit doc is reviewed.

**Goal:** confirm every hardcoded LLM prompt string in the codebase references the **L3 ledger schema** correctly — field names, role strings, format conventions. No drift between what we tell the LLM to produce and what the consumers expect to read. The patterns doc is the canonical reference; if a prompt and the patterns doc disagree, the patterns doc wins (it is locked from real code that runs).

**Files to grep for prompt strings (system prompts, user prompts, format instructions, rubric text):**

- `nodes/_otr_outline.py` — writer outline prompt
- `nodes/_otr_line_composer.py` — per-line dialogue prompt
- `nodes/script_critic.py` — critic system prompt + rubric
- `nodes/_otr_legacy_writer.py` — the old `SCRIPT_SYSTEM_PROMPT`. Verify it is not load-bearing for the new path; if so, audit it; if dead, document as deprecated.
- `nodes/story_orchestrator.py` — any module-level prompt constants (`SCRIPT_SYSTEM_PROMPT`, `SCAFFOLDING_PREAMBLE`, etc.)
- Any `LLMDirector` prompt material if reachable from the new path

**For each prompt found, audit:**

1. Does it reference `[VOICE: NAME, traits]` format? Confirm: `NAME` is the cast member name from `cast[i].name`, NOT `char_id`. `traits` is what the writer derives from `beat.mood`. If the prompt says `[VOICE: c01, ...]` that's wrong — should be `[VOICE: MARLOW, ...]`.
2. Does it reference `speaker_role` values? Confirm exact strings match `VALID_SPEAKER_ROLES` from `_otr_speaker_role.py`: `character`, `announcer`, `music_open`, `music_close`, `music_inter`, `sfx`. No `narrator`, `voiceover`, `music_intro`, or other near-misses.
3. Does it reference any ledger field by name? `line_id`, `char_id`, `text`, `traits`, `beat_id`, `shot_id`, `cast`, `lines`, `meta`, `episode_id` — all must match the schema. No abbreviations, no plurals/singulars swap.
4. Does the format example show the right column order if applicable? e.g. `[VOICE: NAME, traits] dialogue` — not `[VOICE: traits, NAME] dialogue`.
5. Does the prompt reference any DEAD field names (`script_lines`, `content`, `type`, `scene_break`) that came from the old parser-list contract? If yes, those references are stale — flag for rewrite.

**Output:** a doc at `docs/2026-05-XX-llm-prompt-audit.md` listing each prompt found, with:

- File + line range
- Current text snippet (key parts)
- Audit verdict — `CLEAN` / `NEEDS UPDATE` / `DEAD CODE`
- Recommended fix if any

**Hard rule:** no edits in this pass. Audit only. Review the doc, then a second pass applies fixes.

**Acceptance:** doc exists at the path above, every prompt source file in the list is covered, every prompt has one of the three verdicts assigned, recommended-fix column populated for every `NEEDS UPDATE` row.

---

## v2.0-beta candidates

### Animated backgrounds (3-layer composite, 16 GB only)

Promotes the current 2-layer composite (procgen-base + HuMo-overlay, BUG-092) into a 3-layer composite. **8 GB tier does NOT get a background layer** (procgen sides only — keeps 8 GB lean).

```
TOP:    Procgen / CRT audio-reactive overlay -- `lighten` blend, ~0.3 opacity
MID:    HuMo lip-sync portrait -- center pillarbox during dialogue, opaque
BOTTOM: Animated background (model TBD) -- full canvas, opaque
```

**Why CRT-on-top in lighten mode is more truthful:** a failing broadcast's scanlines + audio-peak flicker should cover the WHOLE frame including the speaker's face — the interference doesn't politely stop at the pillarbox edges. Lighten mode takes max(CRT, underlying) per channel so artifacts ride on top without erasing detail.

**Render budget (locked 2026-04-29 PM — render-native + slow-mo, model-agnostic):**
- Render at the chosen model's native fps, then slow to 12 fps via ffmpeg `setpts=PTS*2,fps=12`. The slow-mo IS the SIGNAL LOST broadcast-degraded aesthetic.
- 1-2 clips per SCENE (not per shot). Loop across the scene's duration via `-stream_loop -1` with optional crossfade or ping-pong reverse.
- For LTX: 193 frames per clip = 8 sec native = 16 sec apparent after 2× slow-mo. LTX uses 8× temporal VAE compression so frame counts must be `8n + 1`. 193 = 24*8 + 1. Max 257.
- For Wan: frame-count math TBD per model card during implementation.
- Distilled 4-8 steps (default 6 for LTX; Wan TBD).

**Per-episode wall-clock estimate:** smoke (1 scene) ~50 s; short (3 scenes) ~2.5 min; medium (5 scenes) ~4 min. Negligible vs HuMo (~10 min per dialogue line).

**Frame-count widget shape (model-specific names locked at impl):**
```
frames:         dropdown of valid frame counts for chosen model
steps:          distilled step dropdown
slow_mo_factor: float (default 2.0)
target_fps:    int (default 12)
```

#### Background-model selection — LOCKED 2026-04-30

**Round-robin verdict:** Keep the background layer cheap, stable, and visually appropriate for being blurred/degraded under the HuMo dialogue pillarbox. Foundation-model chasing for a layer that gets slowed to 12 fps and composited under a foreground is the wrong engineering bet.

| Candidate | Size on disk | Peak VRAM | Role | Verdict |
|---|---|---|---|---|
| **LTXV 0.9.x 2B distilled fp16** | ~5 GB | ~7-8 GB w/ VAE | **Default (16 GB)** | **LOCK.** Fits the degraded-broadcast aesthetic perfectly. 193 frames (8n+1), 4-8 distilled steps, then ffmpeg slow-mo to 12 fps. Both ChatGPT + Gemini endorsed. |
| **Still + Parallax + Interpolation** | ~5-6 GB (FLUX still only) | ~7 GB | **Default (8 GB)** | **PLAN B / 8 GB PATH.** Lowest risk, highly deterministic Ken-Burns + frame interp on FLUX stills. Likely enough motion for radio drama without diffusion overhead. ChatGPT's smallest-change biggest-payoff suggestion. |
| **Wan 2.2 5B native FP8** | ~6 GB | ~8-9 GB w/ VAE | Fallback | Keep as a fallback if LTXV introduces unacceptable motion artifacts during live-test. Also serves 8 GB tier as optional B-roll lane. |
| **LTX-2 19B / 2.3 22B GGUF** | 12-14 GB | 14-17 GB w/ VAE decode spike | **REJECTED** | **DO NOT USE FOR BACKGROUNDS.** Audio-video foundation models are a paradigm mismatch and too heavy for a sidecar background layer on a 16 GB VRAM ceiling. VAE temporal decode adds 2-3 GB at decode → OOM. ChatGPT also flagged "1.1" version label as community packaging, not a confirmed upstream tag. |
| **HunyuanVideo distilled** | varies | varies | Not recommended | ChatGPT mentions; operationally heavier than LTXV. Skip. |
| **Stable Video 3 (8B)** | unknown | unknown | Suspect | NVIDIA round suggested with hallucinated specifics; do not pursue without independent verification. |

**Quantization gotchas on Blackwell sm_120 (both ChatGPT + Gemini):** Don't depend on FP8 / NVFP4 paths for video models yet — Blackwell support arrives in layers (PyTorch → CUDA kernels → custom ops → quant backends → custom nodes), and ComfyUI custom video nodes are exactly where "advertised support" and "production-safe support" diverge. Prefer fp16 / bf16 paths that already work.

**Pin format locked:**
```yaml
background_video:
  family: "ltxv"
  upstream_repo: "Lightricks/LTX-Video"
  model_file: "<exact 0.9.x safetensors filename to confirm at impl>"
  upstream_commit: "<HF commit SHA at impl>"
  comfyui_node_repo: "<exact custom node repo>"
  comfyui_node_commit: "<SHA at impl>"
  precision: "fp16"   # prefer over fp8 for stability on this layer
  frames_rule: "8n+1"
  target_frames: 193
  sampler_steps: 6
  postprocess: "setpts=PTS*2,fps=12"
```

#### TTS palette expansion — LOCKED LADDER 2026-04-30

NOT replacing the canonical pipeline (Bark + Kokoro + MusicGen + AudioGen → master mix). EXPANDING the per-character voice palette. Round-robin consult 2026-04-30 produced strong agreement on direction.

**Production add-order ladder (Parler-TTS REJECTED — owner pref; vintage sound stays in the deterministic DSP chain):**

| Priority | Engine | License | Peak VRAM | C7-deterministic? | Verdict |
|---|---|---|---|---|---|
| **1** | **Kokoro** (current) | MIT | ~1 GB | Yes | **KEEP.** Undisputed workhorse for strict lip-sync and clean narration. Gemini calls "undisputed king of low-VRAM deterministic phoneme TTS." |
| **2** | **Bark** (current) | MIT | ~6 GB | Yes (vram_sentinel + length-sort batching shipped) | **KEEP.** Unmatched for period vibe, character texture, and emotional color. |
| **3** | **CosyVoice 2** | Apache-2.0 | ~3-4 GB | Yes (flow-matching ODE solver + fixed seed = byte-identical) | **ADD NEXT.** Strongest production candidate for expanding the dramatic voice palette. Both ChatGPT + Gemini endorsed. |
| **4** | **Piper** | MIT | ~1 GB | Yes | **8 GB / UTILITY FALLBACK.** Tiny, deterministic, fast. Ideal for minor announcer roles or 8 GB emergency fallback. ChatGPT's recommendation for utility voices. |
| **5** | **CosyVoice 3** | Apache-2.0 | unknown | Unverified | **RESEARCH LANE.** Both flag as too new for production. Needs strict C7 hash proof before promotion. NVIDIA round claimed v3.2.1 production-ready with hallucinated commit SHA; ignore that signal. |
| **6** | **Qwen3-TTS** | needs license audit | unknown | **C7 RISK** | **RESEARCH LANE.** Gemini flags autoregressive + flow-matching hybrid as hard to make byte-identical. Highly expressive but requires deep C7 verification before any merge. |

**REJECTED candidates:**
- **Parler-TTS Mini** — owner preference; vintage broadcast sound stays in the deterministic DSP mastering chain (band-limit + tube saturation + plate flavor + noise floor + AM EQ).
- **Fish Speech** — license incompatible with MIT downstream.
- **XTTS / Tortoise / StyleTTS family** — license ambiguity, Windows friction, C7 determinism risk. Evaluate only if a specific gap appears that priorities 1-4 don't fill.

**C7 qualification protocol (apply to any new TTS before merge):**
1. Same prompt + same seed + same model revision + same driver/torch/CUDA/cuDNN + same batch size + same output format.
2. Run 10 repeated generations across cold start, warm start, and process restarts.
3. Hash final WAV bytes. If any hashes differ → engine is NOT qualified for OTR.

**Period-style controls — locked position:** Vintage broadcast sound lives in the deterministic DSP mastering chain (band-limit, tube saturation, plate flavor, noise floor, AM EQ shaping). TTS engines provide diction / cadence / timbre baseline only. Any model offering "1940s radio" as a text-prompted style is out of scope — we own the vintage sound, the model doesn't get to drift it.

**Pin format to lock once each engine ships:**
```yaml
tts_palette:
  engines:
    - name: "kokoro" / "bark" / "cosyvoice2" / "piper"
      upstream_repo: "<exact repo>"
      model_revision: "<tag/SHA>"
      tokenizer_revision: "<tag/SHA>"
      vocoder_revision: "<tag/SHA>"
      decode_mode: "<greedy|ode_solver|other>"
      sample_rate: "<Hz>"
      wav_hash_test: true
      role: "<character|announcer|narrator|utility>"
```

#### LLM palette expansion — QUEUED 2026-05-03 EVENING (paired with CosyVoice 2 add)

Same shape as the TTS ladder above: NOT replacing the canonical script-writer (Mistral-Nemo 12B), EXPANDING the per-role LLM palette so the writer pool can be voiced for tone (period radio drama, hard-boiled detective, broadcast announcer) instead of one general-purpose model carrying everything. Queued for the same beta cycle as the CosyVoice 2 TTS add — both are voice/character expansion work, both gate on the same C7 + VRAM verification protocol.

**Production add-order ladder (writer lane):**

| Priority | Model | License | Peak VRAM (est) | C7-deterministic? | Verdict |
|---|---|---|---|---|---|
| **1** | **Mistral-Nemo 12B** (current canonical) | Apache-2.0 | ~22.8 GB FP16 / ~7-8 GB int4 | Yes (deterministic with fixed seed + temperature 0) | **KEEP.** Default story-writer per `otr_scifi_16gb_full.json`. Don't replace. |
| **2** | **talkie-lm/talkie-1930-13b-it** (instruct variant — supersedes the earlier `destnyrr/talkie-1930-13b-base-gptq-int4` queue entry) | needs license audit | ~7-8 GB (13B int4) | needs verification | **PROMOTE TO NEXT-UP.** Instruct-tuned 1930s broadcast LLM. The instruct variant is what's actively trending on HF; better fit than the raw base for OTR's prompt-engineered writer prompts. Pair-add with CosyVoice 2 in the same beta cycle. |
| **3** | **Qwen/Qwen3.6-27B** (or `unsloth/Qwen3.6-27B-GGUF` for the pre-quantized GGUF) | Apache-2.0 | ~7 GB int4 GPTQ / ~6 GB GGUF Q4 | needs verification | **TIER-1 ALTERNATIVE.** Qwen3 series has top-tier creative-writing reputation; legitimately could replace Mistral-Nemo as primary writer if A/B test on the same prompt favors it. Unsloth GGUF quant means zero DIY quantization work. |

**Production add-order ladder (utility lane — NEW 2026-05-03 EVENING):**

Separate from the writer palette. Utility LLMs are for tasks where deterministic instruction-following + small footprint + Apache license matter MORE than period prose flavor. Capabilities target: summarization, structured extraction, classification, function-calling, normalization passes.

| Priority | Model | License | Peak VRAM (est) | Use case | Verdict |
|---|---|---|---|---|---|
| **1** | **ibm-granite/granite-4.1-8b** | Apache-2.0 (verified 2026-05-03) | ~5 GB int4 / ~16 GB BF16 (8.79B params, 17.5 GB on disk) | Title compression from news_seed (currently the news_seed_fallback path produces 80-char filename slugs like `signal_lost_what_a_decade_of_gene_therapy_research_f_...` — Granite would compress to 4-word punchy title); cast normalize pass (queued LLM cleanup); treatment.txt structured extraction; ledger forensics tool-use | **TIER-1.** IBM's "diverse domains, including business applications" framing is the OPPOSITE of what we want for the writer lane, but the EXACT shape we want for utility tasks. Strong instruction-following + tool-use + function-calling. |

**C7 qualification protocol (apply to any new LLM before merge):**
1. Same prompt + same seed + temperature 0 + same model revision + same tokenizer revision + same draft length cap.
2. Run 10 repeated generations across cold start, warm start, and process restarts.
3. Hash final draft text bytes. If any hashes differ at temperature 0 → engine is NOT qualified for OTR.
4. **Period-tone smoke pass:** generate 5 short scripts with the writer prompt and a fixed seed; spot-check that the model does NOT slip modern slang, modern brand names, or post-1950 cultural references into a script tagged for the 1940s setting. Failure mode: model that ignores period framing and emits anachronisms gets demoted to RESEARCH LANE pending prompt-engineering work.

**Pin format to lock once each LLM ships:**
```yaml
llm_palette:
  writers:
    - name: "mistral-nemo-12b" / "talkie-1930-13b-it" / "qwen3.6-27b-gguf-q4"
      upstream_repo: "<exact HF repo>"
      model_revision: "<tag/SHA>"
      tokenizer_revision: "<tag/SHA>"
      quant_format: "<fp16|int4-gptq|gguf-q4|int8|...>"
      context_cap: "<tokens>"
      temperature_default: 0.0
      draft_hash_test: true
      role: "<canonical|period-broadcast|hardboiled|announcer-narration|...>"
  utility:
    - name: "granite-4.1-8b"
      upstream_repo: "ibm-granite/granite-4.1-8b"
      model_revision: "<tag/SHA>"
      tokenizer_revision: "<tag/SHA>"
      quant_format: "int4-gptq | int8 | bf16"
      context_cap: "<tokens>"
      temperature_default: 0.0
      draft_hash_test: true
      role: "<title-compress|cast-normalize|treatment-extract|ledger-forensics|...>"
```

**Wired-in alongside what:** the writer-profile dropdown in `LLMScriptWriter` would gain new options (`Talkie-1930-it (Period Broadcast)`, `Qwen3.6-27B (Creative Alternative)`) that load via the same loader path used by Mistral-Nemo. Switch is per-episode at queue time, not per-line. The utility lane (Granite 4.1 8B) wires into a NEW node `LLMUtilityRunner` (or extends an existing utility hook) for the small structured-output tasks that don't need a full writer; it co-loads alongside the writer profile because their VRAM footprints (5 GB + 7-8 GB int4) sum to ~13 GB, comfortably under the 14.5 GB ceiling. CosyVoice 2 add (TTS priority 3 above) is independent at the audio engine layer; all three (writer-add, utility-add, TTS-add) can ship in the same v2.0-beta cut without touching each other's code paths.

**Rejected from this round (size or alignment mismatch):**
- **Anything 100B+** (DeepSeek-V4-Pro 862B, MiMo-V2.5 311B, Kimi-K2.6 1.1T, Mistral-Medium-128B, Ling-1T) — exceeds 16 GB VRAM even at int4
- **Multimodal `Image-Text-to-Text`** variants (Qwen image families, Gemma-4 31B-it has IMG variants) — wrong tool for text-only OTR writing
- **`text-to-image` / `text-to-video`** (SeeSee21, SulphurAI) — wrong domain entirely
- **`HauhauCS/Qwen3.6-27B-Uncensored-...-Aggressive`** — explicitly conflicts with OTR's safe-for-work / no-profanity content standard
- **`google/gemma-4-31B-it`** + **`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning`** — both interesting Tier-2 candidates but deferred until after the Tier-1 writer A/B (Mistral-Nemo vs Talkie-1930-it vs Qwen3.6-27B) lands a winner. Re-evaluate then.
- **`ibm-granite/granite-4.1-30b`** — bigger Granite sibling loses the small-footprint advantage that makes the 8B compelling for the utility lane.

**Defer to v2.0-beta** — same trigger as the TTS expansion. Land BUG-LOCAL-031+ first, then the v2.0-alpha → v2.0-beta cut, then this palette work in beta cycle 1.

### LLM character normalize pass

Currently cast cleanup is two layers: (1) regex blocklist `_SFX_CAST_BLOCKLIST_PATTERNS` (BUG-091 + BUG-097), (2) fuzzy `_consolidate_similar_cast_rows_with_aliases` (BUG-098). Both deterministic, limited to KNOWN patterns. An LLM-based normalize after fuzzy dedup could catch semantic aliases neither layer sees: `KEVIN VOICEOVER` → `KEVIN STENDAHL`, `(captain)` lowercase → `CAPTAIN`, `DR. AMELIA HARTFIELD` → `AMELIA`.

**Constraints:** conservative prompt ("ONLY merge when names CLEARLY refer to the same character; when in doubt, do NOT merge"); hard-cap merge-set ≤50% of cast (flags hallucination); only run on `optimization_profile = "Pro (Ultra Quality)"` (adds 2-5 min wall time); feed first 1500 chars of script_text + first sentence of each character's first line.

**Defer to v2.0-beta** — by then we have a real corpus of run logs showing common emission patterns, so the prompt can be data-informed instead of guesswork-driven.

---

## v2.1 candidates

### Configurable show name (replace hardcoded "Signal Lost")

**Status:** queued 2026-05-03 LATE EVENING. Real-shippability blocker — anyone wanting to fork OTR for their own show ("Twilight Zone", "Lights Out", "The Hitchhiker") currently has to grep + sed across the codebase.

**Sites that hardcode "Signal Lost":**
- `nodes/video_engine.py:1484` — `out_path = ... f"signal_lost_{safe_title}_{ts}.mp4"` (filename prefix)
- `nodes/story_orchestrator.py:9089` — announcer closing line `"This has been Signal Lost. {episode_title}. Stay safe."`
- `nodes/story_orchestrator.py:6216` — last-resort title fallback `"Signal Lost Transmission {ts}"`
- `nodes/video_engine.py:1322` — last-resort title fallback `"Signal Lost {ts}"`
- (probably more — full grep needed before scoping)

**Fix architecture:** add a `show_name` field to `ProjectState` (already loaded by Director + ScriptWriter), plumb it through everywhere the literal `"Signal Lost"` appears. Default to `"Signal Lost"` for backwards-compat. Surface as a top-level widget on `OTR_ProjectStateLoader` (or whichever node currently owns project_state) so users can flip it without code edits.

**Verify:** grep for `Signal Lost` returns ZERO source-file hits after the change (all references go through `project_state.show_name`); test fixture with `show_name="Twilight Zone"` produces filenames like `twilight_zone_<title>_<ts>.mp4` and announcer closings like "This has been Twilight Zone."

**Why v2.1 not v2.0:** v2.0 ships as branded "Signal Lost" — that's fine for the launch. The brand-portability work is its own scoped sprint and shouldn't gate the v2.0 release.

### Per-shot / per-scene face variation via PuLID-FLUX

**Status:** queued 2026-05-03 EVENING. **Defer to v2.1** — landed AFTER v2.0 ships clean.

**Context:** v2.0 ships with the BUG-LOCAL-078 portrait pass (`OTR_BatchFluxPortraitRender`). Each character gets ONE canonical portrait per episode — fully dynamic, fresh on every run, no stored stock characters, no cross-episode face library. HuMo references that single portrait for every line of that character's dialogue. Within-episode consistency goes from ~5/10 (env-still tier-4 fallback) to ~9/10 (single canonical portrait). For an anthology series with fresh cast every episode, single-portrait-per-character is the correct architecture. **v2.1 should NOT change that default.**

**What v2.1 ADDS** (opt-in, not default):

Per-shot or per-scene FACE VARIATION for the same character — same identity, different STATE. The character is recognizably the same person across the whole episode (single PuLID identity reference), but each shot/scene can render that face in a different STATE that reflects the story:

- Scene 1: clean, composed (just entered the scenario)
- Scene 3: sweat, dirt, dilated pupils (mid-crisis)
- Scene 5: bloodied, exhausted, scarred (post-climax)
- Scene 6: composed but visibly changed (denouement)

PuLID-FLUX is the canonical solution: it extracts the FACE IDENTITY from a reference image and re-renders it under a new prompt. So the workflow is:

```
ROUND 1 (text-only FLUX): render the character's seed portrait from
        ledger.cast[i].appearance text. This becomes the IDENTITY ANCHOR.
        Same as v2.0's portrait pass. Save to portraits/<char>_seed.png.

ROUND 2..N (PuLID-FLUX, per shot or per scene):
        For each ledger.scenes[i] OR ledger.shots[i] entry, render a
        new face image using:
          - PuLID identity reference  =  portraits/<char>_seed.png
          - prompt                    =  v2.0's portrait composition base
                                          + scene/shot-specific state
                                          modifier (sweat, blood, etc)
        Save to portraits/<char>_scene{N}.png.

        State modifier sources, in priority order:
          (a) ledger.scenes[i].character_state[char_id] (if LLMDirector
              populates it -- new ledger field for v2.1)
          (b) ledger.shots[i].mood + character_position_in_arc
          (c) ledger.lines[i].traits (per-line emotion tag)
          (d) Default ladder by scene index: scene_1=clean,
              mid_scene=mid, last_scene=worn

HuMo's portrait_path lookup (v2.1 update):
        Currently picks ledger.cast[i].portrait_path (single canonical).
        v2.1 adds tier 0: ledger.scenes[scene_id].cast_portraits[char_id]
        if populated, falls back to tier 1 (cast canonical) otherwise.
```

**What this BUYS** (per-shot variation locked to single identity):
- Story-driven visual evolution. The character ages / accumulates damage /
  emotionally shifts as the episode progresses, but it's recognizably them.
- Higher emotional payoff in the final montage. Scene 1 vs scene 5 of the
  same character looks DIFFERENT (right) instead of IDENTICAL (wrong, but
  what v2.0 ships).
- Anthology format unchanged. No persistent face library. Each episode
  builds its own seed + variations from scratch and discards them at the
  next run.

**What this COSTS:**
- PuLID-FLUX install: `ComfyUI-PuLID-Flux-Enhanced` custom node + ~1-2 GB
  PuLID model weights + ~250 MB InsightFace `antelopev2` face detection.
- VRAM: ~3 GB extra on top of FLUX dev fp8 (~12 GB). Total ~15 GB. Tight
  but fits the 16 GB ceiling.
- Render time: 2x portrait time per character per scene/shot variant.
  For a 5-scene episode with 3 characters: 3 seed portraits + 15 scene
  variants = 18 FLUX renders, ~3-5 minutes added per episode (vs v2.0's
  ~30-60 sec for the seed pass alone).
- Code: extend `OTR_BatchFluxPortraitRender` with a v2 mode that loops
  scenes after the seed pass; new ledger field `cast_portraits` per scene
  populated by LLMDirector; HuMo's `_find_portrait` updated to prefer
  per-scene over canonical when present. Estimated ~4-6 hours of code +
  test work.

**Acceptance criteria for v2.1 ship:**
1. Single full episode renders with per-scene face variation enabled.
2. Visible state shift across scenes (verified by ffprobe + manual frame
   inspection — scene 1 portrait vs scene 5 portrait should be the SAME
   FACE but DIFFERENT STATE).
3. C7 audio byte-identity holds (visual changes don't touch the audio path).
4. Performance budget: <5 minutes added per episode at 5 scenes / 3
   characters.
5. Toggle defaults to OFF so v2.0 single-portrait behavior is the default.
   Users opt in by flipping a widget.

**Deferred from this lane (separate v2.x work, NOT in v2.1 scope):**
- Cross-episode face registry (recurring characters, stored library) —
  conflicts with anthology design philosophy; revisit only if OTR pivots
  to a serialized format.
- Face-locking on HuMo's OUTPUT video (not just the portrait input) —
  much harder, requires video-level identity injection. HuMo's intrinsic
  per-frame variation is acceptable for now.
- Multiple portrait ANGLES per character (frontal + 3/4 + side) — would
  require HuMo upgrade to consume multiple references. Out of scope.

---

## Discarded — do not revisit

- Flash Attention 2/3 on sm_120
- Pinning torch < 2.10
- Weight streaming from system RAM via ComfyUI-Manager
- Asynchronous weight streamer as a fallback for 16 GB OOM
- "Shift Bark to HuggingFace implementation" (already on it)
- Speculating on unreleased Visual unified latent space
- Visual 2.0 Gate 0 probe (WorldMirror / HunyuanWorld / WorldStereo / WorldPlay-5B) — retired 2026-04-17. VisualBridge + Poll + Renderer harness stays as the harness; the backends are the active video stack
- `ComfyUI-*-Wrapper` repos as primary runtime (pull flash_attn, wrap overhead)
- v2v chaining (deep-fries output by 3rd generation)
- Single-image LoRA training on the laptop during live orchestration (thrash risk)
- SD 1.5 anchors as final style — pivoted to FLUX-native
- Subprocess pattern for HuMo orchestration (BUG-076 OTR_PostAudioVideoPipeline + render_humo_batch.py orchestrator) — superseded 2026-04-27 by in-graph nodes (BUG-078). Subprocess scripts remain as ad-hoc CLI smoke tools but the production path is in-graph. `OTR_PostAudioVideoPipeline` class kept registered with `(retired)` title for back-compat with old workflow JSONs
- Blanket `git clean -fX` — the existing `scripts/_*.py` ignore is too broad and would nuke `_consult_*.py`, `yoga_watchdog.py`, and other legitimately-local files. Use targeted `git clean -fX -- <pattern>` instead

---

## References

- `CLAUDE.md` — project rules, platform pins, Desktop Commander git pattern
- `docs/BUG_LOG.md` — live bug tracking
- `docs/ROADMAP_HISTORY.md` — historical session logs and shipped-work archive
- `docs/2026-04-12-otr-v2-visual-sidecar-design.md` — v2 design spec
- `docs/2026-05-02-v2.0-beta-sprint-qa/` — round-robin QA on Sprint 1/2/3 plan (this session)
- `docs/2026-05-09-ledger-consumer-rewrite-patterns.md` — **L3 consumer rewrite patterns doc** (canonical reference for the 7-consumer ledger sprint; pattern 1 = `load_ledger` posture, pattern 2 = role filters with judgment rule, pattern 3 = voice fallback, pattern 4 = write-back contract, pattern 5 = `production_plan_json` demotion, pattern 6 = hermetic test fixture, pattern 7 = canonical 4-test plan)
- `nodes/_otr_ledger_consumers.py` — read-side helper module (L3-strict, raises ValueError on legacy list)
- `nodes/_otr_ledger.py` — write-side helper module (existing, `patch_line_fields` + `save_ledger_safe` + `set_meta` + new `patch_line_text` for atomic text+metrics updates)
- Survival guide / Bug Bible: https://github.com/jbrick2070/comfyui-custom-node-survival-guide

---

## Pre-ship v2.0 — ecosystem review checklist

Quick scan before tagging v2.0-alpha → v2.0. Verify each upstream
release either (a) doesn't break OTR's pinned versions or (b) is
worth pulling in for the v2.0 release notes. Added 2026-05-07.

### ComfyUI Core & Frontend
- v1.44.18 (2026-05-06) and v1.44.17 (2026-05-05) — review changelog
  for anything affecting the LTX 2.3 path, MultimodalGuider, RES4LYF
  compatibility, or Blackwell/CUDA 13 attention paths.
- Releases: https://github.com/Comfy-Org/ComfyUI_frontend/releases
- Changelog: https://docs.comfy.org/changelog

### ComfyUI-GGUF — native GGUF weight loading
- v1.1.10 (2026-01-12), with continuous repo commits.
- Repo: https://github.com/city96/ComfyUI-GGUF
- Why care: opens a smaller-VRAM path for LTX 2.3 (the GGUF
  Q5_K_M quants of the 22B-distilled exist on HF). Could become
  the "32 GB RAM" budget option below the current v0_9 default
  if GGUF + euler_cfg_pp produces equivalent motion at ~half the
  weight footprint vs the BF16 fused 46 GB.

### ComfyUI-Ollama nodes — LLM integration / agent tooling
- Continuous Q1/Q2 2026 updates, including DeepSeek-R1 and Qwen
  3.5 architecture support.
- Describer / agent variant: https://github.com/alisson-anjos/ComfyUI-Ollama-Describer
- Native workflows: https://github.com/slyt/comfyui-ollama-nodes
- Why care: OTR currently uses transformers + Mistral-Nemo for
  story / critic / brief LLMs. Ollama would give an HTTP-server
  pattern with model swap by name (no per-call load), DeepSeek-R1
  for the critic role, and Qwen 3.5 for shorter beat-level
  rewrites. Worth a benchmark spike before v2.0 ships in case
  one of them obsoletes the current LLM stack.

### Google Gemma 2026 Developer Challenge
- Launched 2026-05-06.
- Link: https://dev.to/challenges/google-gemma-2026-05-06
- Why care: OTR's LTX 2.3 path uses Gemma 3 12B (FP4 mixed) as
  its text encoder, and the legacy story/critic LLM was Gemma-4
  before the Mistral-Nemo migration. If the challenge surfaces
  Gemma-tuned techniques or new finetunes (e.g. better motion
  prompt adherence, period-specific tonal control for the
  1940s OTR aesthetic), worth folding into either the prompt
  pipeline or the LTX encoder layer. Submission window may also
  be a forcing function to publish OTR's Gemma usage pattern as
  a contest entry — free marketing for the project.

---

## Daily operating cadence

- First thing: read this file, `CLAUDE.md`, `docs/BUG_LOG.md` header, `git log --oneline -5` on current branch.
- LHM is always on — poll `http://localhost:8085/data.json` (or `outputs/libre_tail.py`) before asking Jeffrey for system status.
- After every code change: AST parse + three regression suites (Bug Bible regression in survival-guide repo, `tests/test_dropdown_guardrails.py`, `tests/test_core.py`). Don't report "done" until green.
- One `git push` attempt max — if it fails, hand a cmd block with `cd /d` included.
- Verify every push: local HEAD == origin HEAD, no 0-byte files, no BOM, workflow JSONs valid, all node classes registered in `__init__.py`.
- Log bugs the moment they surface. Don't batch. Promote `Bible candidate: yes` to the survival guide only after the fix is verified AND a real run confirms the behavioural fix.
- Round-robin consult before non-trivial design decisions (CLAUDE.md "Round-Robin Consultation" rule). Save transcripts under `docs/<date>-<topic>/`.
- Never use PowerShell for git operations — always cmd shell via Desktop Commander (PowerShell mangles `&&` and commit message quoting).
