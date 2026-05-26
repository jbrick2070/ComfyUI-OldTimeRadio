# Downstream Brief Consumer Audit (Sprint 8 prep)

- **Repo:** `ComfyUI-OldTimeRadio` @ `v2.0-alpha`
- **HEAD at audit time:** `91007e7` (Sprint 7C close -- payload_null typed repair)
- **Audit date:** 2026-05-25
- **Author:** Lead (post v4-plan close, pre-Sprint-8 wiring)
- **Decision context:** `session_handoff.md` item 1 -- `meta.story_brief` is the single source of truth for every downstream creative prompt; brief schema v2 adds five fields, one consumer wires per commit.

This audit is read-only. It classifies every node that reads (or should read) `meta.story_brief` and the sidecar `meta.story_brief_terms` against the four-class taxonomy from the session handoff. No consumer is edited here. The follow-up sprint wires consumers one commit at a time, starting with MusicGenTheme (the confirmed live-run miss).

---

## 0. Producer schema (current, v1)

Stamped onto `meta` by `nodes/_otr_story_brief.run_story_brief_reflection` (post-composition reflection, technical-slot LLM call through the shared `structured_call` ladder). The 8-key delta:

| Key | Type | Notes |
|---|---|---|
| `story_brief` | str (<= 300 chars) | One-sentence visual atmosphere brief. Anonymised (Sprint 3G). |
| `story_brief_status` | `"ok" \| "failed" \| "absent"` | Observable failure mode. |
| `story_brief_error` | str \| None | Rejection class on failure (`json_parse_failed`, `schema_validation_failed`, content reject codes). |
| `story_brief_model` | str | Technical-slot model id at call time. |
| `story_brief_prompt_version` | str | `"v1"` today. Bump on prompt body changes consumers must observe. |
| `story_brief_source` | str | `"post_script_reflection"`. |
| `story_brief_char_count` | int | `len(story_brief)`. |
| `story_brief_terms` | dict | Three sidecar arrays. See below. |

`story_brief_terms` (sidecar -- shape enforced by `StoryBriefModel`):

| Sub-key | Type | Length cap | Today's use |
|---|---|---|---|
| `setting` | `list[str]` | <= 10, each <= 24 chars | Scene/env nouns. MusicGenTheme reads top-2; FLUX env path also reads via lighting helper. |
| `lighting` | `list[str]` | <= 10, each <= 24 chars | Lighting nouns. Joined with `atmosphere` by `get_story_brief_lighting`. |
| `atmosphere` | `list[str]` | <= 10, each <= 24 chars | Atmosphere nouns. Joined with `lighting`; MusicGenTheme separately reads as mood adjectives (the C case). |

---

## 1. Brief schema v2 additions (per `session_handoff.md`)

These are the fields the wiring sprint must teach the producer to emit, and the consumers to read:

| New v2 field | Type | Replaces / covers |
|---|---|---|
| `music_mood_terms` | `list[str]` | The MusicGenTheme miss. Mood vocabulary tuned for MusicGen, NOT visual atmosphere -- replaces today's `_compose_music_prompt`-reads-`atmosphere` improvisation and the over-filtered `get_story_brief_music_mood` 16-word vocab. |
| `visual_palette` | `list[str]` | Palette descriptors (color, texture, material) for FLUX env + portrait + radio. Today FLUX consumers read `story_brief` prose or `lighting`/`atmosphere` term arrays; neither is a clean palette. |
| `atmosphere_line` | str | Single-sentence atmosphere line. **Naming collision flagged** -- v1 already has `story_brief_terms.atmosphere` as a list of nouns. Recommend the v2 single-sentence field be named `atmosphere_line` (or live at `meta.atmosphere`, separate from `story_brief_terms`) to avoid quietly replacing the existing array. See Open Decision A below. |
| `tempo_hint` | `"slow" \| "moderate" \| "driving"` (or similar enum) | Pacing cue for music cue length + LTX motion intensity. New surface. |
| `key_objects` | `list[str]` | Named props / settings the brief promised. New surface for FLUX env + HuMo. |

**v2 producer impact (out of this audit's scope, scoped here for the wiring sprint):**

- Extend `StoryBriefModel` (`nodes/_otr_story_brief.py:159`) with the new fields, all with safe defaults so a v1 ledger replayed against v2 code degrades cleanly to today's behaviour.
- Extend the reflection prompt body to ask the LLM for the new fields (one structured JSON object, no Markdown -- same shape as today).
- Extend `_success_delta` / `_failure_sentinel` (`_otr_story_brief.py:736 / 706`) to include the new fields (sentinel emits empty list / empty string defaults so the read helper's `if not value: fall through to legacy` pattern just works).
- Bump `_PROMPT_VERSION` v1 -> v2 so the version stamp records the shape change.

---

## 2. The shared read contract (`_otr_brief_reader.py`)

Per the session handoff: **every downstream consumer reads through one helper** -- no inline `meta.get(...).get(...)` chains. New module `nodes/_otr_brief_reader.py` exposes a single shared read function:

```python
def _read_brief_field(meta, field_name, default):
    """Read a story-brief field through the canonical contract.

    - `field_name` is dotted: 'story_brief', 'terms.setting',
      'music_mood_terms', 'visual_palette', 'atmosphere_line',
      'tempo_hint', 'key_objects'.
    - `default` is returned when meta is missing, the field is
      absent, or story_brief_status != 'ok' (the C5b helper convention
      every existing consumer already relies on).
    - Returns the literal value when present; never raises.
    """
```

**Why centralise:** if v3 ever renames a field (`music_mood_terms` -> `music_mood`), one edit in `_otr_brief_reader.py` updates every consumer. Today the field-name contract is enforced by N independent `meta["story_brief_terms"]["atmosphere"]` reads.

The five C5b helpers (`get_story_brief_full`, `get_story_brief_ltx`, `get_story_brief_lighting`, `get_story_brief_music_mood`, `get_story_brief_status`) stay -- they are shape adapters tuned per consumer (sentence trim, vocabulary intersection, joining). They become wrappers around `_read_brief_field` internally so the field name lives in exactly one place.

---

## 3. Consumer classification

| # | Consumer node | File:line | Reads today | Class | Notes |
|---|---|---|---|---|---|
| 1 | `OTR_MusicGenTheme` | `nodes/musicgen_theme.py:327-417` (`_compose_music_prompt`); helper imports at L63-64; diagnostic log at L544-549 | `story_brief_terms.atmosphere` (top 3 directly), `story_brief_terms.setting` (top 2 directly); the helper `get_story_brief_music_mood` is read but ONLY for the diagnostic log line -- NOT threaded into the prompt | **C** | Reads the right key (atmosphere) but the field's values are visually tuned, not music tuned. Live-run console: `[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[] style_slug_diag=...` -- the `mood_terms=[]` is the helper's 16-word-vocab intersection dropping every visual atmosphere term. The actual prompt is NOT empty (it pulls atmosphere directly), but it's misaligned. v2 adds `music_mood_terms` -- a dedicated field tuned for MusicGen vocabulary. Wiring: `_compose_music_prompt` reads `music_mood_terms` first; falls back to today's atmosphere-as-mood path when v2 brief is absent / legacy. |
| 2 | `OTR_BatchLTXRender` | `nodes/batch_ltx_render.py:438-484` (`_build_motion_prompt`) | `get_story_brief_ltx(meta, max_chars=90)` (sentence-trimmed full prose), `get_story_brief_status(meta)` for log line | **A** | Reads the right key, threads the right value. 240-char total budget + motion-verb position guard already handle pathological briefs. Future v2 enrichment: `tempo_hint` could inform whether the motion verb is "drift" (slow) vs "whip-pans" (driving) -- nice-to-have, not required. |
| 3 | `OTR_BatchFluxPortraitRender` | `visual/batch_flux_portrait_render.py:100-150` (`_build_portrait_prompt`); helper imports at L331-336 | `get_story_brief_lighting(meta)` (lighting + atmosphere terms joined), `get_story_brief_status(meta)` for log line | **A** | Reads the right key. Lighting leads the prompt (BUG-LOCAL-250 follow-up). v2 enrichment: `visual_palette` would add color/material descriptors the portrait wants (e.g. "warm tungsten, brushed aluminum"). |
| 4 | `OTR_BatchFluxRender` (env stills) | `visual/batch_flux_render.py:405-460` (`_parse_env_prompts`) | `get_story_brief_full(meta)` (full prose -- LEADS the env prompt), `get_story_brief_status(meta)` | **A** | Reads the right key. Brief leads the env prompt body (BUG-LOCAL-250). v2 enrichment: `visual_palette` for color discipline; `key_objects` to bias toward named props the brief promised. |
| 5 | `OTR_BatchFluxRender` (radio bookend) | `visual/batch_flux_render.py:259-339` (`_build_dynamic_radio_prompt`); helper imports at L232-256, L1243-1250 | `get_story_brief_full(led)` (primary radio descriptor), `get_story_brief_status` (logged) | **A** | Reads the right key. Tiered fallback (brief -> episode_id slug -> hardcoded `_RADIO_FALLBACK_PROMPT`) already works. v2 enrichment: `visual_palette` could replace the hand-tuned cinematic style suffix on the radio prompt. |
| 6 | `OTR_VideoPlan` (`_resolve_era_tail`) | `nodes/otr_video_plan.py:147-183` | `get_story_brief_lighting(meta)`, `get_story_brief_status(meta)` | **A** | Reads the right key. Replaces the retired `_ERA_TAIL_BY_STYLE` style-slug lookup (BUG-LOCAL-250). v2 enrichment: same as portrait -- palette + atmosphere line. |
| 7 | `OTR_BatchHumoRender` (`_build_pos_prompt`) | `nodes/batch_humo_render.py:1187-1196`; status probe at L1694-1708 | `get_story_brief_lighting(meta or {})` (lighting + atmosphere joined), `get_story_brief_status` (logged once per run) | **A** | Reads the right key. Lighting follows the speaker description, before `_DEFAULT_POS_SUFFIX`. v2 enrichment: `atmosphere_line` (the single-sentence version) -- HuMo's lip-sync clips want a stable cinematic mood line more than a term list. |

**No B-class consumers.** (Type B = "reads the right key, but the field doesn't exist yet in v1.") In v1 every helper-mediated read resolves to a field that does exist; the gap is misalignment (C) or under-enrichment (A wanting more).

**No D-class consumers worth the wiring cost today.**

The handoff flagged "FLUX prompts, HuMo prompts, possibly the title scratchpad -- TBD" as D candidates. Three candidate sites were checked:

| Candidate | Site | Reads brief today? | D-class? | Notes |
|---|---|---|---|---|
| Title scratchpad | `nodes/OTR_LedgerScriptWriter.py` `_generate_title_from_script` (Sprint 3E -- DETAILS -> CANDIDATES -> TITLE) | No | **No** | The title path already grounds against `_build_title_excerpt_set` -- opening / middle / ending lines + premise + arc verdict (the Sprint 3E richer excerpt set). Adding `meta.story_brief` would be redundant signal (the brief IS a post-composition reflection of the same excerpts). Defer until / unless a live-run shows title drift the existing grounding doesn't catch. |
| Style picker | `nodes/_otr_style_picker.py` (Pass 1 inventor + Pass 2 chooser) | No | **No (causal)** | The brief is a POST-script reflection. Style is picked PRE-script. The style picker cannot read a brief that has not been written yet. Confirmed causal block, not a wiring gap. |
| News interpreter / Outline / Casting / Continuity / Critic / Reroll | various | No | **No (causal)** | Every upstream creative pass runs before the brief is generated. Critic + reroll run AFTER composition but BEFORE the reflection (reflection is the last writer step). None of these can read the brief without rebuilding the pipeline ordering. |

If a future sprint inverts the pipeline (e.g. an "outline reflection" pass producing a pre-script brief), that surfaces a real D-class candidate. Today, the only D-class candidate is the title path, and it has rich grounding already.

---

## 4. One-commit-per-consumer wiring order

Per the handoff: one commit per consumer keeps the diff small and the regression auditable.

| Order | Consumer | Why this order | Sprint label |
|---|---|---|---|
| 1 | **MusicGenTheme** | The confirmed live-run miss. Highest signal-to-noise -- a single new field (`music_mood_terms`) replaces the entire atmosphere-as-mood improvisation. The diagnostic log line at L544-549 already prints `mood_terms=...` -- the success metric is "this list is non-empty on the next live run". | **8.1 MusicGenTheme** |
| 2 | **Producer v2 schema** | Done as part of commit 1 (the schema add cannot land separately from its first consumer, or the consumer's read returns the v1 fallback and the new path is never exercised). Practically: commit 1 = producer v2 add + MusicGenTheme read + `_otr_brief_reader.py` shared helper + tests. | (folded into 8.1) |
| 3 | **FLUX env** (`_parse_env_prompts`) | Highest visual leverage. Today reads full prose; v2 `visual_palette` + `key_objects` give the env scene named-prop bias the brief currently has to imply. | **8.2 FLUX env** |
| 4 | **FLUX portrait** (`_build_portrait_prompt`) | Already reads lighting; v2 `visual_palette` + `atmosphere_line` upgrade the portrait composition. | **8.3 FLUX portrait** |
| 5 | **FLUX radio bookend** (`_build_dynamic_radio_prompt`) | Smallest diff. The radio still is a single prop; `visual_palette` cleans up the hand-tuned style suffix. | **8.4 FLUX radio** |
| 6 | **LTX motion** (`_build_motion_prompt`) | `tempo_hint` informs motion verb selection. Optional enrichment -- A-class today, so no live-run pressure. | **8.5 LTX tempo** |
| 7 | **HuMo lip-sync** (`_build_pos_prompt`) | `atmosphere_line` replaces / augments the lighting+atmosphere term join. HuMo wants a stable cinematic mood line per clip. | **8.6 HuMo atmosphere** |
| 8 | **OTR_VideoPlan** (`_resolve_era_tail`) | Sweeps the era tail to v2 palette + atmosphere_line. PASS 3 composite cleanup. | **8.7 VideoPlan tail** |

**Test gate per commit:** add or extend a unit test that pins (a) the new field reads through `_read_brief_field`, (b) absent / failed / empty paths fall through to the v1 helper behaviour. The existing `tests/test_story_brief_helpers_c5b.py` suite is the template -- new tests live next to it (`tests/test_brief_reader_*.py` per consumer).

**Regression gate per commit (per CLAUDE.md):** Bug Bible + full OTR suite + LLM-slot sweep after each, unprompted.

**Live-run cadence:** the v2 producer change is observable from commit 1 onward -- the writer's `meta.story_brief_*` keys gain the new fields. Operator live-run after commit 1 confirms MusicGenTheme reads non-empty mood; subsequent consumer commits ride along on opportunistic live runs (not a hard gate per commit).

---

## 5. Open decisions for Jeffrey

A. **`atmosphere` naming collision.** v1 emits `story_brief_terms.atmosphere = list[str]` (term array). v2 plan adds `atmosphere` as a single sentence. Two ways to land it cleanly:

   * **(A1)** v2's single-sentence field lives at `meta.story_brief_atmosphere_line` (or just `meta.atmosphere_line`), separate from `story_brief_terms`. `story_brief_terms.atmosphere` (the array) stays untouched. **Recommended** -- zero collision, zero rename, every existing consumer of the array keeps working.
   * **(A2)** v2 renames the array to `story_brief_terms.atmosphere_terms` and reclaims the bare `atmosphere` key for the single sentence. Cleaner schema, but every consumer of the array gets a same-commit rename.

B. **Wiring sprint 8.x or a different label?** This audit calls it "Sprint 8.x" by convention (v4 plan ends at 7C). If the wiring is part of a different roadmap track, rename in the Status Board before the first commit lands.

C. **Should `_otr_brief_reader.py` ship as part of commit 1, or as its own preceding commit?** Two options:

   * **(C1)** Ship together with MusicGenTheme (commit 1 carries: producer v2 fields + reader module + MusicGenTheme rewire + tests). Larger commit but the reader's first real caller proves it works.
   * **(C2)** Ship the reader module first as its own commit with unit tests but no consumers wired; commit 2 then wires MusicGenTheme. Smaller diffs but the reader sits unused for one commit. **Defer to Jeffrey** -- both are auditable.

D. **Title scratchpad as D-class?** This audit declines on the grounds that Sprint 3E's excerpt grounding already covers the title path. If a live run shows title drift the brief would catch, revisit.

---

## 6. Out of scope for this audit

- The v2 brief's prompt body (what the reflection LLM is told to produce). Wiring sprint owns the prompt rewrite; this audit only specifies the schema shape consumers read.
- Schema versioning / migration semantics. A v1 ledger replayed against v2 code falls through cleanly because every new helper returns a safe default; no migration script needed.
- The deferred v2 follow-ups from the session handoff (StoryCriticReport silent-default landmine, BUG-LOCAL-275 -- now FIXED, continuity_slice on rerolls -- now LANDED, etc.). Those are separate sprints.

---

## Appendix A: Live-run console signal (BUG context)

From `session_handoff.md` item 1, the live-run line that motivated this audit:

```
[OTR_MusicGenTheme] story_brief_status=ok mood_terms=[]
style_slug_diag=sanctioned_trade_battle
```

Decoded:

- `story_brief_status=ok` -- the reflection pass succeeded; the brief is non-empty.
- `mood_terms=[]` -- `get_story_brief_music_mood(meta)` returned an empty list. This is the 16-term `_MUSIC_MOOD_VOCAB` intersection in `nodes/_otr_story_brief_helpers.py:35-39` dropping every visual-tuned atmosphere term the reflection produced.
- `style_slug_diag=sanctioned_trade_battle` -- the style preset slug (diagnostic only since the 2026-05-18 Path F change that retired style-driven music prompts).

The actual MusicGenTheme prompt body is NOT empty -- `_compose_music_prompt` (`nodes/musicgen_theme.py:370-371`) reads `atmosphere[:3]` directly, bypassing the helper's vocab filter. But those top-3 atmosphere terms were chosen for a visual atmosphere brief, not for MusicGen. The v2 `music_mood_terms` field gives MusicGen its own field, tuned for the music model's vocabulary.

Closing this miss is commit 1 of the wiring sprint.
