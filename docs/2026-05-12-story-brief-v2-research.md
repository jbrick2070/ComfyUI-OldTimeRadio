# `meta.story_brief` v2 — research paper

**Author:** Cowork research pass (Claude)
**Date:** 2026-05-12
**Status:** Analysis only — no code changes. Input to round-robin review.
**Scope:** Inventory every post-script-write prompt assembly site that currently keys off `meta.style` (the picker slug) or `meta.visual_plan.genre`, identify the recently reintroduced "genre" surface that the problem statement flags for removal, and surface candidates for `story_brief` integration. Plus prior art that the build sprint can borrow from.

This file is a **discovery document**, not a build spec. It does not propose code. It lists sites, traces the actual current strings, flags conflicts, and surfaces open questions.

---

## 0. TL;DR

Five things matter, in this order.

**1. The "virulent" `_GENRE_BY_STYLE` table is real, lives in one file, and has no LLM or visual consumer today.** It was reintroduced on 2026-05-12 in voice-path-cleanbreak Sprint 6.1 (commented "Replaces the hardcoded 'audio drama' fallback Sprint 2 used"). The comment block claims it feeds "FLUX scene-prompt composition (style_tail + genre)" but a full grep across `nodes/`, `visual/`, and tests finds zero FLUX, LTX, HuMo, or MusicGen consumers. The only live readers are two cosmetic surfaces in `nodes/video_engine.py`: the post-roll telemetry HUD card and the `_treatment.txt` companion file. Both fall back through `style or genre or "sci-fi"` — i.e. if `style` is present the `genre` value is never read at all. Sprint 6.1 added a 10-entry mapping, a strict resolver, a preview helper, a guardrail test, and a stamping call site, all to feed two display lines that already work from `meta.style`.

**2. There is exactly ONE post-script-write LLM call in the LPL writer path today.** It is `_generate_title_from_script` (OTR_LedgerScriptWriter.py:574, called at line 2235 in section J.5). It reads the full assembled `lines[]` and regenerates a title when the existing title is in `_STUCK_TITLE_DEFAULTS`. Everything else that looks like a "post-script LLM" — `news_close_brief`, `script_brief`, `casting_brief`, `key_terms` — is **upstream of writing**, produced by `news_interpreter` from the raw RSS article before composition begins. Those briefs feed into the writer, not out of it.

**3. There is excellent prior art for `meta.story_brief`.** `_generate_ltx_style_brief()` in legacy `story_orchestrator.py:3418` already has a working, length-bounded, era-neutral, no-people-no-faces prompt template (`_LTX_STYLE_BRIEF_PROMPT`, line 3398) that produces a 20-40-word single-sentence visual brief. It was orphaned by BUG-LOCAL-112 (2026-05-06) when LTX stopped consuming it for prompt-dilution reasons. The field name `ledger.meta.ltx_style_brief` is still mentioned in `batch_ltx_render.py:396` as "still stamped by OTR_LedgerScriptWriter" — but the LPL writer (`OTR_LedgerScriptWriter.py`) does not call the legacy generator, so the stamp is not actually happening on current ledgers. Net: a complete, tested reflection-pass prompt is sitting in legacy code, ready to be lifted, retimed, and re-pointed at a new field.

**4. Eight in-scope post-script prompt-assembly sites currently consume `style`, `genre`, or hardcoded era language.** The full inventory is in §3. The two most actionable findings:
- `visual/batch_flux_portrait_render.py::_build_portrait_prompt` defaults `style_anchor` to the literal string `"1940s noir radio drama style"`. This is a hardcoded era anchor inside a portrait generator and directly contradicts the era-neutral directive that landed with the 10-preset style set on 2026-05-10.
- `nodes/otr_video_plan.py::_DEFAULT_STYLE_TAIL` is `"cinematic, 35mm film look, 1980s broadcast aesthetic, subtle film grain, volumetric lighting"` — another hardcoded era anchor, this time at module scope, applied to every PASS 1 / PASS 2 / PASS 3 FLUX composite prompt.

Both of these "leak" a period that the picker slug never asked for.

**5. MusicGen already does the hybrid pattern the problem statement proposes.** `_resolve_cue_from_style` in `nodes/musicgen_theme.py` composes `palette[style][cue_id]` + `_mood_suffix(meta.news.script_brief)` + `_PROMPT_TAIL`. The slug anchors instrument register; the brief layers mood. This is the cleanest precedent in the codebase for the "slug stays as genre anchor, brief feeds in as supplementary mood/flavor" pattern described in problem-statement §2a. Worth mining the exact composition order before designing the FLUX/LTX/HuMo integrations.

---

## 1. The "virulent genre" — provenance, footprint, why it shouldn't survive

### 1.1 What `_GENRE_BY_STYLE` is

A 10-entry `dict[str, str]` in `nodes/OTR_LedgerScriptWriter.py` lines 246-257 mapping each style slug to a short genre phrase:

```
closed_room_suspense       -> thriller audio drama
detective_case_file        -> detective audio drama
pulp_serial_cliffhanger    -> pulp serial audio drama
mission_control_procedural -> procedural audio drama
deep_space_distress_call   -> sci-fi audio drama
noir_interrogation         -> noir audio drama
small_town_uncanny         -> uncanny audio drama
radio_newsroom_emergency   -> newsroom audio drama
haunted_broadcast_signal   -> horror audio drama
laboratory_containment     -> containment audio drama
```

Two resolvers wrap it:
- `_resolve_genre(style)` — strict, raises on empty or unknown slug per standing directive #1.
- `_preview_genre(style)` — best-effort, used by UI/demo paths.

### 1.2 When and why it was reintroduced

Sprint 6.1 of voice-path-cleanbreak, committed 2026-05-12. The block comment at OTR_LedgerScriptWriter.py:236-244 says:

> Genre table for the `meta.visual_plan.genre` stamp. Replaces the hardcoded "audio drama" fallback Sprint 2 used. The genre string surfaces in:
>   - FLUX scene-prompt composition (style_tail + genre)
>   - episode metadata (treatment txt, video info card)
>
> Drift guard: tests/test_musicgen_style_palette.py asserts every entry in `_STYLE_PICKER_SEED_POOL` has an explicit row in this table.

The stamp site is at OTR_LedgerScriptWriter.py:2400:
```
meta["visual_plan"] = {
    ...
    "style":      resolved["style"],
    "genre":      _resolve_genre(resolved["style"]),
}
```

### 1.3 Where it's actually consumed

A full grep finds three live readers across all production code paths:

| File | Line | Surface | What it reads |
|---|---|---|---|
| `nodes/otr_video_plan.py` | 306 | `_visual_plan_from_script_json` | `"genre": visual_plan.get("genre") or ""` — stored on the projected `director` dict, but **no downstream consumer in this file reads `director["genre"]`**. The four `build_*` helpers and `compose_shot_prompt` consume `style`, `era_tail`, `style_tail`, and `portrait_prompt`, never `genre`. |
| `nodes/video_engine.py` | 711 | `_parse_hud_data` | `"style": style or genre or "sci-fi"` — `genre` is only read if `style` is falsy, which never happens for a properly-stamped ledger (`_resolve_genre` raises before that point). |
| `nodes/video_engine.py` | 836 | `_TelemetryHUDRenderer._build_left` | `self.data.get("style", self.data.get("genre", "?"))` — defensive back-compat for ledgers that pre-date the 2026-05-02 "style" key cleanup. Same pattern as 711: `genre` only fires if `style` is missing. |
| `nodes/video_engine.py` | 1075 | `_write_story_treatment` | `style = style or genre or "audio drama"` — treatment text header. Same pattern. |

The block comment's claim about "FLUX scene-prompt composition (style_tail + genre)" is **stale**. FLUX does not consume `genre` anywhere. The FLUX scene-prompt composers are `_parse_env_prompts` (uses `style_suffix` widget) and `compose_shot_prompt` (uses `style_tail` + `era_tail`, both keyed off `style`). The genre string never enters a FLUX prompt.

### 1.4 What this means for the cleanbreak

The problem statement (§6) says `genre` is being deleted in a parallel housekeeping pass and not to propose integration with it. That's the right call. Concretely, removal touches:

- `_GENRE_BY_STYLE`, `_resolve_genre`, `_preview_genre` in `OTR_LedgerScriptWriter.py:236-301`
- The stamp at `OTR_LedgerScriptWriter.py:2400` (drop the `"genre"` key from `visual_plan`)
- The projection at `otr_video_plan.py:306` (drop `"genre"`)
- The three video_engine fall-throughs (collapse `style or genre or "..."` to just `style or "..."`)
- `tests/test_musicgen_style_palette.py` lines 229-331 (the genre drift guard + the `_resolve_genre` / `_preview_genre` tests)
- Block comments in OTR_LedgerScriptWriter.py:236-244, 2356-2370 referring to "S6.1" and `_GENRE_BY_STYLE`

The `meta.style` slug already does the job everywhere `genre` is read. There is no information loss in removing `genre` — `_GENRE_BY_STYLE` is a deterministic function of `style`, so any downstream consumer that wants the human-readable genre string can compute it locally without persisting it.

### 1.5 Why this matters to the `story_brief` design

`meta.visual_plan.genre` is a **categorical projection** of `meta.style`. `meta.story_brief` will be a **descriptive reflection** on `lines[]`. The two answer different questions:

- `genre` ("thriller audio drama") — what bucket the writer started in
- `story_brief` ("single-room interrogation under a swinging bare bulb; rain-streaked window; one detective, one suspect, sweat and cigarette smoke; 1947 LA grime") — what the episode actually became

Keeping `genre` alive while introducing `story_brief` would let two competing flavor sources coexist and bid for the same prompt real estate, which is the exact failure mode the problem statement is trying to fix. Deleting `genre` first leaves a clean slate for `story_brief` to land in.

---

## 2. Post-script-write LLM call inventory

Sites where an LLM is called **after** the script has been assembled into `lines[]` and the cast is locked. Short list.

### 2.1 In the LPL writer path (`OTR_LedgerScriptWriter.py`)

| Site | Section | Purpose | Reads | Writes |
|---|---|---|---|---|
| `_generate_title_from_script` | J.5, line 2205-2243 | Regenerate a title from the final assembled script when the existing title is in `_STUCK_TITLE_DEFAULTS` (empty, "the last frequency", "untitled", "signal lost", "(pending)", etc.) | Full script text from `assemble_script_text_from_ledger(led.data)` | `meta["episode_title"]`, `meta["title_source"] = "llm_post_composition"`, optional `meta["title_substitution"]` |

That's it. Every other LLM call in the writer is upstream of script assembly:
- `_otr_outline.py` — outline generation (pre-write)
- `_otr_line_composer.py` — per-line composition (during write)
- `_otr_cast_contract.py` — cast lock (mid-write)
- `_otr_ledger_reviewer.py` — review pass (mid-write)
- `_otr_style_picker.py` — two-pass style picker (pre-write)
- `news_interpreter.py` — produces all four `meta.news.*_brief` fields (pre-write, from the raw RSS article)

### 2.2 In legacy code that the LPL path replaced

| Site | File | Status |
|---|---|---|
| `_generate_ltx_style_brief` | `story_orchestrator.py:3418` | **Orphaned.** Last live consumer (LTX prompt prepend) was removed by BUG-LOCAL-112 on 2026-05-06. The LPL writer does not call this generator, so `meta.ltx_style_brief` is not being stamped on current ledgers despite the comment in `batch_ltx_render.py:396` claiming it is. This is the prior art for `meta.story_brief`. See §4. |

### 2.3 What "post-script-write" means architecturally

The problem statement §3 places the reflection pass "between LPL writer exit (script + cast locked) and FreezeCascade entry." Today there is no node in that position — the LPL writer exits, FreezeCascade runs, and downstream visual nodes start consuming the frozen ledger. The reflection pass would be a new step, either:
- Inside `OTR_LedgerScriptWriter.execute()` after section K.5 / before return (single-process call, no new node, ledger stamped before the writer returns)
- A new dedicated node between the writer and FreezeCascade in the workflow JSON (separate concern, separate test surface)

Both are viable. Open question — see §6.

---

## 3. In-scope post-story-gen prompt assembly inventory

Every site where a text prompt is assembled before being passed to a FLUX, LTX, HuMo, or MusicGen call. Bark / Kokoro / AudioGen / ProcSFX are excluded per problem-statement §2a.

For each site: file + line, current template/scaffolding, what feeds scene-specific flavor today, model target, and a one-line `story_brief` integration sketch (sketch only — not a build spec).

### 3.1 FLUX — environment renders

**File:** `visual/batch_flux_render.py`
**Function:** `_parse_env_prompts` (line 233)
**Current template:**
```
"{env_token.description}, {style_suffix}"
```
where `style_suffix` defaults to `_DEFAULT_STYLE_SUFFIX`:
```
"cinematic, 35mm film, anamorphic lens, volumetric lighting,
 heavy vignette, muted color grade, sharp focus"
```
**Flavor source today:** the description from env-tokens emitted upstream by `otr_video_plan.build_pass2_scene_prompts` (which already concatenates `_DEFAULT_STYLE_TAIL`), plus the node's own `style_suffix` widget. No `style` slug, no `genre`, no `meta.news.*_brief` read — the env description from the planner carries all of it.
**Model:** FLUX (Comfy-Org safetensors).
**`story_brief` sketch:** Append `meta.story_brief` to the description before the style_suffix tail. The order matters: env description first (subject), brief second (specific scene flavor), style_suffix last (universal cinematic register). Concatenate in `_parse_env_prompts` rather than in the upstream planner so the planner's tokens don't double-bake the brief.

### 3.2 FLUX — radio bookend still

**File:** `visual/batch_flux_render.py`
**Function:** `_build_dynamic_radio_prompt(led)` (line 73)
**Current template:** tier chain with these resolution candidates:
1. `meta.gen_params_initial.style` (the slug)
2. `meta.gen_params_initial.style_custom` (free-text override when style="custom")
3. `scenes[0].env` or `scenes[0].description` (first-scene fallback)
4. `episode_id` slug
5. `_RADIO_FALLBACK_PROMPT` (hardcoded: `"sci-fi retrofuturistic radio broadcast unit, glowing CRT frequency display, copper vacuum tubes haloed in plasma, brushed steel chassis with art-deco engraving, dim amber and cyan rim lighting, dust-mote atmosphere, ..."`)

The resolved descriptor is wrapped as:
```
"{descriptor_capped} radio broadcast unit, set in {scene_hint}, {_RADIO_PROMPT_SUFFIX}"
```
**Flavor source today:** the picker slug (or its free-text override), capped at 80 chars, plus a 60-char first-scene-env hint. No story reflection — the radio's aesthetic is hypothesized at picker time.
**Model:** FLUX (Comfy-Org safetensors).
**`story_brief` sketch:** The radio bookend is the strongest candidate in the codebase for `story_brief` integration — the prompt explicitly tries to "look at the story" but only has the slug and the first scene's env tag. Replace the tier-4 (`scenes[0].env`) and tier-5 (`episode_id`) tiers with `meta.story_brief` when present, falling through to the current chain when absent. The tier-1 slug stays as a coarse anchor; the brief replaces the specific-setting tier. (BUG-LOCAL-024 comment quotes Jeffrey directly: *"we always need flux to look at the story and render that radio for the music, announcer and sfx."* The brief is the concretization of that ask.)

### 3.3 FLUX — character portraits

**File:** `visual/batch_flux_portrait_render.py`
**Function:** `_build_portrait_prompt(speaker, appearance, style_anchor)` (line 98)
**Current template:**
```
"{style_anchor},
 head and shoulders portrait of {speaker},
 {appearance},
 neutral expression, centered composition, frontal pose facing camera,
 soft studio lighting, 35mm film grain,
 no other characters in frame, no background props"
```
**Flavor source today:** `style_anchor` defaults to the literal string **`"1940s noir radio drama style"`** when the caller doesn't pass one. The `appearance` field comes from `cast[].character_description` (Tier 1 cast contract output).
**Model:** FLUX (Comfy-Org safetensors).
**`story_brief` sketch:** Two issues here. (1) The default `style_anchor` is a hardcoded era literal and directly violates the era-neutral directive that landed 2026-05-10 — this is a separate cleanbreak item regardless of `story_brief`. (2) For `story_brief` integration: portraits are character-anchored, not scene-anchored, so the brief's setting/lighting/object content is mostly noise for the portrait composition. But the brief's **lighting and atmosphere** cues are relevant for matching the portrait look to the episode. Candidate: split the brief into lighting/atmosphere phrase fragments at consumption time, append only those, or accept the noise and append the whole brief at the tail. Flag for round-robin — this is the site where the question "does the brief help or hurt?" is most acute.

### 3.4 LTX — environment / motion clips

**File:** `nodes/batch_ltx_render.py`
**Function:** `_build_ltx_role_prompt(role, line, ledger)` (line 404)
**Current template:** Just the per-role entry from `_PROMPT_BY_ROLE` (line 350). Example for `announcer`:
```
"Continuous shot, same console throughout. Tuning dial needle
 sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille
 trembles. Dust motes drift. Slow handheld dolly forward."
```
Five role variants (announcer, music_open, music_close, music_inter, sfx), each <160 chars. The `line` and `ledger` arguments are accepted for API stability but **deliberately unused** post-BUG-LOCAL-112.
**Flavor source today:** None — the prompt is brutally minimal by design. The visual identity comes from the FLUX radio bookend still feeding LTXVImgToVideoConditionOnly (the i2v anchor). Per BUG-LOCAL-112 round-robin: appending per-episode style brief, scene_env, and style tone to LTX prompts pushed them to 600-800 chars and diluted motion verbs into a static set tableau. The fix was to feed LTX only the motion-centric template and let the i2v anchor carry visual identity.
**Model:** LTX-Video v2.3.
**`story_brief` sketch:** **Do not integrate `story_brief` into the LTX motion prompt.** BUG-LOCAL-112's round-robin convergence is recent (2026-05-06) and the regression risk is high — adding 15-40 words of descriptive flavor would push the prompt back over the 200-300 char threshold that dilutes motion. The right place for `story_brief` to influence LTX output is **indirectly, via the FLUX radio bookend still** (§3.2). The bookend is the i2v anchor; if the brief shapes the bookend, the brief shapes what LTX renders motion on top of. This preserves BUG-LOCAL-112's fix and still gets the per-episode flavor into the LTX output. This is worth being explicit about in the build sprint — LTX looks tempting because it's a video model with a prompt, but the correct integration shape is upstream of LTX.

### 3.5 HuMo — lip-sync video

**File:** `nodes/batch_humo_render.py`
**Function:** `_build_pos_prompt(speaker, ln, cast)` (line 1143)
**Current template:**
```
"{speaker_desc}, {_DEFAULT_POS_SUFFIX}"
```
where `speaker_desc` is the matching cast member's `character_description` (Tier 1 cast contract output), and `_DEFAULT_POS_SUFFIX` (line 498) is:
```
"dimly lit interior, ambient cinematic lighting,
 35mm film grain, shallow depth of field"
```
**Flavor source today:** the cast member's description, plus a static aesthetic suffix. No `style` slug, no `genre`, no scene flavor. The negative prompt is a fixed ByteDance-derived Chinese suppression string.
**Model:** HuMo (Wan 2.1 / Kijai pack).
**`story_brief` sketch:** HuMo is the right place to inject `story_brief` for character-line clips — the model needs to know that the speaker is in a "single-room interrogation under a swinging bare bulb" so the lip-sync clip's environment matches the episode's bookend renders. Two viable integration shapes:
  - **Clean append:** `speaker_desc + story_brief + _DEFAULT_POS_SUFFIX` — the brief sits between subject and aesthetic.
  - **Replace `_DEFAULT_POS_SUFFIX`:** `speaker_desc + story_brief` — the brief is the suffix; drop the static `_DEFAULT_POS_SUFFIX` because the brief already provides lighting and atmosphere.

The second option is cleaner but riskier (the static suffix was tuned for HuMo's specific lighting failure modes). Flag for round-robin. Verified per `reference_humo_per_clip_wall_time` memory: HuMo runs ~10-12 min per character line on the 5080, so prompt-quality regressions are very expensive to detect through real renders — any HuMo prompt change wants to go through round-robin before a soak.

### 3.6 MusicGen — theme cues (opening, closing, interstitial)

**File:** `nodes/musicgen_theme.py`
**Function:** `_resolve_cue_from_style(cue_id, style, mood_suffix)` (line 377)
**Current template:**
```
"{_STYLE_PALETTE[style][cue_id]}{mood_suffix}{_PROMPT_TAIL}"
```
Where:
- `_STYLE_PALETTE[style][cue_id]` is a 10-style × 3-cue palette of era-neutral instrumentation phrases (e.g. for `noir_interrogation` / `opening`: `"muted solo trumpet, low double bass walk, smoky tenor saxophone, dim and atmospheric"`).
- `_mood_suffix(script_brief)` mines `meta.news.script_brief` for any of 10 mood keywords (`betrayal`, `discovery`, `loss`, `urgent`, `isolation`, `danger`, `mystery`, `triumph`, `conflict`, `silence`) and concatenates the matching mood tags (e.g. `"minor mode, unresolved tension"`).
- `_PROMPT_TAIL` is `", instrumental only, no dialogue, no vocals"`.

Hard-fails on unknown `style` slug — no default palette.
**Flavor source today:** Already hybrid. `style` slug → palette (genre anchor). `meta.news.script_brief` → keyword scan → mood tags (supplementary mood). No LLM at the music-render side.
**Model:** MusicGen-medium (transformers, ~6 GB VRAM).
**`story_brief` sketch:** MusicGen is the **template** for the integration pattern, not just another consumer. Two viable changes:
  - **Add `story_brief` as a third input to `_mood_suffix`:** keyword-scan the brief alongside `script_brief` for mood keywords; merge the matches; concatenate. Cheap, preserves determinism, doesn't change MusicGen output character.
  - **Replace `script_brief` with `story_brief` as the mood source:** `story_brief` is a post-script reflection so it's strictly more aligned with the music's narrative function than the pre-script news brief. But this loses the news-story mood signal entirely.

The first option is the safer integration shape and fits the problem statement's "supplementary mood" framing. The second is cleaner architecturally (one brief, one source) but loses signal. Flag for round-robin.

Note that this is the only site that *already* reads a `meta.news.*_brief` field. The pattern — `palette[style][cue] + brief_derived_mood + universal_tail` — is what FLUX env, FLUX radio, and HuMo should look like after `story_brief` lands.

### 3.7 `otr_video_plan` — PASS 1 / PASS 2 / PASS 3 composite assembly

**File:** `nodes/otr_video_plan.py`
**Functions:** `build_pass1_char_prompts` (line 310), `build_pass2_scene_prompts` (line 384), `build_shot_plan` (line 441). All three call `compose_shot_prompt` (line 254) which concatenates:
```
"{portrait}, {scene_visual}, {shot_hint}, {era_tail}, {style_tail}"
```
**Flavor source today:**
- `portrait` — `cast[].character_description` via `resolve_character_portrait` (line 164). Falls back through cast.portrait_prompt → generic template. No `genre` consumer.
- `scene_visual` — `scenes[i].visual_prompt` from `meta.visual_plan.scenes`. Currently always empty in the LPL writer path (the writer emits `meta.visual_plan.scenes = []` intentionally — comment at OTR_LedgerScriptWriter.py:2380-2383). So in practice `scene_visual` is always synthesized from the scene_id placeholder.
- `era_tail` — `_ERA_TAIL_BY_STYLE[style]` (line 93). 10-entry table parallel to MusicGen's `_STYLE_PALETTE`, keyed by the same slugs. Era-neutral visual descriptors (e.g. for `noir_interrogation`: `"single overhead lamp, deep cast shadows, smoke-filtered air, hard contrast on the subject"`).
- `style_tail` — defaults to `_DEFAULT_STYLE_TAIL` (line 78): **`"cinematic, 35mm film look, 1980s broadcast aesthetic, subtle film grain, volumetric lighting"`** — another hardcoded era anchor at module scope, this time "1980s" rather than "1940s".

**Model:** Output is emitted as env-tokens consumed by `OTR_BatchFluxRender` — so the actual model is FLUX, but this file is the prompt assembler that feeds FLUX.
**`story_brief` sketch:** This is the **highest-leverage** integration site because it feeds every FLUX shot. Two cuts:
  - **Append the brief between `scene_visual` and `era_tail`:** `portrait + scene_visual + story_brief + shot_hint + era_tail + style_tail`. The brief replaces the empty `scene_visual` content that the LPL writer doesn't currently emit, while the slug-keyed `era_tail` keeps style-specific lighting consistent.
  - **Replace `scene_visual` with the brief entirely:** since the LPL writer emits empty scenes, `scene_visual` is currently doing no work. The brief could become the de-facto scene-visual source, with the era_tail and style_tail still providing slug-keyed flavor. Cleaner; eliminates the dead scene-visual path.

Secondary issue: the hardcoded `"1980s broadcast aesthetic"` in `_DEFAULT_STYLE_TAIL` should probably be deleted as part of the same cleanbreak — see §5 below.

### 3.8 `video_engine` HUD card + treatment text (consumer-only, no LLM)

**File:** `nodes/video_engine.py`
**Functions:** `_parse_hud_data` (line 599), `_TelemetryHUDRenderer._build_left` (line 814), `_write_story_treatment` (line 1016)
**Current consumption:** All three read `style` and `genre` from the projected director dict and print them on the HUD / in the treatment text. No LLM, no model.
**`story_brief` sketch:** The HUD has space for one or two extra lines. Adding a `STORY:` row showing `meta.story_brief` (truncated to one display line) would let the viewer see what the episode actually is rather than just its style slug. The treatment text has even more room — append a "STORY BRIEF" section between the existing "Title / Style / Produced" header and the news-seed block. Both are decorative integrations and can be deferred.

### 3.9 Summary table

| # | File | Function | Model | Currently uses `style`? | Currently uses `genre`? | Hardcoded era literals? | `story_brief` priority |
|---|---|---|---|---|---|---|---|
| 1 | `visual/batch_flux_render.py` | `_parse_env_prompts` | FLUX | indirect (via planner tokens) | no | no | medium |
| 2 | `visual/batch_flux_render.py` | `_build_dynamic_radio_prompt` | FLUX | yes (tier 1) | no | indirect (in `_RADIO_FALLBACK_PROMPT`) | high |
| 3 | `visual/batch_flux_portrait_render.py` | `_build_portrait_prompt` | FLUX | as `style_anchor` arg | no | **yes — "1940s noir radio drama style"** | low (cleanbreak the era literal separately) |
| 4 | `nodes/batch_ltx_render.py` | `_build_ltx_role_prompt` | LTX | no (deliberately) | no | no | **do not integrate** — go via the FLUX bookend instead |
| 5 | `nodes/batch_humo_render.py` | `_build_pos_prompt` | HuMo | no | no | no (`_DEFAULT_POS_SUFFIX` is era-neutral) | high |
| 6 | `nodes/musicgen_theme.py` | `_resolve_cue_from_style` | MusicGen | yes (palette key) | no | no | medium (already hybrid via `script_brief`) |
| 7 | `nodes/otr_video_plan.py` | `compose_shot_prompt` (called by PASS 1/2/3) | FLUX (via planner tokens) | yes (`era_tail` key) | no | **yes — "1980s broadcast aesthetic" in `_DEFAULT_STYLE_TAIL`** | high (feeds every shot) |
| 8 | `nodes/video_engine.py` | `_parse_hud_data` / `_TelemetryHUDRenderer._build_left` / `_write_story_treatment` | none (display only) | yes | yes (fall-through only) | no | low (decorative) |

---

## 4. Prior art — the orphaned `_LTX_STYLE_BRIEF_PROMPT`

`story_orchestrator.py:3398-3415` defines a complete reflection-pass prompt that already does most of what `meta.story_brief` needs. Quoted verbatim because the build sprint will want to mine it:

```
You are writing a single-sentence VISUAL STYLE BRIEF for the
broadcast equipment shown on screen during an audio drama. Describe
ONLY the equipment / room aesthetic appropriate to this story's
setting and style. NO people, NO characters, NO action -- just the
look of the broadcasting equipment and the room it sits in.

Story style: {style}
Story snippet: {story_snippet}

Output ONE sentence (20-40 words) describing the broadcast equipment
and its room. The sentence should:
- Match the story's setting (extract from the snippet: lunar base,
  deep-space vessel, seabase, mars colony, orbital station,
  near-future newsroom, industrial-decay site, whatever fits)
- Use equipment design language that fits the setting AND style --
  do not default to any specific era's hardware unless the story
  explicitly implies it
- Include lighting and atmosphere cues that fit the style
- NOT mention people, hands, faces, voices, or anyone speaking
- Be ONE sentence with no preamble

Examples (one near-future newsroom, one deep-space vessel, one
industrial decay -- spanning the style range so no single hardware
era dominates):
- Near-future newsroom broadcast desk, edge-lit glass console with
  floating waveform overlays, cool overhead daylight, condensation
  rings on a steel coffee cup, hum of HVAC.
- Deep-space science vessel comms console, holographic dial readouts,
  recycled-atmosphere haze, speaker grille mounted into a curved
  bulkhead, magnetic dust drifting through volumetric beams.
- Rust-belt repurposed factory broadcast loft, scavenged industrial
  speaker bolted to a corroded I-beam, sodium-vapor work lamps,
  oil-stained concrete floor, occasional sparks from exposed wiring.

Visual brief:
```

The Python wrapper (`_generate_ltx_style_brief`, line 3418) handles:
- 80-token output budget on Mistral-Nemo (~5-10s cost)
- temperature 0.7 / top_p 0.9
- prefix stripping ("Visual brief:", "BRIEF:", quotes, leading bullets)
- single-line forcing (`text.strip().split("\n")[0]`)
- 300-char hard cap with sentence-boundary trim
- non-fatal failure path — returns `""` on any exception, caller falls through

### What the new design has to change

| Aspect | Legacy `_LTX_STYLE_BRIEF_PROMPT` | New `meta.story_brief` proposal |
|---|---|---|
| **Subject** | The broadcast equipment in the room | The scene flavor of the episode — setting, lighting, atmosphere, period, specific objects |
| **Input** | `style` slug + `story_snippet` (raw text fragment, taken **before** the script existed in finished form) | `lines[]` (the assembled final script) + `cast[]` |
| **Timing** | Pre-write or mid-write — story_snippet was the outline/early-draft text | Post-write — script and cast are locked |
| **Length** | 20-40 words, 300 char hard cap | 15-40 words per problem statement §4.2 |
| **Scope** | "Broadcast equipment and its room" — narrowed to the radio prop's surroundings | Full scene flavor — broader |
| **Era guidance** | "do not default to any specific era's hardware unless the story explicitly implies it" — already era-neutral | Same — keep this exact line |
| **No people** | Explicit | Should stay explicit — FLUX env / radio bookend / LTX i2v anchor all want people-free renders |
| **Examples** | 3 examples spanning the style range | Same approach, refresh examples to span the new 10-preset style set rather than the older era language |
| **Failure mode** | Returns `""`, caller falls through | Same — see open question 6.3 |

### What can be lifted unchanged

The Python wrapper's output cleanup is solid and worth keeping verbatim. Specifically:
- Prefix stripping
- Single-line forcing
- Smart-quote stripping
- Sentence-boundary length trim
- Non-fatal exception handling

The prompt template needs a rewrite (input shape changes from snippet → full script, scope broadens from radio room → full scene flavor), but the bullet-point output discipline, the explicit no-people clause, the example-driven era-neutrality nudge, and the "ONE sentence, no preamble" footer should all stay.

### What is broken about the legacy stamping path

The comment at `batch_ltx_render.py:396` claims `ledger.meta.ltx_style_brief` is "still stamped by OTR_LedgerScriptWriter" but a grep of `OTR_LedgerScriptWriter.py` finds no call to `_generate_ltx_style_brief`. The LPL writer simply doesn't invoke this code path. So:
- Current ledgers do not have `meta.ltx_style_brief` populated.
- The comment in `batch_ltx_render.py` is wrong.
- The function in `story_orchestrator.py` is dead code.

This means there is zero migration burden — the field name `meta.ltx_style_brief` can be retired in favor of `meta.story_brief` without coordinating with any live consumer.

---

## 5. Secondary cleanbreak candidates surfaced by this inventory

Not in the `story_brief` build sprint, but worth flagging now because they shape the surface area:

### 5.1 Hardcoded era literals in two FLUX assemblers

- `visual/batch_flux_portrait_render.py:107` — `style_anchor` defaults to `"1940s noir radio drama style"`. Contradicts the era-neutral directive from the 2026-05-10 style preset set.
- `nodes/otr_video_plan.py:79` — `_DEFAULT_STYLE_TAIL` includes `"1980s broadcast aesthetic"`. Same problem, different decade.

Both should be replaced with era-neutral language before `story_brief` lands, otherwise the brief and the hardcoded literals will fight in the prompt. Suggested replacement: keep the cinematic-grammar parts ("cinematic, 35mm film grain, volumetric lighting") and drop the era word.

### 5.2 The `_RADIO_FALLBACK_PROMPT` is sci-fi-flavored

`visual/batch_flux_render.py:54-61` has the hardcoded radio fallback baked as a sci-fi retrofuturist still. This fires when every other tier fails. If `meta.story_brief` becomes a new tier in `_build_dynamic_radio_prompt`, the hardcoded fallback becomes much rarer — but it still exists. Worth a separate decision about whether the fallback should be more era-neutral, or whether it should be removed entirely now that the brief tier covers most real cases.

### 5.3 `meta.visual_plan.scenes` is always empty

`OTR_LedgerScriptWriter.py:2398` deliberately stamps `"scenes": []` — the writer doesn't emit scene-level visual blocking, comment at line 2380 says so. This means `otr_video_plan.extract_scenes` always returns `[]`, which means `build_pass2_scene_prompts` emits zero tokens, which means PASS 2 of the video plan does nothing on real episodes. If `story_brief` is the new scene-level signal, PASS 2's role needs to be reconsidered: either fold PASS 2 into PASS 3 (use the brief as the scene flavor for every composite), or have the reflection pass emit `meta.scenes[].brief` as v3 to revive PASS 2 properly. Per problem statement §2 — per-scene briefs are explicitly v3, not v2. So PASS 2 stays dead until v3.

### 5.4 The `_STYLE_PICKER_SEED_POOL` is the single source of truth for slug enumeration — except where it isn't

Four tables in three files all enumerate the same 10 style slugs:
1. `_STYLE_PICKER_SEED_POOL` in `_otr_style_picker.py` (canonical source)
2. `_GENRE_BY_STYLE` in `OTR_LedgerScriptWriter.py` (Sprint 6.1 reintroduction — proposed for deletion)
3. `_STYLE_PALETTE` in `musicgen_theme.py`
4. `_MOOD_TAGS` in `musicgen_theme.py` (orthogonal — keyword set, not style-keyed)
5. `_ERA_TAIL_BY_STYLE` in `otr_video_plan.py`

The drift guard at `tests/test_musicgen_style_palette.py` already asserts these tables stay in sync. Adding `story_brief` will not introduce a new style-keyed table — the brief is freeform prose, not a slug-keyed lookup. So the table-count stays at three (palette + era_tail + picker_pool) post-genre-deletion. Good.

---

## 6. Open questions for round-robin review

Questions where the inventory surfaces a real judgment call and a single right answer is not obvious from the existing standing directives. Listed for reviewer weigh-in per problem statement §4.4.

### 6.1 Brief word-count window

Problem statement §4.2 proposes 15-40 words. Legacy `_LTX_STYLE_BRIEF_PROMPT` used 20-40 words. The example in §3 of the problem statement (`"single-room interrogation under a swinging bare bulb; rain-streaked window; one detective, one suspect, sweat and cigarette smoke; 1947 LA grime"`) is ~22 words. Question: is 15-40 the right window, or should it be 20-40 (matching the legacy prompt that already produced acceptable output)? 15-word briefs may be too short to give downstream consumers four distinct flavor cues (setting, lighting, atmosphere, period markers, specific objects = 5 categories).

### 6.2 Reflection pass position — inside the writer, or as a separate node?

Two options:
- **Inside `OTR_LedgerScriptWriter.execute()`:** call the reflection pass after section K.5 (visual_plan stamp) and before the return. Single-process, no new node, no workflow JSON edit, no ledger schema bump. But the writer is already long (2400+ lines) and adding a post-script LLM call to it grows the surface.
- **New dedicated node between writer and FreezeCascade:** a `OTR_StoryBriefReflection` node that takes `script_json` in, calls the LLM, stamps `meta.story_brief`, and emits the updated `script_json`. Separate test surface, separate failure mode, requires a workflow JSON edit and a fail-loud guardrail on the empty-string output path.

Tradeoff: cohesion-with-writer vs separation-of-concerns. Worth a round-robin call.

### 6.3 Failure mode — empty string or raise?

Standing directive #1 says no silent fallbacks on production surfaces. But the problem statement §6 lists this exact question as still-open. The legacy `_generate_ltx_style_brief` returned `""` on any exception, and callers treated empty as "fall back to existing behavior" — which is the standard non-fatal pattern. Empty-string-stamp would let downstream consumers continue working on older ledgers and on rare LLM failures; raise would force the writer to abort the entire episode on a 5-second flavor-text generation failure. The cost-benefit favors empty-string-stamp, but it's a documented violation of standing directive #1 and needs explicit reviewer signoff.

### 6.4 Should `style` slug and `story_brief` ever conflict, and if so, how?

Concrete scenario: writer picks `noir_interrogation` at the picker stage, the LLM drifts and writes a script set on a Mars colony, the reflection pass produces a brief like `"Mars colony hydroponics deck, sodium-vapor lamps, oxygen-recycler hum, cracked pressure visor on the workbench, red dust on every surface, 2087 atmosphere"`. The MusicGen palette is now playing muted trumpet and smoky tenor saxophone over Mars hydroponics. Question: is this a feature (slug = picker's hypothesis, brief = actual story, MusicGen continues honoring picker intent for genre consistency) or a bug (the entire point of `story_brief` is to make downstream match the story, so MusicGen should follow the brief)? Recommend reviewer position: feature. The slug owns audio-track genre identity (because MusicGen output is hard to "drift" out of without sounding wrong); the brief owns visual scene-flavor (because FLUX/LTX/HuMo prompts are flexible enough to render the brief without the slug's genre register). But this is exactly the kind of edge case that round-robin should pressure-test.

### 6.5 Token budget on the reflection call

Typical episode is 6-12 character lines plus 4-8 announcer/SFX/music entries — call it 15-25 ledger rows. Each row is 30-150 tokens of dialogue + 20-30 tokens of metadata. Full assembled script is 500-1500 tokens. Cast block adds another 200-400 tokens. Reflection prompt template adds ~250 tokens. Total: 1000-2200 tokens of input.

The LPL writer's `context_cap` is 8192 with `max_new_tokens` reserved separately, so 2200 input + 80 output fits comfortably. But: long-line episodes (15+ minute runs) can blow past 2200. The legacy `_generate_ltx_style_brief` only fed a `story_snippet` capped at 500 chars — much smaller. Question: do we truncate the script before feeding it (lose tail context), summarize the script first (extra LLM call), or trust the input cap and accept that very long episodes won't fit? Recommendation: feed the assembled script verbatim, rely on the writer's existing `_LLM_CACHE["context_cap"]` truncation guard. The reflection pass's output is short enough that input crowding is unlikely.

### 6.6 What happens to `meta.ltx_style_brief`?

The legacy field is mentioned in `batch_ltx_render.py:396` as "still stamped by OTR_LedgerScriptWriter" but it isn't actually stamped on current ledgers. Two options:
- Retire the field name entirely and use only `meta.story_brief` going forward. Update the LTX comment to remove the stale reference. Clean.
- Keep `meta.ltx_style_brief` as an alias / re-export of `meta.story_brief` for ledgers in flight. Pointless — there are no ledgers in flight using the legacy field.

Per the standing "no legacy back-compat" feedback memory, retire the name. The `meta.story_brief` name in the problem statement is the canonical going-forward field.

---

## 7. Recommended next steps (analysis only, not a build plan)

In this order:

1. **Round-robin §1.1-1.5 (delete `_GENRE_BY_STYLE`).** Confirm with ChatGPT + Gemini that the genre table has no downstream consumer outside the two cosmetic display surfaces, and that those surfaces work correctly with `style` alone. Once confirmed, the cleanbreak is a small focused commit (deletes touched in §1.4).

2. **Round-robin the open questions in §6.** Specifically 6.2 (writer-inline vs new node), 6.3 (empty-string vs raise), 6.4 (slug-vs-brief conflict policy), and 6.5 (token budget). The other two (6.1 word-count, 6.6 retire `ltx_style_brief`) have a clear default — flag them but don't burn a round-robin slot.

3. **Round-robin the `story_brief` reflection-pass prompt itself.** Take the legacy `_LTX_STYLE_BRIEF_PROMPT` as the starting point (§4), rewrite for the new input shape (`lines[]` + `cast[]` instead of `style + story_snippet`) and broader scope (full scene flavor, not just the radio room), and put it through ChatGPT → Gemini → synthesize per the round-robin pattern.

4. **Round-robin the integration shape at each of the 8 sites in §3.9.** The non-obvious calls are:
   - §3.3 portrait builder — does the brief help or hurt? Probably narrow the brief to its lighting/atmosphere fragment, but flag.
   - §3.4 LTX — do not integrate directly; verify reviewers agree the indirect-via-FLUX-bookend path is correct.
   - §3.5 HuMo — clean append vs replace `_DEFAULT_POS_SUFFIX`. Lean toward clean append (lower regression risk on a 10-minute-per-clip model).
   - §3.6 MusicGen — keyword-scan the brief alongside `script_brief`, don't replace.
   - §3.7 `otr_video_plan` — append between scene_visual and era_tail, or replace the always-empty scene_visual entirely. Probably the latter.

5. **Separate cleanbreak: era literals (§5.1).** Either bundle with the `genre` cleanbreak or do as a follow-up commit. Either way, do this before `story_brief` ships so the brief doesn't fight a hardcoded "1940s" or "1980s" in the same prompt.

6. **Build sprint:** straightforward after the above lands. One new reflection pass (or new node), one new ledger field, N append-edits across the inventory sites. Estimate: 2-3 commits.

---

## 8. Files referenced

The complete set of files touched or read by this analysis, for reviewer cross-checking:

- `nodes/OTR_LedgerScriptWriter.py` — lines 236-301 (genre table + resolvers), 574+ (title regen), 2205-2243 (J.5 post-composition regen call), 2356-2402 (K.5 visual_plan stamp + genre call site)
- `nodes/musicgen_theme.py` — lines 70-175 (`_PROMPT_TAIL`, `_STYLE_PALETTE`, `_MOOD_TAGS`), 360-397 (`_mood_suffix`, `_resolve_cue_from_style`), 459-503 (`render` consumption of `style` + `script_brief`)
- `nodes/otr_video_plan.py` — lines 78-108 (`_DEFAULT_STYLE_TAIL`, `_ERA_TAIL_BY_STYLE`), 156-273 (`resolve_era_tail`, `resolve_character_portrait`, `compose_shot_prompt`), 276-307 (`_visual_plan_from_script_json` — the only `genre` reader in the file), 310-712 (the three `build_*` helpers)
- `nodes/batch_ltx_render.py` — lines 300-376 (`_PROMPT_BY_ROLE` + the BUG-LOCAL-112 comment block), 378-414 (`_build_ltx_role_prompt`), 396-403 (the stale `ltx_style_brief` comment)
- `nodes/batch_humo_render.py` — line 498 (`_DEFAULT_POS_SUFFIX`), 1143-1162 (`_build_pos_prompt`)
- `nodes/video_engine.py` — lines 599-719 (`_parse_hud_data`), 833-842 (HUD left panel `STYLE` row), 1016-1192 (`_write_story_treatment`), 1286-1295 (call-site style/genre derivation)
- `visual/batch_flux_render.py` — lines 30-70 (`_DEFAULT_FALLBACK`, `_DEFAULT_STYLE_SUFFIX`, `_RADIO_FALLBACK_PROMPT`, `_RADIO_PROMPT_SUFFIX`), 73-202 (`_build_dynamic_radio_prompt`), 233-265 (`_parse_env_prompts`)
- `visual/batch_flux_portrait_render.py` — lines 98-123 (`_build_portrait_prompt` + the "1940s noir radio drama style" default)
- `nodes/story_orchestrator.py` — lines 3398-3416 (`_LTX_STYLE_BRIEF_PROMPT`), 3418-3472 (`_generate_ltx_style_brief`)
- `tests/test_musicgen_style_palette.py` — lines 229-331 (drift guards on `_GENRE_BY_STYLE` and `_STYLE_PALETTE`)
- `docs/2026-05-12-voice-path-cleanbreak-S6-S8-qa.md` — lines 121-353 (the Sprint 6 design doc that introduced `_GENRE_BY_STYLE`)
- `docs/BUG_LOG.md` — entries 1498-1543 (the BUG-LOCAL-112 LTX prompt-dilution diagnosis and fix)

End of research paper.
