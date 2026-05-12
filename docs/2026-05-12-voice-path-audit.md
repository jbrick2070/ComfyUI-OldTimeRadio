# Voice-Path Audit — Post-LFC Clean-Break (commit 1aed66d, tag v2.0-alpha-cleanbreak)

**Author:** Cowork session, 2026-05-12
**Branch:** v2.0-alpha
**HEAD:** `1aed66d` (LFC commit 12.20: 8th smoke check pins G5 invariant + criterion 5 polish)
**Tag note:** Jeffrey's session prompt placed `v2.0-alpha-cleanbreak` at `1aed66d`. The tag in the repo actually resolves to `f582a38`. HEAD is `1aed66d`. Minor — the tag is set, the sprint did land.
**Scope:** Read-only audit. No code changes. This doc is the input for a round-robin (ChatGPT + Gemini + Claude).

---

## 0. Standing directive (Jeffrey, 2026-05-12) — applies to this audit and every fix that follows

**No legacy. No back-compat. No secondary paths.** The upstream (writer + freeze cascade + ledger) was rebuilt clean over the 12.3 → 12.20 sprint specifically so the downstream could be rewritten clean too. Anything in the voice path that exists "for legacy workflows" or "as a fallback" or "during transition" is in scope to be deleted.

This is an extension of the ROADMAP "STANDING DIRECTIVE — NO LEGACY BACK-COMPAT" (Jeffrey, 2026-05-11). That entry covered renamed nodes / class names / field names. This audit applies the same rule to:

- Director-derived production_plan secondary inputs on voice nodes (Pattern 3 voice_map fallback in Bark, AudioGen's unused production_plan_json wire, Sequencer's unused production_plan_json wire).
- `OTR_LLMDirector` itself (delete, do not deprecate).
- Pre-L3 parser-list reader code in `BatchKokoroGenerator` (migrate to ledger consumers, no shim for the old shape).
- Hardcoded period defaults in `MusicGenTheme` `CUE_DEFAULTS` (delete; ledger-derived prompts only).
- Legacy single-line nodes `OTR_BarkTTS`, `OTR_SFXGenerator`, `OTR_VoiceRender` if not wired by any active workflow.
- Any "during transition" or "back-compat shim" language in the fix list — replaced with delete-and-rewire.

Saved workflow JSONs that reference deleted nodes / wires are **expected to be rewritten from scratch** against the new graph. The workflow JSON is the canonical surface — there is no parallel "legacy workflow" path to preserve.

**Acceptance rule for the follow-up commit (call it `voice-path-cleanbreak`):**

1. `grep -rn "production_plan_json" nodes/` returns hits only in `musicgen_theme.py` (if MusicGen keeps the socket during its ledger-aware rewrite — see §6) and `story_orchestrator.py::LLMDirector` (if Director is not yet deleted in the same commit).
2. `grep -rn "_voice_preset_for_character\|voice_assignments" nodes/batch_bark_generator.py nodes/batch_kokoro_generator.py` returns zero hits.
3. `grep -rn "1940s\|vintage\|old time radio" nodes/musicgen_theme.py` returns zero hits.
4. `workflows/otr_scifi_16gb_full.json` contains no `OTR_LLMDirector` node and no `production_plan_json` wires (other than MusicGen's transient socket if Phase 1 lands separately).
5. `__init__.py` does not register `OTR_LLMDirector`, `OTR_BarkTTS`, `OTR_SFXGenerator`, `OTR_VoiceRender`, or `OTR_BatchKokoroGenerator` (the legacy script-list reader) — either each class file is deleted, or the registration block is removed.
6. Bug Bible regression holds at its current baseline. New voice-path tests added in lockstep.

If any of those grep hits land non-zero, the commit hasn't finished the job.

---

## 1. The question

After the 12.3 → 12.20 sprint (robust L3 ledger, per-line polish, three-bucket meta layout, freeze cascade with try/finally unload, standalone Phase 4/5/6 nodes, G1–G5 interlocks), do the downstream voice nodes read from the L3 ledger as their single source of truth, or are they still reading from older script-text / script_json / Director surfaces that predated the ledger?

---

## 2. Topology — what the active workflow JSON actually wires

`workflows/otr_scifi_16gb_full.json` is the canonical production graph.

Two upstream "source" nodes feed all voice-side consumers:

| Source node | Output socket | Format on the wire |
|---|---|---|
| `OTR_LedgerFreezeCascade` (id 62) | `script_json` | **L3 ledger JSON** (post-freeze snapshot of in-flight ledger). Naming is legacy — the wire is named `script_json` but the payload is a ledger `dict` with `cast`, `lines[]`, `meta{}`, `schema_version="l3-2026-05-14"`. |
| `OTR_LLMDirector` (id 2) | `production_plan_json` | **Director-derived plan**, LLM-generated *from `script_text`*. Contains `voice_assignments`, `sfx_plan`, `music_plan`, etc. Pre-L3 surface; redundant where ledger has the same data. |

**Wires landing in voice-side consumers:**

| Link | From | To | Wire name |
|---|---|---|---|
|  2 | FreezeCascade.script_json | SceneSequencer.script_json | L3 ledger |
|  4 | Director.production_plan_json | SceneSequencer.production_plan_json | plan |
| 12 | FreezeCascade.script_json | BatchBarkGenerator.script_json | L3 ledger |
| 13 | Director.production_plan_json | BatchBarkGenerator.production_plan_json | plan |
| 19 | FreezeCascade.script_json | KokoroAnnouncer.script_json | L3 ledger |
| 21 | Director.production_plan_json | MusicGenTheme.production_plan_json | plan |
| 24 | FreezeCascade.script_json | BatchAudioGenGenerator.script_json | L3 ledger |
| 26 | Director.production_plan_json | BatchAudioGenGenerator.production_plan_json | plan |

`OTR_AudioEnhance` (id 4) and `OTR_EpisodeAssembler` (id 7) take only audio wires; they touch the ledger from disk via `_otr_ledger.in_flight_ledger_path()`.

---

## 3. Per-node table — source of truth + status

| # | Node (file) | Reads text/role/char_id/style from | L3-native? | Director-aware? | Status |
|---|---|---|---|---|---|
| 1 | `OTR_BatchBarkGenerator` (`batch_bark_generator.py`) | `_OTRLC.load_ledger(script_json)` + `iter_lines(roles={"character"})`; cast.voice_preset preferred, Director `voice_assignments` fallback (Pattern 3) | yes | yes (fallback only) | **Needs cleanup** — read path is clean, but the Director `voice_map` fallback at lines 519-524 + the `_voice_preset_for_character` helper + the `production_plan_json` socket all need to be deleted per the no-back-compat rule. Cast.voice_preset is the only valid source; empty → hard fail. |
| 2 | `OTR_KokoroAnnouncer` (`kokoro_announcer.py`) | `_OTRLC.load_ledger(script_json)` + `iter_lines(roles={"announcer"})`; seeded grab-bag from announcer pool (no Director read) | yes | no | **OK** — ledger-only. |
| 3 | `OTR_BatchAudioGenGenerator` (`batch_audiogen_generator.py`) | `_OTRLC.load_ledger(script_json)` + `iter_lines(roles={"sfx"})`; uses `line["text"]` as cue | yes | unused (plan wired but not read for SFX content) | **Needs cleanup** — content reads are clean, but the `production_plan_json` socket is dead weight. Delete socket + the `production_plan_or_empty` call. Cut workflow link 26. |
| 4 | `OTR_BatchProceduralSFX` (`batch_procedural_sfx.py`) | `_OTRLC.load_ledger(script_json)` + `iter_lines(roles={"sfx"})` | yes | n/a | **OK** — ledger-only. Not in active workflow but ready when wired. |
| 5 | `OTR_SceneSequencer` (`scene_sequencer.py`) | `_OTRLC.load_ledger(script_json)` + `iter_lines()` (no role filter) | yes | unused for line content | **Needs cleanup** — line iteration is clean, but the `production_plan_json` socket is unused for content. Delete socket + `production_plan_or_empty` call. Cut workflow link 4. |
| 6 | `OTR_EpisodeAssembler` (`scene_sequencer.py`) | In-flight ledger from disk via `_OTRL_PATHS.in_flight_ledger_path()` + `load_ledger_safe` | yes | no | **OK** — audited clean in previous sprint; takes no `script_json` input. |
| 7 | `OTR_AudioEnhance` (`audio_enhance.py`) | None on read; writes to ledger via in-flight path | n/a | no | **OK** — non-text consumer. |
| 8 | `OTR_MusicGenTheme` (`musicgen_theme.py`) | `json.loads(production_plan_json)["music_plan"]` ONLY. Falls back to `CUE_DEFAULTS` at lines 48–74 (hardcoded "1940s old time radio opening theme, warm brass fanfare, upright bass…"). **No ledger read at all.** | **no** | **only** | **HIGH-RISK** — zero L3 awareness. Hardcoded era literals. Already tracked as ROADMAP D2. |
| 9 | `OTR_BatchKokoroGenerator` (`batch_kokoro_generator.py`) | `json.loads(script_json)` then `for i, item in enumerate(script): item.get("type") == "dialogue"` — **pre-L3 parser-list shape**. Uses `character_name`, `voice_traits`, `line` fields that don't exist on L3. Director `voice_assignments` for preset lookup. | **no** | yes (only) | **DELETE OR REWRITE.** Two options per the no-back-compat rule: (a) delete the file + drop the registration if Kokoro-for-character isn't an active workflow target, or (b) rewrite from scratch to mirror `batch_bark_generator.py` (ledger consumers, cast.voice_preset only, no Director fallback). No middle-ground "migrate but keep legacy parse" path. |
| 10 | `OTR_LLMDirector` (`story_orchestrator.py:4644`) | `script_text` (raw rendered prose from writer) | n/a (upstream of ledger consumers) | self | **DELETE.** Standing Directive says delete legacy. Director's outputs are either redundant (`voice_assignments` overridden by cast, `sfx_plan` ignored, `production_plan` unused) or actively wrong (`music_plan` derived from text without seeing the L3 ledger). Delete the class, drop the registration, cut every workflow link. |
| 11 | `OTR_BarkTTS` (`bark_tts.py`) | Single-line input; legacy node | n/a | n/a | **DELETE.** Not wired by any active workflow. File + registration both go. |
| 12 | `OTR_SFXGenerator` (`sfx_generator.py`) | Single-line input; legacy node | n/a | n/a | **DELETE.** Same as above — file + registration. |
| 13 | `OTR_VoiceRender` (`voice_render.py`) | Per-line; legacy | n/a | n/a | **DELETE.** Same. |

---

## 4. The three real findings

### Finding A — MusicGenTheme has zero L3 awareness (HIGH)

`musicgen_theme.py:48–74` hardcodes three "1940s old time radio" cue defaults that are used whenever the Director's `music_plan` is missing the cue (which is most of the time on the v2 path — Director is being pushed toward "unwired" per Pattern 5 in ROADMAP). Even when the Director's plan *is* present, it was generated from `script_text` without seeing the L3 ledger's `meta.gen_params_initial.style`, `meta.news.script_brief`, or the news-derived mood signal.

This is the only voice-side consumer with **zero** read path into the L3 ledger.

ROADMAP D2 already tracks this as a deferred follow-up, with xfail-strict canary `test_musicgen_does_not_default_to_period_cues` armed in `tests/test_downstream_prompt_contract.py`. The fix path is unblocked (narrative plane is stable as of the news_interpreter sprint).

### Finding B — BatchKokoroGenerator is stuck on the pre-L3 parser-list shape (HIGH if wired)

`batch_kokoro_generator.py:116` does `script = json.loads(script_json)`, no `_otr_ledger_consumers` import, and iterates expecting `{type, character_name, voice_traits, line}` per item. Under the current `OTR_LedgerFreezeCascade.script_json` payload (a ledger dict, not a list), this would:
- Iterate the dict's top-level keys (`"cast"`, `"lines"`, `"meta"`, `"schema_version"`…) instead of lines, or
- Crash on `item.get("type") == "dialogue"` because `item` is a string.

The node is **not wired** in `otr_scifi_16gb_full.json`, so it is not failing today. It is also not behind a `pytest.importorskip` or deprecation flag, and it remains registered in `__init__.py:120` (`"OTR_BatchKokoroGenerator"`). Any user-edited workflow that uses Kokoro for character dialogue (the 4GB code path) wires straight into a broken node.

This was missed by the "7 of 7 consumers shipped green" sprint because the 7-consumer list (per `docs/2026-05-09-ledger-consumer-rewrite-patterns.md` and ROADMAP) did **not** include BatchKokoroGenerator. The Kokoro work in the sprint was the **announcer** path (`KokoroAnnouncer`), not the character-dialogue alternative (`BatchKokoroGenerator`).

### Finding C — LLMDirector is still wired and emits a parallel, derivative production_plan (MEDIUM)

The Director (`story_orchestrator.LLMDirector`) takes `script_text` as input and produces a 4-output plan (`production_plan_json`, `voice_map_json`, `sfx_plan_json`, `music_plan_json`). In v2 this is now redundant or worse:

- **`voice_assignments`** — Pattern 3 in the consumer-rewrite doc says: prefer `cast.voice_preset` (v2 cast contract), fall back to Director's `voice_map` only when missing. The cast contract is now load-bearing (`led.data["cast"]` is the canonical source per BUG_LOG `script-writing-architecture` entry §10). So Director's voice_assignments is either ignored (good) or silently overrides cast in edge cases (bad). Latent drift.
- **`sfx_plan`** — The L3 ledger has authoritative `sfx_cue` per line (sfx are first-class lines in v2). `OTR_BatchAudioGenGenerator` reads `line["text"]` from the ledger, not Director's sfx_plan. Director's sfx_plan is dead weight here.
- **`music_plan`** — This is the **only** music input. MusicGen reads it via `production_plan_json` and nowhere else. Finding A above.

The Director also re-runs an LLM call (~2000 max_new_tokens at temperature 0.4) every workflow execution — VRAM + wallclock cost for a derivative output the consumers mostly ignore.

---

## 5. Drift signals to watch

1. `cast.voice_preset` empty → silent fallthrough to `_voice_preset_for_character(name, Director_voice_map, traits)`. Cast contract is supposed to guarantee voice_preset is populated; if a cast row ever ships with empty/None voice_preset, the Director path takes over. No telemetry today.
2. `meta.news.script_brief` present but MusicGen never sees it. The opening/closing cues will always reflect Director's `music_plan` (or `CUE_DEFAULTS` if Director's plan is missing/malformed).
3. `meta.gen_params_initial.style` present (e.g., `deep_space_distress_call`) but MusicGen still produces "warm brass fanfare, upright bass" because of `CUE_DEFAULTS`.
4. Any user wiring `OTR_BatchKokoroGenerator` from `FreezeCascade.script_json` → undefined behavior (crash or empty results, depending on which `.get` call raises first).

---

## 6. Concrete wiring fix list

### Workflow JSON (`workflows/otr_scifi_16gb_full.json`) — single canonical surface, rewritten clean

The workflow JSON is the single canonical surface. There is no parallel legacy-workflow path to preserve, and no soft "transition" of leaving stale wires in place. Every change below lands in one commit.

| Change | Rationale |
|---|---|
| **Delete `OTR_LLMDirector(2)` node entirely** | Standing Directive — delete, don't deprecate. Removes the ~2000-token LLM call + VRAM cost from every run. |
| **Cut link 4** (Director → SceneSequencer.production_plan_json) | Sequencer never reads plan content. |
| **Cut link 13** (Director → Bark.production_plan_json) | Bark's Director fallback is removed; cast.voice_preset is sole source. |
| **Cut link 21** (Director → MusicGen.production_plan_json) | MusicGen is rewritten to read the ledger directly. Replaced by a new wire from `FreezeCascade.script_json → MusicGen.script_json`. |
| **Cut link 26** (Director → AudioGen.production_plan_json) | AudioGen never reads plan content. |
| **Add new link** (FreezeCascade.script_json → MusicGen.script_json) | MusicGen's only data input becomes the L3 ledger, matching every other voice consumer. |
| **No "legacy_archive" copy of the old JSON** | Per the no-back-compat rule. Old saved JSONs are expected to be rewritten by hand against the new graph. |

### Node code — delete the old surfaces, do not deprecate

| File | Change |
|---|---|
| `nodes/musicgen_theme.py` | **Rewrite to ledger-only.** Drop the `production_plan_json` socket entirely; add a required `script_json: STRING` input. Read `meta.gen_params_initial.style` (slug) and `meta.news.script_brief` (when present) and synthesize the three cue prompts from those two fields. **Delete `CUE_DEFAULTS`** at lines 48-74 — no hardcoded "1940s old time radio" / "warm brass fanfare" defaults survive. Empty/missing style is a hard fail, not a degrade. Style→prompt strategy (deterministic table vs. tiny LLM call) is round-robin question §10.1. Flip the armed `test_musicgen_does_not_default_to_period_cues` canary in lockstep. |
| `nodes/batch_kokoro_generator.py` | **Either delete the file + drop the registration, OR rewrite from scratch.** If Kokoro-for-character is no longer a target workflow (Bark is the production path on the 5080), delete it. If it stays, rewrite mirroring `batch_bark_generator.py` exactly: ledger consumers + cast.voice_preset + per-line `patch_line_fields`. **No legacy `_voice_preset_for_character`, no `production_plan_json` socket, no `voice_assignments` lookup.** Round-robin question §10.4 confirms which. |
| `nodes/batch_bark_generator.py` | **Delete the Director fallback.** Replace lines 519-524 (`preset_from_cast` / Director `voice_map` branch) with a hard `raise ValueError("cast.voice_preset missing or malformed for {name} -- writer cast contract violation")`. **Delete `_voice_preset_for_character` (lines 147-180), delete the `production_plan_json` socket, delete the `production_plan_or_empty` call.** Cast contract is the only voice source. |
| `nodes/scene_sequencer.py` | **Delete the `production_plan_json` socket + `production_plan_or_empty` call from `SceneSequencer`.** Plan content was already unused; the socket itself is the legacy. EpisodeAssembler is already ledger-only and stays. |
| `nodes/batch_audiogen_generator.py` | **Delete the `production_plan_json` socket + `production_plan_or_empty` call.** Same reason — unused for SFX content. |
| `nodes/batch_procedural_sfx.py` | **Delete the `production_plan_json` socket + `production_plan_or_empty` call.** Same reason. |
| `nodes/story_orchestrator.py::LLMDirector` | **Delete the entire class.** Also delete: `DIRECTOR_PROMPT`, `KOKORO_VOICE_RULES`, `BARK_VOICE_RULES`, `DirectorJSONParseError`, `_build_director_json_repair_prompt`, `_validate_director_plan`, `_randomize_character_names`, `_extract_json` if exclusive to the director, `_strip_json_comments` if exclusive — anything used only by `LLMDirector`. Grep for `LLMDirector` / `_director_` / `production_plan` across `nodes/` after the delete; expected zero hits outside MusicGen's transient socket if rewritten in a separate commit. |
| `nodes/bark_tts.py` | **Delete the file** + drop registration in `__init__.py`. |
| `nodes/sfx_generator.py` | **Delete the file** + drop registration. |
| `nodes/voice_render.py` | **Delete the file** + drop registration. |
| `__init__.py` | Drop registrations for `OTR_LLMDirector`, `OTR_BarkTTS`, `OTR_SFXGenerator`, `OTR_VoiceRender`, and `OTR_BatchKokoroGenerator` (unless rewritten). Drop the `NodeName` legacy-alias mirror loop (lines 219-229) if no remaining workflow relies on the bare-name registration — confirm via grep. |

### Tests — add and flip; delete obsolete

| Test | Action |
|---|---|
| `tests/test_downstream_prompt_contract.py::test_musicgen_does_not_default_to_period_cues` | **Flip xfail-strict → PASS** in the same commit that lands MusicGen's ledger-aware rewrite. Per the canary mechanic, leaving the marker would XPASS-fail under strict. |
| `tests/test_batch_kokoro_generator.py` | **Either delete the file (if BatchKokoroGenerator is deleted), or rewrite it** as a 4-case mirror of `test_batch_bark_generator.py`. |
| `tests/test_workflow_json_guardrails.py` | Add three asserts: (a) `OTR_LLMDirector` not present in `otr_scifi_16gb_full.json`; (b) no `production_plan_json` wires in the workflow; (c) `OTR_MusicGenTheme` has a `script_json` wire from `OTR_LedgerFreezeCascade.script_json`. |
| `tests/test_bark_cast_contract.py` (new or extended) | Add a case asserting Bark **raises ValueError** when a line's `cast.voice_preset` is empty — proves the Director fallback is gone. |
| Any test pinning the old `OTR_LLMDirector` / `OTR_BarkTTS` / `OTR_SFXGenerator` / `OTR_VoiceRender` / pre-L3 `OTR_BatchKokoroGenerator` contracts | **Delete.** Same rule as the upstream legacy-prune §32–§35 in ROADMAP. |
| `tests/test_otr_ledger_consumers.py` | Add coverage for any new helper used by MusicGen (e.g. style + news.script_brief accessor) if one is introduced. |

### Tests to add or flip

| Test | Action |
|---|---|
| `tests/test_downstream_prompt_contract.py::test_musicgen_does_not_default_to_period_cues` | Flip xfail-strict → PASS when MusicGen ledger-aware lands. |
| `tests/test_batch_kokoro_generator.py` (new) | Mirror `test_batch_bark_generator.py`: 4 cases (clean L3 read, role-filter, legacy-list raises ValueError, line_id write-back). |
| `tests/test_workflow_json_guardrails.py` | Add guardrail asserting `OTR_MusicGenTheme` has `script_json` wired (link from FreezeCascade) when MusicGen ledger-aware lands. Add guardrail asserting `OTR_LLMDirector` is **not** present in `otr_scifi_16gb_full.json` after the cleanup commit. |
| `tests/test_bark_cast_contract.py` | Add a case asserting Bark **raises** on empty cast.voice_preset (instead of silently falling back to Director). |

---

## 7. Risk ranking — all items in scope under no-back-compat directive

Under §0 every item below is mandatory for the cleanbreak commit. Ranking reflects implementation order and surface area, not optionality.

| Rank | Item | Why | Effort |
|---|---|---|---|
| 1 | MusicGenTheme ledger-aware rewrite (delete `CUE_DEFAULTS`, delete `production_plan_json` socket, add `script_json` socket, derive cues from `meta.gen_params_initial.style` + `meta.news.script_brief`) | Largest behavior change. User-visible audio mood mismatch today. Round-robin question §10.1 picks the style→prompt strategy. | Medium — node rewrite + workflow re-wire + canary flip. |
| 2 | Delete `OTR_LLMDirector` class + drop registration + cut all four workflow links | Removes the parallel derivative-plan surface entirely. Drops ~2000-token LLM call + VRAM cost from every run. | Low to medium — class delete + grep-pass for any leftover `production_plan` reads. |
| 3 | Delete Bark Director-fallback + delete `production_plan_json` socket from Bark, Sequencer, AudioGen, ProcSFX | Standing Directive — every secondary path goes. Hard-fail on empty `cast.voice_preset`. | Low — surgical deletions in 4 files. |
| 4 | Delete `OTR_BatchKokoroGenerator` OR rewrite it ledger-native (no middle ground) | Pre-L3 parser-list reader is a footgun and a back-compat artifact. Round-robin question §10.4 picks delete vs. rewrite. | Trivial (delete) or low (rewrite). |
| 5 | Delete `nodes/bark_tts.py`, `nodes/sfx_generator.py`, `nodes/voice_render.py` + drop registrations | Unused legacy nodes. Footgun cleanup. | Trivial. |
| 6 | Workflow JSON re-wire: cut Director, cut all `production_plan_json` wires, add FreezeCascade→MusicGen `script_json` wire | Single canonical surface, rewritten clean. No `legacy_archive` copy. | Trivial JSON edit + load-test in ComfyUI Desktop. |
| 7 | Test deletions + canary flip + new guardrail asserts | Lockstep with the deletions. | Low. |
| 8 | Drop `__init__.py` `NodeName` bare-name legacy-alias mirror loop (lines 219-229) if no longer needed | Last back-compat residue in the registration layer. | Trivial — confirm via grep then delete. |

---

## 8. Source-of-truth recommendation (single sentence per field)

| Field | Single source of truth | Today's status |
|---|---|---|
| Per-line `text` | `ledger.lines[i].text` | ✅ all voice nodes (except BatchKokoroGenerator) read from here |
| Per-line `speaker_role` (`character` / `announcer` / `sfx` / `music_inter`) | `ledger.lines[i].speaker_role` (via `iter_lines(roles=...)`) | ✅ enforced by `_otr_ledger_consumers.iter_lines` |
| Per-line `char_id` | `ledger.lines[i].char_id`, joined to `ledger.cast` via `speaker_name(led, line)` | ✅ correct in all migrated consumers |
| Per-character `voice_preset` | `ledger.cast[char].voice_preset` (cast contract) | ⚠️ Bark falls back to Director.voice_map on miss → **fix:** delete the fallback, hard-fail on miss |
| Episode `style` | `ledger.meta.gen_params_initial.style` (snake_case slug, e.g. `closed_room_suspense`) | ❌ MusicGen blind → **fix:** MusicGen rewrite |
| Episode `news_brief` (mood signal for music) | `ledger.meta.news.script_brief` | ❌ MusicGen blind → **fix:** MusicGen rewrite |
| SFX cue content | `ledger.lines[i].text` for `sfx` role | ✅ AudioGen + ProcSFX |
| Music cue content | `ledger.meta.gen_params_initial.style` + `ledger.meta.news.script_brief` | ❌ today: Director `music_plan` + hardcoded "1940s" `CUE_DEFAULTS` → **fix:** MusicGen rewrite, delete `CUE_DEFAULTS`, delete Director |
| `production_plan_json` wire | **DELETED.** No node retains the socket post-cleanbreak. | wired into Bark / Sequencer / AudioGen / MusicGen today → **fix:** cut every wire, delete every socket |

---

## 9. Out-of-scope for this audit

- HuMo / LTX / FLUX / VideoComposite path. Per ROADMAP "Video pipeline recon (read-only confirmation, post-consumer #7)" — those nodes read ledger from disk; confirmed in consumer-7 audit (SignalLostVideoRenderer / `video_engine.py`). Not re-audited here.
- The `_voice_backends/` abstraction (`bark.py` / `kokoro.py` / `_protocol.py` / `__init__.py`). These are backend drivers, not data consumers. They are downstream of `_otr_voice_resolver.parse_voice_spec()` which is itself called only from `cast.voice_preset` (Phase 0+ Cast Contract §3 + Voice Backend Abstraction §3).
- The Phase 4 / Phase 5 / Phase 6 standalone LFC nodes. Those operate on the ledger pre-freeze; they are not voice consumers.
- The `OTR_LedgerScriptReviewer` re-export shim deletion (Standing Directive). That's commit 12.3 follow-up, not this audit.

---

## 10. Questions for the round-robin

The no-back-compat directive (§0) answers the previous "delete vs. deprecate" questions in advance. Remaining open questions are design choices inside the deletion, not whether to delete.

1. **MusicGen style→prompt strategy.** Deterministic style-slug → cue-prompt mapping table (no LLM, fast, predictable, easy to test) vs. a tiny LLM call that takes `style + news.script_brief` and emits 3 cue prompts (richer per-episode variation, costs ~200 tokens × 3 per run, adds one more LLM-agnostic control-plane call). The narrative plane picked the LLM-call approach in the news_interpreter sprint. Music is more constrained — table might be sufficient. Round-robin: which?
2. **Cast.voice_preset invariant — where does the hard-fail land?** Bark's current Director-voice_map fallback (slated for deletion) papers over any case where cast.voice_preset is empty/None. With the fallback gone, an empty preset crashes the audio render mid-line. Where is the right place to detect and fail loud? Options: (a) at the writer's cast-lock exit (earliest), (b) at the freeze-cascade Phase 0 audit (mid), (c) at Bark's iter_lines entry (latest, what the current §6 fix proposes). Round-robin: which gate?
3. **`OTR_BatchKokoroGenerator` — delete or rewrite?** The character-Kokoro path was the 4GB-card option ("Obsidian"). The 5080 production path is Bark. Is Kokoro-for-character still a target on any active or near-future workflow, or can the file + class + tests be deleted outright? Round-robin: keep or cut?
4. **Director-deletion sweep — anything outside `nodes/` reads its outputs?** Before the delete commit, what's the right grep set? Candidates: `scripts/*.py`, `tools/*.py`, `viewer/*.py`, `web/*.js`, any saved workflow JSON under `workflows/`, the survival-guide repo. Round-robin: is the grep set complete, or is there a forgotten reader?

---

## 11. Verdict

The post-clean-break L3 ledger **is** the de-facto single source of truth for text + speaker_role + char_id on all five migrated voice consumers (Bark, KokoroAnnouncer, AudioGen, ProcSFX, SceneSequencer/EpisodeAssembler/AudioEnhance). The reads are clean. The legacy **surfaces around those reads** — `production_plan_json` sockets, Director-derived fallbacks, hardcoded era defaults, pre-L3 parser-list readers, unused legacy single-line nodes — are what's left to delete.

Per the §0 directive, the cleanbreak commit deletes every one of those surfaces in a single auditable pass:

1. **MusicGenTheme** — rewrite ledger-native, delete `CUE_DEFAULTS`, delete `production_plan_json` socket. The only remaining design decision (§10.1) is style→prompt strategy.
2. **`OTR_LLMDirector`** — delete the class. Cut every workflow link. Drop every fallback that consumed its outputs.
3. **Bark Director-fallback + `production_plan_json` sockets on Bark / Sequencer / AudioGen / ProcSFX** — delete. Cast.voice_preset is the sole voice source; empty is a hard fail (§10.2 decides where the gate sits).
4. **`OTR_BatchKokoroGenerator`** — delete or rewrite, §10.3 decides. No middle ground.
5. **`OTR_BarkTTS`, `OTR_SFXGenerator`, `OTR_VoiceRender`** — delete files and registrations.
6. **Workflow JSON** — rewritten clean. No `legacy_archive`. Saved workflows referencing deleted nodes are expected to be redrawn.
7. **Tests** — canary flip, guardrails added, every test pinning deleted contracts removed.

The upstream is clean. The downstream finishes clean in one commit.

No code edits in this session. Round-robin first, then sprint.
