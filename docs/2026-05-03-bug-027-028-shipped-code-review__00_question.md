# Question -- 2026-05-03

# Code review: BUG-LOCAL-027 + BUG-LOCAL-028 fixes shipped 2026-05-03 EVENING

## Project context

ComfyUI custom-node radio-drama generator (OTR "SIGNAL LOST"). Windows / RTX 5080 Laptop / 16 GB VRAM. 100% local. Pipeline: LLM script writer → critique-and-revise pass → audio cascade (Bark for character dialogue, Kokoro for ANNOUNCER, MusicGen, AudioGen) → FLUX env stills → HuMo lipsync → LTX motion → VideoComposite → RTX upscale.

Two soak-revealed bugs were fixed in commit `f1467a2` (with read-side coordination in batch_humo_render and batch_ltx_render). Round-robin consult was SKIPPED at the user's direct directive ("yes ofrget rop8u7hnd robins just fix fix fix"). User now wants a retrospective consult for peace of mind — NO new fixes from you, just a verdict on whether the shipped fixes are sound and what (if anything) was missed.

Tests at the time of commit: 23 new tests + 155 cumulative regression all green; Bug Bible 24 passed / 1 skipped / 1 xfailed in 1.24s. Round-robin questions follow.

---

## BUG-LOCAL-027: critique/revision pass strips all CHARACTER dialogue

**Symptom (3 separate runs over 2 hours):** ScriptWriter draft has 18 healthy `[N] CHARNAME: text` dialogue lines; critique-and-revise pass returns a "revised" script containing ONLY `=== SCENE N ===`, `ENV:`, `SFX #N:` lines (zero character dialogue). Acceptance gate (similarity ratio + length ratio) accepts it because surface metrics pass. Downstream Bark TTS finds 0 character dialogue lines; final audio has only narration + SFX.

**Root cause analysis (confirmed via source dive):**
1. **Parser blindness.** `_count_character_lines` regex was `r'^\s*\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'` — required line to START with optional whitespace + uppercase name. The writer's actual output uses `[12] FLETCHER WELLS: text` numbered-bracket prefix, so the regex never matched and returned `{}` for both draft and revised. The per-character preservation gate iterated `draft_char_counts` (empty dict) → no-op → revision accepted regardless.
2. **Gate too narrow.** Even with parser working, the per-character check only catches "FLETCHER dropped from 8 to 1." If the revision wipes ALL characters at once, no individual character drops below the floor (they all dropped from N to 0, but the loop doesn't compare totals).
3. **Secondary contributor:** revision pass uses `temperature` from caller — for "maximum chaos" creativity = 0.95. High temp + critique demanding "fix every flagged problem" can push the model into pure-prose rewriting where it drops dialogue.

**Fix (3-part) — DIFF SUMMARY:**

```python
# nodes/story_orchestrator.py:6916 — Part 1 (regex)
# OLD:
pattern = r'^\s*\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'
# NEW:
pattern = r'^\s*(?:\[\d+\]\s+)?\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'
# Added optional non-capturing group `(?:\[\d+\]\s+)?` to accept [N] prefix.

# nodes/story_orchestrator.py:6924 — Part 1b (structural-token exclude tightening)
# OLD:
if char_name not in _struct_exclude:
    character_counts[char_name] = ...
# NEW:
first_word = char_name.split()[0] if char_name else ""
if char_name not in _struct_exclude and first_word not in _struct_exclude:
    character_counts[char_name] = ...
# So multi-word headers like "ACT 2:" no longer slip through (first_word = "ACT" is in exclude set).

# nodes/story_orchestrator.py — Part 2 (total-collapse hard gate, after the per-character loop)
import math as _math
draft_total = sum(draft_char_counts.values())
revised_total = sum(revised_char_counts.values())
if draft_total >= 3:
    min_revised = max(1, _math.ceil(draft_total * 0.5))
    if revised_total < min_revised:
        log.warning("[Critique] Total character-line count collapsed from %d to %d "
                    "(minimum %d, threshold=50%% of draft). Revision appears to be "
                    "SCENE/ENV/SFX-only — keeping original draft.",
                    draft_total, revised_total, min_revised)
        _runtime_log(f"CRITIQUE: CRITIQUE_REJECTED - total character lines collapsed "
                     f"from {draft_total} to {revised_total} (min={min_revised}, threshold=50%%)")
        return draft_text

# nodes/story_orchestrator.py — Part 3 (revision prompt clause, prepended to the existing RULES block)
# Added these lines BEFORE the existing RULES list:
ABSOLUTE REQUIREMENT — DIALOGUE MUST SURVIVE THE REVISION:
The revised script MUST contain CHARACTER dialogue lines. Producing a script with only
SCENE/ENV/SFX/MUSIC scaffolding and zero spoken character lines is a TOTAL FAILURE — the
radio drama becomes silent narration. Every CHARACTER speaker present in the draft MUST
appear in the revision. You may rewrite their lines for sharper dialogue, emotional
grounding, or pacing — but you may NEVER delete a character's voice entirely. If you
find yourself writing only ENV: and SFX: tags with no CHARACTER: lines, STOP and
re-include the dialogue.

# Also added to the RULES block:
# "CRITICAL: Every spoken line MUST use the format 'CHARACTER_NAME: dialogue text'... 
#  The optional line-number prefix '[N]' from the draft (e.g. '[12] FLETCHER WELLS: ...')
#  may be kept or omitted — both formats parse correctly."
```

**Tests added** (`tests/test_critique_dialogue_preservation.py`, 14 passing):
- parser handles bare `CHARNAME:` and `[N] CHARNAME:` and mixed
- structural tokens (SCENE/ACT/MUSIC/etc) excluded by exact-match AND first-word-match
- ANNOUNCER counted as character
- gate REJECTS total dialogue wipe (the actual recurring failure case)
- gate REJECTS announcer-only revision
- gate ACCEPTS minor dialogue trim (83% retention)
- gate ACCEPTS at exactly 50% threshold
- gate SKIPS short drafts (`< 3` lines)
- prompt has no unescaped `{}` braces (BUG-026 footgun gate)

---

## BUG-LOCAL-028: FLUX env stills + radio bookend save to legacy flat dirs

**Symptom (every soak since Phase B per-episode reorg on 2026-05-02):** Per-episode workspace at `output/otr/episodes/<ep>/` has `audio/` `videos/` `composited/` correctly populated. But `stills/` and `portraits/` subdirs DO NOT EXIST. FLUX env stills end up in flat `output/otr/stills/` (213 PNGs accumulated since 4/26 with shared global counter); radio bookend ends up in `output/otr/_legacy_stills/`. VideoComposite + BatchHumoRender + BatchLTXRender look in per-episode dirs and find nothing, contributing to "black video" symptom.

**Root cause (confirmed via source dive):** Two unrelated sites had legacy-path defaults:
- `visual/batch_flux_render.py:833` — `_OTRP.otr_stills_dir()` called with no `episode_id`. The helper at `nodes/_otr_paths.py:208-218` returns `output/otr/_legacy_stills/` when called without an episode_id.
- `workflows/otr_scifi_16gb_full.json` node id 25 — stock ComfyUI `SaveImage` with hardcoded `filename_prefix="otr/stills/full_env"`. ComfyUI writes to `output/<filename_prefix>_<auto_counter>_.png`. Path doesn't change per-episode because the widget is static.

**Fix (4 sites — write + read alignment) — DIFF SUMMARY:**

```python
# Site 1 (writer, radio bookend): visual/batch_flux_render.py
# OLD:
stills_dir = _OTRP.otr_stills_dir()
# NEW:
stills_dir = _OTRP.otr_stills_dir(episode_id)
# `episode_id` was already in scope (resolved from in-flight ledger singleton at line 768/772
# via the same Phase G discovery path used by BUG-LOCAL-021).

# Site 2 (writer, env stills): NEW node nodes/otr_save_to_episode_workspace.py + register in __init__.py
# Replaces stock SaveImage. Reads in_flight_ledger_path() singleton at runtime to derive
# episode_id, routes images to otr_stills_dir(ep) or otr_portraits_dir(ep) based on
# `role_kind` widget ("stills" | "portraits"). Falls back to legacy dirs (preserving
# existing behavior) if no singleton is available. Never raises — save failure logs and
# silently skips. Workflow JSON node 25 retyped from SaveImage to OTR_SaveToEpisodeWorkspace.

class SaveToEpisodeWorkspace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "role_kind": (["stills", "portraits"], {"default": "stills"}),
            "filename_pattern": ("STRING", {"default": "full_env"}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"

    def save(self, images, role_kind="stills", filename_pattern="full_env", **_):
        episode_id = _resolve_episode_id()  # in-flight singleton, None if absent
        target_dir = _resolve_target_dir(role_kind, episode_id)  # per-ep or legacy
        target_dir.mkdir(parents=True, exist_ok=True)
        next_idx = _next_index(target_dir, filename_pattern)  # per-episode counter starts at 1
        for img in images:
            pil_img = _tensor_to_pil(img)
            pil_img.save(target_dir / f"{filename_pattern}_{next_idx:05d}_.png")
            next_idx += 1
        return {"ui": {"images": [...]}}

# Site 3 (reader, BatchHumoRender env-still binding): nodes/batch_humo_render.py
# OLD (in _resolve_cast_stills_from_ledger and _find_portrait):
for pattern in ("otr/stills/full_env_*.png", "otr_stills/full_env_*.png"):
    for p in portraits_dir.glob(pattern): ...
# NEW:
for pattern in ("otr/episodes/*/stills/full_env_*.png",
                "otr/stills/full_env_*.png",
                "otr_stills/full_env_*.png"):
    for p in portraits_dir.glob(pattern): ...
# Added per-episode glob pattern alongside legacy patterns. Mtime-based freshness filter
# (fresh_floor = ledger_mtime - 60s) in same function still enforces episode-correctness,
# so cross-episode pollution is mathematically impossible.

# Site 4 (reader, BatchLTXRender radio bookend): nodes/batch_ltx_render.py
# OLD:
fs_path = otr_stills_dir() / f"radio_bookend_{eid}.png"
# NEW:
fs_path = otr_stills_dir(eid) / f"radio_bookend_{eid}.png"
```

**Tests added** (`tests/test_save_to_episode_workspace.py`, 8 passing):
- INPUT_TYPES shape correct
- OUTPUT_NODE = True (so ComfyUI runs it every queue)
- with active singleton → resolves to per-episode dir
- role_kind="portraits" → routes to portraits/ subdir (not stills/)
- no singleton → falls back to legacy `_legacy_stills/`
- per-episode counter starts at 1 (not shared with global)
- save NEVER raises on mkdir failure
- node registered in NODE_CLASS_MAPPINGS

---

## Asks (please answer all)

1. **Are the BUG-027 fixes structurally sound?** Specifically:
   - Is the regex change `(?:\[\d+\]\s+)?` correct? Any edge cases where it would FAIL to match the writer's output, or FALSELY match non-dialogue lines?
   - Is the 0.5 threshold for total-collapse correct, or does it need adjustment? What about edge cases like "draft has 4 lines, revision has 2" — that's exactly at threshold, should it be > or >= ?
   - Is the prompt addition safe (no `.format()` footgun like BUG-026)? Does the language risk over-correcting (model now ALWAYS keeps dialogue even when revision should legitimately reduce it)?
   - Should the per-character `min_line_count_per_character` floor (default=2) be tightened or loosened in light of the new total-collapse gate?

2. **Are the BUG-028 fixes structurally sound?** Specifically:
   - Is the new `OTR_SaveToEpisodeWorkspace` node API correct (INPUT_TYPES + OUTPUT_NODE + RETURN_TYPES)? Any ComfyUI-specific gotchas?
   - The fall-back-to-legacy behavior when no singleton is active — is this safe, or does it risk the same wrong-episode pollution that BUG-LOCAL-014/021 fixed for other paths?
   - The added per-episode glob `otr/episodes/*/stills/full_env_*.png` in batch_humo_render — does the wildcard `*` create any cross-episode leak risk that the mtime filter wouldn't catch?
   - Is the per-episode counter (starts at 1 per ep_dir) correct, or should it be globally monotonic for some reason I missed?

3. **What did I miss?**
   - Other read sites that might still expect legacy paths? (I checked video_composite.py — already correct. Checked batch_humo_render.py and batch_ltx_render.py — fixed.)
   - Edge cases where the fixes interact badly with each other?
   - Failure modes specific to ComfyUI's hot-reload or sys.modules caching?
   - Race conditions in the in_flight_ledger_path() singleton lookup?

4. **Per-bug "% chance fix is needed" estimate:**
   - For each of the 5 fix elements (BUG-027 part 1 regex, part 2 gate, part 3 prompt; BUG-028 site 1 writer, site 2-4 read alignment), give a percentage estimate of how likely YOU think a follow-up fix will be needed in the next 2 weeks of soak runs (0% = bulletproof, 100% = will definitely break in production).
   - Give your reasoning briefly.

5. **Skipped round-robin context:** This consult was deferred until AFTER the fixes shipped per direct user override. Do you see any risk that having shipped first + consulted second creates blind spots a pre-implementation consult would have caught?

Audio C7 byte-identity must hold. VRAM ceiling 14.5 GB on audio path / 15.5 GB on video path. Round-robin context: ChatGPT, Gemini, NVIDIA Nemotron round-robin. Synthesis goes to Claude (the original author of these fixes) for final decision.
