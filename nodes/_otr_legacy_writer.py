"""nodes/_otr_legacy_writer.py

Legacy LLMScriptWriter (v1.x) extracted from story_orchestrator.py.

Behavior is BYTE-IDENTICAL to the pre-extraction class with ONE targeted
deviation: the _CURRENT_LLM_MODEL write-through. The class body's
`global _CURRENT_LLM_MODEL; _CURRENT_LLM_MODEL = model_id` was replaced
with `_so._CURRENT_LLM_MODEL = model_id` so the Director's read-back
(via its own `global` in story_orchestrator.LLMDirector.direct()) picks
up the writer's chosen model. Without this fix the chains drift to
different module namespaces. Documented inline at the call site.

The contract (INPUT_TYPES, RETURN_TYPES, RETURN_NAMES, FUNCTION,
CATEGORY) is preserved exactly so saved workflow JSONs binding to
OTR_LLMScriptWriter keep working without modification.

This module remains the production default for OTR_LLMScriptWriter. The
v2.0 path (OTR_LedgerScriptWriter) is registered as a separate explicit
node; users opt in by adding it to a workflow.

Public surface:
    LegacyLLMScriptWriter -- the extracted class (renamed for clarity).

The original name LLMScriptWriter is preserved in story_orchestrator.py
as a re-import alias.

Design notes (Phase 3 extraction, 2026-05-10):
    Of the 47 module-level names the class body references, 46 are
    defined BEFORE the original class location (line 4305 pre-extraction)
    in story_orchestrator.py. Those are eager-aliasable from a
    partially-loaded _so module during circular import. Exactly one
    name -- _looks_like_non_character_cast_name -- is defined AFTER
    (originally line 10402); it is wrapped in a late-binding callable
    so the class body can reference it as a bare name without
    triggering an AttributeError during _so's partial-init.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime, timedelta

log = logging.getLogger("OTR")

# Dual-mode story_orchestrator import. ComfyUI loads this module as
# nodes._otr_legacy_writer (package-relative import works). Test gates
# and standalone scripts insert the nodes/ directory into sys.path and
# import by bare name (relative import fails). Both paths must work.
try:
    from . import story_orchestrator as _so
except ImportError:
    import sys as _sys
    import pathlib as _pathlib
    _NODES_DIR = _pathlib.Path(__file__).resolve().parent
    _COMFY_ROOT = _NODES_DIR.parent.parent.parent
    for _p in (str(_NODES_DIR), str(_COMFY_ROOT)):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    import story_orchestrator as _so  # type: ignore

ProjectState = _so.ProjectState
vram_snapshot = _so.vram_snapshot
vram_reset_peak = _so.vram_reset_peak
_LEMMY_RNG = _so._LEMMY_RNG
_LEMMY_HISTORY = _so._LEMMY_HISTORY
SCAFFOLDING_PREAMBLE = _so.SCAFFOLDING_PREAMBLE
SCRIPT_SYSTEM_PROMPT = _so.SCRIPT_SYSTEM_PROMPT
_DIALOGUE_FALSE_POSITIVES = _so._DIALOGUE_FALSE_POSITIVES
_FIRST_NAMES = _so._FIRST_NAMES
_LAST_NAMES = _so._LAST_NAMES
_RE_TITLE_LINE = _so._RE_TITLE_LINE
_STUCK_TITLE_DEFAULTS = _so._STUCK_TITLE_DEFAULTS
_TOKEN_RATIO_ACT_CHUNK = _so._TOKEN_RATIO_ACT_CHUNK
_TOKEN_RATIO_ACT_OBSIDIAN = _so._TOKEN_RATIO_ACT_OBSIDIAN
_TOKEN_RATIO_DIALOGUE = _so._TOKEN_RATIO_DIALOGUE
_TOKEN_RATIO_MIXED = _so._TOKEN_RATIO_MIXED
_VOICE_PROFILES = _so._VOICE_PROFILES
_VOICE_TRAITS = _so._VOICE_TRAITS
_check_parse_ok = _so._check_parse_ok
_check_voice_consistency = _so._check_voice_consistency
_cleanup_character_names = _so._cleanup_character_names
_content_filter = _so._content_filter
_derive_title_from_script_lines = _so._derive_title_from_script_lines
_extract_all_dialogue = _so._extract_all_dialogue
_extract_title_from_script_text = _so._extract_title_from_script_text
_fetch_science_news = _so._fetch_science_news
_flush_vram_keep_llm = _so._flush_vram_keep_llm
_generate_ltx_style_brief = _so._generate_ltx_style_brief
_generate_with_llm = _so._generate_with_llm
_inject_scene_transitions = _so._inject_scene_transitions
_is_inline_narration = _so._is_inline_narration
_load_canon_for_writer = _so._load_canon_for_writer
_log_scene_checkpoint = _so._log_scene_checkpoint
_normalize_dialogue_names = _so._normalize_dialogue_names
_run_with_timeout = _so._run_with_timeout
_runtime_log = _so._runtime_log
_tail_at_sentence_boundary = _so._tail_at_sentence_boundary
_truncate_at_sentence_boundary = _so._truncate_at_sentence_boundary
_unload_llm = _so._unload_llm


def _looks_like_non_character_cast_name(*args, **kwargs):
    return _so._looks_like_non_character_cast_name(*args, **kwargs)


# ===========================================================================
# Class body (extracted from story_orchestrator.py during Phase 3 of the
# v2.0 LPL sprint; class declaration line renamed
# LLMScriptWriter -> LegacyLLMScriptWriter; one block (MASTER SWITCH
# INHERITANCE) edited for write-through to _so. All other lines unchanged.)
# ===========================================================================
class LegacyLLMScriptWriter:
    """Fetches real science news, generates a full radio drama script via LLM."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "write_script"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("script_text", "script_json", "news_used", "estimated_minutes")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "episode_title": ("STRING", {
                    "default": "",
                    "tooltip": "Episode title (leave blank for Gemma to generate one; see BUG-LOCAL-035)"
                }),
                "target_words": ("INT", {
                    "default": 700, "min": 30, "max": 10000, "step": 10,
                    "tooltip": "Target spoken dialogue words at ~140 wpm: 30=ultra-smoke (~13s, ~3 lines, BUG-128/129 routing check), 100=smoke test (~45s, ~6 lines), 200=quick (~85s), 350=2.5min, 700=5min, 1400=10min, 2100=15min, 3500=25min. Step-down works directly via this widget. For one-click ultra-smoke pick target_length='30 words (smoke, 1 act)' (also lowers the 18-line floor to 3); for 100-word smoke pick target_length='tiny (smoke, 1 act)'. Both presets override this widget."
                }),
                "num_characters": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Speaking characters (plus announcer). 1=monologue/diary mode (one voice carries the entire episode, ANNOUNCER bookends still apply). Auto-clamped to 4 when target_words <= 700, or 3 when <= 420 (clamps respect user's explicit 1 or 2 -- only bump UP from default)."
                }),
            },
            "optional": {
                # 2026-04-26 PM: Mistral-Nemo restored as default while
                # the rest of the pipeline (HuMo orchestrator, concat,
                # upscale, episodes folder) is being shaken down. Live
                # trials of two different RP fine-tunes (Captain-Eris-Violet
                # and Mag-Mell-R1, both Mistral-Nemo-derivative 12B merges)
                # produced short or empty episodes -- they wrote nice
                # creative draft prose but did not engage with the
                # structured rescue prompts (Open-Close spines, WORD_EXTEND,
                # Announcer bookends). Mistral-Nemo base cleared
                # BUG-061/062/063 format hardening and is the validated
                # path. Captain-Eris-Violet and Mag-Mell remain in the
                # dropdown for users who want creative voice; the planned
                # follow-up is a two-LLM split (creative model writes the
                # script, structured model handles cleanup phases).
                "model_id": (["mistralai/Mistral-Nemo-Instruct-2407",
                              "google/gemma-4-E2B-it",
                              "google/gemma-4-E4B-it",
                              "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
                              "Nitral-AI/Captain-Eris_Violet-V0.420-12B (EXPERIMENTAL)",
                              "inflatebot/MN-12B-Mag-Mell-R1 (EXPERIMENTAL)"], {
                    "default": "mistralai/Mistral-Nemo-Instruct-2407",
                    "tooltip": "Hugging Face model ID for LLM. "
                               "Mistral-Nemo is the production default "
                               "(12B, 4-bit NF4, _cap=6144). It cleared "
                               "BUG-061/062/063 format hardening. "
                               "Gemma 4 E2B (effective 2B) and E4B "
                               "(effective 4B) are Google's edge-targeted "
                               "lightweight models, untested on Blackwell "
                               "yet. Qwen-2.5-14B is alpha. Captain-Eris "
                               "and MN-12B-Mag-Mell-R1 are EXPERIMENTAL "
                               "RP fine-tunes -- they produce vibey draft "
                               "prose but routinely short-output "
                               "structured rescue prompts (BUG-109 retry "
                               "loop cannot rescue them at high "
                               "creativity / short target_length). Pair "
                               "them as the story model with Mistral-Nemo "
                               "as cleanup_model_id, or just stay on "
                               "Mistral-Nemo. Suffix tags are stripped "
                               "before HF lookup."
                }),
                # 2026-04-26 PM two-LLM split (task #56): creative LLM
                # writes the script (draft / revision / arc / spines /
                # autotitle / per-act gen / cast names) while the
                # cleanup LLM handles structured rescue + polish phases
                # (Grammarian, WORD_EXTEND, LLM_RESCUE, ANNOUNCER
                # bookends, FormatNorm, Director plan). Lets users pair
                # an RP fine-tune like Captain-Eris-Violet (rich dialogue
                # voice but fails structured prompts) with Mistral-Nemo
                # base (validated against every format gate). Default
                # "auto" uses the same model for both roles --
                # backward-compatible with every saved workflow.
                "cleanup_model_id": ([
                    "auto (use story model)",
                    "mistralai/Mistral-Nemo-Instruct-2407",
                    "google/gemma-4-E2B-it",
                    "google/gemma-4-E4B-it",
                    "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
                    "Nitral-AI/Captain-Eris_Violet-V0.420-12B (EXPERIMENTAL)",
                    "inflatebot/MN-12B-Mag-Mell-R1 (EXPERIMENTAL)",
                ], {
                    "default": "auto (use story model)",
                    "tooltip": "Optional second LLM for structured rescue "
                               "+ polish phases (Grammarian, WORD_EXTEND, "
                               "LLM_RESCUE, ANNOUNCER bookends, "
                               "FormatNorm, Director). 'auto' uses the "
                               "story model -- backward compat. Pair an "
                               "RP fine-tune as the story model with "
                               "Mistral-Nemo base here for the best of "
                               "both worlds: rich character voice + "
                               "format compliance."
                }),
                "custom_premise": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "(optional) type a custom story premise here -- "
                        "overrides the RSS news fetch"
                    ),
                    "tooltip": (
                        "OPTIONAL premise override.  Leave empty (default) "
                        "to let the RSS news fetcher pick a real-world "
                        "headline as the episode seed.  Type a premise "
                        "here to BYPASS the news pipeline entirely -- "
                        "your text becomes the spine seed and the "
                        "OpenClose 3-spine evaluator is skipped (direct "
                        "path to script writer).\n\n"
                        "Use cases:\n"
                        "  - test a specific story idea ('two scientists "
                        "trapped in a lunar greenhouse during a solar "
                        "storm')\n"
                        "  - reproduce a previous run with controlled "
                        "inputs\n"
                        "  - work offline / skip RSS when news re-rank "
                        "picked a weak headline\n"
                        "  - write a series with continuity ('three "
                        "months after the Vela incident...')\n\n"
                        "Ledger stamps meta.custom_premise_set so "
                        "post-mortem can tell which seed path produced "
                        "the run."
                    ),
                }),
                # news_headlines and temperature removed in v2.0 - both were dead
                # params (news_headlines was never wired to RSS, temperature was
                # overridden by creativity dial). Kept in write_script() signature
                # for backward compat but no longer exposed as widgets.
                "include_act_breaks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Include act breaks with sponsor messages (authentic style)"
                }),
                "self_critique": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Checks & Critiques: Draft -> Critique -> Revise loop "
                        "for higher story quality (adds ~2 extra LLM passes). "
                        "Defaults ON for stable / high-quality scripts. "
                        "Edge-case empty-output crashes are caught by the "
                        "BUG-LOCAL-085 empty-draft guard. Flip OFF for "
                        "faster cheaper runs at the cost of polish."
                    ),
                }),
                "open_close": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Open-Close Expansion: generate 3 competing story "
                        "outlines, evaluator picks the best before full "
                        "script (adds ~4 small LLM passes). Defaults ON "
                        "for stable / high-quality scripts. Cascading "
                        "1-token / empty-output cases are caught by "
                        "BUG-LOCAL-090 Director fallback. Flip OFF for "
                        "faster cheaper runs at the cost of plot variety."
                    ),
                }),
                "target_length": (["30 words (smoke, 1 act)", "tiny (smoke, 1 act)", "short (3 acts)", "medium (5 acts)", "long (7-8 acts)", "epic (10+ acts)"], {
                    "default": "medium (5 acts)",
                    "tooltip": "Act structure preset. '30 words (smoke, 1 act)' = fastest possible end-to-end pipeline check (~13s audio, ~3 dialogue lines, forces target_words=30 + lowers the 18-line floor; pair with num_characters=2). 'tiny (smoke, 1 act)' = 100-word smoke (~45s audio, ~6 HuMo clips). Short=3 acts, Medium=5, Long=7-8, Epic=10+. More acts spread your target_words across more scenes."
                }),
                "style": ([
                    "tense claustrophobic",
                    "space opera epic",
                    "psychological slow-burn",
                    "hard-sci-fi procedural",
                    "noir mystery",
                    "chaotic black-mirror",
                    "cosmic dread",
                    "dystopian unease",
                    "post-apocalyptic decay",
                    "cyberpunk neon-noir",
                ], {
                    "default": "tense claustrophobic",
                    "tooltip": (
                        "Tonal direction PRESET. Pick a preset OR "
                        "leave it on any preset and use style_custom "
                        "below to override with your own free-text "
                        "phrase (style_custom wins when non-empty).\n\n"
                        "The chosen value is interpolated into ~12 "
                        "LLM prompt templates (writer / spine "
                        "evaluator / critic / reviser / arc-enhancer) "
                        "AND used VERBATIM as the FLUX radio still's "
                        "leading aesthetic descriptor (the radio HuMo "
                        "I2V ref for non-dialogue lines).\n\n"
                        "2026-04-30 consolidation: this widget "
                        "replaced genre_flavor + style_variant, which "
                        "were redundant."
                    ),
                }),
                "style_custom": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "OPTIONAL free-text override.  When non-empty, "
                        "this string replaces the style preset above "
                        "for ALL downstream use (LLM prompts AND FLUX "
                        "radio prompt).  Leave empty to use the preset.\n\n"
                        "Be evocative -- the FLUX radio still echoes "
                        "your exact wording.  Example overrides:\n"
                        "  rust-belt cyber-noir\n"
                        "  cosmic dread, fog-bound coast\n"
                        "  1970s soviet brutalism\n"
                        "  neon-drenched broadcast cathedral\n"
                        "Override beats preset for prompt richness."
                    ),
                }),
                "creativity": (["safe & tight", "balanced", "wild & rough", "maximum chaos"], {
                    "default": "balanced",
                    "tooltip": "Creativity dial - overrides temperature/top_p (safe=0.6, balanced=0.85, wild=0.92, chaos=0.95)"
                }),
                "arc_enhancer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Structural coherence pass: rewrites the opening & closing dialogue to ensure a 'seed' in the intro pays off in the finale."
                }),
                "optimization_profile": (["Pro (Ultra Quality)", "Standard", "Obsidian (UNSTABLE/4GB)"], {
                    "default": "Standard",
                    "tooltip": "Master switch for multi-pass generation. Obsidian is for 4GB hardware only; it is unstable and disables all iterative passes."
                }),
                "perfect_run_spacesaver": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When ON: at the end of the workflow, wipe every intermediate "
                        "file under output/otr/episodes/<episode_id>/ -- stills, "
                        "portraits, per-line video pieces, the 832x480 composite "
                        "intermediate, MusicGen wavs, AudioGen wavs, Bark wavs, "
                        "the procgen mp4. KEEP only: the final upscaled mp4 in "
                        "output/otr/obs/<episode_id>.mp4, the production ledger "
                        "(<episode_id>_ledger.json), and the treatment text "
                        "(<episode_id>_treatment.txt). Useful for unattended "
                        "long-running batches where you want one tidy deliverable "
                        "per episode instead of ~1 GB of working files. DEFAULT OFF "
                        "so you can re-upscale, A/B compare, or audit a run later. "
                        "Stamped into ledger.meta.perfect_run_spacesaver and read "
                        "by OTR_RTXUpscale at the end of the workflow."
                    ),
                }),
                # v1.4 Theme C - optional series bible. Socket input only, no widget.
                # BUG-LOCAL-027: project_state MUST remain the last entry in optional.
                # Socket-only inputs at the tail cannot shift widget slots even if the
                # widgets_values mapper regresses. Do not add widget-backed params
                # after this line.
                "project_state": ("PROJECT_STATE", {
                    "tooltip": "Optional: Project State Loader output. When wired, series bible preamble is injected into the script prompt."
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Always re-execute: news changes daily (Section 12)."""
        return time.time()

    def _generate_cast_names_via_llm(self, num_names, style, story_context,
                                     model_id, episode_fingerprint,
                                     from_outline=False):
        """Generate character names that organically fit the story.

        from_outline=True  - story_context is the winning outline. Extract
                             the names Gemma already chose while plotting so
                             names blend naturally with the world and sound.
        from_outline=False - story_context is a news headline hook. Invent
                             names suited to the genre and science theme.

        voice_preset is assigned from the English pool by seeded RNG.
        Returns a list of profile dicts, or None on failure.
        """
        if num_names <= 0:
            return []

        _runtime_log(f"CAST_LLM: {'Extracting' if from_outline else 'Generating'} "
                     f"{num_names} names ({'from outline' if from_outline else 'from context'})")

        if from_outline:
            names_prompt = f"""You are a script supervisor finalizing the cast for a {style.replace('_', ' ')} audio drama.

Below is the WINNING STORY OUTLINE. It already contains character names chosen to fit the world and story.

YOUR TASK: Extract exactly {num_names} character name(s) from this outline. Choose names that:
- Sound crisp and distinct when spoken aloud - easy to tell apart by ear
- Fit the tone and world of this story
- Have no two characters sharing the same last name

OUTLINE:
{story_context[:2000]}

Output ONLY {num_names} line(s) in this exact format, nothing else:
FIRSTNAME LASTNAME: their role or key trait in one short phrase"""
        else:
            names_prompt = f"""You are a casting director for a {style.replace('_', ' ')} audio drama.

Generate exactly {num_names} character name(s) that sound crisp and memorable when spoken aloud.

Science theme (for tonal inspiration only - do NOT write a story):
{story_context[:300]}

RULES:
- FIRST + LAST name only - no titles like "Dr." or "Agent"
- Names must be easy to distinguish from each other by ear in an audio drama
- No two characters share the same last name
- Avoid sci-fi clich-s: Chen, Reyes, Kira, Jake, Marco, Elena, Voss, Hayes
- Mix genders if {num_names} > 1

Output ONLY {num_names} line(s) in this exact format, nothing else:
FIRSTNAME LASTNAME: role or personality in one short phrase"""

        try:
            raw = _run_with_timeout(
                lambda: _generate_with_llm(
                    names_prompt,
                    model_id=model_id,
                    max_new_tokens=num_names * 30 + 20,
                    temperature=0.85,
                ),
                timeout_sec=120,
                phase_label="CastNames",
            )
        except Exception as e:
            log.warning("[CastNames] LLM call failed: %s", e)
            _runtime_log(f"CAST_LLM: failed ({e})")
            return None

        # Parse "FIRSTNAME LASTNAME: description" lines
        profiles = []
        seen_last_names = set()
        rng = random.Random(f"{episode_fingerprint}_voices")
        male_pool   = [vp[0] for vp in _VOICE_PROFILES if vp[1] == "male"]
        female_pool = [vp[0] for vp in _VOICE_PROFILES if vp[1] == "female"]
        rng.shuffle(male_pool)
        rng.shuffle(female_pool)
        m_idx = f_idx = 0

        for line in raw.strip().splitlines():
            line = line.strip()
            # Accept "FIRSTNAME LASTNAME: description" or "- FIRSTNAME LASTNAME: ..."
            line = re.sub(r'^[-*\d.)\s]+', '', line).strip()
            match = re.match(
                r'^([A-Z][A-Za-z\-\']+)\s+([A-Z][A-Za-z\-\']+)\s*:\s*(.+)$',
                line
            )
            if not match:
                # Try case-insensitive version and normalise to upper
                match = re.match(
                    r'^([A-Za-z\-\']+)\s+([A-Za-z\-\']+)\s*:\s*(.+)$',
                    line
                )
                if not match:
                    continue

            first, last, desc = match.group(1), match.group(2), match.group(3)
            name = f"{first.upper()} {last.upper()}"
            last_up = last.upper()

            # Skip duplicate last names
            if last_up in seen_last_names:
                log.debug("[CastNames] Skipping %s - duplicate last name", name)
                continue
            seen_last_names.add(last_up)

            # Infer gender from description keywords for voice preset matching
            desc_lower = desc.lower()
            if any(w in desc_lower for w in ("female", "woman", "she", "her", "scientist woman")):
                gender = "female"
            elif any(w in desc_lower for w in ("male", "man", "he", "his")):
                gender = "male"
            else:
                gender = rng.choice(["male", "female"])

            # Assign voice preset from the appropriate pool, round-robin
            if gender == "female" and female_pool:
                preset = female_pool[f_idx % len(female_pool)]
                f_idx += 1
            elif male_pool:
                preset = male_pool[m_idx % len(male_pool)]
                m_idx += 1
            else:
                preset = "v2/en_speaker_1"

            profiles.append({
                "name": name,
                "gender": gender,
                "age": "adult",
                "demeanor": desc.strip(),
                "notes": desc.strip(),
                "voice_preset": preset,
            })

            if len(profiles) >= num_names:
                break

        if profiles:
            _runtime_log(f"CAST_LLM: {len(profiles)} names generated: "
                         f"{', '.join(p['name'] for p in profiles)}")
        else:
            _runtime_log("CAST_LLM: parse failed - no valid names extracted")

        return profiles if len(profiles) >= num_names else None

    def write_script(self, episode_title,
                     target_words, num_characters, model_id="mistralai/Mistral-Nemo-Instruct-2407",
                     cleanup_model_id="auto (use story model)",
                     custom_premise="", news_headlines=3, temperature=0.8,
                     include_act_breaks=True, self_critique=True,
                     open_close=True,
                     target_length="medium (5 acts)",
                     style="tense claustrophobic",
                     style_custom="",
                     creativity="balanced",
                     arc_enhancer=True,
                     project_state=None,
                     optimization_profile="Standard",
                     perfect_run_spacesaver=False):
        force_lemmy = False # internal alias for clarity below (removed from widget to match INPUT_TYPES)

        # 2026-04-30 STYLE OVERRIDE: when style_custom is non-empty,
        # it replaces the dropdown preset for ALL downstream use
        # (LLM prompts, FLUX radio still, ledger gen_params).  Keeps
        # the dropdown as a quick-pick UI and lets power users feed
        # FLUX more evocative phrases like "rust-belt cyber-noir".
        if isinstance(style_custom, str) and style_custom.strip():
            _override_tone = style_custom.strip()
            _runtime_log(
                f"ScriptWriter: STYLE_OVERRIDE preset {style!r} replaced "
                f"by style_custom={_override_tone!r}"
            )
            style = _override_tone

        target_words = int(target_words)

        # 2026-04-29 SMOKE-TEST PRESET. The "tiny (smoke, 1 act)" target
        # length forces target_words=100 (below the widget min of 350)
        # for fastest end-to-end pipeline validation. num_characters is
        # left as the user-chosen value -- chars and words are
        # independent dimensions in the matrix and the user may want
        # 2 chars + 100 words OR 5 chars + 100 words. To get 2 chars,
        # set the num_characters widget to 2 directly (it accepts
        # min=2). Override BEFORE the early ledger init so the
        # gen_params snapshot reflects the effective target_words.
        #
        # 2026-05-01 ULTRA-SMOKE PRESET (30 words). The "30 words
        # (smoke, 1 act)" target length forces target_words=30 for
        # the absolute fastest end-to-end check -- a few HuMo clips
        # plus the BUG-128/129 routing path exercised on a couple
        # of speaker_role variants without committing to a 27-min
        # render. Detected before the "tiny" check so the longer
        # prefix wins (both start the same in lower()).
        if isinstance(target_length, str) and target_length.lower().startswith("30 words"):
            target_words = 30
            _runtime_log(
                "ScriptWriter: ULTRA-SMOKE preset detected (target_length=30 words) "
                f"-> target_words=30 forced (num_characters={num_characters} unchanged)"
            )
        elif isinstance(target_length, str) and target_length.lower().startswith("tiny"):
            target_words = 100
            _runtime_log(
                "ScriptWriter: SMOKE-TEST preset detected (target_length=tiny) "
                f"-> target_words=100 forced (num_characters={num_characters} unchanged)"
            )

        _runtime_log(f"ScriptWriter: target_words={target_words} (~{max(1, round(target_words / 140))} min at 140 wpm)")

        # 2026-04-29: EARLY LEDGER INIT.
        # Previously the ledger was created inside GemmaHeartbeatStreamer when
        # the body-pass started -- which is AFTER NewsFetcher, model load,
        # OpenClose Spine (3 outlines + evaluator), and auto-title. On a
        # Gemma-4-E2B run with maximum chaos that's a 5+ minute observability
        # gap where /otr/latest_ledger returns the previous run's file. The
        # early init here creates a fresh pending_<ts>_ledger.json on disk
        # the moment write_script() is entered, stamps schema_version + git
        # commit + initial gen_params snapshot, and then saves so the watcher
        # / external tail tools see live state from t=0. Subsequent phases
        # mutate this same ledger (via the singleton) and overwrite the file
        # via Ledger.save() at each gate.
        try:
            from .production_ledger import new_ledger as _new_ledger
            from . import _otr_ledger as _OTRL
            _early_led = _new_ledger()
            _early_led.save()  # write pending_<ts>_ledger.json immediately
            try:
                _OTRL.set_meta(_early_led, "git_commit", _OTRL.lookup_git_commit())
            except Exception:  # noqa: BLE001 -- meta-stamp never blocks
                pass
            # Initial gen_params snapshot. Forward-compatible with the
            # spine-ledger ticket's full meta.gen_params bundle.
            try:
                _early_led.data.setdefault("meta", {})["gen_params_initial"] = {
                    "model_id":             str(model_id),
                    "cleanup_model_id":     str(cleanup_model_id),
                    "target_words":         int(target_words),
                    "num_characters":       int(num_characters),
                    "target_length":        str(target_length),
                    "style":                str(style),
                    "style_custom":         str(style_custom),
                    "creativity":           str(creativity),
                    "optimization_profile": str(optimization_profile),
                    "arc_enhancer":         bool(arc_enhancer),
                    "include_act_breaks":   bool(include_act_breaks),
                    "self_critique":        bool(self_critique),
                    "open_close":           bool(open_close),
                    "custom_premise_set":   bool(custom_premise),
                    "perfect_run_spacesaver": bool(perfect_run_spacesaver),
                }
                # Stamp the spacesaver flag at the meta TOP level too --
                # OTR_RTXUpscale reads it at the end of the workflow to
                # decide whether to wipe per-episode intermediates after
                # the final mp4 lands in otr/obs/. Top-level (not nested
                # under gen_params_initial) so the read is one indirection.
                _early_led.data.setdefault("meta", {})["perfect_run_spacesaver"] = bool(perfect_run_spacesaver)
                _early_led.save()
            except Exception:  # noqa: BLE001 -- snapshot is observability, never blocks
                pass
            # Log path stamping. Tells anyone reading the ledger
            # (watcher script, /otr/latest_ledger consumer, post-mortem
            # tail) exactly which log files captured the verbose state
            # of THIS run. ComfyUI core log catches stdout (Loading LLM
            # model, [OpenClose] Starting..., parse-fatal traces, etc.).
            # OTR runtime log catches the throttled heartbeat (tok/s,
            # scene count, dialogue count, VRAM peaks). Both paths are
            # absolute so a watcher process can `tail -f` them without
            # knowing the user's directory layout.
            try:
                _log_paths = {}
                _otr_log = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "otr_runtime.log",
                )
                _log_paths["otr_runtime"] = _otr_log
                # ComfyUI core log discovery: probe candidate paths
                # by mtime recency and pick the live one. Handles
                # ComfyUI Desktop (Electron) on Win/macOS/Linux PLUS
                # legacy portable layouts. Returns None if no log
                # file exists yet -- the running process may not have
                # flushed its first line at the moment we stamp the
                # ledger; downstream tail tools should re-probe via
                # the helper if comfyui_core is None.
                try:
                    from ._otr_paths import comfyui_log_path as _comfy_log_lookup
                    _log_paths["comfyui_core"] = _comfy_log_lookup()
                except Exception:  # noqa: BLE001 -- helper may fail in CLI/test env
                    _log_paths["comfyui_core"] = None
                _early_led.data.setdefault("meta", {})["log_paths"] = _log_paths
                _early_led.save()
            except Exception:  # noqa: BLE001 -- log-path stamp is observability, never blocks
                pass
            _runtime_log(
                f"EARLY_LEDGER: pending ledger initialized at write_script entry "
                f"(model_id={model_id}, target_words={target_words})"
            )
        except Exception as _early_err:  # noqa: BLE001 -- early init never blocks
            log.warning("[ScriptWriter] Early ledger init failed: %s", _early_err)

        # 2026-04-26 PM two-LLM split (BUG-LOCAL-068 follow-up).
        # Resolve effective cleanup model. "auto (use story model)" means
        # "use the same model for everything" -- backward compatible with
        # every saved workflow that pre-dates this widget.
        if (not cleanup_model_id
                or str(cleanup_model_id).strip().lower().startswith("auto")):
            _effective_cleanup_id = model_id
            _two_llm_active = False
        else:
            _effective_cleanup_id = str(cleanup_model_id).strip()
            _two_llm_active = (_effective_cleanup_id != model_id)
        if _two_llm_active:
            _runtime_log(
                f"ScriptWriter: TWO_LLM_SPLIT active "
                f"creative={model_id!r} cleanup={_effective_cleanup_id!r}"
            )
        else:
            _runtime_log(
                f"ScriptWriter: SINGLE_LLM mode "
                f"model={model_id!r}"
            )

        # -- OPTIMIZATION PROFILE OVERRIDES --
        # Obsidian mode is "One-Shot": no critique, no open-close, no arc-enhancer.
        # This prevents the "slow to a crawl" effect on 4GB cards where multiple
        # LLM passes cause excessive offloading overhead.
        if optimization_profile == "Obsidian (UNSTABLE/4GB)":
            _runtime_log("ScriptWriter: OBSIDIAN PROFILE ACTIVE - forcing One-Shot mode. NOTE: 4GB hardware may still see ~9GB total footprint.")
            log.warning("[LLMScriptWriter] Obsidian Profile: 4GB VRAM IS CURRENTLY UNSTABLE. Total usage will likely exceed physical VRAM.")
            self_critique = False
            open_close = False
            arc_enhancer = False
        elif optimization_profile == "Standard":
            # Standard skips Open-Close (very heavy) but keeps Critique and Arc Enhancer
            # for reasonable quality.
            if open_close:
                log.info("[LLMScriptWriter] Standard Profile: Open-Close was ON but typically skipped in Standard. Allowing user's True choice.")
            else:
                open_close = False
        
        # Pro (Ultra) keeps whatever the widgets say (defaults to all ON).

        # -- MASTER SWITCH INHERITANCE --
        # Save explicitly chosen model so Director can use it automatically.
        # Phase 3 extraction note: write-through to story_orchestrator's
        # module namespace (where LLMDirector.direct() reads it back via
        # `global _CURRENT_LLM_MODEL`). Pre-extraction this was a single
        # `global _CURRENT_LLM_MODEL; _CURRENT_LLM_MODEL = model_id`
        # statement; that becomes a no-op-then-cross-module-sync after
        # the class moves to its own module. The targeted deviation from
        # byte-identical is documented in BUG_LOG / Phase 3 report.
        _so._CURRENT_LLM_MODEL = model_id


        # -- PROJECT STATE (v1.4 Theme C) --
        # Resolve the series bible. If the socket is wired, use the dict from
        # the upstream ProjectStateLoader. Otherwise fall back to the on-disk
        # project_state.json (or defaults if the file does not exist).
        # This call is read-only and cheap - safe for the generation path.
        try:
            if project_state is None:
                _project_state_obj = ProjectState.load()
            else:
                _project_state_obj = ProjectState.from_dict(project_state)
            project_state_preamble = _project_state_obj.prompt_preamble()
        except Exception as e:
            _runtime_log(f"ScriptWriter: project_state load failed, continuing without preamble: {e}")
            project_state_preamble = ""
        _runtime_log(f"ScriptWriter: project_state_preamble_chars={len(project_state_preamble)}")

        # v1.4 Theme C - VRAM telemetry. Reset peak so the per-phase high
        # water mark reflects this script writer run only, then snapshot on
        # entry, after model load (via best-effort hook below), and on exit.
        vram_reset_peak("script_writer_entry")
        vram_snapshot("script_writer_entry")

        # -- DIAGNOSTIC: log feature flags so we can confirm they're received --
        _runtime_log(f"ScriptWriter: PARAMS open_close={open_close} self_critique={self_critique} "
                     f"custom_premise={'(set)' if custom_premise else '(empty)'} "
                     f"target_words={target_words} chars={num_characters} "
                     f"length={target_length} style={style} creativity={creativity} arc_enhancer={arc_enhancer}")

        # ======================================================================
        # CREATIVITY DIAL - temperature/top_p mapping
        # The creativity widget overrides the raw temperature value with curated
        # presets so the user doesn't have to think in floats.
        # ======================================================================
        temp_map = {
            "safe & tight": 0.6,
            "balanced": 0.85,
            "wild & rough": 0.92,
            "maximum chaos": 0.95,  # BUG-014: 1.35 caused total format collapse; 0.95 stays creative but respects structure
        }
        top_p_map = {
            "safe & tight": 0.9,
            "balanced": 0.95,
            "wild & rough": 0.98,
            "maximum chaos": 0.99,
        }
        active_temp = temp_map.get(creativity, 0.85)
        active_top_p = top_p_map.get(creativity, 0.95)
        # Override the temperature variable used everywhere downstream
        temperature = active_temp
        _runtime_log(f"ScriptWriter: CREATIVITY {creativity} - temp={active_temp} top_p={active_top_p}")

        # ======================================================================
        # LENGTH + STYLE DIRECTIVES
        # These get injected into the user prompt to force dialogue VOLUME
        # rather than [PAUSE/BEAT] padding. Targets the "Zoom call pacing" bug.
        # ======================================================================
        # HARD MINIMUMS - word-count based enforcement (BUG-012/020 fix).
        # Widget is now target_words directly. No conversion needed.
        _target_words = target_words
        # ~8 lines per minute at 140 wpm. The 18-line floor is a baseline
        # for >=100 word episodes; for the 30-word ultra-smoke preset
        # 18 lines would force ~1.6 words/line (incoherent). Drop the
        # floor for very short runs.
        if target_words <= 50:
            _min_lines = 3  # 30-word ultra-smoke -> 3-5 lines, ~6-10 words/line
        else:
            _min_lines = max(18, target_words // 18)
        _act_label = {
            "short (3 acts)": "3 acts",
            "medium (5 acts)": "5 acts",
            "long (7-8 acts)": "7-8 acts",
            "epic (10+ acts)": "10+ acts",
        }.get(target_length, "5 acts")
        _extend_hint = (" If your first draft is shorter, EXTEND the middle acts "
                        "with more conflict, more interruptions, and more reaction beats."
                        if target_words >= 1120 else "")
        _subplot_hint = " Allow sub-plots." if target_words >= 2520 else ""
        # BUG-007 root cause fix: short acts + short runtime made the LLM
        # produce narration instead of tagged dialogue. Force the format
        # explicitly when act count is low.
        #
        # BUG-LOCAL-062 fix (2026-04-24): previous wording insisted on
        # 'CHARACTER_NAME: dialogue' bare-colon format, which Mistral Nemo
        # IGNORED roughly half the time (it emits [NAME, mood] dialogue
        # shorthand instead) AND occasionally collapsed to narration-only output
        # (run #2 produced 0 dialogue lines, the parser had nothing to work
        # with, final MP4 was 48.5s instead of ~7 min). Now tolerant of both
        # bracketed shorthand AND bare-colon forms (post-BUG-LOCAL-061 the
        # parser accepts both) AND explicitly forbids narration-only output.
        _format_hint = ""
        if _act_label == "3 acts":
            _format_hint = (
                " CRITICAL FORMAT RULE: Every spoken line MUST be TAGGED with the "
                "speaking character's name. Two tag forms are accepted - pick one "
                "and use it consistently: "
                "(A) '[NAME, gender, age, mood] dialogue text' or "
                "(B) 'NAME: dialogue text' (all-caps name, colon, dialogue). "
                "NEVER write untagged prose, NEVER write stage directions without "
                "a speaker tag, NEVER let the episode collapse to ANNOUNCER-only "
                "narration. An episode without real character conversations is "
                "REJECTED. Even with only 3 acts, every line must be tagged and "
                "the conversation between characters must carry the story."
            )
        # BUG-LOCAL-132 fix (2026-05-01): for ultra-smoke (<=50 words)
        # the prompt must impose an UPPER BOUND, not a floor. Previous
        # wording was "AT LEAST {N} words ... do not stop until you've
        # written at least {N}" -- which is a floor with no ceiling.
        # The LLM took the 30-word smoke target as permission to write
        # 234 words (run signal_lost_skindeep_microneedle_..._163736),
        # turning a ~25-min smoke into a ~3 hr render. Smoke runs
        # need a HARD ceiling so the verification stays fast.
        if _target_words <= 50:
            _ceiling = max(_target_words + 10, int(_target_words * 1.4))
            length_instruction = (
                f"MANDATORY ULTRA-SMOKE: {_act_label}, EXACTLY {_target_words} words "
                f"of spoken dialogue (range: {_target_words}-{_ceiling} words HARD CAP, "
                f"NOT counting ANNOUNCER). Minimum {_min_lines} dialogue lines, maximum "
                f"{max(_min_lines + 1, int(_min_lines * 1.5))} lines. This is a SMOKE TEST "
                f"intended to verify the pipeline end-to-end in ~25 minutes; do NOT pad. "
                f"At ~140 wpm this is ~{max(3, int(_target_words / 140 * 60))} seconds of "
                f"audio. STOP writing as soon as you have a beginning, middle, and end "
                f"that fit in {_ceiling} words. Long monologues are FORBIDDEN."
                f"{_format_hint}"
            )
        else:
            length_instruction = (
                f"MANDATORY: {_act_label}, AT LEAST {_target_words} words of spoken dialogue "
                f"(minimum {_min_lines} dialogue lines, NOT counting ANNOUNCER).{_subplot_hint} "
                f"This script will be read aloud by voice actors at ~140 words per minute, "
                f"so {_target_words} words = ~{max(1, round(_target_words / 140))} minutes of audio. "
                f"Do NOT stop until you have written at least {_target_words} words of character dialogue."
                f"{_extend_hint}{_format_hint}"
            )
        style_instruction = f"Style: {style.upper()}. Lean hard into that tone throughout - every line should reflect this tone."

        # Bark health check moved to LLMDirector to prevent VRAM OOM during script generation.
        log.info(f"[LLMScriptWriter] Feature flags: open_close={open_close}, "
                 f"self_critique={self_critique}, custom_premise={'set' if custom_premise else 'empty'}")

        # ======================================================================
        # PHASE 1: PRE-FLIGHT & INPUT VALIDATION (v1.1)
        # Catch bad configs before burning RTX 5080 compute time.
        # ======================================================================

        # Collect guardrail warnings to display in UI
        guardrail_warnings = []

        # -- 1a. Parameter sanity checks --
        # Short episodes: too many characters starves dialogue per character
        if target_words <= 700 and num_characters > 4:
            log.warning("[PreFlight] target_words=%d with %d characters is too many - "
                        "clamping to 4 characters for short episode", target_words, num_characters)
            _runtime_log(f"PREFLIGHT: Clamped num_characters to 4 ({target_words}-word episode)")
            guardrail_warnings.append(f"[!] Auto-clamped {num_characters} -> 4 characters ({target_words}-word episode max: 4)")
            num_characters = 4
        if target_words <= 420 and num_characters > 3:
            log.warning("[PreFlight] target_words=%d with %d characters is too many - "
                        "clamping to 3 characters for very short episode", target_words, num_characters)
            _runtime_log(f"PREFLIGHT: Clamped num_characters to 3 ({target_words}-word episode)")
            guardrail_warnings.append(f"[!] Auto-clamped {num_characters} -> 3 characters ({target_words}-word episode max: 3)")
            num_characters = 3

        # Long episodes: too few characters can't sustain narrative tension
        # 2026-04-29: respect user's explicit pick of 1 or 2 (monologue or
        # duo). Only auto-bump if user picked the default 4 with insufficient
        # acts handling, OR if user picked a non-1/2 value below 3. The
        # rationale: 1-char monologue is a legitimate narrative form
        # (audio diary, war journal, last-broadcaster), and forcing it
        # back to 3 silently destroys the user's intent.
        _act_count_for_clamp = {"30 words (smoke, 1 act)": 1,
                                "tiny (smoke, 1 act)": 1, "short (3 acts)": 3,
                                "medium (5 acts)": 5, "long (7-8 acts)": 8,
                                "epic (10+ acts)": 12}.get(target_length, 5)
        if _act_count_for_clamp >= 7 and num_characters < 3 and num_characters not in (1, 2):
            log.warning("[PreFlight] %d characters too few for %s - clamping to 3",
                        num_characters, target_length)
            _runtime_log(f"PREFLIGHT: Clamped num_characters to 3 (too few for {target_length})")
            guardrail_warnings.append(f"[!] Auto-clamped {num_characters} -> 3 characters ({target_length} requires minimum 3)")
            num_characters = 3
        elif _act_count_for_clamp >= 7 and num_characters in (1, 2):
            _runtime_log(
                f"PREFLIGHT: respecting user-explicit num_characters={num_characters} "
                f"on {target_length} episode (would normally bump to 3; skipped because "
                f"1=monologue and 2=duo are deliberate narrative choices)"
            )

        if target_words <= 420 and include_act_breaks:
            log.warning("[PreFlight] Act breaks disabled for %d-word episode (too short)", target_words)
            _runtime_log("PREFLIGHT: Act breaks disabled (episode too short)")
            guardrail_warnings.append("[!] Act breaks disabled (too short for <=420-word episodes)")
            include_act_breaks = False

        # Obsidian profile + long episode = severe truncation (2500 token cap)
        if optimization_profile == "Obsidian (UNSTABLE/4GB)" and target_words > 1400:
            log.warning("[PreFlight] Obsidian profile with %d-word episode will truncate badly - "
                        "clamping to 1400 words", target_words)
            _runtime_log(f"PREFLIGHT: Clamped target_words from {target_words} to 1400 (Obsidian token cap)")
            guardrail_warnings.append(f"[!] Auto-clamped {target_words} -> 1400 words (Obsidian profile max: 1400)")
            target_words = 1400

        # -- 1b. Custom premise enforcement --
        # When user provides a premise, skip RSS entirely - zero context contamination
        if custom_premise:
            open_close = False  # User already knows what story they want
            log.info("[PreFlight] Custom premise set - bypassing RSS fetch and Open-Close")
            _runtime_log("PREFLIGHT: Custom premise detected - RSS bypassed, Open-Close disabled")

        # -- 1c. Global token budgeting --
        # target_words comes directly from the widget. ~5 chars/word average.
        target_chars = target_words * 5  # Hard cap for downstream length enforcement

        # -- 1d. Episode fingerprint for reproducibility --
        import hashlib
        fingerprint_data = f"{episode_title}|{style}|{target_words}|{num_characters}|{temperature}"
        episode_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:12]
        _runtime_log(f"ScriptWriter: FINGERPRINT {episode_fingerprint} | {episode_title} | {style}")

        # -- Deterministic seeding from episode fingerprint --
        # Same fingerprint - same torch RNG state - reproducible Gemma generation.
        try:
            import torch
            seed = int(episode_fingerprint, 16) % (2**31 - 1)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            random.seed(seed)
            _runtime_log(f"ScriptWriter: SEED {seed} (from fingerprint {episode_fingerprint})")
        except Exception as _seed_err:
            log.warning(f"[LLMScriptWriter] Could not set deterministic seed: {_seed_err}")

        # ======================================================================
        # RSS FETCH (or custom premise bypass)
        # ======================================================================

        if custom_premise:
            # Custom premise mode - build minimal news block from premise text
            news = [{
                "headline": episode_title or "Custom Episode",
                "summary": custom_premise[:500],
                "full_text": custom_premise,
                "source": "User Premise",
                "date": str(datetime.now().date()),
                "link": "",
            }]
            news_block = f"CUSTOM PREMISE (provided by user):\n{custom_premise}"
        else:
            # -- 1e. RSS fetch with deterministic fallback --
            # 2026-04-29: pass style + model_id + profile so the
            # LLM-rank curator picks narratively-fit headlines for this
            # specific episode's genre. Also enables history dedup so
            # back-to-back runs don't get the same Orion Flywheel.
            try:
                news = _fetch_science_news(
                    style=style,
                    model_id=model_id,
                    optimization_profile=optimization_profile,
                )
            except Exception as rss_err:
                log.warning("[PreFlight] RSS fetch failed: %s - using fallback seed", rss_err)
                _runtime_log(f"PREFLIGHT: RSS_FALLBACK - {rss_err}")
                # Deterministic fallback seeds - real science, manually curated
                _FALLBACK_SEEDS = [
                    {
                        "headline": "Deep-sea microbes found thriving in high-pressure volcanic vents challenge limits of life",
                        "summary": "Researchers discover extremophile bacteria colonies at 4,000m depth near hydrothermal vents "
                                   "that metabolize hydrogen sulfide at temperatures exceeding 120C, suggesting life may exist "
                                   "in similar conditions on Europa and Enceladus.",
                        "full_text": "Researchers discover extremophile bacteria colonies at 4,000m depth near hydrothermal vents "
                                     "that metabolize hydrogen sulfide at temperatures exceeding 120C. The organisms use a novel "
                                     "chemosynthetic pathway never observed before, converting volcanic minerals directly into "
                                     "cellular energy without sunlight. This discovery challenges our understanding of the minimum "
                                     "requirements for life and has major implications for astrobiology missions targeting ocean "
                                     "worlds like Europa and Enceladus.",
                        "source": "Nature Geoscience (fallback seed)",
                        "date": str(datetime.now().date()),
                        "link": "",
                    },
                    {
                        "headline": "Quantum entanglement maintained at room temperature for first time using diamond lattice",
                        "summary": "A team at ETH Zurich demonstrates stable quantum entanglement between nitrogen-vacancy centers "
                                   "in diamond at 22C for over 100 microseconds, eliminating the need for near-absolute-zero cooling.",
                        "full_text": "A team at ETH Zurich demonstrates stable quantum entanglement between nitrogen-vacancy centers "
                                   "in diamond at room temperature for over 100 microseconds. The breakthrough uses a novel spin-echo "
                                   "protocol that actively corrects thermal decoherence in real time. If scaled, the technique could "
                                   "enable practical quantum sensors for medical imaging and navigation systems that operate outside "
                                   "laboratory conditions.",
                        "source": "Physical Review Letters (fallback seed)",
                        "date": str(datetime.now().date()),
                        "link": "",
                    },
                    {
                        "headline": "CRISPR-based gene drive successfully suppresses invasive mosquito population in contained trial",
                        "summary": "A controlled field trial in Burkina Faso demonstrates that a CRISPR gene drive targeting female "
                                   "fertility reduced Anopheles gambiae populations by 90 percent within 8 generations.",
                        "full_text": "A controlled field trial demonstrates that a CRISPR gene drive targeting female fertility in "
                                     "Anopheles gambiae mosquitoes reduced populations by 90 percent within 8 generations inside a "
                                     "contained outdoor enclosure. The drive spread to 95 percent of the population within 4 generations. "
                                     "Researchers emphasize the need for further ecological impact studies before any open-release trials, "
                                     "but the results represent the most successful demonstration of gene drive technology in a near-wild setting.",
                        "source": "Science (fallback seed)",
                        "date": str(datetime.now().date()),
                        "link": "",
                    },
                ]
                news = [random.choice(_FALLBACK_SEEDS)]

            # -- 1f. Headline sanitization --
            # Strip emojis, cap length, normalize whitespace to prevent prompt injection
            for n in news:
                # Remove emojis and non-ASCII decorators
                n["headline"] = re.sub(r'[^\x20-\x7E]', '', n["headline"]).strip()[:280]
                # Normalize whitespace
                n["headline"] = re.sub(r'\s+', ' ', n["headline"])
            # -- 1g. NEWS SUMMARIZATION PASS --
            # Instead of jamming raw article text (often 5K-20K chars of prose,
            # ads, boilerplate) into the script prompt, distill it into a dense
            # fact summary. This gives the script LLM ALL the science without
            # blowing the context window.
            for n in news:
                _raw = n.get("full_text", n.get("summary", ""))
                if len(_raw) < 500:
                    # Short text — no summarization needed
                    continue
                _runtime_log(
                    f"NEWS_SUMMARY: Summarizing '{n['headline'][:60]}' "
                    f"({len(_raw)} chars) via LLM"
                )
                _summary_prompt = (
                    "You are a science news analyst preparing source material for a radio drama writer.\n"
                    "The writer will turn this article into a dramatic audio story with characters and dialogue.\n\n"
                    "Extract EVERY important fact into a dense bullet-point summary, organized for storytelling.\n\n"
                    "RULES:\n"
                    "- Keep ALL names, numbers, dates, locations, institutions, and technical terms\n"
                    "- Keep ALL cause-and-effect relationships and scientific mechanisms\n"
                    "- Keep quotes from researchers or officials — these become character dialogue\n"
                    "- Highlight human stakes: who benefits, who is at risk, what could go wrong\n"
                    "- Highlight dramatic tension: ethical dilemmas, competing interests, unknowns\n"
                    "- Note sensory details useful for audio drama: sounds, environments, settings\n"
                    "- Remove ads, navigation text, subscription prompts, and boilerplate\n"
                    "- Remove repetitive phrasing — say each fact exactly once\n"
                    "- Output ONLY the bullet-point summary, no preamble\n\n"
                    f"HEADLINE: {n['headline']}\n"
                    f"SOURCE: {n['source']}\n\n"
                    f"FULL ARTICLE TEXT:\n{_raw}\n\n"
                    "DENSE FACT SUMMARY FOR RADIO DRAMA WRITER:"
                )
                try:
                    _summarized = _run_with_timeout(
                        lambda: _generate_with_llm(
                            _summary_prompt,
                            model_id=model_id,
                            max_new_tokens=800,
                            temperature=0.2,
                            optimization_profile=optimization_profile,
                        ),
                        timeout_sec=60,
                        phase_label="NewsSummary",
                    )
                    if _summarized and len(_summarized.strip()) > 100:
                        _runtime_log(
                            f"NEWS_SUMMARY: Distilled {len(_raw)} chars -> "
                            f"{len(_summarized.strip())} chars"
                        )
                        n["full_text"] = _summarized.strip()
                    else:
                        _runtime_log("NEWS_SUMMARY: Summary too short, keeping original text")
                        # Fall back to capped original
                        if len(_raw) > 12000:
                            n["full_text"] = _raw[:12000] + "\n[... article truncated at 12,000 chars]"
                except Exception as _e:
                    log.warning("[NEWS_SUMMARY] Summarization failed: %s — keeping original text", _e)
                    if len(_raw) > 12000:
                        n["full_text"] = _raw[:12000] + "\n[... article truncated at 12,000 chars]"

        news_block = "\n".join(
            f"- {n['headline']} ({n['source']}, {n['date']})\n\n{n.get('full_text', n['summary'])}"
            for n in news
        )
        news_json = json.dumps(news, indent=2)

        # Calculate target words
        # target_words and target_chars already computed in Phase 1 pre-flight

        # -- Easter egg: 11% chance Lemmy appears as a character --
        # A grizzled, seen-it-all engineer/mechanic who speaks in blunt,
        # colorful metaphors. Rare enough to be a surprise, frequent enough
        # that regulars will notice. Named after Lemmy Kilmister.
        # force_lemmy=True overrides for testing (validates voice collision fix).
        # Use _LEMMY_RNG (SystemRandom) instead of seeded `random` so the 11%
        # is actually 11% per run, not frozen by the per-episode fingerprint seed.
        _natural_roll = _LEMMY_RNG.random() < 0.11
        
        # Lemmy Telemetry Counter
        global _LEMMY_HISTORY
        _LEMMY_HISTORY.append(_natural_roll)
        if len(_LEMMY_HISTORY) > 50:
            _LEMMY_HISTORY.pop(0)
        _hits = sum(_LEMMY_HISTORY)
        _rate = (_hits / len(_LEMMY_HISTORY)) * 100
        _runtime_log(f"TELEMETRY: Lemmy hit rate [{_hits}/{len(_LEMMY_HISTORY)}] = {_rate:.1f}%")
        
        lemmy_roll = force_lemmy or _natural_roll
        if force_lemmy:
            _lemmy_source = "[EMOJI] Lemmy was summoned by the boss (force toggle ON)"
        elif _natural_roll:
            _lemmy_source = "[EMOJI] Lemmy rolled in on his own (lucky 11%)"
        else:
            _lemmy_source = "[EMOJI] Lemmy stayed in the garage tonight"
        log.info(f"[LLMScriptWriter] {_lemmy_source}  [force={force_lemmy}, rng_hit={_natural_roll}]")
        lemmy_directive = ""
        if lemmy_roll:
            lemmy_directive = (
                "\nSPECIAL CHARACTER REQUIREMENT: One of the characters MUST be named LEMMY - "
                "a resourceful, slightly unconventional engineer/mechanic who operates on the fringes "
                "of authority but proves essential in critical moments. He has a hands-on technical "
                "mindset, more comfortable solving problems directly than following protocol. "
                "Personality: dryly humorous, pragmatic, rough around the edges, but loyal and "
                "dependable when it counts. He questions leadership, bends rules, but his instincts "
                "are sharp under pressure. In the team dynamic Lemmy is the fixer and improviser - "
                "he adapts quickly, thinks creatively, and keeps things moving when plans fall apart. "
                "Give him at least 3 lines of dialogue. Use the name LEMMY consistently "
                "(not ENGINEER LEMMY, just LEMMY).\n"
                "LEMMY SFX REQUIREMENT: Before LEMMY's FIRST line of dialogue, you MUST include "
                "exactly this SFX cue on its own line:\n"
                "[SFX: heavy wrench strike on metal pipe, single resonant clank]\n"
                "This is his signature sound - it plays once, the first time he appears, nowhere else.\n"
            )
            log.info("[LLMScriptWriter] - Lemmy Easter egg activated (11%% roll) - wrench SFX cued")

        # -- Gemma owns character names - they become canonical character_ids --
        # We do NOT pre-seed names. Gemma invents its own character names while
        # writing. Those names are stable pipeline keys used by BatchBark and
        # SceneSequencer. The Director adds a procedural display_name (e.g.
        # "BLAKE ARCHER") for human-facing output only - never as a pipeline key.

        # -- Phase 1b: Model Selection & Prompting --
        # v1.4 Fix: Small models (2B) suffer from "Model Collapse" if the
        # prompt is too complex. We swap to a "LITE" version for these.
        # Check for 2B specifically (avoiding false hits on 26b or 31b)
        is_small_model = any(tag in model_id.lower() for tag in ("2b-it", "2b_it", "small")) or (model_id.lower().endswith("2b"))

        # 2026-04-29: prepend SIGNAL LOST canon (tonal rules, period
        # rules, recurring motifs, used premises/twists/motifs) to
        # the writer's system prompt so the LLM has explicit
        # anti-repeat guidance and tonal anchors. Skipped on small
        # models to avoid Model Collapse from prompt size.
        # 2026-04-30: future-proofing for E2B-only users. Previously
        # the canon block was SKIPPED entirely on small models; that
        # left the writer with NO 1947-period anchor and contributed
        # to A15-A19 anachronism leakage caught by the critic. Now
        # small models get a COMPACT canon (period rules only --
        # ~200 tokens vs ~800 for the full block) so the period
        # discipline still reaches them without bloating the prompt.
        canon_block = _load_canon_for_writer(
            skip=False,
            compact=is_small_model,
        )
        system_base = canon_block + SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT
        if is_small_model:
            # Gemma 2B Lite role prevents prose and header hallucinations
            lite_role = "<system_role>STRICT OTR TAGS ONLY. No prose. Start every line with a tag.</system_role>"
            system_base = lite_role + "\n\n" + SCRIPT_SYSTEM_PROMPT
            
        approx_minutes = max(1, round(target_words / 140))
        system = system_base.format(
            approx_minutes=approx_minutes,
            target_words=target_words,
            news_block=news_block,
            num_characters=num_characters,
        )

        # -- PRE-ROLL DETERMINISTIC CAST ROSTER --
        seed_str = f"{episode_title}_{target_words}_{style}_{time.time()}"
        cast_rng = random.Random(seed_str)
        
        # `pre_rolled_cast` stays a list[str] for backward compat with every
        # downstream consumer (name cleanup, fallback_cast, set difference
        # against parsed names, etc.). The new `pre_rolled_cast_traits` dict
        # carries (gender, age, tone, energy, register, signature) per name
        # so the writer prompt can inject voice profiles AND the post-script
        # voice-consistency check can verify the LLM kept the profiles
        # stable across every [VOICE:] tag.
        pre_rolled_cast = []
        pre_rolled_cast_traits = {}
        seen_first = set()
        seen_last = set()
        seen_traits_idx = set()
        num_non_announcers = max(1, num_characters)

        # Injected Lemmy: if he rolled in, he occupies one of the cast slots
        # so he appears in the MANDATORY CAST ROSTER (ensuring the writer
        # uses him). Lemmy gets the first trait slot deterministically so
        # his voice profile is reproducible across episodes.
        if lemmy_roll:
            pre_rolled_cast.append("LEMMY")
            pre_rolled_cast_traits["LEMMY"] = _VOICE_TRAITS[0]
            seen_first.add("LEMMY")
            seen_traits_idx.add(0)

        while len(pre_rolled_cast) < num_non_announcers:
            f_name = cast_rng.choice(_FIRST_NAMES).upper()
            l_name = cast_rng.choice(_LAST_NAMES).upper()
            if f_name in seen_first or l_name in seen_last:
                continue
            # Pick an unused voice profile so every cast member sounds
            # distinct. If the requested cast size exceeds the trait pool,
            # cycle the pool rather than fail.
            available_traits = [
                i for i in range(len(_VOICE_TRAITS)) if i not in seen_traits_idx
            ]
            if not available_traits:
                seen_traits_idx.clear()
                available_traits = list(range(len(_VOICE_TRAITS)))
            trait_idx = cast_rng.choice(available_traits)
            full_name = f"{f_name} {l_name}"
            seen_first.add(f_name)
            seen_last.add(l_name)
            seen_traits_idx.add(trait_idx)
            pre_rolled_cast.append(full_name)
            pre_rolled_cast_traits[full_name] = _VOICE_TRAITS[trait_idx]

        # Build the cast roster block with per-character voice profiles
        # injected. Each line carries the fixed (gender, age, tone, energy)
        # quad plus the vocab register and signature speech tic. The writer
        # prompt references these explicitly so the LLM does not improvise
        # traits inside each [VOICE:] tag.
        _cast_lines = []
        for _name in pre_rolled_cast:
            _traits = pre_rolled_cast_traits.get(_name)
            if not _traits:
                _cast_lines.append(f"- {_name}")
                continue
            _g, _a, _t, _e, _reg, _sig = _traits
            _cast_lines.append(
                f"- {_name} ({_g}, {_a}, {_t}, {_e} energy)"
                f" - register: {_reg}; signature: {_sig}"
            )

        cast_roster_block = (
            "MANDATORY CAST ROSTER WITH VOICE PROFILES:\n"
            f"You MUST use exactly these {num_non_announcers} character names "
            "and no others for your speaking roles.\n"
            "Each character has a fixed voice profile -- keep it CONSISTENT "
            "across every [VOICE:] tag for that character:\n"
            + "\n".join(_cast_lines) + "\n"
            "Preserve spelling exactly. Do not introduce substitute names, "
            "nicknames, or titles. When you write a [VOICE: NAME, ...] tag "
            "for a character, use THAT character's gender, age, tone, and "
            "energy from the profile above -- not improvised values. "
            "If ANNOUNCER is present, it does not count as a cast invention."
        )

        # -- Write canonical cast to config/episode_cast.txt --
        # Single source of truth for the name cleanup pass downstream.
        # Pre-rolled traits include gender, so the cast config now records
        # gender at pre-roll time instead of waiting on the Director.
        _cast_config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "config", "episode_cast.txt"
        )
        try:
            os.makedirs(os.path.dirname(_cast_config_path), exist_ok=True)
            with open(_cast_config_path, "w", encoding="utf-8") as _cf:
                _cf.write("# Episode Cast - auto-generated per episode\n")
                for _cname in pre_rolled_cast:
                    _ctraits = pre_rolled_cast_traits.get(_cname)
                    _cgender = _ctraits[0] if _ctraits else "unknown"
                    _cf.write(f"{_cname} | {_cgender}\n")
            _runtime_log(f"ScriptWriter: CAST_CONFIG written: {len(pre_rolled_cast)} characters -> {_cast_config_path}")
        except Exception as _cast_err:
            log.warning("[ScriptWriter] Failed to write cast config: %s", _cast_err)

        # -- Open-Close Expansion --
        # BUG-LOCAL-005 v2 (round-robin verdict 2026-05-02): ultra-smoke MUST
        # bypass OpenClose entirely. The 3-outline evaluator holds three
        # parallel KV caches simultaneously (Gemini's calculation: ~6 GB
        # at 16k context BF16), which is the actual source of the
        # 29.5-GB-on-16-GB OOM, not the final write_script forward pass.
        # Detect the preset BEFORE _open_close_expansion fires and short-
        # circuit. Same logic also skips for "tiny (smoke, 1 act)" since
        # that preset is also a pipeline ping where outline competition
        # buys nothing.
        is_ultra_smoke = (
            isinstance(target_length, str)
            and target_length.lower().startswith("30 words")
        )
        is_tiny_smoke = (
            isinstance(target_length, str)
            and target_length.lower().startswith("tiny")
        )
        is_smoke_preset = is_ultra_smoke or is_tiny_smoke

        winning_outline = ""
        _runtime_log(
            f"ScriptWriter: OPEN-CLOSE CHECK: open_close={open_close} "
            f"(type={type(open_close).__name__}), custom_premise='{custom_premise}' "
            f"(bool={bool(custom_premise)}), is_ultra_smoke={is_ultra_smoke} "
            f"is_tiny_smoke={is_tiny_smoke}, "
            f"condition={open_close and not custom_premise and not is_smoke_preset}"
        )
        if open_close and not custom_premise and not is_smoke_preset:
            winning_outline = self._open_close_expansion(
                system, style, news_block, num_characters,
                target_words, lemmy_directive,
                model_id, temperature, cast_roster_block=cast_roster_block
            )
        elif is_smoke_preset:
            _runtime_log(
                "ScriptWriter: OPEN-CLOSE SKIPPED for smoke preset "
                f"(target_length={target_length!r}) -- prevents the 3-outline "
                "evaluator from holding parallel KV caches that drove the "
                "BUG-LOCAL-004 OOM."
            )

        # -- Auto-title from spine (added 2026-04-26) --
        # If the user did not supply a strong title, ask the LLM for one
        # grounded in the winning outline. The spine knows more about what
        # this episode IS than a user typing in a placeholder. Override
        # only when the user value is empty, "auto", or a known stuck-default.
        _user_title_clean = (episode_title or "").strip().lower()
        _wants_auto_title = (
            not _user_title_clean
            or _user_title_clean == "auto"
            or _user_title_clean in _STUCK_TITLE_DEFAULTS
        )
        if winning_outline and _wants_auto_title:
            try:
                _spine_title = self._generate_title_from_spine(
                    winning_outline=winning_outline,
                    style=style,
                    news_block=news_block,
                    model_id=model_id,
                    temperature=temperature,
                    optimization_profile=optimization_profile,
                )
                if _spine_title:
                    log.info(
                        "[ScriptWriter] AUTO_TITLE_FROM_SPINE: %r (was: %r)",
                        _spine_title, episode_title
                    )
                    _runtime_log(
                        f"AUTO_TITLE_FROM_SPINE: {_spine_title!r} "
                        f"(was {episode_title!r})"
                    )
                    episode_title = _spine_title
            except Exception as _t_err:
                log.warning(
                    "[ScriptWriter] Auto-title generation failed: %s "
                    "- falling back to user/LLM/derived chain", _t_err
                )

        # -- Build final script prompt --
        # Mode label must match the logic in _open_close_expansion_inner so the
        # downstream prompt asks the model to expand a PITCH (long episodes) or
        # an OUTLINE (short episodes) accordingly.
        oc_mode_label = "PITCH" if target_words >= 2100 else "OUTLINE"

        # BUG-LOCAL-005 fix (2026-05-02). The "30 words (smoke, 1 act)" preset
        # forces target_words=30, which is too short to fit the standard
        # prompt's TITLE + SCENE + ENV + SFX + ANNOUNCER opening + multiple
        # character lines + ANNOUNCER closing + MUSIC structure. The model
        # degrades to either prose (no [VOICE: ...] markers) or empty output.
        # Symptom captured 2026-05-02: 571 tokens generated, 0 scenes / 0
        # dialogue lines / 0 characters parsed; OpenClose 3-outline evaluator
        # returned 0 chars on all 3 focuses; downstream OOM at peak 29.5 GB
        # after the parse-retry loop.
        #
        # This branch swaps in a minimal ULTRA_SMOKE prompt that (a) keeps the
        # structural markers the parser greps for ([VOICE: ...], === SCENE 1
        # ===, [SFX: ...], [MUSIC: ...]) and (b) tells the model that this is
        # a pipeline ping, not a story -- ~30 words across 4 short lines is
        # the explicit budget. Output is parseable by the existing v1/v2
        # bracket-form parser at story_orchestrator.py:3021. Detected before
        # the winning_outline branch so OpenClose's empty-result fallback
        # doesn't reach the standard else-branch with an unsuitable prompt.
        # is_ultra_smoke + is_tiny_smoke + is_smoke_preset already computed above
        # at the OpenClose bypass (~line 5034). Re-using the same names here.
        if is_ultra_smoke:
            user_prompt = f"""Write a 30-WORD SMOKE-TEST fragment of "SIGNAL LOST". This is a PIPELINE PING, not a story -- the goal is to exercise the parser + audio + video chain end-to-end with the smallest possible script.

EPISODE TITLE: {episode_title if episode_title else "(invent a 2-word evocative title)"}
GENRE: {style.replace("_", " ")}
CHARACTERS: {num_characters} speaking roles plus ANNOUNCER
{cast_roster_block}
TOTAL DIALOGUE BUDGET: ~30 words across the 4 lines below. Each line must be SHORT.

REQUIRED OUTPUT (exactly this structure -- do NOT add scenes, do NOT add characters, do NOT add commentary or markdown):

TITLE: <your 2-word title>
=== SCENE 1 ===
[ENV: one-sentence setting -- 8 words max]
[SFX: one establishing sound]
[VOICE: ANNOUNCER, female, 50s, authoritative, calm] Short opening sentence under 12 words.
[VOICE: CHARACTER_1_NAME, gender, age, tone, energy] Short dialogue line under 8 words.
[VOICE: CHARACTER_2_NAME, gender, age, tone, energy] Short dialogue line under 8 words.
[VOICE: ANNOUNCER, female, 50s, authoritative, calm] Short closing sentence under 10 words.
[MUSIC: Closing theme]

CRITICAL FORMAT RULES (BUG-007 enforcement -- the parser depends on these):
- EVERY dialogue line MUST start with `[VOICE: NAME, ...]` followed by the line. No exceptions.
- The bracket-form `[VOICE: ...]` is the ONLY accepted dialogue tag here. Do NOT use bare `CHARACTER:` form, do NOT use `[CHARACTER, traits]` form, do NOT use prose narration.
- The `=== SCENE 1 ===` marker MUST appear exactly once, on its own line, before any dialogue.
- The TITLE: line MUST be the very first line of output.
- Replace CHARACTER_1_NAME and CHARACTER_2_NAME with actual character names from the roster above. Do not leave the placeholder names.
- DO NOT output anything before TITLE: or after [MUSIC: Closing theme]. No explanations, no analysis, no markdown."""
        elif is_tiny_smoke:
            # BUG-LOCAL-005 v3 (round-robin follow-up 2026-05-02): apply the
            # smoke-template pattern to the older "tiny (smoke, 1 act)" 100-word
            # preset too. Same structural marker contract as ultra-smoke,
            # scaled to ~6-8 voice lines (~100 words). Reviewers ChatGPT 5.5
            # and NVIDIA Nemotron 49B explicitly recommended this when
            # reviewing the ultra-smoke fix; the OpenClose bypass already
            # fires for tiny-smoke so this prompt completes the parity.
            user_prompt = f"""Write a 100-WORD SMOKE-TEST fragment of "SIGNAL LOST". This is a PIPELINE PING with slightly more breathing room than the 30-word ultra-smoke -- still NOT a story. The goal is to exercise the parser + audio + video chain end-to-end on a script just large enough to surface multi-line bugs (line ordering, beat boundaries, music chunking) without committing to a 5-minute render.

EPISODE TITLE: {episode_title if episode_title else "(invent a 3-word evocative title)"}
GENRE: {style.replace("_", " ")}
CHARACTERS: {num_characters} speaking roles plus ANNOUNCER
{cast_roster_block}
TOTAL DIALOGUE BUDGET: ~100 words across the 7 lines below. Each line should land between 8 and 18 words.

REQUIRED OUTPUT (exactly this structure -- do NOT add scenes, do NOT add characters, do NOT add commentary or markdown):

TITLE: <your 3-word title>
=== SCENE 1 ===
[ENV: one-sentence setting]
[SFX: one establishing sound]
[VOICE: ANNOUNCER, female, 50s, authoritative, calm] Opening sentence -- introduce time, place, premise. 14 words max.
[VOICE: CHARACTER_1_NAME, gender, age, tone, energy] First dramatic line. 12 words max.
[VOICE: CHARACTER_2_NAME, gender, age, tone, energy] Reaction or counter-line. 12 words max.
[SFX: action sound]
[VOICE: CHARACTER_1_NAME, gender, age, tone, energy] Second beat. 12 words max.
[VOICE: CHARACTER_2_NAME, gender, age, tone, energy] Closing exchange. 12 words max.
[VOICE: ANNOUNCER, female, 50s, authoritative, calm] Closing sentence -- echo the science hook. 14 words max.
[MUSIC: Closing theme]

CRITICAL FORMAT RULES (BUG-007 enforcement -- the parser depends on these):
- EVERY dialogue line MUST start with `[VOICE: NAME, ...]` followed by the line. No exceptions.
- The bracket-form `[VOICE: ...]` is the ONLY accepted dialogue tag here. Do NOT use bare `CHARACTER:` form, do NOT use `[CHARACTER, traits]` form, do NOT use prose narration.
- The `=== SCENE 1 ===` marker MUST appear exactly once, on its own line, before any dialogue.
- The TITLE: line MUST be the very first line of output.
- Replace CHARACTER_1_NAME and CHARACTER_2_NAME with actual character names from the roster above. Do not leave the placeholder names.
- DO NOT output anything before TITLE: or after [MUSIC: Closing theme]. No explanations, no analysis, no markdown."""
        elif winning_outline:
            user_prompt = f"""Write a complete episode of "SIGNAL LOST" based on the WINNING {oc_mode_label} below.

LENGTH DIRECTIVE: {length_instruction}
STYLE DIRECTIVE: {style_instruction}

WINNING {oc_mode_label} (selected by evaluator from 3 competing concepts):
{winning_outline}

EPISODE TITLE: {episode_title if episode_title else "(generate a unique, evocative title for THIS episode)"}
GENRE: {style.replace("_", " ")}
CHARACTERS: {num_characters} speaking roles plus ANNOUNCER
{cast_roster_block}
TARGET LENGTH: ~{target_words} words
{"STRUCTURAL BREAKS: Include 2-3 act breaks marked with [ACT TWO], [ACT THREE] etc." if include_act_breaks else ""}
{lemmy_directive}

REMEMBER: The {oc_mode_label.lower()} above is your premise and story spine. {"Invent the full scene structure, acts, and SFX based on it." if oc_mode_label == "PITCH" else "Follow its structure, characters, and arc."} Flesh it out with sharp dialogue, atmospheric [SFX:] and [ENV:] tags, and real emotional stakes.

REQUIRED FIRST LINE: The VERY FIRST line of your output MUST be exactly:
TITLE: <your chosen title here>
The title must be unique to this episode - do NOT use "The Last Frequency", "Untitled", "Signal Lost", or "Episode". Draw it from the premise, characters, or a striking image in the story.

Begin the full script now. Follow this structure exactly:
TITLE: <your chosen title>
=== SCENE 1 ===
[ENV: location description, ambient noise, vibe]
[SFX: establishing sound]
(beat)
[VOICE: ANNOUNCER, <male|female - ALTERNATE per episode, do NOT default to male>, <40s|50s|60s>, authoritative, calm] [Opening introduction - time, place, character names and roles, science hook, tagline. REQUIRED. Always first.]
[VOICE: CHARACTER_NAME, gender, age, tone, energy] First dramatic line - drop us in medias res.
[VOICE: CHARACTER_NAME, gender, age, tone, energy] Response line.
(beat)
[SFX: action sound]
...
[VOICE: ANNOUNCER, <same gender/age as opening>, authoritative, calm] [Hard-science epilogue - cite ONLY the real article provided above. Headline, source, date. No invented IDs.]
[MUSIC: Closing theme]"""
        else:
            user_prompt = f"""Write a complete episode of "SIGNAL LOST" - a contemporary sci-fi audio drama anthology.

LENGTH DIRECTIVE: {length_instruction}
STYLE DIRECTIVE: {style_instruction}

EPISODE TITLE: {episode_title if episode_title else "(generate a unique, evocative title for THIS episode)"}
GENRE: {style.replace("_", " ")}
CHARACTERS: {num_characters} speaking roles plus ANNOUNCER
{cast_roster_block}
TARGET LENGTH: ~{target_words} words
{"STRUCTURAL BREAKS: Include 2-3 act breaks marked with [ACT TWO], [ACT THREE] etc." if include_act_breaks else ""}
{lemmy_directive}
{"PREMISE: " + custom_premise if custom_premise else "The news headlines above ARE the premise. Extrapolate them. What's the next terrifying or profound step?"}

STORY ARC SEED: Use Arc Type {random.choice("ABCDEFGHIJKL")} from the Story Arc Engine above. Commit fully to that structure.

REMEMBER: Story first. Make the listener CARE about these people before you scare them with science. Write dialogue that sounds like real humans under pressure - not scientists reading papers.

REQUIRED FIRST LINE: The VERY FIRST line of your output MUST be exactly:
TITLE: <your chosen title here>
The title must be unique to this episode - do NOT use "The Last Frequency", "Untitled", "Signal Lost", or "Episode". Draw it from the premise, characters, or a striking image in the story.

Begin the full script now. Follow this structure exactly:
TITLE: <your chosen title>
=== SCENE 1 ===
[ENV: location description, ambient noise, vibe]
[SFX: establishing sound]
(beat)
[VOICE: ANNOUNCER, <male|female - ALTERNATE per episode, do NOT default to male>, <40s|50s|60s>, authoritative, calm] [Opening introduction - time, place, character names and roles, science hook, tagline. REQUIRED. Always first.]
[VOICE: CHARACTER_NAME, gender, age, tone, energy] First dramatic line - drop us in medias res.
[VOICE: CHARACTER_NAME, gender, age, tone, energy] Response line.
(beat)
[SFX: action sound]
...
[VOICE: ANNOUNCER, <same gender/age as opening>, authoritative, calm] [Hard-science epilogue - cite ONLY the real article provided above. Headline, source, date. No invented IDs.]
[MUSIC: Closing theme]"""

        # v1.4 Theme C - prepend the series bible preamble so every downstream
        # phase (outline, draft, critique, revise, arc enhancer) sees the same
        # locked decisions. Empty preamble degrades gracefully to v1.3 behavior.
        if project_state_preamble:
            full_prompt = f"[SERIES BIBLE]\n{project_state_preamble}\n\n{system}\n\n{user_prompt}"
        else:
            full_prompt = f"{system}\n\n{user_prompt}"

        log.info(f"[LLMScriptWriter] Generating {target_words}-word episode "
                 f"'{episode_title}' ({style}) using {model_id}")
        log.info(f"[LLMScriptWriter] News seed: {news[0]['headline']} | {news[0]['source']}")

        # For episodes > 5 min, generate act-by-act to avoid token truncation.
        # 8,192 max_new_tokens - 6,000 words. A 25-min episode needs ~3,250 words
        # which fits, but 45-min needs ~5,850 which is tight. Chunked generation
        # ensures we never hit the ceiling and produces more coherent long scripts.

        if target_words <= 700 or optimization_profile == "Obsidian (UNSTABLE/4GB)":
            # Short episodes (or Obsidian 4GB tier): single-pass generation.
            # Floor at 1024 - even a 1-min episode needs enough tokens to
            # complete canonical structure (ENV, SFX, VOICE tags, beats).
            # Without the floor, 1-min = 260 tokens, which truncates mid-scene.

            # BUG-012 FIX: Cap KV cache for direct generation in Obsidian profile.
            # Standard: 8192 limit. Obsidian: 2500 limit (protects 4GB VRAM ceiling).
            if optimization_profile == "Obsidian (UNSTABLE/4GB)":
                if target_words > 700:
                    log.warning("[LLMScriptWriter] Obsidian profile forced single-pass on %d-word target. "
                                "Expect shorter overall length.", target_words)
                max_new_tokens = max(int(target_words * _TOKEN_RATIO_DIALOGUE), 1024)
                max_new_tokens = min(max_new_tokens, 2500)
            else:
                max_new_tokens = max(int(target_words * _TOKEN_RATIO_DIALOGUE), 1024)
                max_new_tokens = min(max_new_tokens, 8192)

            # BUG-LOCAL-005 v2 fix (round-robin verdict 2026-05-02): clamp
            # max_new_tokens for the smoke presets. The 30-word ultra-smoke
            # captured 571 tokens of degenerate output before parse-fail;
            # all three reviewers (ChatGPT 5.5, Gemini 3.1, NVIDIA Nemotron 49B)
            # flagged this as a runaway-generation symptom that the prompt
            # alone won't stop. 256 fits the ~30-word ultra-smoke template
            # (TITLE + scene marker + 4 [VOICE: ...] lines + [MUSIC:]) with
            # generous slack. Tiny smoke gets 384 -- ~100 words across more
            # lines but still tightly capped vs the 1024 floor that produced
            # the runaway.
            if is_ultra_smoke:
                max_new_tokens = min(max_new_tokens, 256)
            elif is_tiny_smoke:
                max_new_tokens = min(max_new_tokens, 384)

            # BUG-LOCAL-004 fix (2026-05-02). Force a hard VRAM flush before
            # the main script-writer call when prior LLM phases (NewsSummary,
            # CAST_CONFIG, OpenClose 3-outline + evaluator + synthesizer) have
            # already accumulated KV cache + activation peaks. Symptom captured
            # 2026-05-02: peak_gb=29.498 on a 16 GB device after OpenClose
            # returned 3x 0-char outlines and write_script proceeded; next
            # _generate_with_llm OOMed at the model.generate() forward pass.
            # _generate_with_llm has a torch.cuda.empty_cache() in its finally
            # block but that fires AFTER each call; here we flush BEFORE the
            # call so cumulative state from prior calls in this turn doesn't
            # ride along. Keeps the LLM weights resident (CLAUDE.md rule:
            # use _flush_vram_keep_llm() between LLM phases, not
            # force_vram_offload()).
            try:
                _flush_vram_keep_llm()
                _runtime_log("ScriptWriter: VRAM flushed before main generation (BUG-LOCAL-004)")
            except Exception as _flush_err:  # noqa: BLE001 -- never block on flush
                log.debug(
                    "[LLMScriptWriter] _flush_vram_keep_llm pre-main-gen failed: %s",
                    _flush_err,
                )

            # BUG-LOCAL-004 fix (2026-05-02). Hard parse-retry cap. The 30-word
            # ultra-smoke previously could fail to parse (0 lines), trigger
            # WORD_EXTEND or related retry paths, and accumulate VRAM until
            # OOM. Bounded retry: at most MAX_PARSE_RETRIES attempts; on
            # exhaustion, accept whatever the last attempt returned and let
            # the parse-fail observability stamp it in the ledger rather than
            # OOM during a 4th forward pass.
            MAX_PARSE_RETRIES = 2
            _parse_attempt = 0
            script_text = ""
            while _parse_attempt < MAX_PARSE_RETRIES:
                _parse_attempt += 1
                _runtime_log(
                    f"ScriptWriter: GENERATE attempt={_parse_attempt}/{MAX_PARSE_RETRIES} "
                    f"max_new_tokens={max_new_tokens} target_words={target_words}"
                )
                script_text = _generate_with_llm(
                    full_prompt,
                    model_id=model_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=active_top_p,
                    optimization_profile=optimization_profile,
                    live_ledger=True,  # L1.5: stream cast + lines to ledger
                )
                # Branch-aware parseability check via the module-level helper
                # `_check_parse_ok` (BUG-LOCAL-004/005, round-robin verdict
                # 2026-05-02). Helper extracted so tests can lock the contract
                # without driving the full write_script() method.
                _check = _check_parse_ok(
                    script_text,
                    is_ultra_smoke=is_ultra_smoke,
                    is_tiny_smoke=is_tiny_smoke,
                )
                _has_scene = _check["has_scene"]
                _voice_hits = _check["voice_hits"]
                _bare_hits = _check["bare_hits"]
                _parse_ok = _check["parse_ok"]
                if _parse_ok:
                    _runtime_log(
                        f"ScriptWriter: PARSE_OK attempt={_parse_attempt} "
                        f"has_scene={_has_scene} voice_hits={_voice_hits} "
                        f"bare_hits={_bare_hits} "
                        f"smoke=ultra:{is_ultra_smoke}/tiny:{is_tiny_smoke}"
                    )
                    break
                _runtime_log(
                    f"ScriptWriter: PARSE_FAIL attempt={_parse_attempt} "
                    f"has_scene={_has_scene} voice_hits={_voice_hits} "
                    f"bare_hits={_bare_hits} "
                    f"smoke=ultra:{is_ultra_smoke}/tiny:{is_tiny_smoke} -- "
                    + (
                        f"flushing VRAM and retrying"
                        if _parse_attempt < MAX_PARSE_RETRIES
                        else "MAX_PARSE_RETRIES_EXCEEDED, accepting last output"
                    )
                )
                if _parse_attempt < MAX_PARSE_RETRIES:
                    try:
                        _flush_vram_keep_llm()
                    except Exception:  # noqa: BLE001
                        pass
        else:
            # Long episodes: chunked act-by-act generation
            script_text = self._generate_chunked(
                system, episode_title, style, num_characters,
                target_words, custom_premise, news_block,
                include_act_breaks, model_id, temperature,
                target_length=target_length,
                lemmy_directive=lemmy_directive,
                top_p=active_top_p,
                cast_roster_block=cast_roster_block,
                optimization_profile=optimization_profile
            )

        # -- v1.1 CHECKS & CRITIQUES LOOP -------------------------------------
        # Three-pass refinement: Draft - Critique - Revise
        # v1.5 HARDENING: For long scripts (>3 acts), run CRITIQUE-ONLY
        # (structural analysis without global rewrite) to avoid the
        # "Summarization Collapse" where the LLM condensed 5 acts into ~30 lines.
        # The critique findings feed into the Arc Enhancer's spine for
        # targeted opening/closing polish.
        # ----------------------------------------------------------------------
        _runtime_log(f"ScriptWriter: CRITIQUE CHECK: self_critique={self_critique}")
        
        # Determine act count for critique strategy
        actual_act_count = 1
        if "target_length" in locals() and target_length:
            _act_map = {"short (3 acts)": 3, "medium (5 acts)": 5, "long (7-8 acts)": 8, "epic (10+ acts)": 12}
            actual_act_count = _act_map.get(target_length, 1)

        # BUG-021 FIX: Critique and arc enhancer are structural passes, not
        # creative ones. Cap their temperature so maximum chaos creativity
        # does not produce sloppy formatting in post-generation cleanup.
        _structural_temp = min(temperature, 0.6)
        _runtime_log(f"ScriptWriter: structural_temp={_structural_temp} (creativity temp={temperature})")

        if self_critique and actual_act_count <= 3:
            # Short scripts: full critique + revision (safe - script fits in context)
            _runtime_log("ScriptWriter: >>> ENTERING critique_and_revise (full)")
            script_text = self._critique_and_revise(
                script_text, style, target_words, model_id, _structural_temp,
                optimization_profile=optimization_profile
            )
            _runtime_log("ScriptWriter: <<< EXITED critique_and_revise")
        elif self_critique:
            # v1.5: For long scripts, critique now runs UPSTREAM in the Story Editor
            # (before act generation), not as a post-generation pass. The critique
            # guides each act's writing via per-act briefs. The findings are already
            # stored on self._last_critique_findings from _generate_chunked().
            _runtime_log(f"ScriptWriter: Critique ran upstream via Story Editor ({actual_act_count} acts)")


        # -- ARC ENHANCER (v1.3 flagship feature) ------------------------------
        # Paired opening + closing bookend rewrite to ensure narrative coherence.
        # ----------------------------------------------------------------------
        if arc_enhancer:
            _runtime_log("ScriptWriter: >>> ENTERING arc_enhancer")
            # v1.5: Pass critique findings (if any) so the Arc Enhancer
            # can address structural weaknesses when polishing start/end.
            _findings = getattr(self, '_last_critique_findings', '') or ''
            _act_sums = getattr(self, '_last_act_summaries', []) or []
            script_text = self._execute_arc_enhancer(
                script_text, style, episode_title, news_block, model_id, _structural_temp,
                optimization_profile=optimization_profile,
                critique_findings=_findings,
                act_summaries=_act_sums
            )
            _runtime_log("ScriptWriter: <<< EXITED arc_enhancer")

        # -- SCENE DEDUPLICATION (BUG-026 fix) --------------------------------
        # LLMs sometimes place the climax mid-script AND at the end, or the
        # revision/extension pass duplicates closing sequences. Detect near-
        # identical scenes and remove the earlier copy (the last occurrence is
        # structurally correct for a climax/closing).
        _scene_splits = re.split(r'(===\s*SCENE\s+\d+\s*===)', script_text, flags=re.IGNORECASE)
        if len(_scene_splits) >= 5:  # At least 2 full scenes (header+body pairs)
            # Reassemble into (header, body) pairs
            _scenes = []
            for i in range(1, len(_scene_splits) - 1, 2):
                _scenes.append((_scene_splits[i], _scene_splits[i + 1]))
            _preamble = _scene_splits[0]  # Text before first scene header

            # Compare dialogue content between scenes (strip whitespace/tags for comparison)
            def _dialogue_fingerprint(body):
                """Extract just dialogue words for similarity comparison."""
                lines = _extract_all_dialogue(body)
                return " ".join(d.strip().lower() for _, d in lines)

            _dupes_removed = 0
            _keep = list(range(len(_scenes)))  # indices to keep
            for i in range(len(_scenes)):
                if i not in _keep:
                    continue
                fp_i = _dialogue_fingerprint(_scenes[i][1])
                if len(fp_i) < 50:  # Too short to compare meaningfully
                    continue
                for j in range(i + 1, len(_scenes)):
                    if j not in _keep:
                        continue
                    fp_j = _dialogue_fingerprint(_scenes[j][1])
                    if len(fp_j) < 50:
                        continue
                    # Check similarity using simple overlap ratio
                    _shorter = min(len(fp_i), len(fp_j))
                    _longer = max(len(fp_i), len(fp_j))
                    if _shorter == 0:
                        continue
                    # If one fingerprint contains 80%+ of the other, it's a dupe
                    _common = sum(1 for c1, c2 in zip(fp_i, fp_j) if c1 == c2)
                    _similarity = _common / _longer
                    if _similarity > 0.75:
                        # Remove the EARLIER scene (keep the later one as the climax)
                        _keep.remove(i)
                        _dupes_removed += 1
                        _runtime_log(
                            f"DEDUP: Scene {i+1} is {_similarity:.0%} similar to "
                            f"Scene {j+1} - removing earlier copy"
                        )
                        break  # Scene i is already removed, move on

            if _dupes_removed > 0:
                # Rebuild script with remaining scenes, renumbering
                _new_parts = [_preamble]
                for new_num, orig_idx in enumerate(_keep, 1):
                    header, body = _scenes[orig_idx]
                    # Renumber the scene header
                    renumbered = re.sub(
                        r'===\s*SCENE\s+\d+\s*===',
                        f'=== SCENE {new_num} ===',
                        header, flags=re.IGNORECASE
                    )
                    _new_parts.append(renumbered)
                    _new_parts.append(body)
                script_text = "".join(_new_parts)
                _runtime_log(
                    f"DEDUP: Removed {_dupes_removed} duplicate scene(s), "
                    f"renumbered {len(_keep)} remaining scenes"
                )
                log.warning(
                    "[BUG-026] Removed %d duplicate scene(s) from script. "
                    "LLM placed climax mid-script and repeated it at the end.",
                    _dupes_removed
                )

        # -- Content safety filter - catch anything the prompt policy missed --
        script_text, blocked = _content_filter(script_text)
        if blocked:
            log.warning("[LLMScriptWriter] Content filter caught %d word(s) - replaced with minced oaths",
                        len(blocked))

        # -- FIX-4 (v1.2): Stock-name leak guard -------------------------------
        # Gemma sometimes types the wrong character name inside dialogue body -
        # e.g. "it keeps spiking when you talk about the frequencies, Rex"
        # when the intended character is VEX. This is NOT a hardcoded blocklist:
        # we extract the real roster from [VOICE: NAME, ...] tags, then scan
        # direct-address tokens (", Name." or "Name,") in dialogue body. Any
        # capitalized proper-noun-looking token that is NOT in the roster gets
        # replaced with the phonetically closest roster name via difflib.
        # Pure structural fix - no baked names anywhere.
        try:
            import difflib
            # v1.4 HACK: Strip markdown ** bolding before roster extraction
            _clean_script = re.sub(r'\*\*(\[.*?\])\*\*', r'\1', script_text)
            _roster = set(re.findall(r'\[VOICE:\s*([A-Z][A-Z0-9_ -]+)\s*,', _clean_script))
            if _roster:
                _roster_list = sorted(_roster)
                _leaks_fixed = 0
                # Match direct-address tokens in dialogue body.
                # 1. Title-case: "Rex." or ", Maya," - common in narrative speech.
                # 2. ALL-CAPS: "REX" or "MAYA" - common in direct address inside dialogue.
                # Token length 2-8 chars, followed by punctuation or whitespace.
                _addr_pat = re.compile(
                    r'(?<=[,\s])'
                    r'([A-Z][a-z]{1,7}|[A-Z]{2,8})'
                    r'(?=[.,!?\s])'
                )
                def _leak_fix(m):
                    nonlocal _leaks_fixed
                    token = m.group(1)
                    upper = token.upper()
                    if upper in _roster:
                        return token  # legit roster name
                    # Common English words - skip
                    if token.lower() in {
                        "the", "and", "but", "for", "with", "from", "into", "that",
                        "this", "then", "than", "when", "what", "will", "were",
                        "been", "have", "just", "only", "some", "such", "very",
                        "now", "yes", "no", "ok", "okay", "sir", "maam", "doctor",
                        "captain", "commander", "listen", "look", "hey", "wait",
                        "stop", "god", "lord", "earth", "mars", "moon", "sun",
                        "orion", "nasa", "please", "thanks", "maybe", "never",
                        "always", "forever", "tonight", "tomorrow", "yesterday",
                    }:
                        return token
                    # Phonetic match to closest roster name (cutoff 0.80 - strictly
                    # tuned to catch typos like 'Marten'->'Martin' without falsely
                    # capturing normal English prose or SFX tags).
                    match = difflib.get_close_matches(upper, _roster_list, n=1, cutoff=0.80)
                    if match:
                        _leaks_fixed += 1
                        # Preserve title-case for dialogue flow
                        return match[0].title()
                    return token
                script_text = _addr_pat.sub(_leak_fix, script_text)
                if _leaks_fixed:
                    log.warning(
                        "[LLMScriptWriter] NameLeakGuard: repaired %d typo/leak(s) "
                        "in dialogue body (roster=%s)",
                        _leaks_fixed, sorted(_roster)
                    )
        except Exception as _e:
            log.warning("[LLMScriptWriter] NameLeakGuard skipped: %s", _e)

        # -- Citation hallucination guard --------------------------------------
        # Gemma sometimes invents plausible-looking ArXiv IDs (arXiv:2401.XXXXX)
        # even when told not to. These look authoritative but are fabricated.
        # Detect and warn - the IDs are left in the text (stripping would create
        # jarring gaps) but the log makes the problem visible for review.
        _arxiv_pat = re.compile(r'\barXiv:\s*\d{4}\.\d{4,5}\b', re.IGNORECASE)
        _doi_pat   = re.compile(r'\bdoi\.org/10\.\d{4,}/\S+', re.IGNORECASE)
        hallucinated_ids = _arxiv_pat.findall(script_text) + _doi_pat.findall(script_text)

        # Cross-check against the real article source
        real_source_text = " ".join(
            f"{n['headline']} {n['source']} {n.get('full_text', n['summary'])}"
            for n in news
        ).lower()

        bad_ids = []
        for hid in hallucinated_ids:
            # If the ID string doesn't appear in any form in the real article
            # content we provided, it's almost certainly hallucinated
            if hid.lower().replace(" ", "") not in real_source_text.replace(" ", ""):
                bad_ids.append(hid)

        if bad_ids:
            log.warning(
                "[CitationGuard] %d likely hallucinated citation ID(s) detected: %s - "
                "Gemma invented these. Review the epilogue before publishing.",
                len(bad_ids), ", ".join(bad_ids)
            )
        elif hallucinated_ids:
            log.info("[CitationGuard] %d citation ID(s) found - appear to match source material.",
                     len(hallucinated_ids))

        # -- CitationGuard 2: strip numeric bracket references -----------------
        # Gemma sometimes outputs [1], [2], article #3 when the prompt uses
        # bracket-style placeholders like [SOURCE] as format examples. These
        # become broken grammar when spoken ("According to article .") because
        # _clean_text_for_bark already strips unrecognized bracket tags.
        # Strip them here at the source so the script text is clean before
        # _parse_script() stores it.
        _num_ref_pat = re.compile(
            r'\s*\[\d{1,3}\]'           # [1] [2] [99]
            r'|\s*\(\d{1,3}\)'          # (1) (2)
            r'|\s*article\s+#\s*\d+'    # article #3
            r'|\s*source\s+#\s*\d+'     # source #2
            r'|\s*reference\s+#\s*\d+', # reference #1
            re.IGNORECASE
        )
        stripped_text, nsubs = _num_ref_pat.subn("", script_text)
        if nsubs:
            log.warning(
                "[CitationGuard] Stripped %d numeric citation marker(s) ([1], article #N, etc.) "
                "from script text - update prompt to prevent recurrence.", nsubs
            )
            script_text = stripped_text

        # ══════════════════════════════════════════════════════════════
        # POST-GENERATION PIPELINE (all on raw text, then parse once)
        #
        # Order matters:
        #   1. WORD_EXTEND  — count dialogue words, extend if under 70%
        #   2. ANNOUNCER    — add bookends (sees full extended script)
        #   3. FORMAT_NORM  — clean up everything into canonical format
        #   4. PARSE        — parse clean text into structured JSON
        # ══════════════════════════════════════════════════════════════

        # -- SCENE CHECKPOINT: raw LLM output (pre-pipeline baseline) --
        _log_scene_checkpoint("00_RAW_LLM_OUTPUT", script_text)

        # -- STEP 0: NORMALIZE BOLD DIALOGUE NAMES (BUG-023 fix) --------
        # LLMs at high temperature produce **NAME**, emotion: format.
        # Strip to canonical NAME: before any word-count regex runs.
        _pre_norm_len = len(script_text)
        script_text = _normalize_dialogue_names(script_text)
        if len(script_text) != _pre_norm_len:
            _runtime_log("BOLD_NORM: Stripped Markdown bold from dialogue names")
        _log_scene_checkpoint("01_AFTER_BOLD_NORM", script_text)

        # -- STEP 1: WORD-COUNT ENFORCEMENT (BUG-012/020/025 fix) -----
        # Count dialogue words in raw text using dual-format extraction.
        # Recognizes both bare "NAME: text" AND "[VOICE: NAME, emotion] text"
        # so VOICE-tag scripts are not falsely detected as zero-dialogue.
        _target_words = target_words  # Direct from widget — no conversion needed
        _raw_dialogue_pairs = _extract_all_dialogue(script_text)
        _raw_dialogue_words = sum(
            len(dialogue.split()) for _, dialogue in _raw_dialogue_pairs
        )
        _word_ratio = _raw_dialogue_words / max(1, _target_words)
        _runtime_log(
            f"WORD_ENFORCEMENT: {_raw_dialogue_words} words vs {_target_words} target "
            f"({_word_ratio:.0%}) | @140wpm -> ~{_raw_dialogue_words / 140:.1f} min "
            f"[{len(_raw_dialogue_pairs)} lines detected]"
        )

        # BUG-024: Zero-dialogue detection — creative generation produced
        # only SFX/atmosphere with no character dialogue at all.
        if _raw_dialogue_words == 0:
            _runtime_log(
                "WORD_ENFORCEMENT: [!] ZERO CHARACTER DIALOGUE DETECTED - "
                "script contains only SFX/ANNOUNCER/atmosphere. "
                "WORD_EXTEND will attempt full dialogue generation from cast roster."
            )
            log.warning(
                "[BUG-024] Zero character dialogue in raw script. "
                "Cast roster fallback will be used for extension. "
                "Pre-rolled cast: %s", ", ".join(pre_rolled_cast)
            )

        # BUG-LOCAL-109 (2026-04-29): retry the extension pass up to N
        # times if the LLM's first extension still leaves us under
        # threshold. Captain-Eris with "maximum chaos" + short target
        # words (350) tends to produce a 3-line shell on the first
        # call AND a barely-extended one on the second; the only
        # reliable cure is to keep asking until the model fills out
        # the scenes, with a "no-progress -> abort" guard so we don't
        # spin forever. Threshold also tightened from 70% to 80%.
        # (wormhole_swallowing_phobos 2026-04-29 produced 65/350 words
        # = 18.6% on a single extension; the retry loop closes that
        # gap or surfaces the LLM's actual ceiling.)
        _BUG_109_RATIO_TARGET = 0.80
        _BUG_109_MAX_RETRIES = 3
        _retry_count = 0
        while (
            _word_ratio < _BUG_109_RATIO_TARGET
            and _target_words > 150
            and _retry_count < _BUG_109_MAX_RETRIES
        ):
            _retry_count += 1
            _deficit = _target_words - _raw_dialogue_words
            _prev_words = _raw_dialogue_words
            _runtime_log(
                f"WORD_ENFORCEMENT: UNDER THRESHOLD ({_word_ratio:.0%} < "
                f"{int(_BUG_109_RATIO_TARGET * 100)}%) -- attempt "
                f"{_retry_count}/{_BUG_109_MAX_RETRIES}, deficit "
                f"{_deficit} words"
            )
            # WORD_EXTEND is structured rescue -- use cleanup model.
            script_text = self._extend_script_dialogue(
                script_text, _deficit, _target_words,
                _effective_cleanup_id, style, optimization_profile,
                fallback_cast=pre_rolled_cast
            )
            # Recount after extension (dual-format)
            _raw_dialogue_pairs = _extract_all_dialogue(script_text)
            _raw_dialogue_words = sum(
                len(dialogue.split()) for _, dialogue in _raw_dialogue_pairs
            )
            _word_ratio = _raw_dialogue_words / max(1, _target_words)
            _runtime_log(
                f"WORD_ENFORCEMENT: Post-extension {_retry_count}: "
                f"{_raw_dialogue_words} words ({_word_ratio:.0%}) | "
                f"+{_raw_dialogue_words - _prev_words} this pass | "
                f"~{_raw_dialogue_words / 140:.1f} min"
            )
            # No-progress guard: if the LLM didn't add words this
            # pass, stop retrying -- it has nothing more to say.
            if _raw_dialogue_words <= _prev_words:
                _runtime_log(
                    f"WORD_ENFORCEMENT: extension {_retry_count} added 0 "
                    f"words, model stalled at {_word_ratio:.0%} -- aborting "
                    f"retries"
                )
                break

        if _word_ratio < _BUG_109_RATIO_TARGET and _target_words > 150:
            log.warning(
                "[BUG-109] Script ended at %d/%d words (%.0f%%) after "
                "%d extension pass(es). LLM under-delivered; downstream "
                "audio will be %d sec instead of ~%d sec target.",
                _raw_dialogue_words, _target_words, _word_ratio * 100,
                _retry_count, int(_raw_dialogue_words / 140 * 60),
                int(_target_words / 140 * 60),
            )

        # BUG-LOCAL-109b (visibility): surface per-cast / per-scene
        # gaps so we can see at a glance whether the script was
        # uniformly thin (small word target) vs. structurally lopsided
        # (some scenes / characters with zero dialogue). This is
        # diagnostic-only -- it does not re-prompt; that would
        # require another tool round-trip and is queued for a
        # follow-up commit. The runtime log lets us spot whether
        # bug 109 is a "global thinness" or "structural gap" failure
        # mode on the next render.
        try:
            _post_pairs = _raw_dialogue_pairs
            _names_with_lines = {n for n, _ in _post_pairs}
            _missing_cast = sorted(
                set(pre_rolled_cast) - _names_with_lines
            ) if pre_rolled_cast else []
            _scene_count = max(1, len(re.findall(r'=== SCENE', script_text)))
            # Approximate per-scene line counts by splitting on === SCENE
            _scene_blocks = re.split(r'^===\s*SCENE', script_text, flags=re.MULTILINE)
            _empty_scenes = sum(
                1 for blk in _scene_blocks[1:]  # skip pre-scene preamble
                if not _extract_all_dialogue("=== SCENE" + blk)
            )
            if _missing_cast:
                _runtime_log(
                    f"BUG-109b: cast members with 0 lines: "
                    f"{', '.join(_missing_cast)}"
                )
            if _empty_scenes:
                _runtime_log(
                    f"BUG-109b: {_empty_scenes}/{_scene_count} scene(s) "
                    f"have 0 dialogue lines"
                )
        except Exception as _exc:
            _runtime_log(f"BUG-109b: gap-audit skipped ({_exc})")

        _log_scene_checkpoint("02_AFTER_WORD_EXTEND", script_text)

        # -- STEP 2: ANNOUNCER BOOKENDS (on raw text) -----------------
        # Check if ANNOUNCER lines exist. If not, generate and inject.
        # Runs after word extension so the ANNOUNCER sees the full story.
        #
        # BUG-LOCAL-131 fix (2026-05-01): the previous regex only
        # matched the bare-colon format `^ANNOUNCER:` but the modern
        # LLM prompt asks for the bracketed VOICE-tag format
        # `[VOICE: ANNOUNCER, gender, age, ...]`. The bare-colon
        # check almost always failed against modern output, which
        # forced the fallback _generate_announcer_bookends path to
        # fire every run -- and when that secondary LLM call also
        # produced empty output, the episode shipped with NO
        # ANNOUNCER lines at all. Now matches BOTH formats so a
        # native LLM emission is recognized correctly.
        _ANNOUNCER_BOOKEND_RX = re.compile(
            r'^(?:ANNOUNCER\s*:|\[VOICE\s*:\s*ANNOUNCER\b)',
            re.MULTILINE | re.IGNORECASE,
        )
        # Widen the head/tail sniff windows -- 500 chars is too narrow
        # when the LLM emits a long opening preamble before the first
        # ANNOUNCER tag (and similarly for the tail). 2000 chars on
        # each end captures the announcer slot under any reasonable
        # narrative pacing.
        _has_announcer_open = bool(_ANNOUNCER_BOOKEND_RX.search(
            script_text[:2000]
        ))
        _has_announcer_close = bool(_ANNOUNCER_BOOKEND_RX.search(
            script_text[-2000:]
        ))
        if not _has_announcer_open or not _has_announcer_close:
            _runtime_log(
                f"ANNOUNCER_RAW: Missing bookends (open={_has_announcer_open}, "
                f"close={_has_announcer_close}) - generating via LLM"
            )
            # Extract character names from raw text for context (BUG-025:
            # uses dual-format extraction so VOICE-tag names are included)
            _char_names = {name for name, _ in _raw_dialogue_pairs}
            # Extract news headline
            _news_head = ""
            for nb_line in news_block.split("\n"):
                clean = nb_line.strip()
                if clean and not clean.startswith("CUSTOM") and not clean.startswith("---"):
                    _news_head = clean[:300]
                    break
            # ANNOUNCER bookend gen is structured -- use cleanup model.
            opening_text, closing_text = self._generate_announcer_bookends(
                [], episode_title, style,
                _news_head, _char_names, _effective_cleanup_id,
                optimization_profile,
            )
            # BUG-LOCAL-131 ultimate fallback (2026-05-01): if the
            # secondary LLM call (_generate_announcer_bookends) also
            # produced empty output, the episode would have shipped
            # WITHOUT ANNOUNCER lines at all (Jeffrey: "we should
            # always have an announcer close end and end as
            # requirement"). Backfill with deterministic placeholder
            # text keyed off the episode title so the bookend is
            # ALWAYS present, even when both LLM passes fail.
            _safe_title = (
                str(episode_title).strip()
                if isinstance(episode_title, str) and episode_title.strip()
                else "tonight's broadcast"
            )
            if not opening_text:
                opening_text = (
                    f"From the static between worlds, you're listening "
                    f"to Signal Lost. Tonight: {_safe_title}."
                )
                _runtime_log(
                    "ANNOUNCER_RAW: secondary LLM produced empty opening "
                    "-- using deterministic fallback (BUG-131 safety net)"
                )
            if not closing_text:
                closing_text = (
                    f"And so concludes tonight's signal. "
                    f"Until next we tune in."
                )
                _runtime_log(
                    "ANNOUNCER_RAW: secondary LLM produced empty closing "
                    "-- using deterministic fallback (BUG-131 safety net)"
                )
            if not _has_announcer_open and opening_text:
                script_text = f"ANNOUNCER: {opening_text}\n\n{script_text}"
                _runtime_log(f"ANNOUNCER_RAW: Prepended opening ({len(opening_text)} chars)")
            if not _has_announcer_close and closing_text:
                script_text = f"{script_text}\n\nANNOUNCER: {closing_text}"
                _runtime_log(f"ANNOUNCER_RAW: Appended closing ({len(closing_text)} chars)")
        _log_scene_checkpoint("03_AFTER_ANNOUNCER", script_text)

        # -- STEP 3: FORMAT NORMALIZER (Creative → Strict) ------------
        # Now the script has extensions + announcer. One pass cleans
        # everything into canonical format before the parser runs.
        # FORMAT_NORM is structured polish -- use cleanup model.
        script_text = self._normalize_script_format(
            script_text, _effective_cleanup_id, optimization_profile
        )
        _log_scene_checkpoint("04_AFTER_FORMAT_NORM", script_text)

        # -- STEP 3b: NAME CLEANUP (Python fuzzy match) ---------------
        # Read canonical cast from config/episode_cast.txt and fix any
        # hallucinated name variants the LLM produced under high creativity.
        # Pure Python - no LLM call, no VRAM cost, runs in milliseconds.
        script_text = _cleanup_character_names(script_text, _cast_config_path, pre_rolled_cast)
        _log_scene_checkpoint("05_AFTER_NAME_CLEANUP", script_text)

        # -- STEP 3c: GRAMMARIAN (final logic + grammar polish) -------
        # Light LLM pass at temp 0.3. Fixes grammar, catches logic gaps,
        # ensures dialogue reads naturally. Does NOT add content, rename
        # characters, or change the story. Pure copy-edit.
        # Grammarian is structured polish -- use cleanup model.
        script_text = self._grammarian_pass(
            script_text, _effective_cleanup_id, optimization_profile
        )
        _log_scene_checkpoint("06_AFTER_GRAMMARIAN", script_text)

        # 2026-04-26 BUG-LOCAL-066: capture pre-parse dialogue expectation
        # so the post-parse floor check can detect silent dialogue loss.
        # Counts the same three forms FORMAT_NORM/Grammarian count: bare
        # NAME:, [VOICE: NAME], and [NAME, mood] shorthand. Used purely as
        # a sanity expectation -- if the parser returns far fewer dialogue
        # tokens than this number, something dropped lines silently and
        # the LLM_RESCUE path should attempt recovery.
        _expected_pre_parse_dialogue = (
            len(re.findall(
                r'^[A-Z][A-Z0-9_ ]{1,19}:\s*.+$',
                script_text, re.MULTILINE,
            ))
            + len(re.findall(r'\[VOICE:', script_text, re.IGNORECASE))
            + len(re.findall(
                r'^\[[A-Z][A-Z0-9_ ]{1,20}(?:,\s*.+?)?\]\s*\S',
                script_text, re.MULTILINE,
            ))
        )

        # -- STEP 4: PARSE into structured JSON -----------------------
        # BUG-LOCAL-037: capture the LLM-emitted "TITLE: <...>" line BEFORE
        # we feed script_text to the parser, then strip it. Otherwise the
        # parser reads it as a "TITLE" character speaking the title text,
        # polluting the cast roster and confusing the self-critique pass.
        _early_llm_title = _extract_title_from_script_text(script_text)
        script_text = _RE_TITLE_LINE.sub("", script_text).lstrip()
        if _early_llm_title:
            _runtime_log(
                f"TITLE_STRIP | extracted='{_early_llm_title}' | "
                f"removed leading TITLE: line(s) before parse"
            )

        # Single parse on the fully prepared text.
        # LLM_RESCUE fires only if parser gets 0 dialogue lines.
        try:
            script_lines = self._parse_script(script_text)
        except ValueError as parse_err:
            if "0 dialogue lines" in str(parse_err) and len(script_text) > 500:
                _runtime_log("LLM_RESCUE: Parser found 0 dialogue - attempting LLM reparse")
                # LLM_RESCUE is structured rescue -- use cleanup model.
                rescued_text = self._llm_reparse_rescue(
                    script_text, _effective_cleanup_id, optimization_profile
                )
                if rescued_text and rescued_text != script_text:
                    _runtime_log(f"LLM_RESCUE: Got {len(rescued_text)} chars back - retrying parse")
                    script_lines = self._parse_script(rescued_text)
                    _runtime_log(f"LLM_RESCUE: Reparse recovered {len([l for l in script_lines if l.get('type') == 'dialogue'])} dialogue lines")
                else:
                    _runtime_log("LLM_RESCUE: Rescue pass returned nothing useful - re-raising")
                    raise
            else:
                raise

        # -- SCENE CHECKPOINT: post-parse (structured view) -----------
        # Count scene_break entries in script_lines. If this drops
        # below the 06_AFTER_GRAMMARIAN count, the parser lost scenes.
        _parsed_scene_tokens = [
            str(ln.get("scene", "")).strip().upper()
            for ln in script_lines
            if ln.get("type") == "scene_break"
        ]
        _runtime_log(
            f"SCENE_TRACK: 07_AFTER_PARSE | count={len(_parsed_scene_tokens)} "
            f"| tokens={_parsed_scene_tokens}"
        )

        # BUG-LOCAL-038: dialogue-token checkpoint. If this lands at 0 while
        # the streaming heartbeat saw dialogue, the loss is in FORMAT_NORM /
        # GRAMMARIAN / _parse_script -- not in BatchBark. Makes the silent
        # drop visible in one grep.
        _parsed_dialogue_tokens = [
            ln for ln in script_lines if ln.get("type") == "dialogue"
        ]
        _parsed_dialogue_chars = sorted({
            str(ln.get("character_name", "")).strip()
            for ln in _parsed_dialogue_tokens
            if ln.get("character_name")
        })
        _runtime_log(
            f"DIALOGUE_TRACK: 07_AFTER_PARSE | count={len(_parsed_dialogue_tokens)} "
            f"| characters={_parsed_dialogue_chars}"
        )

        # 2026-04-26 BUG-LOCAL-066: post-parse dialogue floor.
        # Until now LLM_RESCUE only fired when the parser raised on 0
        # dialogue. But Grammarian's polish can rewrite dialogue into a
        # form the parser doesn't recognize, dropping 15+ lines silently
        # while still passing Grammarian's own 80% safety check (which
        # was asymmetric -- pre counted shorthand, post did not). The
        # floor check below catches that drop: if the parser produced
        # <50% of the dialogue lines we expected after Grammarian, attempt
        # an LLM_RESCUE reparse on the un-polished text.
        _post_parse_count = len(_parsed_dialogue_tokens)
        if (_expected_pre_parse_dialogue >= 6
                and _post_parse_count < _expected_pre_parse_dialogue * 0.5):
            _runtime_log(
                f"DIALOGUE_FLOOR: parsed {_post_parse_count} dialogue tokens but "
                f"AFTER_GRAMMARIAN had {_expected_pre_parse_dialogue} -- triggering "
                f"LLM_RESCUE reparse"
            )
            try:
                # BUG-066 floor-trigger LLM_RESCUE -- structured reparse,
                # use cleanup model.
                rescued_text = self._llm_reparse_rescue(
                    script_text, _effective_cleanup_id, optimization_profile
                )
            except Exception as _resc_err:
                rescued_text = None
                _runtime_log(f"LLM_RESCUE: floor-trigger failed ({_resc_err})")
            if rescued_text and rescued_text != script_text:
                rescued_lines = self._parse_script(rescued_text)
                rescued_dialogue = [
                    ln for ln in rescued_lines if ln.get("type") == "dialogue"
                ]
                if len(rescued_dialogue) > _post_parse_count:
                    _runtime_log(
                        f"LLM_RESCUE: floor-trigger recovered "
                        f"{len(rescued_dialogue)} dialogue lines "
                        f"(was {_post_parse_count}, expected ~"
                        f"{_expected_pre_parse_dialogue})"
                    )
                    script_lines = rescued_lines
                    _parsed_dialogue_tokens = rescued_dialogue
                    _parsed_dialogue_chars = sorted({
                        str(ln.get("character_name", "")).strip()
                        for ln in _parsed_dialogue_tokens
                        if ln.get("character_name")
                    })
                    _runtime_log(
                        f"DIALOGUE_TRACK: 07_AFTER_PARSE_RESCUED | "
                        f"count={len(_parsed_dialogue_tokens)} "
                        f"| characters={_parsed_dialogue_chars}"
                    )
                else:
                    _runtime_log(
                        f"LLM_RESCUE: floor-trigger rescue returned only "
                        f"{len(rescued_dialogue)} dialogue lines -- keeping "
                        f"original parser output"
                    )

        # Log guardrail warnings (visible in otr_runtime.log) but keep script_json as pure JSON
        # BUG-016: Never prepend comments to script_json - downstream nodes call json.loads() on it
        if guardrail_warnings:
            for w in guardrail_warnings:
                _runtime_log(f"GUARDRAIL_UI: {w}")

        # ------------------------------------------------------------------
        # BUG-LOCAL-035 TITLE_STUCK FIX: resolve a real episode title and
        # prepend a title token to script_lines so downstream nodes (video,
        # assembler) can read it without falling back to a widget default.
        # Resolution order:
        #   1. user-supplied episode_title (widget)
        #   2. "TITLE: ..." line emitted by the LLM in script_text
        #   3. derived from the first environment token (deterministic)
        #   4. timestamped genre fallback (last resort, still unique)
        # Any result matching _STUCK_TITLE_DEFAULTS is rejected at each step.
        # ------------------------------------------------------------------
        _resolved_title = (episode_title or "").strip()
        _title_source = "user"
        if not _resolved_title or _resolved_title.lower() in _STUCK_TITLE_DEFAULTS:
            # BUG-LOCAL-037: prefer the title we captured BEFORE strip.
            # Falls back to a fresh extract for safety if the early capture
            # was empty for any reason (e.g. self-critique re-emitted one).
            _llm_title = _early_llm_title or _extract_title_from_script_text(script_text)
            if _llm_title and _llm_title.lower() not in _STUCK_TITLE_DEFAULTS:
                _resolved_title = _llm_title
                _title_source = "llm"
            else:
                _derived_title = _derive_title_from_script_lines(
                    script_lines, style
                )
                if _derived_title and _derived_title.lower() not in _STUCK_TITLE_DEFAULTS:
                    _resolved_title = _derived_title
                    _title_source = "derived"
                else:
                    _resolved_title = f"Signal Lost Transmission {int(time.time()) % 100000}"
                    _title_source = "timestamp_fallback"
        _runtime_log(
            f"TITLE_TRACE | source={_title_source} | resolved='{_resolved_title}' "
            f"| user_widget='{episode_title}'"
        )
        # Prepend as first script_lines token so downstream nodes can read it.
        # Token type 'title' is silently skipped by all existing iterators
        # (they filter on dialogue/sfx/scene_break/etc) - safe addition.
        script_lines = [{"type": "title", "value": _resolved_title}] + script_lines

        script_json = json.dumps(script_lines, indent=2)

        # Estimate actual minutes (radio drama pacing ~140 wpm)
        word_count = sum(len(line.get("line", "").split()) for line in script_lines
                         if line.get("type") == "dialogue")
        est_minutes = max(1, round(word_count / 140, 1))

        # -- Phase 1g: Cast map verification --
        # Extract unique character names from parsed script for downstream matching
        script_characters = set()
        for item in script_lines:
            if item.get("type") == "dialogue":
                cname = item.get("character_name", "").upper().strip()
                if cname:
                    script_characters.add(cname)
        _runtime_log(f"ScriptWriter: CAST_MAP {sorted(script_characters)} | "
                     f"{len(script_lines)} lines | ~{word_count} words | ~{est_minutes} min")

        # -- Phase 3d: QA debug dump --
        # Save minimal JSON payload alongside the output for reproducibility
        try:
            qa_data = {
                "fingerprint": episode_fingerprint,
                "timestamp": datetime.now().isoformat(),
                "params": {
                    "episode_title": episode_title,
                    "style": style,
                    "target_words": target_words,
                    "num_characters": num_characters,
                    "open_close": open_close,
                    "self_critique": self_critique,
                    "temperature": temperature,
                },
                "news_seed": news[0]["headline"] if news else "none",
                "news_source": news[0].get("source", "unknown") if news else "none",
                "cast": sorted(script_characters),
                "stats": {
                    "dialogue_lines": sum(1 for l in script_lines if l.get("type") == "dialogue"),
                    "sfx_cues": sum(1 for l in script_lines if l.get("type") == "sfx"),
                    "scenes": sum(1 for l in script_lines if l.get("type") == "scene_break"),
                    "word_count": word_count,
                    "est_minutes": est_minutes,
                    "script_chars": len(script_text),
                },
                "guardrails_triggered": [],  # Populated by downstream phases
            }
            # 2026-04-26 PM BUG-LOCAL-067: nest QA debug dumps under
            # output/otr/audio/ alongside the matching ledger.
            qa_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   "output", "otr", "audio",
                                   f"qa_debug_{episode_fingerprint}.json")
            os.makedirs(os.path.dirname(qa_path), exist_ok=True)
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa_data, f, indent=2)
            _runtime_log(f"ScriptWriter: QA_DUMP saved: qa_debug_{episode_fingerprint}.json")
        except Exception as qa_err:
            log.warning("[QA] Debug dump failed: %s", qa_err)

        log.info(f"[LLMScriptWriter] Generated {len(script_lines)} lines, "
                 f"~{word_count} words, ~{est_minutes} min")

        # -- VRAM handoff: unload Gemma before Bark loads ----------------------
        # Gemma and Bark cannot share 16GB VRAM comfortably. Explicitly unload
        # now so BatchBark starts with a clean VRAM slate.
        _unload_llm()
        _runtime_log("ScriptWriter: Gemma unloaded - VRAM freed for Bark")

        # v1.4 Theme C - exit snapshot after Gemma unload. This should be
        # close to the idle baseline; a large value here means the unload
        # path left memory on the table and needs investigation.
        vram_snapshot("script_writer_exit_after_unload")

        # ------------------------------------------------------------------
        # Production Ledger (L1) -- initial snapshot of cast + lines.
        # Writes are best-effort; ledger failures never block the pipeline.
        # Director fills in scenes + shots + voice_presets; SceneSequencer +
        # SignalLostVideo back-fill timings + final paths.
        # ------------------------------------------------------------------
        try:
            # L1.5: do NOT reset the ledger here. The body streamer
            # already created a fresh ledger via new_ledger() in its
            # __init__ and incrementally populated it with cast + lines
            # during generation. Calling new_ledger() now would dump that
            # to an orphan file with an older timestamp and start a new
            # one. Use get_ledger() to overwrite the same file with the
            # final parsed canonical cast + lines.
            from .production_ledger import get_ledger
            led = get_ledger()
            cast_rows = [{"char_id": f"c{idx+1:02d}", "name": n}
                         for idx, n in enumerate(_parsed_dialogue_chars)]
            name_to_cid = {row["name"]: row["char_id"] for row in cast_rows}
            # Derive a running scene_id so downstream shot assignment can
            # key against it. Scenes appear as explicit 'scene_break' items
            # in script_lines; we number them 1..N in the order seen. A
            # script with no scene_break items still gets a single
            # implicit s01 so shots/lines have somewhere to attach.
            scene_rows = []
            line_to_scene = {}
            current_scene = "s01"
            scene_count = 0
            for i, ln in enumerate(script_lines):
                if ln.get("type") == "scene_break":
                    scene_count += 1
                    current_scene = f"s{scene_count:02d}"
                    scene_rows.append({"scene_id": current_scene,
                                       "env": ""})
                line_to_scene[i] = current_scene
            if not scene_rows:
                scene_rows = [{"scene_id": "s01", "env": ""}]
            line_rows = []
            ln_idx = 0
            for i, ln in enumerate(script_lines):
                if ln.get("type") != "dialogue":
                    continue
                ln_idx += 1
                line_rows.append({
                    "line_id":  f"l{ln_idx:03d}",
                    "shot_id":  None,  # Director assigns shots -> L2
                    "char_id":  name_to_cid.get(ln.get("character_name"), None),
                    "text":     str(ln.get("line") or ""),
                    "traits":   str(ln.get("voice_traits") or ""),
                })
            led.set_cast(cast_rows)
            led.set_scenes(scene_rows)
            led.set_lines(line_rows)
            # Voice-consistency soft warnings: walk every [VOICE:] tag in
            # the FINAL script_text and compare LLM-chosen traits against
            # the pre-rolled cast traits. Mismatches are logged for
            # forward analysis (no schema bump, no run abort). After 2-3
            # episodes of warning data accumulate the spine-ledger
            # ticket can use them to size the structural-validation
            # follow-up. Failure of the check is silent; voice_warnings
            # may be absent on legacy ledgers.
            try:
                _voice_warnings = _check_voice_consistency(
                    script_text, pre_rolled_cast_traits
                )
                if _voice_warnings:
                    led.data["voice_warnings"] = _voice_warnings
                    _runtime_log(
                        f"VOICE_CONSISTENCY: {len(_voice_warnings)} drifted "
                        f"[VOICE:] tag(s) recorded to ledger.voice_warnings"
                    )
                else:
                    _runtime_log(
                        "VOICE_CONSISTENCY: all [VOICE:] tags matched "
                        "pre-rolled cast traits (or no cast traits set)"
                    )
            except Exception as _vc_err:  # noqa: BLE001
                log.warning("[ScriptWriter] voice consistency check failed: %s", _vc_err)

            # ----------------------------------------------------------------
            # BUG-LOCAL-110 Layer 2 (2026-05-05, round-robin verified):
            # stamp the canonical resolved title at the TOP LEVEL of the
            # ledger (alongside episode_id, commit, total_episode_dur_s)
            # so video_engine and other consumers can read a clean
            # `ledger.title` instead of trying to walk script_lines looking
            # for the "type": "title" token. Also stamp meta.title_source
            # for forensics ("user", "llm", "derived", "timestamp_fallback").
            # _resolved_title and _title_source are computed earlier in
            # this method via the BUG-LOCAL-035 fallback chain.
            try:
                _rt = str(_resolved_title or "").strip()
                _ts_src = str(_title_source or "unknown").strip() or "unknown"
                if _rt:
                    led.data["title"] = _rt
                    led.data.setdefault("meta", {})["title_source"] = _ts_src
                    _runtime_log(
                        f"BUG-110: stamped ledger.title={_rt!r} "
                        f"(source={_ts_src})"
                    )
            except Exception as _title_stamp_err:  # noqa: BLE001
                log.warning(
                    "[ScriptWriter] BUG-110 ledger.title stamp failed "
                    "(non-fatal): %s", _title_stamp_err,
                )

            # ----------------------------------------------------------------
            # LTX style brief (Jeffrey directive 2026-05-05): generate ONE
            # per-episode visual style brief that flavors the radio
            # broadcast set to match the story's sci-fi setting. Stamped
            # to ledger.meta.ltx_style_brief; nodes/batch_ltx_render
            # prepends this to every per-line LTX prompt so each episode's
            # LTX clips feel setting-appropriate instead of all looking
            # like the same vintage 1940s vacuum-tube set. Non-fatal on
            # failure -- the role templates work fine standalone.
            try:
                _story_snippet = (
                    (news_block or "").strip()[:400]
                    or script_text[:400]
                )
                _ltx_brief = _generate_ltx_style_brief(
                    style=style,
                    story_snippet=_story_snippet,
                    model_id=model_id,
                    optimization_profile=optimization_profile,
                )
                if _ltx_brief:
                    led.data.setdefault("meta", {})["ltx_style_brief"] = _ltx_brief
                    _runtime_log(
                        f"LTX_STYLE_BRIEF: {_ltx_brief[:120]}"
                        + ("..." if len(_ltx_brief) > 120 else "")
                    )
            except Exception as _brief_err:  # noqa: BLE001
                log.warning(
                    "[ScriptWriter] LTX style brief generation failed "
                    "(non-fatal): %s", _brief_err,
                )

            led.save()
        except Exception as _e:  # noqa: BLE001
            log.warning("[Ledger] ScriptWriter-stage snapshot failed: %s", _e)

        return (script_text, script_json, news_json, est_minutes)

    # -------------------------------------------------------------------------
    # OPEN-CLOSE EXPANSION - 3 competing outlines - evaluator picks winner
    # -------------------------------------------------------------------------

    def _open_close_expansion(self, system, style, news_block,
                              num_characters, target_words,
                              lemmy_directive, model_id, temperature,
                              cast_roster_block="", optimization_profile="Standard"):
        """Generate 3 competing story outlines with different priorities,
        then have an evaluator pick the best one.

        Outline A: Prioritizes character conflict and emotional stakes.
        Outline B: Prioritizes scientific rigor and technical tension.
        Outline C: Prioritizes atmosphere, pacing, and environmental tension.

        The evaluator receives all 3 and selects the strongest narrative,
        optionally merging the best elements from each.

        Returns the winning outline text, or empty string on failure.
        """
        try:
            return self._open_close_expansion_inner(
                system, style, news_block, num_characters,
                target_words, lemmy_directive,
                model_id, temperature,
                cast_roster_block=cast_roster_block,
                optimization_profile=optimization_profile
            )
        except Exception as e:
            log.error("[OpenClose] Top-level failure: %s - falling back to v1.0 direct generation", e)
            _runtime_log(f"OPENCLOSE: OPENCLOSE_FALLBACK - top-level error: {e}")
            return ""

    def _open_close_expansion_inner(self, system, style, news_block,
                                     num_characters, target_words,
                                     lemmy_directive, model_id, temperature,
                                     cast_roster_block="", optimization_profile="Standard"):
        """Inner implementation of Open-Close expansion (wrapped for safety)."""
        log.info("[OpenClose] Starting Open-Close expansion (3 outlines + evaluator)...")
        _runtime_log("OPENCLOSE: Generating 3 competing outlines")

        # -- PITCH MODE (Gemini round 3) --
        # For long episodes (>= 15 min) the 3 full structured outlines bottleneck
        # the run (~10-15 min just for the open-close phase on SDPA). For long
        # episodes we switch to "pitch mode" - a 3-5 sentence logline per concept,
        # ~100 words, no act structure. Saves ~80% of open-close inference time.
        # The full script generator still invents the scene structure downstream.
        # v1.5: 7-Line Micro-Spine Protocol
        # Instead of generating full ~450-token outlines that blow the KV cache
        # and take ~4 min each, we generate ultra-condensed 7-line structural
        # spines (~100 tokens). This cuts Open-Close from ~12 min to ~2 min,
        # eliminates VRAM_CEILING_EXCEEDED warnings during outline generation,
        # and produces tighter narrative structures for act expansion.
        is_pitch_mode = target_words >= 2100
        if is_pitch_mode:
            mode_label = "PITCH"
            outline_max_tokens = 250
            OUTLINE_MIN = 100
            OUTLINE_MAX = 1500
            _runtime_log(
                f"OPENCLOSE: PITCH_MODE enabled for {target_words}-word run "
                f"(max_new_tokens={outline_max_tokens})"
            )
        else:
            mode_label = "SPINE"
            outline_max_tokens = 150   # 7-line spine: ~100 tokens actual output
            OUTLINE_MIN = 80
            OUTLINE_MAX = 1200
            _runtime_log(
                f"OPENCLOSE: SPINE_MODE enabled for {target_words}-word run "
                f"(max_new_tokens={outline_max_tokens})"
            )
        mode_lower = mode_label.lower()

        arc_choices = random.sample("ABCDEFGHIJKL", 3)

        outline_focuses = [
            ("CHARACTER-DRIVEN",
             "Focus on intense interpersonal conflict. Give each character a secret, "
             "a fear, and a breaking point. The science is the pressure cooker - "
             "the people are the story. Make us feel their desperation."),
            ("SCIENCE-DRIVEN",
             "Focus on scientific rigor and technical problem-solving. The plot should "
             "hinge on a real physics/biology constraint that characters must solve under "
             "pressure. Think Apollo 13 - the math IS the drama."),
            ("ATMOSPHERE-DRIVEN",
             "Focus on environmental dread and sensory immersion. Use sound design cues "
             "([SFX:], [ENV:]) heavily. Build a world the listener can HEAR - creaking metal, "
             "distant alarms, breathing in a spacesuit. Slow-burn tension."),
        ]

        # v1.4 Theme B - 3-outline evaluator re-enabled.
        #
        # History: this flag was introduced in v1.3 to mitigate token-stream
        # corruption from CONCURRENT generation across threads. The underlying
        # _generate_with_llm shares a single cached model, a single streamer,
        # and a single CUDA context, so parallel calls are undefined behavior.
        #
        # The ROADMAP hard rule is "Sequential execution only. ComfyUI manages
        # the queue." So we re-enable the evaluator in SEQUENTIAL mode - the
        # three outlines are generated one at a time through the loop below.
        # Per-outline budget is already tuned: 450 tok / 480s wall. Three
        # serial outlines at ~2 tok/s - 12 minutes worst case for OUTLINE mode,
        # under 3 minutes for PITCH mode. This is the cost of diversity: the
        # evaluator gets three genuinely different focuses (CHARACTER-DRIVEN,
        # SCIENCE-DRIVEN, ATMOSPHERE-DRIVEN) and picks the strongest one.
        #
        # Do NOT wrap the loop in a ThreadPoolExecutor. Parallel generation on
        # a shared Gemma model will corrupt the token streams - the same bug
        # this flag was originally put in place to prevent.
        ENABLE_3_OUTLINE_EVALUATOR = True

        if not ENABLE_3_OUTLINE_EVALUATOR:
            outline_focuses = [
                ("STORY-DRIVEN", "Focus on a balanced narrative arc, strong characters, and scientific plausibility.")
            ]
        else:
            _runtime_log(
                f"OPENCLOSE: 3-outline evaluator ACTIVE (sequential) - "
                f"{len(outline_focuses)} focuses: "
                f"{', '.join(name for name, _ in outline_focuses)}"
            )

        outlines = []
        for i, (focus_name, focus_desc) in enumerate(outline_focuses):
            if is_pitch_mode:
                # PITCH mode: lightweight 3-5 sentence logline per concept
                concept_body = f"""Generate a distinct story PITCH for a {style.replace("_", " ")} radio drama episode.

PRIORITY: {focus_name}
{focus_desc}

CRITICAL CONSTRAINTS:
- Exactly 3 to 5 sentences. 50-100 words total.
- High-level logline only. No act structure. No scene breakdown. No dialogue.
- Hook + core conflict + science angle.
- The science must be rooted in the real headlines from the system prompt above.

ARC TYPE: Use Arc Type {arc_choices[i]} from the Story Arc Engine above.
TARGET LENGTH (downstream script): {target_words} words
{lemmy_directive}

Begin your PITCH now:"""
            else:
                # SPINE mode: 7-line micro-variation protocol
                # Each line maps to a foundational dramatic function:
                # 1=Inciting Incident, 2=Protagonist Goal, 3=First Obstacle,
                # 4=Midpoint Twist, 5=Climax Prep, 6=Climax, 7=Epilogue
                concept_body = f"""Generate a 7-LINE STORY SPINE for a {style.replace("_", " ")} radio drama.

PRIORITY: {focus_name}
{focus_desc}

CRITICAL: The science news headlines in the system prompt above ARE your raw material. Your premise MUST be rooted in those real headlines - extrapolate the science to its most dramatic, terrifying, or profound next step.

ARC TYPE: Use Arc Type {arc_choices[i]} from the Story Arc Engine.
{cast_roster_block if cast_roster_block else f"CHARACTERS: {num_characters} speaking roles plus ANNOUNCER"}
TARGET LENGTH: {target_words} words
{lemmy_directive}

RULES:
- Output EXACTLY 7 numbered lines. No more, no fewer.
- No dialogue. No scene descriptions. Pure structural beats.
- Each line is ONE sentence describing WHAT HAPPENS.

FORMAT:
1. INCITING INCIDENT: [What disrupts the status quo - rooted in the real science headline]
2. PROTAGONIST GOAL: [What the lead character must achieve to resolve/contain the incident]
3. FIRST OBSTACLE: [Primary conflict, antagonistic force, or system failure driving tension]
4. MIDPOINT TWIST: [Reversal that changes everything - hidden pattern or critical new info]
5. CLIMAX PREPARATION: [Stakes are set for the final confrontation - consequences are clear]
6. CLIMAX: [Definitive resolution of the core conflict - earned, not ambiguous]
7. SCIENTIFIC EPILOGUE: [Real-world grounding - cite the actual science source]

Write your 7-line spine now:"""

            outline_prompt = f"{system}\n\n{concept_body}"

            try:
                outline_text = _run_with_timeout(
                    lambda op=outline_prompt: _generate_with_llm(
                        op,
                        model_id=model_id,
                        max_new_tokens=outline_max_tokens,
                        temperature=min(1.0, temperature + 0.1) if temperature < 1.0 else temperature,
                        optimization_profile=optimization_profile
                    ),
                    timeout_sec=480,   # was 300 - raised to 8min for SDPA @ ~2 tok/s
                    phase_label=f"OpenClose-{mode_label}-{focus_name}",
                )
                outlines.append((focus_name, outline_text))
                log.info("[OpenClose] %s %s generated (%d chars)",
                         mode_label, focus_name, len(outline_text))
                _runtime_log(f"OPENCLOSE: {mode_label} {focus_name} done ({len(outline_text)} chars)")
                
                # v1.5.1: Lightweight flush - clear KV cache fragments between spines
                # but keep LLM weights on GPU to avoid the ~13s reload penalty.
                _flush_vram_keep_llm()
            except Exception as e:
                log.warning("[OpenClose] %s %s failed: %s", mode_label, focus_name, e)
                outlines.append((focus_name, ""))
                _flush_vram_keep_llm()

        # -- Phase 2a: Open-Close boundary enforcement --
        # Discard outlines outside the mode-specific char range before evaluator.
        # OUTLINE_MIN / OUTLINE_MAX are set above based on pitch vs outline mode.
        valid_outlines = []
        for name, text in outlines:
            if not text or len(text) < OUTLINE_MIN:
                log.warning("[OpenClose] Outline %s too short (%d chars < %d) - discarded",
                            name, len(text) if text else 0, OUTLINE_MIN)
                _runtime_log(f"OPENCLOSE: DISCARDED {name} (too short: {len(text) if text else 0} chars)")
                continue
            if len(text) > OUTLINE_MAX:
                log.warning("[OpenClose] Outline %s too long (%d chars > %d) - truncating",
                            name, len(text), OUTLINE_MAX)
                text = text[:OUTLINE_MAX] + "\n[... outline truncated]"
                _runtime_log(f"OPENCLOSE: TRUNCATED {name} to {OUTLINE_MAX} chars")
            valid_outlines.append((name, text))
        if not valid_outlines:
            log.warning("[OpenClose] All outlines failed - falling back to direct generation")
            _runtime_log("OPENCLOSE: All outlines failed")
            return ""

        if len(valid_outlines) == 1:
            log.info("[OpenClose] Only 1 outline survived - using it directly")
            return valid_outlines[0][1]

        # -- EVALUATOR: pick the best outline --
        log.info("[OpenClose] Evaluating %d outlines...", len(valid_outlines))
        _runtime_log("OPENCLOSE: Evaluator picking winner")

        outlines_block = ""

        for idx, (name, text) in enumerate(valid_outlines, 1):
            outlines_block += f"\n--- {mode_label} {idx} ({name}) ---\n{text}\n"

        eval_prompt = f"""You are a veteran radio drama showrunner selecting the best story concept for production.

Below are {len(valid_outlines)} competing {mode_lower}s for a {style.replace("_", " ")} episode.

Evaluate each on:
1. HOOK STRENGTH: Would a listener stay past the first 30 seconds?
2. CHARACTER DEPTH: Do the characters feel real and distinct?
3. NARRATIVE ARC: Is there clear escalation, a satisfying climax, and earned resolution?
4. SCIENTIFIC PLAUSIBILITY: Is the science grounded or handwavy?
5. AUDIO POTENTIAL: Will this sound amazing as a radio drama? Strong SFX moments?
6. EAR FLOW: Does the premise lend itself to short, punchy, spoken-aloud dialogue (X Minus One / Suspense style)? Will lines be 5-15 words, rhythmic, easy to say in one breath? Reject outlines that imply long expository monologues or tongue-twister jargon.
7. DIALOGUE DENSITY (HARD FLOOR): The winner MUST support character-to-character conversation across the whole runtime. Reject any outline that reads like pure atmosphere, montage, or narrator-monologue - the downstream script must be a real dialogue drama, not a soundscape piece. If a candidate puts all the action into [SFX:] cues and leaves characters silent, penalize it heavily.

{outlines_block}

YOUR DECISION:
First, write ONE sentence about each {mode_lower}'s biggest strength and weakness.
Then state: "WINNER: {mode_label} N" (the number).
Finally, if elements from a losing {mode_lower} would strengthen the winner, list them as "MERGE: [element]".

Output the WINNING {mode_label} in full at the end, incorporating any merged elements.
Label it "FINAL {mode_label}:" on its own line before the text."""

        try:
            eval_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    eval_prompt,
                    model_id=model_id,
                    max_new_tokens=800,
                    temperature=max(0.3, temperature - 0.3),
                    optimization_profile=optimization_profile
                ),
                timeout_sec=300,
                phase_label="OpenClose-Evaluator",
            )
            log.info("[OpenClose] Evaluator complete (%d chars)", len(eval_text))
            _runtime_log(f"OPENCLOSE: Evaluator done ({len(eval_text)} chars)")
            
            # v1.5.1: Lightweight flush - keep LLM on GPU for story editor.
            _flush_vram_keep_llm()
        except Exception as e:
            log.warning("[OpenClose] Evaluator failed: %s - using first outline", e)
            _flush_vram_keep_llm()
            return valid_outlines[0][1]

        # Extract the final concept from evaluator output.
        # Marker is mode-specific: "FINAL PITCH:" or "FINAL OUTLINE:". Try the
        # current mode first, then the other (in case the model picked the wrong
        # header), then the generic fallbacks.
        for marker in (f"FINAL {mode_label}:", "FINAL OUTLINE:", "FINAL PITCH:"):
            marker_idx = eval_text.upper().find(marker.upper())
            if marker_idx >= 0:
                winning = eval_text[marker_idx + len(marker):].strip()
                log.info("[OpenClose] Extracted winning %s via marker '%s' (%d chars)",
                         mode_lower, marker, len(winning))
                return winning

        # If no marker found, try to find "WINNER:" and return corresponding concept
        winner_match = re.search(
            rf'WINNER:\s*(?:{mode_label}|Outline|Pitch)\s*(\d)',
            eval_text, re.IGNORECASE,
        )
        if winner_match:
            winner_idx = int(winner_match.group(1)) - 1
            if 0 <= winner_idx < len(valid_outlines):
                log.info("[OpenClose] Winner is %s %d (%s)",
                         mode_label, winner_idx + 1, valid_outlines[winner_idx][0])
                return valid_outlines[winner_idx][1]

        # Fallback: return the full evaluator output (it usually contains a merged outline)
        log.info("[OpenClose] No clean marker found - using full evaluator output as outline")
        return eval_text

    # -------------------------------------------------------------------------
    # AUTO-TITLE FROM SPINE - one small LLM call between OpenClose winner
    # selection and ScriptWriter draft, so the title is grounded in the
    # actual spine instead of a user-typed placeholder. Added 2026-04-26.
    # -------------------------------------------------------------------------

    def _generate_title_from_spine(self, *, winning_outline, style,
                                   news_block,
                                   model_id, temperature,
                                   optimization_profile="Standard"):
        """Generate a 2-5 word evocative episode title from the winning spine.

        Runs ONE small LLM call after the OpenClose evaluator picks a winner
        but before the ScriptWriter draft fires. Result is used as the
        episode_title for the rest of the pipeline (prompt frame, ledger
        episode_id, downstream announcer bookends).

        Returns the cleaned title string, or "" on any failure (caller
        falls back to the existing user / LLM-emitted / derived chain).
        """
        if not winning_outline or not winning_outline.strip():
            return ""

        # Trim the news block - we only need the first headline (max 200
        # chars) as a tonal cue. Bare string, no prefix; the prompt template
        # below frames it as "Tone hint (optional): ...".
        _news_hint = ""
        if news_block:
            _first_news = news_block.strip().split("\n")[0][:200]
            if _first_news:
                _news_hint = _first_news

        # Prompt v2 (2026-04-26 PM): genre-agnostic, craft-focused. Lets the
        # LLM pick up genre from the spine itself rather than spelling it out,
        # which keeps the title from sliding into on-the-nose genre tropes.
        title_prompt = (
            "You are creating a title for a single episode of a radio drama.\n\n"
            f"Tone hint (optional): {_news_hint.strip() or '(none)'}\n\n"
            f"Episode spine: {winning_outline[:1200]}\n\n"
            "Write ONE evocative episode title in 2-5 words.\n\n"
            "The title must:\n"
            " - draw from a vivid image, key object, character, or thematic "
            "tension in the spine\n"
            " - match the implied genre and tone (do not name the genre "
            "explicitly)\n"
            " - feel specific and memorable, not generic or placeholder\n"
            " - avoid cliches like \"The Beginning\", \"Final Chapter\", "
            "\"Untitled\", or \"Episode X\"\n"
            " - avoid repeating obvious words from the spine unless used in "
            "a fresh or symbolic way\n\n"
            "Style guidance:\n"
            " - prefer concrete nouns + subtle intrigue\n"
            " - light ambiguity is good; confusion is not\n"
            " - aim for something that could sit on a vintage radio listing "
            "or modern podcast feed\n\n"
            "Output ONLY the title text on a single line. Nothing else."
        )
        # style / style are still accepted as parameters for
        # logging + future style-pinning, but no longer baked into the prompt.

        try:
            raw = _run_with_timeout(
                lambda: _generate_with_llm(
                    title_prompt,
                    model_id=model_id,
                    max_new_tokens=24,
                    temperature=max(0.4, min(1.0, temperature)),
                    optimization_profile=optimization_profile,
                ),
                timeout_sec=60,
                phase_label="AutoTitle-Spine",
            )
        except Exception as _err:
            log.warning("[AutoTitle] LLM call failed: %s", _err)
            return ""

        if not raw:
            return ""

        # Take first non-empty line, strip junk.
        candidate = ""
        for ln in raw.strip().split("\n"):
            ln = ln.strip()
            if ln:
                candidate = ln
                break
        if not candidate:
            return ""

        # Strip "Title:" / "**" / smart-quote / asterisk wrappers.
        # Handles "**Title:** Pulse", "Title: \"Pulse\"", "**Pulse**", etc.
        candidate = re.sub(
            r'^\s*(?:\*\*)?\s*(?:TITLE|Title)\s*:\s*(?:\*\*)?\s*',
            '', candidate
        )
        # Iteratively strip wrapping whitespace, asterisks, ASCII / smart
        # quotes, single quotes from BOTH ends until the value is stable.
        _wrap_chars = '"“”‘’*\' \t'
        prev = None
        while candidate != prev:
            prev = candidate
            candidate = candidate.strip(_wrap_chars)
        if not candidate:
            return ""

        # Reject stuck defaults and obviously-too-long output (full sentence
        # leaking through from the model).
        if candidate.lower() in _STUCK_TITLE_DEFAULTS:
            log.info("[AutoTitle] Rejected stuck default: %r", candidate)
            return ""
        if len(candidate.split()) > 10:
            log.info("[AutoTitle] Rejected overlong title (%d words): %r",
                     len(candidate.split()), candidate)
            return ""

        log.info("[AutoTitle] Spine-derived title: %r", candidate)
        return candidate

    # -------------------------------------------------------------------------
    # CHECKS & CRITIQUES - Draft -> Critique -> Revise
    # -------------------------------------------------------------------------

    def _run_critique_only(self, draft_text, style, target_words,
                           model_id, temperature, optimization_profile="Standard"):
        """Critique-only pass for long scripts (>3 acts).

        Runs the same structural critique as _critique_and_revise Pass 2,
        but SKIPS Pass 3 (global rewrite) to prevent summarization collapse.
        The critique findings are returned as text and stored on self so the
        Arc Enhancer can incorporate them into its opening/closing polish.

        Returns the critique text, or empty string on failure.
        """
        log.info("[Critique] Starting critique-only pass (no rewrite - long script protection)")
        _runtime_log("CRITIQUE_ONLY: Generating structural analysis")

        # Truncate for critique context - keep first 4000 + last 4000 chars
        # to see beginning AND ending without blowing the context window.
        draft_for_critique = draft_text
        if len(draft_text) > 8000:
            draft_for_critique = (
                draft_text[:4000]
                + "\n\n[... MIDDLE ACTS OMITTED FOR BREVITY ...]\n\n"
                + draft_text[-4000:]
            )

        critique_prompt = f"""You are a HARSH but constructive script editor for a {style.replace("_", " ")} radio drama.

Below is a multi-act draft script. Your job is to identify SPECIFIC weaknesses. Do NOT rewrite anything.

Output a numbered list of 5-8 concrete problems, each one sentence. Focus on:
1. OPENING HOOK: Does the first 30 seconds grab the listener? Is the announcer's intro compelling?
2. STORY ARC: Does tension rise across acts? Is the climax earned? Does anything feel skipped?
3. CHARACTER VOICE: Do characters sound distinct from each other? Or interchangeable?
4. DIALOGUE QUALITY: Natural spoken English? 5-15 words per line? Contractions used?
5. ENDING PAYOFF: Does the closing connect back to the opening? Is the epilogue grounded in real science?
6. PACING: Any dead spots or rushed sections between acts?
7. AUDIO DESIGN: Are [SFX:] and [ENV:] tags specific and atmospheric?
8. START-TO-END COHERENCE: Does the final act honor the promises made in Act 1?

Be brutal. Be specific. Name the exact act or line that's weak.
Do NOT include any script text in your response - critique ONLY.

DRAFT SCRIPT:
{draft_for_critique}

YOUR CRITIQUE (numbered list only):"""

        try:
            critique_tokens = min(600, max(200, len(draft_text) // 25))
            critique_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    critique_prompt,
                    model_id=model_id,
                    max_new_tokens=critique_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=180,
                phase_label="Critique-Only",
            )
            log.info("[Critique] Critique-only pass complete (%d chars)", len(critique_text))
            _runtime_log(f"CRITIQUE_ONLY: Complete ({len(critique_text)} chars)")

            # Validate it looks like a critique, not a rewrite
            _markers = re.findall(r'^\s*\d+[\.)\:]', critique_text, re.MULTILINE)
            if len(_markers) < 2:
                log.warning("[Critique] Critique-only output doesn't look like a numbered list - discarding")
                return ""

            return critique_text
        except Exception as e:
            log.warning("[Critique] Critique-only pass failed: %s", e)
            _runtime_log(f"CRITIQUE_ONLY: Failed - {e}")
            return ""

    @staticmethod
    def _normalize_voice_format_to_standard(text):
        """Convert ULTRA_SMOKE-style ``[VOICE: NAME, attrs, ...]: text`` lines
        into the canonical ``NAME: text`` format that ``_count_character_lines``
        and the rest of the critique/revise pipeline expect.

        BUG-LOCAL-027 extension (2026-05-03 EVENING, Jeffrey directive): the
        ULTRA_SMOKE preset's PARSE_OK validator counts ``[VOICE: ...]`` markers
        as voice_hits, but ``_count_character_lines`` only matches bare
        ``CHARNAME:`` and ``[N] CHARNAME:`` formats. Without normalization, the
        critique-pipeline counter returns ``{}`` for ULTRA_SMOKE drafts and the
        total-collapse hard gate has no signal to enforce the dialogue floor.

        Per Jeffrey 2026-05-03: "ULTRA_SMOKE need to abide by all the rules".
        Normalizing here means ULTRA_SMOKE goes through the SAME critique +
        gate machinery as the standard short(3 acts) path -- one source of
        truth for dialogue preservation.

        Two transformations:

        1. **Standalone VOICE-prefix lines.** Lines of the form
           ``[VOICE: NAME, attr, attr, ...]: text`` become ``NAME: text``.
           Captures the speaker name (first comma-separated token after
           ``[VOICE:``) and treats the rest as voice metadata to discard.

        2. **Inline VOICE blocks within dialogue.** Lines that already start
           with ``[N] CHARNAME:`` or ``CHARNAME:`` but contain a trailing
           ``[VOICE: ...]`` block in the dialogue text get the inline VOICE
           block stripped (Bark / Kokoro don't speak voice metadata aloud).

        Idempotent: text that is already in standard format passes through
        unchanged. C7-safe: deterministic regex transformation of the same
        input always produces the same output.
        """
        if not text:
            return text

        # Pattern 1: standalone [VOICE: NAME, ...]: text -> NAME: text
        # Capture the first identifier after [VOICE:; treat everything up to
        # the closing ] as voice metadata.
        _voice_line_pat = re.compile(
            r'^(\s*)\[VOICE:\s*([A-Z][A-Z0-9_ \-]*?)(?:\s*,[^\]]*)?\]\s*:\s*(.*)$'
        )
        # Pattern 2: inline [VOICE: ...] block to strip from within a dialogue
        # line that already starts with a CHARNAME: marker.
        _inline_voice_pat = re.compile(r'\s*\[VOICE:[^\]]*\]\s*:?\s*')

        out_lines = []
        for line in text.split('\n'):
            m = _voice_line_pat.match(line)
            if m:
                indent, name, content = m.group(1), m.group(2).strip(), m.group(3)
                out_lines.append(f"{indent}{name}: {content}")
                continue
            # Already standard or no VOICE prefix -- strip any inline VOICE
            # blocks that appear after the colon (don't break the structure).
            stripped = _inline_voice_pat.sub(' ', line)
            out_lines.append(stripped)
        return '\n'.join(out_lines)

    @staticmethod
    def _count_character_lines(text):
        """Count dialogue lines per character in script text.

        Returns a dict {CHARACTER_NAME: line_count}. Matches the pattern
        'NAME: dialogue' where NAME is uppercase letters/digits/underscores/spaces.
        Excludes structural tokens: TITLE, SCENE, ACT, SFX, ENV, MUSIC, BEAT,
        PAUSE, NARRATOR, SYSTEM_SENTINEL. ANNOUNCER is counted as a character.

        Args:
            text: Script text to analyze.

        Returns:
            Dict mapping character names to dialogue line counts.
        """
        if not text:
            return {}

        # Structural tokens to exclude (but not ANNOUNCER - it's a real character)
        _struct_exclude = frozenset([
            "TITLE", "SCENE", "ACT", "SFX", "ENV", "MUSIC", "BEAT",
            "PAUSE", "NARRATOR", "SYSTEM_SENTINEL"
        ])

        # Pattern: NAME: dialogue (uppercase name, optional parenthetical emotion,
        # colon, then content). Allow optional asterisks (character emphasis).
        # BUG-LOCAL-027 fix (2026-05-03): also accept the writer's actual
        # ``[N] CHARNAME: dialogue`` numbered-bracket format. The original
        # regex required the line to START with the uppercase name, so any
        # script with the ``[12] FLETCHER WELLS:`` prefix returned ``{}``
        # for both draft and revised — the per-character preservation gate
        # at line ~7174 then iterated an empty dict and accepted any
        # dialogue-stripped revision. Three runs on 2026-05-03 between
        # 22:00 and 00:16 shipped revisions with zero character lines
        # because of this. The new optional non-capturing group
        # ``(?:\[\d+\]\s+)?`` makes the prefix optional so BOTH
        # ``CHARNAME:`` and ``[N] CHARNAME:`` formats parse correctly.
        pattern = r'^\s*(?:\[\d+\]\s+)?\*{0,2}([A-Z][A-Z0-9_ ]+?)\*{0,2}\s*(?:\([^)]*\))?\s*:'

        character_counts = {}
        for line in text.split('\n'):
            match = re.match(pattern, line)
            if match:
                char_name = match.group(1).strip()
                # BUG-LOCAL-027 fix (2026-05-03): exclude structural tokens
                # by EXACT match OR first-word match, so "ACT 2", "SCENE 3",
                # "MUSIC theme" etc all get filtered. Prior to this the
                # exact-match exclude let "ACT 2:" line headers count as a
                # character, which inflated draft_total and could cause the
                # total-collapse gate to misfire.
                first_word = char_name.split()[0] if char_name else ""
                if char_name not in _struct_exclude and first_word not in _struct_exclude:
                    character_counts[char_name] = character_counts.get(char_name, 0) + 1

        return character_counts

    def _critique_and_revise(self, draft_text, style, target_words,
                             model_id, temperature, optimization_profile="Standard",
                             min_line_count_per_character=2):
        """Three-pass refinement: the LLM critiques its own draft, then revises.

        Pass 1 (already done): Draft generation (the script_text we received).
        Pass 2 (Critique):     LLM acts as a harsh script editor. Outputs a
                               numbered improvement plan - NO rewriting.
        Pass 3 (Revision):     LLM receives draft + critique, rewrites the
                               script implementing the specific fixes.

        Args:
            min_line_count_per_character: Minimum dialogue lines required per character
                in the revised script. Revisions that drop any character below this
                threshold are rejected. Default 2.

        Returns the revised script text, or the original draft if critique
        fails or produces nothing useful.
        """
        log.info("[Critique] Starting Checks & Critiques loop (Draft -> Critique -> Revise)...")
        _runtime_log("CRITIQUE: Starting self-critique pass")

        # BUG-LOCAL-027 extension (2026-05-03 EVENING, Jeffrey directive):
        # Normalize ULTRA_SMOKE-style ``[VOICE: NAME, attrs, ...]: text`` lines
        # into canonical ``NAME: text`` BEFORE the critique pass runs. This
        # ensures the critique LLM, the per-character preservation gate, AND
        # the total-collapse hard gate all see the same canonical format
        # regardless of whether the writer was in ULTRA_SMOKE mode or standard
        # short(3 acts) mode. Prior to this normalization, ULTRA_SMOKE drafts
        # parsed as ``draft={}`` in the counter and the gate was a no-op.
        # Idempotent on already-standard text; C7-safe (deterministic).
        _pre_norm_len = len(draft_text)
        draft_text = self._normalize_voice_format_to_standard(draft_text)
        if len(draft_text) != _pre_norm_len:
            _runtime_log(
                f"CRITIQUE: ULTRA_SMOKE format normalized "
                f"({_pre_norm_len} -> {len(draft_text)} chars)"
            )

        # -- Truncate draft for critique context --
        # Keep the full draft but cap at ~12k chars to stay within context window.
        # The critique doesn't need every word - it needs the structure and flow.
        draft_for_critique = draft_text
        if len(draft_text) > 12000:
            # Keep first 6000 + last 6000 so critique sees beginning AND ending
            draft_for_critique = (
                draft_text[:6000]
                + "\n\n[... MIDDLE SECTION OMITTED FOR BREVITY ...]\n\n"
                + draft_text[-6000:]
            )

        # -- Pass 2: CRITIQUE --
        critique_prompt = f"""You are a HARSH but constructive script editor for a {style.replace("_", " ")} radio drama.

Below is a draft script. Your job is to identify SPECIFIC weaknesses. Do NOT rewrite anything.

Output a numbered list of 5-8 concrete problems, each one sentence. Focus on:
1. STORY ARC: Does it have a clear hook, rising tension, climax, and resolution? Or does it meander?
2. CHARACTER: Do characters sound distinct? Do they have clear motivations? Or are they interchangeable talking heads?
3. DIALOGUE: Does it sound like real humans under pressure? Or stilted/expository?
4. PACING: Are there dead spots? Does tension build or stay flat?
5. SCIENCE: Is the science grounded in real physics/biology? Any obvious handwaving?
6. ENDING: Does the resolution feel earned or rushed? Does the epilogue connect to the story?
7. AUDIO DESIGN: Are [SFX:] and [ENV:] tags used effectively to build atmosphere? Or sparse/generic?
8. EAR TEST (CRITICAL): Read every line aloud in your head. Does it sound like natural spoken English a real person would say in 5-15 words? Flag any line that is: longer than 15 words, full of jargon, missing contractions, or reads like written prose instead of speech. Flag any character name that is hard to say aloud or longer than 2 syllables.

Be brutal. Be specific. Name the exact scene or line that's weak.
Do NOT include any script text in your response - critique ONLY.

DRAFT SCRIPT:
{draft_for_critique}

YOUR CRITIQUE (numbered list only):"""

        try:
            critique_tokens = min(800, max(300, len(draft_text) // 20))
            critique_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    critique_prompt,
                    model_id=model_id,
                    max_new_tokens=critique_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=300,
                phase_label="Critique-Pass",
            )
            log.info("[Critique] Critique pass complete (%d chars)", len(critique_text))
            _runtime_log(f"CRITIQUE: Critique pass done ({len(critique_text)} chars)")
        except Exception as e:
            log.warning("[Critique] Critique pass failed: %s - returning original draft", e)
            _runtime_log(f"CRITIQUE: Failed - {e}")
            return f"{draft_text}\n\n[SYSTEM_SENTINEL: TIMEOUT_FALLBACK]"

        # Sanity check: critique should be a numbered list, not a rewrite
        if not critique_text or len(critique_text) < 50:
            log.warning("[Critique] Critique too short (%d chars) - skipping revision",
                        len(critique_text) if critique_text else 0)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED - critique too short")
            return draft_text

        # -- Phase 2c: Critique format validation --
        # Verify the critique looks like a numbered list, not a rewrite
        _critique_markers = re.findall(r'^\s*\d+[\.\):]', critique_text, re.MULTILINE)
        _critique_keywords = sum(1 for kw in ["weak", "issue", "problem", "flat", "generic",
                                               "missing", "rushed", "unclear", "improve"]
                                 if kw in critique_text.lower())
        if len(_critique_markers) < 2 and _critique_keywords < 2:
            log.warning("[Critique] Critique doesn't look like a numbered list "
                        "(%d markers, %d keywords) - may be a rewrite, skipping revision",
                        len(_critique_markers), _critique_keywords)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED - critique format invalid")
            return draft_text

        # -- Pass 3: REVISION --
        log.info("[Critique] Starting revision pass with %d-char critique...", len(critique_text))
        _runtime_log("CRITIQUE: Starting revision pass")

        revision_prompt = f"""You are the original writer of this {style.replace("_", " ")} radio drama script.
A tough editor has reviewed your draft and provided specific critique.

YOUR TASK: Rewrite the COMPLETE script, implementing every critique point below.
Keep everything that already works. Fix only what the editor flagged.

ABSOLUTE REQUIREMENT — DIALOGUE MUST SURVIVE THE REVISION:
The revised script MUST contain CHARACTER dialogue lines. Producing a script with only SCENE/ENV/SFX/MUSIC scaffolding and zero spoken character lines is a TOTAL FAILURE — the radio drama becomes silent narration. Every CHARACTER speaker present in the draft MUST appear in the revision. You may rewrite their lines for sharper dialogue, emotional grounding, or pacing — but you may NEVER delete a character's voice entirely. If you find yourself writing only ENV: and SFX: tags with no CHARACTER: lines, STOP and re-include the dialogue.

RULES:
- Output the FULL revised script - not a summary, not highlights, the COMPLETE script.
- CRITICAL: Every spoken line MUST use the format 'CHARACTER_NAME: dialogue text' (all caps name, colon, space, then dialogue). Also preserve [SFX:], [ENV:], (beat), === SCENE N === tags. The optional line-number prefix '[N]' from the draft (e.g. '[12] FLETCHER WELLS: ...') may be kept or omitted — both formats parse correctly.
- Do NOT add new characters unless the critique specifically demands it.
- Do NOT change character names.
- Do NOT remove the ANNOUNCER opening or closing epilogue.
- Keep the same approximate length (~{target_words} words).
- Make dialogue sharper, more natural, more emotionally grounded.
- Strengthen the story arc wherever the critique identifies weakness.
- CRITICAL: Do NOT reduce any character below {min_line_count_per_character} dialogue lines. Every character present in the draft must still appear with at least {min_line_count_per_character} lines in the revision.

EDITOR'S CRITIQUE:
{critique_text}

ORIGINAL DRAFT:
{draft_text}

REVISED SCRIPT (complete, from === SCENE 1 === to [MUSIC: Closing theme]):"""

        try:
            # FIX-1 (v1.2): Size revision budget from DRAFT LENGTH, not target_words.
            # Previously used target_words*2.0 which gave ~2080 tokens for 8-min eps -
            # but an 8-min draft runs ~10k chars (~2500 tokens), so the revision pass
            # got decapitated mid-Scene 4. Scene 4 is where the ending lives, which is
            # why every critique flagged "weak ending". Not a writing bug - a budget bug.
            # Formula: draft_chars / 3.5 chars-per-token * 1.25 safety margin.
            draft_token_estimate = int(len(draft_text) / 3.5)
            revision_tokens = max(int(draft_token_estimate * 1.25), int(target_words * _TOKEN_RATIO_MIXED), 2048)
            revision_tokens = min(revision_tokens, 8192)
            log.info("[Critique] Revision token budget: %d (draft_est=%d, target_words=%d)",
                     revision_tokens, draft_token_estimate, target_words)
            # BUG-005 fix: scale wall-clock budget to episode length AND draft size.
            # SDPA on 4-expert MoE models runs ~2-3 tok/s, so a 22k-char revision needs
            # ~700-1100s. The previous fixed 600s killed every long episode.
            # Formula: max(600, target_words/2.3, len(draft)*0.05)
            revision_timeout = int(max(
                600,
                target_words / 2.3,  # ~60s per 140 words
                len(draft_text) * 0.05,
            ))
            log.info("[Critique] Revision wall-clock budget: %ds (target_words=%d, draft=%d chars)",
                     revision_timeout, target_words, len(draft_text))
            revised_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    revision_prompt,
                    model_id=model_id,
                    max_new_tokens=revision_tokens,
                    temperature=temperature,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=revision_timeout,
                phase_label="Revision-Pass",
            )
            log.info("[Critique] Revision pass complete (%d chars)", len(revised_text))
            _runtime_log(f"CRITIQUE: Revision done ({len(revised_text)} chars)")
        except Exception as e:
            log.warning("[Critique] Revision pass failed: %s - returning original draft", e)
            _runtime_log(f"CRITIQUE: Revision failed - {e}")
            return f"{draft_text}\n\n[SYSTEM_SENTINEL: TIMEOUT_FALLBACK]"

        # BUG-LOCAL-027 extension: normalize the revised text too, so the
        # downstream gate counter sees consistent format regardless of what
        # variant the revision LLM emitted. The revision prompt asks for
        # ``CHARACTER_NAME: dialogue`` but the model may slip back into
        # ``[VOICE: ...]`` shape under high-temp creativity.
        _pre_norm_revised_len = len(revised_text)
        revised_text = self._normalize_voice_format_to_standard(revised_text)
        if len(revised_text) != _pre_norm_revised_len:
            _runtime_log(
                f"CRITIQUE: revised text VOICE-format normalized "
                f"({_pre_norm_revised_len} -> {len(revised_text)} chars)"
            )

        # -- Phase 2b: Critique length & format guardrails --

        # BUG-LOCAL-085 guard: if the initial draft pass produced 0
        # chars (LLM emitted nothing, prompt-truncation edge case,
        # token-boundary issue, etc.), accept whatever the revision
        # produced rather than crashing on division-by-zero in the
        # diagnostic log lines below. Without this guard the whole
        # script-writer node aborts even though the revision pass
        # successfully produced a complete script.
        if len(draft_text) == 0:
            log.warning(
                "[Critique] Empty draft (0 chars) -- accepting revision "
                "of length %d as the script.",
                len(revised_text),
            )
            _runtime_log(
                f"CRITIQUE: Empty draft, using revision ({len(revised_text)} chars)"
            )
            return revised_text

        # Check 1: Revision must be at least 60% of draft length (not a summary)
        if len(revised_text) < len(draft_text) * 0.6:
            log.warning(
                "[Critique] Revision too short (%d chars vs %d draft) - "
                "LLM may have summarized instead of rewriting. Keeping original draft.",
                len(revised_text), len(draft_text)
            )
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED - revision too short")
            return draft_text

        # Check 2: Revision must not exceed 250% of draft length (runaway expansion)
        if len(revised_text) > len(draft_text) * 2.5:
            log.warning(
                "[Critique] Revision too long (%d chars vs %d draft, %.0f%%) - "
                "LLM expanded beyond acceptable bounds. Keeping original draft.",
                len(revised_text), len(draft_text),
                len(revised_text) / len(draft_text) * 100
            )
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED - revision too long (%.0f%%)" %
                         (len(revised_text) / len(draft_text) * 100))
            return draft_text

        # Check 3: Levenshtein similarity ratio - catch both lazy copies and hallucinations
        # Use simple character overlap ratio (fast approximation of edit distance)
        def _char_overlap_ratio(a, b):
            """Fast character-level similarity: shared chars / max length."""
            if not a or not b:
                return 0.0
            from collections import Counter
            ca, cb = Counter(a.lower()), Counter(b.lower())
            shared = sum((ca & cb).values())
            return shared / max(len(a), len(b))

        similarity = _char_overlap_ratio(draft_text, revised_text)
        _runtime_log(f"CRITIQUE: Similarity ratio: {similarity:.3f}")

        if similarity > 0.95:
            log.warning("[Critique] Revision too similar to draft (%.1f%% overlap) - "
                        "LLM likely copied instead of revising. Keeping original draft.",
                        similarity * 100)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED - revision is a copy (%.1f%%)" % (similarity * 100))
            return draft_text

        if similarity < 0.35:
            log.warning("[Critique] Revision too different from draft (%.1f%% overlap) - "
                        "LLM may have hallucinated a new story. Keeping original draft.",
                        similarity * 100)
            _runtime_log("CRITIQUE: CRITIQUE_SKIPPED - revision is a hallucination (%.1f%%)" % (similarity * 100))
            return draft_text

        # Check 4: Character line count preservation - ensure no character drops below minimum
        draft_char_counts = self._count_character_lines(draft_text)
        revised_char_counts = self._count_character_lines(revised_text)

        _runtime_log(f"CRITIQUE: Character line counts - draft={draft_char_counts} revised={revised_char_counts}")

        # For each character in draft with >= min_lines, verify it still meets the floor in revision
        for char_name, draft_count in draft_char_counts.items():
            if draft_count >= min_line_count_per_character:
                revised_count = revised_char_counts.get(char_name, 0)
                if revised_count < min_line_count_per_character:
                    log.warning(
                        "[Critique] Character '%s' dropped from %d to %d lines (floor=%d) - "
                        "revision violates character preservation constraint. Keeping original draft.",
                        char_name, draft_count, revised_count, min_line_count_per_character
                    )
                    _runtime_log(f"CRITIQUE: CRITIQUE_REJECTED - character '{char_name}' dropped from {draft_count} to {revised_count} lines (floor={min_line_count_per_character})")
                    return draft_text

        # BUG-LOCAL-027 hard total-collapse gate (2026-05-03): belt-and-
        # suspenders for the per-character check above. The per-character
        # loop catches "FLETCHER dropped from 8 to 1"; this catches "every
        # character wiped at once" (the actual failure mode observed on
        # 2026-05-03 — three soak runs returned revisions with zero
        # character lines because the model under temp=0.95 padded the
        # output with SCENE/ENV/SFX prose and dropped every CHARACTER:
        # line). Threshold: revised total must be >= 50% of draft total
        # whenever the draft had >= 3 character lines. Below 3 lines the
        # draft itself is too short to apply a meaningful ratio — let the
        # per-character check (with min_line_count_per_character floor)
        # handle that case.
        import math as _math
        draft_total = sum(draft_char_counts.values())
        revised_total = sum(revised_char_counts.values())
        if draft_total >= 3:
            min_revised = max(1, _math.ceil(draft_total * 0.5))
            if revised_total < min_revised:
                log.warning(
                    "[Critique] Total character-line count collapsed from %d "
                    "to %d (minimum %d, threshold=50%% of draft). Revision "
                    "appears to be SCENE/ENV/SFX-only — keeping original draft.",
                    draft_total, revised_total, min_revised,
                )
                _runtime_log(
                    f"CRITIQUE: CRITIQUE_REJECTED - total character lines "
                    f"collapsed from {draft_total} to {revised_total} "
                    f"(min={min_revised}, threshold=50%%)"
                )
                return draft_text

        log.info("[Critique] Checks & Critiques complete - revised script accepted "
                 "(similarity=%.1f%%, length ratio=%.0f%%).",
                 similarity * 100, len(revised_text) / len(draft_text) * 100)
        _runtime_log("CRITIQUE: Revised script accepted (sim=%.1f%%, len=%.0f%%)" %
                     (similarity * 100, len(revised_text) / len(draft_text) * 100))
        return revised_text

    def _generate_chunked(self, system, title, style, num_chars,
                          target_words, premise, news_block, act_breaks,
                          model_id, temperature, target_length="medium (5 acts)",
                          lemmy_directive="", top_p=0.95,
                          cast_roster_block="", optimization_profile="Standard"):
        """Generate long scripts act-by-act to avoid token truncation.

        Step 1: Generate an outline (characters, plot beats, act structure)
        Step 2: Generate each act using the outline + previous act as context
        Step 3: Concatenate into the final script
        """
        # v1.5 FIX: Respect the target_length widget for act counts
        # Map: short=3, medium=5, long=8, epic=12
        _act_map = {
            "30 words (smoke, 1 act)": 1,
            "tiny (smoke, 1 act)": 1,
            "short (3 acts)":  3,
            "medium (5 acts)": 5,
            "long (7-8 acts)": 8,
            "epic (10+ acts)": 12
        }
        num_acts = _act_map.get(target_length, 5)
        
        # v1.5 FIX: Increased inflation factor to 1.5 (from 1.2).
        # Gemma/Nemo aggressively summarize if not pushed. 1.5x target ensures
        # that even with 'lazy' generation, we land near the user's intent.
        inflated_target = int(target_words * 1.5)
        words_per_act = inflated_target // num_acts

        # Step 1: Outline
        outline_prompt = f"""{system}

Create a detailed OUTLINE for a {target_words}-word episode of "SIGNAL LOST."
Title: {title}
Style: {style.replace("_", " ")}
Characters: {num_chars} speaking roles plus ANNOUNCER
{cast_roster_block}
{lemmy_directive}

Return:
- Character list: name, role, gender, personality, and what they PERSONALLY have at stake (~50/50 male/female split)
- Time period and setting (derived from the science news)
- {num_acts}-act structure: inciting incident, escalation beats, twist/resolution - focus on HUMAN drama, not science exposition
- At least one moment of humor, warmth, or unexpected humanity
- The ANNOUNCER's hard-science epilogue topic and sources to cite
- Key SFX and music cues

STORY ARC SEED: Use Arc Type {random.choice("ABCDEFGH")} from the Story Arc Engine. Commit fully to that structure.

Remember: This is a DRAMA that happens to involve science, not a science report with characters. Give every character something personal to lose.

{"Premise: " + premise if premise else "The news headlines ARE the premise. Extrapolate the science into its most dramatic next step."}

Outline only - do NOT write dialogue yet."""

        # BUG-011 FIX: Reduce KV Cache allocation overhead.
        # Outline is instructed to be under 400 words. max_new_tokens=1500 pre-allocates
        # excessive KV cache, which immediately overflows the 4GB ceiling and causes 100% GPU
        # PCIe memory thrashing (0.1 tok/sec behavior, appearing as a hang).
        outline_budget = 600 if optimization_profile == "Obsidian (UNSTABLE/4GB)" else 1200
        
        # BUG-021 FIX: Cap outline temperature at 0.7. The outline is the
        # structural skeleton -- character names, act structure, plot beats.
        # Under maximum chaos the outline hallucinates names and sprawling
        # act counts that infect every downstream act. Creative acts can
        # improvise wildly on top of a solid outline.
        _outline_temp = min(temperature, 0.7)
        _outline_top_p = min(top_p, 0.95)
        log.info(f"[ScriptWriter] Generating outline ({num_acts} acts) [KV Budget: {outline_budget}] [temp={_outline_temp}]")
        outline = _generate_with_llm(outline_prompt, model_id=model_id,
                                         max_new_tokens=outline_budget, temperature=_outline_temp, top_p=_outline_top_p,
                                         optimization_profile=optimization_profile)

        # Step 2: Generate each act with Context Engineering
        # Instead of dumping raw previous text, we summarize what happened
        # and signpost key character states for continuity.
        acts = []
        act_summaries = []  # Running narrative memory

        # -- Step 1b: CRITIQUE THE OUTLINE (v1.5 - Story Editor) ----------
        # Before writing ANY dialogue, have the LLM critique its own outline.
        # This catches structural weaknesses BEFORE they infect the acts.
        # The critique generates per-act briefs that guide each act's writing.
        # Key insight from research: critique guides writing, not patches it.
        outline_critique = ""
        act_briefs = {}  # {act_num: "brief for what this act should accomplish"}
        
        try:
            # v1.5.1: Lightweight flush - keep LLM on GPU for critique.
            _flush_vram_keep_llm()
            
            _runtime_log("STORY_EDITOR: Critiquing outline before act generation")
            _brief_lines = []
            for n in range(1, num_acts + 1):
                _brief_lines.append(f"ACT {n} BRIEF: [What Act {n} must accomplish dramatically - 1-2 sentences]")
            _brief_format = "\n".join(_brief_lines)
            
            editor_prompt = f"""You are a veteran radio drama story editor. Below is an outline for a {num_acts}-act episode.

OUTLINE:
{_truncate_at_sentence_boundary(outline, 2000)}

YOUR TASK: Briefly critique this outline, then write a 1-2 sentence BRIEF for each act describing what it must accomplish dramatically.

FORMAT YOUR RESPONSE EXACTLY AS:
CRITIQUE: [2-3 sentences identifying the outline's biggest weakness and how to fix it]

{_brief_format}

QUALITY TARGETS:
- Each brief should specify the EMOTIONAL STATE characters should be in
- Each brief should name a KEY DRAMATIC MOMENT that must happen
- Each brief should note any SFX or atmosphere cues that would enhance the scene"""
            
            editor_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    editor_prompt,
                    model_id=model_id,
                    max_new_tokens=min(600, 80 * num_acts),
                    temperature=0.3,
                    top_p=0.9,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=120,
                phase_label="Story-Editor",
            )
            
            # v1.5.1: Lightweight flush - keep LLM on GPU for act generation.
            _flush_vram_keep_llm()
            
            # Parse act briefs from the editor text
            critique_match = re.search(r'CRITIQUE:\s*(.+?)(?=ACT \d+ BRIEF:)', editor_text, re.DOTALL | re.IGNORECASE)
            if critique_match:
                outline_critique = critique_match.group(1).strip()
                _runtime_log(f"STORY_EDITOR: Critique: {outline_critique[:120]}")
            
            # Extract per-act briefs
            for act_n in range(1, num_acts + 1):
                brief_match = re.search(
                    rf'ACT {act_n} BRIEF:\s*(.+?)(?=ACT \d+ BRIEF:|$)',
                    editor_text, re.DOTALL | re.IGNORECASE
                )
                if brief_match:
                    act_briefs[act_n] = brief_match.group(1).strip()[:300]
            
            _runtime_log(f"STORY_EDITOR: Generated {len(act_briefs)} act briefs")
            log.info("[StoryEditor] Outline critique complete: %d chars, %d act briefs",
                     len(outline_critique), len(act_briefs))
            
            # Store critique for downstream Arc Enhancer
            self._last_critique_findings = outline_critique
            
        except Exception as _editor_err:
            log.warning("[StoryEditor] Story editor pass failed: %s - continuing without briefs", _editor_err)
            _runtime_log(f"STORY_EDITOR: Failed - {_editor_err}")

        for act_num in range(1, num_acts + 1):
            # S29: Allow users to cancel long script generation
            try:
                import comfy.model_management
                comfy.model_management.throw_exception_if_processing_interrupted()
            except ImportError:
                pass

            # -- Context Engineering: curated memory instead of raw dump --
            if acts:
                # Summarize previous act for tight context (not raw 2000 chars)
                if not act_summaries:
                    # Generate a quick summary of Act 1 for Act 2's context
                    _act_text_for_summary = _truncate_at_sentence_boundary(acts[-1], 3000)
                    summary_prompt = f"""Summarize the following radio drama act in 3-5 sentences.
Focus on: what happened, how each character's emotional state changed, what's at stake going into the next act, and any unresolved tensions.
Do NOT include dialogue. Just narrative summary.

ACT TEXT:
{_act_text_for_summary}

SUMMARY:"""
                    try:
                        summary = _generate_with_llm(
                            summary_prompt, model_id=model_id,
                            max_new_tokens=200, temperature=0.3,
                            optimization_profile=optimization_profile
                        )
                        act_summaries.append(summary)
                        _runtime_log(f"ScriptWriter: Act {act_num-1} summarized for context")
                    except Exception:
                        # Fallback: sentence-boundary truncation (no mid-sentence cuts)
                        act_summaries.append(_truncate_at_sentence_boundary(acts[-1], 1500))
                else:
                    # Summarize the latest act and append to running memory
                    _act_text_for_summary = _truncate_at_sentence_boundary(acts[-1], 3000)
                    summary_prompt = f"""Summarize the following radio drama act in 3-5 sentences.
Focus on: what happened, how each character's emotional state changed, what's at stake going into the next act, and any unresolved tensions.

ACT TEXT:
{_act_text_for_summary}

SUMMARY:"""
                    try:
                        summary = _generate_with_llm(
                            summary_prompt, model_id=model_id,
                            max_new_tokens=200, temperature=0.3,
                            optimization_profile=optimization_profile
                        )
                        act_summaries.append(summary)
                    except Exception:
                        act_summaries.append(_truncate_at_sentence_boundary(acts[-1], 1500))

                # -- Phase 3a: Chunked context hardening --
                # Validate each summary - if too short, fall back to mechanical summary
                for s_idx in range(len(act_summaries)):
                    if len(act_summaries[s_idx].strip()) < 50:
                        log.warning("[ContextEng] Act %d summary too short (%d chars) - using mechanical fallback",
                                    s_idx + 1, len(act_summaries[s_idx]))
                        # Mechanical fallback: scene titles + last 8 dialogue lines
                        act_lines = acts[s_idx].strip().splitlines()
                        scene_titles = [l.strip() for l in act_lines if "===" in l]
                        dialogue_lines = [l.strip() for l in act_lines if "[VOICE:" in l][-8:]
                        act_summaries[s_idx] = (
                            "Scenes: " + "; ".join(scene_titles) + "\n"
                            "Key dialogue: " + " / ".join(dialogue_lines)
                        )[:800]

                # Build signposted context: all summaries + last 500 chars of raw text
                context_block = "STORY SO FAR (summaries of previous acts):\n"
                for s_idx, s_text in enumerate(act_summaries, 1):
                    context_block += f"  Act {s_idx}: {s_text.strip()}\n"

                # -- Phase 3b: Sentence-boundary tail (v1.4 Theme B) --
                # Walks forward from the cut point to the next sentence start so
                # the Gemma prompt never sees a tail that begins mid-word.
                last_lines = _tail_at_sentence_boundary(acts[-1], 500)
                if len(acts[-1]) > 500:
                    last_lines = "... [truncated]\n" + last_lines
                context_block += f"\nLAST LINES (for dialogue continuity):\n{last_lines}"
            else:
                context_block = "(beginning of episode)"

            # v1.5 FIX: Truncate outline for later acts to reduce KV cache pressure.
            # Acts 1-2 get the full outline; Acts 3+ get a compressed version.
            act_outline = outline if act_num <= 2 else _truncate_at_sentence_boundary(outline, 800)

            # v1.5: Build Story Editor guidance block for this act
            editor_guidance = ""
            act_brief = act_briefs.get(act_num, "")
            if act_brief or outline_critique:
                editor_guidance = "\nSTORY EDITOR GUIDANCE:\n"
                if outline_critique:
                    editor_guidance += f"Overall note: {outline_critique[:200]}\n"
                if act_brief:
                    editor_guidance += f"THIS ACT must accomplish: {act_brief}\n"

            act_prompt = f"""You are writing Act {act_num} of {num_acts} for a radio drama called "SIGNAL LOST".

OUTLINE:
{act_outline}
{editor_guidance}
{context_block}

Now write ACT {act_num} of {num_acts} in full script format.
Target: ~{words_per_act} words for this act. 
STRICT REQUIREMENT: Focus on deep character reactions and atmospheric descriptions. If you run out of plot, expand the dialogue with conflicting emotions and technical disagreements. Do NOT summarize. Do NOT skip any plot points. Write every single beat in full dialogue form. Every character must have space to breathe and react.
{"This is the OPENING - start with [MUSIC: Opening theme] and ANNOUNCER setting time/place/characters. Then drop us IN MEDIAS RES." if act_num == 1 else ""}
{"This is the FINAL ACT - build to the twist, then ANNOUNCER delivers the hard-science epilogue. CITATION RULE: cite ONLY the real article provided in the news block above - its exact source name and date. NEVER use numbered references like [1], [2], article #N - always say the source name directly (e.g. 'According to Science Daily, published April 3, 2026...'). Do NOT invent ArXiv IDs or paper titles. End with [MUSIC: Closing theme]." if act_num == num_acts else ""}
{"Include an act break marker [ACT " + str(act_num + 1) + "] at the end of this act." if act_breaks and act_num < num_acts else ""}

CONTINUITY CHECK: Before writing, review the story-so-far summaries above. Ensure characters reference earlier events naturally. No amnesia - people remember what just happened to them.

Write Act {act_num} now:"""


            _runtime_log(f"ScriptWriter: Generating Act {act_num}/{num_acts}")
            
            # v2.0: Content-aware act token budget using _TOKEN_RATIO constants.
            # Standard ceiling raised from 2048 to 4096 to prevent silent truncation
            # on long acts (epic 10+ act episodes). Obsidian stays at 2048 for VRAM.
            if optimization_profile == "Obsidian (UNSTABLE/4GB)":
                act_budget = min(2048, int(words_per_act * _TOKEN_RATIO_ACT_OBSIDIAN))
            else:
                act_budget = max(1024, min(4096, int(words_per_act * _TOKEN_RATIO_ACT_CHUNK)))
                
            act_text = _generate_with_llm(act_prompt, model_id=model_id,
                                              max_new_tokens=act_budget, temperature=temperature, top_p=top_p,
                                              optimization_profile=optimization_profile)
            acts.append(act_text)

            # v1.5.1: Lightweight flush - keep LLM on GPU between acts.
            # Full model eviction here was causing ~13s reload per act (up to 8 acts).
            _flush_vram_keep_llm()
            _runtime_log(f"ScriptWriter: Act {act_num} VRAM flushed (lightweight -- LLM retained)")

        # v1.5: Store act summaries for the Arc Enhancer to use when
        # polishing the opening/closing. These are richer than the plot spine
        # the Arc Enhancer extracts on its own.
        self._last_act_summaries = act_summaries

        # BUG-021 FIX: Strip hallucinated act markers beyond the target count.
        # Under maximum chaos, the LLM may inject [ACT N] markers for acts
        # beyond num_acts within a single act's output text. Remove them so
        # downstream parsers see the correct act structure.
        combined = "\n\n".join(acts)
        for phantom_act in range(num_acts + 1, num_acts + 20):
            combined = re.sub(
                rf'\[ACT\s+{phantom_act}\]',
                '',
                combined,
                flags=re.IGNORECASE
            )
        _hallucinated = len(re.findall(r'\[ACT\s+\d+\]', combined, re.IGNORECASE))
        _runtime_log(f"ScriptWriter: Act marker cleanup done ({num_acts} target, {_hallucinated} markers remain)")

        return combined

    def _execute_arc_enhancer(self, script_text, style, title, news_block, model_id, temperature, optimization_profile="Standard", critique_findings="", act_summaries=None):
        """Phase A-C: Paired opening + closing bookend rewrite for narrative coherence.
        
        v1.5: Now accepts optional critique_findings and act_summaries.
        When present, these give the bookend rewriter a complete picture of
        the story's structure so the opening and closing mesh perfectly.
        """
        _runtime_log("ARC_EN_HANCER: Starting structural coherence pass")
        original_script_backup = script_text

        # Phase A: Extraction
        bookends = self._get_bookends(script_text)
        if not bookends:
            _runtime_log("ARC_ENHANCER: Failed to extract bookends - skipping pass")
            return script_text

        opening_orig, closing_orig = bookends

        # Phase A: Structural Coherence Scoring (Observability)
        arc_score, arc_checks = self._score_arc_coherence(opening_orig, closing_orig, script_text)
        checks_str = ", ".join(f"{k}={v}" for k, v in arc_checks.items())
        _runtime_log(f"ARC_ENHANCER: Arc score: {arc_score}/5 ({checks_str})")

        # Plot Spine Injection: extract middle-act summary so Phase B rewrite
        # honors the journey instead of hallucinating contradictions.
        plot_spine = self._extract_plot_spine(script_text, opening_orig, closing_orig)

        # v1.4 Theme B - surface the spine in the runtime log so the showrunner
        # can see exactly what the bookend rewriter was told about the middle.
        _runtime_log(f"ARC_ENHANCER: Plot spine: {plot_spine[:150]}")

        # Phase A score floor flag - if score < 3/5, the first Phase B pass will
        # be followed by one automatic retry. The retry threshold is the same
        # contract used by tests/vram_profile_test.py for the arc coherence check.
        _arc_retry_warranted = arc_score < 3

        # Phase B: Architectural Echo call
        # We use a lower temperature (0.6) for tighter structural alignment
        # v1.5: Inject critique findings + act summaries if available
        critique_block = ""
        if critique_findings:
            critique_block += f"""\nEDITOR CRITIQUE (address these weaknesses in your rewrite):
{critique_findings[:800]}
"""
            _runtime_log(f"ARC_ENHANCER: Injecting {len(critique_findings)} chars of critique findings")
        
        # v1.5: If act summaries are available from chunked generation,
        # they provide a richer story picture than the extracted plot spine.
        act_summary_block = ""
        if act_summaries:
            act_summary_block = "\nACT-BY-ACT JOURNEY (use this to ensure opening seeds and closing payoffs match the actual story):\n"
            for s_idx, s_text in enumerate(act_summaries, 1):
                act_summary_block += f"  Act {s_idx}: {s_text.strip()}\n"
            _runtime_log(f"ARC_ENHANCER: Injecting {len(act_summaries)} act summaries for start/end coherence")

        echo_prompt = f"""You are a structural script editor for the radio drama anthology "SIGNAL LOST".
YOUR TASK: Rewrite the OPENING and CLOSING dialogue blocks below to create a "narrative echo".

DIRECTIONS:
1. Plant a NARRATIVE SEED in the Opening Block. This can be a cryptic mention of an object, a specific fear, a recurring sound cue, or a foreshadowed choice.
2. Harvest the PAYOFF in the Closing Block. The seed MUST resolve, pivot, or be explained in a way that provides emotional or structural closure to the episode.
3. Preserve the CHARACTER NAMES and VOICES exactly as they appear in the original text.
4. Preserve all CANONICAL TAGS ([VOICE:], [SFX:], [ENV:], (beat)) exactly.
5. Do NOT change the meaning of the science headline context provided.
6. Do NOT contradict the MIDDLE EVENTS summary below. The closing must honor what happened in the middle of the story - no resurrected characters, no forgotten revelations, no reversed outcomes.
7. Return ONLY the rewritten blocks inside the XML tags below.

STYLE: {style.replace("_", " ")}
TITLE: {title}
SCIENCE CONTEXT: {news_block}

MIDDLE EVENTS (do not contradict):
{plot_spine}
{act_summary_block}{critique_block}
ORIGINAL OPENING BLOCK:
{opening_orig}

ORIGINAL CLOSING BLOCK:
{closing_orig}

Format your response exactly as:
<opening>
[Revised Opening Block]
</opening>
<closing>
[Revised Closing Block]
</closing>"""

        try:
            echo_response = _run_with_timeout(
                lambda: _generate_with_llm(
                    echo_prompt,
                    model_id=model_id,
                    max_new_tokens=1000,
                    temperature=0.6,
                    optimization_profile=optimization_profile
                ),
                timeout_sec=300,
                phase_label="Arc-Enhancer-Echo",
            )

            # Phase C: Injection + Echo Phrase Extraction
            try:
                opening_new = echo_response.split("<opening>")[1].split("</opening>")[0].strip()
                closing_new = echo_response.split("<closing>")[1].split("</closing>")[0].strip()

                if opening_new and closing_new:
                    # Extract echo phrase: find longest common noun between opening and closing rewrite
                    opening_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', opening_new))
                    closing_nouns = set(re.findall(r'\b[A-Z][a-z]+\b', closing_new))
                    echo_phrase = list(opening_nouns & closing_nouns)[0] if (opening_nouns & closing_nouns) else "(no direct echo)"

                    # Safe replacement for opening
                    script_text = script_text.replace(opening_orig, opening_new, 1)

                    # Safe replacement for closing (work from the end to avoid collisions)
                    parts = script_text.rsplit(closing_orig, 1)
                    if len(parts) == 2:
                        script_text = parts[0] + closing_new + parts[1]

                    _runtime_log(f"ARC_ENHANCER: Pass 1 complete (echo phrase = {echo_phrase})")

                    # v1.4 Theme B - Phase A score floor retry.
                    # If the initial arc score was below the 3/5 floor, run a
                    # second Phase B+C pass using the already-injected script as
                    # the new base. One retry only - more would drift the text.
                    if _arc_retry_warranted:
                        _runtime_log(
                            f"ARC_ENHANCER: Score was {arc_score}/5 (below 3/5 floor) "
                            f"- triggering retry pass"
                        )
                        retry_bookends = self._get_bookends(script_text)
                        if retry_bookends:
                            opening_retry, closing_retry = retry_bookends
                            retry_spine = self._extract_plot_spine(
                                script_text, opening_retry, closing_retry
                            )
                            retry_prompt = echo_prompt.replace(
                                f"ORIGINAL OPENING BLOCK:\n{opening_orig}",
                                f"ORIGINAL OPENING BLOCK:\n{opening_retry}",
                            ).replace(
                                f"ORIGINAL CLOSING BLOCK:\n{closing_orig}",
                                f"ORIGINAL CLOSING BLOCK:\n{closing_retry}",
                            ).replace(
                                f"{plot_spine}",
                                f"{retry_spine}",
                            )
                            try:
                                retry_response = _run_with_timeout(
                                    lambda: _generate_with_llm(
                                        retry_prompt,
                                        model_id=model_id,
                                        max_new_tokens=1000,
                                        temperature=0.6,
                                        optimization_profile=optimization_profile
                                    ),
                                    timeout_sec=300,
                                    phase_label="Arc-Enhancer-Retry",
                                )
                                opening_r = retry_response.split("<opening>")[1].split("</opening>")[0].strip()
                                closing_r = retry_response.split("<closing>")[1].split("</closing>")[0].strip()
                                if opening_r and closing_r:
                                    script_text = script_text.replace(opening_retry, opening_r, 1)
                                    parts_r = script_text.rsplit(closing_retry, 1)
                                    if len(parts_r) == 2:
                                        script_text = parts_r[0] + closing_r + parts_r[1]
                                    # Re-score so the log tells the truth about the
                                    # final state, not just the initial state.
                                    retry_score, retry_checks = self._score_arc_coherence(
                                        opening_r, closing_r, script_text
                                    )
                                    retry_checks_str = ", ".join(
                                        f"{k}={v}" for k, v in retry_checks.items()
                                    )
                                    _runtime_log(
                                        f"ARC_ENHANCER: Pass 2 complete "
                                        f"arc_score={retry_score}/5 ({retry_checks_str})"
                                    )
                                else:
                                    _runtime_log("ARC_ENHANCER: Retry returned empty tags - keeping pass 1 result")
                            except Exception as retry_err:
                                log.warning("[ArcEnhancer] Retry pass failed: %s", retry_err)
                                _runtime_log(f"ARC_ENHANCER: Retry failed - keeping pass 1 result ({retry_err})")
                        else:
                            _runtime_log("ARC_ENHANCER: Retry skipped - could not re-extract bookends after pass 1")
                    else:
                        _runtime_log(
                            f"ARC_ENHANCER: Score {arc_score}/5 meets floor - no retry needed"
                        )
                else:
                    _runtime_log("ARC_ENHANCER: LLM returned empty tags - skipping injection")
            except (IndexError, ValueError):
                log.warning("[ArcEnhancer] Failed to parse XML tags from echo response")
                _runtime_log("ARC_ENHANCER: Parsing error - response format invalid")

        except Exception as e:
            log.warning("[ArcEnhancer] Phase B/C pass failed: %s", e)
            _runtime_log(f"ARC_ENHANCER: Failed - {e} (reverting to original)")
            # Revert to raw LLM output to prevent len(text)=1 crash
            return original_script_backup

        # v1.4 Theme B - automatic scene transition injection.
        # Runs regardless of how Phase B/C fared so even a failed arc pass
        # still gets the structural handoff benefit for downstream audio.
        try:
            script_text, _transition_count = _inject_scene_transitions(script_text)
            if _transition_count > 0:
                _runtime_log(
                    f"ARC_ENHANCER: Injected {_transition_count} scene transition "
                    f"cue(s) at weak handoffs"
                )
            else:
                _runtime_log("ARC_ENHANCER: No weak scene handoffs detected")
        except Exception as transition_err:
            log.warning("[ArcEnhancer] Scene transition injection failed: %s", transition_err)
            _runtime_log(f"ARC_ENHANCER: Transition injection failed - {transition_err}")

        return script_text

    def _score_arc_coherence(self, opening_text, closing_text, script_text):
        """Phase A: Structural coherence check - 5-point scoring for narrative completeness."""
        score = 0
        checks = {}

        # Check 1: Truncation detector - does closing end mid-sentence?
        closing_lines = closing_text.strip().split('\n')
        last_line = closing_lines[-1].strip() if closing_lines else ""
        terminal_chars = {'.', '!', '?', '"'}
        # Pass if last line ends with terminal char AND not a connective word
        last_word = last_line.split()[-1].rstrip('.,!?;:"') if last_line.split() else ""
        connective_words = {'the', 'and', 'to', 'a', 'an', 'or', 'but', 'as', 'is', 'of', 'in', 'be'}
        checks['truncation'] = (bool(last_line) and
                                any(last_line.endswith(c) for c in terminal_chars) and
                                last_word.lower() not in connective_words)
        if checks['truncation']:
            score += 1

        # Check 2: Weak final scene - count [VOICE:] tags, need -4 lines
        voice_count = len(re.findall(r'\[VOICE:', closing_text, re.IGNORECASE))
        checks['strong_scene'] = voice_count >= 4
        if checks['strong_scene']:
            score += 1

        # Check 3: Premise payoff - any capitalized keyword overlap (opening - closing)
        opening_caps = set(re.findall(r'\b[A-Z][a-z]+\b', opening_text))
        closing_caps = set(re.findall(r'\b[A-Z][a-z]+\b', closing_text))
        checks['payoff'] = len(opening_caps & closing_caps) > 0
        if checks['payoff']:
            score += 1

        # Check 4: Tonal echo - repeated words (>4 chars) between opening and closing
        opening_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', opening_text))
        closing_words = set(w.lower() for w in re.findall(r'\b\w{4,}\b', closing_text))
        checks['echo'] = len(opening_words & closing_words) >= 2
        if checks['echo']:
            score += 1

        # Check 5: Epilogue presence - ANNOUNCER in final 500 chars
        epilogue_region = script_text[-500:] if len(script_text) > 500 else script_text
        checks['epilogue'] = 'ANNOUNCER' in epilogue_region
        if checks['epilogue']:
            score += 1

        return score, checks

    def _extract_plot_spine(self, script_text, opening_orig, closing_orig):
        """Extract a ~50-word middle-act summary so Phase B rewrites honor continuity.

        Pulls dialogue and scene headers from the region BETWEEN the opening and
        closing blocks, then truncates to ~50 words. This gives the Phase B rewriter
        knowledge of the middle acts without bloating the token budget.
        """
        # Find the middle region (everything between opening and closing)
        open_end = script_text.find(opening_orig)
        close_start = script_text.rfind(closing_orig)

        if open_end == -1 or close_start == -1 or close_start <= open_end:
            return "(middle events unavailable)"

        middle_region = script_text[open_end + len(opening_orig):close_start]

        # Extract scene markers and voice lines, strip formatting for a clean spine
        spine_parts = []
        for raw_line in middle_region.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            # Keep scene markers as structural anchors
            scene_match = re.match(r'===\s*SCENE\s+(\d+)\s*===', line, re.IGNORECASE)
            if scene_match:
                spine_parts.append(f"[Scene {scene_match.group(1)}]")
                continue
            # Extract dialogue content from voice tags
            voice_match = re.match(r'\[VOICE:\s*([^,\]]+)[^\]]*\]\s*(.+)$', line, re.IGNORECASE)
            if voice_match:
                speaker = voice_match.group(1).strip()
                dialogue = voice_match.group(2).strip()
                spine_parts.append(f"{speaker}: {dialogue}")

        # Truncate to ~50 words to keep Phase B prompt lean (~60 tokens)
        full_spine = " ".join(spine_parts)
        words = full_spine.split()
        if len(words) > 50:
            full_spine = " ".join(words[:50]) + "..."

        return full_spine if full_spine else "(middle events unavailable)"

    def _get_bookends(self, script_text):
        """Extract opening and closing dialogue blocks for the coherence pass."""
        # --- 1. OPENING BLOCK ---
        # Find Scene 1
        scene1_match = re.search(r'===\s*SCENE\s+1\s*===', script_text, re.IGNORECASE)
        if not scene1_match:
            return None

        # Focus on the first ~25 lines of Scene 1 to find dialogue
        body_start = script_text[scene1_match.end():]
        # Find all Voice tags in the first 4000 chars of Scene 1
        voices = list(re.finditer(r'\[VOICE:', body_start[:4000], re.IGNORECASE))
        if len(voices) < 4:
            return None

        # Opening block: from first voice to end of 8th voice (or last available)
        v_count = min(len(voices), 8)
        target_v = voices[v_count - 1]
        line_end = body_start.find("\n", target_v.end())
        if line_end == -1: line_end = len(body_start)
        opening_block = body_start[:line_end].strip()

        # --- 2. CLOSING BLOCK ---
        # Find the last scene (climax)
        # We look for the last SCENE marker before the EPILOGUE or Closing Music
        end_marker = re.search(r'===\s*EPILOGUE\s*===|\[MUSIC:\s*Closing theme\]', script_text, re.IGNORECASE)
        climax_boundary = end_marker.start() if end_marker else len(script_text)

        climax_area = script_text[:climax_boundary]
        scenes = list(re.finditer(r'===\s*SCENE\s+\d+\s*===', climax_area, re.IGNORECASE))
        if not scenes:
            return None

        last_scene_body = climax_area[scenes[-1].end():]
        # Find voice tags in the last scene
        last_voices = list(re.finditer(r'\[VOICE:', last_scene_body, re.IGNORECASE))
        if len(last_voices) < 3:
            return None

        # Closing block: pull at most the last 8 dialogue lines
        v_count_climax = min(len(last_voices), 8)
        start_idx = last_voices[-v_count_climax].start()
        closing_block = last_scene_body[start_idx:].strip()

        # Sanity check: ensure these blocks actually exist in the text (for later replace)
        if opening_block in script_text and closing_block in script_text:
            return opening_block, closing_block

        return None

    # False-positive tag names that look like CHARACTER: but aren't speakers.
    # Shared between skip heuristic and chunk dialogue counting.
    _FORMAT_NORM_NON_CHARS = {
        "SCENE", "ACT", "NOTE", "TARGET", "STYLE", "SFX",
        "ENV", "NARRATOR", "OPENING", "CLOSING", "MUSIC",
    }

    def _normalize_script_format(self, script_text, model_id, optimization_profile="Standard"):
        """Creative-to-Strict pass: reformat any dialogue style into Canonical 1.0.

        Uses the same LLM (already loaded in VRAM) at low temperature to
        rewrite the script into strict format. This prevents PARSE_FATAL when
        the creative pass produces non-standard dialogue formatting.

        Routing:
          - Tightened skip heuristic: only skip when dialogue AND scenes AND
            cast are ALL present. This catches "ghost runs" where max chaos
            produces parseable-looking dialogue but no CAST or scene markers.
          - Single-pass for short scripts (<=50 dialogue lines OR <2 scenes).
          - Chunked by scene marker for long scripts, like the Grammarian.

        Returns the normalized script text, or the original if normalization fails.
        """
        # ----- Count every structural signal ------------------------------
        voice_tag_count = len(re.findall(r'\[VOICE:', script_text, re.IGNORECASE))
        _all_canonical = re.findall(
            r'^([A-Z][A-Z0-9_ ]{1,25}):\s+.+$', script_text, re.MULTILINE
        )
        canonical_count = sum(
            1 for name in _all_canonical
            if name.strip() not in self._FORMAT_NORM_NON_CHARS
        )
        unique_chars = {
            name.strip() for name in _all_canonical
            if name.strip() not in self._FORMAT_NORM_NON_CHARS
        }
        scene_count = len(re.findall(
            r'===\s*SCENE\s+\S+.*?===', script_text, re.IGNORECASE
        ))
        total_dialogue = canonical_count + voice_tag_count

        # ----- Tightened skip heuristic -----------------------------------
        # Previously: skipped if dialogue count was "enough" regardless of
        # whether CAST or scene markers existed. That silently let ghost
        # runs (Run 011/012) bypass FORMAT_NORM entirely. Now ALL three
        # structural elements must be present before we skip.
        #
        # BUG-LOCAL-038 refinement: require real [VOICE:] tags, not bare
        # canonical_count. Mistral's native output is `NAME: dialogue` with
        # no bracket tag -- that format matches _FORMAT_NORM_NON_CHARS less
        # and looks canonical to this counter, but _parse_script's four VOICE
        # patterns do NOT accept it. Skipping FORMAT_NORM on bare NAME: runs
        # is how the dialogue tokens vanish before BatchBark.
        has_dialogue  = (voice_tag_count >= 3)
        has_scenes    = scene_count >= 1
        # 2026-04-29: relaxed cast assertion to >= 1 to support 1-character
        # monologue mode. Previously a 1-char (announcer + single voice)
        # script would be flagged as needing FORMAT_NORM rescue even when
        # dialogue + scenes were present, triggering an unnecessary cleanup
        # pass that could mangle the legitimate single-character output.
        has_cast      = len(unique_chars) >= 1

        if has_dialogue and has_scenes and has_cast:
            _runtime_log(
                f"FORMAT_NORM: Skipped - dialogue={canonical_count}+"
                f"{voice_tag_count}V, scenes={scene_count}, "
                f"cast={len(unique_chars)} (all present)"
            )
            return script_text

        _runtime_log(
            f"FORMAT_NORM: Running (dialogue={canonical_count}+"
            f"{voice_tag_count}V, scenes={scene_count}, "
            f"cast={len(unique_chars)}) - "
            f"missing: "
            f"{'dialogue ' if not has_dialogue else ''}"
            f"{'scenes ' if not has_scenes else ''}"
            f"{'cast' if not has_cast else ''}"
        )

        # ----- Route: chunked for long scripts, single for short ----------
        _CHUNK_THRESHOLD = 50  # dialogue lines before chunking
        if total_dialogue > _CHUNK_THRESHOLD and scene_count >= 2:
            return self._normalize_chunked(
                script_text, model_id, optimization_profile,
                canonical_count, voice_tag_count,
            )
        return self._normalize_single_pass(
            script_text, model_id, optimization_profile,
            canonical_count, voice_tag_count,
        )

    # -------------------------------------------------------------------------
    # FORMAT_NORM: single-pass and chunked implementations
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_normalize_prompt(script_text, *, is_segment=False):
        """Shared strict-normalizer prompt. Same rules for full script and
        per-scene chunk; chunked variant adds a segment-awareness preamble."""
        preamble = (
            "The text below is ONE SEGMENT of a longer script. Preserve "
            "every scene marker verbatim and do not merge or split scenes.\n"
            if is_segment else ""
        )
        return f"""You are a strict script normalizer. Your ONLY task is to reformat input text into the exact canonical format defined below.
{preamble}

HARD CONSTRAINTS:
- Do NOT add, remove, summarize, or rewrite ANY dialogue or content.
- Do NOT infer or guess any missing text.
- Do NOT paraphrase.
- Only transform formatting.
- Output plain text ONLY.
- Do NOT use Markdown, code blocks, or quotes.
- If something is unclear or malformed, preserve it as-is but normalize its formatting.

CANONICAL FORMAT RULES:

1. STRIP FORMATTING
- Remove ALL Markdown symbols (such as *, **, _, `, etc.).
- Remove ALL quotation marks around dialogue.

2. CHARACTER NAMES
- Convert all character names to ALL CAPS.
- Replace any underscores in names with spaces.
- Standardize the name NARRATOR to ANNOUNCER.
- Do NOT rename any other characters.

3. DIALOGUE STRUCTURE (STRICT)
Every dialogue line MUST be in exactly ONE of these two output forms:
  [VOICE: CHARACTER NAME, traits] dialogue text
  CHARACTER NAME: dialogue text
Accepted INPUT forms that you may need to convert:
  [CHARACTER NAME, traits] dialogue text  ->  rewrite to [VOICE: CHARACTER NAME, traits] dialogue text
  **CHARACTER NAME:** dialogue text        ->  rewrite to CHARACTER NAME: dialogue text
  CHARACTER NAME (emotion): dialogue text  ->  rewrite to CHARACTER NAME: (emotion) dialogue text
Rules:
- For the CHARACTER NAME: format, use a colon only (never hyphens or other separators), followed by exactly one space.
- For the [VOICE: ...] format, the dialogue text MUST follow immediately after the closing bracket with exactly one space and NO colon.
- Never drop or invent dialogue during conversion - only rewrite the tag shape.

4. STAGE DIRECTIONS / EMOTIONS
If a line contains emotional cues such as:
  NAME, angrily: dialogue
  NAME (angry): dialogue
Then move the emotion into parentheses at the START of the dialogue text:
  CHARACTER NAME: (angrily) dialogue
Rules:
- Do NOT invent new emotions.
- Only relocate emotions that are already present in the text.

5. TAGS (STRICT)
Only the following tags are allowed, and they MUST appear on their own line:
  [SFX: description]
  [ENV: description]
  [MUSIC: description]
  [ACT TWO], [ACT THREE], etc. (act break markers -- word-form numbers, no colon)
Rules:
- Tags MUST be uppercase.
- Tags with descriptions (SFX, ENV, MUSIC) MUST use a colon after the tag name.
- Act break markers use word-form numbers with NO colon: [ACT TWO] not [ACT: 2].
- Normalize malformed tags (for example: sfx-, Sound:, etc.) into one of the allowed forms above.
- Convert any unsupported scene/visual tags (for example: [VFX: ...], [LIGHTING: ...], [CAMERA: ...]) into [ENV: ...] with the same description.
- Preserve act break markers exactly as-is. Do NOT convert them to scene headers or ENV tags.

6. SCENE AND ACT HEADERS
Scene headers: Format EXACTLY as === SCENE N: Title ===
Act break markers: Format EXACTLY as [ACT TWO], [ACT THREE], etc. (on their own line)
Rules:
- Preserve the scene number N if it exists in the input.
- If numbering is missing in the input, keep the original scene title text but apply the header format without inventing a new number.
- Do NOT invent or remove scene numbers, act numbers, or titles; only normalize their formatting.
- Scenes and acts are different structures. Do NOT merge or convert one into the other.

7. ERROR NORMALIZATION
- Fix inconsistent casing (for example, tag names, character names that are obviously the same).
- Fix spacing (remove extra spaces, enforce required single spaces as specified).
- Replace incorrect separators (such as hyphens or equals signs used instead of colons) with the correct ones as defined above.
- Do NOT delete malformed or unclear content -- normalize it while preserving all original text.

FINAL RULE:
Output ONLY the normalized script. No explanations. No extra text. No commentary.

SCRIPT TO REFORMAT:
{script_text}"""

    def _normalize_single_pass(self, script_text, model_id,
                               optimization_profile,
                               canonical_count, voice_tag_count):
        """Single-pass FORMAT_NORM for short scripts (under the chunk threshold).

        BUG-019 token budget preserved: min(1024, max(256, len//4)) keeps the
        LLM from runaway filler while giving enough room for the reformatted
        output. Timeout 75s - if reformatting can't finish in that window,
        the script is too long and should have been chunked instead.
        """
        normalize_prompt = self._build_normalize_prompt(
            script_text, is_segment=False
        )

        _norm_max_tokens = min(1024, max(256, len(script_text) // 4))
        _runtime_log(
            f"FORMAT_NORM: Single-pass | token budget={_norm_max_tokens} | "
            f"input chars={len(script_text)}"
        )

        try:
            normalized = _run_with_timeout(
                lambda: _generate_with_llm(
                    normalize_prompt,
                    model_id=model_id,
                    max_new_tokens=_norm_max_tokens,
                    temperature=0.3,
                    optimization_profile=optimization_profile,
                ),
                timeout_sec=75,
                phase_label="FormatNorm",
            )

            if not normalized or len(normalized.strip()) < len(script_text) * 0.3:
                _runtime_log(
                    f"FORMAT_NORM: Output too short "
                    f"({len(normalized or '')} chars vs {len(script_text)} "
                    f"input) - keeping original"
                )
                return script_text

            # Verify normalization improved dialogue detection
            new_canonical = len(re.findall(
                r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$', normalized, re.MULTILINE
            ))
            new_voice = len(re.findall(r'\[VOICE:', normalized, re.IGNORECASE))

            if new_canonical + new_voice > canonical_count + voice_tag_count:
                _runtime_log(
                    f"FORMAT_NORM: Success - {new_canonical} canonical + "
                    f"{new_voice} VOICE tags (was {canonical_count} + "
                    f"{voice_tag_count})"
                )
                return normalized.strip()
            else:
                _runtime_log(
                    f"FORMAT_NORM: No improvement ({new_canonical} canonical "
                    f"vs {canonical_count}) - keeping original"
                )
                return script_text

        except Exception as e:
            log.warning("[FormatNorm] Normalization pass failed: %s", e)
            _runtime_log(f"FORMAT_NORM: Failed ({e}) - keeping original")
            return script_text

    def _normalize_chunked(self, script_text, model_id,
                           optimization_profile,
                           canonical_count, voice_tag_count):
        """Chunked FORMAT_NORM for long scripts (50+ dialogue lines, 2+ scenes).

        Mirrors _grammarian_chunked: split by === SCENE N === markers,
        normalize each scene independently with a full per-chunk token budget,
        then reassemble. This fixes the ghost-run class of bug where a long
        script silently bailed out via "Output too short" because the 1024
        token cap could not cover the entire reformatted script.

        Each chunk has its own timeout and safety checks; failed chunks keep
        their original text rather than blocking the rest.
        """
        _runtime_log(
            f"FORMAT_NORM: Chunked mode | canonical={canonical_count}, "
            f"voice={voice_tag_count} dialogue lines"
        )

        # ----- Split by scene markers ---------------------------------
        scene_re = re.compile(
            r'(===\s*SCENE\s+\S+.*?===)', re.IGNORECASE
        )
        parts = scene_re.split(script_text)

        # Reassemble: each chunk = marker + body until next marker.
        chunks = []
        current_chunk = ""
        for part in parts:
            if scene_re.match(part):
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part
        if current_chunk.strip():
            chunks.append(current_chunk)

        # Fall back to line-based chunking if scene split produced only 1.
        if len(chunks) <= 1:
            _runtime_log(
                "FORMAT_NORM: No scene markers found - using line-based chunking"
            )
            all_lines = script_text.split('\n')
            chunk_size = 40
            chunks = [
                '\n'.join(all_lines[i:i + chunk_size])
                for i in range(0, len(all_lines), chunk_size)
            ]

        _runtime_log(f"FORMAT_NORM: Split into {len(chunks)} chunks")

        # ----- Normalize each chunk independently ---------------------
        normalized_chunks = []
        total_fixed = 0
        total_kept = 0

        for idx, chunk in enumerate(chunks):
            chunk_num = idx + 1

            # Pre-count for per-chunk safety check.
            _chunk_canon_pre = sum(
                1 for name in re.findall(
                    r'^([A-Z][A-Z0-9_ ]{1,25}):\s+.+$', chunk, re.MULTILINE
                ) if name.strip() not in self._FORMAT_NORM_NON_CHARS
            )
            _chunk_voice_pre = len(re.findall(
                r'\[VOICE:', chunk, re.IGNORECASE
            ))

            chunk_prompt = self._build_normalize_prompt(
                chunk, is_segment=True
            )

            _chunk_tokens = min(1024, max(256, len(chunk) // 4))
            _runtime_log(
                f"FORMAT_NORM: Chunk {chunk_num}/{len(chunks)} | "
                f"pre={_chunk_canon_pre}+{_chunk_voice_pre}V | "
                f"budget={_chunk_tokens}"
            )

            try:
                normalized = _run_with_timeout(
                    lambda: _generate_with_llm(
                        chunk_prompt,
                        model_id=model_id,
                        max_new_tokens=_chunk_tokens,
                        temperature=0.3,
                        optimization_profile=optimization_profile,
                    ),
                    timeout_sec=75,
                    phase_label=f"FormatNorm-Chunk-{chunk_num}",
                )

                if not normalized or len(normalized.strip()) < len(chunk) * 0.3:
                    _runtime_log(
                        f"FORMAT_NORM: Chunk {chunk_num} output too short "
                        f"- keeping original"
                    )
                    normalized_chunks.append(chunk)
                    total_kept += 1
                    continue

                # Per-chunk improvement check: did normalization add structure?
                _chunk_canon_post = len(re.findall(
                    r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$',
                    normalized, re.MULTILINE
                ))
                _chunk_voice_post = len(re.findall(
                    r'\[VOICE:', normalized, re.IGNORECASE
                ))

                if (_chunk_canon_post + _chunk_voice_post <
                        (_chunk_canon_pre + _chunk_voice_pre) * 0.8):
                    _runtime_log(
                        f"FORMAT_NORM: Chunk {chunk_num} lost dialogue "
                        f"({_chunk_canon_post}+{_chunk_voice_post}V vs "
                        f"{_chunk_canon_pre}+{_chunk_voice_pre}V) - "
                        f"keeping original"
                    )
                    normalized_chunks.append(chunk)
                    total_kept += 1
                    continue

                normalized_chunks.append(normalized.strip())
                total_fixed += 1

            except Exception as e:
                log.warning("[FormatNorm] Chunk %d failed: %s", chunk_num, e)
                _runtime_log(
                    f"FORMAT_NORM: Chunk {chunk_num} failed ({e}) "
                    f"- keeping original"
                )
                normalized_chunks.append(chunk)
                total_kept += 1

        # ----- Reassemble ---------------------------------------------
        reassembled = '\n\n'.join(normalized_chunks)

        # Final safety: total dialogue count must hold (80% floor).
        _post_canon = len(re.findall(
            r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$', reassembled, re.MULTILINE
        ))
        _post_voice = len(re.findall(r'\[VOICE:', reassembled, re.IGNORECASE))
        _pre_total = canonical_count + voice_tag_count
        _post_total = _post_canon + _post_voice

        if _pre_total > 0 and _post_total < _pre_total * 0.8:
            _runtime_log(
                f"FORMAT_NORM: Chunked pass lost too many lines overall "
                f"({_post_total} vs {_pre_total}) - keeping original"
            )
            return script_text

        _runtime_log(
            f"FORMAT_NORM: Chunked success | {total_fixed} fixed, "
            f"{total_kept} kept | lines {_pre_total}->{_post_total}"
        )
        return reassembled

    def _grammarian_pass(self, script_text, model_id,
                         optimization_profile="Standard"):
        """Final copy-edit pass: grammar, logic, and readability polish.

        Runs at temp 0.3 (structural). The grammarian does NOT:
          - Add new content, scenes, or dialogue lines
          - Rename characters or change the cast
          - Alter SFX/ENV tags or scene structure
          - Change the story arc or plot

        The grammarian DOES:
          - Fix grammar, punctuation, and spelling in dialogue
          - Smooth awkward phrasing so lines read naturally aloud
          - Flag and fix logic gaps (character in two places at once, etc.)
          - Ensure dialogue attributions are consistent
          - Clean up run-on sentences for radio pacing

        Returns the polished script, or the original if the pass fails.
        """
        # Skip for very short scripts (not worth the VRAM cost)
        if len(script_text) < 500:
            _runtime_log("GRAMMARIAN: Skipped - script too short")
            return script_text

        # Count dialogue lines before - we must not lose any.
        # Three equivalent forms are counted (LLM-agnostic: Gemma leans toward
        # `NAME:` and `[VOICE:...]`, Mistral Nemo toward `[NAME, mood] text`).
        # Missing the shorthand count previously caused the loss-check to
        # silently pass when Mistral's entire dialogue payload was shorthand.
        _pre_lines = len(re.findall(
            r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$', script_text, re.MULTILINE
        ))
        _pre_voice = len(re.findall(r'\[VOICE:', script_text, re.IGNORECASE))
        _pre_shorthand = len(re.findall(
            r'^\[[A-Z][A-Z0-9_ ]{1,20}(?:,\s*.+?)?\]\s*\S',
            script_text, re.MULTILINE,
        ))
        _pre_total = _pre_lines + _pre_voice + _pre_shorthand

        # -----------------------------------------------------------
        # CHUNKED GRAMMARIAN for long scripts (60+ dialogue lines)
        # Split by scene markers, polish each scene independently,
        # then reassemble. Prevents timeout on dense episodes.
        # -----------------------------------------------------------
        _CHUNK_THRESHOLD = 50  # dialogue lines before chunking kicks in

        if _pre_total > _CHUNK_THRESHOLD:
            return self._grammarian_chunked(
                script_text, model_id, optimization_profile,
                _pre_total
            )

        # --- Single-pass grammarian for short scripts ---------------
        return self._grammarian_single(
            script_text, model_id, optimization_profile,
            _pre_total
        )

    def _grammarian_single(self, script_text, model_id,
                           optimization_profile, pre_total):
        """Single-pass grammarian for scripts under the chunk threshold."""

        # Version C (grammarian-v2c-ship): mechanical validator framing.
        # Replaces the old "copy editor" prompt which was inviting Gemma to
        # "smooth" and "polish" slang into English-professor prose.
        grammarian_prompt = f"""You are a mechanical TTS & Parser Validator for a radio drama pipeline.
You have ZERO creative authority. Your only job is minimal, targeted fixes
so the script parses cleanly and reads correctly through text-to-speech.

You MUST PRESERVE EXACTLY:
- Every character's slang, contractions, dialect, fragments, and quirks.
- All original wording, rhythm, and word choice.
- All [SFX:], [ENV:], [VOICE:], [NAME,...], and === SCENE N === markers (verbatim).
- Whichever dialogue tag style the upstream writer used; do NOT convert between
  forms. The parser accepts all three equivalents listed below.

FIX ONLY THESE FIVE THINGS:
1. SPELLING TYPOS that would mispronounce in TTS (e.g., "teh" -> "the",
   "natrual" -> "natural").
2. PARSER SYNTAX: Every spoken line must be TAGGED in ONE of these three
   equivalent forms - keep whichever the upstream used:
       (A) CHARACTER: dialogue
       (B) [VOICE: CHARACTER, traits] dialogue
       (C) [CHARACTER, traits] dialogue
   Fix missing colons, capitalization, or spacing only. Do NOT rewrite form
   A -> B or B -> C; that conversion belongs to the normalizer, not here.
3. PUNCTUATION: Close orphaned quotes, brackets, parentheses. Add a missing
   period only where its absence would slur TTS output.
4. LOGIC CONTRADICTIONS that break continuity within the visible script
   (character in two places, references to events that haven't happened).
5. EXTREME RUN-ONS (80+ words with no stop): insert a period or ellipsis
   to allow a TTS breath. Keep every original word in original order.

STRICTLY FORBIDDEN:
- Do NOT rewrite casual dialogue into formal English.
  Example: "Ain't no way that thing's natural" stays exactly as-is.
  NEVER change it to "There is no way that entity is natural."
- Do NOT remove or replace slang, contractions, or fragments.
- Do NOT add vocabulary, metaphors, or literary flourishes.
- Do NOT restructure sentences for "elegance" or "flow."
- Do NOT add, delete, or reorder any dialogue lines or scenes.
- Do NOT touch anything that already parses and sounds fine spoken aloud.
- Do NOT add commentary, notes, markdown, or explanations.

OUTPUT: The corrected script only. Same format as input. Nothing else.

SCRIPT TO VALIDATE:
{script_text}"""

        # Token budget: same as input (we're polishing, not expanding)
        _gram_max_tokens = min(2048, max(256, len(script_text) // 3))
        _runtime_log(
            f"GRAMMARIAN: Starting polish pass | {pre_total} dialogue lines | "
            f"token budget={_gram_max_tokens}"
        )

        try:
            polished = _run_with_timeout(
                lambda: _generate_with_llm(
                    grammarian_prompt,
                    model_id=model_id,
                    max_new_tokens=_gram_max_tokens,
                    temperature=0.3,
                    optimization_profile=optimization_profile,
                ),
                timeout_sec=150,
                phase_label="Grammarian",
            )

            if not polished or len(polished.strip()) < len(script_text) * 0.5:
                _runtime_log(
                    f"GRAMMARIAN: Output too short ({len(polished or '')} chars "
                    f"vs {len(script_text)} input) - keeping original"
                )
                return script_text

            # Safety check: did we lose dialogue lines?
            # 2026-04-26 BUG-LOCAL-066: counter MUST be symmetric with the
            # pre_total computation -- pre_total counts shorthand +
            # bare-colon + voice, so post must too. Adding underscores to
            # the bare-colon char class keeps it aligned with the rest of
            # the pipeline (`OSCAR_KANE: line` form).
            _post_lines = len(re.findall(
                r'^[A-Z][A-Z0-9_ ]{1,19}:\s*.+$', polished, re.MULTILINE
            ))
            _post_voice = len(re.findall(r'\[VOICE:', polished, re.IGNORECASE))
            _post_shorthand = len(re.findall(
                r'^\[[A-Z][A-Z0-9_ ]{1,20}(?:,\s*.+?)?\]\s*\S',
                polished, re.MULTILINE,
            ))
            _post_total = _post_lines + _post_voice + _post_shorthand

            if _post_total < pre_total * 0.8:
                _runtime_log(
                    f"GRAMMARIAN: Rejected - lost too many lines "
                    f"({_post_total} vs {pre_total}) - keeping original"
                )
                return script_text

            _runtime_log(
                f"GRAMMARIAN: Success | lines {pre_total}->{_post_total} | "
                f"chars {len(script_text)}->{len(polished.strip())}"
            )
            return polished.strip()

        except Exception as e:
            log.warning("[Grammarian] Polish pass failed: %s", e)
            _runtime_log(f"GRAMMARIAN: Failed ({e}) - keeping original")
            return script_text

    def _grammarian_chunked(self, script_text, model_id,
                            optimization_profile, pre_total):
        """Chunked grammarian for long scripts (50+ dialogue lines).

        Splits the script by === SCENE N === markers, polishes each scene
        independently, then reassembles. If the script has no scene markers,
        falls back to line-based chunking (20 dialogue lines per chunk).

        Each chunk gets its own timeout and safety checks, so one failed
        chunk does not block the rest. Failed chunks keep their original text.
        """
        _runtime_log(
            f"GRAMMARIAN: Chunked mode | {pre_total} dialogue lines "
            f"(threshold 50)"
        )

        # ----- Split by scene markers ---------------------------------
        scene_re = re.compile(
            r'(===\s*SCENE\s+\S+.*?===)', re.IGNORECASE
        )
        parts = scene_re.split(script_text)

        # parts alternates: [pre-scene-text, marker, scene-body, marker, ...]
        # Reassemble into chunks: each chunk = marker + body until next marker
        chunks = []
        current_chunk = ""
        for part in parts:
            if scene_re.match(part):
                # This is a scene marker - save previous chunk, start new one
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = part
            else:
                current_chunk += part
        if current_chunk.strip():
            chunks.append(current_chunk)

        # If splitting produced only 1 chunk, fall back to line-based split
        if len(chunks) <= 1:
            _runtime_log("GRAMMARIAN: No scene markers found - using line-based chunking")
            all_lines = script_text.split('\n')
            chunk_size = 40  # lines per chunk (not dialogue lines, raw lines)
            chunks = [
                '\n'.join(all_lines[i:i + chunk_size])
                for i in range(0, len(all_lines), chunk_size)
            ]

        _runtime_log(f"GRAMMARIAN: Split into {len(chunks)} chunks")

        # ----- Polish each chunk independently -------------------------
        polished_chunks = []
        total_fixed = 0
        total_kept = 0

        for idx, chunk in enumerate(chunks):
            chunk_num = idx + 1

            # Count dialogue lines in this chunk
            _chunk_dl = len(re.findall(
                r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$', chunk, re.MULTILINE
            ))
            _chunk_dl += len(re.findall(r'\[VOICE:', chunk, re.IGNORECASE))

            # Skip chunks with no dialogue (pure SFX/ENV blocks)
            if _chunk_dl == 0:
                _runtime_log(
                    f"GRAMMARIAN: Chunk {chunk_num}/{len(chunks)} "
                    f"skipped (no dialogue)"
                )
                polished_chunks.append(chunk)
                continue

            _chunk_tokens = min(1024, max(128, len(chunk) // 3))
            _runtime_log(
                f"GRAMMARIAN: Chunk {chunk_num}/{len(chunks)} | "
                f"{_chunk_dl} lines | budget={_chunk_tokens}"
            )

            # Version C chunked variant (grammarian-v2c-ship): same rules as
            # single-pass, with the segment-awareness clause so chunks do not
            # flag references to scenes they cannot see as contradictions.
            chunk_prompt = f"""You are a mechanical TTS & Parser Validator for a radio drama pipeline. The text below is ONE SEGMENT of a longer script. Ignore references to scenes you cannot see -- do not flag them as contradictions.
You have ZERO creative authority. Your only job is minimal, targeted fixes
so the script parses cleanly and reads correctly through text-to-speech.

You MUST PRESERVE EXACTLY:
- Every character's slang, contractions, dialect, fragments, and quirks.
- All original wording, rhythm, and word choice.
- All [SFX:], [ENV:], [VOICE:], [NAME,...], and === SCENE N === markers (verbatim).
- Whichever dialogue tag style the upstream writer used; do NOT convert between
  forms. The parser accepts all three equivalents listed below.

FIX ONLY THESE FIVE THINGS:
1. SPELLING TYPOS that would mispronounce in TTS (e.g., "teh" -> "the",
   "natrual" -> "natural").
2. PARSER SYNTAX: Every spoken line must be TAGGED in ONE of these three
   equivalent forms - keep whichever the upstream used:
       (A) CHARACTER: dialogue
       (B) [VOICE: CHARACTER, traits] dialogue
       (C) [CHARACTER, traits] dialogue
   Fix missing colons, capitalization, or spacing only. Do NOT rewrite form
   A -> B or B -> C; that conversion belongs to the normalizer, not here.
3. PUNCTUATION: Close orphaned quotes, brackets, parentheses. Add a missing
   period only where its absence would slur TTS output.
4. LOGIC CONTRADICTIONS that break continuity within the visible script
   (character in two places, references to events that haven't happened).
5. EXTREME RUN-ONS (80+ words with no stop): insert a period or ellipsis
   to allow a TTS breath. Keep every original word in original order.

STRICTLY FORBIDDEN:
- Do NOT rewrite casual dialogue into formal English.
  Example: "Ain't no way that thing's natural" stays exactly as-is.
  NEVER change it to "There is no way that entity is natural."
- Do NOT remove or replace slang, contractions, or fragments.
- Do NOT add vocabulary, metaphors, or literary flourishes.
- Do NOT restructure sentences for "elegance" or "flow."
- Do NOT add, delete, or reorder any dialogue lines or scenes.
- Do NOT touch anything that already parses and sounds fine spoken aloud.
- Do NOT add commentary, notes, markdown, or explanations.

OUTPUT: The corrected script only. Same format as input. Nothing else.

SCRIPT SEGMENT TO VALIDATE:
{chunk}"""

            try:
                polished = _run_with_timeout(
                    lambda: _generate_with_llm(
                        chunk_prompt,
                        model_id=model_id,
                        max_new_tokens=_chunk_tokens,
                        temperature=0.3,
                        optimization_profile=optimization_profile,
                    ),
                    timeout_sec=90,
                    phase_label=f"Grammarian-Chunk-{chunk_num}",
                )

                if not polished or len(polished.strip()) < len(chunk) * 0.5:
                    _runtime_log(
                        f"GRAMMARIAN: Chunk {chunk_num} output too short "
                        f"- keeping original"
                    )
                    polished_chunks.append(chunk)
                    total_kept += 1
                    continue

                # Safety: did we lose dialogue lines in this chunk?
                _post_dl = len(re.findall(
                    r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$', polished, re.MULTILINE
                ))
                _post_dl += len(re.findall(
                    r'\[VOICE:', polished, re.IGNORECASE
                ))

                if _post_dl < _chunk_dl * 0.8:
                    _runtime_log(
                        f"GRAMMARIAN: Chunk {chunk_num} lost lines "
                        f"({_post_dl} vs {_chunk_dl}) - keeping original"
                    )
                    polished_chunks.append(chunk)
                    total_kept += 1
                    continue

                polished_chunks.append(polished.strip())
                total_fixed += 1

            except Exception as e:
                log.warning("[Grammarian] Chunk %d failed: %s", chunk_num, e)
                _runtime_log(
                    f"GRAMMARIAN: Chunk {chunk_num} failed ({e}) "
                    f"- keeping original"
                )
                polished_chunks.append(chunk)
                total_kept += 1

        # ----- Reassemble ---------------------------------------------
        reassembled = '\n\n'.join(polished_chunks)

        # Final safety: total dialogue count must hold
        _post_lines = len(re.findall(
            r'^[A-Z][A-Z0-9 ]{1,19}:\s*.+$', reassembled, re.MULTILINE
        ))
        _post_voice = len(re.findall(r'\[VOICE:', reassembled, re.IGNORECASE))
        _post_total = _post_lines + _post_voice

        if _post_total < pre_total * 0.8:
            _runtime_log(
                f"GRAMMARIAN: Reassembly lost too many lines "
                f"({_post_total} vs {pre_total}) - reverting to original"
            )
            return script_text

        _runtime_log(
            f"GRAMMARIAN: Chunked complete | {total_fixed} polished, "
            f"{total_kept} kept original | lines {pre_total}->{_post_total} | "
            f"chars {len(script_text)}->{len(reassembled)}"
        )
        return reassembled

    def _extend_script_dialogue(self, script_text, deficit_words,
                                 target_words, model_id, style,
                                 optimization_profile="Standard",
                                 fallback_cast=None):
        """LLM extension pass: add more dialogue to raw script text.

        Called when raw text dialogue word count is <70% of target.
        The LLM reads the existing script and generates additional dialogue
        that fits the existing scenes, characters, and narrative arc.
        New dialogue is appended to the end of the raw script text.

        Returns the extended raw script text (or original on failure).
        """
        _runtime_log(f"WORD_EXTEND: Starting dialogue extension (deficit={deficit_words} words)")

        # Extract characters and dialogue preview from raw text (BUG-025:
        # dual-format extraction covers both bare NAME: and [VOICE: NAME] formats)
        _all_dialogue = _extract_all_dialogue(script_text)
        characters = sorted({name for name, _ in _all_dialogue})

        # BUG-024 fix: When script has zero character dialogue (only SFX/ANNOUNCER),
        # the extraction returns an empty character list. Fall back to the
        # pre-rolled cast names so the extension LLM knows WHO to write dialogue for.
        if not characters and fallback_cast:
            characters = sorted(fallback_cast)
            _runtime_log(
                f"WORD_EXTEND: Zero characters found in script text - "
                f"falling back to pre-rolled cast: {', '.join(characters)}"
            )

        existing_dialogue = [
            f"{name}: {dialogue[:80]}"
            for name, dialogue in _all_dialogue
        ]
        existing_preview = "\n".join(existing_dialogue[:40])
        num_scenes = max(1, len(re.findall(r'=== SCENE', script_text)))

        # Calculate how many new lines we need (avg ~10 words per dialogue line)
        new_lines_needed = max(10, deficit_words // 10)

        # BUG-024: When script has zero dialogue, show the full raw script
        # (SFX cues, scene headers, atmosphere) so the LLM has story context
        # to write dialogue that fits the existing narrative skeleton.
        if existing_preview:
            script_context_block = f"EXISTING SCRIPT PREVIEW:\n{existing_preview}"
        else:
            # Trim raw script to ~2000 chars to fit token budget
            _raw_trimmed = script_text[:2000]
            script_context_block = (
                "WARNING: The script currently has ZERO character dialogue.\n"
                "It contains only SFX cues, scene headers, and atmosphere.\n"
                "You must CREATE all the dialogue from scratch using the characters listed.\n\n"
                f"RAW SCRIPT SKELETON:\n{_raw_trimmed}"
            )

        extend_prompt = f"""You are extending a {style.replace("_", " ")} radio drama script.
The current script has {len(existing_dialogue)} dialogue lines but needs approximately {new_lines_needed} MORE lines
to reach the target of {target_words} words of spoken dialogue.

CHARACTERS IN THE STORY: {", ".join(characters)}
NUMBER OF SCENES: {num_scenes}

{script_context_block}

TASK: Write {new_lines_needed} NEW dialogue lines that continue and deepen the story.
- Use ONLY the existing characters listed above
- Every line MUST use format: CHARACTER_NAME: dialogue text
- Add conflict, tension, emotional beats, reactions, and reveals
- Develop character relationships — disagreements, alliances, secrets
- Include stage directions in parentheses: (angry), (whispering), (pause)
- Do NOT repeat existing lines
- Do NOT add new characters
- Do NOT write ANNOUNCER lines
- Do NOT write scene headers, SFX, or ENV tags — ONLY dialogue lines

OUTPUT ONLY THE NEW DIALOGUE LINES, one per line:"""

        try:
            # Token budget: ~4 chars per token, ~10 words per line, ~50 chars per line
            _max_tokens = min(2048, max(512, new_lines_needed * 20))
            _runtime_log(f"WORD_EXTEND: Requesting {new_lines_needed} lines, budget={_max_tokens} tokens")

            extended_text = _run_with_timeout(
                lambda: _generate_with_llm(
                    extend_prompt,
                    model_id=model_id,
                    max_new_tokens=_max_tokens,
                    temperature=0.5,  # Moderate - creative but follows instructions
                    optimization_profile=optimization_profile,
                ),
                timeout_sec=90,
                phase_label="WordExtend",
            )

            if not extended_text or len(extended_text.strip()) < 50:
                _runtime_log("WORD_EXTEND: Extension returned too little text - keeping original")
                return script_text

            # Normalize bold dialogue in extension output before filtering
            extended_text = _normalize_dialogue_names(extended_text)

            # Filter extension output — only keep valid dialogue lines
            valid_lines = []
            for raw_line in extended_text.strip().split("\n"):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                m = re.match(r'^([A-Z][A-Z0-9_ ]{1,25}):\s+(.+)', raw_line)
                if m:
                    name = m.group(1).strip()
                    # BUG-LOCAL-036: was `_false_positives` (undefined).
                    # Module-level constant is `_DIALOGUE_FALSE_POSITIVES`.
                    if name not in _DIALOGUE_FALSE_POSITIVES and name in characters:
                        valid_lines.append(raw_line)

            if len(valid_lines) < 3:
                _runtime_log(f"WORD_EXTEND: Only {len(valid_lines)} valid lines - keeping original")
                return script_text

            # Append new dialogue to end of raw script text
            new_block = "\n".join(valid_lines)
            new_word_count = sum(len(line.split()) for line in valid_lines)
            script_text = f"{script_text.rstrip()}\n\n{new_block}\n"
            _runtime_log(
                f"WORD_EXTEND: Appended {len(valid_lines)} lines "
                f"({new_word_count} words) to raw script text"
            )
            return script_text

        except Exception as e:
            log.warning("[WordExtend] Extension pass failed: %s", e)
            _runtime_log(f"WORD_EXTEND: Failed ({e}) - keeping original")
            return script_text

    def _llm_reparse_rescue(self, raw_script, model_id, optimization_profile="Standard"):
        """LLM rescue pass: extract dialogue from a script the regex parser cannot handle.

        Fires ONLY when _parse_script() returns 0 dialogue lines from substantial text.
        The LLM reads the raw script (prose, screenplay, novel-style, whatever format
        the creative pass produced) and extracts every spoken line into strict
        CHARACTER_NAME: dialogue format.

        Same model already in VRAM, low temperature (0.3), focused extraction task.
        Typically completes in 10-20 seconds. Returns reformatted text or None on failure.
        """
        import time

        _runtime_log("LLM_RESCUE: Starting dialogue extraction rescue pass")

        # Truncate input to avoid blowing context - keep first 8000 chars
        truncated = raw_script[:8000]

        rescue_prompt = f"""Extract all spoken dialogue from the script below and reformat into EXACTLY this structure:

=== SCENE 1 ===
[ENV: location description]
[SFX: sound effect description]
CHARACTER_NAME: Their exact spoken dialogue.
CHARACTER_NAME: Their exact reply.
(beat)
=== SCENE 2 ===
CHARACTER_NAME: Next scene dialogue.

FORMAT RULES:
- Scene breaks: === SCENE N ===
- Environment: [ENV: description]
- Sound effects: [SFX: description]
- Dialogue: CHARACTER_NAME: exact words (name in ALL CAPS, colon, space, dialogue)
- Pauses: (beat)
- First and last dialogue lines should be ANNOUNCER
- Preserve exact dialogue words. Do not rewrite, summarize, or add new lines.
- Output ONLY the reformatted script. No commentary.

SCRIPT:
{truncated}

REFORMATTED:"""

        try:
            start = time.time()
            rescued = _generate_with_llm(
                rescue_prompt,
                model_id=model_id,
                max_new_tokens=min(4096, int(len(truncated) / 2.5)),
                temperature=0.3,
                top_p=0.9,
                optimization_profile=optimization_profile,
            )
            elapsed = time.time() - start
            _runtime_log(f"LLM_RESCUE: Completed in {elapsed:.1f}s ({len(rescued)} chars)")

            if not rescued or len(rescued) < 100:
                _runtime_log("LLM_RESCUE: Output too short - rescue failed")
                return None

            # Sanity check - count dialogue in ANY recognizable format
            import re
            # Pattern 1: CHARACTER_NAME: dialogue
            bare_count = len(re.findall(r'^[A-Z][A-Z0-9_ ]{1,25}:\s+.+', rescued, re.MULTILINE))
            # Pattern 2: [VOICE: NAME, ...] dialogue
            voice_count = len(re.findall(r'\[VOICE:', rescued, re.IGNORECASE))
            dialogue_count = bare_count + voice_count
            _runtime_log(f"LLM_RESCUE: Found {dialogue_count} dialogue lines ({bare_count} bare + {voice_count} [VOICE:])")

            if dialogue_count < 3:
                _runtime_log("LLM_RESCUE: Too few dialogue lines in rescue - giving up")
                return None

            return rescued

        except Exception as e:
            log.warning("[LLM_RESCUE] Rescue pass failed: %s", e)
            _runtime_log(f"LLM_RESCUE: Failed ({e})")
            return None

    def _generate_announcer_bookends(self, script_lines, episode_title,
                                     style, news_headline, character_names,
                                     model_id, optimization_profile="Standard"):
        """LLM micro-pass: generate story-specific ANNOUNCER opening and closing.

        Called by QA_REPAIR when the parser detects missing ANNOUNCER bookends.
        Uses the same loaded LLM at low temperature for a fast ~50-token generation.
        Returns (opening_line, closing_line) as plain strings.
        Falls back to generic canned text if the LLM call fails.
        """
        # Build a brief story summary from the first few dialogue lines
        dialogue_preview = []
        for ln in script_lines:
            if ln.get("type") == "dialogue" and ln.get("character_name") != "ANNOUNCER":
                dialogue_preview.append(f"{ln['character_name']}: {ln.get('line', '')[:80]}")
                if len(dialogue_preview) >= 4:
                    break
        story_glimpse = "\n".join(dialogue_preview) if dialogue_preview else "(no dialogue preview)"

        chars_list = ", ".join(sorted(character_names - {"ANNOUNCER"})) if character_names else "unknown"

        prompt = f"""You are the ANNOUNCER for the radio drama "Signal Lost".
Write exactly TWO lines - an OPENING and a CLOSING - for tonight's episode.

EPISODE: {episode_title}
GENRE: {style}
NEWS SEED: {news_headline[:300] if news_headline else 'science fiction'}
CHARACTERS: {chars_list}
STORY PREVIEW:
{story_glimpse}

RULES:
- OPENING: 2-4 sentences. Include today's date naturally (say "April 12th, 2026" not a timestamp). Name the setting. Mention 1-2 characters by name. End with a hook/tagline. Match the genre tone.
- CLOSING: 1-2 sentences. Wrap up with "This has been Signal Lost" and a brief real-science epilogue tied to the news seed.
- Write ONLY the two lines, labeled OPENING: and CLOSING:
- No stage directions, no [VOICE:] tags, just the spoken words.

OPENING:
"""
        try:
            result = _run_with_timeout(
                lambda: _generate_with_llm(
                    prompt,
                    model_id=model_id,
                    max_new_tokens=200,
                    temperature=0.4,
                    optimization_profile=optimization_profile,
                ),
                timeout_sec=30,
                phase_label="AnnouncerGen",
            )

            if not result or len(result.strip()) < 20:
                _runtime_log("ANNOUNCER_GEN: LLM output too short - using fallback")
                return self._announcer_fallback()

            # Parse OPENING: and CLOSING: from result
            opening = ""
            closing = ""
            current = None
            for raw_line in result.strip().splitlines():
                stripped = raw_line.strip()
                if stripped.upper().startswith("OPENING:"):
                    current = "opening"
                    text_after = stripped[len("OPENING:"):].strip().strip('"')
                    if text_after:
                        opening = text_after
                elif stripped.upper().startswith("CLOSING:"):
                    current = "closing"
                    text_after = stripped[len("CLOSING:"):].strip().strip('"')
                    if text_after:
                        closing = text_after
                elif current == "opening" and not opening:
                    opening = stripped.strip('"')
                elif current == "opening" and opening and not stripped.upper().startswith("CLOSING"):
                    opening += " " + stripped.strip('"')
                elif current == "closing":
                    if closing:
                        closing += " " + stripped.strip('"')
                    else:
                        closing = stripped.strip('"')

            if not opening or len(opening) < 15:
                _runtime_log("ANNOUNCER_GEN: Could not parse opening - using fallback")
                return self._announcer_fallback()
            if not closing or len(closing) < 10:
                closing = f"And so the transmission ends. This has been Signal Lost. {episode_title}. Stay safe."

            _runtime_log(f"ANNOUNCER_GEN: Generated opening ({len(opening)} chars) + closing ({len(closing)} chars)")
            return (opening, closing)

        except Exception as e:
            log.warning("[AnnouncerGen] LLM micro-pass failed: %s", e)
            _runtime_log(f"ANNOUNCER_GEN: Failed ({e}) - using fallback")
            return self._announcer_fallback()

    @staticmethod
    def _announcer_fallback():
        """Canned ANNOUNCER text when LLM generation fails."""
        return (
            "Welcome to Signal Lost. Tonight's broadcast takes us into the unknown.",
            "And so the transmission ends. This has been Signal Lost. Stay safe.",
        )

    # Descriptor words that indicate a missing character name (Gemma dropped the NAME field)
    _GENDER_WORDS = frozenset([
        "male", "female", "man", "woman", "boy", "girl", "nonbinary",
        "young", "old", "older", "elderly", "middle", "teen",
    ])

    def _parse_script(self, text):
        """Parse raw script text into structured Canonical Audio Tokens.

        Robust against the most common Gemma formatting failure: omitting the
        character NAME as the first field in [VOICE:] tags, producing malformed
        tags like [VOICE: male, 40s, calm] that would silently create "MALE" as
        a character. Those are caught, logged, and assigned a positional fallback
        name (CHAR_A, CHAR_B, ...) so Bark still produces audio.
        """
        lines = []
        _fallback_counter = 0   # incremented in the for-loop below for CHAR_A / CHAR_B fallback names

        # OTR Canonical 1.0 RegEx Patterns
        # BUG-009 fix: accept both `=== SCENE N ===` and `=== SCENE N ***` (Gemma
        # occasionally uses asterisks as the closing delimiter, which silently
        # broke scene splitting and merged Act 3 into Act 2's last scene).
        scene_pat = re.compile(r'^===\s*SCENE\s+(.+?)\s*(?:===|\*\*\*)', re.IGNORECASE)
        env_pat   = re.compile(r'^\[ENV:\s*(.+?)\]',          re.IGNORECASE)
        sfx_pat   = re.compile(r'^\[SFX:\s*(.+?)\]',          re.IGNORECASE)
        beat_pat  = re.compile(r'^\(beat\)$', re.IGNORECASE)

        # Voice patterns - ordered from most to least specific:
        # v1 (canonical): [VOICE: NAME, traits] dialogue on same line
        voice_inline_pat = re.compile(r'^\[VOICE:\s*(.+?),\s*(.+?)\]\s*(.+)$', re.IGNORECASE)
        # v2 (no-traits): [VOICE: NAME] dialogue on same line
        voice_notrait_pat = re.compile(r'^\[VOICE:\s*([A-Z][A-Z0-9_ ]+?)\]\s*(.+)$', re.IGNORECASE)
        # v3 (tag only): [VOICE: NAME, traits] with dialogue on NEXT line
        voice_tagonly_pat = re.compile(r'^\[VOICE:\s*(.+?)(?:,\s*(.+?))?\]\s*$', re.IGNORECASE)
        # v4 (shorthand): [ANNOUNCER, traits] or [ANNOUNCER] as a standalone tag (Mistral Nemo style)
        voice_shorthand_pat = re.compile(r'^\[([A-Z][A-Z0-9_ ]{1,20})(?:,\s*(.+?))?\]\s*$', re.IGNORECASE)
        # v4-inline: [CHARACTER, traits] dialogue on the SAME line (Mistral Nemo's most common form)
        # BUG-LOCAL-061 fix (2026-04-24): previous regex set only accepted the
        # bracketed tag alone (v4 standalone) or the VOICE:-prefixed inline form
        # (v1/v2). Mistral Nemo emits `[EDNA, Female, 40s, urgent] Dammit!`
        # inline without the VOICE: prefix; every such line fell through the
        # parser and produced `[BatchBark] Found 0 dialogue lines`.
        voice_shorthand_inline_pat = re.compile(
            r'^\[([A-Z][A-Z0-9_ ]{1,20})(?:,\s*(.+?))?\]\s*(.+)$',
            re.IGNORECASE,
        )

        raw_lines = text.strip().splitlines()
        i = 0
        while i < len(raw_lines):
            raw_line = raw_lines[i]
            s = raw_line.strip()
            # v1.4 Markdown Bolding Hallucination Fix:
            # Gemma 2B often generates **[VOICE:...]** or **=== SCENE ===**
            # Strip outer asterisks before matching tags.
            s = re.sub(r'^\*+|(?<=\])\*+$', '', s).strip()
            # Also strip italic markers flanking a tag: *[VOICE:..]* or _[VOICE:..]_
            s = re.sub(r'^[_*]+|[_*]+$', '', s).strip()

            if not s:
                i += 1
                continue

            # v1.4 Theme B - Timeout fallback sentinel path.
            if s.startswith("[SYSTEM_SENTINEL:"):
                i += 1
                continue

            m = scene_pat.match(s)
            if m:
                lines.append({"type": "scene_break", "scene": m.group(1)})
                i += 1
                continue

            m = env_pat.match(s)
            if m:
                lines.append({"type": "environment", "description": m.group(1)})
                i += 1
                continue

            m = sfx_pat.match(s)
            if m:
                lines.append({"type": "sfx", "description": m.group(1)})
                i += 1
                continue

            m = beat_pat.match(s)
            if m:
                lines.append({"type": "pause", "kind": "beat", "duration_ms": 200})
                i += 1
                continue

            # -- VOICE TAG MATCHING (4 variants) --------------------------

            # v1: [VOICE: NAME, traits] dialogue - inline
            m = voice_inline_pat.match(s)
            if m:
                raw_name     = m.group(1).strip()
                voice_traits = m.group(2).strip()
                dialogue     = m.group(3).strip().strip('"\u201c\u201d*_')
                if raw_name.lower() in self._GENDER_WORDS:
                    _fallback_counter += 1
                    fallback_name = f"CHAR_{chr(64 + _fallback_counter)}"
                    log.warning("[ScriptParser] Malformed VOICE tag - name field is a descriptor word '%s'. Assigning fallback '%s'.", raw_name, fallback_name)
                    voice_traits = f"{raw_name}, {voice_traits}"
                    character_name = fallback_name
                else:
                    character_name = raw_name.upper()
                # BUG-LOCAL-100: detect inline-narration pattern. When the
                # captured "dialogue" is actually third-person stage direction
                # ("Lev bursts onto deck...") with the real dialogue on the
                # NEXT line, look ahead and use that instead. Without this,
                # Bark TTS reads stage directions aloud (Stellar Shadows
                # 2026-04-28: l002 + l003).
                if _is_inline_narration(character_name, dialogue):
                    j = i + 1
                    while j < len(raw_lines) and not raw_lines[j].strip():
                        j += 1
                    next_s = raw_lines[j].strip() if j < len(raw_lines) else ""
                    next_s_clean = re.sub(r'^[*_]+|[*_]+$', '', next_s).strip()
                    if next_s_clean and not next_s_clean.startswith('[') and not next_s_clean.startswith('='):
                        log.info(
                            "[ScriptParser] BUG-100: rejected inline narration for %s, "
                            "using next-line dialogue",
                            character_name,
                        )
                        dialogue = next_s_clean.strip('"\u201c\u201d*_')
                        lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": voice_traits, "line": dialogue})
                        i = j + 1  # consume both lines
                        continue
                    log.warning(
                        "[ScriptParser] BUG-100: inline narration detected for %s "
                        "but no usable next-line dialogue found",
                        character_name,
                    )
                lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": voice_traits, "line": dialogue})
                i += 1
                continue

            # v2: [VOICE: NAME] dialogue - no traits, inline
            m = voice_notrait_pat.match(s)
            if m:
                character_name = m.group(1).strip().upper()
                dialogue = m.group(2).strip().strip('"\u201c\u201d*_')
                # BUG-LOCAL-100: same inline-narration check as v1.
                if _is_inline_narration(character_name, dialogue):
                    j = i + 1
                    while j < len(raw_lines) and not raw_lines[j].strip():
                        j += 1
                    next_s = raw_lines[j].strip() if j < len(raw_lines) else ""
                    next_s_clean = re.sub(r'^[*_]+|[*_]+$', '', next_s).strip()
                    if next_s_clean and not next_s_clean.startswith('[') and not next_s_clean.startswith('='):
                        log.info(
                            "[ScriptParser] BUG-100: rejected inline narration for %s, "
                            "using next-line dialogue",
                            character_name,
                        )
                        dialogue = next_s_clean.strip('"\u201c\u201d*_')
                        lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": "", "line": dialogue})
                        i = j + 1
                        continue
                    log.warning(
                        "[ScriptParser] BUG-100: inline narration detected for %s "
                        "but no usable next-line dialogue found",
                        character_name,
                    )
                lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": "", "line": dialogue})
                i += 1
                continue

            # v3: [VOICE: NAME, traits] tag-only - look ahead for dialogue on NEXT line
            m = voice_tagonly_pat.match(s)
            if m:
                raw_name     = m.group(1).strip()
                voice_traits = (m.group(2) or "").strip()
                # Skip non-VOICE bracket tags that could match (e.g. [MUSIC:...])
                # Only handle if the raw_name looks like a real character name (uppercase letters)
                _first_word_v3 = raw_name.upper().split()[0] if raw_name.strip() else ""
                if re.match(r'^[A-Z][A-Z0-9_ ]*$', raw_name, re.IGNORECASE) and _first_word_v3 not in (
                    "MUSIC", "SFX", "ENV", "BEAT", "PAUSE", "SYSTEM_SENTINEL",
                    "ACT", "SCENE", "TRANSITION", "CONTINUED", "CONT", "END",
                ):
                    # Peek at next non-empty line for dialogue
                    j = i + 1
                    while j < len(raw_lines) and not raw_lines[j].strip():
                        j += 1
                    next_s = raw_lines[j].strip() if j < len(raw_lines) else ""
                    next_s_clean = re.sub(r'^[*_]+|[*_]+$', '', next_s).strip()
                    # Accept as dialogue if next line is NOT a tag and not empty
                    if next_s_clean and not next_s_clean.startswith('[') and not next_s_clean.startswith('='):
                        dialogue = next_s_clean.strip('"\u201c\u201d*_')
                        if raw_name.lower() in self._GENDER_WORDS:
                            _fallback_counter += 1
                            character_name = f"CHAR_{chr(64 + _fallback_counter)}"
                        else:
                            character_name = raw_name.upper()
                        lines.append({"type": "dialogue", "character_name": character_name, "voice_traits": voice_traits, "line": dialogue})
                        i = j + 1  # consume both tag line and dialogue line
                        continue
                    # else: fall through as direction

            # v4-inline: [CHARACTER, traits] dialogue on the SAME line (BUG-LOCAL-061)
            # Mistral Nemo's most common dialogue form. Must run before v4
            # standalone because v4 standalone requires `]\s*$` and so never
            # matches an inline-dialogue line.
            m = voice_shorthand_inline_pat.match(s)
            if m:
                raw_name     = m.group(1).strip()
                voice_traits = (m.group(2) or "").strip()
                dialogue     = m.group(3).strip().strip('"\u201c\u201d*_')
                upper_name   = raw_name.upper()
                _first_word_v4i = upper_name.split()[0] if upper_name.strip() else ""
                # Reject structural bracketed tags (`[VOICE: ...]` handled by
                # v1/v2/v3, `[ENV:...]` / `[SFX:...]` already matched above,
                # but `[SCENE ONE] description` or `[ACT TWO]` etc must not
                # register as character dialogue).
                if _first_word_v4i not in (
                    "ENV", "SFX", "MUSIC", "BEAT", "PAUSE", "ACT", "SCENE",
                    "TRANSITION", "CONTINUED", "CONT", "END", "FADE", "CUT",
                    "INT", "EXT", "VOICE",
                ) and not _looks_like_non_character_cast_name(upper_name):
                    # BUG-LOCAL-091: content-based blocklist catches
                    # multi-word SFX/stage-direction tags whose FIRST
                    # word looks innocent (e.g. `[ALARM BLARING] ...`,
                    # `[CHAMBER FLICKERS BRIGHTLY] ...`,
                    # `[BACK AT THE LAB] ...`). Without this guard
                    # those phrases get registered as cast names,
                    # poison the cast map, and crash Director's
                    # JSON output (BUG-LOCAL-090 root cause).
                    if raw_name.lower() in self._GENDER_WORDS:
                        _fallback_counter += 1
                        character_name = f"CHAR_{chr(64 + _fallback_counter)}"
                        voice_traits = f"{raw_name}, {voice_traits}".strip(", ")
                    else:
                        character_name = upper_name
                    lines.append({
                        "type": "dialogue",
                        "character_name": character_name,
                        "voice_traits": voice_traits,
                        "line": dialogue,
                    })
                    i += 1
                    continue
                elif _looks_like_non_character_cast_name(upper_name):
                    # Re-route to SFX so the cue isn't lost entirely.
                    # The dialogue text after the bracket is descriptive
                    # narration of the cue; record it on the SFX entry's
                    # description field for downstream AudioGen.
                    log.info(
                        "[ScriptParser] BUG-091: rejected '%s' as cast "
                        "(SFX/stage-direction); routing to sfx",
                        upper_name,
                    )
                    lines.append({
                        "type": "sfx",
                        "description": f"{upper_name}: {dialogue}".strip(": "),
                    })
                    i += 1
                    continue

            # v4: [CHARACTER, traits] shorthand (e.g. [ANNOUNCER, female, 50s, calm])
            # Used by Mistral Nemo when it omits the VOICE: prefix
            m = voice_shorthand_pat.match(s)
            if m:
                raw_name     = m.group(1).strip()
                voice_traits = (m.group(2) or "").strip()
                # Must look like a character name (not a known structural tag)
                upper_name = raw_name.upper()
                _first_word_v4 = upper_name.split()[0] if upper_name.strip() else ""
                # BUG-LOCAL-091: same content-based filter as v4-inline
                # to reject multi-word SFX/stage-direction tags whose
                # first word slipped past the structural blocklist.
                if (
                    _first_word_v4 not in (
                        "ENV", "SFX", "MUSIC", "BEAT", "PAUSE", "ACT", "SCENE",
                        "TRANSITION", "CONTINUED", "CONT", "END",
                    )
                    and not _looks_like_non_character_cast_name(upper_name)
                ):
                    j = i + 1
                    while j < len(raw_lines) and not raw_lines[j].strip():
                        j += 1
                    next_s = raw_lines[j].strip() if j < len(raw_lines) else ""
                    next_s_clean = re.sub(r'^[*_]+|[*_]+$', '', next_s).strip()
                    if next_s_clean and not next_s_clean.startswith('[') and not next_s_clean.startswith('='):
                        dialogue = next_s_clean.strip('"\u201c\u201d*_')
                        lines.append({"type": "dialogue", "character_name": upper_name, "voice_traits": voice_traits, "line": dialogue})
                        i = j + 1
                        continue
                elif _looks_like_non_character_cast_name(upper_name):
                    # BUG-091: reject as cast; route the bracketed
                    # cue to SFX so it isn't dropped silently.
                    log.info(
                        "[ScriptParser] BUG-091: rejected '%s' as cast "
                        "(SFX/stage-direction); routing to sfx",
                        upper_name,
                    )
                    lines.append({"type": "sfx", "description": upper_name})
                    i += 1
                    continue

            # v5: bare `NAME: dialogue` (e.g. `DRACULA MALONE: We're gonna get out of here.`)
            # BUG-LOCAL-038: Mistral Nemo emits this form natively and FORMAT_NORM
            # used to be the only thing rewriting it into `[VOICE: NAME, traits]`.
            # Register it as a first-class pattern so FORMAT_NORM becomes a
            # nice-to-have (adds traits + voice cues) rather than load-bearing
            # (prevents silent dialogue-token drop into BatchBark).
            #
            # Pattern accepts 0-2 leading asterisks, an optional (parenthetical)
            # emotion tag after the name, and ignores any trailing asterisks.
            # Structural tokens are blacklisted so `SCENE: ...`, `ACT 1: ...`,
            # `TITLE: ...` (BUG-LOCAL-037), `ENV: ...`, etc never register as
            # dialogue. ANNOUNCER is intentionally allowed -- BatchBark counts
            # + skips it and routes to the Kokoro bus.
            # 2026-04-26 BUG-LOCAL-066: added `_` to char class so names
            # the production_plan emits with underscores (`OSCAR_KANE`,
            # `RUFUS_HALPERT`, `AUTHORITY_VOICE`) survive the parser. All
            # the upstream counters (`_RE_BARE_DIALOGUE`, FORMAT_NORM,
            # Grammarian) accept underscores; without this fix the parser
            # silently dropped any underscored character's lines.
            _m_v5 = re.match(
                r'^(?:\*{0,2})([A-Z][A-Z0-9_ ]{0,24})(?:\*{0,2})'
                r'(?:\s*\(([^)]*)\))?\s*:\s+(.+)$',
                s,
            )
            if _m_v5:
                _v5_raw_name = _m_v5.group(1).strip()
                _v5_first_word = _v5_raw_name.split()[0] if _v5_raw_name else ""
                _v5_structural = {
                    "ENV", "SFX", "MUSIC", "BEAT", "PAUSE", "ACT", "SCENE",
                    "TRANSITION", "CONTINUED", "CONT", "END", "FADE", "CUT",
                    "INT", "EXT", "OPENING", "CLOSING", "INTERSTITIAL",
                    "TITLE", "NOTE", "TARGET", "STYLE", "NARRATOR",
                }
                # BUG-LOCAL-091: also content-filter v5 bare NAME:
                # form. Captain-Eris-Violet emits lines like
                # `ALARM BLARING: warning text` or
                # `BACK AT THE LAB: descriptive narration` whose
                # first word ("ALARM", "BACK") slips past the
                # structural-token blocklist. Reject and route to SFX.
                if (
                    _v5_first_word
                    and _v5_first_word not in _v5_structural
                    and not _looks_like_non_character_cast_name(_v5_raw_name.upper())
                ):
                    _v5_emotion = (_m_v5.group(2) or "").strip()
                    _v5_dialogue = _m_v5.group(3).strip().strip('"*_\u201c\u201d')
                    if _v5_dialogue:
                        lines.append({
                            "type": "dialogue",
                            "character_name": _v5_raw_name.upper(),
                            "voice_traits": _v5_emotion or "unspecified",
                            "line": _v5_dialogue,
                        })
                        i += 1
                        continue
                elif _v5_first_word and _looks_like_non_character_cast_name(_v5_raw_name.upper()):
                    log.info(
                        "[ScriptParser] BUG-091: rejected '%s' as cast "
                        "(bare NAME: form was SFX/stage-direction); "
                        "routing to sfx",
                        _v5_raw_name.upper(),
                    )
                    lines.append({
                        "type": "sfx",
                        "description": _v5_raw_name.upper(),
                    })
                    i += 1
                    continue

            # Fallback: treat as structural direction
            if s and not s.startswith("#") and not s.startswith("---"):
                lines.append({"type": "direction", "text": s})
            i += 1

        malformed = _fallback_counter
        if malformed:
            log.warning(
                "[ScriptParser] %d malformed VOICE tag(s) detected (missing character name). "
                "Update SCRIPT_SYSTEM_PROMPT Section 1 example if this recurs.", malformed
            )

        # BUG-010 fix: hard-abort if extraction produced an empty / no-dialogue
        # script. Previously this silently passed ghost data into SceneSequencer
        # which then crashed Bark / video assembly with cryptic errors.
        dialogue_count = sum(1 for ln in lines if ln.get("type") == "dialogue")
        
        # v1.4 Theme B - Failsafe for 2B models that strip [VOICE:] tags
        # If no dialogue was found but we see `NAME: dialogue` structure, attempt recovery
        #
        # BUG-LOCAL-038: loosened from `dialogue_count == 0` to `< 3` AND raw
        # text has 5+ `NAME:` shape matches. The previous guard short-circuited
        # whenever any malformed VOICE tag registered as one dialogue token,
        # leaving 20+ bare `NAME:` dialogue lines uncovered. The raw-text
        # sanity check prevents the fallback from firing on scripts that are
        # genuinely dialogue-light (e.g. narration-only treatments).
        # 2026-04-26 BUG-LOCAL-066: added `_` -- aligns with the rest of
        # the pipeline so the loose-fallback trigger fires when underscored
        # names are present in the raw text.
        _raw_name_hits = len(re.findall(
            r'^(?:\*{0,2})[A-Z][A-Z0-9_ ]{1,25}(?:\*{0,2})\s*:\s+\S',
            text, re.MULTILINE,
        ))
        _orig_fallback_trigger = (dialogue_count == 0 and len(lines) > 0)
        _loose_fallback_trigger = (dialogue_count < 3 and _raw_name_hits >= 5)
        if _orig_fallback_trigger or _loose_fallback_trigger:
            if _loose_fallback_trigger and not _orig_fallback_trigger:
                log.warning(
                    "[ScriptParser] Only %d dialogue tokens but %d NAME: shape matches in raw text. "
                    "Attempting permissive 2B-fallback parse (BUG-LOCAL-038)...",
                    dialogue_count, _raw_name_hits,
                )
            else:
                log.warning("[ScriptParser] Zero standard tags found. Attempting permissive 2B-fallback parse...")
            _recovered = 0

            # Pass 1: Match 'NAME: dialogue' or '*NAME*: dialogue' or '**NAME:** dialogue'
            #         or 'NAME(angry): dialogue' or '*NAME*(angry): dialogue'
            # BUG-014 fix: accept 0-2 asterisks (not just 0 or 2) around names.
            # Maximum chaos creativity produces *NAME*(emotion): format with single asterisks.
            _structural_names = {
                "ENV", "SFX", "MUSIC", "BEAT", "PAUSE", "ACT", "SCENE",
                "TRANSITION", "CONTINUED", "CONT", "END", "FADE", "CUT",
                "INT", "EXT", "OPENING", "CLOSING", "INTERSTITIAL",
            }
            for ln in lines:
                if ln.get("type") == "direction":
                    text_d = ln["text"]
                    # Strip any leading/trailing asterisks or underscores from the direction text
                    text_d_clean = re.sub(r'^[*_]+\s*|\s*[*_]+$', '', text_d).strip()
                    m = re.match(r'^(?:\*{0,2})([A-Z][A-Z0-9 ]{0,19})(?:\*{0,2})(?:\s*\([^)]*\))?\s*:\s*(.+)$', text_d_clean)
                    if m:
                        cname = m.group(1).strip()
                        # Reject structural tag names that look like characters
                        if cname.split()[0] in _structural_names:
                            continue
                        ln["type"] = "dialogue"
                        ln["character_name"] = cname
                        ln["voice_traits"] = "unspecified"
                        ln["line"] = m.group(2).strip().strip('"*_\u201c\u201d')
                        _recovered += 1

            # Pass 2: Screenplay format (NeMo 12B natural style)
            # Matches: **NAME** or **NAME:** on its own line, followed by optional
            # (parenthetical), then dialogue text on subsequent line(s).
            if _recovered < 3:
                _screenplay_name_pat = re.compile(
                    r'^\*\*([A-Z][A-Z0-9_ ]{0,20})\*\*\s*:?\s*$'
                )
                _paren_pat = re.compile(r'^\(.*\)\s*$')
                # Structural lines that should NOT be treated as dialogue
                _structural_prefixes = (
                    "INT.", "EXT.", "===", "---", "[", "*", "ACT ", "SCENE ",
                    "FADE ", "CUT ", "END ", "TO BE", "**ACT", "**SCENE",
                )
                # Re-parse from raw text since the direction items lost structure
                raw_lines_2 = text.strip().splitlines()
                _new_items = []
                k = 0
                while k < len(raw_lines_2):
                    raw_s = raw_lines_2[k].strip()
                    # Strip markdown bold/italic wrappers
                    clean_s = re.sub(r'^[*_]+|[*_]+$', '', raw_s).strip()
                    nm = _screenplay_name_pat.match(raw_s)
                    if nm:
                        char_name = nm.group(1).strip().upper()
                        # Skip known structural words
                        _fw = char_name.split()[0] if char_name else ""
                        if _fw in ("ACT", "SCENE", "INT", "EXT", "FADE", "CUT",
                                   "END", "MUSIC", "SFX", "ENV", "BEAT", "PAUSE",
                                   "TRANSITION", "CONTINUED", "CONT"):
                            k += 1
                            continue
                        # Collect dialogue lines after the name
                        k += 1
                        # Skip optional parenthetical(s)
                        while k < len(raw_lines_2):
                            next_l = raw_lines_2[k].strip()
                            next_clean = re.sub(r'^[*_]+|[*_]+$', '', next_l).strip()
                            if _paren_pat.match(next_clean):
                                k += 1
                            else:
                                break
                        # Collect dialogue lines until we hit a blank, another name, or structural
                        _dial_parts = []
                        while k < len(raw_lines_2):
                            dl = raw_lines_2[k].strip()
                            dl_clean = re.sub(r'^[*_]+|[*_]+$', '', dl).strip()
                            if not dl_clean:
                                break
                            if _screenplay_name_pat.match(dl):
                                break
                            if _paren_pat.match(dl_clean):
                                k += 1
                                continue
                            if any(dl_clean.upper().startswith(p) for p in _structural_prefixes):
                                break
                            _dial_parts.append(dl_clean.strip('"\u201c\u201d'))
                            k += 1
                        if _dial_parts:
                            # Join multi-line dialogue into one
                            full_dialogue = " ".join(_dial_parts)
                            _new_items.append({
                                "type": "dialogue",
                                "character_name": char_name,
                                "voice_traits": "unspecified",
                                "line": full_dialogue,
                            })
                            _recovered += 1
                    else:
                        k += 1
                if _new_items:
                    # Replace the lines list with screenplay-parsed items
                    # Keep non-direction items (scene_break, sfx, etc.) and add new dialogue
                    structural = [ln for ln in lines if ln.get("type") != "direction"]
                    lines.clear()
                    lines.extend(structural)
                    lines.extend(_new_items)
                    log.info(f"[ScriptParser] Screenplay format: recovered {len(_new_items)} dialogue lines from **NAME** patterns")

            # Pass 3: Bare screenplay format (FormatNorm + Mistral 12B + plain
            # screenplay output). Like Pass 2 but without the `**` bold markdown
            # requirement. The FormatNorm pass strips the bold wrappers and
            # emits CHARACTER:\ndialogue\n\n -- which falls through Pass 1 (no
            # inline colon match because the dialogue is on the next line) and
            # Pass 2 (no `**` markers). Without this pass, a clean cyberpunk
            # 100-word run with self_critique=ON would PARSE_FATAL after
            # FormatNorm normalised the script (2026-05-09 incident).
            #
            # Differences from Pass 2:
            #   - No `**` required around name
            #   - Colon REQUIRED on the name line (prevents false-match against
            #     plain prose lines that happen to start with a capital)
            #   - Allows quotes / apostrophes / periods / hyphens inside the
            #     name so EDWARD "BEE" BEESLY:, MS. KIRBY:, etc. survive
            if _recovered < 3:
                _bare_name_pat = re.compile(
                    r'^([A-Z][A-Z0-9_ "\'\.\-]{0,40}?)\s*:\s*$'
                )
                # Reuse _paren_pat and _structural_prefixes from Pass 2 above
                raw_lines_3 = text.strip().splitlines()
                _new_items_3 = []
                k = 0
                while k < len(raw_lines_3):
                    raw_s3 = raw_lines_3[k].strip()
                    clean_s3 = re.sub(r'^[*_]+|[*_]+$', '', raw_s3).strip()
                    nm3 = _bare_name_pat.match(clean_s3)
                    if nm3:
                        char_name = nm3.group(1).strip().upper()
                        # Strip any embedded quotes from the canonical name so
                        # EDWARD "BEE" BEESLY -> EDWARD BEE BEESLY for cast
                        # matching downstream. Keep dialogue text intact.
                        char_name = re.sub(r'["\'“”]', '', char_name).strip()
                        char_name = re.sub(r'\s+', ' ', char_name)
                        # Skip structural words masquerading as character names
                        _fw3 = char_name.split()[0] if char_name else ""
                        if _fw3 in ("ACT", "SCENE", "INT", "EXT", "FADE", "CUT",
                                    "END", "MUSIC", "SFX", "ENV", "BEAT", "PAUSE",
                                    "TRANSITION", "CONTINUED", "CONT", "OPENING",
                                    "CLOSING", "INTERSTITIAL"):
                            k += 1
                            continue
                        # Collect dialogue lines after the name (same as Pass 2)
                        k += 1
                        while k < len(raw_lines_3):
                            next_l3 = raw_lines_3[k].strip()
                            next_clean3 = re.sub(r'^[*_]+|[*_]+$', '', next_l3).strip()
                            if _paren_pat.match(next_clean3):
                                k += 1
                            else:
                                break
                        _dial_parts3 = []
                        while k < len(raw_lines_3):
                            dl3 = raw_lines_3[k].strip()
                            dl_clean3 = re.sub(r'^[*_]+|[*_]+$', '', dl3).strip()
                            if not dl_clean3:
                                break
                            # Stop if next line is another bare-name marker
                            if _bare_name_pat.match(dl_clean3):
                                break
                            if _paren_pat.match(dl_clean3):
                                k += 1
                                continue
                            if any(dl_clean3.upper().startswith(p) for p in _structural_prefixes):
                                break
                            _dial_parts3.append(dl_clean3.strip('"“”'))
                            k += 1
                        if _dial_parts3:
                            full_dialogue3 = " ".join(_dial_parts3)
                            _new_items_3.append({
                                "type": "dialogue",
                                "character_name": char_name,
                                "voice_traits": "unspecified",
                                "line": full_dialogue3,
                            })
                            _recovered += 1
                    else:
                        k += 1
                if _new_items_3:
                    structural = [ln for ln in lines if ln.get("type") != "direction"]
                    lines.clear()
                    lines.extend(structural)
                    lines.extend(_new_items_3)
                    log.info(
                        "[ScriptParser] Bare screenplay format: recovered %d "
                        "dialogue lines from CHARACTER: NAME patterns "
                        "(2026-05-09 FormatNorm fix)",
                        len(_new_items_3),
                    )

            if _recovered > 0:
                log.info(f"[ScriptParser] Permissive fallback recovered {_recovered} dialogue lines!")
                dialogue_count = _recovered


        if not lines or dialogue_count == 0:
            log.critical(
                "[ScriptParser] FATAL: parsed %d structural lines but %d dialogue lines. "
                "Script extraction failed - refusing to pass empty data downstream.",
                len(lines), dialogue_count,
            )
            _runtime_log(
                f"PARSE_FATAL: lines={len(lines)} dialogue={dialogue_count} "
                f"raw_text_len={len(text)} - aborting"
            )
            with open("FAILED_SCRIPT_DUMP.txt", "w", encoding="utf-8") as f:
                f.write(text)
            raise ValueError(
                f"Script parser produced 0 dialogue lines from {len(text)}-char input. "
                "Aborting run to prevent silent audio failure."
            )

        # -- PRO QA: Flag missing ANNOUNCER bookends --
        # Detection only - actual injection happens at call site via _generate_announcer_bookends()
        # which has access to episode context (title, news, characters) for story-aware text.
        # Fallback canned injection kept for callers that don't use the LLM path (e.g. unit tests).
        dialogue_indices = [i for i, ln in enumerate(lines) if ln.get("type") == "dialogue"]
        if len(dialogue_indices) > 5:
            first_idx = dialogue_indices[0]
            last_idx = dialogue_indices[-1]

            if lines[first_idx]["character_name"] != "ANNOUNCER":
                log.warning("[ScriptParser] PRO QA: Missing ANNOUNCER opening - flagged for LLM repair")
                _runtime_log("QA_REPAIR: Missing ANNOUNCER opening - flagged for LLM generation")
                lines.insert(first_idx, {
                    "type": "dialogue",
                    "character_name": "ANNOUNCER",
                    "voice_traits": "male, 50s, authoritative, calm",
                    "line": "__NEEDS_LLM_OPENING__",
                })
                dialogue_indices = [i for i, ln in enumerate(lines) if ln.get("type") == "dialogue"]
                last_idx = dialogue_indices[-1]

            if lines[last_idx]["character_name"] != "ANNOUNCER":
                log.warning("[ScriptParser] PRO QA: Missing ANNOUNCER closing - flagged for LLM repair")
                _runtime_log("QA_REPAIR: Missing ANNOUNCER closing - flagged for LLM generation")
                lines.insert(last_idx + 1, {
                    "type": "dialogue",
                    "character_name": "ANNOUNCER",
                    "voice_traits": "male, 50s, authoritative, calm",
                    "line": "__NEEDS_LLM_CLOSING__",
                })

        return lines



__all__ = ["LegacyLLMScriptWriter"]


if __name__ == "__main__":
    import sys
    import pathlib

    try:
        from . import story_orchestrator as _so_test  # type: ignore
    except ImportError:
        _NODES_DIR = pathlib.Path(__file__).resolve().parent
        _COMFY_ROOT = _NODES_DIR.parent.parent.parent
        for _p in (str(_NODES_DIR), str(_COMFY_ROOT)):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        import story_orchestrator as _so_test  # type: ignore

    cls = LegacyLLMScriptWriter
    obj = cls()
    assert obj is not None
    print("[1/5] PASS: instantiation")

    it = cls.INPUT_TYPES()
    assert isinstance(it, dict) and "required" in it
    print("[2/5] PASS: INPUT_TYPES well-formed")

    assert cls.RETURN_TYPES == ("STRING", "STRING", "STRING", "INT")
    print("[3/5] PASS: RETURN_TYPES")

    assert cls.RETURN_NAMES == (
        "script_text", "script_json", "news_used", "estimated_minutes"
    )
    print("[4/5] PASS: RETURN_NAMES")

    assert hasattr(_so_test, "_runtime_log")
    print("[5/5] PASS: story_orchestrator helpers reachable")

    print("LEGACY SELF-TEST PASS: 5/5")
