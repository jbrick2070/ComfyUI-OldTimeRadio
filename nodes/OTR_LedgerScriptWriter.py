"""OTR_LedgerScriptWriter — v2.0 LPL writer with legacy-style widget surface restored 2026-05-10.

Pipeline (unchanged from v2.0 LPL):

    1. Validate + normalize inputs (legacy widget set restored).
    2. Resolve effective values:
       - news_seed = custom_premise verbatim if non-empty,
         else RSS auto-fetch via story_orchestrator._fetch_science_news.
       - style = style_custom if non-empty, else style combo (with
         "let the story decide" sentinel deferred to a two-pass
         picker once the LLM is loaded, see _otr_style_picker).
       - target_words from widget, optionally overridden by smoke
         target_length presets ("30 words", "tiny"). Words are the
         single canonical length unit for story writing; seconds is
         only computed post-hoc for the est_minutes output socket.
       - creativity → (temperature, top_p) preset map.
    3. Load LLM via _otr_model_loader.
    4. generate_outline (validated against OutlineSchema).
    5. new_ledger + episode_canon + set_cast.
    6. Per-beat loop:
         - character / announcer → compose_line (uses creativity temp/top_p)
         - non-voiced            → use beat.sfx_cue / beat.intent verbatim
    7. set_lines + speaker_role post-patch.
    8. Post-composition title regen (Jeffrey 2026-05-10): when the user
       left episode_title blank, ask the LLM to title the episode from
       the FINAL assembled dialogue. The prompt sees ONLY the composed
       dialogue -- not news_seed, not style, not outline metadata.
       User-typed title still wins; outline.title is the last-resort
       fallback if the LLM call fails or its output is rejected by the
       guardrails. canon.title is updated and episode_canon.json is
       written here (deferred from step 5 specifically for this).
    8b. Spoken-title substitution: the per-line composer ran in step 6
        with canon_header containing the outline TITLE, so the
        announcer / character lines may have baked the OLD title into
        spoken dialogue ("Tonight on: <outline.title>"). When the regen
        produced a different final title, substitute case-insensitive
        whole-phrase occurrences of the old title with the new title
        across ledger.lines[].text AND script_text_parts. Stamps
        meta.title_substitution for forensics.
    9. Stamp meta block (gen_params_initial, episode_title, title_source,
       title_substitution, perfect_run_spacesaver, creativity,
       optimization_profile).
   10. Save ledger.

Output contract (UNCHANGED from prior v2.0):
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("script_text", "script_json", "news_used",
                    "estimated_minutes")

Widget surface (post-Phase-3 cleanup 2026-05-11):
    required:
        episode_title     STRING  (optional override; empty -> LLM regen
                                   from final dialogue post-composition)
        target_words      INT     (canonical length unit; radio ~140 wpm
                                   conversion is only for the est_minutes
                                   output, never for story planning)
        num_characters    INT     (1-6 speaking characters; 1 = monologue)
    optional:
        seed              INT     (C7 byte-identity seed; shuffle-on
                                   randomizes per Queue Prompt)
        model_id          combo   (HF model -- story LLM)
        custom_premise    STRING  (RSS override; empty triggers feed fetch)
        include_act_breaks BOOLEAN (True -> outline LLM plans music_inter
                                    beats between acts; False -> continuous)
        act_count         INT     (1-7 acts; 0 = auto-derive default from
                                   target_words. Live-clamped to
                                   [default..max] by the JS extension at
                                   web/js/otr_act_count_widget.js)
        style             combo   (tonal preset; "let the story decide"
                                   defers to two-pass LLM picker)
        style_custom      STRING  (free-text override; empty falls back
                                   to style combo)
        creativity        combo   (maps to temperature + top_p preset)
        optimization_profile combo (VRAM-tier; only Standard validated
                                    today, others fall back to Standard)
        perfect_run_spacesaver BOOLEAN (stamped on ledger.meta for
                                        RTXUpscale spacesaver)

Notes:
    - news_seed RSS fetch lifted from story_orchestrator._fetch_science_news.
      Feeds, dedup, style-aware re-ranking all reused as-is. Falls back to a
      deterministic synthetic seed only when feedparser is unavailable or
      every feed times out.
    - open_close DROPPED per user 2026-05-10 — the 3-spine evaluator added
      ~4 extra LLM passes and v2 LPL outline pipeline doesn't need it.
    - No edits to any other shipped file (_otr_outline, _otr_canon,
      _otr_line_composer, _otr_model_loader, production_ledger, _otr_ledger).
    - No model loads / GPU at import.
    - UTF-8 no BOM. Safe-for-work content.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerScriptWriter"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOICED_ROLES = {"character", "announcer"}
"""Speaker roles that produce spoken dialogue. These trigger an LLM
compose_line call. Other roles (music_*, sfx) skip the LLM."""

NON_VOICED_ROLES = {"music_open", "music_close", "music_inter", "sfx"}
"""Speaker roles that render as [SFX: ...] tokens with no LLM call."""

DEFAULT_TRAITS = "neutral"
"""Fallback traits string when a beat has no mood. Mirrors the
'traits = beat.mood or "neutral"' rule from the kickoff prompt."""

DEFAULT_MODEL_ID = "mistralai/Mistral-Nemo-Instruct-2407"
"""Default story-writing LLM. Validated production path (cleared
BUG-061/062/063 format hardening)."""

LAST_LINES_WINDOW = 5
"""Rolling context window size for compose_line. Each character /
announcer beat appends to the window; non-voiced beats do not.

Phase 1 (2026-05-11): bumped from 3 to 5 per synthesis §6.D --
Mistral-Nemo handles the wider window cleanly within the 800-tok
composer prompt budget, and the extra context smooths line-to-line
voice consistency (especially in multi-character scenes where the
prior 3-line window often dropped one speaker's last beat)."""

WORD_BUDGET_RATIO_LO = 0.7
WORD_BUDGET_RATIO_HI = 1.3
"""Acceptable band for sum(beat.target_words) / target_words. Outside
this band logs WARNING but does not fail the run."""

WORDS_PER_MINUTE_ESTIMATE = 140
"""Word-per-minute estimate for the est_minutes output socket only.
Story planning is words-only; this constant is never used to derive
a target_seconds input to the LLM. Mirrors legacy at
story_orchestrator.py:6584."""


# ---------------------------------------------------------------------------
# Creativity preset maps (lifted verbatim from legacy at
# _otr_legacy_writer.py:755-768; BUG-014 chaos clamp preserved)
# ---------------------------------------------------------------------------

_CREATIVITY_TEMP_MAP = {
    "safe & tight":   0.6,
    "balanced":       0.85,
    "wild & rough":   0.92,
    "maximum chaos":  0.95,  # BUG-014: 1.35 caused total format collapse
}

_CREATIVITY_TOP_P_MAP = {
    "safe & tight":   0.9,
    "balanced":       0.95,
    "wild & rough":   0.98,
    "maximum chaos":  0.99,
}

_CREATIVITY_CHOICES = list(_CREATIVITY_TEMP_MAP.keys())


# target_length widget removed 2026-05-11 (post-Phase-3 cleanup pass).
# The old "short (3 acts)" / "medium (5 acts)" / "long (7-8 acts)" combo
# is replaced by the typed `act_count` widget + `target_words` (driven
# by web/js/otr_act_count_widget.js). Smoke presets are gone with it --
# for a 30-word smoke run, type target_words=30 directly. Cleaner UX,
# one source of truth for episode shape.


# ---------------------------------------------------------------------------
# Style widget surface — three-way (Jeffrey 2026-05-10):
#   1. Free-text override (`style_custom`) wins when non-empty.
#   2. `style` combo set to "let the story decide" -> LLM analyzes the
#      news story and proposes a 3-6 word tonal descriptor; that
#      descriptor flows into both character generation and script
#      generation. This is the SAVED DEFAULT in
#      workflows/otr_scifi_16gb_full.json so a fresh load runs the
#      auto-derive path with no user intervention.
#   3. Any other combo entry -> used verbatim.
# Both axes — story (custom_premise/RSS) AND style (combo/auto/custom) —
# drive story content; the user wants both selectable.
# ---------------------------------------------------------------------------

_STYLE_AUTO_SENTINEL = "let the story decide"

_STYLE_CHOICES = [
    _STYLE_AUTO_SENTINEL,
    "closed room suspense",
    "detective case file",
    "pulp serial cliffhanger",
    "mission control procedural",
    "deep space distress call",
    "noir interrogation",
    "small town uncanny",
    "radio newsroom emergency",
    "haunted broadcast signal",
    "laboratory containment",
]

_LLM_STYLE_FALLBACK = "mission control procedural"
"""Hardcoded slug used by the RSS reranker (`_fetch_rss_seed_or_die`)
ONLY, when style is still pending (sentinel selected, picker hasn't
fired yet — chicken-and-egg: RSS fetch happens BEFORE the style
picker since the picker needs the article to derive style from).
NOT used as a fallback by the picker itself — see
`_otr_style_picker.pick_style`, which raises
`StyleGenerationFailedError` on any failure path per Jeffrey
2026-05-10: a failed picker fails the workflow."""


# Pool fed to the two-pass style picker as "seed flavors" (inspiration
# only; not echoed back). Same 10 slugs as the user-facing dropdown
# minus the auto-sentinel — the sentinel is a UX label, not a style.
# Random sample of 5 per call (deterministic via writer's seed RNG
# for C7 byte-identity).
_STYLE_PICKER_SEED_POOL: tuple[str, ...] = (
    "closed_room_suspense",
    "detective_case_file",
    "pulp_serial_cliffhanger",
    "mission_control_procedural",
    "deep_space_distress_call",
    "noir_interrogation",
    "small_town_uncanny",
    "radio_newsroom_emergency",
    "haunted_broadcast_signal",
    "laboratory_containment",
)


# Voice-path-cleanbreak Sprint 6.1 (2026-05-12). Genre table for the
# meta.visual_plan.genre stamp. Replaces the hardcoded "audio drama"
# fallback Sprint 2 used. The genre string surfaces in:
#   - SignalLostVideo HUD overlay
#   - FLUX scene-prompt composition (style_tail + genre)
#   - episode metadata (treatment txt, video info card)
#
# Drift guard: tests/test_musicgen_style_palette.py asserts every
# entry in _STYLE_PICKER_SEED_POOL has an explicit row in this table
# (mechanical fallback below is a safety net, not the contract).
_GENRE_BY_STYLE: dict[str, str] = {
    "closed_room_suspense":       "thriller audio drama",
    "detective_case_file":        "detective audio drama",
    "pulp_serial_cliffhanger":    "pulp serial audio drama",
    "mission_control_procedural": "procedural audio drama",
    "deep_space_distress_call":   "sci-fi audio drama",
    "noir_interrogation":         "noir audio drama",
    "small_town_uncanny":         "uncanny audio drama",
    "radio_newsroom_emergency":   "newsroom audio drama",
    "haunted_broadcast_signal":   "horror audio drama",
    "laboratory_containment":     "containment audio drama",
}


def _resolve_genre(style: str) -> str:
    """Resolve a style slug to a HUD/FLUX-friendly genre string.

    Standing directive (no silent fallbacks): unknown style slugs use
    a mechanical "<words> audio drama" fallback that's loud (visibly
    non-curated, suggests the slug needs an explicit table entry) but
    never empty. Drift guard in tests/test_musicgen_style_palette.py
    catches any new _STYLE_PICKER_SEED_POOL entry that's missing here.
    """
    if style in _GENRE_BY_STYLE:
        return _GENRE_BY_STYLE[style]
    words = (style or "").replace("_", " ").strip()
    return f"{words} audio drama" if words else "audio drama"


# ---------------------------------------------------------------------------
# Title regeneration (post-composition, news-seed-free per Jeffrey 2026-05-10)
# ---------------------------------------------------------------------------

_STUCK_TITLE_DEFAULTS = frozenset({
    "",
    "the last frequency",
    "untitled",
    "episode",
    "signal lost",
    "custom episode",
    "pending",
    "(pending)",
})
"""Reject set for post-composition title regen. Mirrors the legacy
story_orchestrator._STUCK_TITLE_DEFAULTS set, plus "(pending)" guard
in case a future canon_header placeholder leaks into the LLM output."""

_TITLE_PREFIX_RE = None  # compiled lazily inside the helper to keep
                          # this module's import surface stdlib-only.


# ---------------------------------------------------------------------------
# Model dropdown (matches legacy + cleanup model dropdown)
# ---------------------------------------------------------------------------

_MODEL_CHOICES = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen2.5-14B-Instruct [ALPHA]",
    "Nitral-AI/Captain-Eris_Violet-V0.420-12B (EXPERIMENTAL)",
    "inflatebot/MN-12B-Mag-Mell-R1 (EXPERIMENTAL)",
]

_OPTIMIZATION_PROFILE_CHOICES = [
    "Standard",
    "Pro (Ultra Quality)",
    "Obsidian (UNSTABLE/4GB)",
]


# ---------------------------------------------------------------------------
# Truncating generate_fn wrapper (top_p parametrized 2026-05-10)
# ---------------------------------------------------------------------------


# Tier 2 fix #16 (2026-05-11): rolling-buffer StoppingCriteria
# class, hoisted to module scope so it is defined ONCE per process
# rather than once per generate_fn build. transformers stays a lazy
# import — the class is constructed only the first time a stop=
# kwarg is non-empty, then cached.
_SUBSTRING_STOP_CLASS = None


def _get_substring_stop_class():
    """Return (and lazily build + cache) the _SubstringStop class.

    The rolling buffer cuts per-step decode cost ~50x vs the previous
    "decode last 64 tokens every step" approach. We only decode the
    tokens newly emitted since the prior __call__, append to a
    running tail capped at `tail_window` chars, and substring-match
    each stop string against the tail.
    """
    global _SUBSTRING_STOP_CLASS  # noqa: PLW0603
    if _SUBSTRING_STOP_CLASS is not None:
        return _SUBSTRING_STOP_CLASS
    from transformers import StoppingCriteria  # type: ignore

    class _SubstringStop(StoppingCriteria):
        def __init__(
            self,
            tokenizer,
            stops: tuple[str, ...],
            prompt_len: int,
            tail_window: int = 64,
        ) -> None:
            super().__init__()
            self._tok = tokenizer
            self._stops = stops
            self._last_seen = int(prompt_len)
            self._tail = ""
            self._tail_window = int(tail_window)

        def __call__(self, input_ids, scores, **kwargs):  # noqa: D401
            ids = input_ids[0]
            cur_len = int(ids.shape[0])
            if cur_len <= self._last_seen:
                return False
            new_ids = ids[self._last_seen:cur_len]
            self._last_seen = cur_len
            try:
                new_text = self._tok.decode(
                    new_ids, skip_special_tokens=True,
                )
            except Exception:  # noqa: BLE001
                return False
            self._tail = (self._tail + new_text)[-self._tail_window:]
            return any(s in self._tail for s in self._stops)

    _SUBSTRING_STOP_CLASS = _SubstringStop
    return _SUBSTRING_STOP_CLASS


def _build_truncating_generate_fn(
    cache_entry: dict,
    *,
    top_p: float = 0.92,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
):
    """Return a generate_fn that left-truncates oversized prompts and
    forwards sampling controls to model.generate.

    Closure captures the four episode-level sampling knobs from the
    writer widgets: top_p, min_p, repetition_penalty. The per-call
    args (`temperature`, `max_new_tokens`, optional `stop`) are
    whatever the line composer / outline / picker passes.

    Phase 4 v4 (2026-05-11): min_p and repetition_penalty added as
    closure-captured params, plus per-call `stop` support via a
    StoppingCriteria subclass that matches on substring at the tail
    of the decoded output. Defaults are conservative for the 7B-14B
    class:
      top_p              = 0.92   (current default, preserved)
      min_p              = 0.0    (disabled; 0.05 is the safe non-
                                   trivial improvement)
      repetition_penalty = 1.0    (disabled; 1.03 is gentle and
                                   doesn't damage short outputs)
    Each widget overrides per-episode from the workflow.

    Cap math: max_input_tokens = max(64, context_cap - max_new_tokens).
    Truncation is left-side (drops oldest tokens, preserves most
    recent context).
    """
    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]
    context_cap = int(cache_entry.get("context_cap") or 8192)
    active_top_p = float(top_p)
    active_min_p = float(min_p or 0.0)
    active_rep_penalty = float(repetition_penalty or 1.0)
    # Tier 1 fix #8 (2026-05-11): one-shot warning + auto-fallback
    # for transformers versions < 4.43 that don't accept `min_p` as
    # a kwarg on model.generate. Closure-scoped mutable cell so the
    # disable persists across calls within one run without spamming
    # the warning more than once.
    _min_p_unsupported = [False]

    def generate_fn(messages, *, temperature, max_new_tokens, stop=None):
        import torch  # local import; never load torch at module import
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        max_input_tokens = max(64, context_cap - int(max_new_tokens))
        input_len = inputs["input_ids"].shape[-1]
        if input_len > max_input_tokens:
            trunc = input_len - max_input_tokens
            inputs["input_ids"] = inputs["input_ids"][:, trunc:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, trunc:]
            log.warning(
                "[OTR_LedgerScriptWriter] PROMPT_GUARD: Truncated "
                "%d -> %d tokens (context_cap=%d, max_new_tokens=%d)",
                input_len, max_input_tokens, context_cap, max_new_tokens,
            )

        gen_kwargs = {
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": active_top_p,
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": tokenizer.eos_token_id,
        }
        # Only forward non-default values so older transformers
        # versions that don't accept `min_p` as a kwarg keep working
        # silently when the widget is at its disabled default.
        if active_min_p > 0.0 and not _min_p_unsupported[0]:
            gen_kwargs["min_p"] = active_min_p
        if active_rep_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = active_rep_penalty

        # Stop-string support (Phase 4 v4). Tier 2 fix #16
        # (2026-05-11): the StoppingCriteria subclass is now defined
        # once at module scope by `_get_substring_stop_class()` and
        # reuses a rolling buffer instead of decoding the last 64
        # tokens every step. Falls back silently on import error
        # (stop strings are quality nice-to-have, not correctness).
        if stop:
            try:
                from transformers import (  # noqa: I001
                    StoppingCriteriaList,
                )
                prompt_len_now = inputs["input_ids"].shape[1]
                stop_strings = tuple(s for s in stop if s)
                _SubstringStop = _get_substring_stop_class()
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                    _SubstringStop(
                        tokenizer, stop_strings, prompt_len_now,
                    ),
                ])
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "[OTR_LedgerScriptWriter] stop-strings disabled: %s",
                    exc,
                )

        with torch.no_grad():
            try:
                out = model.generate(**inputs, **gen_kwargs)
            except TypeError as exc:
                # Tier 1 fix #8: min_p kwarg unsupported on
                # transformers < 4.43. Warn once and retry without it
                # for the rest of this run.
                if "min_p" in gen_kwargs and "min_p" in str(exc):
                    log.warning(
                        "[OTR_LedgerScriptWriter] min_p kwarg not "
                        "supported by this transformers version; "
                        "disabling for the remainder of this run "
                        "(error was: %s)",
                        exc,
                    )
                    _min_p_unsupported[0] = True
                    gen_kwargs.pop("min_p", None)
                    out = model.generate(**inputs, **gen_kwargs)
                else:
                    raise
        prompt_len = inputs["input_ids"].shape[1]
        decoded = tokenizer.decode(
            out[0][prompt_len:], skip_special_tokens=True,
        )
        # Tier 1 fix #5 (2026-05-11): StoppingCriteria halts
        # generation but leaves the trigger bytes in the output
        # buffer. Slice at the first stop substring so leaked
        # bracketed/parenthesized tails don't survive into the
        # composer's strip_line_formatting -> ledger pipeline. With
        # polish OFF (default), this is the last guard before the
        # text lands. Earliest-match wins.
        if stop:
            cut = len(decoded)
            for s in stop:
                if not s:
                    continue
                idx = decoded.find(s)
                if idx >= 0 and idx < cut:
                    cut = idx
            decoded = decoded[:cut]
        return decoded

    return generate_fn


# ---------------------------------------------------------------------------
# Pure helpers (testable without model load)
# ---------------------------------------------------------------------------


# NOTE: the prior single-shot picker `_generate_style_via_llm` was
# replaced by the two-pass picker in `nodes/_otr_style_picker.py`
# (commit landing 2026-05-10). The two-pass design (Pass 1 inventor
# producing 5 distinct candidates + Pass 2 chooser picking one)
# fixed the mode-collapse problem the single-shot picker suffered
# from -- every Mistral-Nemo run defaulted to "tense industrial
# procedural" or close. The fail-loud policy from commit 62e85f2
# carries through: any picker failure raises
# `_otr_style_picker.StyleGenerationFailedError` and halts the
# workflow. See the picker module for design rationale.


def _generate_title_from_script(
    generate_fn,
    assembled_script: str,
    *,
    temperature: float = 0.85,
) -> str:
    """Generate a 2-5 word episode title from the FINAL composed dialogue.

    Per Jeffrey 2026-05-10: "title should generate only AFTER the whole
    story is done via the LLM, nothing with the news seed". So the prompt
    sees ONLY the assembled dialogue. No news_seed, no style hint, no
    outline metadata. The intent is a title grounded purely in what the
    listener will actually hear.

    Returns the cleaned title, or empty string on any failure (LLM raise,
    stuck-default rejection, overlong leak, smart-quote-only wrappers
    that strip to nothing). Caller falls back to outline.title on "".

    `generate_fn` matches the (messages, *, temperature, max_new_tokens)
    contract returned by `_build_truncating_generate_fn`.

    Temperature is clamped to [0.4, 1.0] regardless of caller value to
    keep title output stable (legacy parity at
    _otr_legacy_writer.py:2987).
    """
    import re

    text = (assembled_script or "").strip()
    if not text:
        return ""

    # Cap the dialogue excerpt. Title-generation only needs broad strokes
    # of the story; full transcripts blow the context budget on long runs.
    excerpt = text[:3000]

    sys_msg = (
        "You are titling a single episode of a sci-fi radio drama. "
        "You receive the final dialogue transcript and propose an "
        "evocative 2-5 word episode title."
    )
    user_msg = (
        f"Final episode dialogue:\n{excerpt}\n\n"
        "Write ONE evocative episode title in 2 to 5 words. The title must:\n"
        " - draw from a vivid image, key object, character, or thematic "
        "tension actually present in the dialogue\n"
        " - feel specific and memorable, not generic\n"
        " - avoid cliches like \"The Beginning\", \"Final Chapter\", "
        "\"Untitled\", or \"Episode X\"\n\n"
        "Output ONLY the title text on a single line. No quotes. No "
        "preamble. No explanation."
    )

    clamped_temp = max(0.4, min(1.0, float(temperature)))

    try:
        raw = generate_fn(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=clamped_temp,
            max_new_tokens=24,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_LedgerScriptWriter] title LLM-regen failed (%s); "
            "caller will fall back to outline.title",
            exc,
        )
        return ""

    if not raw:
        return ""

    # Take first non-empty line.
    candidate = ""
    for ln in raw.strip().splitlines():
        ln = ln.strip()
        if ln:
            candidate = ln
            break
    if not candidate:
        return ""

    # Strip "Title:" / "**Title:**" / "TITLE:" wrappers.
    candidate = re.sub(
        r'^\s*(?:\*\*)?\s*(?:TITLE|Title|title)\s*:\s*(?:\*\*)?\s*',
        '', candidate,
    )

    # Iteratively strip ASCII + smart quotes, asterisks, whitespace.
    _wrap_chars = '"“”‘’*\' \t'
    prev = None
    while candidate != prev:
        prev = candidate
        candidate = candidate.strip(_wrap_chars)

    # Trailing punctuation often leaks from the model.
    candidate = candidate.rstrip(".,;:!?")
    candidate = candidate.strip()

    if not candidate:
        return ""

    # Reject stuck defaults.
    if candidate.lower() in _STUCK_TITLE_DEFAULTS:
        log.info(
            "[OTR_LedgerScriptWriter] title regen rejected stuck default: %r",
            candidate,
        )
        return ""

    # Reject full-sentence leaks. Legacy threshold = 10 words; mirror it.
    word_count = len(candidate.split())
    if word_count > 10:
        log.info(
            "[OTR_LedgerScriptWriter] title regen rejected overlong (%d "
            "words): %r",
            word_count, candidate,
        )
        return ""

    # Outline.title schema allows 3-80 chars; enforce upper bound here
    # too so the regenerated title stays drop-in compatible with the
    # canon.title field downstream.
    if len(candidate) > 80:
        candidate = candidate[:80].rstrip()

    log.info(
        "[OTR_LedgerScriptWriter] title regen -> %r (from %d-char script)",
        candidate, len(text),
    )
    return candidate


_TITLE_SUB_MIN_LEN = 4
"""Minimum length of old title to attempt substitution. Guards against
false matches on tiny outline titles like "ICE" or "Sun" hitting
unrelated dialogue words."""


def _substitute_title_in_text(
    text: str,
    old_title: str,
    new_title: str,
) -> tuple[str, int]:
    """Case-insensitive whole-phrase substitution of ``old_title`` with
    ``new_title`` in ``text``. Returns ``(new_text, n_subs)``.

    Whole-phrase: anchored with negative-lookbehind / negative-lookahead
    on word characters so "Pulse" doesn't match "Pulsewave". Min-length
    guard prevents short common-word titles from spuriously matching.
    No-ops when old_title == new_title or either is empty.
    """
    import re as _re

    if not text or not old_title or not new_title:
        return (text or "", 0)
    if old_title == new_title:
        return (text, 0)
    if len(old_title) < _TITLE_SUB_MIN_LEN:
        return (text, 0)

    patt = _re.compile(
        r"(?<!\w)" + _re.escape(old_title) + r"(?!\w)",
        flags=_re.IGNORECASE,
    )
    new_text, n_subs = patt.subn(new_title, text)
    return (new_text, n_subs)


def _resolve_creativity(creativity: str) -> tuple[float, float]:
    """Map a creativity widget value to (temperature, top_p).

    Unknown values default to balanced (0.85 / 0.95). Returns floats.
    """
    temp = _CREATIVITY_TEMP_MAP.get(creativity, _CREATIVITY_TEMP_MAP["balanced"])
    top_p = _CREATIVITY_TOP_P_MAP.get(creativity, _CREATIVITY_TOP_P_MAP["balanced"])
    return (float(temp), float(top_p))


def _resolve_target_words(target_words) -> int:
    """Clamp target_words to the schema minimum.

    Smoke-preset target_length override path removed 2026-05-11
    (post-Phase-3 cleanup) along with the target_length widget. For
    a smoke run type target_words=30 directly.
    """
    return max(5, int(target_words))


def _fetch_rss_seed_or_die(style: str, model_id: str) -> dict:
    """Run the story_orchestrator RSS fetcher and return the article dict.

    Lifts the exact path the legacy writer used. `style` is mapped to
    the closest legacy slug for the LLM re-rank step; if the fetcher
    returns None (every feed failed) we raise loudly -- the legacy
    writer behaved the same.

    Return shape (commit 3 of the news_interpreter sprint, ADR
    docs/news_interpreter_adr.md section 9.1): a dict with keys
    ``headline``, ``summary``, ``full_text``, ``source``, ``date``,
    ``link``, plus a computed ``seed_text`` for back-compat with
    consumers that still treat news_seed as a plain string. Previously
    this function returned only the seed_text string; the richer
    return lets the news_interpreter stage read the full body that
    the cast LLM never sees today.
    """
    try:
        try:
            from . import story_orchestrator as _so
        except ImportError:
            import story_orchestrator as _so  # type: ignore
        # Style normalization: re-ranker expects a slug like "hard_sci_fi";
        # use the closest match or fall back to the canonical default.
        slug = (style or "").lower().replace(" ", "_").replace("-", "_")
        if slug not in {
            "closed_room_suspense",
            "detective_case_file",
            "pulp_serial_cliffhanger",
            "mission_control_procedural",
            "deep_space_distress_call",
            "noir_interrogation",
            "small_town_uncanny",
            "radio_newsroom_emergency",
            "haunted_broadcast_signal",
            "laboratory_containment",
        }:
            slug = "mission_control_procedural"
        news = _so._fetch_science_news(
            max_feeds=10, style=slug, model_id=model_id,
            optimization_profile="Standard",
        )
        if not news:
            raise RuntimeError(
                "RSS fetcher returned no articles (all feeds failed or "
                "all candidates already used)"
            )
        # news is a list[dict] like [{headline, summary, full_text, source, link, date}, ...]
        # The orchestrator returns either a list or a single dict depending
        # on version. Normalize both shapes.
        if isinstance(news, dict):
            article = news
        elif isinstance(news, list) and news:
            article = news[0]
        else:
            raise RuntimeError(f"unexpected fetcher return shape: {type(news).__name__}")
        seed_text = " ".join(filter(None, [
            (article.get("headline") or "").strip(),
            (article.get("summary") or "").strip(),
        ]))
        if not seed_text:
            seed_text = (article.get("full_text") or "").strip()
        if not seed_text:
            raise RuntimeError("fetched article had empty headline/summary/full_text")
        log.info(
            "[OTR_LedgerScriptWriter] RSS_FETCH OK: source=%s, len=%d, head=%r",
            article.get("source") or "?", len(seed_text), seed_text[:80],
        )
        return {
            "headline":  (article.get("headline") or "").strip(),
            "summary":   (article.get("summary") or "").strip(),
            "full_text": (article.get("full_text") or "").strip(),
            "source":    (article.get("source") or "").strip(),
            "date":      (article.get("date") or "").strip(),
            "link":      (article.get("link") or "").strip(),
            "seed_text": seed_text,
        }
    except Exception as exc:
        # Loud raise: the writer requires a real seed to function. The
        # workflow can override via custom_premise if RSS is unavailable.
        raise RuntimeError(
            f"[OTR_LedgerScriptWriter] RSS fetch failed: {exc}. "
            f"Type a non-empty value into the `custom_premise` widget to "
            f"bypass the RSS pipeline.",
        ) from exc


def _resolve_inputs(
    episode_title: str = "",
    target_words: int = 350,
    num_characters: int = 2,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    custom_premise: str = "",
    include_act_breaks: bool = True,
    act_count: int = 0,
    style: str = _STYLE_AUTO_SENTINEL,
    style_custom: str = "",
    creativity: str = "balanced",
    optimization_profile: str = "Standard",
    perfect_run_spacesaver: bool = False,
    # Phase 4 v4 (2026-05-11) sampling knobs. Tier 2 fix #17
    # defaults flipped to 0.05 / 1.03 (validated improvement over
    # disabled baseline on the small-LLM class).
    min_p: float = 0.05,
    repetition_penalty: float = 1.03,
    max_new_tokens_cap: int = 200,
    enable_polish_pass: bool = False,
) -> dict:
    """Resolve raw widget values into the effective set used by the run.

    Returns a single dict. Logs at INFO for branches that override the
    widget value (RSS fetch, smoke preset, style_custom override).

    Both story and style follow the same dual-axis pattern (per Jeffrey
    2026-05-10 "there is the story, and the style — those two drive
    story content so we need both"):

      - story:   custom_premise verbatim > RSS auto-fetch.
      - style:   style_custom verbatim > `style` combo verbatim, EXCEPT
                 when combo == `_STYLE_AUTO_SENTINEL`, in which case
                 the writer defers to ``_generate_style_via_llm``
                 once the model is loaded. This helper returns
                 ``style_pending=True`` on the dict in that case so
                 the caller knows to call the LLM.

    Resolution order for the final style string:
      1. style_custom (free-text, takes precedence)
      2. style combo verbatim if != _STYLE_AUTO_SENTINEL
      3. LLM-generated (caller fills `resolved["style"]` post-load)
    """
    target_words = _resolve_target_words(target_words)
    num_characters = max(1, min(6, int(num_characters)))

    # Phase 2A (2026-05-11): act_count resolution. 0 (default) means
    # auto-derive via _otr_episode_budget.default_act_count. Any
    # non-zero value gets validated by compute_episode_budget in
    # run() -- including the [default..max] band check.
    act_count_int = max(0, min(7, int(act_count or 0)))
    if act_count_int == 0:
        try:
            from . import _otr_episode_budget as _OTRB  # type: ignore
            act_count_int = _OTRB.default_act_count(target_words)
        except Exception as exc:  # noqa: BLE001
            # If target_words is below 30, default_act_count raises;
            # fall through to act_count=1 and let compute_episode_budget
            # surface the structured InvalidEpisodeBudgetError in run().
            log.warning(
                "[OTR_LedgerScriptWriter] act_count auto-derive failed "
                "(target_words=%d): %s -- defaulting to 1",
                target_words, exc,
            )
            act_count_int = 1
    temperature, top_p = _resolve_creativity(creativity)
    custom = (custom_premise or "").strip()

    sc = (style_custom or "").strip()
    style_combo = (style or "").strip()
    if sc:
        resolved_style = sc
        style_source = "style_custom"
        style_pending = False
    elif style_combo and style_combo != _STYLE_AUTO_SENTINEL:
        resolved_style = style_combo
        style_source = "style_combo"
        style_pending = False
    else:
        # auto / empty -> defer to LLM post-load.
        resolved_style = ""        # caller fills
        style_source = "llm_auto"
        style_pending = True

    if custom:
        # Custom premise path: synthesize the same dict shape RSS
        # would produce so news_interpreter sees a uniform article
        # surface no matter how the story entered the writer.
        news_article = {
            "headline":  "",
            "summary":   "",
            "full_text": custom,
            "source":    "User Seed",
            "date":      "",
            "link":      "",
            "seed_text": custom,
        }
        news_seed = custom
        seed_source = "custom_premise"
    else:
        # Best-effort RSS re-rank slug. If style is still pending
        # (auto/LLM), use the hardcoded fallback only for the slug --
        # the writer's final style still gets LLM-proposed from the
        # ACTUAL fetched article below.
        rss_style_slug = resolved_style or _LLM_STYLE_FALLBACK
        news_article = _fetch_rss_seed_or_die(rss_style_slug, model_id)
        news_seed = news_article["seed_text"]
        seed_source = "rss_fetch"

    return {
        "news_seed":            news_seed,
        "news_article":         news_article,
        "seed_source":          seed_source,
        "style":                resolved_style,
        "style_source":         style_source,
        "style_pending":        style_pending,
        "style_combo":          style_combo,
        "style_custom":         sc,
        "target_words":         target_words,
        "num_characters":       num_characters,
        "episode_title":        (episode_title or "").strip(),
        "model_id":             str(model_id or DEFAULT_MODEL_ID).strip(),
        "include_act_breaks":   bool(include_act_breaks),
        "act_count":            int(act_count_int),
        "creativity":           str(creativity),
        "temperature":          float(temperature),
        "top_p":                float(top_p),
        "optimization_profile": str(optimization_profile),
        "perfect_run_spacesaver": bool(perfect_run_spacesaver),
        # Phase 4 v4 (2026-05-11) sampling knobs. Clamped to widget
        # ranges so a hand-edited workflow JSON can't slip through
        # out-of-band values.
        "min_p":                max(0.0, min(0.5, float(min_p or 0.0))),
        "repetition_penalty":   max(1.0, min(1.2, float(
            repetition_penalty or 1.0,
        ))),
        "max_new_tokens_cap":   max(40, min(400, int(
            max_new_tokens_cap or 200,
        ))),
        "enable_polish_pass":   bool(enable_polish_pass),
    }


def _derive_prev_speaker(
    last_lines: list,
    current_speaker: str,
) -> str:
    """Walk `last_lines` in reverse, return the first speaker NAME
    that is not the current speaker and not "ANNOUNCER".

    Tier 1 fix #4 (2026-05-11). Pre-Tier-1 every LineRequest set
    `prev_speaker = last_lines[-1][0]` which, when the rolling window
    ended on an announcer beat, produced "You are ALICE. You are
    responding to ANNOUNCER." — which breaks the fictional layer
    (characters in radio drama don't hear the narrator).

    The walk skips:
      - empty / blank names
      - "ANNOUNCER" (any case)
      - the current speaker (no "responding to yourself" two-line
        monologues; the WRITE LINE block already drops the clause
        in that case but we belt-and-brace it here)

    Returns "" when no qualifying speaker is found (first character
    line of a scene, or scene composed entirely of self + announcer
    so far). Empty string drops the "You are responding to ..."
    clause cleanly in `_build_user_prompt`.

    Inputs:
      last_lines       writer's rolling window: list[(speaker, text)]
      current_speaker  the speaker we are writing the line FOR

    Pure stdlib, no LLM cost. Never raises.
    """
    cur_u = (current_speaker or "").strip().upper()
    for entry in reversed(last_lines or []):
        if not entry:
            continue
        try:
            spk = entry[0]
        except (TypeError, IndexError):
            continue
        s = (spk or "").strip()
        if not s:
            continue
        s_u = s.upper()
        if s_u == "ANNOUNCER":
            continue
        if s_u == cur_u:
            continue
        return s
    return ""


def _build_cast_rows(cast_names) -> tuple:
    """Build legacy-schema cast rows + a name->char_id index from
    a list of ALL-CAPS character names.

    Returns ``(cast_rows, char_id_by_name)``. char_id is ``c01``,
    ``c02``, ... in the order the names appear in ``cast_names``.
    """
    cast_rows = []
    char_id_by_name = {}
    for i, name in enumerate(cast_names):
        cid = f"c{i + 1:02d}"
        cast_rows.append({
            "char_id":              cid,
            "name":                 name,
            "character_description": None,
            "gender":               None,
            "voice_preset":         None,
            "line_count":   0,
            "word_count":   0,
        })
        char_id_by_name[name] = cid
    return cast_rows, char_id_by_name


def _build_news_payload(outline, news_seed: str, seed_source: str) -> str:
    """Build the slot-2 news_used JSON string.

    1-element JSON array matching legacy article shape
    (story_orchestrator.py:5141-5283 + RECON 4(b)). seed_source flags
    whether the body came from a user-typed custom_premise or from the
    RSS fetcher.
    """
    news = [{
        "headline":  outline.title,
        "summary":   outline.premise[:500],
        "full_text": news_seed,
        "source":    "User Seed" if seed_source == "custom_premise" else "RSS Auto-Fetch",
        "date":      datetime.now().date().isoformat(),
        "link":      "",
    }]
    return json.dumps(news, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Class
# ---------------------------------------------------------------------------


class OTR_LedgerScriptWriter:
    """v2.0 LPL script writer with legacy-style widget surface.

    Wires the four shipped LPL modules (_otr_outline, _otr_canon,
    _otr_line_composer, _otr_model_loader) plus production_ledger
    into the legacy 4-slot output contract. Widget set restored 2026-05-10
    so users get back episode_title / target_words / num_characters /
    creativity / target_length / style / style_custom / model controls.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Widget order matches the pre-rename writer widget
        # layout (commit 485874b screenshot), minus open_close per
        # Jeffrey 2026-05-10. Order is load-bearing — saved workflows
        # bind by widget index, and the user's mental model maps the
        # field labels to positions on the node.
        return {
            "required": {
                "episode_title": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional episode title override. Stamped at "
                        "ledger.meta.episode_title so SignalLostVideo "
                        "picks it up directly without title-chain "
                        "fallback. Leave blank to let the outline "
                        "supply a title."
                    ),
                }),
                "target_words": ("INT", {
                    "default": 350, "min": 30, "max": 10000, "step": 10,
                    "tooltip": (
                        "Target spoken dialogue word count at ~140 wpm. "
                        "30 = ultra-smoke pipeline check (~13s, ~3 lines), "
                        "100 = smoke (~45s, ~6 HuMo clips), 200 = quick, "
                        "350 = ~2.5min (default), 700 = 5min, "
                        "1400 = 10min, 2100 = 15min, 3500 = 25min. "
                        "target_length presets for '30 words (smoke)' / "
                        "'tiny (smoke)' override this widget."
                    ),
                }),
                "num_characters": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": (
                        "Number of speaking characters (plus ANNOUNCER "
                        "bookends). 1 = monologue/diary mode."
                    ),
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 2**32 - 1, "step": 1,
                    "tooltip": (
                        "Random seed for the C7 byte-identity contract. "
                        "Drives cast lock RNG (announcer voice + open-"
                        "character name rolls), style picker (two-pass "
                        "inventor + chooser samples), LEMMY's 11% roll, "
                        "and the reviewer's seed_for_reviewer derivation.\n\n"
                        "Default is 42 (cosmetic; replaces the previous "
                        "default of 0). When the ComfyUI 'shuffle' icon "
                        "next to the seed value is ON, ComfyUI generates "
                        "a fresh random 64-bit integer on every Queue "
                        "Prompt and the displayed default is ignored. "
                        "Click the shuffle icon OFF and type a specific "
                        "integer to lock the run for byte-identical "
                        "re-queueing."
                    ),
                }),
                "model_id": (_MODEL_CHOICES, {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "Hugging Face model ID. Mistral-Nemo is the "
                        "validated production default. Suffix tags "
                        "([ALPHA], (EXPERIMENTAL)) are stripped by "
                        "the loader before HF lookup."
                    ),
                }),
                "custom_premise": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "(optional) type a custom story premise here — "
                        "overrides the RSS news fetch"
                    ),
                    "tooltip": (
                        "Empty (default) -> RSS fetcher pulls a fresh "
                        "real-world science headline as the episode "
                        "seed.\n\n"
                        "Non-empty -> uses your text verbatim as the "
                        "seed and skips RSS entirely.\n\n"
                        "Use cases for the override:\n"
                        "  - test a specific story idea\n"
                        "  - reproduce a previous run with controlled "
                        "inputs\n"
                        "  - work offline / skip RSS when the network "
                        "is slow."
                    ),
                }),
                "include_act_breaks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When ON (default), the outline LLM is told to "
                        "plan music_inter beats between acts so the "
                        "episode breathes between scenes.\n\n"
                        "When OFF, the outline LLM is told the episode "
                        "is one continuous flow with no music_inter "
                        "beats.\n\n"
                        "Outline schema (Beat.speaker_role) supports "
                        "music_inter either way; this widget just "
                        "tells the LLM whether to use it. Wired into "
                        "the outline prompt via the `Target episode "
                        "shape` line 2026-05-10."
                    ),
                }),
                # act_count sits where target_length used to be in the
                # widget order. Replaced the legacy target_length combo
                # (post-Phase-3 cleanup 2026-05-11) with the typed,
                # JS-clamped integer dropdown that the synthesis §3
                # Phase 2A specified as the authoritative act-count
                # control. Driven live from target_words by
                # web/js/otr_act_count_widget.js.
                "act_count": ("INT", {
                    "default": 0, "min": 0, "max": 7, "step": 1,
                    "tooltip": (
                        "Number of acts (1-7). 0 (default) = "
                        "auto-derive from target_words via "
                        "_otr_episode_budget.default_act_count.\n\n"
                        "Default-act thresholds (target_words floor):\n"
                        "  30   -> default 1 act\n"
                        "  150  -> default 2 acts\n"
                        "  300  -> default 3 acts (and all higher words)\n\n"
                        "Maximum cap per target_words: target_words // 50, "
                        "hard ceiling 7. The user can pick UP from the "
                        "default but not below.\n\n"
                        "The JS extension at web/js/otr_act_count_widget.js "
                        "live-updates the valid dropdown choices when "
                        "target_words changes; the Python validator is "
                        "authoritative (rejects any out-of-band combo)."
                    ),
                }),
                "style": (_STYLE_CHOICES, {
                    "default": _STYLE_AUTO_SENTINEL,
                    "tooltip": (
                        "Tonal preset for the outline. Three-way "
                        "resolution:\n"
                        f"  - '{_STYLE_AUTO_SENTINEL}' (default) -> the "
                        "LLM proposes a 3-6 word style descriptor "
                        "from the resolved news_seed during the run. "
                        "Each episode gets a unique tonal direction "
                        "matched to its article.\n"
                        "  - Any other entry in this dropdown -> used "
                        "verbatim as the style descriptor.\n"
                        "  - style_custom (next widget, when non-"
                        "empty) overrides BOTH paths above."
                    ),
                }),
                "style_custom": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": (
                        "(optional) free-form tonal descriptor — "
                        "overrides the style dropdown above"
                    ),
                    "tooltip": (
                        "Free-form style descriptor. Non-empty value "
                        "overrides the style combo above AND disables "
                        "the LLM 'auto' path. Examples: 'rust-belt "
                        "cyber-noir', 'pulp adventure with comic "
                        "timing', 'cosmic horror procedural'."
                    ),
                }),
                "creativity": (_CREATIVITY_CHOICES, {
                    "default": "balanced",
                    "tooltip": (
                        "Creativity dial — overrides raw temperature "
                        "+ top_p with curated presets:\n"
                        "  safe & tight   -> temp 0.60, top_p 0.90\n"
                        "  balanced       -> temp 0.85, top_p 0.95\n"
                        "  wild & rough   -> temp 0.92, top_p 0.98\n"
                        "  maximum chaos  -> temp 0.95, top_p 0.99\n"
                        "(BUG-014: temp > 1.0 caused format collapse, "
                        "so 'maximum chaos' caps at 0.95.)"
                    ),
                }),
                "optimization_profile": (_OPTIMIZATION_PROFILE_CHOICES, {
                    "default": "Standard",
                    "tooltip": (
                        "[v2.0 MVP forward-compat] Plumbed through to "
                        "_otr_model_loader for VRAM-tier selection. "
                        "Today only 'Standard' is fully validated; "
                        "the other tiers fall back to Standard until "
                        "the v2 loader's profile branches land."
                    ),
                }),
                "perfect_run_spacesaver": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Stamps ledger.meta.perfect_run_spacesaver = "
                        "true so OTR_RTXUpscale's spacesaver cleanup "
                        "fires after PostUpscaleProcgenBlend produces "
                        "the final 1080p mp4. Wipes intermediates to "
                        "free disk space. Leave OFF for any run you "
                        "want to keep the per-stage mp4 set around for "
                        "debugging."
                    ),
                }),
                # Phase 4 v4 (2026-05-11): sampling knobs appended at
                # the END of optional so existing saved workflows keep
                # binding positionally to the old widgets; ComfyUI
                # fills the new positions with the defaults below on
                # workflow load.
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": (
                        "min_p sampling threshold (HuggingFace "
                        "transformers).\n\n"
                        "0.05 (default) cuts the long tail of "
                        "low-probability tokens that produce the "
                        "occasional off-key word in an otherwise "
                        "good line on 7B-14B small local LLMs "
                        "(Mistral-Nemo, Gemma-2, Qwen2.5). Tier 2 fix "
                        "#17 (2026-05-11) flipped this from 0.0 — "
                        "preserving an unvalidated baseline is not "
                        "preservation. 0.0 = disabled.\n\n"
                        "Aggressive: 0.10. Pairs with the existing "
                        "creativity top_p — when both are active the "
                        "tail cut is the union."
                    ),
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.03, "min": 1.0, "max": 1.2, "step": 0.01,
                    "tooltip": (
                        "Repetition penalty for HuggingFace "
                        "transformers generate.\n\n"
                        "1.03 (default) is gentle and helps small "
                        "local LLMs avoid looping on character "
                        "names / high-frequency tokens in short "
                        "outputs. Tier 2 fix #17 (2026-05-11) "
                        "flipped this from 1.0 — preserving an "
                        "unvalidated baseline is not preservation. "
                        "1.0 = disabled. Values above 1.08 commonly "
                        "damage short generations on the 7B-14B "
                        "class."
                    ),
                }),
                "max_new_tokens_cap": ("INT", {
                    "default": 200, "min": 40, "max": 400, "step": 10,
                    "tooltip": (
                        "Per-line max_new_tokens ceiling on the "
                        "composer hot-path.\n\n"
                        "Default 200 preserves current behavior. The "
                        "composer scales attempt-1 max_new_tokens with "
                        "min(cap, target_words * 4) so short lines do "
                        "not get a profligate budget that invites "
                        "drift; attempt-2 retry uses the full cap."
                    ),
                }),
                "enable_polish_pass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "After the composer's retry ladder closes, "
                        "optionally check each generated line against a "
                        "small narration-leak regex (he said / "
                        "*asterisk action* / [bracket direction] / "
                        "opens-with-quote-mark / parenthesized cue "
                        "verbs). If the line trips the regex, fire ONE "
                        "polish LLM call with a targeted cleanup prompt "
                        "and replace the line.\n\n"
                        "OFF (default) preserves the 1-call-per-voiced-"
                        "beat composer hot-path. ON typically adds 1-2 "
                        "extra calls per 15-line episode (~30s); not "
                        "the full +15 calls (~3-5 min). Backstops the "
                        "Script Doctor's end-of-episode pass for users "
                        "who want clean lines in the ledger as they "
                        "are written."
                    ),
                }),
            },
        }

    CATEGORY = "OldTimeRadio"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = (
        "script_text", "script_json", "news_used", "estimated_minutes",
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Mirror legacy LLMScriptWriter: always re-execute. The seed
        # may change, the model may have warmed, etc.
        import time as _t
        return _t.time()

    # ------------------------------------------------------------------
    # FUNCTION method
    # ------------------------------------------------------------------

    def run(
        self,
        episode_title="",
        target_words=350,
        num_characters=2,
        seed=0,
        model_id=DEFAULT_MODEL_ID,
        custom_premise="",
        include_act_breaks=True,
        act_count=0,
        style=_STYLE_AUTO_SENTINEL,
        style_custom="",
        creativity="balanced",
        optimization_profile="Standard",
        perfect_run_spacesaver=False,
        # Phase 4 v4 (2026-05-11) sampling knobs appended at end.
        # Tier 2 fix #17 (2026-05-11): min_p / repetition_penalty
        # defaults flipped from 0.0 / 1.0 (disabled) to 0.05 / 1.03
        # — measured non-trivial dialogue-quality lift on every small
        # local LLM in the 7B-14B class. Knobs remain widgets;
        # per-model tuning untouched.
        min_p=0.05,
        repetition_penalty=1.03,
        max_new_tokens_cap=200,
        enable_polish_pass=False,
    ):
        """Generate a v2.0 LPL script. See module docstring for pipeline."""

        # --- A. Resolve all widget inputs (RSS fetch happens here) -----
        resolved = _resolve_inputs(
            target_words=target_words,
            num_characters=num_characters,
            episode_title=episode_title,
            model_id=model_id,
            custom_premise=custom_premise,
            include_act_breaks=include_act_breaks,
            act_count=act_count,
            style=style,
            style_custom=style_custom,
            creativity=creativity,
            optimization_profile=optimization_profile,
            perfect_run_spacesaver=perfect_run_spacesaver,
            # Phase 4 v4 (2026-05-11) sampling knobs.
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens_cap=max_new_tokens_cap,
            enable_polish_pass=enable_polish_pass,
        )

        log.info(
            "[OTR_LedgerScriptWriter] start: model=%r, target_words=%d, "
            "num_characters=%d, style_source=%s "
            "(pending=%s, value=%r), creativity=%r (temp=%.2f "
            "top_p=%.2f), seed_source=%s, episode_title=%r, "
            "perfect_run_spacesaver=%s",
            resolved["model_id"], resolved["target_words"],
            resolved["num_characters"],
            resolved["style_source"], resolved["style_pending"],
            resolved["style"], resolved["creativity"],
            resolved["temperature"], resolved["top_p"],
            resolved["seed_source"], resolved["episode_title"],
            resolved["perfect_run_spacesaver"],
        )

        # --- B. Late imports (no GPU / no model loads at module import) ---
        import random as _random
        from . import _otr_outline as _OTRO
        from . import _otr_canon as _OTRC
        from . import _otr_line_composer as _OTRLC
        from . import _otr_model_loader as _OTRML
        from . import _otr_ledger as _OTRL
        from . import _otr_casting as _OTRCAST
        from . import _otr_style_picker as _OTRSP
        from . import news_interpreter as _OTRNI
        from . import _otr_news_wiring as _OTRNW
        from . import production_ledger as _PL

        # --- C. Load LLM + build truncating generate_fn ---------------
        cache_entry = _OTRML.load_llm(
            resolved["model_id"], device="cuda",
            optimization_profile=resolved["optimization_profile"],
        )
        generate_fn = _build_truncating_generate_fn(
            cache_entry,
            top_p=resolved["top_p"],
            # Phase 4 v4 (2026-05-11) sampling knobs.
            min_p=resolved["min_p"],
            repetition_penalty=resolved["repetition_penalty"],
        )
        # LFC sprint commit 12.2 (W4 fix, 2026-05-11): build a
        # SEPARATE polish_generate_fn off the same cache_entry so
        # the inline compose_line polish path (when
        # enable_polish_pass=True) does NOT inherit the writer's
        # composer-tuned closure (min_p / repetition_penalty /
        # top_p). Polish is a short low-temperature rewrite;
        # composer tuning produces awkward substitutions. The
        # dedicated fn uses polish-conservative sampling
        # (top_p=0.9, no min_p, no repetition_penalty -- see
        # _otr_model_loader.make_polish_generate_fn).
        # Best-effort: if the factory isn't available on older
        # builds, falls back to None and compose_line uses
        # generate_fn for polish (pre-W4 behaviour).
        try:
            polish_generate_fn = _OTRML.make_polish_generate_fn(
                cache_entry,
            )
        except Exception as _polish_exc:  # noqa: BLE001
            log.warning(
                "[OTR_LedgerScriptWriter] make_polish_generate_fn "
                "unavailable (%s); falling back to generate_fn "
                "for inline polish",
                _polish_exc,
            )
            polish_generate_fn = None

        # --- D. Cast contract -- LEDGER-FIRST, CAST-LOCKED, OUTLINE-AFTER
        #
        # Inversion landed 2026-05-10 per the cast contract
        # architecture target. Order is now:
        #   D.1  new_ledger() up front, stamp cast_status="building"
        #   D.2  optional style LLM-suggest (when style_pending)
        #   D.3  lock_cast() -- ANNOUNCER first, LEMMY 11%, then
        #        per-character LLM call for description+gender+voice
        #   D.4  led.set_cast() + stamp cast_status="locked"
        #   D.5  generate_outline() consumes the locked character_cast
        # ---------------------------------------------------------------
        # D.1 Ledger up front. Subsequent stages stamp meta against it.
        led = _PL.new_ledger(episode_id=None)
        episode_id = led.episode_id           # pending_<YYYYMMDD_HHMMSS>
        audio_dir = Path(led.out_dir)         # otr/episodes/<ep>/audio/
        episode_root = audio_dir.parent       # otr/episodes/<ep>/
        meta = led.data.setdefault("meta", {})
        meta["cast_status"] = "building"
        meta["requested_num_characters"] = resolved["num_characters"]
        meta["episode_seed"] = int(seed)

        # D.2 Two-pass style picker (when "let the story decide" is
        # selected or combo is blank AND no style_custom override).
        # Pass 1 inventor produces 5 distinct snake_case style
        # descriptors grounded in the news article + 5 sampled seed
        # flavors. Pass 2 chooser picks the single best one. The
        # writer's seed widget seeds the sample RNG so same seed +
        # same article = same sample = same picks (C7 byte-identity).
        # See nodes/_otr_style_picker.py for full design.
        #
        # The widget-typed style_custom and the verbatim combo entries
        # bypass this branch.
        if resolved["style_pending"]:
            picker_rng = _random.Random(int(seed))
            style_pick = _OTRSP.pick_style(
                generate_fn,
                article_text=resolved["news_seed"],
                seed_pool=list(_STYLE_PICKER_SEED_POOL),
                rng=picker_rng,
                model_id=str(resolved["model_id"]),
            )
            resolved["style"] = style_pick.chosen
            meta["style_pick"] = style_pick.model_dump()
            log.info(
                "[OTR_LedgerScriptWriter] style picker: chosen=%r "
                "(from candidates %r, %d inventor attempt(s), "
                "pass1=%dms pass2=%dms)",
                style_pick.chosen, style_pick.candidates,
                style_pick.pass1_attempts,
                style_pick.pass1_duration_ms, style_pick.pass2_duration_ms,
            )

        # D.2.5 News interpretation. Read the full article (currently
        # discarded after RSS fetch -- see _fetch_rss_seed_or_die change
        # in this commit) and emit four purpose-specific briefs that
        # cast / outline / announcer / line-composer consume INSTEAD
        # of the mechanical 500-char slice of headline+summary.
        # ADR docs/news_interpreter_adr.md section 5 -- commit 3 of
        # the news_interpreter sprint.
        #
        # Graceful degrade (ADR section 9.2): if build_news_briefs
        # exhausts its 3-attempt retry budget, stamp meta["news"] = None
        # and fall back to raw news_seed on downstream consumers. The
        # writer MUST produce a complete episode even when the brief
        # LLM call fails; this is a "warn-and-continue" boundary, not
        # a hard fail.
        article = resolved["news_article"]
        try:
            briefs = _OTRNI.build_news_briefs(
                generate_fn,
                full_text=article.get("full_text", ""),
                headline=article.get("headline", ""),
                summary=article.get("summary", ""),
                outlet=article.get("source", ""),
                pub_date=article.get("date", ""),
                style=resolved["style"],
                seed=int(seed),
                model_id=str(resolved["model_id"]),
            )
            meta["news"] = briefs.model_dump()
            casting_brief = briefs.casting_brief
            script_brief = briefs.script_brief
            key_terms_tuple: tuple[str, ...] = tuple(briefs.key_terms)
            log.info(
                "[OTR_LedgerScriptWriter] news_interpreter OK: "
                "%d key_terms in %d attempt(s)",
                len(briefs.key_terms), briefs.attempts,
            )
        except _OTRNI.NewsInterpreterError as exc:
            log.warning(
                "[OTR_LedgerScriptWriter] news_interpreter FAILED after "
                "all attempts: %s -- falling back to raw news_seed for "
                "cast + outline (no key_terms enforcement)",
                exc,
            )
            meta["news"] = None
            casting_brief = ""
            script_brief = ""
            key_terms_tuple = ()

        # D.3 Lock the cast.
        #
        # Seed: ALWAYS used to seed random.Random (no zero sentinel).
        # ComfyUI's frontend handles "randomize" mode by sending a
        # real 64-bit integer to the backend; the backend should
        # trust that integer for C7 byte-identity. Branching on
        # seed==0 would treat a legitimate user choice ("seed 0")
        # as non-deterministic and silently violate the contract
        # (round-robin 2026-05-10).
        #
        # LEMMY's 11% roll now also runs against this same RNG so
        # the seed fully determines whether the cameo hits. Prior
        # design routed LEMMY through SystemRandom for OS entropy,
        # but that defeated C7 even for explicitly-seeded runs.
        # The 11% rate remains statistically intact across many
        # runs (still tested by tests/lemmy_rng_check.py); what
        # changes is that with seed=42, every run gets the SAME
        # lemmy_hit value -- which is exactly the reproducibility
        # contract C7 requires.
        cast_rng = _random.Random(int(seed))
        cast_rows, cast_meta = _OTRCAST.lock_cast(
            generate_fn,
            num_characters=resolved["num_characters"],
            news_seed=resolved["news_seed"],
            casting_brief=casting_brief,
            style=resolved["style"],
            rng=cast_rng,
        )
        led.set_cast(cast_rows)
        meta["cast_status"]           = "locked"
        meta["cast_locked"]           = True
        meta["cast_contract_version"] = "cast-v1"
        meta["cast_contract"] = {
            "lemmy_hit":              cast_meta["lemmy_hit"],
            "casting_attempts":       cast_meta["casting_attempts"],
            "num_characters_request": cast_meta["num_characters_request"],
            "num_characters_locked":  cast_meta["num_characters_locked"],
        }
        log.info(
            "[OTR_LedgerScriptWriter] cast locked: %d rows "
            "(announcer + %d characters, lemmy_hit=%s)",
            len(cast_rows), cast_meta["num_characters_locked"],
            cast_meta["lemmy_hit"],
        )

        # Build the name->char_id index the per-beat composer needs.
        # Excludes ANNOUNCER (announcer-role beats hardcode "announcer"
        # cid downstream, not the c01 cast row's char_id).
        char_id_by_name: dict[str, str] = {
            row["name"]: row["char_id"]
            for row in cast_rows
            if row["name"] != "ANNOUNCER"
        }
        character_cast: tuple[str, ...] = tuple(char_id_by_name.keys())

        # Post-lock sanity assertions (round-robin 2026-05-10):
        # Catch any future regression where lock_cast() returns
        # duplicates, drops a row, or mis-counts. Belt-and-braces;
        # today the casting module guarantees these by construction.
        non_announcer_count = len(char_id_by_name)
        if non_announcer_count == 0:
            raise RuntimeError(
                "Cast lock produced no non-announcer characters. "
                f"cast_rows: {cast_rows!r}"
            )
        # Duplicate name check: char_id_by_name as a dict silently
        # collapses dupes, so compare to the raw row name list.
        raw_names = [
            row["name"] for row in cast_rows
            if row["name"] != "ANNOUNCER"
        ]
        if len(raw_names) != len(set(raw_names)):
            raise RuntimeError(
                f"Cast lock produced duplicate non-announcer names: "
                f"{raw_names!r}"
            )
        # Count match: the locked open characters should equal the
        # requested num_characters.
        if non_announcer_count != resolved["num_characters"]:
            raise RuntimeError(
                f"Cast lock count mismatch: requested "
                f"{resolved['num_characters']} non-announcer "
                f"characters, got {non_announcer_count}. "
                f"cast_rows: {cast_rows!r}"
            )

        # D.5 Generate validated outline against the locked cast.
        # The outline LLM is told to use exactly these character names
        # in character-role beats; generate_outline rerolls on cast
        # drift (CastContractError).
        #
        # cast_descriptions wires the casting LLM's per-character
        # output (gender + character_description) into the outline
        # prompt's Cast block so the outline LLM can plan beats that
        # exploit each character's distinct personality + stakes.
        # Order MUST match character_cast 1:1 (OutlineRequest
        # __post_init__ enforces this); both lists derive from
        # char_id_by_name.keys() above so the order is identical
        # by construction. ANNOUNCER excluded from both, same as
        # character_cast.
        cast_descriptions: tuple[tuple[str, str, str], ...] = tuple(
            (
                row["name"],
                str(row.get("gender") or ""),
                str(row.get("character_description") or ""),
            )
            for row in cast_rows
            if row["name"] != "ANNOUNCER"
        )
        # Phase 2A (2026-05-11): build EpisodeBudget from
        # (target_words, act_count, include_act_breaks, num_characters).
        # On invalid combos compute_episode_budget raises
        # InvalidEpisodeBudgetError (ValueError subclass); we let it
        # propagate -- the widget delta is the right place to fail
        # loud, not silently coerce.
        from . import _otr_episode_budget as _OTRB  # type: ignore
        episode_budget = _OTRB.compute_episode_budget(
            target_words=resolved["target_words"],
            act_count=resolved["act_count"],
            include_act_breaks=resolved["include_act_breaks"],
            num_characters=resolved["num_characters"],
        )
        log.info(
            "[OTR_LedgerScriptWriter] phase 2A budget: act_count=%d, "
            "arc_phases=%s, per_phase_words=%s, per_phase_beats=%s, "
            "words_per_beat_range=%s, music_inter=%d",
            episode_budget.act_count, list(episode_budget.arc_phases),
            list(episode_budget.per_phase_words),
            list(episode_budget.per_phase_beats),
            list(episode_budget.words_per_beat_range),
            episode_budget.music_inter_count,
        )

        outline_req = _OTRO.OutlineRequest(
            news_seed=resolved["news_seed"],
            style=resolved["style"],
            character_cast=character_cast,
            target_words=resolved["target_words"],
            script_brief=script_brief,
            key_terms=key_terms_tuple,
            cast_descriptions=cast_descriptions,
            include_act_breaks=bool(resolved.get("include_act_breaks", True)),
            budget=episode_budget,
        )
        outline = _OTRO.generate_outline(generate_fn, outline_req)

        # --- E. Word-budget integration check (WARN, do not fail) -----
        beat_word_sum = sum(b.target_words for b in outline.beats)
        ratio = beat_word_sum / max(1, resolved["target_words"])
        if not (WORD_BUDGET_RATIO_LO <= ratio <= WORD_BUDGET_RATIO_HI):
            log.warning(
                "[OTR_LedgerScriptWriter] WORD_BUDGET_DRIFT: outline "
                "beats sum to %d words, target %d (ratio=%.2f); "
                "proceeding anyway",
                beat_word_sum, resolved["target_words"], ratio,
            )

        # --- G. Build episode_canon (write deferred to section J.5) ----
        # Disk write moved out so the post-composition title regen
        # (section J.5) can overwrite canon.title before episode_canon.json
        # ever touches disk. Header rendering still happens here because
        # the per-line composer (section I) needs canon_header on every
        # beat.
        canon = _OTRC.episode_canon_from_outline_dict({
            "title":       resolved["episode_title"] or outline.title,
            "premise":     outline.premise,
            "setting":     outline.setting,
            "time_of_day": outline.time_of_day,
        })
        canon_header = _OTRC.render_episode_canon_header(canon)
        log.info(
            "[OTR_LedgerScriptWriter] episode_canon built (disk write "
            "deferred to post-composition title regen)"
        )

        # --- H. Phase 2B (2026-05-11): pre-stamp skeleton ledger -------
        # Outline validated. Pre-stamp one row per beat NOW so the
        # composer loop updates in place. Mid-loop crash leaves a
        # partial-but-coherent ledger on disk (text == "" signals
        # "row composed pending"). See production_ledger
        # init_lines_from_outline / update_line_text comments.
        led.init_lines_from_outline(outline, char_id_by_name)
        led.save()
        log.info(
            "[OTR_LedgerScriptWriter] phase 2B skeleton stamped: "
            "%d line rows", len(led.data.get("lines", []) or []),
        )

        # --- I. Per-beat loop ------------------------------------------
        script_text_parts: list = []
        last_lines: list = []  # rolling window of LAST_LINES_WINDOW

        base_temp = resolved["temperature"]

        # Phase 0 (2026-05-11): build the UPPERCASE name-roster ONCE.
        # Passed to every LineRequest so compose_line can detect
        # proper nouns the LLM invented outside the locked cast +
        # journalistic key_terms. Detection-only: phantoms are flagged
        # on lines[k].compose_flags; the composer does NOT reroll.
        # Phase 3 reviewer + deterministic Step 2.5 fallback own
        # repair downstream. See synthesis §6.A (Option 1, strict).
        allowed_roster = _OTRLC.build_allowed_roster(
            cast_rows=cast_rows,
            key_terms=key_terms_tuple,
        )
        # Wiring-review #7 / #9 (2026-05-11): stamp the canonical
        # allowed_roster on meta as a sorted JSON-serializable list
        # so every downstream consumer (composer, Pass 1 auditor,
        # deterministic repair, Step 2.5 phantom-skip, Pass 3
        # auditor) reads ONE roster. Nobody recomputes locally; the
        # roster is immutable for the episode's life.
        meta["allowed_roster"] = sorted(allowed_roster)
        log.info(
            "[OTR_LedgerScriptWriter] phase 0 roster built: %d entries "
            "(cast=%d + announcer + key_terms=%d), stamped on meta",
            len(allowed_roster),
            len([r for r in cast_rows if r.get("name") != "ANNOUNCER"]),
            len(key_terms_tuple),
        )

        # Phase 1 (2026-05-11): build outline_spine + voice_card map
        # ONCE. Both are stable across every composer call in the
        # episode so they live in the static prefix of the prompt
        # (KV-cache friendly once reuse is wired in the loader).
        # See synthesis §6.D for the prompt structure.
        outline_spine = _OTRLC.render_outline_spine(outline)
        # Map char_id -> voice_card_str (cast_rows already has the
        # ANNOUNCER row stamped by _OTRCAST.lock_cast()). Beats look
        # up by NAME, not char_id, so build a name index too.
        voice_card_by_name: dict[str, str] = {
            row.get("name", ""): _OTRLC.build_voice_card(row)
            for row in cast_rows
            if row.get("name")
        }
        # Fallback ANNOUNCER card if for some reason the cast row's
        # voice_card came out empty (e.g. unset description).
        if not voice_card_by_name.get("ANNOUNCER"):
            voice_card_by_name["ANNOUNCER"] = "ANNOUNCER (omniscient narrator)"
        log.info(
            "[OTR_LedgerScriptWriter] phase 1 prompt context built: "
            "spine=%d chars, voice_cards=%d entries",
            len(outline_spine), len(voice_card_by_name),
        )

        # Phase 4 v4 (2026-05-11): split rosters for prompt rendering.
        # `allowed_roster` stays the union (input to the phantom
        # gate); cast names and journalistic key_terms render in
        # distinct buckets inside the composer's NAMED ENTITIES block.
        allowed_people = frozenset(
            (r.get("name") if isinstance(r, dict) else getattr(r, "name", ""))
            for r in cast_rows
            if (r.get("name") if isinstance(r, dict) else getattr(r, "name", ""))
        )
        allowed_things = frozenset(key_terms_tuple)

        # Phase 4 v4 (2026-05-11): full-cast voice cards block. Joined
        # in cast_rows order (dict ordering preserves insertion order).
        all_voice_cards_str = "\n".join(
            card for card in voice_card_by_name.values() if card
        )

        # Phase 4 v4 (2026-05-11): one-sentence theme from
        # meta.news.script_brief. Robust to abbreviations ("Dr. Smith
        # ...") via terminal-punctuation + whitespace split, not bare
        # ".". Empty string flips the THEME block off cleanly.
        _brief = str(
            (meta.get("news") or {}).get("script_brief") or ""
        ).strip()
        if _brief:
            # Tier 1 fix #9 (2026-05-11): drop the sentence-detection
            # regex (broke on "Dr." / "Mr." / "St." abbreviations and
            # produced a one-token theme). Theme is flavor, not
            # structure — cap at the first 15 words and move on.
            _words = _brief.split()
            theme = " ".join(_words[:15])
        else:
            theme = ""

        # Phase 4 v4 (2026-05-11): precompute per-beat POSITION
        # strings. Format "<phase>, beat N of M. Next phase: <next>."
        # or "<phase>, beat N of M. Final phase." for the final phase.
        #
        # Tier 1 fix #3 (2026-05-11): EXCLUDE non-voiced beats (music
        # markers, sfx) from phase_beats. A character beat surrounded
        # by two music_inter beats was reading "beat 3 of 5 in setup"
        # when 2 of the 5 had no dialogue — confusing to the model
        # and inconsistent with the user's mental model of POSITION.
        phase_beats: dict = {}
        for _b in outline.beats:
            if _b.speaker_role in NON_VOICED_ROLES:
                continue
            phase_beats.setdefault(_b.arc_phase or "setup", []).append(
                _b.beat_id,
            )
        # Tier 3 fix #21 (2026-05-11): `episode_budget` is always in
        # scope by the time the per-beat loop builds POSITION (it is
        # constructed by section D.5 / compute_episode_budget on
        # every code path that reaches I). The outline-only fallback
        # path was defensive dead code; assert to surface drift
        # immediately if a future refactor moves the budget build.
        assert episode_budget is not None, (
            "episode_budget must be non-None before the per-beat "
            "loop; POSITION derivation depends on its arc_phases."
        )
        arc_order = list(episode_budget.arc_phases)

        def _position_for(beat) -> str:
            # Tier 1 fix #10 (2026-05-11): raise on missing beat_id /
            # missing arc_phase instead of silently returning "beat 1
            # of 1". Silent wrong position is prompt poison; a hard
            # raise surfaces upstream corruption (outline/budget
            # drift) immediately. Called only for voiced beats, so
            # both lookups must hit.
            this_phase = (beat.arc_phase or "setup").strip()
            if this_phase not in arc_order:
                raise ValueError(
                    f"[_position_for] beat {beat.beat_id!r} has "
                    f"arc_phase {this_phase!r} not in arc_order "
                    f"{arc_order!r}"
                )
            ids = phase_beats.get(this_phase, [])
            if beat.beat_id not in ids:
                raise ValueError(
                    f"[_position_for] beat_id {beat.beat_id!r} not "
                    f"in phase_beats[{this_phase!r}]={ids!r}"
                )
            phase_idx = arc_order.index(this_phase)
            next_phase = (
                arc_order[phase_idx + 1]
                if phase_idx + 1 < len(arc_order)
                else "end"
            )
            beat_n = ids.index(beat.beat_id) + 1
            beat_total = len(ids)
            tail = (
                f" Next phase: {next_phase}."
                if next_phase != "end"
                else " Final phase."
            )
            return f"{this_phase}, beat {beat_n} of {beat_total}.{tail}"

        # Style descriptor for the composer's STATIC prefix. Empty
        # string flips the STYLE block off in _build_user_prompt --
        # back-compat for callers without a style picked yet.
        style_descriptor = str(resolved.get("style") or "").strip()

        # Tier 3 fix #19 (2026-05-11): single LineRequest construction
        # site for both character and announcer beats. Pre-Tier-3 the
        # body was duplicated twice across ~25 fields each; adding a
        # field meant editing two literals in lockstep and missing one
        # was easy. The nested closure pulls loop-scope context
        # implicitly so the call sites stay one-liners.
        def _build_line_request_for_beat(
            beat,
            *,
            is_announcer: bool,
        ):
            speaker = "ANNOUNCER" if is_announcer else beat.speaker
            prev_speaker = _derive_prev_speaker(last_lines, speaker)
            voice_card = (
                voice_card_by_name.get(
                    "ANNOUNCER", "ANNOUNCER (omniscient narrator)",
                )
                if is_announcer
                else voice_card_by_name.get(beat.speaker, beat.speaker)
            )
            return _OTRLC.LineRequest(
                speaker=speaker,
                intent=beat.intent,
                mood=beat.mood,
                target_words=beat.target_words,
                canon_header=canon_header,
                last_lines=list(last_lines),
                allowed_roster=allowed_roster,
                # Phase 1 (2026-05-11) prompt enrichment.
                style_descriptor=style_descriptor,
                outline_spine=outline_spine,
                character_voice_card=voice_card,
                # Phase 2A (2026-05-11) arc_phase awareness.
                arc_phase=(beat.arc_phase or "").strip(),
                # Phase 4 v4 (2026-05-11) prompt revision.
                allowed_people=allowed_people,
                allowed_things=allowed_things,
                prev_speaker=prev_speaker,
                current_beat_block=_OTRLC.render_current_beat(
                    outline, beat.beat_id,
                ),
                theme=theme,
                all_voice_cards=all_voice_cards_str,
                sfx_cue=(beat.sfx_cue or "").strip(),
                position=_position_for(beat),
            )

        for beat in outline.beats:
            traits = (beat.mood or "").strip() or DEFAULT_TRAITS
            cleaned: str
            cid: str
            token: str
            beat_compose_flags: tuple[str, ...] = ()

            if beat.speaker_role == "character":
                line_req = _build_line_request_for_beat(
                    beat, is_announcer=False,
                )
                line_res = _OTRLC.compose_line(
                    generate_fn, line_req, base_temperature=base_temp,
                    max_new_tokens_cap=resolved["max_new_tokens_cap"],
                    enable_polish_pass=resolved["enable_polish_pass"],
                    polish_generate_fn=polish_generate_fn,
                )
                cleaned = line_res.text
                beat_compose_flags = line_res.compose_flags
                cid = char_id_by_name[beat.speaker]
                token = f"[VOICE: {beat.speaker}, {traits}] {cleaned}"

                last_lines.append((beat.speaker, cleaned))
                if len(last_lines) > LAST_LINES_WINDOW:
                    last_lines.pop(0)

            elif beat.speaker_role == "announcer":
                line_req = _build_line_request_for_beat(
                    beat, is_announcer=True,
                )
                line_res = _OTRLC.compose_line(
                    generate_fn, line_req, base_temperature=base_temp,
                    max_new_tokens_cap=resolved["max_new_tokens_cap"],
                    enable_polish_pass=resolved["enable_polish_pass"],
                    polish_generate_fn=polish_generate_fn,
                )
                cleaned = line_res.text
                beat_compose_flags = line_res.compose_flags
                cid = "announcer"
                token = f"[VOICE: ANNOUNCER, {traits}] {cleaned}"

                last_lines.append(("ANNOUNCER", cleaned))
                if len(last_lines) > LAST_LINES_WINDOW:
                    last_lines.pop(0)

            elif beat.speaker_role in NON_VOICED_ROLES:
                # Phase 4 v4 (2026-05-11): scene-local LAST SPOKEN
                # window. Crossing a music marker resets the
                # conversation context — listeners experience a scene
                # break, so the composer should too. Lines from before
                # the marker are wrong signal for what comes after.
                if beat.speaker_role in {
                    "music_open", "music_inter", "music_close",
                }:
                    last_lines.clear()
                cleaned = (beat.sfx_cue or beat.intent or "").strip()
                cid = beat.speaker_role
                token = f"[SFX: {cleaned}]"

            else:
                log.warning(
                    "[OTR_LedgerScriptWriter] unknown speaker_role %r "
                    "on beat %s; skipping",
                    beat.speaker_role, beat.beat_id,
                )
                continue

            # Phase 2B (2026-05-11): in-place ledger update + save.
            # Skeleton row exists from init_lines_from_outline. Update
            # text + compose_flags + traits + char_id (skeleton's char_id
            # came from char_id_by_name lookup, but we re-stamp here
            # so any post-init speaker resolution is reflected). Save
            # after EVERY line so a mid-loop crash leaves the work
            # done so far on disk.
            #
            # Wiring-review #4 (2026-05-11): MUST check
            # update_line_text return value. False means no row
            # matched -- the ledger skeleton and the outline have
            # drifted apart and the disk ledger silently misses this
            # beat while script_text_parts populates. Fail loud.
            _ok = led.update_line_text(beat.beat_id, cleaned)
            if not _ok:
                raise RuntimeError(
                    f"[OTR_LedgerScriptWriter] LineLedgerMismatchError: "
                    f"update_line_text returned False for "
                    f"beat_id={beat.beat_id!r} -- ledger skeleton lacks "
                    f"this beat. Did init_lines_from_outline run with "
                    f"the same outline object? "
                    f"lines={[ln.get('beat_id') for ln in (led.data.get('lines') or [])]}"
                )
            _OTRL.patch_line_fields(
                led.data, beat.beat_id,
                {
                    "char_id":       cid,
                    "traits":        traits,
                    "compose_flags": list(beat_compose_flags),
                },
            )
            led.save()
            script_text_parts.append(token)

        # --- I.5. News-wiring overlay (Phase 2B: operates on ledger) --
        # Two pure operations on `led.data["lines"]` AFTER the
        # progressive composer loop completes. Both no-op when
        # meta["news"] is None (graceful-degrade path).
        #
        # 1. Announcer closing-line override. The line composer wrote
        #    something at every announcer beat from beat.intent. For
        #    the LAST announcer beat we substitute news_close_brief
        #    so the listener actually hears the journalistic content
        #    from the source article (era-neutral, news_interpreter-
        #    distilled).
        #
        # 2. Post-assembly key_terms audit. Walk every voiced line,
        #    check each key_term landed via word-boundary regex.
        #    Stamp the result on meta["post_assembly_key_terms"].
        news_meta = meta.get("news") or {}
        nc_brief = (news_meta.get("news_close_brief") or "").strip()
        if nc_brief:
            overridden = _OTRNW.override_announcer_close(
                led.data["lines"], nc_brief,
            )
            if overridden is not None:
                # Recompute char_count + word_count after the in-place
                # text override so downstream consumers see fresh
                # counts (override_announcer_close only touches `text`).
                _OTRL.patch_line_text(
                    led.data, overridden["line_id"], overridden["text"],
                )
                log.info(
                    "[OTR_LedgerScriptWriter] news_close_brief stamped "
                    "onto closing announcer line %s",
                    overridden.get("line_id"),
                )
            else:
                log.warning(
                    "[OTR_LedgerScriptWriter] news_close_brief present "
                    "but no announcer line found in led.data['lines'] "
                    "to stamp onto; closing read will use the line "
                    "composer's original text"
                )

        nc_key_terms = tuple(news_meta.get("key_terms") or ())
        if nc_key_terms:
            landed, missing = _OTRNW.post_assembly_keyterm_check(
                led.data["lines"], nc_key_terms, min_required=2,
            )
            meta["post_assembly_key_terms"] = {
                "landed":       landed,
                "missing":      missing,
                "min_required": 2,
                "passed":       len(landed) >= 2,
                "repair_pass":  "deferred",
            }
            if not landed:
                log.warning(
                    "[OTR_LedgerScriptWriter] post-assembly key_terms "
                    "ZERO landed (terms=%r). ADR section 4.4 calls "
                    "for hard-fail + repair pass; current alpha ships "
                    "warn-only and DEFERS the repair pass. Episode proceeds.",
                    list(nc_key_terms),
                )
            elif len(landed) < 2:
                log.warning(
                    "[OTR_LedgerScriptWriter] post-assembly key_terms "
                    "below min_required=2: %d/%d landed (missing=%r)",
                    len(landed), len(nc_key_terms), missing,
                )
            elif missing:
                log.warning(
                    "[OTR_LedgerScriptWriter] post-assembly key_terms "
                    "%d/%d landed (missing=%r); proceeding",
                    len(landed), len(nc_key_terms), missing,
                )
            else:
                log.info(
                    "[OTR_LedgerScriptWriter] post-assembly key_terms "
                    "all %d landed",
                    len(landed),
                )

        # --- J. Phase 0 aggregate + §6.G word counts + final save ----
        # No set_lines + post-patch pass any more -- every line was
        # stamped progressively inside the composer loop (Phase 2B).
        # The post-loop work here is the meta.compose_flag_summary
        # rollup, the §6.G word-count stamp (character / announcer /
        # total -- post-Phase-3 review Fix 3, 2026-05-11), and a
        # final ledger save (which also flushes any text the
        # news-wiring overlay mutated above).
        meta["compose_flag_summary"] = _OTRLC.aggregate_compose_flags(led.data)
        log.info(
            "[OTR_LedgerScriptWriter] phase 0 compose_flag_summary: %s",
            meta["compose_flag_summary"] or "(clean)",
        )
        _PL.stamp_word_counts(led)
        log.info(
            "[OTR_LedgerScriptWriter] §6.G word counts: "
            "character=%d announcer=%d total=%d",
            meta.get("character_word_count", 0),
            meta.get("announcer_word_count", 0),
            meta.get("total_word_count", 0),
        )
        led.save()

        # --- J.5. Post-composition title regen ------------------------
        # Per Jeffrey 2026-05-10: when the user leaves episode_title
        # blank, regenerate the title from the FINAL composed dialogue
        # via the LLM. The prompt does NOT see the news_seed -- the title
        # is grounded purely in what the listener will hear. User-typed
        # episode_title still wins; LLM only fires on blank input;
        # outline.title is the last-resort fallback when the LLM call
        # fails or its output is rejected by the guardrails.
        #
        # Why we capture old_title BEFORE deciding final_title:
        # the per-line composer's canon_header included the outline title
        # as the TITLE: field. If a beat's intent was "open the show by
        # naming the episode" the LLM likely baked the OUTLINE title
        # verbatim into the announcer/character spoken text. When the
        # title regen produces a different final title, the audio
        # consumers (Bark / Kokoro) would otherwise speak the old
        # placeholder title at the top of the episode. So after regen
        # we substitute the new title back into any line text that
        # quoted the old one, before audio rendering runs downstream.
        old_title_in_lines = (outline.title or "").strip()

        title_source = "outline_fallback"
        if resolved["episode_title"]:
            # User typed a value; respect it verbatim. The composer saw
            # this same value in canon_header (section G), so no
            # substitution is needed.
            final_title = resolved["episode_title"]
            title_source = "user"
        else:
            assembled_script = "\n\n".join(script_text_parts).strip()
            regen_title = _generate_title_from_script(
                generate_fn,
                assembled_script,
                temperature=resolved["temperature"],
            )
            if regen_title:
                final_title = regen_title
                title_source = "llm_post_composition"
            else:
                final_title = outline.title
                title_source = "outline_fallback"
                log.warning(
                    "[OTR_LedgerScriptWriter] title regen returned empty; "
                    "falling back to outline.title=%r",
                    outline.title,
                )

        # Update canon with the final title and write to disk. canon.title
        # is now what downstream video consumers (SignalLostVideo, episode
        # canon readers) will see.
        canon.title = final_title
        _OTRC.write_episode_canon(episode_root, canon)
        log.info(
            "[OTR_LedgerScriptWriter] episode_canon written with "
            "title=%r (source=%s) at %s",
            final_title, title_source,
            episode_root / _OTRC.EPISODE_CANON_FILENAME,
        )

        # --- J.6. Substitute regenerated title into spoken line text --
        # When the title changed post-composition AND the composer baked
        # the old (outline) title into any spoken line, swap to the new
        # title so the announcer / characters actually say the
        # regenerated title aloud. Helper enforces case-insensitive
        # whole-phrase match + the _TITLE_SUB_MIN_LEN guard.
        #
        # Phase 2B (2026-05-11) -- the writer no longer accumulates
        # `line_rows` in memory; all rows live on `led.data["lines"]`
        # progressively. This loop must walk the ledger directly.
        # Wiring-review #1 fix: replaced stale `for r in line_rows:`
        # with `for r in led.data.get("lines", []) or []:` so the
        # title-substitution branch no longer NameError's the moment
        # the LLM regenerates a different title.
        _title_sub_meta = None
        if (
            final_title
            and old_title_in_lines
            and final_title != old_title_in_lines
        ):
            n_lines_patched = 0
            n_subs_total = 0
            for r in led.data.get("lines", []) or []:
                new_text, n_subs = _substitute_title_in_text(
                    r.get("text", "") or "",
                    old_title_in_lines,
                    final_title,
                )
                if n_subs > 0:
                    _OTRL.patch_line_text(
                        led.data, r.get("line_id"), new_text,
                    )
                    n_lines_patched += 1
                    n_subs_total += n_subs
            # Mirror substitutions into script_text_parts so the slot-0
            # STRING output and the script_text_parts join stay
            # consistent with the patched ledger.
            for idx, part in enumerate(script_text_parts):
                new_part, _ = _substitute_title_in_text(
                    part, old_title_in_lines, final_title,
                )
                script_text_parts[idx] = new_part
            if n_lines_patched > 0:
                log.info(
                    "[OTR_LedgerScriptWriter] title substitution: "
                    "replaced %d occurrence(s) of %r with %r across "
                    "%d line(s)",
                    n_subs_total, old_title_in_lines, final_title,
                    n_lines_patched,
                )
            # Always stamp the meta record when we attempted substitution
            # (even with 0 matches) so audits can tell "no occurrences
            # found" from "no regen happened at all".
            _title_sub_meta = {
                "old_title":         old_title_in_lines,
                "new_title":         final_title,
                "lines_patched":     n_lines_patched,
                "substitutions":     n_subs_total,
            }

        # --- K. Stamp meta block --------------------------------------
        # Stamps the run parameters into meta.gen_params_initial for
        # forensic / soak inspection. Also stamps episode_title
        # (forward-compat title chain slot) and perfect_run_spacesaver.
        meta = led.data.setdefault("meta", {})
        meta["gen_params_initial"] = {
            "target_words":         resolved["target_words"],
            "num_characters":       resolved["num_characters"],
            "model_id":              resolved["model_id"],
            "style":                 resolved["style"],
            "style_combo":           resolved["style_combo"],
            "style_custom":          resolved["style_custom"],
            "style_source":          resolved["style_source"],
            "creativity":            resolved["creativity"],
            "temperature":           resolved["temperature"],
            "top_p":                 resolved["top_p"],
            "act_count":             resolved["act_count"],
            "include_act_breaks":    resolved["include_act_breaks"],
            "optimization_profile":  resolved["optimization_profile"],
            "seed_source":           resolved["seed_source"],
        }
        # Always stamp the resolved final title (user / LLM regen / outline
        # fallback). title_source records which branch won so downstream
        # consumers and BUG_LOG forensics can tell user-typed from
        # LLM-regenerated runs without inspecting widget state.
        meta["episode_title"] = final_title
        meta["title_source"] = title_source
        if _title_sub_meta is not None:
            meta["title_substitution"] = _title_sub_meta
        if resolved["perfect_run_spacesaver"]:
            meta["perfect_run_spacesaver"] = True

        # K.5 -- voice-path-cleanbreak Sprint 2 + Sprint 6 (2026-05-12).
        # Stamp the visual_plan + style fields that OTR_VideoPlan and
        # OTR_SignalLostVideo previously read from
        # OTR_LLMDirector.production_plan_json.
        #
        # Sprint 6 changes vs Sprint 2:
        #   - genre: was hardcoded "audio drama"; now resolved from style
        #     via _GENRE_BY_STYLE (S6.1). Style-specific genre strings
        #     surface in the SignalLostVideo HUD and FLUX prompts.
        #   - voice_assignments: was persisted to meta; now derived at
        #     render time from led["cast"] via
        #     _otr_ledger_consumers.voice_assignments_from_cast (S6.2).
        #     Cast is the canonical source; persisting a derived view
        #     invited drift.
        #   - notes: was mirrored from character_description into both
        #     portrait_prompt and notes; now portrait_prompt is the only
        #     character description surface (S6.2).
        #
        # portrait_prompt is the cast row's character_description. The
        # downstream FLUX composer (compose_shot_prompt) appends era_tail
        # + style_tail at render time, so this short, content-focused
        # field is the right Tier-1 input. The 3-tier fallback in
        # resolve_character_portrait already covers the empty case.
        #
        # scenes is intentionally empty -- the writer doesn't emit
        # scene-level visual blocking today. OTR_VideoPlan handles the
        # empty list gracefully (extract_scenes returns [] and the
        # caller drives the per-shot composition off beats instead).
        _cast_rows = led.data.get("cast") or []
        _visual_chars = {}
        for _row in _cast_rows:
            if not isinstance(_row, dict):
                continue
            _name = _row.get("name")
            if not _name:
                continue
            _desc = (_row.get("character_description") or "").strip()
            _visual_chars[_name] = {
                "portrait_prompt": _desc,
            }
        meta["visual_plan"] = {
            "characters": _visual_chars,
            "scenes":     [],
            "style":      resolved["style"],
            "genre":      _resolve_genre(resolved["style"]),
        }
        meta["style"] = resolved["style"]

        # --- L. Assemble return values --------------------------------
        # Tier 1 fix #2 (2026-05-11): derive final script_text from the
        # CANONICAL ledger rows, not from the in-flight script_text_parts
        # list. Post-loop mutations (news_close_brief announcer override
        # in I.5, title substitution in J.6) write to led.data["lines"]
        # but were not always mirrored back into script_text_parts. The
        # script_text_parts list is now diagnostic-only; the ledger is
        # the source of truth for the slot-0 STRING output.
        script_text = _PL.assemble_script_text_from_ledger(led.data)
        script_json = json.dumps(led.data, indent=2, ensure_ascii=False)
        news_json = _build_news_payload(
            outline, resolved["news_seed"], resolved["seed_source"],
        )

        actual_word_count = sum(
            int(r.get("word_count") or 0) for r in led.data["lines"]
        )
        est_minutes = max(
            1, round(actual_word_count / WORDS_PER_MINUTE_ESTIMATE, 1),
        )

        # --- M. Save ledger -------------------------------------------
        saved_path = led.save()
        log.info(
            "[OTR_LedgerScriptWriter] DONE: episode_id=%s, lines=%d, "
            "words=%d, est_minutes=%s, ledger=%s",
            episode_id, len(led.data["lines"]), actual_word_count,
            est_minutes, saved_path,
        )
        return (script_text, script_json, news_json, est_minutes)


# ---------------------------------------------------------------------------
# Self-test (no-model smoke)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import traceback

    failures: list = []

    # 1. Class instantiation.
    try:
        cls = OTR_LedgerScriptWriter
        obj = cls()
        assert obj is not None
        print("[1/9] PASS: class instantiation")
    except Exception:
        failures.append(("1/9 class instantiation", traceback.format_exc()))
        print("[1/9] FAIL: class instantiation")

    # 2. INPUT_TYPES schema introspection.
    #     Post-Phase-3 cleanup 2026-05-11: legacy widgets dropped
    #     (cleanup_model_id, self_critique, target_length,
    #     arc_enhancer). act_count sits where target_length was.
    try:
        spec = cls.INPUT_TYPES()
        assert "required" in spec, "missing required block"
        assert "optional" in spec, "missing optional block"
        # Required block: episode_title, target_words, num_characters.
        req_keys = list(spec["required"].keys())
        assert req_keys == ["episode_title", "target_words", "num_characters"], \
            f"required widget order drift: {req_keys}"
        # Optional block: the clean set after Phase 0-3 cleanup.
        for k in ("seed", "model_id", "custom_premise",
                  "include_act_breaks", "act_count", "style",
                  "style_custom", "creativity",
                  "optimization_profile", "perfect_run_spacesaver"):
            assert k in spec["optional"], f"optional missing key: {k}"
        # Legacy widgets MUST be absent post-cleanup.
        for legacy in ("cleanup_model_id", "self_critique",
                       "target_length", "arc_enhancer", "open_close"):
            assert legacy not in spec["optional"], \
                f"legacy widget {legacy!r} resurrected"
        # seed widget: INT, 0..2^32-1, default 42 (post-cleanup
        # cosmetic flip; shuffle-on randomizes regardless).
        seed_type, seed_meta = spec["optional"]["seed"]
        assert seed_type == "INT", f"seed type drift: {seed_type!r}"
        assert seed_meta["min"] == 0
        assert seed_meta["max"] == 2**32 - 1
        assert seed_meta["default"] == 42, \
            f"seed default drift: {seed_meta['default']!r}"
        # episode_title is a STRING (default empty).
        et_type, et_meta = spec["required"]["episode_title"]
        assert et_type == "STRING"
        assert et_meta.get("default") == ""
        # target_words INT clamps + default 350.
        tw_type, tw_meta = spec["required"]["target_words"]
        assert tw_type == "INT"
        assert tw_meta["min"] == 30 and tw_meta["max"] == 10000
        assert tw_meta["default"] == 350, \
            f"target_words default drift: {tw_meta['default']!r}"
        # num_characters INT clamps
        nc_type, nc_meta = spec["required"]["num_characters"]
        assert nc_type == "INT"
        assert nc_meta["min"] == 1 and nc_meta["max"] == 6
        # custom_premise is multiline STRING with empty default
        cp_type, cp_meta = spec["optional"]["custom_premise"]
        assert cp_type == "STRING"
        assert cp_meta.get("multiline") is True
        assert cp_meta.get("default") == ""
        # creativity options match the preset map keys
        cr_choices, _ = spec["optional"]["creativity"]
        assert cr_choices == _CREATIVITY_CHOICES, \
            f"creativity dropdown drift: {cr_choices}"
        # act_count INT (0 = auto-derive sentinel; JS clamps to
        # [default..max] at the UI layer per target_words).
        ac_type, ac_meta = spec["optional"]["act_count"]
        assert ac_type == "INT", f"act_count type drift: {ac_type!r}"
        assert ac_meta["min"] == 0 and ac_meta["max"] == 7
        # style combo: first entry is the LLM-auto sentinel.
        st_choices, st_meta = spec["optional"]["style"]
        assert isinstance(st_choices, list) and len(st_choices) >= 4
        assert st_choices[0] == _STYLE_AUTO_SENTINEL, \
            f"style[0] drift: {st_choices[0]!r}"
        assert st_meta.get("default") == _STYLE_AUTO_SENTINEL
        # style_custom is multiline STRING free-text override
        sc_type, sc_meta = spec["optional"]["style_custom"]
        assert sc_type == "STRING"
        assert sc_meta.get("multiline") is True
        assert sc_meta.get("default") == ""
        n_optional = len(spec["optional"])
        assert n_optional == 10, \
            f"optional widget count drift: {n_optional} (expected 10 post-cleanup)"
        print("[2/9] PASS: INPUT_TYPES schema (10 optional widgets after Phase 0-3 cleanup)")
    except Exception:
        failures.append(("2/9 INPUT_TYPES", traceback.format_exc()))
        print("[2/9] FAIL: INPUT_TYPES schema")

    # 3. Locked output contract.
    try:
        assert cls.RETURN_TYPES == ("STRING", "STRING", "STRING", "INT"), \
            f"RETURN_TYPES drift: {cls.RETURN_TYPES}"
        assert cls.RETURN_NAMES == (
            "script_text", "script_json", "news_used", "estimated_minutes",
        ), f"RETURN_NAMES drift: {cls.RETURN_NAMES}"
        assert cls.FUNCTION == "run"
        assert cls.CATEGORY == "OldTimeRadio"
        print("[3/9] PASS: output contract")
    except Exception:
        failures.append(("3/9 output contract", traceback.format_exc()))
        print("[3/9] FAIL: output contract")

    # 4. _build_truncating_generate_fn returns a callable; top_p override.
    try:
        fake_cache = {"model": None, "tokenizer": None, "context_cap": 8192}
        gen_default = _build_truncating_generate_fn(fake_cache)
        gen_custom = _build_truncating_generate_fn(fake_cache, top_p=0.99)
        assert callable(gen_default)
        assert callable(gen_custom)
        print("[4/9] PASS: truncating generate_fn build (default + top_p override)")
    except Exception:
        failures.append(("4/9 generate_fn build", traceback.format_exc()))
        print("[4/9] FAIL: truncating generate_fn build")

    # 5. _resolve_creativity / 3-way style resolution / _resolve_target_words.
    try:
        # 5a. creativity presets land on the right (temp, top_p) tuple.
        for name, (et, ep) in zip(
            _CREATIVITY_CHOICES,
            [(0.6, 0.9), (0.85, 0.95), (0.92, 0.98), (0.95, 0.99)],
        ):
            t, p = _resolve_creativity(name)
            assert (t, p) == (et, ep), f"creativity {name} -> ({t},{p}) != ({et},{ep})"

        # 5b. Unknown creativity falls back to balanced.
        t, p = _resolve_creativity("???")
        assert (t, p) == (0.85, 0.95)

        # 5c. _resolve_target_words clamps to schema minimum.
        # (Smoke-preset force logic was retired with the
        # target_length widget 2026-05-11; type target_words=30
        # directly for smoke runs.)
        assert _resolve_target_words(350) == 350
        assert _resolve_target_words(1400) == 1400
        assert _resolve_target_words(0) == 5, "min-clamp guard"

        print("[5/9] PASS: resolver helpers (creativity + target_words clamp)")
    except Exception:
        failures.append(("5/9 resolver helpers", traceback.format_exc()))
        print("[5/9] FAIL: resolver helpers")

    # 6. _resolve_inputs 3-way style resolution (custom_premise path).
    try:
        # 6a. style_custom non-empty wins over combo.
        out = _resolve_inputs(
            target_words=350, num_characters=2,
            custom_premise="A real seed for testing.",
            style="noir mystery",
            style_custom="rust-belt cyber-noir",
            creativity="balanced",
        )
        assert out["news_seed"] == "A real seed for testing."
        assert out["seed_source"] == "custom_premise"
        assert out["style"] == "rust-belt cyber-noir", out["style"]
        assert out["style_source"] == "style_custom"
        assert out["style_pending"] is False
        assert out["target_words"] == 350
        assert "target_seconds" not in out, \
            f"target_seconds must not appear in resolved dict (words-only contract per Jeffrey 2026-05-10)"
        assert out["temperature"] == 0.85 and out["top_p"] == 0.95

        # 6b. Combo (non-auto, non-empty) used verbatim when style_custom blank.
        out = _resolve_inputs(
            target_words=350, num_characters=2,
            custom_premise="seed",
            style="noir mystery",
            style_custom="",
        )
        assert out["style"] == "noir mystery"
        assert out["style_source"] == "style_combo"
        assert out["style_pending"] is False

        # 6c. Auto sentinel -> style_pending=True, style stays empty.
        out = _resolve_inputs(
            target_words=350, num_characters=2,
            custom_premise="seed",
            style=_STYLE_AUTO_SENTINEL,
            style_custom="",
        )
        assert out["style"] == ""
        assert out["style_source"] == "llm_auto"
        assert out["style_pending"] is True

        # 6d. Empty style combo also routes to LLM auto.
        out = _resolve_inputs(
            target_words=350, num_characters=2,
            custom_premise="seed",
            style="",
            style_custom="",
        )
        assert out["style_pending"] is True
        assert out["style_source"] == "llm_auto"

        print("[6/9] PASS: _resolve_inputs (custom_premise + 3-way style resolution)")
    except Exception:
        failures.append(("6/8_resolve_inputs custom + style 3-way", traceback.format_exc()))
        print("[6/9] FAIL: _resolve_inputs(custom_premise + style 3-way)")

    # 7. Two-pass style picker smoke. The picker module
    #    (nodes/_otr_style_picker.py) has its own dedicated test
    #    file (tests/test_otr_style_picker.py) with 45 cases
    #    covering grammar / parse / chooser / model / end-to-end.
    #    This in-writer smoke only proves the picker module is
    #    importable AND produces a StylePick model on a happy path,
    #    so writer-only refactors can't ship a stale picker
    #    integration.
    try:
        import random as _random_smoke
        from nodes import _otr_style_picker as _OTRSP_smoke

        _five = [
            "decommissioned_dish_signal",
            "midnight_newsroom_emergency",
            "vacuum_chamber_breach",
            "haunted_repeater_loop",
            "frozen_telemetry_archive",
        ]
        _responses = ["\n".join(_five), "vacuum_chamber_breach"]
        _idx = [0]

        def _smoke_gen(messages, *, temperature, max_new_tokens):
            r = _responses[_idx[0]]
            _idx[0] += 1
            return r

        pick = _OTRSP_smoke.pick_style(
            _smoke_gen,
            article_text="Smoke test article body about a real science story.",
            seed_pool=list(_STYLE_PICKER_SEED_POOL),
            rng=_random_smoke.Random(42),
            model_id="smoke",
        )
        assert pick.chosen == "vacuum_chamber_breach", \
            f"expected chooser pick, got {pick.chosen!r}"
        assert pick.candidates == _five, \
            f"expected canned candidates, got {pick.candidates!r}"
        assert pick.pass1_attempts == 1
        assert len(pick.seed_sample) == 5
        assert len(pick.article_hash) == 64

        # Fail-loud check: empty article precondition raises.
        try:
            _OTRSP_smoke.pick_style(
                _smoke_gen, article_text="",
                seed_pool=list(_STYLE_PICKER_SEED_POOL),
                rng=_random_smoke.Random(0), model_id="smoke",
            )
            raise AssertionError(
                "expected StyleGenerationFailedError on empty article"
            )
        except _OTRSP_smoke.StyleGenerationFailedError:
            pass  # expected

        print("[7/9] PASS: _otr_style_picker integration smoke (happy + precondition)")
    except Exception:
        failures.append(("7/9 _otr_style_picker integration smoke", traceback.format_exc()))
        print("[7/9] FAIL: _otr_style_picker integration smoke")

    # 8. _generate_title_from_script post-composition title regen.
    try:
        SAMPLE_SCRIPT = (
            "[VOICE: ANNOUNCER, neutral] Tonight, on Tales From Beyond.\n\n"
            "[VOICE: AEGEUS, tense] The signal -- it's repeating itself.\n\n"
            "[VOICE: PHOEBE, alarmed] That's impossible. The dish was "
            "decommissioned six years ago.\n\n"
            "[SFX: low rumble of static]"
        )

        # 8a. Empty script -> "" (no LLM call attempted).
        def _trap(*a, **kw):
            raise AssertionError("LLM should not be called on empty script")
        assert _generate_title_from_script(_trap, "") == ""
        assert _generate_title_from_script(_trap, "   \n  \n") == ""

        # 8b. LLM raises -> "".
        def _raises(*a, **kw):
            raise RuntimeError("LLM offline")
        assert _generate_title_from_script(_raises, SAMPLE_SCRIPT) == ""

        # 8c. Clean 3-word title -> verbatim.
        def _clean(*a, **kw):
            return "The Echo Below"
        assert _generate_title_from_script(_clean, SAMPLE_SCRIPT) == "The Echo Below"

        # 8d. Wrapped "**Title:** \"Pulse\"" -> "Pulse".
        def _wrapped(*a, **kw):
            return '**Title:** "Pulse"'
        assert _generate_title_from_script(_wrapped, SAMPLE_SCRIPT) == "Pulse"

        # 8e. Smart quotes -> stripped.
        def _smart(*a, **kw):
            return "“Agri-Crash”"
        assert _generate_title_from_script(_smart, SAMPLE_SCRIPT) == "Agri-Crash"

        # 8f. Multi-line: only first non-empty line, drop explanation.
        def _explain(*a, **kw):
            return "Pulse Out\n\nThis title evokes the magnetic pulse."
        assert _generate_title_from_script(_explain, SAMPLE_SCRIPT) == "Pulse Out"

        # 8g. Stuck defaults rejected.
        for stuck in ("Untitled", "Signal Lost", "Episode", "the last frequency"):
            def _stuck(*a, **kw):
                return stuck
            assert _generate_title_from_script(_stuck, SAMPLE_SCRIPT) == "", \
                f"stuck default {stuck!r} should be rejected"

        # 8h. Full-sentence leak (>10 words) rejected.
        def _leak(*a, **kw):
            return (
                "Here is a title that the model leaked as a complete "
                "English sentence well over ten words long indeed"
            )
        assert _generate_title_from_script(_leak, SAMPLE_SCRIPT) == ""

        # 8i. Empty LLM output -> "".
        def _empty(*a, **kw):
            return ""
        assert _generate_title_from_script(_empty, SAMPLE_SCRIPT) == ""

        # 8j. 80+ char title gets truncated to 80.
        def _long(*a, **kw):
            return "X" * 90 + " A Title"
        result_long = _generate_title_from_script(_long, SAMPLE_SCRIPT)
        # The full output is "XXX..." (~98 chars, 2 words). Words count
        # <= 10 so it passes the overlong-sentence gate; length cap then
        # trims to 80.
        assert len(result_long) <= 80, f"title truncation failed: len={len(result_long)}"

        # 8k. Trailing punctuation stripped.
        def _punct(*a, **kw):
            return "Final Frequency."
        assert _generate_title_from_script(_punct, SAMPLE_SCRIPT) == "Final Frequency"

        # 8l. News seed must NOT be in the prompt the helper builds.
        # We capture what generate_fn receives and assert news_seed-like
        # tokens never appear. This is the Jeffrey 2026-05-10 contract.
        captured: dict = {}
        def _capture(messages, **kw):
            captured["messages"] = messages
            return "Clean Title"
        _generate_title_from_script(_capture, SAMPLE_SCRIPT)
        full_prompt_text = " ".join(
            m.get("content", "") for m in captured["messages"]
        )
        # The sample script DOES appear (that's the whole point); but
        # nothing news-seed-flavored should leak. Assert the system msg
        # and user msg do not mention "news", "headline", "article",
        # "RSS", or the word "seed".
        for forbidden in ("news", "headline", "article", "RSS", "seed"):
            assert forbidden.lower() not in full_prompt_text.lower(), (
                f"title-regen prompt leaked forbidden token {forbidden!r}: "
                f"{full_prompt_text[:200]}..."
            )

        print("[8/9] PASS: _generate_title_from_script (12 paths + news-seed-free contract)")
    except Exception:
        failures.append(("8/9 _generate_title_from_script", traceback.format_exc()))
        print("[8/9] FAIL: _generate_title_from_script")

    # 9. _substitute_title_in_text — spoken-title swap for J.6.
    try:
        # 9a. Exact whole-phrase swap.
        out, n = _substitute_title_in_text(
            "Tonight on Tales From Beyond: The Echo Below. Listen close.",
            "The Echo Below", "Pulse Out",
        )
        assert out == "Tonight on Tales From Beyond: Pulse Out. Listen close.", out
        assert n == 1

        # 9b. Case-insensitive match preserves new-title casing.
        out, n = _substitute_title_in_text(
            "the echo below is closing.", "The Echo Below", "Pulse Out",
        )
        assert out == "Pulse Out is closing.", out
        assert n == 1

        # 9c. Multiple occurrences in one string.
        out, n = _substitute_title_in_text(
            "The Echo Below opens. Stay with us for The Echo Below.",
            "The Echo Below", "Pulse Out",
        )
        assert n == 2
        assert "The Echo Below" not in out

        # 9d. Whole-phrase boundary: substring inside a longer word stays put.
        out, n = _substitute_title_in_text(
            "Pulseware engineering at the Pulse station.",
            "Pulse", "Surge",
        )
        # Min-length guard kicks in here (len("Pulse")=5 >= 4), and the
        # word-boundary regex must NOT replace "Pulse" inside "Pulseware".
        assert "Pulseware" in out, f"word-boundary broken: {out!r}"
        assert "Surge station" in out, f"standalone word missed: {out!r}"
        assert n == 1

        # 9e. Min-length guard rejects too-short titles.
        out, n = _substitute_title_in_text(
            "ICE forms on the dome.", "ICE", "GLASS",
        )
        assert n == 0, f"min-length guard failed: out={out!r} n={n}"
        assert out == "ICE forms on the dome."

        # 9f. Same-title no-op (saves regex compile).
        out, n = _substitute_title_in_text(
            "The Echo Below.", "The Echo Below", "The Echo Below",
        )
        assert out == "The Echo Below."
        assert n == 0

        # 9g. Empty inputs return safely.
        assert _substitute_title_in_text("", "old", "new") == ("", 0)
        assert _substitute_title_in_text("text", "", "new") == ("text", 0)
        assert _substitute_title_in_text("text", "old", "") == ("text", 0)

        # 9h. Regex-special chars in old title are escaped.
        out, n = _substitute_title_in_text(
            "Tonight: The Frequency (Lost). End.",
            "The Frequency (Lost)", "Pulse",
        )
        assert "Pulse" in out and "(Lost)" not in out, out
        assert n == 1

        # 9i. No match (title not present) returns text unchanged.
        out, n = _substitute_title_in_text(
            "Nothing to see here, move along.",
            "The Echo Below", "Pulse Out",
        )
        assert out == "Nothing to see here, move along."
        assert n == 0

        print("[9/9] PASS: _substitute_title_in_text (9 paths)")
    except Exception:
        failures.append(("9/9 _substitute_title_in_text", traceback.format_exc()))
        print("[9/9] FAIL: _substitute_title_in_text")

    # Summary.
    if not failures:
        print("\nSELF-TEST PASS: 9/9")
        sys.exit(0)
    else:
        print(f"\nSELF-TEST FAIL: {len(failures)} of 9")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
