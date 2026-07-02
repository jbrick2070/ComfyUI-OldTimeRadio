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
         - non-voiced (music_*)  → render-contract rows, text stays empty
    7. set_lines + speaker_role post-patch.
    8. Post-composition title regen (Jeffrey 2026-05-10): when the user
       left episode_title blank, ask the LLM to title the episode from
       the FINAL assembled story material. The prompt sees ONLY the
       composed dialogue excerpts + the outline premise -- not
       news_seed, not style, not RSS metadata.

       Sprint 3E (2026-05-25) -- scratchpad + late binding:
        - Title generation is a forced scratchpad pass: the model
          extracts 3 concrete physical details from the script,
          drafts 3 candidate titles, then emits a final TITLE: line;
          Python parses the title from the last TITLE: line. The
          excerpt set spans the whole arc (opening / middle / ending
          lines + premise) so the title is not titled off the
          opening act alone.
        - Late binding: the per-line composer in step 6 ran with the
          literal `EPISODE_TITLE: TBD` in canon_header, so NO
          provisional / outline title is ever spoken in dialogue.
          Because the real title is bound late (after the script
          exists), the fragile post-hoc verbatim string substitution
          of the old title in spoken lines is removed entirely -- it
          only ever caught verbatim quotes and let paraphrases slip
          through, and with `TBD` in the header there is no old
          title to substitute.

       User-typed title still wins; outline.title is the last-resort
       fallback if the LLM call fails or its output is rejected by the
       guardrails. canon.title is updated and episode_canon.json is
       written here (deferred from step 5 specifically for this).
    9. Stamp meta block (gen_params_initial, episode_title, title_source,
       perfect_run_spacesaver, creativity, optimization_profile).
   10. Save ledger.

Output contract:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("script_text", "script_json", "news_used",
                    "estimated_minutes", "technical_model")

Widget surface (current as of 2026-05-23):
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
        creative_writing_model combo (HF LLM -- narrative passes: outline,
                                      cast, dialogue, polish, style picker)
        technical_model   combo   (HF LLM -- structured passes: JSON
                                   validators, GBNF grammar, reviewer,
                                   cast contract, format normalization)
        custom_premise    STRING  (RSS override; empty triggers feed fetch)
        include_act_breaks BOOLEAN (True -> outline LLM plans music_inter
                                    beats between acts; False -> continuous)
        act_count         combo   ('auto' derives the act count from
                                   target_words; '1'-'7' set it explicitly)
        style             combo   (tonal preset; "let the story decide"
                                   defers to two-pass LLM picker)
        style_custom      STRING  (free-text override; empty falls back
                                   to style combo)
        creativity        combo   (maps to temperature + top_p preset)
        perfect_run_spacesaver BOOLEAN (stamped on ledger.meta for
                                        RTXUpscale spacesaver)
        min_p             FLOAT   (sampling tail cut; 0.0 disables)
        repetition_penalty FLOAT  (anti-loop penalty; 1.0 disables)
        max_new_tokens_cap INT    (per-line composer token ceiling)
        enable_polish_pass BOOLEAN (optional post-compose narration-leak
                                    check)

    The optimization_profile combo was a widget here until 2026-05-23;
    removed in the ROADMAP PRIORITY 2 UI simplification (only "Standard"
    was ever validated). _resolve_inputs still defaults it to "Standard"
    and stamps it to meta, so the loader plumbing is intact.

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
import os
import re
from datetime import datetime
from pathlib import Path

# S30 B2a: catalog drives the dropdown_choices() for the two model widgets
# at INPUT_TYPES() registration time. Pure-Python module, no torch /
# transformers / GPU work -- safe at module-import time.
from . import _otr_model_catalog as _otr_model_catalog  # noqa: E501

# Sprint C C5a2 (2026-05-15) module-level import per E-22 / RR-B4. The
# reflection pure module is wired into execute() at K.5.5 -- see the
# reflection call site below the K.5 visual_plan stamp. Module-level
# import (not hot-path) so a typo / refactor surfaces at module load
# time rather than during the first script generation.
from ._otr_story_brief import run_story_brief_reflection

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerScriptWriter"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOICED_ROLES = {"character", "announcer"}
"""Speaker roles that produce spoken dialogue. These trigger an LLM
compose_line call. Other roles (music_*) skip the LLM."""

NON_VOICED_ROLES = {"music_open", "music_close", "music_inter"}
"""Speaker roles that are pure render contracts: no LLM call, no
transcript text (the 'sfx' role was removed 2026-07-01, rip-sfx-broll)."""

DEFAULT_TRAITS = "neutral"
"""Fallback traits string when a beat has no mood. Mirrors the
'traits = beat.mood or "neutral"' rule from the kickoff prompt."""

# S30 B7: _otr_model_catalog.DEFAULT_LLM literal DELETED. The canonical default
# lives in _otr_model_catalog.DEFAULT_LLM (already imported at the
# top of this module). Every call site that used _otr_model_catalog.DEFAULT_LLM
# now references _otr_model_catalog.DEFAULT_LLM directly. B7's
# forbidden-pattern sweep locks the symbol name out of any new
# runtime code.

LAST_LINES_WINDOW = 5

# Story-scaffold UI toggle (2026-06-24) -- the OTR_ENABLE_STYLE_GRAMMAR value the
# process STARTED with (the headless/operator env, or None when unset). The
# writer's `story_scaffold` widget can force the scaffold on/off per run (it sets
# the env so every downstream config + module read is consistent); the "auto"
# setting restores THIS baseline so an on/off run never leaks to the next prompt
# in a long-lived server.
_OTR_SCAFFOLD_ENV_BASELINE = os.environ.get("OTR_ENABLE_STYLE_GRAMMAR")
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

# BUG-LOCAL-260: operator control for the LEMMY easter-egg cameo. The
# roll itself is OS-entropy (cast_pools.roll_lemmy, decoupled from the
# C7 seed); this widget lets the operator override the ~11% chance.
_LEMMY_CAMEO_CHOICES = ["roll (~11% chance)", "always include", "never include"]
_LEMMY_CAMEO_FORCE = {
    "roll (~11% chance)": None,    # natural ~11% OS-entropy roll
    "always include": True,         # force the cameo into the cast
    "never include": False,         # keep the cameo out of the cast
}


# target_length widget removed 2026-05-11 (post-Phase-3 cleanup pass).
# The old "short (3 acts)" / "medium (5 acts)" / "long (7-8 acts)" combo
# is replaced by the `act_count` combo widget + `target_words`. Smoke
# presets are gone with it -- for a 30-word smoke run, type
# target_words=30 directly. Cleaner UX, one source of truth for
# episode shape.


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


# Sprint C C3 (2026-05-15): _GENRE_BY_STYLE table + _resolve_genre +
# _preview_genre helpers deleted. The meta.visual_plan.genre stamp they
# fed (formerly emitted by section K.5 below) is retired. Downstream
# consumers (HUD overlay, FLUX scene-prompt composition, treatment txt,
# video info card) fall back to `meta.style` directly -- the slug
# carries enough information for those surfaces, and the parallel
# "genre" string was a derived denormalization that needed a separate
# keep-in-sync contract for no real win. Per the no-legacy-back-compat
# standing directive: deleted outright, no shim, no alias. Any caller
# that still imports `_resolve_genre` / `_preview_genre` /
# `_GENRE_BY_STYLE` from OTR_LedgerScriptWriter will get
# AttributeError -- intentional, so dead wirings fail loud.


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
# Model dropdowns -- S30 B2a two-widget surface.
#
# The hardcoded _MODEL_CHOICES list was deleted in B2a; both writer slots
# now build their dropdown live from `_otr_model_catalog.dropdown_choices()`,
# which scans the local HF cache and applies the [NOT DOWNLOADED] suffix
# to curated entries not yet on disk. The single legacy "model_id" widget
# was replaced by `creative_writing_model` + `technical_model`; broadcast
# as two STRING output sockets at the end of RETURN_NAMES.
# ---------------------------------------------------------------------------


# The optimization_profile widget was removed from INPUT_TYPES on
# 2026-05-23 (UI simplification, ROADMAP PRIORITY 2): of its VRAM tiers
# only "Standard" was ever validated. This tier list is retained --
# _resolve_inputs keeps its "Standard" default and the meta plumbing,
# so re-exposing the widget when the v2 loader's profile branches land
# is a one-line INPUT_TYPES add against this list.
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

        def __call__(self, input_ids, scores, **kwargs):  # noqa: D401  # kept: scores required by HF StoppingCriteria contract
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


# ---------------------------------------------------------------------------
# S30 B2b: writer-side LLM slot scheduler.
#
# Encapsulates per-slot generate_fn construction + request_slot
# invocation + transition counting. Two configurable slots:
#   - creative   (narrative passes: outline, cast, dialogue, polish,
#                 style picker invention, title regen)
#   - technical  (structured passes: GBNF / JSON validators,
#                 reviewer verdicts, news_interpreter, style chooser,
#                 cast contract schema validation, critic)
#
# Each for_slot(slot) call returns a fresh generate_fn closure tied to
# that slot. The closure invokes _otr_model_loader.request_slot at call
# time, so when the user picks a different technical_model than the
# creative_writing_model, crossing a slot boundary transparently
# triggers the loader's full teardown + reload. When both slots
# resolve to the same model id (default), every call cache-hits on
# the resident model -- zero transitions.
#
# Polish always routes to the creative slot (the W4 fix exists to keep
# polish sampling distinct from composer sampling; it has nothing to
# do with the creative-vs-technical model split).
# ---------------------------------------------------------------------------


class _SlotScheduler:
    """Writer-side slot scheduler for the S30 two-model selector.

    Holds the resolved per-slot model ids + the writer's sampling
    config. for_slot(slot) returns a generate_fn closure that lazily
    request_slot's the right model on every invocation. for_polish()
    returns a polish-tuned closure that always routes through the
    creative slot.

    Counts transitions and per-slot calls for forensic meta stamping
    (meta["slot_transitions"], meta["slot_calls_by_slot"]).
    """

    _ALLOWED_SLOTS = ("creative", "technical")

    def __init__(
        self,
        *,
        creative_id: str,
        technical_id: str,
        top_p: float,
        min_p: float,
        repetition_penalty: float,
    ):
        self.ids = {
            "creative": creative_id,
            "technical": technical_id,
        }
        self.sampling = {
            "top_p": float(top_p),
            "min_p": float(min_p or 0.0),
            "repetition_penalty": float(repetition_penalty or 1.0),
        }
        self.transitions = 0
        self.calls_by_slot = {"creative": 0, "technical": 0}
        self._last_resolved_id: str | None = None
        # S32 B6: per-helper / per-phase accounting for forensic meta
        # stamping. `slot_calls_by_helper` maps helper-name -> per-slot
        # call counts; `slot_transitions_by_phase` is the ordered list
        # of (phase_label, from_slot, to_slot, from_id, to_id) tuples
        # captured every time a slot transition fires.
        self.slot_calls_by_helper: dict[str, dict[str, int]] = {}
        self.slot_transitions_by_phase: list[dict] = []
        self._current_helper: str | None = None

    def _account_and_get_entry(self, slot: str) -> dict:
        """Acquire the right cache entry for `slot`. Updates transition
        count + per-slot call count. Lazy import keeps the writer's
        module-level import surface stdlib-only."""
        from . import _otr_model_loader as _OTRML

        resolved_id = self.ids[slot]
        cache_entry = _OTRML.request_slot(slot, resolved_id)
        if (
            self._last_resolved_id is not None
            and self._last_resolved_id != resolved_id
        ):
            self.transitions += 1
            # S32 B6: capture the transition with phase context.
            # `_current_helper` is set by the writer via the
            # `helper_context()` manager around each helper call.
            self.slot_transitions_by_phase.append({
                "phase": self._current_helper or "<unknown>",
                "from_slot": None,  # populated below from prior id
                "to_slot": slot,
                "from_id": self._last_resolved_id,
                "to_id": resolved_id,
            })
            # Backfill from_slot: which slot did `_last_resolved_id`
            # belong to? Look it up in self.ids.
            for s, sid in self.ids.items():
                if sid == self._last_resolved_id:
                    self.slot_transitions_by_phase[-1]["from_slot"] = s
                    break
        self._last_resolved_id = resolved_id
        self.calls_by_slot[slot] = self.calls_by_slot.get(slot, 0) + 1
        # S32 B6: per-helper accounting. When `_current_helper` is
        # unset (helper context not entered), bucket calls under
        # `"<unattributed>"` so we still capture totals; in practice
        # the writer wraps every helper call site so this fallback
        # bucket should stay at 0 in production.
        helper = self._current_helper or "<unattributed>"
        bucket = self.slot_calls_by_helper.setdefault(
            helper, {"creative": 0, "technical": 0}
        )
        bucket[slot] = bucket.get(slot, 0) + 1
        return cache_entry

    def helper_context(self, helper_name: str):
        """Context manager: attribute slot calls made within `with` to
        `helper_name`. Used by the writer to wrap each helper call so
        the per-helper bucket in `slot_calls_by_helper` and the
        `phase` field on `slot_transitions_by_phase` get populated.
        """
        scheduler = self

        class _HelperCtx:
            def __enter__(self):
                self._prior = scheduler._current_helper
                scheduler._current_helper = helper_name
                return scheduler

            def __exit__(self, exc_type, exc, tb):
                scheduler._current_helper = self._prior
                return False

        return _HelperCtx()

    def for_slot(self, slot: str):
        """Return a generate_fn closure that targets `slot`. Each call
        ensures the right model is resident before generation fires."""
        if slot not in self._ALLOWED_SLOTS:
            raise ValueError(
                f"_SlotScheduler.for_slot: slot must be one of "
                f"{self._ALLOWED_SLOTS!r}; got {slot!r}"
            )
        scheduler = self

        def generate_fn(messages, *, temperature, max_new_tokens, stop=None):
            cache_entry = scheduler._account_and_get_entry(slot)
            base = _build_truncating_generate_fn(
                cache_entry, **scheduler.sampling,
            )
            return base(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                stop=stop,
            )

        return generate_fn

    def for_polish(self):
        """Return a conservative-sampling generate_fn on the creative
        slot. Retained as a scheduler primitive (wraps the kept
        make_polish_generate_fn) after the 2026-05-29 lean-down removed
        the polish *feature* (widget + compose_line pass + symbols);
        no production caller remains, but the slot-routing contract +
        its tests keep this creative-slot conservative-sampling helper."""
        scheduler = self

        def polish_fn(messages, *, temperature, max_new_tokens):
            cache_entry = scheduler._account_and_get_entry("creative")
            from . import _otr_model_loader as _OTRML

            base = _OTRML.make_polish_generate_fn(cache_entry)
            return base(
                messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        return polish_fn



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
    # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged remote
    # entry has no model/tokenizer/context_cap to close over; return the
    # remote generate_fn before capturing local handles below. The remote
    # model does its own prompt budgeting server-side and honours the
    # caller's per-call temperature + stop. Zero local VRAM.
    if cache_entry.get("provider") == "openrouter":
        from . import _otr_openrouter_backend as _orb
        return _orb.make_openrouter_generate_fn(cache_entry)
    # [Comfy Credits] sibling remote seam (2026-06-01). Same provider-tag
    # dispatch as OpenRouter: a credit-billed entry has no model/tokenizer/
    # context_cap to close over; return the remote generate_fn before
    # capturing local handles below. Server-side budgeting; zero local VRAM.
    if cache_entry.get("provider") == "comfy_credits":
        from . import _otr_comfy_backend as _occ
        return _occ.make_comfy_credits_generate_fn(cache_entry)
    # [Ollama] LOCAL llama.cpp/Ollama lane (2026-06-04). Same provider-tag
    # dispatch, but the endpoint is a local daemon on 127.0.0.1 -- no model/
    # tokenizer/context_cap to close over; zero ComfyUI-process VRAM. Fail-closed
    # local-only (never cloud), no API key, no credit cost.
    if cache_entry.get("provider") == "ollama":
        from . import _otr_ollama_backend as _oll
        return _oll.make_ollama_generate_fn(cache_entry)
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
    # BUG-LOCAL-262: probe the tokenizer's chat template once per model
    # residency. None = not yet probed; True/False = supports a system
    # role or not. Gemma-2's template hard-rejects the system role, so
    # normalize_messages_for_tokenizer folds system content into the
    # first user turn. Closure-cell idiom matches `_min_p_unsupported`.
    _system_role_supported = [None]

    def generate_fn(messages, *, temperature, max_new_tokens, stop=None):
        import torch  # local import; never load torch at module import
        from . import _otr_loader_backends as _OTRLB
        if _system_role_supported[0] is None:
            _system_role_supported[0] = (
                _OTRLB.tokenizer_supports_system_role(tokenizer)
            )
        if not _system_role_supported[0]:
            messages = _OTRLB.normalize_messages_for_tokenizer(
                tokenizer, messages,
            )
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


def _build_title_excerpt_set(
    assembled_script: str,
    *,
    head_lines: int = 6,
    mid_lines: int = 6,
    tail_lines: int = 6,
) -> dict:
    """Slice the assembled script into opening / middle / ending excerpts.

    Sprint 3E (2026-05-25): the title pass used to receive one thin
    head-of-script slice (`assembled_script[:3000]`), which on a long
    episode is the opening act only -- the model titled the show off
    the setup and never saw the climax or the ending. This helper
    splits the script into three windows so the title prompt sees the
    whole arc: how the episode opens, what happens in its middle, and
    how it lands.

    Splits on the blank-line-delimited token blocks produced by the
    per-beat loop (each `[VOICE: ...]` block is one
    item joined by "\\n\\n"). Returns a dict with `opening_lines`,
    `middle_lines`, `ending_lines` strings; empty strings when the
    script is empty. Pure stdlib, never raises.
    """
    text = (assembled_script or "").strip()
    if not text:
        return {
            "opening_lines": "",
            "middle_lines":  "",
            "ending_lines":  "",
        }
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    n = len(blocks)
    if n == 0:
        return {
            "opening_lines": "",
            "middle_lines":  "",
            "ending_lines":  "",
        }
    opening = blocks[:head_lines]
    ending = blocks[-tail_lines:] if n > tail_lines else []
    # Middle window centred on the script's midpoint, excluding any
    # block already claimed by the opening or ending window so the
    # three excerpts do not overlap on a short episode.
    mid_center = n // 2
    mid_start = max(0, mid_center - mid_lines // 2)
    middle = blocks[mid_start:mid_start + mid_lines]
    claimed = set(range(0, len(opening)))
    if ending:
        claimed |= set(range(n - len(ending), n))
    middle = [
        b for i, b in enumerate(blocks[mid_start:mid_start + mid_lines])
        if (mid_start + i) not in claimed
    ]
    return {
        "opening_lines": "\n".join(opening),
        "middle_lines":  "\n".join(middle),
        "ending_lines":  "\n".join(ending),
    }


def _generate_title_from_script(
    generate_fn,
    assembled_script: str,
    *,
    temperature: float = 0.85,
    premise: str = "",
    arc_verdict: str = "",
) -> str:
    """Generate a 2-5 word episode title via a forced scratchpad pass.

    Per Jeffrey 2026-05-10: "title should generate only AFTER the whole
    story is done via the LLM, nothing with the news seed". The prompt
    sees ONLY the finished story material -- the assembled dialogue
    excerpts plus the outline premise (which is the story spine the
    listener experiences, not the news article). No news_seed, no style
    hint, no RSS metadata.

    Sprint 3E (2026-05-25): single-shot -> forced scratchpad. The model
    must first extract 3 concrete physical details from the script,
    draft 3 candidate titles, then emit a final `TITLE:` line. Python
    parses the title from the LAST `TITLE:` line in the output. The
    scratchpad makes the model ground the title in concrete imagery
    rather than free-associating off the opening act. The whole
    scratchpad + final `TITLE:` line is produced by ONE LLM call.

    The excerpt set (opening / middle / ending lines, premise, and an
    optional `arc_verdict`) is built by `_build_title_excerpt_set` +
    passed in by the writer so the model titles the whole arc, not just
    the head of the transcript. `arc_verdict` is optional -- the
    Sprint 5B whole-script critic that emits it is not built yet, so
    today the writer passes ""; the ARC block flips off cleanly when
    empty.

    Returns the cleaned title, or empty string on any failure (LLM
    raise, no parseable `TITLE:` line, stuck-default rejection, overlong
    leak, smart-quote-only wrappers that strip to nothing). Caller falls
    back to outline.title on "".

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

    excerpts = _build_title_excerpt_set(text)
    premise_str = (premise or "").strip()
    arc_str = (arc_verdict or "").strip()

    # Assemble the story-material block. Each window is capped so the
    # combined prompt stays inside the composer token budget on long
    # episodes; title generation only needs broad strokes per window.
    parts: list[str] = []
    if excerpts["opening_lines"]:
        parts.append(
            f"HOW IT OPENS:\n{excerpts['opening_lines'][:1200]}"
        )
    if excerpts["middle_lines"]:
        parts.append(
            f"THE MIDDLE:\n{excerpts['middle_lines'][:1200]}"
        )
    if excerpts["ending_lines"]:
        parts.append(
            f"HOW IT ENDS:\n{excerpts['ending_lines'][:1200]}"
        )
    if premise_str:
        parts.append(f"PREMISE:\n{premise_str[:600]}")
    if arc_str:
        parts.append(f"ARC:\n{arc_str[:300]}")
    story_block = "\n\n".join(parts)

    sys_msg = (
        "You are titling a single episode of a sci-fi radio drama. "
        "You receive the finished story material and propose an "
        "evocative 2-5 word episode title. You work on a scratchpad "
        "first, then commit to a final answer."
    )
    user_msg = (
        f"{story_block}\n\n"
        "Title this episode. Work through these steps in order:\n\n"
        "DETAILS: list 3 concrete physical details actually present "
        "in the story above -- a specific object, place, sound, or "
        "image, one per line.\n"
        "CANDIDATES: draft 3 candidate episode titles, each 2 to 5 "
        "words, each drawing on one of those details, one per line.\n"
        "TITLE: on the final line, write the single best title from "
        "your candidates.\n\n"
        "Rules for the final title:\n"
        " - 2 to 5 words\n"
        " - draw from a vivid image, key object, character, or "
        "thematic tension actually present in the story\n"
        " - feel specific and memorable, not generic\n"
        " - avoid cliches like \"The Beginning\", \"Final Chapter\", "
        "\"Untitled\", or \"Episode X\"\n\n"
        "Output the DETAILS, CANDIDATES, and TITLE sections. The final "
        "line MUST begin with \"TITLE:\" followed by the chosen title "
        "and nothing else."
    )

    clamped_temp = max(0.4, min(1.0, float(temperature)))

    try:
        raw = generate_fn(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=clamped_temp,
            # Scratchpad needs room for 3 details + 3 candidates + the
            # final TITLE: line. 24 tokens (the pre-scratchpad budget)
            # would truncate before the model ever reached TITLE:.
            max_new_tokens=160,
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

    # Parse the title from the LAST line that begins with TITLE:. The
    # scratchpad's CANDIDATES block does not use the TITLE: prefix, so
    # the last TITLE: line is unambiguously the model's committed pick.
    title_re = re.compile(
        r'^\s*(?:\*\*)?\s*(?:TITLE|Title|title)\s*:\s*(?:\*\*)?\s*(.+?)\s*$'
    )
    candidate = ""
    for ln in raw.splitlines():
        m = title_re.match(ln)
        if m and m.group(1).strip():
            candidate = m.group(1).strip()
    if not candidate:
        log.info(
            "[OTR_LedgerScriptWriter] title scratchpad produced no "
            "parseable TITLE: line; caller will fall back to "
            "outline.title (raw head: %r)",
            raw.strip()[:160],
        )
        return ""

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
        "[OTR_LedgerScriptWriter] title regen -> %r (scratchpad pass, "
        "from %d-char script)",
        candidate, len(text),
    )
    return candidate


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


def _resolve_cast_rng_seed() -> tuple[int, str]:
    """Return (seed, source) for the per-episode cast RNG.

    BUG-LOCAL-269: the cast is no longer pinned by the `seed` widget.
    A fixed `seed` reproduced ONE cast forever -- every episode opened
    with the identical characters (seed 42 always rolled HAYES VANCE /
    GULLIVER REEVES / JIMBO BLACK). Production now draws a fresh
    OS-entropy seed each episode so the cast genuinely varies.

    The OTR_CAST_SEED environment variable forces a fixed seed -- used
    by the C7 audio byte-identity regression, which needs a
    reproducible cast. Set it in ComfyUI's environment before a
    baseline-capture or regression run. This mirrors BUG-LOCAL-260's
    LEMMY decoupling: random in production, explicit force path for C7.
    """
    import os
    import random
    env = os.environ.get("OTR_CAST_SEED", "").strip()
    if env:
        return int(env), "OTR_CAST_SEED override"
    return random.SystemRandom().getrandbits(32), "OS entropy"


def _resolve_style_rng_seed() -> tuple[int, str]:
    """Return (seed, source) for the per-episode style-picker RNG.

    BUG-LOCAL-270 (twin of BUG-LOCAL-269): the style picker's Pass-1
    inventor samples 5 "seed flavors" from the inspiration pool; that
    sampling RNG was seeded from the `seed` widget, so a fixed seed
    sampled the identical 5 flavors every episode. Production now draws
    a fresh OS-entropy seed each episode so the sampled flavors vary.

    The OTR_STYLE_SEED environment variable forces a fixed seed -- the
    C7 audio byte-identity reproducibility path. Set it in ComfyUI's
    environment before a baseline-capture or regression run.
    """
    import os
    import random
    env = os.environ.get("OTR_STYLE_SEED", "").strip()
    if env:
        return int(env), "OTR_STYLE_SEED override"
    return random.SystemRandom().getrandbits(32), "OS entropy"


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
    # S30 B2a: split single model_id input into the two writer-surface
    # slots. Labels passed in may carry the [NOT DOWNLOADED] suffix from
    # the dropdown; _strip_label_suffix normalizes both before they hit
    # the meta block or any consumer.
    creative_writing_model: str = _otr_model_catalog.DEFAULT_LLM,
    technical_model: str = _otr_model_catalog.DEFAULT_LLM,
    custom_premise: str = "",
    include_act_breaks: bool = True,
    act_count: str = "auto",
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
    # Sprint 10B Wave 1 Agent B: Stage 3 validators flag.
    enable_production_stage3_validators: bool = False,
    # Sprint 2.2 (2026-05-28): when True, news_interpreter exhaustion
    # halts the run rather than graceful-degrading to meta["news"]=None.
    news_briefs_required: bool = True,
    # Build 4 (2026-05-28): grouped-exchange dialogue path. When True the
    # render loop pre-passes voiced beat groups through compose_exchange.
    use_exchange: bool = False,
    # OpenRouter 4-dropdown router (2026-06-01, S2): the two slot-slug
    # pickers. PASSIVE bindings -- threaded into the resolved dict here and
    # consumed by slot resolution in S3. Default "" so an old workflow with
    # no slot widgets resolves them as unset -> the S3 fallback chain.
    openrouter_slot_a_model: str = "",
    openrouter_slot_b_model: str = "",
    comfy_slot_a_model: str = "",
    comfy_slot_b_model: str = "",
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
    # S30 B2a: normalize each model id by stripping the [NOT DOWNLOADED]
    # dropdown suffix. Raw widget values never reach a consumer / meta
    # stamp -- catalog._strip_label_suffix is the single normalization
    # point. Default both inputs to _otr_model_catalog.DEFAULT_LLM so an empty widget
    # value (e.g. an old workflow with shorter widgets_values vector)
    # still produces a usable id.
    creative_writing_model = _otr_model_catalog._strip_label_suffix(
        str(creative_writing_model or _otr_model_catalog.DEFAULT_LLM)
    )
    technical_model = _otr_model_catalog._strip_label_suffix(
        str(technical_model or _otr_model_catalog.DEFAULT_LLM)
    )

    target_words = _resolve_target_words(target_words)
    num_characters = max(1, min(6, int(num_characters)))

    # Phase 2A: act_count resolution. The widget is a combo --
    # "auto" (the default) means auto-derive via
    # _otr_episode_budget.auto_act_count, which scales the act count up
    # with target_words (fewest acts whose widened-beat budget fits the
    # length); "1".."7" set it explicitly. An explicit pick is validated
    # against the [default..max] band by compute_episode_budget in run().
    _act_count_raw = str(act_count).strip().lower()
    if _act_count_raw in ("", "auto"):
        act_count_int = 0
    else:
        try:
            act_count_int = max(1, min(7, int(_act_count_raw)))
        except (TypeError, ValueError):
            act_count_int = 0
    if act_count_int == 0:
        try:
            from . import _otr_episode_budget as _OTRB  # type: ignore
            act_count_int = _OTRB.auto_act_count(target_words)
        except Exception as exc:  # noqa: BLE001
            # If target_words is below 30, auto_act_count raises (via
            # default_act_count); fall through to act_count=1 and let
            # compute_episode_budget surface the structured
            # InvalidEpisodeBudgetError in run().
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
        # S31 B6 Fix 1: pass `technical_model`. Post-S31 B3, the RSS
        # rerank path inside `_fetch_rss_seed_or_die` routes through
        # `_otr_model_loader.request_slot("technical", model_id)` (both
        # call sites: `_llm_rank_news_candidates` headline rank and
        # `_llm_rerank_with_bodies` body rerank). Passing
        # `creative_writing_model` here would make the slot label
        # ("technical") and the resolved id (creative model) disagree
        # in differing-slots mode -- the slot scheduler would load the
        # creative model under the technical slot label, defeating the
        # whole point of two-slot routing. In default config (creative
        # == technical) the two ids are identical so the fix is a
        # no-op at runtime; in differing-slots config (S32 forward)
        # this is load-bearing.
        news_article = _fetch_rss_seed_or_die(
            rss_style_slug, technical_model,
        )
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
        # S30 B2b: per-slot keys ONLY. The legacy `model_id` key is
        # deleted outright; consumers route via creative_writing_model
        # / technical_model. No "stamp both" hedge.
        "creative_writing_model": creative_writing_model,
        "technical_model":        technical_model,
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
        # Sprint 10B Wave 1 Agent B Stage 3 validators flag.
        "enable_production_stage3_validators":
            bool(enable_production_stage3_validators),
        # Sprint 2.2 (2026-05-28): news-brief hard-halt toggle.
        "news_briefs_required": bool(news_briefs_required),
        # Build 4 (2026-05-28): grouped-exchange dialogue path toggle.
        "use_exchange": bool(use_exchange),
        # S2 (2026-06-01): slot-slug picker values, threaded through for the
        # S3 resolver. Stored raw (the placeholder sentinel / empty value is
        # interpreted as "unset" at resolution time, not here).
        "openrouter_slot_a_model": str(openrouter_slot_a_model or ""),
        "openrouter_slot_b_model": str(openrouter_slot_b_model or ""),
        "comfy_slot_a_model": str(comfy_slot_a_model or ""),
        "comfy_slot_b_model": str(comfy_slot_b_model or ""),
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


def _otr_body_gate_hint(reasons, sq_entry) -> str:
    """Turn KILL-1 body-gate validation reasons into ONE concrete reroll note.

    SPLITS the machine reasons from ``validate_composed_grounding``:
      ``ungrounded_crisis:<toks>``  -> name the offending generic words to drop;
      ``missing_conflict_object:<obj>`` -> name the premise object to ground in.
    Falls back to the beat's ``conflict_object`` when reasons carry no object.
    Deterministic, ASCII, SFW; never raises."""
    parts: list[str] = []
    for r in reasons or ():
        r = str(r)
        if r.startswith("ungrounded_crisis:"):
            toks = [t for t in r.split(":", 1)[1].split(",") if t]
            if toks:
                parts.append(
                    "Do not lean on generic crisis machinery ("
                    + ", ".join(toks)
                    + "); say what is actually happening in this scene's "
                    "own terms."
                )
        elif r.startswith("missing_conflict_object:"):
            obj = r.split(":", 1)[1].strip()
            if obj:
                parts.append(
                    "Ground the line in the actual conflict over " + obj + "."
                )
    if not parts and isinstance(sq_entry, dict):
        obj = str(sq_entry.get("conflict_object", "") or "").strip()
        if obj:
            parts.append("Ground the line in the conflict over " + obj + ".")
    return " ".join(parts).strip()


def _otr_cast_fullnames(req) -> "tuple[str, ...]":
    """The episode's LOCKED cast FULL names for the C4 (S3) roster-caps signal:
    the multi-word entries of the UPPERCASE ``allowed_roster`` on the line
    request. NASA/UCLA-safe -- full names ONLY (a name contains a space), never
    a single ALL-CAPS token. Longest-first (so a substring name never shadows a
    longer one). Pure; never raises."""
    try:
        roster = getattr(req, "allowed_roster", ()) or ()
        return tuple(sorted(
            {str(n).strip() for n in roster if n and " " in str(n).strip()},
            key=len, reverse=True,
        ))
    except Exception:  # noqa: BLE001
        return ()


def _otr_allcaps_cast_hits(text, fullnames) -> int:
    """Count ALL-CAPS occurrences of a locked cast FULL name in ``text`` -- the
    gemma shout-leak (``...when CLARISSE GORDON claim...``). Case-SENSITIVE on
    the uppercase literal (a normal-case ``Clarisse Gordon`` is NOT a defect),
    word-boundary anchored; mirrors ``scrub_roster_vocative``'s matching. Pure;
    never raises."""
    try:
        s = "" if text is None else str(text)
        if not s or not fullnames:
            return 0
        n = 0
        for name in fullnames:
            up = str(name).strip().upper()
            if " " in up and re.search(rf"\b{re.escape(up)}\b", s):
                n += 1
        return n
    except Exception:  # noqa: BLE001
        return 0


def _otr_roster_caps_midclause(text, fullnames) -> bool:
    """True when an ALL-CAPS locked cast FULL name survives a leading/trailing
    vocative scrub -- i.e. it sits MID-CLAUSE (a grammatical subject/object,
    ``...when CLARISSE GORDON claim...``) where an in-place strip would mangle
    the sentence, so the body gate must REROLL rather than strip. A pure
    leading/trailing vocative is scrubbed by ``scrub_roster_vocative`` and does
    NOT trip this. Pure; never raises."""
    try:
        from . import _otr_line_hygiene as _HY
        scrubbed = _HY.scrub_roster_vocative(text, fullnames)
        return _otr_allcaps_cast_hits(scrubbed, fullnames) > 0
    except Exception:  # noqa: BLE001
        return False


def _otr_body_score(text, bg_entry, grounded_nouns, entity_policy, req) -> int:
    """C4 (S3) total-order defect score for ONE shipped character line -- LOWER
    is cleaner. The body gate scores the SHIPPED text of BOTH the original and a
    reroll and keeps the lower (ORIGINAL on tie), so a reroll is accepted only
    when it is genuinely better, not merely re-grounded. Weighted so grounding
    dominates, then a hard (unscrubable) leak, then truncation / run-on, then a
    cast-name shout as the tie-break:

        10*grounding_failed + 3*hard_leak + 2*trunc + 2*run_on + 1*roster_caps

    The run-on cap MATCHES C2's one-breath gate exactly (``derive_one_breath_cap``
    + relaxed ``max_clause_markers = max(3, cap // 8)``) so a budget-length
    multi-clause line C2 deliberately allowed is not counted against a reroll
    here. Pure + deterministic; never raises (any feature that errors scores 0,
    never a crash mid-render)."""
    from . import _otr_story_quality_l12 as _SQL12
    from . import _otr_line_hygiene as _HY
    try:
        s = "" if text is None else str(text)
        try:
            _roles = _SQL12.CLIMAX_CLASS_ROLES | {_SQL12.BEAT_ROLE_PRESSURE}
            g_ok, _ = _SQL12.validate_composed_grounding(
                s, bg_entry, grounded_nouns,
                max_ungrounded=0,
                require_conflict_object_on_roles=_roles,
            )
            grounding_failed = 0 if g_ok else 1
        except Exception:  # noqa: BLE001
            grounding_failed = 0
        try:
            hard_leak = 1 if _HY.verify_and_repair_line(
                s, policy=entity_policy,
            ).needs_recompose else 0
        except Exception:  # noqa: BLE001
            hard_leak = 0
        try:
            trunc = 1 if _HY.is_truncated(s) else 0
        except Exception:  # noqa: BLE001
            trunc = 0
        try:
            _cap = _HY.derive_one_breath_cap(
                getattr(req, "words_per_beat_range", (0, 0)))
            run_on = 1 if _HY.flag_one_breath(
                s, max_words=_cap, max_clause_markers=max(3, _cap // 8),
            )[0] else 0
        except Exception:  # noqa: BLE001
            run_on = 0
        roster_caps = _otr_allcaps_cast_hits(s, _otr_cast_fullnames(req))
        return (
            10 * grounding_failed
            + 3 * hard_leak
            + 2 * trunc
            + 2 * run_on
            + 1 * roster_caps
        )
    except Exception:  # noqa: BLE001 -- scoring must never break a render
        return 0


def _apply_story_scaffold_env(scaffold) -> str:
    """Resolve the ``story_scaffold`` widget into ``OTR_ENABLE_STYLE_GRAMMAR``.

    The widget is the single user-facing control over the whole bundled scaffold
    (style grammar + the KILL-1 body gate + the outline announcer-close gate,
    which all read that env). ``"on"`` / ``"off"`` override the env for THIS run;
    ``"auto"`` (or any unknown value) restores ``_OTR_SCAFFOLD_ENV_BASELINE`` --
    the value the process started with -- so an on/off run never leaks to the
    next prompt in a long-lived server. Pure side effect on ``os.environ``;
    returns the normalized scaffold string. Never raises."""
    s = str(scaffold or "auto").strip().lower()
    if s == "on":
        os.environ["OTR_ENABLE_STYLE_GRAMMAR"] = "1"
    elif s == "off":
        os.environ["OTR_ENABLE_STYLE_GRAMMAR"] = "0"
    else:  # "auto" (or any unknown value) -- respect the process baseline
        if _OTR_SCAFFOLD_ENV_BASELINE is None:
            os.environ.pop("OTR_ENABLE_STYLE_GRAMMAR", None)
        else:
            os.environ["OTR_ENABLE_STYLE_GRAMMAR"] = _OTR_SCAFFOLD_ENV_BASELINE
    return s


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
        #
        # 2026-06-01 four-dropdown router (S2): two slot-slug pickers are
        # APPENDED at the END of optional (never inserted) so existing
        # widget indices [0..18] stay put -- saved workflows bind by index.
        # The creative slot's DEFAULT is conditional: a freshly-dropped node
        # defaults to openrouter:slot-a when remote is enabled, else local
        # Mistral-Nemo. (Saved widgets_values always win; defaults apply to
        # fresh nodes only.) technical stays local -- never auto-flipped.
        # All dropdown builders are network-free (S0 disk cache only).
        try:
            from . import _otr_openrouter_backend as _orb
            _remote_on = _orb.openrouter_enabled()
            _slot_a_id = _orb.SLOT_A_ID
        except Exception:  # noqa: BLE001 -- INPUT_TYPES must never raise
            _remote_on = False
            _slot_a_id = "openrouter:slot-a"
        _creative_default = (
            _slot_a_id if _remote_on else _otr_model_catalog.DEFAULT_LLM
        )
        _slot_a_choices = _otr_model_catalog.openrouter_catalog_dropdown_choices("a")
        _slot_b_choices = _otr_model_catalog.openrouter_catalog_dropdown_choices("b")
        # Comfy Credits slot pickers (2026-06-01). Choices come from the
        # PINNED partner-node catalog (network-free); the lane shows the
        # "(enable Comfy Credits)" sentinel until OTR_ENABLE_COMFY_CREDITS=1.
        _comfy_slot_a_choices = _otr_model_catalog.comfy_catalog_dropdown_choices("a")
        _comfy_slot_b_choices = _otr_model_catalog.comfy_catalog_dropdown_choices("b")
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
                # S30 B2a: single model_id widget replaced by two slots.
                # The catalog dropdown_choices() call scans the local HF
                # cache live and applies the [NOT DOWNLOADED] suffix to
                # curated entries that aren't on disk yet. Labels are
                # stripped via _otr_model_catalog._strip_label_suffix
                # before any consumer / meta stamp gets the value -- raw
                # widget strings never reach downstream nodes.
                "creative_writing_model": (
                    _otr_model_catalog.dropdown_choices(),
                    {
                        "default": _creative_default,
                        "tooltip": (
                            "LLM for the creative/narrative passes "
                            "(outline, cast, dialogue composer, polish, "
                            "style picker invention). Mistral-Nemo is "
                            "the C7 byte-identical audio baseline. "
                            "Suffix tags like [NOT DOWNLOADED] are "
                            "stripped before HF lookup. To use a remote "
                            "OpenRouter model, set OPENROUTER_API_KEY and "
                            "pick OpenRouter A/B "
                            "(see docs/openrouter-setup.md)."
                        ),
                    },
                ),
                "technical_model": (
                    _otr_model_catalog.dropdown_choices(),
                    {
                        "default": _otr_model_catalog.DEFAULT_LLM,
                        "tooltip": (
                            "LLM for the technical/structured passes "
                            "(JSON validators, GBNF grammar output, "
                            "reviewer verdicts, cast contract checks, "
                            "format normalization, news interpreter). "
                            "Default matches creative_writing_model so "
                            "the single-model audio baseline holds; "
                            "pick a smaller model here when you want "
                            "Slot 1 != Slot 2 routing for VRAM headroom. "
                            "To use a remote OpenRouter model, set "
                            "OPENROUTER_API_KEY and "
                            "pick OpenRouter A/B (see docs/openrouter-setup.md)."
                        ),
                    },
                ),
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
                # (post-Phase-3 cleanup 2026-05-11) with this act_count
                # combo: "auto" derives the act count from target_words,
                # "1".."7" set it explicitly. compute_episode_budget is
                # the authoritative validator.
                "act_count": (
                    ["auto", "1", "2", "3", "4", "5", "6", "7"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Number of acts. 'auto' (the default) sizes "
                            "the act count from target_words via "
                            "_otr_episode_budget.default_act_count; pick "
                            "1-7 to set it explicitly.\n\n"
                            "Auto thresholds (target_words floor):\n"
                            "  30   -> 1 act\n"
                            "  150  -> 2 acts\n"
                            "  300  -> 3 acts (and all higher word counts)\n\n"
                            "compute_episode_budget is authoritative and "
                            "rejects an out-of-band explicit pick at run "
                            "time (cap = target_words // 50, ceiling 7)."
                        ),
                    },
                ),
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
                # BUG-LOCAL-260: operator control for the LEMMY cameo.
                # The natural roll is OS-entropy (~11%, decoupled from
                # the seed); this widget lets the operator force it.
                "lemmy_cameo": (
                    _LEMMY_CAMEO_CHOICES,
                    {
                        "default": "roll (~11% chance)",
                        "tooltip": (
                            "LEMMY easter-egg cameo -- the gravelly "
                            "engineer who occasionally joins the cast.\n\n"
                            "  roll (~11% chance) -- default; LEMMY may "
                            "appear at random. The roll uses OS entropy "
                            "and is NOT tied to the seed (BUG-LOCAL-260), "
                            "so a fixed seed no longer pins him on or "
                            "off.\n"
                            "  always include -- force LEMMY into the "
                            "cast this run.\n"
                            "  never include -- keep LEMMY out this "
                            "run.\n\n"
                            "'always' / 'never' consume one of the "
                            "num_characters slots, exactly as a natural "
                            "roll does."
                        ),
                    },
                ),
                # Build 4 (2026-05-28, GO_FORWARD_PLAN_v10): grouped
                # exchange dialogue path. OFF (default) keeps the per-beat
                # composer; PD1 byte-identity holds. ON runs a pre-pass
                # that groups 2-3 consecutive voiced beats and renders
                # each group as one exchange (compose_exchange) using the
                # Build 3 slot_drama_contracts + the Build 2 Tier-A
                # integrity check (one block per slot, repair-by-group
                # once, then legacy fallback). ANNOUNCER/MUSIC beats and
                # trailing singletons keep their existing pass; any
                # failure falls back to the legacy composer per beat so
                # audio is never blocked.
                "use_exchange": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Build 4 grouped-exchange dialogue. OFF (default) "
                        "keeps the per-beat composer; PD1 byte-identity "
                        "holds. ON groups 2-3 consecutive voiced beats and "
                        "renders each as one exchange (compose_exchange) "
                        "with the Build 3 contracts + Build 2 Tier-A "
                        "check; one block per slot, one repair-by-group, "
                        "then legacy fallback. ANNOUNCER/MUSIC + trailing "
                        "singletons keep their pass. Any failure falls "
                        "back to legacy per beat -- audio is never "
                        "blocked. Validate VRAM <= 14.5 GB + zero slot "
                        "drift on a live N=3 run."
                    ),
                }),
                # Sprint 10B Wave 1 Agent B (2026-05-27): in-line
                # Stage 3 validators on the legacy dialogue composer.
                # Catches speaker leaks, banned phrases, length drift,
                # pronoun mismatches, on-beat misses on the rendered
                # text BEFORE the ledger is frozen. Lines with error-
                # severity findings get ONE repair regenerate attempt
                # with the finding messages threaded in as the reroll
                # hint. Warn-severity findings are stamped without
                # regenerating. The final findings (post-repair if it
                # ran) land on meta.lines[].validation_findings for
                # soak audit. Default OFF so the legacy PD1 byte-
                # identity contract holds out-of-the-box; flip ON for
                # production smokes + the Section 6.2 A/B.
                "enable_production_stage3_validators": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Sprint 10B Wave 1 Agent B. OFF (default) "
                        "preserves PD1 byte-identity on the legacy "
                        "path -- no validators run. ON wires Stage 3 "
                        "validators (speaker-leak, banned-phrase, "
                        "length, pronoun, on-beat) into the production "
                        "compose_line for every character dialogue "
                        "beat. Errors trigger ONE repair regenerate "
                        "(hint = the validator findings); warns stamp "
                        "without regenerating. Findings stamped on "
                        "meta.lines[].validation_findings. Adds at "
                        "most one extra creative-slot LLM call per "
                        "flagged beat (~1-3s on Mistral-Nemo); a "
                        "clean line costs zero extra. Flip ON for "
                        "production smokes; OFF for the byte-identity "
                        "regression run."
                    ),
                }),
                # Sprint 2.2 (2026-05-28) -- Jeffrey 2026-05-27
                # directive: when build_news_briefs exhausts its
                # retry budget, HALT the run rather than silently
                # falling back to raw news_seed with no key_terms
                # enforcement. Defaults TRUE per the directive:
                # "the whole workflow needs to stop and re-roll news
                # until it works and stamps the ledger." The
                # operator re-queues on red graph; news_interpreter
                # pulls fresh from RSS each queue, so the re-queue
                # IS the re-roll. Set FALSE for the back-compat
                # graceful-degrade path (early-stage tests + the
                # rare "we know the brief is bad but want a draft
                # anyway" operator workflow).
                "news_briefs_required": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Sprint 2.2 (2026-05-28). ON (default): "
                        "build_news_briefs exhaustion HALTS the "
                        "writer with a red graph; operator re-"
                        "queues to re-roll news from RSS. OFF: "
                        "graceful-degrade -- meta['news']=None, "
                        "downstream consumers fall back to raw "
                        "news_seed with no key_terms enforcement. "
                        "Production should ship ON."
                    ),
                }),
                # S2 (2026-06-01): the two OpenRouter slot-slug pickers,
                # APPENDED at the END of optional (indices [19]/[20]) so the
                # existing [0..18] widget order is untouched. PASSIVE: a pick
                # here binds a real slug to openrouter:slot-a/b but does NOT
                # activate remote -- it is used only when creative_writing_model
                # / technical_model selects that handle. Choices come from the
                # S0 disk cache (network-free); remote-disabled shows the
                # "(enable OpenRouter)" sentinel. Resolution + preservation
                # land in S3.
                "openrouter_slot_a_model": (
                    _slot_a_choices,
                    {
                        "default": _slot_a_choices[0],
                        "tooltip": (
                            "OpenRouter model slug bound to the "
                            "'openrouter:slot-a' handle (the creative slot). "
                            "Passive: only used when creative_writing_model "
                            "is set to 'openrouter:slot-a'. Choices are the "
                            "cached OpenRouter catalog (run the refresh "
                            "script); shows '(enable OpenRouter)' until "
                            "OPENROUTER_API_KEY is "
                            "set. A saved slug is preserved even if absent "
                            "from a stale cache. See docs/openrouter-setup.md."
                        ),
                    },
                ),
                "openrouter_slot_b_model": (
                    _slot_b_choices,
                    {
                        "default": _slot_b_choices[0],
                        "tooltip": (
                            "OpenRouter model slug bound to the "
                            "'openrouter:slot-b' handle (the technical slot). "
                            "Passive: only used when technical_model is set "
                            "to 'openrouter:slot-b'. Choices are the cached "
                            "OpenRouter catalog; shows '(enable OpenRouter)' "
                            "until remote is enabled. Set "
                            "OTR_OPENROUTER_SLOT_B_REQUIRE_JSON=1 to limit "
                            "this slot to structured-output models. See "
                            "docs/openrouter-setup.md."
                        ),
                    },
                ),
                # Comfy Credits slot-slug pickers (2026-06-01), APPENDED at the
                # END of optional (indices [21]/[22]) so the existing [0..20]
                # widget order is untouched. PASSIVE: a pick binds a real slug
                # to comfy:slot-a/b but does NOT activate the lane -- it is used
                # only when creative_writing_model / technical_model selects
                # that handle. Choices come from the pinned partner-node catalog
                # (network-free); shows "(enable Comfy Credits)" until
                # OTR_ENABLE_COMFY_CREDITS=1.
                "comfy_slot_a_model": (
                    _comfy_slot_a_choices,
                    {
                        "default": _comfy_slot_a_choices[0],
                        "tooltip": (
                            "Comfy Credits model slug bound to the "
                            "'comfy:slot-a' handle (the creative slot). "
                            "Passive: only used when creative_writing_model "
                            "is set to 'comfy:slot-a'. Choices are the pinned "
                            "ComfyUI partner-node catalog; shows '(enable "
                            "Comfy Credits)' until OTR_ENABLE_COMFY_CREDITS=1 "
                            "and a Comfy account with credits is logged in. "
                            "Credit-billed. See docs/comfy-credits-setup.md."
                        ),
                    },
                ),
                "comfy_slot_b_model": (
                    _comfy_slot_b_choices,
                    {
                        "default": _comfy_slot_b_choices[0],
                        "tooltip": (
                            "Comfy Credits model slug bound to the "
                            "'comfy:slot-b' handle (the technical slot). "
                            "Passive: only used when technical_model is set "
                            "to 'comfy:slot-b'. Choices are the pinned "
                            "ComfyUI partner-node catalog; shows '(enable "
                            "Comfy Credits)' until the lane is enabled. "
                            "Credit-billed. See docs/comfy-credits-setup.md."
                        ),
                    },
                ),
                # Refine loop (v1, 2026-06-23) -- APPENDED at the END of optional
                # (the next widgets_values index) so existing widget indices are
                # untouched (BUG-LOCAL-097). The iterative story-REVISION loop.
                "refine_target_grade": (
                    ["Off", "C+", "B", "B+", "A"],
                    {
                        "default": "Off",
                        "tooltip": (
                            "Iterative story-REVISION loop (v1): keep REWRITING "
                            "the story (revising the existing draft) until it "
                            "reaches this grade, then ship -- or stop at a hard "
                            "cap of 5 passes. Off = disabled (single pass, the "
                            "default, byte-identical). B (~75) is a reachable "
                            "target for a local model; A (~90) may never be hit "
                            "(it then ships the last revision). Local-only. The "
                            "env vars OTR_STORY_REFINE_BAR / OTR_STORY_REFINE_"
                            "PASSES override this widget for headless runs."
                        ),
                    },
                ),
                # Story-scaffold toggle (2026-06-24) -- APPENDED at the END of
                # optional (next widgets_values index, BUG-LOCAL-097) so existing
                # widget indices are untouched. The single user-facing control
                # over the whole bundled scaffold (style grammar + the KILL-1
                # body-output gate + the announcer non-outcome close).
                "story_scaffold": (
                    ["auto", "on", "off"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "How much the radio-drama SCAFFOLD shapes the story. "
                            "off = a story drawn straight from the news seed (the "
                            "base prompt only -- no style catalog, no climax-"
                            "shape grammar, no grounding gate; the writer's own "
                            "take). on = the news story shaped by ONE of the ~100 "
                            "radio-drama styles (varied climax + ending + the "
                            "premise-grounding body gate). auto (default) = follow "
                            "the OTR_ENABLE_STYLE_GRAMMAR env / its default (ON). "
                            "on/off override that env for THIS run."
                        ),
                    },
                ),
            },
            # ComfyUI injects the logged-in account's credentials into these
            # hidden inputs at execution time (the API-nodes auth convention).
            # The writer threads them to _otr_comfy_backend.set_auth() so the
            # Comfy Credits lane can make the credit-billed call. They are NOT
            # widgets (absent from widgets_values) and are never logged.
            "hidden": {
                "auth_token_comfy_org": "AUTH_TOKEN_COMFY_ORG",
                "api_key_comfy_org": "API_KEY_COMFY_ORG",
            },
        }

    CATEGORY = "OldTimeRadio"
    FUNCTION = "run"
    # S30 B2a: two new STRING outputs broadcast the resolved model IDs
    # for downstream consumers. B2a wires the widget surface only --
    # the cascade consumer is wired in B3, the writer's internal slot
    # routing comes in B2b.
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "script_text", "script_json", "news_used", "estimated_minutes",
        "technical_model",
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

    def _refine_loop(self, _rcfg, _core_kwargs):
        """Iterative story-REVISION loop (v1, 2026-06-23). Called by run() ONLY
        when refine is enabled (effective_passes >= 2). Re-runs the writer body
        (self.run with _refine_active=True) up to N times sharing ONE cast seed;
        grades each pass and, below the target, REVISES (prior_macro + critique)
        and retries. Keep-best by grade; commit the winner; clean up losers."""
        from . import _otr_story_select as _OTRSEL
        import hashlib
        import random as _rnd
        try:
            import torch as _torch
        except Exception:  # noqa: BLE001
            _torch = None

        def _seed(ns):
            h = int(hashlib.sha256(ns.encode("utf-8")).hexdigest(), 16)
            if _torch is not None:
                _torch.manual_seed(h % (2 ** 64))
            _rnd.seed(h % (2 ** 32))

        def _build_prior_macro(_outline):
            if _outline is None:
                return ""
            try:
                _beats = getattr(_outline, "beats", []) or []
                _intents = [
                    str(getattr(b, "intent", "") or "")
                    for b in _beats
                    if str(getattr(b, "speaker_role", "") or "")
                    in ("character", "announcer")
                ]
                _bs = "; ".join(f"{j + 1}. {t}" for j, t in enumerate(_intents))
                return (
                    f"Title: {getattr(_outline, 'title', '')}\n"
                    f"Premise: {getattr(_outline, 'premise', '')}\n"
                    f"Setting: {getattr(_outline, 'setting', '')}\n"
                    f"Beats: {_bs[:600]}"
                )
            except Exception:  # noqa: BLE001
                return ""

        forced_seed = _resolve_cast_rng_seed()[0]   # ONE cast seed for all passes
        log.info(
            "[refine] ON: target=%s bar=%d max_passes=%d cast_seed=%d",
            _rcfg.target_grade, _rcfg.bar, _rcfg.effective_passes, forced_seed,
        )
        candidates = []
        prior_macro, prior_critique = "", ""
        for i in range(_rcfg.effective_passes):
            _seed(f"{forced_seed}:refine:{i}")
            try:
                out = self.run(
                    **_core_kwargs,
                    _refine_active=True,
                    _refine_prior_macro=prior_macro,
                    _refine_prior_critique=prior_critique,
                    _refine_forced_cast_seed=forced_seed,
                )
            except Exception:  # noqa: BLE001
                if i == 0:
                    raise   # pass-0 failure == the existing LOUD writer failure
                log.warning("[refine] pass %d compose failed; skipping", i)
                continue
            last = getattr(self, "_refine_last", {}) or {}
            _seed(f"{forced_seed}:refine:{i}:grade")
            grade = _OTRSEL.grade_story(
                out[0], last.get("premise", ""),
                generate_fn=last.get("creative_fn"),
            )
            log.info(
                "[refine] pass %d/%d grade=%d (target=%d) weakness=%r",
                i, _rcfg.effective_passes, grade.score_0_100, _rcfg.bar,
                str(grade.biggest_weakness)[:80],
            )
            cand = {"i": i, "grade": grade, "out": out, "last": last}
            candidates.append(cand)
            if grade.score_0_100 >= _rcfg.bar:
                break   # early-stop: this pass hit the target; it is the last
            prior_macro = _build_prior_macro(last.get("outline"))
            # T2 (2026-06-23): when the per-pass result exposes the 5B critic's
            # StoryCriticReport, build the next-pass critique from the critic
            # ADAPTER (arc_verdict -> failing_axes + reroll-target hints) so the
            # re-plan is steered by the structural critic, not only by
            # grade_story.biggest_weakness. The writer's compose body does NOT
            # currently stash a report (the 5B critic runs DOWNSTREAM in the
            # freeze cascade), so today this falls back to the grader weakness
            # -- byte-identical to the pre-T2 refine loop. The plumbing is ready
            # for a future increment that runs the critic in-pass.
            _critic_report = last.get("story_critic_report")
            _critic_hint = ""
            if _critic_report is not None:
                _faxes, _regen = _OTRSEL.critic_report_to_refine_signals(
                    _critic_report
                )
                _critic_hint = _OTRSEL.critique_to_hint(_regen)
                try:
                    _cled = (last or {}).get("led")
                    if _cled is not None and isinstance(
                        getattr(_cled, "data", None), dict
                    ):
                        _csq = _cled.data.setdefault("meta", {}).setdefault(
                            "story_quality", {}
                        )
                        if isinstance(_csq, dict):
                            _csq["critic_failing_axes"] = list(_faxes)
                            _csq["critic_regeneration_hint"] = _regen
                except Exception:  # noqa: BLE001 -- telemetry never breaks the run
                    pass
            if _critic_hint:
                prior_critique = _critic_hint
            else:
                prior_critique = (
                    "" if grade.error_type
                    else _OTRSEL.critique_to_hint(grade.biggest_weakness)
                )
        # Keep-BEST across passes (operator 2026-06-23). Revision is NOT monotonic
        # -- the cap case can leave an EARLIER pass scoring higher than the last
        # (a live gemma episode saw pass1=72 then drift to pass4=65). Ship the
        # HIGHEST-grade pass, not the last; tie -> the earliest pass (cleaner draft,
        # fewer edits). The telemetry block below re-saves the WINNER's ledger LAST,
        # so the downstream latest-ledger handoff ships THIS pass even when it is
        # not the final one composed. Earlier-pass dirs are NOT deleted (deleting
        # raced the freeze -- the PendingSweep / operator reclaims them).
        # Keep-best via the shared pure helper (T2): highest grade, ties -> the
        # earliest pass (candidates are in append/pass order). Identical result
        # to the prior max(grade, -i) comparator.
        winner = candidates[_OTRSEL.keep_best_index(
            [c["grade"].score_0_100 for c in candidates]
        )]
        _reached = winner["grade"].score_0_100 >= _rcfg.bar
        _stop = "bar_reached" if _reached else "cap_reached_below_bar"
        if not _reached:
            log.warning(
                "[refine] cap reached BELOW bar: BEST grade=%d < target=%d (%s); "
                "shipping the best of %d passes -- consider a lower target grade",
                winner["grade"].score_0_100, _rcfg.bar, _rcfg.target_grade,
                len(candidates),
            )
        # Stamp refine telemetry on the WINNER's ledger (merged) + re-save.
        try:
            _wled = (winner["last"] or {}).get("led")
            if _wled is not None and isinstance(getattr(_wled, "data", None), dict):
                _sq = _wled.data.setdefault("meta", {}).setdefault("story_quality", {})
                if isinstance(_sq, dict):
                    _sq["refine_loop"] = {
                        "requested_passes": _rcfg.requested_passes,
                        "effective_passes": _rcfg.effective_passes,
                        "max_passes": _rcfg.max_passes,
                        "bar": _rcfg.bar,
                        "target_grade": _rcfg.target_grade,
                        "override_source": _rcfg.override_source,
                        "winner_pass": winner["i"],
                        "winner_grade": winner["grade"].score_0_100,
                        "stop_reason": _stop,
                        "target_reached": bool(_reached),
                        "provider": _rcfg.provider,
                        "clamp_reason": _rcfg.clamp_reason,
                        "passes": [
                            {
                                "pass_index": c["i"],
                                "score_0_100": c["grade"].score_0_100,
                                "grade_error_type": c["grade"].error_type,
                                "grade_delta": (
                                    None if c["i"] == 0
                                    else c["grade"].score_0_100
                                    - candidates[0]["grade"].score_0_100
                                ),
                            }
                            for c in candidates
                        ],
                    }
                    _wled.save()
        except Exception:  # noqa: BLE001 -- telemetry must never break the run
            log.warning("[refine] telemetry stamp failed")
        return winner["out"]

    def run(
        self,
        episode_title="",
        target_words=350,
        num_characters=2,
        # S30 B2a: single model_id widget split into two surface widgets.
        # Both default to _otr_model_catalog.DEFAULT_LLM so the audio C7 baseline is
        # unchanged when the user accepts defaults. B2b adds the internal
        # routing that uses technical_model on structured passes; in B2a
        # both ids feed the same legacy generation path.
        creative_writing_model=_otr_model_catalog.DEFAULT_LLM,
        technical_model=_otr_model_catalog.DEFAULT_LLM,
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
        # BUG-LOCAL-260: operator control for the LEMMY cameo. Maps to
        # force_lemmy (None = natural ~11% OS-entropy roll).
        lemmy_cameo="roll (~11% chance)",
        # Sprint 10B Wave 1 Agent B (2026-05-27): in-line Stage 3
        # validators on the legacy compose_line path. Default False
        # preserves PD1 byte-identity.
        enable_production_stage3_validators=False,
        # Sprint 2.2 (2026-05-28): hard-halt when news_interpreter
        # exhausts retries. Default True per Jeffrey 2026-05-27
        # directive ("news brief must write -- if it doesn't, the
        # whole workflow needs to stop and re-roll news"). Set False
        # for back-compat graceful-degrade.
        news_briefs_required=True,
        # Build 4 (2026-05-28): grouped-exchange dialogue path (default OFF).
        use_exchange=False,
        # S2 (2026-06-01): the two OpenRouter slot-slug picker widgets.
        # Default "" so an old workflow without these widgets resolves them
        # as unset (-> S3 fallback chain). ComfyUI passes the live widget
        # value (a slug or the "(enable OpenRouter)" sentinel) by keyword.
        openrouter_slot_a_model="",
        openrouter_slot_b_model="",
        # Comfy Credits slot pickers (2026-06-01), appended after the
        # OpenRouter pair. Default "" => unset (resolves to recommended).
        comfy_slot_a_model="",
        comfy_slot_b_model="",
        # ComfyUI-injected hidden auth (API-nodes convention). None when the
        # operator is not logged in / the Comfy Credits lane is unused.
        auth_token_comfy_org=None,
        api_key_comfy_org=None,
        refine_target_grade="Off",
        # Story-scaffold UI toggle (2026-06-24): auto/on/off. Governs the whole
        # bundled scaffold via OTR_ENABLE_STYLE_GRAMMAR (see the resolver at the
        # top of the body). Default "auto" => env/default => byte-identical.
        story_scaffold="auto",
        # Refine loop (v1, 2026-06-23) -- keyword-only overrides set ONLY by
        # _refine_loop when a refine pass re-enters this body. All default to the
        # no-op so a normal (non-refine) call is byte-identical.
        *,
        _refine_active=False,
        _refine_prior_macro="",
        _refine_prior_critique="",
        _refine_forced_cast_seed=None,
    ):
        """Generate a v2.0 LPL script. See the module docstring for the pipeline.

        The optional default-OFF iterative story-REVISION loop (v1, 2026-06-23)
        is delegated to ``_refine_loop`` on the INITIAL call; each refine pass
        re-enters this body with ``_refine_active=True`` and runs it directly."""
        # Story-scaffold UI toggle (2026-06-24) -- resolve the widget into the
        # process env FIRST, before generate_outline + every style-grammar read,
        # so this single control governs the whole bundled scaffold: the style
        # grammar + the KILL-1 body-output gate (via
        # _otr_config.style_grammar_enabled) AND the outline announcer-close gate
        # (which reads OTR_ENABLE_STYLE_GRAMMAR directly). "on"/"off" override the
        # env for THIS run; "auto" restores the import-time baseline so an on/off
        # run never leaks to the next prompt in a long-lived server. A local
        # `import os` binds the name first -- run() has a later function-local
        # `import os`, which makes os function-local for the whole body (the
        # 096ef64 UnboundLocalError gotcha).
        import os
        _scaffold = _apply_story_scaffold_env(story_scaffold)
        if _scaffold in ("on", "off"):
            log.info(
                "[OTR_LedgerScriptWriter] story_scaffold=%s -> "
                "OTR_ENABLE_STYLE_GRAMMAR=%s (widget override)",
                _scaffold, os.environ.get("OTR_ENABLE_STYLE_GRAMMAR"),
            )
        if not _refine_active:
            from . import _otr_story_select as _OTRSEL_GATE
            _rcfg = _OTRSEL_GATE.resolve_refine_passes(
                creative_writing_model, widget_target=refine_target_grade,
            )
            if _rcfg.effective_passes >= 2:
                _core = {
                    k: v for k, v in locals().items()
                    if k not in (
                        "self", "refine_target_grade", "_refine_active",
                        "_refine_prior_macro", "_refine_prior_critique",
                        "_refine_forced_cast_seed", "_OTRSEL_GATE", "_rcfg",
                    )
                }
                return self._refine_loop(_rcfg, _core)

        # BUG-LOCAL-296 (2026-05-31): reset the OpenRouter per-RUN cost
        # budget at the top of every episode. The budget is a module-level
        # accumulator in _otr_openrouter_backend; reset_run_budget() was
        # defined + exported but NEVER wired into the live path, so in a
        # persistent headless server (the Scheduled Task launcher) the
        # "per-run" ceiling actually accumulated across EVERY remote episode
        # and would spuriously fail-closed after a few runs. The writer is
        # the single per-episode entry that precedes all remote LLM calls
        # (its own passes + the downstream cascade share the process global),
        # so ONE reset here scopes the ceiling to one episode. PD1: a budget
        # reset must never be load-bearing for the writer -- any import/attr
        # failure is swallowed.
        try:
            from . import _otr_openrouter_backend as _orb_budget
            _orb_budget.reset_run_budget()
            # S3 (2026-06-01): record the slot-picker widget values so the
            # backend resolves a handle (openrouter:slot-a/b) to the OPERATOR'S
            # chosen slug, demoting the env (OPENROUTER_MODEL_A/B) to a
            # fallback. Set from the RAW widget args BEFORE _resolve_inputs so
            # the binding is live even for the RSS rerank's technical-slot load
            # inside _resolve_inputs. Best-effort: on any failure resolution
            # falls back to env, so a binding hiccup never blocks the run.
            _orb_budget.set_slot_bindings(
                slot_a=openrouter_slot_a_model,
                slot_b=openrouter_slot_b_model,
            )
            # Comfy Credits sibling (2026-06-01): reset its per-run budget,
            # bind the slot pickers, and capture the ComfyUI-injected hidden
            # auth so the credit-billed call has a credential. Best-effort:
            # any hiccup leaves the lane to fail closed at call time, never
            # blocking the run (PD1).
            from . import _otr_comfy_backend as _occ_budget
            _occ_budget.reset_run_budget()
            _occ_budget.set_slot_bindings(
                slot_a=comfy_slot_a_model,
                slot_b=comfy_slot_b_model,
            )
            _occ_budget.set_auth(
                auth_token=auth_token_comfy_org,
                api_key=api_key_comfy_org,
            )
        except Exception:  # noqa: BLE001 -- budget/binding setup is best-effort
            pass

        # --- A. Resolve all widget inputs (RSS fetch happens here) -----
        resolved = _resolve_inputs(
            target_words=target_words,
            num_characters=num_characters,
            episode_title=episode_title,
            creative_writing_model=creative_writing_model,
            technical_model=technical_model,
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
            # Sprint 10B Wave 1 Agent B: propagate Stage 3 flag.
            enable_production_stage3_validators=enable_production_stage3_validators,
            # Sprint 2.2 (2026-05-28): hard-halt toggle.
            news_briefs_required=news_briefs_required,
            # Build 4 (2026-05-28): grouped-exchange dialogue path toggle.
            use_exchange=use_exchange,
            # S2 (2026-06-01): thread the slot-slug picker values through.
            openrouter_slot_a_model=openrouter_slot_a_model,
            openrouter_slot_b_model=openrouter_slot_b_model,
            comfy_slot_a_model=comfy_slot_a_model,
            comfy_slot_b_model=comfy_slot_b_model,
        )

        log.info(
            "[OTR_LedgerScriptWriter] start: creative_model=%r, "
            "technical_model=%r, target_words=%d, num_characters=%d, "
            "style_source=%s (pending=%s, value=%r), creativity=%r "
            "(temp=%.2f top_p=%.2f), seed_source=%s, episode_title=%r, "
            "perfect_run_spacesaver=%s",
            resolved["creative_writing_model"],
            resolved["technical_model"],
            resolved["target_words"],
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
        from . import _otr_continuity as _OTRCONT
        # Lean-down 2026-05-29: the multiturn dispatch was deleted; the
        # in-loop Stage1Plan adapters it hosted moved to
        # _otr_legacy_to_stage1_adapter, which the kept Stage 3
        # validators path imports here. Pure module; no LLM, no model
        # loads at import time.
        from . import _otr_legacy_to_stage1_adapter as _OTRL2S1
        from . import _otr_config as _OTRCFG
        from . import _otr_line_hygiene as _OTRHY  # leak-floor-v2 EntityPolicy
        from . import _otr_story_quality_l12 as _OTRSQL12
        from . import _otr_story_select as _OTRSEL
        from . import _otr_pitch_room as _OTRPR
        from . import _otr_style_catalog as _OTRSTYLE
        # KILL 2 (2026-06-24): hoist the style-grammar gate to ONE variable so every
        # story_scaffold branch below (the pre-outline StoryContract, the
        # OutlineRequest style fields, the safe-open capture, the news coda) reads
        # the same value. _apply_story_scaffold_env (above) already applied the
        # widget override to OTR_ENABLE_STYLE_GRAMMAR; OFF => every new branch is
        # skipped => byte-identical.
        _style_grammar_on = _OTRCFG.style_grammar_enabled()

        # --- C. Slot scheduler -- B2b two-slot LLM routing -------------
        # Replaces the single _OTRML.load_llm + _build_truncating_generate_fn
        # + make_polish_generate_fn block. The scheduler exposes per-slot
        # generate_fn closures that lazily request_slot on each call.
        # When creative_writing_model == technical_model (default) every
        # call cache-hits on one resident model and no transitions fire;
        # when they differ, crossing a slot boundary triggers a full
        # loader teardown + reload.
        #
        # Sub-pass routing (S30 routing table; B2b lands top-level
        # phases. Per-sub-pass routing inside compose_line / pick_style
        # / lock_cast / build_news_briefs stays single-fn for now;
        # the helpers receive whichever slot's fn the writer hands
        # them):
        #   Outline             -> creative
        #   Cast lock           -> creative
        #   Dialogue composer   -> creative
        #   Polish              -> creative (via for_polish)
        #   Title regen         -> creative
        #   Style picker        -> per-sub-pass (S32 B2): pass 1
        #                          inventor -> creative, pass 2
        #                          chooser -> technical; pick_style
        #                          dispatches each pass internally.
        #   News interpreter    -> technical (GBNF + pydantic + V0-V3)
        #
        # slot-interleave: when news_interpreter runs after the style
        # picker (creative -> technical) and before cast lock
        # (technical -> creative), one transition lands per direction.
        # Documented at the call sites below.
        slot_scheduler = _SlotScheduler(
            creative_id=resolved["creative_writing_model"],
            technical_id=resolved["technical_model"],
            top_p=resolved["top_p"],
            # Phase 4 v4 (2026-05-11) sampling knobs.
            min_p=resolved["min_p"],
            repetition_penalty=resolved["repetition_penalty"],
        )
        # LLM slot: creative -- bulk writer path (outline, cast,
        # dialogue, polish, style picker, title regen).
        creative_generate_fn = slot_scheduler.for_slot("creative")
        # LLM slot: technical -- structured passes (news_interpreter
        # in B2b; B4b adds RSS news rerank).
        technical_generate_fn = slot_scheduler.for_slot("technical")

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
        # Story-quality v2 is BAKED IN (operator 2026-06-28): the dialogue-craft
        # spine (objective gate, body-gate text-score, cliche span-repair,
        # one-breath budget cap, news-coda bridge, two-principal scan + telemetry)
        # IS the engine, not an opt-in lever. Always enabled -- the
        # OTR_STORY_QUALITY_V2 env kill-switch is removed so the improvement can
        # never silently regress. (Default was already True since 2026-06-23.)
        meta["story_quality_v2_enabled"] = True

        # Ledger durability P1 (2026-05-19): persist a skeleton ledger to
        # disk NOW, before the style-picker / news-interpreter / cast /
        # outline LLM phases run. Those phases are several minutes and the
        # most failure-prone part of the writer; pre-fix the first
        # led.save() was not until the Phase 2B outline stamp far below,
        # so a crash in any earlier phase left zero ledger on disk for the
        # run. The skeleton is a valid sparse ledger (episode_id, meta
        # seed, cast_status="building"); every later led.save() overwrites
        # it with real progress. Goal: a ledger on disk for every run,
        # regardless of how far it gets.
        _skeleton_path = led.save()
        log.info(
            "[OTR_LedgerScriptWriter] skeleton ledger saved up front: %s",
            _skeleton_path,
        )

        # BUG-LOCAL-290 (2026-05-27): sweep stale `pending_*` dirs.
        # Every writer error before line composition leaves a
        # 0-line pending_* dir on disk forever (17 accumulated on
        # 2026-05-27 alone). Run a sweep here -- AFTER the current
        # run's own skeleton is stamped (so it self-excludes via
        # the 2-hour age threshold) and BEFORE any expensive work
        # starts. PD1: the sweep helper never raises; a
        # filesystem failure logs a warning and the writer
        # proceeds.
        try:
            from . import _otr_paths as _OTRP
            from ._otr_pending_cleanup import sweep_empty_pending_dirs
            _episodes_root = _OTRP.otr_episodes_root()
            _sweep_report = sweep_empty_pending_dirs(_episodes_root)
            if _sweep_report.deleted:
                log.info(
                    "[OTR_LedgerScriptWriter] BUG-LOCAL-290 pending "
                    "sweep: deleted %d stale dir(s) before run start.",
                    len(_sweep_report.deleted),
                )
        except Exception as _sweep_exc:  # noqa: BLE001 -- non-fatal
            log.warning(
                "[OTR_LedgerScriptWriter] BUG-LOCAL-290 pending sweep "
                "raised %s: %s -- continuing without sweep.",
                type(_sweep_exc).__name__, str(_sweep_exc)[:200],
            )

        # D.2 Two-pass style picker (when "let the story decide" is
        # selected or combo is blank AND no style_custom override).
        # Pass 1 inventor produces 5 distinct snake_case style
        # descriptors grounded in the news article + 5 sampled seed
        # flavors. Pass 2 chooser picks the single best one.
        #
        # BUG-LOCAL-270 (twin of BUG-LOCAL-269): the seed-flavor sample
        # RNG is NO LONGER tied to the `seed` widget -- a fixed seed
        # sampled the identical 5 inspiration flavors every episode.
        # _resolve_style_rng_seed() draws a fresh OS-entropy seed per
        # episode; set OTR_STYLE_SEED to force a fixed sample for the
        # C7 audio byte-identity regression.
        # See nodes/_otr_style_picker.py for full design.
        #
        # The widget-typed style_custom and the verbatim combo entries
        # bypass this branch.
        if resolved["style_pending"]:
            style_rng_seed, style_rng_source = _resolve_style_rng_seed()
            picker_rng = _random.Random(style_rng_seed)
            log.info(
                "[OTR_LedgerScriptWriter] style picker RNG seed=%d (%s) "
                "-- seed-flavor sampling randomized per episode "
                "(BUG-LOCAL-270)",
                style_rng_seed, style_rng_source,
            )
            # LLM slot: per-sub-pass -- style picker pass 1 (inventor)
            # -> creative_fn (style invention is a narrative pass that
            # recombines seed flavors creatively); pass 2 (chooser)
            # -> technical_fn (index/grammar-checked short-output
            # structured pass). This per-sub-pass routing landed in
            # S32 B2 inside pick_style itself (nodes/_otr_style_picker.py);
            # the writer no longer routes both passes through one fn.
            # S32 B1 paired-contract wiring: pass BOTH generators.
            # S32 B6: `helper_context` attributes all slot calls
            # inside the block to "pick_style" so per-helper /
            # per-phase meta tracking gets a clean phase label.
            # Resolve the creative slot to its effective model slug for
            # style-pass telemetry (over/under-generation per model). The
            # inventor (pass 1) runs on the creative slot, so this is the
            # model whose draw the counts describe. Only an OpenRouter
            # handle needs resolving (-> its bound remote slug); a local
            # model id is recorded verbatim. Telemetry label only -- never
            # break the run over it.
            _creative_model = str(resolved["creative_writing_model"])
            _creative_slug = _creative_model
            if _creative_model.startswith("openrouter:"):
                try:
                    from . import _otr_openrouter_backend as _orb_slug
                    _creative_slug = _orb_slug.resolve_slug(_creative_model)
                except Exception:  # noqa: BLE001 -- telemetry label only
                    _creative_slug = _creative_model
            with slot_scheduler.helper_context("pick_style"):
                style_pick = _OTRSP.pick_style(
                    creative_fn=creative_generate_fn,
                    technical_fn=technical_generate_fn,
                    article_text=resolved["news_seed"],
                    seed_pool=list(_STYLE_PICKER_SEED_POOL),
                    rng=picker_rng,
                    model_id=_creative_model,
                    model_slug=_creative_slug,
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
            # LLM slot: technical -- news_interpreter emits GBNF +
            # pydantic-validated briefs (V0-V3 schema). Structured-
            # output pass; routes to the technical_model slot.
            # slot-interleave: creative (style picker) -> technical
            # (here). One transition when the two slot ids differ.
            # build_news_briefs runs every V0-V3 sub-pass on the
            # technical slot -- structured-output JSON.
            # S32 B6: helper_context attribution.
            with slot_scheduler.helper_context("build_news_briefs"):
                briefs = _OTRNI.build_news_briefs(
                    technical_fn=technical_generate_fn,
                    full_text=article.get("full_text", ""),
                    headline=article.get("headline", ""),
                    summary=article.get("summary", ""),
                    outlet=article.get("source", ""),
                    pub_date=article.get("date", ""),
                    style=resolved["style"],
                    # The `seed` widget was removed (BUG-LOCAL-269/270
                    # follow-up); a constant keeps the news-interpreter
                    # cache key stable across the seed dimension.
                    seed=0,
                    model_id=str(resolved["technical_model"]),
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
            # Sprint 2.2 (2026-05-28) -- Jeffrey 2026-05-27 directive:
            # "news brief must write -- if it doesn't, the whole
            # workflow needs to stop and re-roll news until it works
            # and stamps the ledger." The pre-Sprint-2.2 graceful-
            # degrade path (`meta["news"] = None`) silently lost the
            # script_brief + key_terms enforcement that downstream
            # consumers depend on, and Sprint 2.1's DramaticState
            # stamp ended up keyed off an empty brief on every halt.
            # The Sprint 2.2 fix: HALT loud by default; operator re-
            # queues the run (which pulls fresh from RSS, effectively
            # re-rolling news). The `news_briefs_required` toggle
            # (default True) lets the original graceful-degrade
            # surface persist for the tests + early-stage callers
            # that depend on it.
            _news_required = bool(
                resolved.get("news_briefs_required", True)
            )
            # Soak/headless escape hatch: an explicit env override lets a
            # batch run degrade (raw news_seed) instead of halting on a
            # single fabricated key_term, without editing the graph widget.
            # Production leaves this unset so the widget default governs.
            import os  # stdlib; local import matches this file's convention
            if os.environ.get("OTR_NEWS_BRIEFS_REQUIRED") == "0":
                _news_required = False
                log.warning(
                    "[OTR_LedgerScriptWriter] OTR_NEWS_BRIEFS_REQUIRED=0 "
                    "-- soak/headless escape hatch active; degrading "
                    "instead of halting on news_interpreter failure: %s",
                    exc,
                )
            if _news_required:
                log.error(
                    "[OTR_LedgerScriptWriter] news_interpreter "
                    "FAILED after all attempts AND news_briefs_"
                    "required=True (Sprint 2.2 default): %s -- "
                    "HALTING the run. Operator should re-queue; "
                    "news_interpreter will pull fresh from RSS.",
                    exc,
                )
                # Stamp the failure on meta before raising so the
                # operator (or a future re-queue heuristic) can see
                # what the failed brief was.
                meta["news"] = None
                meta["news_briefs_halt_reason"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                try:
                    led.save()
                except Exception:  # noqa: BLE001
                    pass
                raise
            log.warning(
                "[OTR_LedgerScriptWriter] news_interpreter FAILED after "
                "all attempts: %s -- news_briefs_required=False; "
                "falling back to raw news_seed for cast + outline "
                "(no key_terms enforcement). Sprint 2.2: this is the "
                "back-compat branch; production should ship True.",
                exc,
            )
            meta["news"] = None
            casting_brief = ""
            script_brief = ""
            key_terms_tuple = ()

        # D.3 Lock the cast.
        #
        # Cast RNG: TRUE per-episode randomization (BUG-LOCAL-269).
        # The cast is NO LONGER pinned by the `seed` widget. A fixed
        # seed reproduced ONE cast forever -- every episode opened with
        # the identical characters (seed 42 always rolled HAYES VANCE /
        # GULLIVER REEVES / JIMBO BLACK, out of a ~5,500-combo name
        # pool). _resolve_cast_rng_seed() now draws a fresh OS-entropy
        # seed each episode so the cast genuinely varies; set the
        # OTR_CAST_SEED env var to force a fixed cast for the C7 audio
        # byte-identity regression. This extends BUG-LOCAL-260's LEMMY
        # decoupling to the cast names + announcer pick: random in
        # production, with an explicit force path for the C7 gate.
        #
        # The legacy `seed` widget has been REMOVED from the node --
        # it drove no per-episode variety once the cast (here), the
        # style picker (BUG-LOCAL-270), and the LEMMY cameo
        # (BUG-LOCAL-260) were each decoupled from it.
        if _refine_forced_cast_seed is not None:
            # Refine loop: all passes share ONE cast seed so the cast (and the
            # cast-keyed determinism) stays stable while the outline is revised.
            cast_seed = int(_refine_forced_cast_seed)
            cast_seed_source = "refine_forced"
        else:
            cast_seed, cast_seed_source = _resolve_cast_rng_seed()
        cast_rng = _random.Random(cast_seed)
        log.info(
            "[OTR_LedgerScriptWriter] cast RNG seed=%d (%s) -- cast "
            "randomized per episode (BUG-LOCAL-269)",
            cast_seed, cast_seed_source,
        )
        # LLM slot: creative -- cast lock generates per-character
        # narrative descriptions (gender + character_description).
        # slot-interleave: technical (news_interpreter) -> creative
        # (here). One transition when the two slot ids differ.
        # S32 B1 paired-contract wiring: pass BOTH generators.
        # B1 routes generation through creative; B3 flips schema
        # validation to technical_fn (fail-fast per D2).
        # S32 B6: helper_context attribution.
        # lemmy_cameo widget -> force_lemmy override (BUG-LOCAL-260).
        # None lets cast_pools.roll_lemmy's OS-entropy ~11% decide;
        # True / False force the cameo into / out of the cast.
        lemmy_force = _LEMMY_CAMEO_FORCE.get(lemmy_cameo)
        with slot_scheduler.helper_context("lock_cast"):
            cast_rows, cast_meta = _OTRCAST.lock_cast(
                creative_fn=creative_generate_fn,
                num_characters=resolved["num_characters"],
                news_seed=resolved["news_seed"],
                casting_brief=casting_brief,
                style=resolved["style"],
                rng=cast_rng,
                cast_seed=cast_seed,
                force_lemmy=lemmy_force,
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
            # Sprint 2 (a): persist the cast RNG seed so OTR_CastLock can REPLAY
            # the deterministic bark voice assignment byte-identically. It drives
            # the whole cast rng and is OS-entropy per episode, so it cannot be
            # reconstructed downstream -- it must travel in the frozen ledger.
            "cast_seed":              int(cast_seed),
            "cast_seed_source":       str(cast_seed_source),
        }
        # VC chunk 3 (2026-06-22): carry the per-character voice-fit slots
        # (gender/timbre/role/age_band/speech_signature/description_digest) into
        # the frozen ledger meta so OTR_CastLock's bank caster can match on
        # timbre/age, not just gender. Free-form meta -> ledger schema unchanged.
        meta["cast_voice_slots"] = cast_meta.get("cast_voice_slots") or {}
        # VC chunk 4 (2026-06-22): carry the HYBRID LLM voice-fit decision
        # (proposed/accepted voice_ref_id + reproducibility keys) into the frozen
        # ledger meta so OTR_CastLock can honour the accepted proposal (and fall
        # closed to the deterministic scorer otherwise). Free-form meta.
        meta["voice_cast_decision"] = cast_meta.get("voice_cast_decision") or {}
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
        # G1 (story-quality v2, 2026-06-28): stamp the per-beat word budget so the
        # reroll rebuilder (_otr_reroll) + story_quality_scan derive the SAME
        # one-breath cap as the first pass (derive_one_breath_cap). v2-ONLY -- the
        # key stays OFF the ledger for a v2-OFF render (byte-identical) and the
        # scan/reroll then fall back to the legacy 28-word cap.
        meta["words_per_beat_range"] = list(
            episode_budget.words_per_beat_range)

        # KILL 2 (2026-06-24): build ONE StoryContract pre-outline (cast_seed-keyed,
        # selected from script_brief/news_seed) so the SAME radio style steers the
        # macro prompt, the climax shape, and the body. OFF => contract stays None
        # => no style fields on OutlineRequest, no meta.story_contract => byte-
        # identical. build_story_contract never raises on a missing style, but the
        # call is wrapped LOUD per CLAUDE.md so a defect can never break the writer.
        contract = None
        if _style_grammar_on:
            try:
                contract = _OTRSTYLE.build_story_contract(
                    cast_seed,
                    script_brief,
                    str(resolved.get("news_seed", "") or ""),
                    meta,
                )
                meta.setdefault("story_quality", {})
                meta["story_contract"] = {
                    "slug": contract.slug,
                    "label": contract.label,
                    "ending_tag": contract.ending_tag,
                    # 2026-06-25: carry the selected style's sound_world into the
                    # ledger meta. It was DROPPED here before, so the episode
                    # canon's sound_palette (derived from it) was always empty
                    # even though the style had a rich audio world.
                    "sound_world": contract.sound_world,
                }
            except Exception as _contract_exc:  # noqa: BLE001 -- never break the writer
                log.warning(
                    "[OTR_LedgerScriptWriter] story-contract build skipped (%s); "
                    "style falls back to the premise-keyed draw", _contract_exc,
                )
                contract = None

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
            prior_macro=_refine_prior_macro,
            prior_critique=_refine_prior_critique,
            style_grammar=(contract.grammar if contract else ""),
            story_engine=(contract.story_engine if contract else ""),
        )
        # T1 PITCH ROOM (2026-06-23) -- THE primary story-architecture lever.
        # Default OFF (OTR_ENABLE_PITCH_ROOM). When ON (and NOT a refine sub-pass
        # -- the refine loop owns premise stability via prior_macro/prior_critique),
        # generate 3 forcibly-divergent premises, taste-select one, and REPLACE
        # script_brief with the winner's distilled brief before the outline is
        # generated. OFF => no pitch call, no meta.story_quality.pitch key =>
        # byte-identical. run_pitch_room never raises (internal fallback), but the
        # gate import + call are wrapped defensively per CLAUDE.md.
        if not _refine_active and _OTRPR.pitch_room_enabled():
            try:
                outline_req, _pitch_meta = _OTRPR.run_pitch_room(
                    outline_req,
                    generate_fn=creative_generate_fn,
                    local_model=resolved["creative_writing_model"],
                    frontier_cfg=None,
                    seed_context=cast_seed,
                    meta=meta,
                )
            except Exception as _pitch_exc:  # noqa: BLE001 -- never break the writer
                log.warning(
                    "[OTR_LedgerScriptWriter] pitch room skipped (%s); using the "
                    "original brief", _pitch_exc,
                )
        # LLM slot: creative -- outline drives the episode narrative
        # arc (beats, characters, structure). Single creative pass.
        # Sprint D D2b: thread creative_repo_id so the outline phase
        # prompt routes via _otr_creative_prompt_router. At default
        # config (Mistral-Nemo) the resolver returns _SYSTEM_PROMPT
        # by object identity so audio C7 holds.
        # Sprint 0 (v4 plan): helper_context attribution.
        #
        # Best-of-N structural story-refine selector (2026-06-23). Default OFF:
        # OTR_STORY_BEST_OF_N unset/0/1 => one generate_outline call,
        # byte-identical to the pre-selector pipeline (no selector entry, no
        # meta.story_quality.best_of_n key). When >= 2 AND the creative writer
        # is local, generate N candidate outlines under cast_seed-keyed seeds +
        # structural diversity_hints, score each with the PURE scorer (raw
        # intents, no build_sq_data), and keep the best. Remote writers clamp to
        # N=1 unless the operator opts in (OTR_STORY_BEST_OF_N_ALLOW_REMOTE).
        # build_sq_data still runs exactly ONCE downstream (F2 block) on the
        # winning outline -- the selector never calls it.
        if _refine_active:
            # Refine loop owns outline variety via prior_macro/prior_critique;
            # bypass best-of-N entirely (exactly one outline per refine pass).
            _bon_requested_n, _bon_effective_n, _bon_clamp_reason = (
                1, 1, "refine_bypass",
            )
        else:
            _bon_requested_n, _bon_effective_n, _bon_clamp_reason = (
                _OTRSEL.resolve_best_of_n(resolved)
            )
        if _bon_effective_n >= 2:
            _bon_model = str(resolved["creative_writing_model"])
            _bon_is_remote = _bon_model.startswith(("openrouter:", "comfy:"))

            def _gen_outline(_req):
                return _OTRO.generate_outline(
                    creative_generate_fn,
                    _req,
                    creative_repo_id=resolved["creative_writing_model"],
                )

            def _bon_cost_probe():
                # Cumulative remote spend so far (USD) from the OpenRouter
                # backend's per-run accounting; the selector records the
                # per-candidate delta. Best-effort; never raises.
                try:
                    from . import _otr_openrouter_backend as _OROB
                    snap = _OROB.resolved_models_snapshot()
                    return float(sum(
                        float((v or {}).get("cost_usd", 0.0) or 0.0)
                        for v in snap.values()
                    ))
                except Exception:  # noqa: BLE001
                    return 0.0

            with slot_scheduler.helper_context("generate_outline"):
                outline = _OTRSEL.select_best_outline(
                    _gen_outline,
                    outline_req,
                    cast_seed=cast_seed,
                    n=_bon_effective_n,
                    meta=meta,
                    roster=outline_req.character_cast,
                    cost_probe=_bon_cost_probe if _bon_is_remote else None,
                    # T4 (2026-06-23): deterministic on-mic-climax staging
                    # penalty. Env-gated (OTR_ENABLE_STAGING_PENALTY); default
                    # OFF => None => byte-identical selection.
                    penalty=_OTRSEL.resolve_staging_penalty(),
                )
            # Stamp the telemetry the selector cannot know: requested_n +
            # clamp_reason (gate) + provider (resolved model). Merge, never
            # replace.
            _bon_sq = meta.setdefault("story_quality", {})
            if isinstance(_bon_sq, dict) and isinstance(
                _bon_sq.get("best_of_n"), dict
            ):
                _bon_sq["best_of_n"]["requested_n"] = _bon_requested_n
                _bon_sq["best_of_n"]["clamp_reason"] = _bon_clamp_reason
                _bon_sq["best_of_n"]["provider"] = _bon_model
            log.info(
                "[OTR_LedgerScriptWriter] best-of-N ON: requested=%d "
                "effective=%d winner_index=%s clamp=%r provider=%r",
                _bon_requested_n, _bon_effective_n,
                (_bon_sq.get("best_of_n", {}) or {}).get("winner_index")
                if isinstance(_bon_sq, dict) else None,
                _bon_clamp_reason, _bon_model,
            )
        else:
            # Disabled / clamped-to-1: the existing single path runs EXACTLY
            # once. THIS is the byte-identical path -- no selector entry, no
            # meta.story_quality.best_of_n key.
            with slot_scheduler.helper_context("generate_outline"):
                outline = _OTRO.generate_outline(
                    creative_generate_fn,
                    outline_req,
                    creative_repo_id=resolved["creative_writing_model"],
                )

        # --- E. Word-budget integration check (WARN, do not fail) -----
        # Sum VOICED (character) beats only. `resolved["target_words"]`
        # is a voiced-dialogue-only budget; announcer / music_inter
        # beats carry a fixed, non-scaling overhead (announcer beats are
        # hardcoded ~15 words each) and are excluded from the word
        # budget per outline budget rules §6.G. Summing every beat let
        # that fixed overhead trip a false WORD_BUDGET_DRIFT on small
        # targets (ratio 2.00 on the 2026-05-25 30-word smoke run).
        # Mirrors validate_outline_against_budget validator #1.
        voiced_beats = [
            b for b in outline.beats
            if getattr(b, "speaker_role", "") == "character"
        ]
        beat_word_sum = sum(b.target_words for b in voiced_beats)
        ratio = beat_word_sum / max(1, resolved["target_words"])
        if not (WORD_BUDGET_RATIO_LO <= ratio <= WORD_BUDGET_RATIO_HI):
            log.warning(
                "[OTR_LedgerScriptWriter] WORD_BUDGET_DRIFT: outline "
                "voiced beats sum to %d words, target %d (ratio=%.2f); "
                "proceeding anyway",
                beat_word_sum, resolved["target_words"], ratio,
            )

        # KILL 2 / announcer OPEN (2026-06-24): capture the no-spoiler open brief
        # NOW -- after the outline is final but BEFORE build_sq_data (below)
        # mutates beat.intent in place and KILL 4 enriches the setup beat. The open
        # is then composed by INPUT STARVATION from these setup-framed fields only
        # (never script_brief). OFF => stays None => the original intro path runs.
        safe_open_brief = None
        if _style_grammar_on:
            _open_status_quo = ""
            for _b in outline.beats:
                if str(getattr(_b, "speaker_role", "")) == "character":
                    _open_status_quo = _OTRLC.clean_one_line(
                        str(getattr(_b, "intent", "") or ""), 200,
                    )
                    break
            safe_open_brief = _OTRLC.SafeOpenBrief(
                setting=str(getattr(outline, "setting", "") or ""),
                time_of_day=str(getattr(outline, "time_of_day", "") or ""),
                opening_status_quo=_open_status_quo,
                cast=tuple(character_cast),
                era=str(meta.get("period", "") or ""),
            )

        # --- F2. Story-Quality LIFT L1/L2 (2026-06-23) -- deterministic,
        # UPSTREAM beat-plan shaping. OFF by default (env OTR_STORY_QUALITY_L12).
        # When ON: build the writer-side sq dict[beat_id -> {beat_role,
        # conflict_object, conflict_type, ...}] and ground the GENERIC crisis
        # nouns in each beat.intent (intent ONLY) so the weak local writer cannot
        # collapse every premise into the same "console standoff". The composer
        # threads beat_role/conflict_object/conflict_type into LineRequest. Flag
        # OFF => empty dict, no intent mutation, no field populated => the prompt
        # is byte-identical to the pre-LIFT pipeline. NEVER raises into the
        # writer (the LIFT must never break audio).
        _sq_by_beat: dict = {}
        # KILL 1 (2026-06-24 assumption-audit) -- the grounded premise noun
        # palette for the in-loop BODY-OUTPUT gate. Populated below ONLY when the
        # grounding build runs (lever on); stays empty when off, so the gate is
        # skipped and the render is byte-identical.
        _grounded_nouns: frozenset = frozenset()
        # Story-grammar build (2026-06-24, C5) -- the per-beat ending injection.
        # _ending_template is the climax beat's on-mic ending instruction;
        # _climax_beat_id is the beat that receives it (section I). Both stay ""
        # whenever the style-grammar lever is off => no LineRequest carries an
        # ending_template => byte-identical.
        _ending_template: str = ""
        _climax_beat_id: str = ""
        # The style-grammar lever (climax SHAPE selection) is BUNDLED with L1/L2:
        # when on, it deterministically picks a radio-drama style per episode and
        # feeds that style's ending-taxonomy class as the climax ROLE, runs the
        # L12 build path so the role flows through build_sq_data, and injects the
        # matching final-beat ending template at the climax beat. OFF => no style,
        # climax stays irreversible_choice (the build_sq_data default).
        _climax_role = _OTRSQL12.BEAT_ROLE_IRREVERSIBLE_CHOICE
        _style_slug = ""
        _ending_tag = ""
        if _style_grammar_on:
            try:
                if contract is not None:
                    # KILL 2 (2026-06-24): the climax SHAPE comes from the ONE
                    # pre-outline StoryContract (selected from script_brief/
                    # news_seed BEFORE generate_outline), not a second premise-keyed
                    # select_style draw here -- one style source per episode.
                    _style_slug = contract.slug
                    _ending_tag = contract.ending_tag
                    _ending_template = contract.ending_template
                else:
                    # Defensive: flag on but the contract build raised (already
                    # LOUD-logged) -> fall back to the original premise-keyed draw
                    # so the climax shape is still chosen.
                    _premise_str = str(getattr(outline, "premise", "") or "")
                    _style_slug = _OTRSTYLE.select_style(_premise_str, meta, cast_seed)
                    _ending_tag = str(
                        (_OTRSTYLE.get_style(_style_slug) or {}).get("ending_tag", "")
                    )
                    _ending_template = _OTRSTYLE.ending_template_for(_style_slug)
                if _ending_tag in _OTRSQL12.CLIMAX_CLASS_ROLES:
                    _climax_role = _ending_tag
            except Exception as exc:  # noqa: BLE001 -- grammar must never break audio
                log.warning(
                    "[OTR_LedgerScriptWriter] style-grammar select skipped (%s); "
                    "climax stays irreversible_choice", exc,
                )
                _style_grammar_on = False
                _ending_template = ""
        try:
            if _OTRCFG.story_quality_l12_enabled() or _style_grammar_on:
                _l12_roster = [
                    str(r.get("name") or "")
                    for r in (led.data.get("cast") or [])
                    if isinstance(r, dict) and r.get("name")
                ]
                _sq_by_beat = _OTRSQL12.build_sq_data(
                    list(outline.beats),
                    meta,
                    str(getattr(outline, "premise", "") or ""),
                    cast_seed,
                    roster=_l12_roster,
                    climax_role=_climax_role,
                )
                meta["story_quality_l12_enabled"] = True
                # KILL 1 (2026-06-24): the grounded premise palette the in-loop
                # BODY-OUTPUT gate validates the SHIPPED line against -- roster
                # names + news_seed + premise + title/logline nouns. Stamped on
                # meta so a freeze-cascade reroll rebuild (build_reroll_line_
                # request) composes against the SAME grounding.
                _grounded_nouns = _OTRSQL12.premise_noun_palette(
                    _l12_roster,
                    str(resolved.get("news_seed", "") or ""),
                    str(getattr(outline, "premise", "") or ""),
                    *_OTRSQL12.premise_texts(meta),
                )
                meta["grounded_nouns"] = sorted(_grounded_nouns)
                # The climax-class beat (exactly one, the last voiced character
                # beat) is the one that receives the ending template.
                if _style_grammar_on:
                    for _bid, _ent in _sq_by_beat.items():
                        if _ent.get("beat_role") in _OTRSQL12.CLIMAX_CLASS_ROLES:
                            _climax_beat_id = str(_bid)
                            break
                    if not _climax_beat_id:
                        # No climax beat resolved (e.g. zero character beats) ->
                        # nothing to inject; keep the render byte-clean.
                        _ending_template = ""
                # Telemetry (MERGE -- the scrub's L5a setdefault/update keeps
                # these): the per-episode distinct conflict slots are the
                # cross-episode SAMENESS measure (compare distinct object/type
                # counts across a soak). ungrounded_crisis (shipped-text density)
                # is stamped later by the scrub over the final spoken lines.
                _l12_objs = sorted({
                    str(v.get("conflict_object", ""))
                    for v in _sq_by_beat.values() if v.get("conflict_object")
                })
                _l12_types = sorted({
                    str(v.get("conflict_type", ""))
                    for v in _sq_by_beat.values() if v.get("conflict_type")
                })
                _l12_sq = meta.setdefault("story_quality", {})
                if isinstance(_l12_sq, dict):
                    _l12_sq.update({
                        "l12_domain": _OTRSQL12.select_domain(
                            meta, str(getattr(outline, "premise", "") or "")
                        ),
                        "conflict_objects": _l12_objs,
                        "conflict_types": _l12_types,
                    })
                # Story-grammar telemetry: which style + climax class was chosen,
                # and the crisis-noun count on the (grounded) final beat -- the
                # soak target is ~0 (the climax is no longer a console standoff).
                if _style_grammar_on and isinstance(_l12_sq, dict):
                    _final_crisis = -1
                    try:
                        _grounded = _OTRSQL12.premise_noun_palette(
                            _l12_roster,
                            str(getattr(outline, "premise", "") or ""),
                            *_OTRSQL12.premise_texts(meta),
                        )
                        for _b in outline.beats:
                            if str(getattr(_b, "beat_id", "")) == _climax_beat_id:
                                _final_crisis = _OTRSQL12.count_ungrounded_crisis(
                                    str(getattr(_b, "intent", "") or ""), _grounded,
                                )
                                break
                    except Exception:  # noqa: BLE001 -- telemetry never breaks audio
                        _final_crisis = -1
                    _l12_sq.update({
                        "style_slug": _style_slug,
                        "ending_tag": _ending_tag,
                        "final_beat_crisis_nouns": _final_crisis,
                    })
                    meta["story_quality_grammar_enabled"] = True
                    log.info(
                        "[OTR_LedgerScriptWriter] story-grammar ON: style=%s "
                        "ending_tag=%s climax_beat=%s final_beat_crisis=%d",
                        _style_slug, _ending_tag, _climax_beat_id, _final_crisis,
                    )
                log.info(
                    "[OTR_LedgerScriptWriter] story-quality L1/L2 ON: shaped "
                    "%d voiced beat(s) (domain=%s, %d distinct conflict objects)",
                    len(_sq_by_beat), _l12_sq.get("l12_domain")
                    if isinstance(_l12_sq, dict) else "?", len(_l12_objs),
                )
        except Exception as exc:  # noqa: BLE001 -- the LIFT must never break audio
            log.warning(
                "[OTR_LedgerScriptWriter] story-quality L1/L2 skipped (%s); "
                "proceeding with the unshaped outline", exc,
            )
            _sq_by_beat = {}
            _grounded_nouns = frozenset()
            _ending_template = ""
            _climax_beat_id = ""

        # --- G. Build episode_canon (write deferred to section J.5) ----
        # Disk write moved out so the post-composition title regen
        # (section J.5) can overwrite canon.title before episode_canon.json
        # ever touches disk. Header rendering still happens here because
        # the per-line composer (section I) needs canon_header on every
        # beat.
        #
        # Sprint 3E (2026-05-25) -- LATE TITLE BINDING. The per-line
        # composer in section I is given a canon_header whose title
        # field is the literal `EPISODE_TITLE: TBD`, NOT a provisional
        # outline / widget title. Reason: any title placed in the
        # header can be baked verbatim into spoken dialogue by a beat
        # whose intent is "open the show by naming the episode". The
        # real title is not chosen until J.5 (after the script
        # exists), so before that point there is no correct title to
        # show -- `TBD` guarantees no provisional title is ever
        # spoken. The old fix for this was a fragile post-hoc verbatim
        # substitution in J.6; with `TBD` in the header there is
        # nothing to substitute and J.6 is gone entirely.
        #
        # `canon` keeps the real title intent (resolved widget title
        # else outline.title) so the J.5 disk write has a sane
        # last-resort value if title regen fails; only the COMPOSITION
        # header is forced to TBD.
        # 2026-06-25: populate the episode canon's sound_palette from the
        # selected style's sound_world (the StoryContract carries it). Without
        # this the written episode_canon.json always had sound_palette=[] -- the
        # style's audio world was selected but never reached the canon/ledger.
        # This feeds the WRITTEN canon + meta only; the per-line composition
        # header (_tbd_canon below) deliberately stays sound_world-free, because
        # sound effects in a line prompt invite stage-direction leak (the
        # _otr_outline design keeps sound_world at the macro prompt only).
        _canon_sound_palette: list = []
        if contract is not None and getattr(contract, "sound_world", ""):
            _canon_sound_palette = [
                part.strip() for part in str(contract.sound_world).split(",")
                if part.strip()
            ]
        canon = _OTRC.episode_canon_from_outline_dict({
            "title":       resolved["episode_title"] or outline.title,
            "premise":     outline.premise,
            "setting":     outline.setting,
            "time_of_day": outline.time_of_day,
            "sound_palette": _canon_sound_palette,
        })
        # Build the composition header from a TBD-titled canon so the
        # composer never sees a real or provisional title. The canon
        # module renders the title field as `TITLE: <value>`; swap
        # that one line to the explicit `EPISODE_TITLE: TBD` literal
        # the Sprint 3E plan specifies (and which downstream prompt
        # readers can scan for unambiguously).
        _tbd_canon = _OTRC.episode_canon_from_outline_dict({
            "title":       "TBD",
            "premise":     outline.premise,
            "setting":     outline.setting,
            "time_of_day": outline.time_of_day,
        })
        canon_header = _OTRC.render_episode_canon_header(_tbd_canon)
        canon_header = canon_header.replace(
            "TITLE: TBD", "EPISODE_TITLE: TBD", 1,
        )
        # C1 + C2 (story-quality R2): derive specificity anchors + the central
        # story-object DETERMINISTICALLY from the curated news key_terms (no LLM
        # call), excluding cast names. Inject the anchors into the per-line
        # canon_header (injection-only, no gate); central_object is consumed at
        # the announcer-close. Idempotent: derive only when absent, inject once
        # via a meta flag (so a resume/retry never duplicates the block).
        try:
            from . import _otr_specificity as _OTRSPEC
        except ImportError:  # pragma: no cover
            import _otr_specificity as _OTRSPEC  # type: ignore
        _spec_kts = (meta.get("news") or {}).get("key_terms") or ()
        if "specificity_anchors" not in meta:
            meta["specificity_anchors"] = _OTRSPEC.derive_specificity_anchors(
                _spec_kts, character_cast)
        if "central_object" not in meta:
            meta["central_object"] = _OTRSPEC.derive_central_object(
                _spec_kts, character_cast)
        if (not meta.get("_specificity_anchors_injected")
                and meta.get("specificity_anchors")):
            canon_header = _OTRSPEC.inject_anchors_into_header(
                canon_header, meta["specificity_anchors"])
            meta["_specificity_anchors_injected"] = True
            log.info(
                "[OTR_LedgerScriptWriter] C1: injected %d specificity anchor(s)"
                " into canon_header; C2 central_object=%r",
                len(meta["specificity_anchors"]),
                (meta.get("central_object") or "")[:40],
            )
        log.info(
            "[OTR_LedgerScriptWriter] episode_canon built; composition "
            "header carries EPISODE_TITLE: TBD (late title binding, "
            "Sprint 3E); disk write deferred to post-composition title "
            "regen"
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

        # --- H.1. B1 (story-quality Phase 1, 2026-06-19): NEWS-DERIVED -----
        # THE SPINE. Cast + ledger lines exist now; the Sprint 1 keystone has
        # already stamped dialogue_slot_id on voiced lines. The opposed wants
        # are now DERIVED FROM meta["news"] at this call site (which has meta
        # + the resident technical generate_fn), replacing the hardcoded
        # _DEFAULT_A/B_WANTS boilerplate that ignored the news (the leg_0013
        # ancient-DNA -> aliens drift). A structured LLM call on the resident
        # technical slot emits the four DramaticState-compatible strings, a
        # post-validator requires >= 1 news key term across wants/question/
        # ending, and any failure degrades to a deterministic news-templated
        # fallback. The helper also guarantees >= 1 entry in
        # meta["news"]["key_terms"] (the turning-slot detail floor for
        # validate_contract). NEVER breaks audio (Prime Directive 1).
        try:
            from ._otr_dramatic_state_llm import (
                derive_news_dramatic_state as _derive_news_ds,
                pick_arc_shape as _pick_arc_shape,
            )
            # F8 (story-engine v1): seeded arc-shape pick (variety). The seed
            # combines the reproducibility style seed (so a pinned smoke is
            # deterministic) with the news source hash (so different stories
            # get different shapes -> the smoke distribution is not single-
            # valued). Stamped on meta["arc_shape"] (additive) and passed into
            # the dramatic-state derivation to steer prompt/validator/fallback.
            try:
                _arc_style_seed = os.environ.get("OTR_STYLE_SEED", "").strip()
                _arc_news_hash = str(
                    (meta.get("news") or {}).get("source_hash") or ""
                )
                _arc_seed = (
                    _arc_style_seed + "|" + _arc_news_hash
                    + "|" + str((meta.get("news") or {}).get("script_brief") or "")[:64]
                )
                _arc_shape = _pick_arc_shape(_arc_seed)
            except Exception:  # noqa: BLE001 -- never break audio
                _arc_shape = ""
            if _arc_shape:
                meta["arc_shape"] = _arc_shape
            # F2 (story-engine v1): the costly choice must land on a
            # CHARACTER beat, never the announcer/music. Build the costly-slot
            # candidate list from CHARACTER voiced beats only so
            # pick_costly_choice_slot can never point costly_choice_beat at an
            # announcer slot (the root of the must_turn audit failures). Fall
            # back to all voiced ids only if no character roles are stamped
            # yet (the contract-build guard below is the authoritative one).
            _all_voice_slot_ids: list[str] = [
                str(ln.get("dialogue_slot_id") or "").strip()
                for ln in (led.data.get("lines") or [])
                if str(ln.get("dialogue_slot_id") or "").strip()
            ]
            _char_voice_slot_ids: list[str] = [
                str(ln.get("dialogue_slot_id") or "").strip()
                for ln in (led.data.get("lines") or [])
                if str(ln.get("dialogue_slot_id") or "").strip()
                and str(ln.get("speaker_role") or "").strip().lower() == "character"
            ]
            _voice_slot_ids: list[str] = _char_voice_slot_ids or _all_voice_slot_ids
            if slot_scheduler is not None:
                with slot_scheduler.helper_context("dramatic_state"):
                    _dramatic_state = _derive_news_ds(
                        meta=meta,
                        cast_rows=led.data.get("cast") or cast_rows or [],
                        voice_slot_ids=_voice_slot_ids,
                        slot_fn=technical_generate_fn,
                        arc_shape=_arc_shape,
                    )
            else:
                _dramatic_state = _derive_news_ds(
                    meta=meta,
                    cast_rows=led.data.get("cast") or cast_rows or [],
                    voice_slot_ids=_voice_slot_ids,
                    slot_fn=technical_generate_fn,
                    arc_shape=_arc_shape,
                )
            meta["dramatic_state"] = _dramatic_state.model_dump()
            led.save()
            log.info(
                "[OTR_LedgerScriptWriter] B1: news-derived dramatic_state "
                "stamped (source=%s, costly_choice_beat=%s, voice_slots=%d).",
                meta.get("dramatic_state_source", "?"),
                _dramatic_state.costly_choice_beat,
                len(_voice_slot_ids),
            )
        except Exception as _exc:  # noqa: BLE001 -- never break audio
            log.warning(
                "[OTR_LedgerScriptWriter] Sprint 2.1: dramatic_state "
                "derivation failed (%s: %s); meta['dramatic_state'] "
                "left absent. Sprint 4 selector + Sprint 5 constraint "
                "checker fall back to the no-DramaticState branch.",
                type(_exc).__name__, str(_exc)[:200],
            )

        # --- H.2. Story-quality Phase 1 (A1, 2026-06-19): NEUTRALIZED ------
        # The Sprint 5.1 constraint-editor diagnostic that stamped
        # meta["editor_constraints"] via check_constraints_from_ledger has
        # been removed. It was a write-only diagnostic with ZERO consumers
        # anywhere in the pipeline (a verify-at-build consumer grep confirmed
        # no meta.get("editor_constraints")/string-key/getattr reader exists)
        # -- pure scoring that never drove a decision. Dropping the call site
        # removes the per-episode pass + a redundant led.save() with no
        # behavioral change. The _otr_editor_constraints module + its unit
        # tests stay in place (physical deletion deferred to a post-spine
        # cleanup, per the Phase-1 plan) but are no longer wired in.

        # --- H.5. Sprint 5A: continuity ledger -------------------------
        # One structured LLM call that reads the finished outline + the
        # locked cast and extracts the episode's ContinuityState -- the
        # narrative facts, each tagged with who knows it and who must not
        # reference it yet, plus the beat index where it becomes true.
        # The per-beat loop below renders a per-speaker continuity slice
        # from this state into every LineRequest, so a character cannot
        # reference a fact they should not yet know. The builder NEVER
        # raises -- on any LLM/schema failure it degrades to a neutral
        # state and the slice renders empty (Prime Directive 1).
        #
        # LLM slot: technical -- structured fact extraction from the
        # outline (JSON object validated against a pydantic schema), not
        # creative prose. The model id arrives via the technical slot
        # callable; no new widget, no model_id parameter (Prime
        # Directive 6). OTR_LedgerScriptWriter.py is exempt from the CI
        # `# LLM slot:` sweep, so this tag is verified by eye.
        with slot_scheduler.helper_context("build_continuity_ledger"):
            continuity_state = _OTRCONT.build_continuity_ledger(
                technical_generate_fn,
                outline,
                cast_rows,
                technical_repo_id=resolved["technical_model"],
            )
        meta["continuity"] = continuity_state.model_dump()
        led.save()
        log.info(
            "[OTR_LedgerScriptWriter] Sprint 5A continuity ledger: "
            "%d fact(s), location=%r, %d active prop(s)",
            len(continuity_state.facts), continuity_state.location,
            len(continuity_state.active_props),
        )
        # render_continuity_slice keys facts to the 0-based beat
        # position in outline.beats -- the same coordinate
        # build_continuity_ledger used for `established_beat`. Build the
        # id -> index map once for the per-beat closure below.
        beat_index_by_id = {
            b.beat_id: i for i, b in enumerate(outline.beats)
        }

        # --- H.6. Build 3 (2026-05-28): per-slot drama contracts -------
        # GO_FORWARD_PLAN_v10 Build 3. For each voiced slot, derive the
        # six deterministic contract fields (speaker / concrete details /
        # state_before / state_after / must_turn) from DramaticState +
        # continuity active_props + news key_terms, attach the two
        # free-text fields (line_job, hidden_pressure), validate per-slot
        # + episode-level (exactly one must_turn), and stamp on
        # meta["slot_drama_contracts"] keyed by slot id. Build 4
        # (compose_exchange) is the sole consumer; nothing in the render
        # path reads it yet, so this build only produces + validates +
        # stamps.
        #
        # The technical-slot LLM writes ONLY line_job + hidden_pressure
        # (SlotJobFields, constrained decode); the other six fields are
        # derived deterministically. The generator is built from the
        # resident technical cache_entry below. A failed/invalid LLM pass
        # regenerates once then falls back to a deterministic minimal
        # contract (build_slot_drama_contract), so no garbage contract
        # reaches the writer (Build 3 gate). The whole block is defensive:
        # any failure leaves the contracts absent and the render path
        # untouched (never break audio). Operator: confirm VRAM <= 14.5 GB
        # and the source distribution (llm/llm_regenerate/minimal) in the
        # slot_drama_contracts_audit log on the next live N=3 run.
        # LLM slot: technical -- structured SlotJobFields constrained
        # decode (rule 6); id from resolved["technical_model"], no
        # model_id widget.
        try:
            from ._otr_slot_drama_contract import (
                build_slot_drama_contract as _build_sdc,
                validate_episode_contracts as _validate_sdc_episode,
                SlotJobFields as _SlotJobFields,
            )

            # Build the technical-slot SlotJobFields generator from the
            # resident technical cache_entry. request_slot reuses the
            # entry build_continuity_ledger just used -- no reload, no new
            # VRAM beyond resident (Prime Directives 1 + 2).
            #
            # BUG-LOCAL-294 (caught live 2026-05-28): run() binds _OTRML /
            # _OTRCG as function-LOCALS inside the gated shadow-pass block,
            # so referencing those bare names here raises UnboundLocalError
            # whenever the shadow pass is OFF. Import the modules under
            # fresh local aliases right before use to sidestep the scope.
            from . import _otr_model_loader as _OTRML_SDC
            from . import _otr_constrained_generate as _OTRCG_SDC
            _sdc_cache = _OTRML_SDC.request_slot(
                "technical", resolved["technical_model"],
            )
            _sdc_gen_fn = _OTRCG_SDC.make_constrained_generate_fn(
                _sdc_cache, _SlotJobFields, heartbeat_label="SlotContract",
            )
            _sdc_active_props = list(
                getattr(continuity_state, "active_props", []) or []
            )
            _sdc_key_terms = list(
                (meta.get("news") or {}).get("key_terms") or []
            )
            _sdc_dramatic = meta.get("dramatic_state") or {}
            _sdc_voiced_beats = [
                b for b in outline.beats
                if str(getattr(b, "dialogue_slot_id", "") or "").strip()
            ]
            # F2 (story-engine v1): must_turn may ONLY land on a CHARACTER
            # voiced beat. Build the character-slot set from the SAME beat
            # list the audit checks; if the dramatic_state's costly slot is
            # not a character beat (the rare all-announcer / empty-cast case),
            # clear it on a COPY so NO contract is marked must_turn -- the
            # audit then reports the episode invalid (acceptable + rare)
            # rather than pinning the turn on the announcer/music rows.
            _sdc_char_slots = {
                str(getattr(b, "dialogue_slot_id", "") or "").strip()
                for b in _sdc_voiced_beats
                if str(getattr(b, "speaker_role", "") or "").strip().lower()
                == "character"
                and str(getattr(b, "dialogue_slot_id", "") or "").strip()
            }
            if isinstance(_sdc_dramatic, dict):
                _sdc_costly = str(
                    _sdc_dramatic.get("costly_choice_beat") or ""
                ).strip()
                if _sdc_costly not in _sdc_char_slots:
                    _sdc_dramatic = dict(_sdc_dramatic)
                    _sdc_dramatic["costly_choice_beat"] = ""
                    log.info(
                        "[OTR_LedgerScriptWriter] F2: costly_choice_beat %r is "
                        "not a character slot (%d character slots); clearing "
                        "must_turn -- no announcer/music turn.",
                        _sdc_costly, len(_sdc_char_slots),
                    )

            _sdc_objs = []
            _sdc_contracts: dict = {}
            _sdc_sources: dict = {}
            for _sdc_i, _sdc_beat in enumerate(_sdc_voiced_beats):
                _sdc_sid = str(
                    getattr(_sdc_beat, "dialogue_slot_id", "") or ""
                ).strip()
                _sdc_speaker = str(
                    getattr(_sdc_beat, "speaker", "") or ""
                ).strip()
                if not _sdc_speaker and str(
                    getattr(_sdc_beat, "speaker_role", "") or ""
                ).strip().lower() == "announcer":
                    _sdc_speaker = "ANNOUNCER"
                if not _sdc_sid or not _sdc_speaker:
                    # Voiced slot without a usable id/speaker -- skip; a
                    # single missing contract is handled downstream.
                    continue
                _sdc_row = {
                    "dialogue_slot_id": _sdc_sid,
                    "speaker": _sdc_speaker,
                }
                try:
                    _sdc_contract, _sdc_source = _build_sdc(
                        _sdc_gen_fn,
                        slot_row=_sdc_row,
                        slot_index=_sdc_i,
                        dramatic_state=_sdc_dramatic,
                        beat_intent=str(
                            getattr(_sdc_beat, "intent", "") or ""
                        ),
                        active_props=_sdc_active_props,
                        key_terms=_sdc_key_terms,
                    )
                except Exception as _sdc_exc:  # noqa: BLE001
                    log.warning(
                        "[OTR_LedgerScriptWriter] Build 3 contract build "
                        "failed for slot %s (%s); skipping that slot.",
                        _sdc_sid, type(_sdc_exc).__name__,
                    )
                    continue
                _sdc_objs.append(_sdc_contract)
                _sdc_contracts[_sdc_sid] = _sdc_contract.model_dump()
                _sdc_sources[_sdc_source] = (
                    _sdc_sources.get(_sdc_source, 0) + 1
                )

            _sdc_ok, _sdc_reasons = _validate_sdc_episode(
                _sdc_objs, _sdc_active_props, _sdc_key_terms,
            )
            meta["slot_drama_contracts"] = _sdc_contracts
            meta["slot_drama_contracts_audit"] = {
                "count": len(_sdc_contracts),
                "sources": _sdc_sources,
                "episode_valid": bool(_sdc_ok),
                "reasons": list(_sdc_reasons[:20]),
            }
            led.save()
            log.info(
                "[OTR_LedgerScriptWriter] Build 3 slot drama contracts: "
                "%d slot(s), sources=%s, episode_valid=%s%s",
                len(_sdc_contracts), _sdc_sources, _sdc_ok,
                "" if _sdc_ok else (
                    " reasons=" + "; ".join(_sdc_reasons[:5])
                ),
            )
        except Exception as _exc:  # noqa: BLE001 -- never break audio
            log.warning(
                "[OTR_LedgerScriptWriter] Build 3 slot drama contract "
                "pass failed (%s: %s); meta['slot_drama_contracts'] left "
                "absent. Build 4 compose_exchange degrades to no-contract.",
                type(_exc).__name__, str(_exc)[:200],
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
        # leak-floor-v2 rule 4 (2026-06-25, DEFAULT-OFF/dark): split real-person
        # / political-figure source entities OUT of the roster so the EXISTING
        # phantom gate REJECTS them (-> reroll); org/place/mission terms
        # (NASA/CERN/JPL) stay. OFF => banned set empty, no key_term filtered,
        # banned_terms=() is a no-op => the roster is byte-identical. Also builds
        # the transient per-episode EntityPolicy threaded onto every LineRequest.
        _lfv2_on = _OTRCFG.leak_floor_v2_enabled()
        _lfv2_banned: frozenset = frozenset()
        _episode_entity_policy = None
        roster_key_terms = key_terms_tuple
        if _lfv2_on:
            try:
                _lfv2_news_text = str(
                    (meta.get("news") or {}).get("script_brief") or ""
                )
            except Exception:  # noqa: BLE001
                _lfv2_news_text = ""
            _lfv2_banned = _OTRLC.build_banned_source_proper_nouns(
                terms=key_terms_tuple, raw_text=_lfv2_news_text,
            )
            if _lfv2_banned:
                _banned_u = {b.upper() for b in _lfv2_banned}
                roster_key_terms = tuple(
                    t for t in key_terms_tuple
                    if str(t).strip().upper() not in _banned_u
                )
        allowed_roster = _OTRLC.build_allowed_roster(
            cast_rows=cast_rows,
            key_terms=roster_key_terms,
            banned_terms=_lfv2_banned,
        )
        if _lfv2_on:
            _episode_entity_policy = _OTRHY.EntityPolicy(
                allowed=allowed_roster,
                banned=frozenset(b.upper() for b in _lfv2_banned),
            )
            meta["leak_floor_v2"] = {
                "active": True,
                "banned": sorted(_lfv2_banned),
                "filtered_key_terms": sorted(
                    {str(t) for t in key_terms_tuple}
                    - {str(t) for t in roster_key_terms}
                ),
            }
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
        # F4 (story-engine v1): name -> gender index from the SAME cast rows,
        # so the line composer can pin the speaker's pronouns (no schema
        # change -- cast[].gender already exists).
        gender_by_name: dict[str, str] = {
            row.get("name", ""): str(row.get("gender", "") or "").strip()
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
        # leak-floor-v2 (2026-06-25): render the FILTERED key_terms in the prompt
        # NAMED ENTITIES bucket so a banned real-person term is not re-injected
        # (roster_key_terms == key_terms_tuple when the flag is off => identical).
        allowed_things = frozenset(roster_key_terms)

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
        # markers) from phase_beats. A character beat surrounded
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

        # --- Sprint 5C (2026-05-25): stamp the composer's static-prefix
        # context onto meta so a LATER node (OTR_LedgerFreezeCascade)
        # can reconstruct a LineRequest for a targeted reroll. These four
        # values are computed writer-locally in section I and were lost
        # once the per-beat loop ended; the freeze cascade runs in its
        # own node invocation and reads the ledger from disk, so it needs
        # them on meta. `allowed_roster` is already stamped above; the
        # per-line ledger rows already carry beat_intent / mood (traits)
        # / target_words / arc_phase (Sprint 3C enrichment). Together
        # that is everything `_otr_reroll.build_reroll_line_request`
        # needs. Stamped here, after all four exist; persisted by the
        # ledger saves that follow in the per-beat loop and section J.
        meta["canon_header"]     = canon_header
        meta["outline_spine"]    = outline_spine
        meta["theme"]            = theme
        meta["style_descriptor"] = style_descriptor

        # Announcer dedicated-pass bookend ids (2026-05-22,
        # BUG-LOCAL-255). `_otr_outline._synthesize_outline` always
        # stamps the FIRST and LAST beats as announcer; those two get
        # purpose-built creative passes -- compose_announcer_intro
        # in-loop on the first, compose_announcer_outro post-loop on
        # the last. Any other announcer beat (none today; act-breaks
        # insert music_inter, not announcer) keeps the shared
        # compose_line path. first==last and the empty list are both
        # guarded at the use sites.
        _announcer_ids = [
            b.beat_id for b in outline.beats
            if b.speaker_role == "announcer"
        ]
        first_announcer_id = _announcer_ids[0] if _announcer_ids else None
        last_announcer_id = _announcer_ids[-1] if _announcer_ids else None
        # news_close_brief drives the outro pass + its deterministic
        # fallback. Hoisted above the loop so the in-loop placeholder
        # for the final announcer beat can use it too.
        nc_brief = str(
            (meta.get("news") or {}).get("news_close_brief") or ""
        ).strip()
        # C2 (story-quality R2): thread the central story-object into the close
        # brief so the S2 "use the central object if set" close lands on a
        # concrete final image. Unchanged when central_object is "". _OTRSPEC was
        # imported above at the C1 anchor-derivation site (same execute scope).
        nc_brief = _OTRSPEC.inject_central_object_into_brief(
            nc_brief, meta.get("central_object") or "")

        # STEP 6 (2026-06-22 story+cast fix, roundtable-converged): a
        # deterministic escalating beat_tension (1..5) over the CHARACTER beats.
        # arc_phase already escalates; beat_tension was never assigned, so the
        # composer's "Tension: N/5" cue never rendered. Compute the ramp ONCE
        # here (character beats only, in outline order) and look it up per beat
        # in the closure below; also stamp the per-line dramatic frame onto meta
        # so the critic can SEE the target and the reroll can RECONSTRUCT it.
        try:
            from ._otr_slot_drama_contract import (
                compute_beat_tension_ramp as _otr_tension_ramp,
            )
            _otr_char_beat_ids = [
                b.beat_id for b in (getattr(outline, "beats", []) or [])
                if getattr(b, "speaker_role", "") == "character"
            ]
            _otr_tension_by_beat = _otr_tension_ramp(_otr_char_beat_ids)
        except Exception:  # noqa: BLE001 -- never break audio
            _otr_tension_by_beat = {}

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
            # Sprint 3.1 (2026-05-28) -- DRAMATIC FRAME wiring.
            # Threads dramatic_question from meta["dramatic_state"]
            # (Sprint 2.1 stamp) and next_turn from the next voiced
            # outline beat's intent. The other Sprint 3 fields
            # (beat_objective / beat_obstacle / beat_turn /
            # beat_subtext / beat_tension) stay empty for Path A
            # since _otr_outline.Beat does not carry the Sprint 2
            # typed-state fields; they activate when a future sprint
            # lifts Path A's outline schema to mirror Stage1Beat.
            # All fields default empty in LineRequest, so this is
            # additive -- legacy callers with no DramaticState see
            # the pre-Sprint-3 prompt byte-identical.
            _ds_meta = meta.get("dramatic_state") or {}
            _dramatic_question = (
                str(_ds_meta.get("dramatic_question") or "").strip()
                if isinstance(_ds_meta, dict) else ""
            )
            _next_turn_text = ""
            try:
                _voiced_beats = [
                    b for b in (getattr(outline, "beats", []) or [])
                    if getattr(b, "speaker_role", "") in (
                        "character", "announcer",
                    )
                ]
                _voiced_ids = [b.beat_id for b in _voiced_beats]
                _here = _voiced_ids.index(beat.beat_id) if beat.beat_id in _voiced_ids else -1
                if 0 <= _here < len(_voiced_beats) - 1:
                    _next_turn_text = (
                        getattr(_voiced_beats[_here + 1], "intent", "") or ""
                    ).strip()
            except Exception:  # noqa: BLE001 -- never break audio
                _next_turn_text = ""

            # --- A5 (story-quality Phase 1): deliver the news-driven drama
            # to the line writer. Map THIS slot's SlotDramaContract (Build 3,
            # now news-derived via B1) onto the composer's per-line dramatic
            # fields so the line PLAYS the objective / obstacle / turn /
            # subtext instead of restating the theme. Single-line Path A
            # (use_exchange OFF); we do NOT mutate locked cast rows --
            # distinctness rides on these per-line fields. Fail-soft: any miss
            # leaves the fields empty and the composer drops the empty blocks
            # (legacy prompt byte-identical). Announcer slots are skipped (the
            # dramatic frame + opposed-want framing is character-centric).
            _a5_obj = _a5_obs = _a5_turn = _a5_sub = ""
            if not is_announcer:
                try:
                    from ._otr_slot_drama_contract import (
                        build_line_dramatic_fields as _a5_fields,
                    )
                    _a5_sid = str(
                        getattr(beat, "dialogue_slot_id", "") or ""
                    ).strip()
                    _a5_contracts = meta.get("slot_drama_contracts") or {}
                    _a5_contract = (
                        _a5_contracts.get(_a5_sid)
                        if isinstance(_a5_contracts, dict) else None
                    )
                    if _a5_contract:
                        _a5_cast = led.data.get("cast") or cast_rows or []
                        _a5_names = [
                            str(r.get("name") or "").strip()
                            for r in _a5_cast
                            if isinstance(r, dict) and str(r.get("name") or "").strip()
                        ]
                        _a5_map = _a5_fields(
                            _a5_contract, _ds_meta,
                            speaker=speaker,
                            a_name=_a5_names[0] if _a5_names else "",
                            b_name=_a5_names[1] if len(_a5_names) > 1 else "",
                        )
                        _a5_obj = _a5_map.get("beat_objective", "")
                        _a5_obs = _a5_map.get("beat_obstacle", "")
                        _a5_turn = _a5_map.get("beat_turn", "")
                        _a5_sub = _a5_map.get("beat_subtext", "")
                except Exception:  # noqa: BLE001 -- never break audio
                    _a5_obj = _a5_obs = _a5_turn = _a5_sub = ""

            # STEP 6: derive this character beat's tension (0 for announcer --
            # announcer lines are excluded from the curve) and STAMP the per-line
            # dramatic frame onto meta. The frame is the single source the critic
            # reads (target_tension) and the reroll reconstructs (objective /
            # obstacle / turn / subtext / tension / dramatic_question / next_turn)
            # -- build_reroll_line_request otherwise loses all of it. META ONLY:
            # the ledger {cast,lines,meta} wire format stays frozen.
            _a5_tension = (
                int(_otr_tension_by_beat.get(beat.beat_id, 0))
                if not is_announcer else 0
            )
            if not is_announcer:
                try:
                    _otr_frames = meta.setdefault("line_dramatic_frame", {})
                    _otr_frames[str(beat.beat_id)] = {
                        "objective": _a5_obj,
                        "obstacle": _a5_obs,
                        "turn": _a5_turn,
                        "subtext": _a5_sub,
                        "tension": _a5_tension,
                        "dramatic_question": _dramatic_question,
                        "next_turn": _next_turn_text,
                    }
                except Exception:  # noqa: BLE001 -- never break audio
                    pass

            return _OTRLC.LineRequest(
                speaker=speaker,
                intent=beat.intent,
                mood=beat.mood,
                target_words=beat.target_words,
                canon_header=canon_header,
                last_lines=list(last_lines),
                allowed_roster=allowed_roster,
                # leak-floor-v2 (2026-06-25): transient per-episode policy (None
                # when the flag is off => compose_line's verifier never runs).
                entity_policy=_episode_entity_policy,
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
                position=_position_for(beat),
                # Sprint 5A (2026-05-25) -- per-speaker continuity slice
                # rendered from the episode ContinuityState. Empty string
                # when this speaker has no continuity signal at this beat;
                # _build_user_prompt drops the block on an empty value.
                continuity_slice=_OTRCONT.render_continuity_slice(
                    continuity_state,
                    speaker,
                    beat_index_by_id.get(beat.beat_id, 0),
                ),
                # Sprint 3.1 (2026-05-28) -- DRAMATIC FRAME fields.
                dramatic_question=_dramatic_question,
                next_turn=_next_turn_text,
                # A5 (2026-06-19) -- the news-driven slot contract, delivered.
                beat_objective=_a5_obj,
                beat_obstacle=_a5_obs,
                beat_turn=_a5_turn,
                beat_subtext=_a5_sub,
                # STEP 6 (2026-06-22) -- the escalating per-beat intensity cue.
                beat_tension=_a5_tension,
                # F4 (story-engine v1) -- speaker gender/pronouns.
                speaker_gender=gender_by_name.get(speaker, ""),
                # G1 (story-quality v2, 2026-06-28) -- the per-beat word budget so
                # the composer's one-breath gate derives a budget-sized cap on the
                # v2 path. Always threaded (consumed ONLY when v2), so a v2-OFF
                # render is byte-identical (the cap stays the legacy 28).
                words_per_beat_range=tuple(
                    episode_budget.words_per_beat_range
                ),
                # Story-quality LIFT L1/L2 (2026-06-23) -- threaded from the
                # writer-side sq dict. Empty unless OTR_STORY_QUALITY_L12 is on
                # (then build_sq_data populated it) => "" => byte-identical.
                beat_role=str(
                    (_sq_by_beat.get(beat.beat_id) or {}).get("beat_role", "")
                ),
                conflict_object=str(
                    (_sq_by_beat.get(beat.beat_id) or {}).get("conflict_object", "")
                ),
                conflict_type=str(
                    (_sq_by_beat.get(beat.beat_id) or {}).get("conflict_type", "")
                ),
                # Story-grammar build (2026-06-24, C4): the style-selected ending
                # instruction, injected ONLY on the climax-class (final character)
                # beat when OTR_ENABLE_STYLE_GRAMMAR is on. "" on every other beat
                # and whenever the lever is off => byte-identical.
                ending_template=(
                    _ending_template
                    if (_ending_template and beat.beat_id == _climax_beat_id)
                    else ""
                ),
                # KILL 1 (2026-06-24) -- the grounded premise palette, carried so
                # a freeze-cascade reroll rebuild keeps the same grounding the
                # in-loop body gate used. Empty when the lever is off.
                grounded_nouns=_grounded_nouns,
            )

        # Sprint 10B Wave 1 Agent B (2026-05-27): build the in-loop
        # Stage1Plan once before the render loop. Needed only when the
        # production Stage 3 validators are on -- they consume the
        # Stage1Plan + per-beat Stage1Beat to score lines. Off (the
        # default) this is a no-op and the legacy path is byte-
        # identical (PD1). Built from in-loop writer state (outline,
        # cast_rows, meta) via the in-loop adapter migrated out of the
        # deleted multiturn module into _otr_legacy_to_stage1_adapter.
        _w1b_stage3_enabled: bool = bool(resolved.get(
            "enable_production_stage3_validators", False,
        ))
        _w0_stage1_plan = None
        if _w1b_stage3_enabled:
            try:
                _w0_stage1_plan = _OTRL2S1.build_inloop_stage1_plan(
                    outline=outline,
                    cast_rows=cast_rows,
                    meta=meta,
                )
                if _w0_stage1_plan is None:
                    log.warning(
                        "[Stage3Validators] build_inloop_stage1_plan "
                        "returned None; Stage 3 validators disabled "
                        "for this episode"
                    )
                else:
                    log.info(
                        "[Stage3Validators] in-loop Stage1Plan built: "
                        "cast=%d, beats=%d, running_facts=%d",
                        len(_w0_stage1_plan.cast),
                        len(_w0_stage1_plan.beats),
                        len(_w0_stage1_plan.running_facts),
                    )
            except Exception as _w0_exc:  # noqa: BLE001 -- PD1
                log.warning(
                    "[Stage3Validators] build_inloop_stage1_plan "
                    "raised %s: %s -- Stage 3 validators disabled for "
                    "this episode",
                    type(_w0_exc).__name__, str(_w0_exc)[:200],
                )
                _w0_stage1_plan = None

        # --- Build 4 (2026-05-28): grouped-exchange pre-pass -----------
        # When use_exchange is ON, render consecutive voiced beat groups
        # as exchanges BEFORE the per-beat loop; the loop then short-
        # circuits each composed beat to the returned text. OFF (default)
        # leaves _ex_lines_by_beat_id empty so the loop is byte-identical
        # to the legacy path (PD1). The whole block is
        # defensive: any failure leaves the map empty and every beat
        # renders via its existing path (never break audio). Runs inside
        # the compose_line helper context so the creative model is
        # resident (same slot the per-beat composer uses).
        # LLM slot: creative -- compose_exchange renders dialogue
        # (subtext / refusal / reversal) via creative_generate_fn (rule 6).
        _ex_use: bool = bool(resolved.get("use_exchange", False))
        _ex_lines_by_beat_id: dict[str, str] = {}
        if _ex_use:
            try:
                from ._otr_compose_exchange import (
                    run_exchange_prepass as _run_ex_prepass,
                    make_tier_a_adapter as _make_tier_a,
                )
                from ._otr_craft_floor import (
                    evaluate_tier_a as _eval_tier_a,
                    normalize_slot_line as _norm_slot_line,
                )
                _ex_tier_a = _make_tier_a(_eval_tier_a, _norm_slot_line)
                _ex_cast = getattr(outline, "cast", None) or cast_rows or []
                with slot_scheduler.helper_context("compose_line"):
                    _ex_lines_by_beat_id = _run_ex_prepass(
                        list(outline.beats),
                        meta.get("slot_drama_contracts") or {},
                        list(_ex_cast),
                        generate_fn=creative_generate_fn,
                        tier_a_check=_ex_tier_a,
                    )
                meta["exchange_prepass_audit"] = {
                    "beats_composed": len(_ex_lines_by_beat_id),
                    "beat_ids": sorted(_ex_lines_by_beat_id.keys()),
                }
                log.info(
                    "[OTR_LedgerScriptWriter] Build 4 use_exchange: %d "
                    "beat(s) composed via grouped exchange.",
                    len(_ex_lines_by_beat_id),
                )
            except Exception as _ex_exc:  # noqa: BLE001 -- never break audio
                log.warning(
                    "[OTR_LedgerScriptWriter] Build 4 exchange pre-pass "
                    "failed (%s: %s); all beats use the legacy "
                    "path.",
                    type(_ex_exc).__name__, str(_ex_exc)[:200],
                )
                _ex_lines_by_beat_id = {}

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
                # Build 4 (2026-05-28): a beat composed by the grouped
                # exchange pre-pass short-circuits the per-beat legacy
                # composer below. When use_exchange is OFF (default)
                # _ex_text is None and the legacy composer runs
                # unchanged -- PD1 byte-identity preserved.
                _ex_text = (
                    _ex_lines_by_beat_id.get(beat.beat_id)
                    if _ex_use else None
                )
                if _ex_text is not None:
                    cleaned = _ex_text
                    beat_compose_flags = ()
                else:
                    # LLM slot: creative -- dialogue composer per-beat
                    # narrative pass. Polish (creative; routed through
                    # polish_generate_fn from the scheduler) handles
                    # the narration-leak cleanup pass when
                    # enable_polish_pass is on.
                    # S32 B6: helper_context attribution. Per-beat
                    # invocation; the context-manager overhead is
                    # constant-time and negligible relative to the LLM
                    # call itself.
                    #
                    # Sprint 10B Wave 1 Agent B (2026-05-27): pass
                    # Stage 3 validator inputs when the widget is on
                    # AND the in-loop Stage1Plan was built successfully.
                    # The validators fire inside compose_line after the
                    # strip pipeline; one repair regenerate on errors,
                    # findings stamped on line_res.validation_findings.
                    _w1b_s3_kwargs = {}
                    if _w1b_stage3_enabled and _w0_stage1_plan is not None:
                        _w1b_s3_beat = (
                            _OTRL2S1.line_request_to_stage1_beat(
                                beat,
                                fallback_index=beat_index_by_id.get(
                                    beat.beat_id, 0,
                                ),
                            )
                        )
                        _w1b_s3_kwargs = dict(
                            enable_stage3_validators=True,
                            stage3_plan=_w0_stage1_plan,
                            stage3_beat=_w1b_s3_beat,
                        )
                    with slot_scheduler.helper_context("compose_line"):
                        line_res = _OTRLC.compose_line(
                            creative_fn=creative_generate_fn,
                            req=line_req,
                            base_temperature=base_temp,
                            max_new_tokens_cap=resolved["max_new_tokens_cap"],
                            creative_repo_id=resolved["creative_writing_model"],
                            **_w1b_s3_kwargs,
                        )
                    cleaned = line_res.text
                    beat_compose_flags = line_res.compose_flags
                    # Sprint 10B Wave 1 Agent B: stamp validator findings
                    # on the ledger row via patch_line_fields below (the
                    # _OTRL.patch_line_fields call inside this loop only
                    # currently stamps char_id/traits/compose_flags --
                    # extend the patch dict here when findings present).
                    if line_res.validation_findings:
                        meta.setdefault(
                            "stage3_findings_per_beat", {},
                        )[beat.beat_id] = list(
                            line_res.validation_findings,
                        )
                # --- KILL 1 (2026-06-24 assumption-audit): deterministic
                # BODY-OUTPUT gate. Validate the SHIPPED character line (NOT
                # beat.intent) against the grounded premise palette -- it must
                # not lean on ungrounded generic crisis machinery, and a
                # climax-class / pressure beat must REFERENCE its premise-
                # anchored conflict_object. ONE guarded reroll with a split,
                # targeted hint; ship the reroll ONLY if it validates, else keep
                # the original (deterministic). Runs after BOTH the exchange
                # (cleaned=_ex_text) and the per-beat composer (cleaned=
                # line_res.text) paths, so the use_exchange bypass is covered.
                # Active only when the grounding build ran (palette + sq dict
                # populated) => byte-identical when the lever is off.
                if _grounded_nouns and _sq_by_beat:
                    _bg_entry = _sq_by_beat.get(beat.beat_id) or {}
                    _bg_roles = (
                        _OTRSQL12.CLIMAX_CLASS_ROLES
                        | {_OTRSQL12.BEAT_ROLE_PRESSURE}
                    )
                    _bg_ok, _bg_reasons = _OTRSQL12.validate_composed_grounding(
                        cleaned, _bg_entry, _grounded_nouns,
                        max_ungrounded=0,
                        require_conflict_object_on_roles=_bg_roles,
                    )
                    _bg_sq = meta.setdefault("story_quality", {})
                    # C4 (S3, story-quality v2): also REROLL on a MID-CLAUSE
                    # roster-caps shout (a locked cast FULL name in ALL CAPS at a
                    # grammatical subject/object position, where an in-place strip
                    # would mangle the clause), and ACCEPT the reroll by a total-
                    # order defect score on the shipped text (below) rather than
                    # grounding alone.
                    _bg_fullnames = _otr_cast_fullnames(line_req)
                    _bg_roster_mid = bool(
                        _bg_fullnames
                        and _otr_roster_caps_midclause(cleaned, _bg_fullnames)
                    )
                    if (not _bg_ok) or _bg_roster_mid:
                        _bg_hint = _otr_body_gate_hint(_bg_reasons, _bg_entry)
                        if _bg_roster_mid:
                            _bg_hint = (
                                (_bg_hint + " ") if _bg_hint else ""
                            ) + (
                                "Do not write a character's full name in "
                                "capital letters; refer to other characters "
                                "normally."
                            )
                        try:
                            with slot_scheduler.helper_context("compose_line"):
                                _bg_res = _OTRLC.compose_line(
                                    creative_fn=creative_generate_fn,
                                    req=line_req,
                                    base_temperature=base_temp,
                                    max_new_tokens_cap=resolved[
                                        "max_new_tokens_cap"
                                    ],
                                    creative_repo_id=resolved[
                                        "creative_writing_model"
                                    ],
                                    reroll_hint=_bg_hint,
                                )
                            _bg_res_ok, _ = _OTRSQL12.validate_composed_grounding(
                                _bg_res.text, _bg_entry, _grounded_nouns,
                                max_ungrounded=0,
                                require_conflict_object_on_roles=_bg_roles,
                            )
                            # C4 ACCEPT: keep the reroll only when it scores
                            # STRICTLY cleaner on the shipped text (lower wins,
                            # ORIGINAL on tie).
                            _bg_rr_ok = bool(_bg_res.text.strip())
                            _use_rr = _bg_rr_ok and (
                                _otr_body_score(
                                    _bg_res.text, _bg_entry, _grounded_nouns,
                                    _episode_entity_policy, line_req,
                                )
                                < _otr_body_score(
                                    cleaned, _bg_entry, _grounded_nouns,
                                    _episode_entity_policy, line_req,
                                )
                            )
                            if _use_rr:
                                cleaned = _bg_res.text
                                beat_compose_flags = (
                                    tuple(_bg_res.compose_flags)
                                    + ("body_gate_reroll",)
                                )
                                if isinstance(_bg_sq, dict):
                                    _bg_sq["body_gate_rerolls"] = int(
                                        _bg_sq.get("body_gate_rerolls", 0)
                                    ) + 1
                            else:
                                if isinstance(_bg_sq, dict):
                                    _bg_sq["body_gate_failed"] = int(
                                        _bg_sq.get("body_gate_failed", 0)
                                    ) + 1
                                log.info(
                                    "[OTR_LedgerScriptWriter] body-gate reroll "
                                    "did not validate for beat %s; keeping "
                                    "original (%s)",
                                    beat.beat_id, ",".join(_bg_reasons)[:160],
                                )
                        except Exception as _bg_exc:  # noqa: BLE001 -- never break audio
                            if isinstance(_bg_sq, dict):
                                _bg_sq["body_gate_failed"] = int(
                                    _bg_sq.get("body_gate_failed", 0)
                                ) + 1
                                _bg_sq["grounding_reroll_failed"] = True
                            log.warning(
                                "[OTR_LedgerScriptWriter] body-gate reroll raised "
                                "(%s: %s) for beat %s; keeping original",
                                type(_bg_exc).__name__, str(_bg_exc)[:160],
                                beat.beat_id,
                            )
                    # Telemetry: the SHIPPED-text ungrounded-crisis density,
                    # accumulated across body beats (final `cleaned`, post-reroll)
                    # -- the soak target vs the flag-off baseline.
                    if isinstance(_bg_sq, dict):
                        _bg_sq["body_gate_ungrounded_crisis"] = int(
                            _bg_sq.get("body_gate_ungrounded_crisis", 0)
                        ) + len(_OTRSQL12.ungrounded_crisis_tokens(
                            cleaned, _grounded_nouns,
                        ))

                cid = char_id_by_name[beat.speaker]
                token = f"[VOICE: {beat.speaker}, {traits}] {cleaned}"

                last_lines.append((beat.speaker, cleaned))
                if len(last_lines) > LAST_LINES_WINDOW:
                    last_lines.pop(0)

            elif beat.speaker_role == "announcer":
                # Announcer dedicated passes (2026-05-22, BUG-LOCAL-255).
                # The first announcer beat gets compose_announcer_intro
                # in-loop; the last gets no in-loop LLM call -- the
                # post-loop compose_announcer_outro pass overwrites it
                # once the script + the intro text both exist. Any
                # other announcer beat (none in the current outline)
                # falls back to the shared compose_line path.
                cid = "announcer"
                if (
                    first_announcer_id is not None
                    and beat.beat_id == first_announcer_id
                ):
                    # LLM slot: creative -- dedicated announcer intro,
                    # a narrative framing pass. Routed through the
                    # writer's creative_writing_model slot; no widget.
                    with slot_scheduler.helper_context(
                        "compose_announcer_intro"
                    ):
                        line_res = _OTRLC.compose_announcer_intro(
                            creative_fn=creative_generate_fn,
                            script_brief=(
                                "" if _style_grammar_on else script_brief
                            ),
                            creative_repo_id=resolved[
                                "creative_writing_model"
                            ],
                            story_scaffold=_style_grammar_on,
                            safe_open_brief=safe_open_brief,
                        )
                    cleaned = line_res.text
                    beat_compose_flags = line_res.compose_flags
                    # KILL 2 telemetry (under flag only): did the safe-open pass
                    # fall back to the deterministic template?
                    if _style_grammar_on:
                        meta.setdefault("story_quality", {})["open_safe_fallback"] = (
                            "open_safe_fallback" in line_res.compose_flags
                        )
                elif (
                    last_announcer_id is not None
                    and beat.beat_id == last_announcer_id
                    and last_announcer_id != first_announcer_id
                ):
                    # No in-loop LLM call. Drop in the deterministic
                    # outro fallback as the placeholder so a mid-loop
                    # crash still leaves a valid closing bookend; the
                    # post-loop outro pass overwrites this row.
                    cleaned = _OTRLC.fallback_announcer_outro(nc_brief)
                    beat_compose_flags = ()
                else:
                    line_req = _build_line_request_for_beat(
                        beat, is_announcer=True,
                    )
                    # LLM slot: creative -- a mid-episode announcer
                    # beat is a narrative write; keep the shared
                    # composer path. S32 B6: helper_context
                    # attribution; constant-time overhead.
                    with slot_scheduler.helper_context("compose_line"):
                        line_res = _OTRLC.compose_line(
                            creative_fn=creative_generate_fn,
                            req=line_req,
                            base_temperature=base_temp,
                            max_new_tokens_cap=resolved[
                                "max_new_tokens_cap"
                            ],
                            creative_repo_id=resolved[
                                "creative_writing_model"
                            ],
                        )
                    cleaned = line_res.text
                    beat_compose_flags = line_res.compose_flags
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
                # (All NON_VOICED_ROLES are music markers post
                # rip-sfx-broll 2026-07-01.)
                last_lines.clear()
                # S1 (2026-06-22) + rip-sfx-broll (2026-07-01): music rows
                # are pure render contracts -- no transcript text, ever.
                # The old [SFX: ...] token emission died with the sfx_cue
                # field; slot-0 authority is assemble_script_text_from_ledger
                # post-loop, which skips empty-text rows.
                cleaned = ""
                cid = beat.speaker_role
                token = ""

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
            # rip-sfx-broll (2026-07-01): music render-contract rows emit
            # an empty token -- skip it (the post-loop
            # assemble_script_text_from_ledger is slot-0's authority and
            # skips empty-text rows the same way).
            if token:
                script_text_parts.append(token)

        # --- I.5. News-wiring overlay (Phase 2B: operates on ledger) --
        # Two operations on `led.data["lines"]` AFTER the progressive
        # composer loop completes.
        #
        # 1. Announcer closing-line pass. The per-beat loop left a
        #    deterministic placeholder on the final announcer beat.
        #    Now that the full script + the intro line both exist,
        #    compose_announcer_outro writes the purpose-built close
        #    (script_brief + news_close_brief + the intro text) and
        #    overwrites that row. This replaces the retired
        #    `override_announcer_close` verbatim stamp -- that helper
        #    matched a private `_speaker_role` key absent from the
        #    ledger's `lines[]` rows, so the close was silently never
        #    applied (BUG-LOCAL-255). Skipped only on a degenerate
        #    outline where the first and last announcer beat coincide
        #    (the intro pass already filled it).
        #
        # 2. Post-assembly key_terms audit. Walk every voiced line,
        #    check each key_term landed via word-boundary regex.
        #    Stamp the result on meta["post_assembly_key_terms"].
        news_meta = meta.get("news") or {}
        if (
            last_announcer_id is not None
            and last_announcer_id != first_announcer_id
        ):
            # Read the composed intro line back from the ledger so the
            # outro prompt can lightly echo its tone.
            intro_text = ""
            for _ln in led.data.get("lines") or []:
                if _ln.get("line_id") == first_announcer_id:
                    intro_text = str(_ln.get("text") or "")
                    break
            # LLM slot: creative -- dedicated announcer outro, a
            # narrative framing pass. Routed through the writer's
            # creative_writing_model slot; no widget.
            # F3 (story-engine v1): thread the resolved ending + the final
            # character line so the close STATES the outcome instead of
            # hedging. Both null-guarded ("" when unavailable) -> the
            # composer's post-check skips cleanly on an unresolved/missing
            # ending.
            # KILL 2 / NEWS CODA (2026-06-24): a dynamic premise->news segue. The
            # LLM writes ONLY a short bridge clause (from the premise + the safe
            # intro tone, never the outcome); the real news_close_brief is appended
            # deterministically so the weak model can't blend the fact away.
            # compose_announcer_outro is UNTOUCHED -- the off / no-brief path runs
            # it verbatim, so the fictional close stays byte-identical.
            if _style_grammar_on and nc_brief.strip():
                with slot_scheduler.helper_context("compose_news_coda"):
                    outro_res = _OTRLC.compose_news_coda(
                        creative_fn=creative_generate_fn,
                        news_close_brief=nc_brief,
                        premise=str(getattr(outline, "premise", "") or ""),
                        intro_text=intro_text,
                        cast_seed=cast_seed,
                        creative_repo_id=resolved["creative_writing_model"],
                        # S2 (story-quality v2, 2026-06-28): system examples +
                        # arc_shape-keyed curated fallback floor.
                        arc_shape=str(meta.get("arc_shape") or ""),
                    )
                if not outro_res.text:
                    # Pathological (brief cleaned to empty) -- never ship an empty
                    # close. Deterministic news outro, LOUD.
                    log.warning(
                        "[OTR_LedgerScriptWriter] news coda produced no text "
                        "(brief=%r); using the deterministic outro fallback",
                        nc_brief,
                    )
                    outro_res = _OTRLC.LineResult(
                        text=_OTRLC.fallback_announcer_outro(nc_brief),
                        compose_flags=("news_coda_fallback", "news_coda_empty_close"),
                    )
            else:
                # The fictional-outro path (flag off, OR on but no news brief).
                # Build its inputs INSIDE the else -- only this path needs them.
                _outro_ending_change = str(
                    (meta.get("dramatic_state") or {}).get("ending_change") or ""
                )
                _outro_final_char_line = ""
                for _ln in reversed(led.data.get("lines") or []):
                    if str(_ln.get("speaker_role") or "").strip() == "character":
                        _t = str(_ln.get("text") or "").strip()
                        if _t:
                            _outro_final_char_line = _t
                            break
                with slot_scheduler.helper_context("compose_announcer_outro"):
                    outro_res = _OTRLC.compose_announcer_outro(
                        creative_fn=creative_generate_fn,
                        script_brief=script_brief,
                        news_close_brief=nc_brief,
                        intro_text=intro_text,
                        creative_repo_id=resolved["creative_writing_model"],
                        ending_change=_outro_ending_change,
                        final_character_line=_outro_final_char_line,
                    )
                if _style_grammar_on:
                    # On-flag but no news brief -> mark it (text unchanged; frozen).
                    import dataclasses as _dc
                    outro_res = _dc.replace(
                        outro_res,
                        compose_flags=outro_res.compose_flags + ("news_coda_no_brief",),
                    )
            # KILL 2 telemetry (under flag only).
            if _style_grammar_on:
                _sqd = meta.setdefault("story_quality", {})
                _sqd["news_coda_emitted"] = bool(nc_brief.strip())
                _sqd["news_coda_fallback"] = (
                    "news_coda_fallback" in outro_res.compose_flags
                )
            # patch_line_text recomputes char_count + word_count in
            # lockstep; patch_line_fields stamps the outro compose_flags
            # so aggregate_compose_flags + soak see the pass result.
            _OTRL.patch_line_text(
                led.data, last_announcer_id, outro_res.text,
            )
            _OTRL.patch_line_fields(
                led.data, last_announcer_id,
                {"compose_flags": list(outro_res.compose_flags)},
            )
            led.save()
            log.info(
                "[OTR_LedgerScriptWriter] announcer outro pass wrote "
                "closing line %s (flags=%s)",
                last_announcer_id, outro_res.compose_flags,
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

        # --- I.6. Dialogue scrubs (operator look-QA 2026-06-10) ------
        # Deterministic, pre-freeze (audio has not rendered yet, so the
        # text edits are safe), LOUD per fix.
        #
        # (a) STAGE-DIRECTION scrub: the composer sometimes embeds a
        # parenthetical/bracketed action inside the LINE TEXT --
        # '..."Observe." (HAYES VANCE removes a vintage pocket watch
        # from...)' -- which the TTS then SPEAKS and the captions
        # display (look-QA round 4). Radio-drama dialogue carries no
        # parentheticals, so every (...) / [...] span is stripped; a
        # line left with <2 words keeps its original text (warned).
        # Symmetric wrapping double-quotes are unwrapped in the same
        # pass.
        _sd_re = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]")
        _sd_fixed = 0
        for _ln in led.data.get("lines") or []:
            if not isinstance(_ln, dict) or _ln.get("skip"):
                continue
            _txt = str(_ln.get("text") or "")
            _new = _sd_re.sub(" ", _txt)
            _new = re.sub(r"\s{2,}", " ", _new).strip()
            if (len(_new) >= 2 and _new[0] == '"' and _new[-1] == '"'
                    and _new.count('"') == 2):
                _new = _new[1:-1].strip()
            if _new == _txt.strip():
                continue
            if len(_new.split()) < 2:
                log.warning(
                    "[OTR_LedgerScriptWriter] stage-direction scrub would "
                    "empty line %s; original text kept", _ln.get("line_id"))
                continue
            log.warning(
                "[OTR_LedgerScriptWriter] stage-direction scrub %s: %r -> %r",
                _ln.get("line_id"), _txt[:70], _new[:70])
            _ln["text"] = _new
            _ln["word_count"] = len(_new.split())
            _sd_fixed += 1
        if _sd_fixed:
            led.save()
            log.warning(
                "[OTR_LedgerScriptWriter] stage-direction scrub fixed %d "
                "line(s) pre-freeze", _sd_fixed)

        # (b) SELF-VOCATIVE scrub: the composer sometimes opens a line
        # with the SPEAKER'S OWN first name as a vocative ("GULLIVER
        # REEVES: Gulliver, have you...") -- a character addressing
        # themselves. Strip the leading self-vocative + separator,
        # re-capitalize, restamp word_count; never touches a line that
        # addresses ANOTHER character.
        _voc_fixed = 0
        _name_by_cid = {
            str(c.get("char_id") or ""): str(c.get("name") or "")
            for c in (led.data.get("cast") or []) if isinstance(c, dict)
        }
        for _ln in led.data.get("lines") or []:
            if not isinstance(_ln, dict) or _ln.get("skip"):
                continue
            _nm = _name_by_cid.get(str(_ln.get("char_id") or ""), "")
            _first = (_nm.split() or [""])[0]
            _txt = str(_ln.get("text") or "")
            if len(_first) < 2 or not _txt:
                continue
            _m = re.match(
                r"^\s*" + re.escape(_first) + r"\s*[,!?—…:;-]+\s*",
                _txt, flags=re.IGNORECASE)
            if not _m:
                continue
            _rest = _txt[_m.end():].lstrip()
            if len(_rest.split()) < 2:
                continue                      # never empty a line
            _fixed = _rest[0].upper() + _rest[1:]
            log.warning(
                "[OTR_LedgerScriptWriter] self-vocative scrub %s (%s): "
                "%r -> %r", _ln.get("line_id"), _nm, _txt[:60], _fixed[:60])
            _ln["text"] = _fixed
            _ln["word_count"] = len(_fixed.split())
            _voc_fixed += 1
        if _voc_fixed:
            led.save()
            log.warning(
                "[OTR_LedgerScriptWriter] self-vocative scrub fixed %d "
                "line(s) pre-freeze", _voc_fixed)

        # (c) SELF-VOCATIVE ATTRIBUTION repair (round 5, 2026-06-10) --
        # the LAST pre-freeze word on speaker identity, deliberately AFTER
        # every text-mutating pass: the b004 acceptance catch was a line
        # REWRITTEN after scrubs (a)/(b) into "Gulliver, it's not just a
        # machine..." while stamped char_id=GULLIVER -- the content belongs
        # to the OTHER character (a self-vocative is an address, and you do
        # not address yourself). Deterministic, no LLM:
        #   * exactly TWO character rows in the cast -> re-attribute the
        #     line to the interlocutor (text kept -- it is now a CORRECT
        #     address), LOUD;
        #   * ambiguous (3+ characters) -> strip the leading vocative like
        #     scrub (b) (which already ran, BEFORE the late rewrites) and
        #     LOUD-warn the attribution; never silent, never raises.
        # Runs pre-freeze and pre-TTS, so the corrected speaker's voice is
        # the one minted. Frozen ledgers are never re-touched (this is the
        # writer; the ledger freezes after J).
        _att_fixed = 0
        _char_rows = [
            c for c in (led.data.get("cast") or [])
            if isinstance(c, dict)
            and str(c.get("name") or "").strip().upper() != "ANNOUNCER"
            and str(c.get("char_id") or "")
        ]
        for _ln in led.data.get("lines") or []:
            if not isinstance(_ln, dict) or _ln.get("skip"):
                continue
            _cid = str(_ln.get("char_id") or "")
            _nm = _name_by_cid.get(_cid, "")
            _first = (_nm.split() or [""])[0]
            _txt = str(_ln.get("text") or "")
            if len(_first) < 2 or not _txt:
                continue
            _m = re.match(
                r"^\s*" + re.escape(_first) + r"\s*[,!?—…:;-]+\s*",
                _txt, flags=re.IGNORECASE)
            if not _m:
                continue
            _others = [c for c in _char_rows
                       if str(c.get("char_id") or "") != _cid]
            if len(_char_rows) == 2 and len(_others) == 1:
                _new_cid = str(_others[0].get("char_id") or "")
                log.warning(
                    "[OTR_LedgerScriptWriter] self-vocative re-attribution "
                    "%s: %s->%s (%r is %s addressing %s)",
                    _ln.get("line_id"), _cid, _new_cid, _txt[:50],
                    _others[0].get("name"), _nm)
                _ln["char_id"] = _new_cid
                _att_fixed += 1
            else:
                _rest = _txt[_m.end():].lstrip()
                if len(_rest.split()) >= 2:
                    _fixed = _rest[0].upper() + _rest[1:]
                    log.warning(
                        "[OTR_LedgerScriptWriter] self-vocative LATE strip "
                        "%s (%s, ambiguous %d-char cast): %r -> %r",
                        _ln.get("line_id"), _nm, len(_char_rows),
                        _txt[:50], _fixed[:50])
                    _ln["text"] = _fixed
                    _ln["word_count"] = len(_fixed.split())
                    _att_fixed += 1
                else:
                    log.warning(
                        "[OTR_LedgerScriptWriter] self-vocative on %s (%s) "
                        "left as-is (strip would empty the line; ambiguous "
                        "cast)", _ln.get("line_id"), _nm)
        if _att_fixed:
            led.save()
            log.warning(
                "[OTR_LedgerScriptWriter] self-vocative attribution pass "
                "fixed %d line(s) pre-freeze", _att_fixed)

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

        # --- J.5. Post-composition title regen (late binding) ---------
        # Per Jeffrey 2026-05-10: when the user leaves episode_title
        # blank, regenerate the title from the FINAL story material via
        # the LLM. The prompt does NOT see the news_seed -- the title is
        # grounded purely in the finished episode. User-typed
        # episode_title still wins; LLM only fires on blank input;
        # outline.title is the last-resort fallback when the LLM call
        # fails or its output is rejected by the guardrails.
        #
        # Sprint 3E (2026-05-25) -- scratchpad + late binding:
        #  - The title is bound LATE, here, after the script exists.
        #    The per-line composer (section I) ran with `EPISODE_TITLE:
        #    TBD` in canon_header, so no provisional / outline title was
        #    ever placed where a beat could speak it. There is no "old
        #    title" baked into dialogue, so the fragile post-hoc
        #    verbatim string substitution (the former section J.6) is
        #    removed entirely -- it only caught verbatim quotes anyway
        #    and let paraphrases slip through.
        #  - `_generate_title_from_script` is now a forced-scratchpad
        #    pass (3 physical details -> 3 candidate titles -> final
        #    TITLE: line) reading the whole-arc excerpt set, not a thin
        #    head-of-script slice. The writer passes the outline
        #    premise as additional grounding (the story spine, not the
        #    news article). `arc_verdict` is left "" -- the Sprint 5B
        #    whole-script critic that would emit it is not built yet.
        title_source = "outline_fallback"
        if resolved["episode_title"]:
            # User typed a value; respect it verbatim.
            final_title = resolved["episode_title"]
            title_source = "user"
        else:
            assembled_script = "\n\n".join(script_text_parts).strip()
            # LLM slot: creative -- title regen is a narrative pass
            # (scratchpad: extract physical details, draft candidates,
            # commit a final title). One LLM call produces the whole
            # scratchpad + the parsed TITLE: line. Routed through the
            # writer's creative_writing_model slot; no widget.
            # Sprint 0 (v4 plan): helper_context attribution.
            with slot_scheduler.helper_context("generate_title"):
                regen_title = _generate_title_from_script(
                    creative_generate_fn,
                    assembled_script,
                    temperature=resolved["temperature"],
                    premise=outline.premise,
                    arc_verdict="",
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
        # canon readers) will see. No spoken-line patching is needed:
        # late binding means dialogue never carried a provisional title.
        canon.title = final_title
        _OTRC.write_episode_canon(episode_root, canon)
        log.info(
            "[OTR_LedgerScriptWriter] episode_canon written with "
            "title=%r (source=%s) at %s",
            final_title, title_source,
            episode_root / _OTRC.EPISODE_CANON_FILENAME,
        )

        # --- K. Stamp meta block --------------------------------------
        # Stamps the run parameters into meta.gen_params_initial for
        # forensic / soak inspection. Also stamps episode_title
        # (forward-compat title chain slot) and perfect_run_spacesaver.
        meta = led.data.setdefault("meta", {})
        meta["gen_params_initial"] = {
            "target_words":         resolved["target_words"],
            "num_characters":       resolved["num_characters"],
            # S30 B2b: the legacy `model_id` key is DELETED outright.
            # Every consumer that previously read meta.gen_params_initial.
            # model_id now reads creative_writing_model + technical_model
            # explicitly (B3 onward).
            "creative_writing_model": resolved["creative_writing_model"],
            "technical_model":        resolved["technical_model"],
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
        # S30 B2b: top-level slot stamps + per-phase routing trace.
        # `gen_params_by_phase` records the slot + resolved model for
        # each writer-level LLM phase that fired. Critic / cascade
        # phases that live in B3+ nodes stamp their own entries when
        # they land.
        meta["creative_writing_model"] = resolved["creative_writing_model"]
        meta["technical_model"]        = resolved["technical_model"]
        meta["slot_transitions"]       = slot_scheduler.transitions
        meta["slot_calls_by_slot"]     = dict(slot_scheduler.calls_by_slot)
        # S32 B6: per-helper / per-phase forensic stamping. Downstream
        # consumers + final QA review can audit (a) which helpers
        # used which slots, and (b) the ordered list of slot
        # transitions captured during this run. Default-config
        # (creative == technical) keeps transitions == 0 and the
        # by-phase list empty; differing-slots populates both.
        meta["slot_calls_by_helper"] = {
            helper: dict(buckets)
            for helper, buckets in slot_scheduler.slot_calls_by_helper.items()
        }
        meta["slot_transitions_by_phase"] = [
            dict(record) for record in slot_scheduler.slot_transitions_by_phase
        ]
        # gen_params_by_phase rows track only the phases the writer
        # invoked. Each row carries the slot + the resolved repo id
        # consulted at call time + the per-slot sampling profile
        # (top_p / min_p / repetition_penalty) for the creative
        # phases. Technical phases use the same closure sampling.
        gen_params_by_phase: dict[str, dict] = {}
        if resolved["style_pending"]:
            gen_params_by_phase["style_picker"] = {
                "slot":  "creative",
                "model": resolved["creative_writing_model"],
            }
        if meta.get("news") is not None:
            gen_params_by_phase["news_interpreter"] = {
                "slot":  "technical",
                "model": resolved["technical_model"],
            }
        gen_params_by_phase["cast_lock"] = {
            "slot":  "creative",
            "model": resolved["creative_writing_model"],
        }
        gen_params_by_phase["outline"] = {
            "slot":  "creative",
            "model": resolved["creative_writing_model"],
        }
        gen_params_by_phase["dialogue_composer"] = {
            "slot":  "creative",
            "model": resolved["creative_writing_model"],
        }
        if title_source == "llm_post_composition":
            gen_params_by_phase["title_regen"] = {
                "slot":  "creative",
                "model": resolved["creative_writing_model"],
            }
        meta["gen_params_by_phase"] = gen_params_by_phase
        # Always stamp the resolved final title (user / LLM regen / outline
        # fallback). title_source records which branch won so downstream
        # consumers and BUG_LOG forensics can tell user-typed from
        # LLM-regenerated runs without inspecting widget state.
        meta["episode_title"] = final_title
        meta["title_source"] = title_source
        # Sprint 3E (2026-05-25): meta.title_substitution is retired.
        # Late title binding means dialogue never carried a provisional
        # title, so there is no post-hoc substitution to record. The
        # former J.6 verbatim-substitution block and its title-swap
        # helper were both removed in this sprint.
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
        # portrait_prompt is the cast row's character_description.
        # (2026-06-10 gap-audit doc fix: the legacy compose_shot_prompt
        # referenced here was DELETED with otr_video_plan.py; the live
        # seam that appends era_tail + style_tail is now
        # _otr_story_brief_helpers.finish_visual_prompt, called by
        # ShotLock M4, the image-prompt deriver, and the render driver's
        # scene composer.) This short, content-focused field is the right
        # Tier-1 input. The 3-tier fallback in resolve_character_portrait
        # already covers the empty case.
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
        }
        meta["style"] = resolved["style"]

        # --- K.5.5. meta.story_brief reflection pass ------------------
        # Per Sprint C final plan Q1 lock (K.5.5, E-01). Writes to meta
        # only; lines untouched; runs on technical_model slot per L-2.
        # Failure path stamps story_brief_status per L-6 fail-loud
        # sentinel pattern. Refinement section 3.6 sync-barrier wording
        # was amended at C0b -- this call inherits the non-blocking
        # BUG-LOCAL-228 timeout contract via technical_generate_fn.
        #
        # Dual-slot OOM analysis (E-15 revised at C1 audit per RR-A1):
        # the loader is single-slot by architecture. request_slot(
        # technical_model) inside technical_generate_fn calls
        # unload_llm() to evict any prior resident model BEFORE loading
        # the next. Peak transient VRAM during the swap is max(
        # creative_size, technical_size), not their sum. No explicit
        # pre-eviction call is needed; regression tests in C5a2 prove
        # the no-OOM property.
        # LLM slot: technical
        # Sprint 0 (v4 plan): helper_context attribution.
        with slot_scheduler.helper_context("story_brief_reflection"):
            _brief_delta = run_story_brief_reflection(
                led,
                technical_generate_fn,
                technical_model_id=resolved["technical_model"],
            )
        meta.update(_brief_delta)

        # --- Story-spine Wave 2: the post-script passes (Stage 2.5 length
        # pass / Stage 3 QA router / Stage 3.5 micro-repair / unload /
        # Stage 4 scrub), in-process + env-gated, DEFAULT ON (opt-out). The
        # orchestrator runs them out of the box; OTR_ENABLE_STORY_SPINE=0
        # restores the writer's byte-identical default path (the unload
        # block in the else branch). When on, the orchestrator performs the
        # writer-LLM unload itself, after the LLM passes (D8). The spine
        # NEVER raises (PD1); the only fail-loud signal is the REJECT gate
        # checked right after this block. No node surface, no workflow-JSON
        # change (D4/D5); model ids come from resolved[...] (QA -> technical
        # slot, length pass + micro-repair -> creative slot).
        try:
            from . import _otr_story_spine as _OTRSPINE
        except ImportError:  # pragma: no cover - standalone / test load
            import _otr_story_spine as _OTRSPINE  # type: ignore
        if _OTRSPINE.enabled():
            _OTRSPINE.run_post_script_spine(
                led, meta, outline,
                creative_generate_fn=creative_generate_fn,
                technical_generate_fn=technical_generate_fn,
                resolved=resolved,
                slot_scheduler=slot_scheduler,
            )
        else:
            # S3 (VRAM): the writer is done with the LLM after
            # story_brief_reflection (the last LLM phase) -- evict it here,
            # after the script and BEFORE the downstream TTS / render phase,
            # so it is not co-resident with Bark / Kokoro / HuMo / FLUX.
            # Never raises (PD1, audio is king); gated by
            # OTR_WRITER_UNLOAD_AFTER_SCRIPT (default on). VRAM-envelope
            # benefit needs an operator GPU smoke to confirm.
            try:
                from . import _otr_writer_vram as _OTRVRAM
            except ImportError:  # pragma: no cover - standalone / test load
                import _otr_writer_vram as _OTRVRAM  # type: ignore
            meta["writer_llm_unload"] = _OTRVRAM.unload_writer_llm_after_script()

        # REJECT gate (go-forward Sprint 3): the story-spine QA router can
        # mark an episode structurally unshippable (a dead ending, a broken
        # turn, an unclear premise -- defects a one-line edit cannot fix).
        # The spine itself NEVER raises (PD1 -- it sets story_verdict=REJECT,
        # unloads the writer LLM, and skips the scrub); the WRITER raises
        # here at its boundary, matching the fail-loud cast-lock pattern
        # above. An aborted run produces no node output, so the graph stops
        # BEFORE the FLUX -> HuMo -> LTX -> Bark render -- no audio is ever
        # produced for a rejected story. A QA crash fails open to PASS and
        # never reaches here, so it stays fail-soft.
        if meta.get("story_verdict") == "REJECT":
            raise RuntimeError(
                f"OTR reject gate: "
                f"{meta.get('story_reject_reason') or 'story rejected'}"
            )

        # Sprint D D2b: stamp creative slot identity into meta so
        # FreezeCascade preserves it via the existing script_json
        # plumb. Sprint C gotcha #4 -- writer was the source of
        # truth for the creative model but never put it into the
        # frozen ledger, so post-freeze diagnostics were blind to
        # which creative model produced the script. The two new
        # meta keys are additive; audio path reads only
        # meta.story_brief so byte identity holds.
        meta["creative_model"] = resolved["creative_writing_model"]
        try:
            _creative_row = _otr_model_catalog._by_repo_id().get(
                resolved["creative_writing_model"],
            )
            meta["creative_prompt_profile"] = (
                _creative_row.prompt_profile if _creative_row else "modern"
            )
        except Exception:  # noqa: BLE001
            meta["creative_prompt_profile"] = "modern"

        # [OpenRouter S5] Remote-LLM provenance. For any slot bound to an
        # OpenRouter handle, stamp provider + virtual handle + resolved
        # slug + basic params + schema-mode so the env-side binding is
        # recorded in the run (the slug is a public model id, not a
        # secret; the API key is never stamped). Empty for local runs, so
        # the offline baseline is byte-identical (C1). Never raises (PD1).
        try:
            from . import _otr_openrouter_backend as _orb
            meta.update(_orb.openrouter_meta_for(
                resolved["creative_writing_model"],
                resolved["technical_model"],
            ))
            # S3 (2026-06-01): also stamp the slug each slot RESOLVES to (on
            # the live bindings/fallback chain) + catalog staleness, so the
            # run records which remote model would serve each slot and how
            # fresh discovery was. {} when remote is disabled (C1 byte-ident).
            meta.update(_orb.openrouter_run_meta())
        except Exception:  # noqa: BLE001 -- provenance must never break a run
            pass

        # NOTE: meta.episode_title is stamped once, by the J.5
        # post-composition title pass (meta["episode_title"] = final_title
        # above). A Sprint-E "K.5.7" block used to re-stamp it here from
        # the raw episode_title widget value -- which ran AFTER J.5 and
        # clobbered the LLM-generated title with "" whenever the widget
        # was left blank, so the video title chain fell to the timestamp
        # last-resort (BUG-LOCAL-236). K.5.7 deleted 2026-05-20; J.5 is
        # the single authority for the title.

        # --- L. Assemble return values --------------------------------
        # Tier 1 fix #2 (2026-05-11): derive final script_text from the
        # CANONICAL ledger rows, not from the in-flight script_text_parts
        # list. Post-loop mutations (the news_close_brief announcer
        # override in I.5) write to led.data["lines"] but were not
        # always mirrored back into script_text_parts. The
        # script_text_parts list is now diagnostic-only; the ledger is
        # the source of truth for the slot-0 STRING output.
        # Sprint 3E (2026-05-25): the former J.6 post-hoc title
        # substitution -- another such ledger-only mutation -- is gone
        # (late title binding means no provisional title in dialogue).
        script_text = _PL.assemble_script_text_from_ledger(led.data)
        # story-ledger DRIFT chunk 2 (2026-06-25): PRE-FREEZE cross-stage
        # consistency guard. contract / outline / canon are the REAL objects
        # here; OTR_CastLock is a DOWNSTREAM node (it re-locks the FROZEN
        # ledger), so the cast source-of-truth is led.data["cast"] -> castlock
        # is None. Audio-safe: non-strict => LOUD warn + meta.consistency_status,
        # NEVER raises (a guard that breaks the writer is worse than the drift;
        # CI enforcement lives in tests/test_ledger_canon_parity.py). Stamped
        # BEFORE the json.dumps so consistency_status ships in the ledger.
        try:
            from . import _otr_ledger_consistency as _OTRLCONS
            _cons_status = _OTRLCONS.evaluate_consistency(
                contract=contract, outline=outline, castlock=None,
                canon=canon, ledger=led.data, strict=False,
            )
            if not _cons_status.get("clean", True):
                log.warning(
                    "[OTR_LedgerScriptWriter] ledger/canon consistency: %d "
                    "defect(s) %s (stamped meta.consistency_status)",
                    _cons_status.get("defect_count", 0),
                    [d.get("field") for d in _cons_status.get("defects", [])],
                )
        except Exception as _cons_exc:  # noqa: BLE001 -- never break the writer
            log.warning(
                "[OTR_LedgerScriptWriter] consistency check skipped: %r",
                _cons_exc,
            )
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
        # S30 B2a: broadcast both resolved model ids on the writer's
        # output sockets. Labels stripped (resolved["creative_writing_model"]
        # / ["technical_model"] are already _strip_label_suffix-normalized).
        # B3 wires `technical_model` into the cascade.
        if _refine_active:
            # Stash what the refine loop in run() needs: the grader fn + premise
            # to grade THIS pass; the outline + cast_seed to build the next
            # REVISE overlay; episode_root for loser cleanup; led to stamp the
            # refine telemetry on the winner.
            self._refine_last = {
                "outline": outline,
                "premise": str(getattr(outline, "premise", "") or ""),
                "creative_fn": creative_generate_fn,
                "cast_seed": cast_seed,
                "episode_root": episode_root,
                "led": led,
                "script_text": script_text,
            }
        return (
            script_text,
            script_json,
            news_json,
            est_minutes,
            resolved["technical_model"],
        )


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
    #     S30 B2a: single `model_id` widget split into
    #     `creative_writing_model` + `technical_model`; optional
    #     widget count grows 10 -> 11.
    try:
        spec = cls.INPUT_TYPES()
        assert "required" in spec, "missing required block"
        assert "optional" in spec, "missing optional block"
        # Required block: episode_title, target_words, num_characters.
        req_keys = list(spec["required"].keys())
        assert req_keys == ["episode_title", "target_words", "num_characters"], \
            f"required widget order drift: {req_keys}"
        # Optional block: the clean set after Phase 0-3 cleanup + B2a
        # two-widget split.
        for k in ("seed", "creative_writing_model", "technical_model",
                  "custom_premise", "include_act_breaks", "act_count",
                  "style", "style_custom", "creativity",
                  "perfect_run_spacesaver", "lemmy_cameo"):
            assert k in spec["optional"], f"optional missing key: {k}"
        # Legacy widgets MUST be absent post-cleanup. `model_id` joins
        # the legacy list at B2a (replaced by the two slot widgets).
        for legacy in ("cleanup_model_id", "self_critique",
                       "target_length", "arc_enhancer", "open_close",
                       "model_id"):
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
        # act_count combo: "auto" sentinel + explicit "1".."7".
        ac_choices, ac_meta = spec["optional"]["act_count"]
        assert ac_choices == ["auto", "1", "2", "3", "4", "5", "6", "7"], \
            f"act_count combo drift: {ac_choices!r}"
        assert ac_meta["default"] == "auto", \
            f"act_count default drift: {ac_meta['default']!r}"
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
        # optimization_profile widget removed 2026-05-23 (UI
        # simplification, ROADMAP PRIORITY 2): of its VRAM tiers only
        # "Standard" was ever validated. _resolve_inputs keeps its
        # "Standard" default + meta plumbing, so the value still flows.
        # `lemmy_cameo` widget added (BUG-LOCAL-260) -- ungated cameo
        # toggle, defaults False. Optional count history: 15 (pre-
        # removal) -> 14 (optimization_profile removed) -> 15
        # (lemmy_cameo added). Current: 11 widget-surface + 4 Phase 4
        # v4 sampling knobs.
        assert n_optional == 16, (
            f"optional widget count drift: {n_optional} "
            f"(expected 16: 11 widget-surface + 4 Phase 4 v4 "
            f"sampling knobs + story_scaffold appended 2026-06-24)"
        )
        # S30 B2a: both model widgets carry the catalog dropdown_choices()
        # output (list of labels). DEFAULT must match catalog.DEFAULT_LLM.
        for slot_key in ("creative_writing_model", "technical_model"):
            choices, meta = spec["optional"][slot_key]
            assert isinstance(choices, list) and choices, (
                f"{slot_key} dropdown empty or wrong shape"
            )
            from nodes._otr_model_catalog import DEFAULT_LLM as _D
            assert meta["default"] == _D, (
                f"{slot_key} default drift: {meta['default']!r} != {_D!r}"
            )
        print("[2/9] PASS: INPUT_TYPES schema (16 optional widgets)")
    except Exception:
        failures.append(("2/9 INPUT_TYPES", traceback.format_exc()))
        print("[2/9] FAIL: INPUT_TYPES schema")

    # 3. Locked output contract.
    #     S30 B2a broadcast both resolved model ids; the 2026-05-29
    #     lean-down removed the zero-consumer creative_writing_model
    #     output. Only technical_model remains; RETURN_TYPES is 5.
    try:
        assert cls.RETURN_TYPES == (
            "STRING", "STRING", "STRING", "INT", "STRING",
        ), f"RETURN_TYPES drift: {cls.RETURN_TYPES}"
        assert cls.RETURN_NAMES == (
            "script_text", "script_json", "news_used", "estimated_minutes",
            "technical_model",
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

        # S32 B1 paired-contract: smoke test passes the same fn to
        # both slots since the mock doesn't distinguish.
        pick = _OTRSP_smoke.pick_style(
            creative_fn=_smoke_gen,
            technical_fn=_smoke_gen,
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
                creative_fn=_smoke_gen, technical_fn=_smoke_gen,
                article_text="",
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

    # 8. _generate_title_from_script -- Sprint 3E scratchpad title pass.
    try:
        SAMPLE_SCRIPT = (
            "[VOICE: ANNOUNCER, neutral] Tonight, on Tales From Beyond.\n\n"
            "[VOICE: AEGEUS, tense] The signal -- it's repeating itself.\n\n"
            "[VOICE: PHOEBE, alarmed] That's impossible. The dish was "
            "decommissioned six years ago."
        )

        # The scratchpad pass parses the title from the LAST line that
        # begins with "TITLE:". Helper for canned scratchpad output.
        def _scratch(details, candidates, final):
            body = "DETAILS:\n" + "\n".join(details)
            body += "\nCANDIDATES:\n" + "\n".join(candidates)
            body += f"\nTITLE: {final}"
            return body

        # 8a. Empty script -> "" (no LLM call attempted).
        def _trap(*a, **kw):
            raise AssertionError("LLM should not be called on empty script")
        assert _generate_title_from_script(_trap, "") == ""
        assert _generate_title_from_script(_trap, "   \n  \n") == ""

        # 8b. LLM raises -> "".
        def _raises(*a, **kw):
            raise RuntimeError("LLM offline")
        assert _generate_title_from_script(_raises, SAMPLE_SCRIPT) == ""

        # 8c. Scratchpad with a clean final TITLE: line -> parsed title.
        def _clean(*a, **kw):
            return _scratch(
                ["a decommissioned dish", "a repeating signal", "static"],
                ["The Echo Below", "Repeating Static", "Dead Dish Signal"],
                "The Echo Below",
            )
        assert _generate_title_from_script(_clean, SAMPLE_SCRIPT) == \
            "The Echo Below"

        # 8d. Last TITLE: line wins; markdown / quote wrappers stripped.
        def _wrapped(*a, **kw):
            return (
                "DETAILS:\n- a dish\n- a signal\n- static\n"
                "CANDIDATES:\nThe Dish\nThe Signal\nPulse\n"
                '**TITLE:** "Pulse"'
            )
        assert _generate_title_from_script(_wrapped, SAMPLE_SCRIPT) == "Pulse"

        # 8e. Smart-quote wrappers on the final title stripped.
        def _smart(*a, **kw):
            return _scratch(
                ["crop", "crash", "field"],
                ["Agri-Crash", "Crop Failure", "The Field"],
                "“Agri-Crash”",
            )
        assert _generate_title_from_script(_smart, SAMPLE_SCRIPT) == \
            "Agri-Crash"

        # 8f. No parseable TITLE: line -> "" (caller falls back).
        def _no_title(*a, **kw):
            return (
                "DETAILS:\n- a dish\n- a signal\n- static\n"
                "CANDIDATES:\nThe Dish\nThe Signal\nPulse\n"
                "I could not decide on a single title."
            )
        assert _generate_title_from_script(_no_title, SAMPLE_SCRIPT) == ""

        # 8g. Stuck defaults rejected even when emitted on a TITLE: line.
        for stuck in ("Untitled", "Signal Lost", "Episode", "the last frequency"):
            def _stuck(*a, _s=stuck, **kw):
                return _scratch(["a", "b", "c"], ["x", "y", "z"], _s)
            assert _generate_title_from_script(_stuck, SAMPLE_SCRIPT) == "", \
                f"stuck default {stuck!r} should be rejected"

        # 8h. Full-sentence leak on the TITLE: line (>10 words) rejected.
        def _leak(*a, **kw):
            return _scratch(
                ["a", "b", "c"], ["x", "y", "z"],
                "Here is a title that the model leaked as a complete "
                "English sentence well over ten words long indeed",
            )
        assert _generate_title_from_script(_leak, SAMPLE_SCRIPT) == ""

        # 8i. Empty LLM output -> "".
        def _empty(*a, **kw):
            return ""
        assert _generate_title_from_script(_empty, SAMPLE_SCRIPT) == ""

        # 8j. 80+ char title on the TITLE: line gets truncated to 80.
        def _long(*a, **kw):
            return _scratch(
                ["a", "b", "c"], ["x", "y", "z"], "X" * 90 + " A Title",
            )
        result_long = _generate_title_from_script(_long, SAMPLE_SCRIPT)
        assert len(result_long) <= 80, \
            f"title truncation failed: len={len(result_long)}"

        # 8k. Trailing punctuation on the TITLE: line stripped.
        def _punct(*a, **kw):
            return _scratch(
                ["a", "b", "c"],
                ["Final Frequency", "Dead Air", "Last Signal"],
                "Final Frequency.",
            )
        assert _generate_title_from_script(_punct, SAMPLE_SCRIPT) == \
            "Final Frequency"

        # 8l. News seed must NOT be in the prompt the helper builds, and
        # the prompt MUST drive a scratchpad (DETAILS / CANDIDATES /
        # TITLE). The premise IS passed through (story spine, not the
        # news article) so it is allowed to appear.
        captured: dict = {}
        def _capture(messages, **kw):
            captured["messages"] = messages
            captured["kw"] = kw
            return _scratch(["a", "b", "c"], ["x", "y", "z"], "Clean Title")
        out_title = _generate_title_from_script(
            _capture, SAMPLE_SCRIPT, premise="A lonely dish hears itself.",
        )
        assert out_title == "Clean Title", out_title
        full_prompt_text = " ".join(
            m.get("content", "") for m in captured["messages"]
        )
        # Nothing news-seed-flavored should leak.
        for forbidden in ("news", "headline", "article", "RSS"):
            assert forbidden.lower() not in full_prompt_text.lower(), (
                f"title-regen prompt leaked forbidden token {forbidden!r}: "
                f"{full_prompt_text[:200]}..."
            )
        # The scratchpad must be forced.
        for required in ("DETAILS", "CANDIDATES", "TITLE:"):
            assert required in full_prompt_text, (
                f"title-regen prompt missing scratchpad marker {required!r}"
            )
        # Scratchpad needs a real token budget, not the old 24.
        assert captured["kw"].get("max_new_tokens", 0) >= 100, (
            "scratchpad pass must request enough tokens to reach the "
            f"final TITLE: line; got {captured['kw'].get('max_new_tokens')}"
        )

        print("[8/9] PASS: _generate_title_from_script (Sprint 3E scratchpad)")
    except Exception:
        failures.append(("8/9 _generate_title_from_script", traceback.format_exc()))
        print("[8/9] FAIL: _generate_title_from_script")

    # 9. _build_title_excerpt_set + Sprint 3E late-binding source check.
    try:
        # 9a. Empty / blank script -> all-empty excerpt dict.
        for blank in ("", "   \n\n  "):
            ex = _build_title_excerpt_set(blank)
            assert ex == {
                "opening_lines": "", "middle_lines": "", "ending_lines": "",
            }, f"blank script excerpt drift: {ex!r}"

        # 9b. Long script: opening / middle / ending windows are non-empty
        # and do not overlap on the head / tail blocks.
        blocks = [f"[VOICE: C, neutral] line {i}" for i in range(40)]
        long_script = "\n\n".join(blocks)
        ex = _build_title_excerpt_set(long_script)
        assert ex["opening_lines"], "opening window empty on long script"
        assert ex["middle_lines"], "middle window empty on long script"
        assert ex["ending_lines"], "ending window empty on long script"
        # The opening window must contain block 0; the ending window the
        # last block; the title pass therefore sees the whole arc.
        assert "line 0" in ex["opening_lines"]
        assert "line 39" in ex["ending_lines"]
        # Whole-arc proof: the ending excerpt carries content the old
        # head-only [:3000] slice would have missed -- on a 40-line
        # script the tail blocks are far past any short head slice.
        assert "line 39" not in ex["opening_lines"], (
            "ending content leaked into opening window"
        )

        # 9c. Short script: windows still resolve, no crash, no overlap
        # explosion (ending excluded when fewer blocks than tail window).
        short = "\n\n".join(blocks[:3])
        ex = _build_title_excerpt_set(short)
        assert ex["opening_lines"], "short script opening empty"

        # 9d. Sprint 3E source contract: the writer uses the literal
        # EPISODE_TITLE: TBD in the composition header, and the fragile
        # post-hoc verbatim substitution is gone.
        from pathlib import Path as _P
        _writer_src = _P(__file__).read_text(encoding="utf-8")
        assert "EPISODE_TITLE: TBD" in _writer_src, (
            "composition header must carry the EPISODE_TITLE: TBD literal"
        )
        # Needles assembled from fragments so the assertion strings
        # themselves do not count as occurrences in the writer source.
        _sub_helper_def = "def _substitute" + "_title_in_text"
        assert _sub_helper_def not in _writer_src, (
            "post-hoc verbatim title substitution helper must be removed"
        )
        _sub_min_guard = "_TITLE_SUB" + "_MIN_LEN"
        assert _sub_min_guard not in _writer_src, (
            "the title-substitution min-length guard must be removed"
        )
        _j6_header = "--- J." + "6."
        assert _j6_header not in _writer_src, (
            "the post-hoc title-substitution section header must be removed"
        )
        # The scratchpad helper must exist and be wired into the title
        # call as the whole-arc excerpt source.
        assert "_build_title_excerpt_set" in _writer_src

        print("[9/9] PASS: _build_title_excerpt_set + late-binding source check")
    except Exception:
        failures.append(("9/9 _build_title_excerpt_set", traceback.format_exc()))
        print("[9/9] FAIL: _build_title_excerpt_set")

    # Summary.
    if not failures:
        print("\nSELF-TEST PASS: 9/9")
        sys.exit(0)
    else:
        print(f"\nSELF-TEST FAIL: {len(failures)} of 9")
        for name, tb in failures:
            print(f"\n--- {name} ---\n{tb}")
        sys.exit(1)
