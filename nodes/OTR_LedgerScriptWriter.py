"""OTR_LedgerScriptWriter — v2.0 LPL writer with legacy-style widget surface restored 2026-05-10.

Pipeline (unchanged from v2.0 LPL):

    1. Validate + normalize inputs (legacy widget set restored).
    2. Resolve effective values:
       - news_seed = custom_premise verbatim if non-empty,
         else RSS auto-fetch via story_orchestrator._fetch_science_news.
       - style = the single deterministic engine call,
         _otr_style_catalog.build_story_contract(), made once cast_seed
         and script_brief both exist (style-engine consolidation,
         2026-07-05). No widget, no LLM picker.
       - act_count from widget (1-6, PBUG-20260825-01); episode length is an observation
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
        num_characters    INT     (REQUESTED speaking characters, 1 = monologue;
                                   a request, not a cap -- the story may use more)
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
        act_count         combo   ('1'-'6' -- THE one length-shaped knob;
                                   always honoured, never derived)
        creativity        combo   (maps to temperature + top_p preset)
        perfect_run_spacesaver BOOLEAN (DEPRECATED 2026-08-08 -- no-op
                                        sentinel; kept to preserve widget
                                        positional layout per BUG-LOCAL-097.
                                        Formerly triggered RTXUpscale's
                                        per-episode cleanup, which was
                                        retired with the RTX-VSR node.)
        min_p             FLOAT   (sampling tail cut; 0.0 disables)
        repetition_penalty FLOAT  (anti-loop penalty; 1.0 disables)
        max_new_tokens_cap INT    (per-line composer token ceiling)

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

import hashlib
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

# S30 B2a: catalog drives the dropdown_choices() for the two model widgets
# at INPUT_TYPES() registration time. Pure-Python module, no torch /
# transformers / GPU work -- safe at module-import time.
from . import _otr_model_catalog as _otr_model_catalog  # noqa: E501
from ._otr_generation_budget import (
    CAPACITY_PHASE_OUTPUT_LIMIT,
    GenerationContextOverflowError,
    GenerationDegeneracyError,
    PromptContextOverflowError,
    fit_output_tokens,
)
from ._otr_text_metrics import canonical_word_count, set_line_text_metrics
from . import _otr_writer_heartbeat as _OTRHB
# S1 platform-portability: the explicit LLM runtime policy (stdlib-only).
from ._otr_shared import llm_policy as _llm_policy

# Stage 2C (multi-modal story schema, 2026-07-05): the story-routing layer
# supplies the source_bank dropdown (list_bank_ids at INPUT_TYPES) and the
# run-intent gate (require_runnable_bank, FIRST statement of run()). The
# module is stdlib-only and LAZY (zero I/O at import; the registries load on
# first call) -- safe at module-import time.
from . import _otr_story_routing as _otr_story_routing

# Stage 3C (2026-07-06): the visual-style layer supplies the visual_style
# dropdown (list_style_ids at INPUT_TYPES) and the fail-loud id gate
# (resolve_visual_style, beside the bank gate). Same lazy/stdlib posture.
from . import _otr_visual_styles as _otr_visual_styles

# Lane-enablement chunk 3 (2026-07-05): the source-payload contract layer --
# the bank's fetcher/interpreter ids resolve to registered callables
# (fail-loud; a bank without a built lane raises SourceContractMissingError,
# never a silent slide into the science path). Stdlib-only + lazy wrapper
# imports -- safe at module-import time, same posture as the routing import.
from . import _otr_source_payload as _otr_source_payload
from . import _otr_roster_gender as _otr_roster_gender
from . import _otr_bank_variants as _otr_bank_variants

# The ONE story-lane authority (2026-07-31): pipeline id -> dispatched runner
# (by NAME, resolved lazily) or the writer's own inline body, plus each
# dispatched lane's declared request-compatibility policy. This table used to
# live in this file, which meant nothing outside the writer could ask a
# question about lanes without importing ComfyUI. Stdlib-only + lazy, same
# posture as the routing import; it never imports back into this module.
from . import _otr_lane_specs as _LANES

# The two operator randomizers (2026-07-31): the `source_bank` roll and the
# `visual_style` roll, each switched on by its OWN dropdown sentinel and
# therefore controllable independently. Pure + stdlib-only + lazy (registry
# reads happen on call, never at import); zero LLM calls.
from . import _otr_rolls as _ROLLS

# Bake-off source-snapshot replay (2026-07-15, r3 ruling B7): a frozen source
# for a base bank, replayed across the base/_v2/_v3 triplet so the pack is the
# only variable. Stdlib-only leaf (imports only _otr_bank_variants) -- safe at
# module-import time, same posture as the routing/source-payload imports.
from . import _otr_source_snapshot as _otr_source_snapshot
from . import _otr_word_delivery as _OTRWD
# MODULE SCOPE ON PURPOSE (item F, 2026-08-17). This module was previously
# imported ONLY inside the `provenance_normalize` branch below, which is true
# for two of the six banks. The announcer's work title is now read on the
# bank-agnostic path, so a branch-local import would raise UnboundLocalError on
# every `original`, `media_archive` and news-lane episode -- the majority of the
# corpus, and invisible to any test that only exercises an adaptation lane.
# `identity_from_meta` never raises and returns an empty title for a lane it
# cannot name, so evaluating it unconditionally is safe by its own contract.
from . import _otr_source_identity as _OTRSID

# Sprint C C5a2 (2026-05-15) module-level import per E-22 / RR-B4. The
# reflection pure module is wired into execute() at K.5.5 -- see the
# reflection call site below the K.5 visual_plan stamp. Module-level
# import (not hot-path) so a typo / refactor surfaces at module load
# time rather than during the first script generation.
from ._otr_story_brief import (
    REJECT_JSON_PARSE as _STORY_BRIEF_REJECT_JSON_PARSE,
    REJECT_SCHEMA as _STORY_BRIEF_REJECT_SCHEMA,
    derive_produced_open_brief,
    run_produced_story_summary,
    run_story_brief_reflection,
)
# The intro-rewrite block classifies its own failures, and a deliberate derive
# outcome must never be filed as an unexpected one. Imported by name so the
# `except` clause cannot silently become a NameError on the failure path --
# which is the one path nobody exercises by accident.
from ._otr_structured_call import StructuredCallFailedError

# Lean-mean order 9, slice 1 (2026-08-23). `_resolve_inputs` and the widget
# menus it is the sole interpreter of moved to their own module -- 484 lines out
# of this file, byte-identically, with a sha256 per block in the commit. They
# are imported BACK under their original names on purpose: every one of these
# is reached for as `OTR_LedgerScriptWriter.<name>` by a test or by the class
# body below, and a seam that renames things is a rewrite wearing a seam's
# clothes. The dependency runs one way -- `_otr_writer_inputs` never imports
# this module -- which is what makes the split legal at all.
from ._otr_writer_inputs import (  # noqa: F401 -- re-exported for callers/tests
    _ACT_COUNT_CHOICES,
    _CREATIVITY_CHOICES,
    _CREATIVITY_TEMP_MAP,
    _CREATIVITY_TOP_P_MAP,
    _DEFAULT_ACT_COUNT,
    _FABLE2_MAX_CAST,
    _LEMMY_CAMEO_CHOICES,
    _LEMMY_CAMEO_FORCE,
    _bank_has_no_source_contract,
    _resolve_creativity,
    _resolve_inputs,
)

# Lean-mean order 9, slice 2 (2026-08-23). The 911-line tail and the ten helpers
# and two contracts only it owns moved to their own module -- 1,514 lines out of
# this file, byte-identically, with a sha256 per block in the commit. The tail
# arrives back as a MIXIN because it reads `self` zero times and moving it as a
# free function would have meant re-indenting 911 lines, which no hash can
# check. Everything else is re-exported under its original spelling: three of
# these helpers are still called from `run()`, and the rest are reached for by
# name in tests. The dependency runs one way -- `_otr_writer_tail` never imports
# this module.
from ._otr_writer_tail import (  # noqa: F401 -- re-exported for callers/tests
    STORY_STYLE_STATUS_SCAFFOLD_OFF,
    TailFinalizer,
    WORDS_PER_MINUTE_ESTIMATE,
    WriterTailContext,
    WriterTailMixin,
    _apply_intro_rewrite_result,
    _build_news_payload,
    _build_title_excerpt_set,
    _generate_title_from_script,
    _resolve_cast_rng_seed,
    _stamp_final_slot_telemetry,
    _stamp_story_style_receipt,
    _title_source_for_custom_override,
)

log = logging.getLogger("OTR")


__all__ = ["OTR_LedgerScriptWriter"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# `VOICED_ROLES` was removed 2026-08-28: this copy had no reader (the loop
# below branches on `speaker_role == "character"` / `NON_VOICED_ROLES`). The
# authority for which roles are spoken is
# `nodes/_otr_spoken_text_policy.py`; the private per-lane copies elsewhere
# are deliberate and stay independent.

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

_SPOKEN_CODA_SOURCES = frozenset({"provenance", "news_close_brief", "none"})
"""Closed vocabulary for ``meta["spoken_coda_source"]``.

Names WHICH fact the announcer's closing line actually spoke:

  * ``provenance``       -- the deterministic ``provenance_coda_line``, appended
                           verbatim on a provenance-owned lane.
  * ``news_close_brief`` -- the interpreter's own note, appended verbatim on an
                           unowned lane (media_archive / the news lanes, where
                           the note is a genuine factual read).
  * ``none``             -- no source fact was spoken; attribution, if any, is
                           carried by the printed credits.

Closed and validated at write time because the corpus audit JOINS on it. A free
string would let a typo widen what the audit accepts, and this field exists
precisely because inferring the spoken fact from prose is what let a URL reach
the air for 84 lines across 30 episodes."""

# target_length widget removed 2026-05-11 (post-Phase-3 cleanup pass).
# The old "short (3 acts)" / "medium (5 acts)" / "long (7-8 acts)" combo
# is replaced by the `act_count` combo widget + `target_words`. Smoke
# presets are gone with it -- for a 30-word smoke run, type
# target_words=30 directly. Cleaner UX, one source of truth for
# episode shape.


# ---------------------------------------------------------------------------
# Style engine (style-engine consolidation, 2026-07-05): there is no
# widget-level style surface anymore. Every episode's style comes from
# exactly ONE call, `_otr_style_catalog.build_story_contract()`, made in
# run() once cast_seed and script_brief both exist. The old `style` /
# `style_custom` widgets, the three-way resolver, the two-pass LLM
# picker (`_otr_style_picker.py`), and the 10-slug seed pool are gone.
# ---------------------------------------------------------------------------


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


# `_OPTIMIZATION_PROFILE_CHOICES` was REMOVED 2026-08-28. Its widget left
# INPUT_TYPES on 2026-05-23 and the list stayed so that "re-exposing the widget
# ... is a one-line INPUT_TYPES add against this list". Three months on nothing
# reads it -- no widget, no workflow, no test -- and the sibling dial it was
# modelled on turned out to LABEL its output without controlling anything
# (vram_context_test.py, removed the same day). A list kept for a hypothetical
# future reader is scaffolding; the one-line add is no harder to write from
# scratch. The live "Standard" default in _resolve_inputs is untouched.


# ---------------------------------------------------------------------------
# Truncating generate_fn wrapper (top_p parametrized 2026-05-10)
# ---------------------------------------------------------------------------


# A-4 (2026-07-30, writer repair): `PromptContextOverflowError` now lives in
# `_otr_generation_budget`, the module that owns capacity arithmetic, and is
# imported above. It is RE-EXPORTED here on purpose: this name is what every
# caller and test in the tree reaches for, and the retry ladder -- documented
# pure, forbidden from importing the writer -- has to be able to name the same
# object to decide whether a capacity failure may re-roll. Two definitions of
# one error would be two policies over one state.
#
# No `__all__` is declared for the re-export: this module's public surface is
# its node classes, and narrowing it to one name to document one import would
# be a lie about the rest. The module-level import above IS the re-export.


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


class _LLMPreflight:
    """Result of _preflight_llm_selection: the ONE validated policy, the two
    normalized slot ids, and an immutable per-slot GGUF load_config (gguf slots
    only)."""

    __slots__ = ("creative_id", "technical_id", "policy", "load_config_by_slot")

    def __init__(self, *, creative_id, technical_id, policy, load_config_by_slot):
        self.creative_id = creative_id
        self.technical_id = technical_id
        self.policy = policy
        self.load_config_by_slot = load_config_by_slot


def _preflight_llm_selection(
    *, creative_writing_model, technical_model,
    llm_device, llm_attn_impl, llm_quant_policy, llm_vram_ceiling_gb,
    gguf_n_ctx, gguf_quant,
) -> "_LLMPreflight":
    """Single early resolution point for the two LLM slots (GGUF row registry,
    2026-07-16).

    Runs AFTER the bank / word-count / refine gates and BEFORE the
    story-scaffold env mutation and any source fetch/rerank. It builds the ONE
    validated LLMRuntimePolicy (byte-identical to the _resolve_inputs build --
    same six widgets, default lane_allowlist), normalizes the two slot ids, and
    resolves an immutable per-slot GGUF load_config. A gguf slot with an
    unknown quant, an out-of-range effective n_ctx, or a missing/malformed
    OTR_GGUF_SEED fails LOUD here, before any story work. No downstream consumer
    rebuilds effective config from live env.
    """
    from . import _otr_gguf_backend as _gguf

    policy = _llm_policy.LLMRuntimePolicy(
        device=str(llm_device),
        attn_impl=str(llm_attn_impl),
        quant_policy=str(llm_quant_policy),
        vram_ceiling_gb=float(llm_vram_ceiling_gb),
        gguf_n_ctx=int(gguf_n_ctx),
        gguf_quant=str(gguf_quant),
    )
    norm_creative = _otr_model_catalog.validate_model_id(creative_writing_model)
    norm_technical = _otr_model_catalog.validate_model_id(technical_model)
    load_config_by_slot: dict[str, Any] = {}
    by_repo = _otr_model_catalog._by_repo_id()
    for slot, mid in (("creative", norm_creative), ("technical", norm_technical)):
        row = by_repo.get(mid)
        if getattr(row, "provider", "local") == "gguf_native":
            load_config_by_slot[slot] = _gguf.build_gguf_load_config(
                repo_id=mid, policy=policy, slot=slot,
            )
    return _LLMPreflight(
        creative_id=norm_creative, technical_id=norm_technical,
        policy=policy, load_config_by_slot=load_config_by_slot,
    )


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
        policy: Any = None,
        load_config_by_slot: dict | None = None,
    ):
        self.ids = {
            "creative": creative_id,
            "technical": technical_id,
        }
        # S1 platform-portability: the frozen LLMRuntimePolicy threaded
        # into every request_slot call (None = nv50 baseline, resolved
        # by request_slot itself).
        self.policy = policy
        # GGUF row registry (2026-07-16): the immutable per-slot GGUF
        # load_config (gguf slots only) threaded into request_slot -> backend
        # load. Empty for non-GGUF runs (request_slot then uses the policy).
        self.load_config_by_slot = load_config_by_slot or {}
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
        cache_entry = _OTRML.request_slot(
            slot, resolved_id, policy=self.policy,
            load_config=self.load_config_by_slot.get(slot),
        )
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

    def _slot_transport_markers(
        self, slot: str,
    ) -> dict[str, bool | str | None]:
        """Return the declared transport capability for one configured slot.

        ``for_slot`` deliberately defers model acquisition until an actual
        generation call. Its wrapper retains the catalog's structured transport
        behavior: local models expose lazy schema binding and remote providers
        retain their JSON-object capability markers.
        """
        try:
            row = _otr_model_catalog._by_repo_id().get(self.ids[slot])
            provider = str(getattr(row, "provider", "") or "")
        except Exception:  # noqa: BLE001 -- capability is a safe false default
            provider = ""
        return {
            "_otr_local_schema_binding": provider == "local",
            "_otr_openrouter": provider == "openrouter",
            "_otr_comfy_credits": provider == "comfy_credits",
            "_otr_google_api": provider == "google_api",
            "_otr_gguf_native": provider == "gguf_native",
            # Providers whose backend accepts a json_object response_format:
            # invoke_structured_slot forces json_object for a schema-less
            # structured pass on these. OpenRouter (frontier prose->JSON) AND
            # native GGUF (llama-cpp json_object). The local transformers lane
            # is excluded (it has no json_object mode).
            "_otr_supports_json_object": provider in ("openrouter", "gguf_native"),
            # The plain scheduler closure does not bind a schema itself.
            # Local-transformers closures expose `_otr_bind_schema`; the SciFi
            # structured invoker uses it to bind each pass's exact Pydantic
            # result type. OpenRouter/GGUF retain their existing json_object
            # response-format behavior and are intentionally unchanged.
            "_otr_response_format": None,
        }

    def for_slot(self, slot: str):
        """Return a generate_fn closure that targets `slot`. Each call
        ensures the right model is resident before generation fires."""
        if slot not in self._ALLOWED_SLOTS:
            raise ValueError(
                f"_SlotScheduler.for_slot: slot must be one of "
                f"{self._ALLOWED_SLOTS!r}; got {slot!r}"
            )
        scheduler = self

        transport_markers = scheduler._slot_transport_markers(slot)

        def _make_generate_fn(schema_model=None):
            def generate_fn(
                messages, *, temperature, max_new_tokens, stop=None,
                response_format=None,
            ):
                cache_entry = scheduler._account_and_get_entry(slot)
                base = _build_truncating_generate_fn(
                    cache_entry,
                    schema_model=schema_model,
                    **scheduler.sampling,
                )
                kwargs = {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "stop": stop,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                return base(messages, **kwargs)

            for marker, value in transport_markers.items():
                setattr(generate_fn, marker, value)
            if transport_markers["_otr_local_schema_binding"]:
                # Lazy schema binding preserves slot accounting and model
                # transitions while making invalid JSON un-sampleable on the
                # local Transformers lane.
                generate_fn._otr_bind_schema = _make_generate_fn  # type: ignore[attr-defined]
                generate_fn._otr_bound_schema_model = schema_model  # type: ignore[attr-defined]
            return generate_fn

        return _make_generate_fn()

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
    schema_model: Any = None,
):
    """Return a generate_fn that NEVER truncates a prompt.

    The name is historical. This wrapper used to left-slice an oversized prompt;
    it no longer can, and no longer does. The output request is fitted to the
    MEASURED prompt (see below), which makes the input allowance at least the
    prompt's own length by construction, and a prompt with no honest room left
    for an artifact raises ``PromptContextOverflowError`` instead of quietly
    losing its system/schema prefix.

    Closure captures the episode-level sampling knobs from the
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

    The output request is first fitted against the measured prompt. This is
    important for structured passes whose artifact-derived reservation can be
    larger than the local context cap: the old ``context_cap - requested``
    arithmetic reduced the input allowance to 64 tokens and silently deleted
    the contract. A prompt is never truncated merely because its caller asked
    for a generous output ceiling.
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
    if cache_entry.get("provider") == "google_api":
        from ._otr_google_api import llm as _gai_llm
        return _gai_llm.make_google_api_generate_fn(cache_entry)
    # [Local OpenAI] External local server lane for Gemma 4 12B. Same
    # provider-tag dispatch; zero ComfyUI-process VRAM.
    if cache_entry.get("provider") == "gguf_native":
        from . import _otr_gguf_backend as _gguf
        # GGUF row registry (2026-07-16): the native GGUF lane now HONORS the
        # writer's episode sampling widgets (previously discarded, so gemma ran
        # at llama-cpp defaults). top_k stays pinned to GGUF_TOP_K inside the
        # backend; the base seed rides the cache_entry from the preflight
        # load_config. Announced behavior change for all gguf rows.
        return _gguf.make_gguf_generate_fn(
            cache_entry,
            sampling={
                "top_p": top_p,
                "min_p": min_p,
                "repeat_penalty": repetition_penalty,
            },
        )
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
    schema_parser = None
    prefix_allowed_tokens_fn = None
    if schema_model is not None:
        from ._otr_constrained_generate import (
            get_cached_transformers_schema_constraint,
        )

        schema_parser, prefix_allowed_tokens_fn = (
            get_cached_transformers_schema_constraint(
                cache_entry, schema_model,
            )
        )

    def generate_fn(messages, *, temperature, max_new_tokens, stop=None):
        import torch  # local import; never load torch at module import
        from . import _otr_loader_backends as _OTRLB
        require_full_output = bool(getattr(
            messages, "_otr_require_full_output_budget", False,
        ))
        reserve_remaining = bool(getattr(
            messages, "_otr_reserve_remaining_output_capacity", False,
        ))
        bounded_capacity = reserve_remaining and max_new_tokens is not None
        fail_on_output_limit = bool(getattr(
            messages, "_otr_fail_on_output_limit", False,
        ))
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

        input_len = inputs["input_ids"].shape[-1]
        requested_max_new_tokens = (
            context_cap if reserve_remaining else max(1, int(max_new_tokens))
        )
        try:
            effective_max_new_tokens = fit_output_tokens(
                requested_max_new_tokens,
                context_cap=context_cap,
                prompt_tokens=input_len,
                label="prompt",
                require_full=require_full_output or bounded_capacity,
            )
        except GenerationContextOverflowError as exc:
            # A prompt that leaves no honest room for a usable artifact is a
            # hard failure for EVERY caller, not only the ones that opt in via
            # `_otr_prompt_must_fit`: left-truncating it would delete the
            # system/schema prefix and the model would answer from whatever
            # fragment survived. Both arms of the old branch already raised the
            # same error, so the flag decided nothing here; the honest guard is
            # unconditional. `prompt_must_fit` still selects fail-loud behavior
            # at the lane preflights that own an artifact's provenance.
            # A-4: the phase travels with the re-wrap rather than being
            # re-derived. `fit_output_tokens` refuses BEFORE the call, so this
            # is always `prompt_no_room` and the ladder must not re-roll it --
            # but the phase is read off the error, not assumed here, so a
            # future pre-call refusal that IS retryable cannot be mislabelled
            # by this line.
            raise PromptContextOverflowError(
                str(exc), phase=exc.phase,
            ) from exc
        if (not reserve_remaining
                and effective_max_new_tokens != requested_max_new_tokens):
            log.warning(
                "[OTR_LedgerScriptWriter] OUTPUT_BUDGET: requested %d -> %d "
                "tokens (prompt_tokens=%d, context_cap=%d)",
                requested_max_new_tokens, effective_max_new_tokens,
                input_len, context_cap,
            )
        # NO PROMPT TRUNCATION HAPPENS HERE, BY CONSTRUCTION. `fit_output_tokens`
        # returns at most `context_cap - input_len`, so `context_cap -
        # effective_max_new_tokens` is always >= input_len: the old PROMPT_GUARD
        # left-slice could never fire once the output budget was fitted to the
        # MEASURED prompt instead of the REQUESTED ceiling. It is deleted rather
        # than left unreachable -- a dead lever is worse than no lever, because
        # the next reader repairs the branch that never runs (operator, 2026-07-11)
        # and the live defect keeps its hiding place. A prompt that genuinely
        # cannot fit now raises PromptContextOverflowError above.
        gen_kwargs = {
            "do_sample": True,
            "temperature": float(temperature),
            "top_p": active_top_p,
            "max_new_tokens": effective_max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }
        # Only forward non-default values so older transformers
        # versions that don't accept `min_p` as a kwarg keep working
        # silently when the widget is at its disabled default.
        if active_min_p > 0.0 and not _min_p_unsupported[0]:
            gen_kwargs["min_p"] = active_min_p
        if active_rep_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = active_rep_penalty
        if prefix_allowed_tokens_fn is not None:
            # Local structured passes are constrained at token selection:
            # tokens that cannot continue a schema-valid JSON document are
            # never sampleable. Keep one beam; constrained sampling does not
            # benefit from multiplying parser state across beams.
            gen_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
            gen_kwargs["num_beams"] = 1

        # THE LIVENESS GUARD (2026-08-13). Installed UNCONDITIONALLY, and NOT
        # behind `if stop:` -- the structured passes never pass `stop`, which is
        # exactly why nothing watched the decode that ran away for 22 minutes.
        #
        # It is also NOT gated on `schema_model is not None`, though the settled
        # design proposed that. A schema gate installs the guard on the lane
        # that now ALSO has structural string ceilings, and skips the lane with
        # full remaining capacity, no schema and no stop. A liveness contract
        # belongs on every local generate() call; the guard costs nothing on a
        # decode that never opens a long string.
        #
        # CONSTRUCTION FAILURE IS LOUD. The stop-string block below swallows its
        # exception because a missing quality stop is a nice-to-have; a missing
        # liveness guard is a silent removal of protection the log then claims
        # to have. So this one raises.
        from transformers import StoppingCriteriaList  # noqa: I001
        try:
            from ._otr_decode_guard import make_degeneracy_criterion
        except ImportError:  # pragma: no cover - flat/standalone import path
            from _otr_decode_guard import (  # type: ignore
                make_degeneracy_criterion,
            )
        # NO try/except AROUND THE CONSTRUCTION. An r1 panel caught the first
        # version claiming "construction failure must be loud" in a comment
        # while the code quietly set the guard to None and ran without it --
        # a comment describing a fix is not a fix. If the guard cannot be
        # built, that is a broken install and the render must say so.
        # The tokenizer is passed ONLY when a schema is bound, which turns on
        # the second signal: the open-string counter that catches an
        # ELABORATION SPIRAL (a runaway that never repeats -- specimen P2,
        # 15,355 tokens, which the cycle detector is structurally blind to).
        # It reads quotes as STRUCTURE, so it must never run on a free-prose
        # or markup pass where a quotation mark is dialogue.
        _degeneracy_guard = make_degeneracy_criterion(
            inputs["input_ids"].shape[1],
            tokenizer=tokenizer if schema_model is not None else None,
        )
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [_degeneracy_guard]
        )

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
                # APPEND, never assign. This block used to own
                # `stopping_criteria` outright; with the liveness guard
                # installed above, assigning here would silently REMOVE the
                # guard on exactly the calls that also asked for stop strings.
                # StoppingCriteriaList ORs its members, so both signals stay
                # live and the guard's `hit` flag remains the discriminator.
                _criteria = list(gen_kwargs.get("stopping_criteria") or [])
                _criteria.append(
                    _SubstringStop(tokenizer, stop_strings, prompt_len_now)
                )
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList(_criteria)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "[OTR_LedgerScriptWriter] stop-strings disabled: %s",
                    exc,
                )

        # LIVE HEARTBEAT (2026-08-13). This transport had NO streamer, and it is
        # the one that ran away: a P3 prose pass burned its whole 14,191-token
        # allowance without a stop token, three times, ~20 minutes each, and
        # nothing was visible while it happened. Read-only -- generate() hands
        # the streamer sampled token ids and this never feeds any back, so the
        # output is byte-identical with it attached.
        _hb = _OTRHB.make_streamer(tokenizer, "OTR_LedgerScriptWriter")
        if _hb is not None:
            gen_kwargs = dict(gen_kwargs, streamer=_hb)

        _applied_seed = _seed_writer_sampling(inputs)
        if _applied_seed is not None:
            log.info("[OTR_LedgerScriptWriter] sampling seeded (%s) -- this "
                     "pass is reproducible", WRITER_SEED_ENV)

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
                    # Re-seed: the failed attempt above already consumed RNG
                    # state, so without this the retry path would diverge from
                    # a run that never hit the TypeError.
                    _seed_writer_sampling(inputs)
                    out = model.generate(**inputs, **gen_kwargs)
                else:
                    raise
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = out[0][prompt_len:]
        try:
            generated_tokens = int(getattr(generated_ids, "shape", [len(generated_ids)])[-1])
        except Exception:  # pragma: no cover - exotic backend sequence shape
            generated_tokens = None
        ended_with_eos = False
        try:
            last_token = int(generated_ids[-1])
            eos = tokenizer.eos_token_id
            eos_values = {int(value) for value in (
                eos if isinstance(eos, (list, tuple, set)) else (eos,)
            ) if value is not None}
            ended_with_eos = last_token in eos_values
        except Exception:  # pragma: no cover - exotic token container
            pass
        # A-1 (2026-07-30, writer repair): DECODE BEFORE THE RAISE.
        # The output-limit raise used to fire HERE, above the decode, so a
        # fail-closed leg threw away the only copy of what the model actually
        # produced -- the ladder received an exception with no artifact, and the
        # OUTPUT_TRUNCATED / OUTPUT_CAP arithmetic below never printed either,
        # because the raise jumped over it. Whoever debugs a truncation needs
        # the completion AND the arithmetic, and both were unreachable at the
        # one moment they exist. Decoding here costs a decode on a leg that is
        # already dying; the success path decodes exactly once, as before.
        decoded = tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        )

        # CLASSIFY THE HALT FIRST (2026-08-13). Order matters and this is the
        # authoritative sequence from the settled design:
        #   1. guard.hit          -> degeneracy, REGARDLESS of generated length
        #   2. at the ceiling, no EOS -> capacity
        #   3. ended_with_eos     -> clean termination
        #   4. otherwise          -> some other criterion, e.g. a stop substring
        # Degeneracy must be tested BEFORE capacity because a halted decode
        # stops with room to spare -- reading it as anything else would report
        # "ended at the provider capacity limit" about a decode that was
        # deliberately stopped with ~11,000 tokens unspent.
        if _degeneracy_guard is not None and getattr(
            _degeneracy_guard, "hit", False
        ):
            telemetry = _degeneracy_guard.telemetry()
            log.error(
                "[OTR_LedgerScriptWriter] DECODE HALTED (%s): the output "
                "repeated a %s-token run verbatim %s times in a row, after %s "
                "generated tokens of a %d-token allowance. The model did not "
                "run out of room and this is not a long artifact -- it was "
                "CYCLING, and the transport stopped it. REROLLABLE: the "
                "ladder's next rung runs at a lower temperature. Telemetry: %s",
                _degeneracy_guard.reason,
                telemetry.get("cycle_tokens"),
                telemetry.get("required_repeats"),
                generated_tokens, effective_max_new_tokens,
                telemetry,
            )
            # Same evidence discipline as the capacity raise below: the head
            # says what the model was writing, the tail says what it was doing
            # when the guard stopped it. For a degeneracy halt the tail is the
            # whole point -- it is where the loop is visible.
            _halt_raw = decoded or ""
            log.error(
                "[OTR_LedgerScriptWriter] RUNAWAY EVIDENCE (%d chars, %s "
                "tokens, halted)\n  HEAD: %s\n  TAIL: %s",
                len(_halt_raw), generated_tokens,
                _halt_raw[:400].replace("\n", " "),
                _halt_raw[-400:].replace("\n", " "),
            )
            raise GenerationDegeneracyError(
                "generation was halted by the in-decode liveness guard: the "
                "output repeated a run of tokens verbatim, which is a decode "
                "that is cycling rather than a long artifact",
                halt_reason=_degeneracy_guard.reason,
                open_string_tokens=None,
                repetition=telemetry,
                raw_completion=decoded,
                prompt_tokens=prompt_len,
                generated_tokens=generated_tokens,
                requested_output_tokens=requested_max_new_tokens,
                effective_output_tokens=effective_max_new_tokens,
                context_cap=context_cap,
                ended_with_eos=ended_with_eos,
            )

        if generated_tokens == effective_max_new_tokens:
            # The model stopped because it ran OUT OF ROOM, not because it was
            # finished. When the room it was given is also LESS than the room
            # its caller asked for, that is the silent catastrophe: the artifact
            # is cut off mid-JSON and the ladder reports a bare JSONDecodeError
            # three times, naming the model instead of the budget. Say the real
            # cause once, LOUDLY, with the whole arithmetic -- a reader of the
            # leg log must never have to reconstruct it.
            if effective_max_new_tokens < requested_max_new_tokens:
                if reserve_remaining:
                    # THE ADVICE WAS WRONG EXACTLY WHEN IT FIRED (2026-08-13).
                    #
                    # A ProviderCapacityMessages pass sets
                    # _otr_reserve_remaining_output_capacity, so requested ==
                    # the whole context window BY DESIGN, and
                    # effective < requested is true the moment the prompt is
                    # non-empty. The old text told the reader to "give this
                    # pass a slot whose window fits prompt+artifact" -- but
                    # this pass already HAS every token there is, so there is
                    # no bigger slot to give it and no config defect to find.
                    # It sent a live session hunting one for twenty minutes.
                    # When the pass reserved everything and still hit the
                    # ceiling without an EOS, the model did not stop.
                    log.error(
                        "[OTR_LedgerScriptWriter] OUTPUT_TRUNCATED: this pass "
                        "reserved ALL remaining output capacity (%d of the "
                        "%d-token window, after a %d-token prompt) and still "
                        "ran to the ceiling. THE MODEL DID NOT STOP -- there "
                        "is no larger slot to move it to, so do not go looking "
                        "for one. Any JSON parse failure below is a runaway "
                        "decode, not a budget defect.",
                        effective_max_new_tokens, context_cap, prompt_len,
                    )
                else:
                    log.error(
                        "[OTR_LedgerScriptWriter] OUTPUT_TRUNCATED: generation "
                        "stopped at the ceiling after a CLAMP. The caller asked "
                        "for %d output tokens; the %d-token context window left "
                        "only %d after a %d-token prompt. Any JSON parse failure "
                        "below is this budget, not the model. Give this pass a "
                        "slot whose window fits prompt+artifact.",
                        requested_max_new_tokens, context_cap,
                        effective_max_new_tokens, prompt_len,
                    )
            else:
                log.warning(
                    "[OTR_LedgerScriptWriter] OUTPUT_CAP: generation stopped at "
                    "the caller's own ceiling (prompt_tokens=%d "
                    "generated_tokens=%d max_new_tokens=%d); output may be "
                    "truncated.",
                    prompt_len, generated_tokens, effective_max_new_tokens,
                )
        if (generated_tokens == effective_max_new_tokens
                and fail_on_output_limit and not ended_with_eos):
            # LOG THE EVIDENCE BEFORE DISCARDING IT (2026-08-13).
            #
            # `raw_completion=decoded` below has been attached to this exception
            # since A-1 and NOTHING has ever read it -- the leg log prints
            # "raw head: <empty>". So at the one moment thousands of tokens of
            # runaway text exist in memory, they are thrown away, and the next
            # reader has to reproduce a 20-minute decode to learn what the model
            # was actually saying. Two runaways in one night were diagnosed by
            # inference for exactly this reason.
            #
            # Head AND tail, because they answer different questions: the head
            # says what the model was writing, the tail says what it was doing
            # when it ran out of room. A verbatim loop or digit run in the tail
            # means degeneracy; varied run-on prose means it was hedging and
            # could not find a way to end the sentence. That distinction decides
            # whether the cure is a decode guard or the pack's own wording, and
            # it is one log line away.
            _raw = decoded or ""
            _head = _raw[:400].replace("\n", " ")
            _tail = _raw[-400:].replace("\n", " ")
            log.error(
                "[OTR_LedgerScriptWriter] RUNAWAY EVIDENCE (%d chars, %d "
                "tokens, ended_with_eos=%s)\n  HEAD: %s\n  TAIL: %s",
                len(_raw), generated_tokens, ended_with_eos, _head, _tail,
            )
            raise PromptContextOverflowError(
                "prose generation exhausted the full remaining provider/context "
                f"capacity ({effective_max_new_tokens} output tokens after a "
                f"{prompt_len}-token prompt); the partial artifact is discarded, "
                "never repaired as prose",
                # A-4: THIS is the phase a re-roll can actually fix -- the call
                # RAN, and sampling is stochastic (nine engines in the live
                # 45-word campaign produced both a pass and a fail on
                # byte-identical code). The message lost its old tail, "not
                # eligible for a prose or structural reroll", because A-4 makes
                # the second half of that false: the ladder may now re-roll
                # this pass. What stays true, and stays said, is that the
                # partial artifact is never handed to a prose repair. Every
                # OTHER transport's capacity refusal carries no phase, so it
                # stays terminal and its own message stays accurate.
                phase=CAPACITY_PHASE_OUTPUT_LIMIT,
                raw_completion=decoded,
                prompt_tokens=prompt_len,
                generated_tokens=generated_tokens,
                requested_output_tokens=requested_max_new_tokens,
                effective_output_tokens=effective_max_new_tokens,
                context_cap=context_cap,
                ended_with_eos=ended_with_eos,
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

    if schema_model is not None:
        generate_fn.schema_model = schema_model  # type: ignore[attr-defined]
        generate_fn.json_schema_parser = schema_parser  # type: ignore[attr-defined]
        generate_fn.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn  # type: ignore[attr-defined]
    return generate_fn


# ---------------------------------------------------------------------------
# Pure helpers (testable without model load)
# ---------------------------------------------------------------------------


try:                                            # package import
    from . import _otr_episode_budget as _OTRB
except ImportError:                              # flat/script import
    import _otr_episode_budget as _OTRB          # type: ignore

#: Set to an integer to make the writer's SAMPLING reproducible. Unset (the
#: default, and production) leaves generation exactly as it was.
WRITER_SEED_ENV = "OTR_WRITER_SEED"


def _seed_writer_sampling(inputs) -> "int | None":
    """Seed torch from (OTR_WRITER_SEED, this prompt) before a generate call.

    WHY THIS EXISTS. ``gen_kwargs`` carries ``do_sample=True`` with no seed and
    no generator, so two runs of the same episode inputs produce DIFFERENT
    scripts. That is correct for production -- every episode should be its own
    -- but it makes a controlled comparison impossible: on 2026-08-22 four
    separate attempts to judge a video change all collapsed because each leg
    wrote a different story, and the operator called it himself: "its a
    different story so not a good comparison".

    KEYED ON THE PROMPT, NOT A CALL COUNTER. An episode makes many generate
    calls; a counter would make the seed depend on call ORDER, so any pass that
    ran conditionally would shift every later seed and the reproduction would
    silently drift. Hashing the actual input tokens makes each call's seed a
    function of what that call is being asked -- order-independent, and equal
    across two runs whose prompts match.

    Returns the seed applied, or None when the variable is unset or unusable.
    Never raises: a bake-off convenience may not be able to break a render.
    """
    raw = os.environ.get(WRITER_SEED_ENV, "").strip()
    if not raw:
        return None
    try:
        base = int(raw)
    except (TypeError, ValueError):
        log.warning(
            "[OTR_LedgerScriptWriter] %s=%r is not an integer; sampling stays "
            "unseeded for this run", WRITER_SEED_ENV, raw)
        return None
    try:
        import zlib

        import torch as _torch

        ids = inputs["input_ids"]
        digest = zlib.crc32(str(ids.tolist()).encode("utf-8"))
        seed = (base * 1_000_003 + digest) & 0x7FFF_FFFF
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        return seed
    except Exception as exc:  # noqa: BLE001 -- never break a render for this
        log.warning(
            "[OTR_LedgerScriptWriter] could not seed sampling (%s); the run "
            "continues UNSEEDED and is not reproducible", exc)
        return None


def _fetch_rss_seed_or_die(
    model_id: str, *, load_config=None, policy=None,
    receipt_sink: dict[str, Any] | None = None,
) -> dict:
    """Run the story_orchestrator RSS fetcher and return the article dict.

    Lifts the exact path the legacy writer used; if the fetcher returns
    None (every feed failed) we raise loudly -- the legacy writer behaved
    the same. Style-engine consolidation (2026-07-05): the old style ->
    slug normalization (with a hardcoded "mission_control_procedural"
    fallback) is removed -- the fetch/rerank chain is style-agnostic now,
    there is no style value yet at this pre-contract sourcing stage.

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
        from . import story_orchestrator as _so
    except ImportError:
        import story_orchestrator as _so  # type: ignore
    try:
        news = _so._fetch_science_news(
            max_feeds=10, model_id=model_id,
            load_config=load_config, policy=policy,
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
        payload = {
            "headline":  (article.get("headline") or "").strip(),
            "summary":   (article.get("summary") or "").strip(),
            "full_text": (article.get("full_text") or "").strip(),
            "source":    (article.get("source") or "").strip(),
            "date":      (article.get("date") or "").strip(),
            "link":      (article.get("link") or "").strip(),
            "seed_text": seed_text,
        }
        if receipt_sink is not None:
            selected_body = payload["full_text"]
            receipt_sink.clear()
            receipt_sink.update({
                "headline": payload["headline"][:240],
                "source": payload["source"][:120],
                "url": payload["link"],
                "date": payload["date"],
                "body_chars": len(selected_body),
                "body_source": str(article.get("_body_source") or ""),
                "rss_content_index": article.get("_rss_content_index"),
                "rss_content_count": int(
                    article.get("_rss_content_count") or 0
                ),
                "body_bytes_utf8": len(selected_body.encode("utf-8")),
                "body_sha256": hashlib.sha256(
                    selected_body.encode("utf-8")
                ).hexdigest(),
                "selected_at": datetime.now().isoformat(timespec="seconds"),
            })
        return payload
    except _so._LLMTimeoutWorkflowPause:
        # 2026-08-25 (PBUG-20260825-04 follow-up): this subtype's own
        # docstring says a timed-out LLM phase must halt the queue so the
        # next stage never races the orphan worker's still-running CUDA
        # kernels -- but the broad `except Exception` below used to catch
        # it here too, one level above story_orchestrator's own two
        # NewsCuration/NewsCurationDeep call sites (already fixed), and
        # recast it as a generic "RSS fetch failed" RuntimeError. That
        # erased the exception's TYPE, so the pause never actually reached
        # the node boundary in the real call chain
        # (_fetch_rss_seed_or_die -> _otr_source_payload's science_rss
        # wrapper, which has no try/except of its own). Re-raise
        # unconditionally so it keeps propagating.
        raise
    except Exception as exc:
        # Loud raise: the writer requires a real seed to function. The
        # workflow can override via custom_premise if RSS is unavailable.
        raise RuntimeError(
            f"[OTR_LedgerScriptWriter] RSS fetch failed: {exc}. "
            f"Type a non-empty value into the `custom_premise` widget to "
            f"bypass the RSS pipeline.",
        ) from exc


def _upstream_identity_names(meta: dict) -> list[str]:
    """Names an upstream creative pass invented for this episode's people.

    Bug Bible 11.61: the cast assigner is about to override these, so they are
    SUPERSEDED and must not reach a per-record prompt. Read only from the two
    lane-neutral structured surfaces: the original lane's
    ``meta.source_meta.selected_concept.cast[].name`` and a source interpreter's
    optional ``meta.news.upstream_identity_names``. A free-text brief cannot be
    mined for identities without firing on healthy Title-Case occupation heads
    ("Film Historian", "Space Force Liaison"), a measured false-positive class.

    Returns [] on every lane that records neither surface, which makes the whole
    mechanism inert there and keeps those prompts byte-identical. Names are
    de-duplicated case-insensitively while preserving first-seen order.
    """
    concept = ((meta or {}).get("source_meta") or {}).get("selected_concept") or {}
    out: list[str] = []
    seen: set[str] = set()

    def _append(value: object) -> None:
        name = str(value or "").strip()
        folded = name.casefold()
        if name and folded not in seen:
            seen.add(folded)
            out.append(name)

    for entry in (concept.get("cast") or []):
        if isinstance(entry, dict):
            _append(entry.get("name"))

    news = (meta or {}).get("news") or {}
    identities = (
        news.get("upstream_identity_names") if isinstance(news, dict) else []
    )
    if isinstance(identities, list):
        for identity in identities:
            if isinstance(identity, str):
                _append(identity)
    return out


def _stamp_news_seed_receipt(
    meta: dict[str, Any],
    resolved: Mapping[str, Any],
) -> None:
    """Promote the selected RSS receipt onto the actual episode ledger."""
    receipt = resolved.get("news_seed_receipt")
    if not receipt:
        return
    if not isinstance(receipt, Mapping):
        raise RuntimeError("news_seed_receipt must be a mapping")
    article = resolved.get("news_article")
    if not isinstance(article, Mapping):
        raise RuntimeError("news_seed_receipt requires a news_article mapping")
    body = str(article.get("full_text") or "")
    expected = {
        "headline": str(article.get("headline") or "")[:240],
        "source": str(article.get("source") or "")[:120],
        "url": str(article.get("link") or ""),
        "date": str(article.get("date") or ""),
        "body_chars": len(body),
        "body_bytes_utf8": len(body.encode("utf-8")),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"news_seed_receipt {key} does not match selected article"
            )
    body_source = receipt.get("body_source")
    if body_source not in {
        "rss_full", "url_scrape", "summary_fallback", "summary_only",
    }:
        raise RuntimeError("news_seed_receipt has an invalid body_source")
    count = receipt.get("rss_content_count")
    index = receipt.get("rss_content_index")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or (
            index is not None
            and (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                or index >= count
            )
        )
    ):
        raise RuntimeError("news_seed_receipt has invalid RSS coordinates")
    if not str(receipt.get("selected_at") or ""):
        raise RuntimeError("news_seed_receipt selected_at is empty")
    meta["news_seed"] = dict(receipt)


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


# (lean-mean 2026-08-23) `_build_cast_rows` was here and is DELETED. It built
# LEGACY-schema cast rows (char_id + seven mostly-None fields) from a list of
# names and had ZERO callers repo-wide. The live cast path is
# `nodes/_otr_casting.lock_cast` via `nodes/cast_lock.py`, which builds richer
# rows and is what every episode actually uses.


# The pipeline -> lane authority moved OUT of this file to
# `nodes/_otr_lane_specs.py` (2026-07-31). `_run_scifi_news_pro_lane`,
# the per-lane wrappers, `_RUNNER_BY_PIPELINE`, `_LEGACY_INLINE_PIPELINES`
# and `_resolve_lane_runner` all lived here; they are GONE, not aliased.
# Lazy runner import is unchanged -- `_LANES.runner_for()` resolves the
# module by name at dispatch time, exactly as the old wrappers did.
# Use `_LANES.is_dispatched(pipeline_id)` for membership and
# `_LANES.runner_for(pipeline_id)` for the callable (None = inline body).


def _make_original_interpreter(*, creative_fn, resolved, meta):
    """Interpreter-SHAPED adapter for the original_radio creative front.

    Returned callable matches the resolve_interpreter contract
    (bank=, payload=, technical_fn=, model_id= -> briefs-like), so the
    writer's D.2 stamping path (validate_interpreter_result +
    casting/script/key_terms unpack + the science halt surface) runs
    byte-identically for both lanes. The payload arg is accepted and
    ignored: the A-branch synthesized it from the spark digest; the
    creative front reads the ATOMS from resolved["source_meta"].

    Side effect (kibitz r3 D3): after the front returns, the sidecar
    delta {selected_concept, pitches, selection_rationale, model_ids} is
    merged into BOTH resolved["source_meta"] and meta["source_meta"] --
    meta was COPIED from resolved at D.1, so a resolved-only mutation
    would never reach the durable ledger.

    Failure posture: OriginalBriefsError / StructuredCallFailedError
    propagate PAST the science degrade except (which catches only
    SourceInterpretError) -- this lane hard-fails, no degrade, no
    news_briefs_required lever.
    """
    def _original_interpret(*, bank, payload, technical_fn, model_id):
        try:
            from . import _otr_original_radio as _OTROR
        except ImportError:  # pragma: no cover -- flat test/standalone load
            import _otr_original_radio as _OTROR  # type: ignore
        source_meta = dict(resolved.get("source_meta") or {})
        spark_atoms = dict(source_meta.get("spark_atoms") or {})
        if not spark_atoms:
            raise _OTROR.OriginalBriefsError(
                "original lane: resolved['source_meta']['spark_atoms'] "
                "missing -- the _resolve_inputs A-branch did not run"
            )
        briefs, delta = _OTROR.build_original_briefs(
            spark_atoms=spark_atoms,
            num_characters=int(resolved["num_characters"]),
            creative_fn=creative_fn,
            technical_fn=technical_fn,
            creative_model_id=str(resolved["creative_writing_model"]),
            technical_model_id=str(model_id),
            pack=_otr_story_routing.resolve_story_pack(bank.source_bank_id),
            operator_hint=str(source_meta.get("operator_hint") or ""),
        )
        source_meta.update(delta)
        resolved["source_meta"] = source_meta
        meta["source_meta"] = dict(source_meta)
        return briefs
    return _original_interpret


def _run_source_interpreter(
    *,
    interpreter,
    bank,
    payload,
    source_meta,
    technical_fn,
    technical_model_id: str,
    slot_scheduler,
    meta,
):
    """Run one bank interpreter with a deterministic same-source fallback.

    Each interpreter retains its own bounded structured-output repair ladder.
    If model-authored output exhausts that ladder, a validated bank-specific
    brief is derived from the same source. Backend, configuration, I/O, and
    contract defects still propagate.
    """
    try:
        with slot_scheduler.helper_context("build_news_briefs"):
            briefs = interpreter(
                bank=bank,
                payload=payload,
                technical_fn=technical_fn,
                model_id=str(technical_model_id),
            )
    except _otr_source_payload.SourceInterpretError as exc:
        used = _otr_source_payload.interpreter_exhaustion_attempts(exc)
        if used <= 0:
            raise
        reason = " ".join(str(exc).split())[:700]
        fallback = _otr_source_payload.build_source_interpreter_fallback(
            bank=bank,
            payload=payload,
            source_meta=source_meta,
            attempts=used,
            failure_reason=reason,
        )
        meta["source_interpreter"] = {
            "schema_version": "source_interpreter_v2",
            "status": "deterministic_same_source_fallback",
            "model_calls": used,
            "model": fallback.model_id,
            "reason": reason,
        }
        log.warning(
            "[OTR_LedgerScriptWriter] interpreter output exhausted; "
            "using the deterministic same-source fallback: %s",
            reason,
        )
        return fallback

    used = max(1, int(getattr(briefs, "attempts", 1) or 1))
    meta["source_interpreter"] = {
        "schema_version": "source_interpreter_v2",
        "status": "accepted",
        "model_calls": used,
        "model": str(technical_model_id),
    }
    return briefs


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


def _compose_and_stamp_announcer_close(
    led,
    meta,
    *,
    first_announcer_id,
    last_announcer_id,
    provenance_owned,
    style_grammar_on,
    effective_spoken_fact,
    nc_brief,
    script_brief,
    premise,
    resolved,
    slot_scheduler,
    creative_generate_fn,
):
    """Compose the announcer close, stamp its receipts, and hand back the result.

    EXTRACTED VERBATIM (B4, 2026-08-07) so tests can exercise the PRODUCTION
    reader. The 2026-08-04 attempt at the citation defect edited
    ``spoken_coda_line()``, which had zero callers, and thirty episodes leaked
    after it "landed" -- a fix applied to a function nobody calls is not a fix.

    WHAT STAYS IN THE CALLER, and why:

    * ``news_meta`` -- defined ABOVE this block and read BELOW it by the
      key-term audit. Moving it here is a ``NameError`` on every episode.
    * ``led.save()`` and the flag log -- the caller persists, so this helper
      stays in-memory and therefore testable without touching disk. The caller
      needs the composed result for its log line, which is why this RETURNS
      rather than mutating and going quiet.
    * the ``last_announcer_id``/``first_announcer_id`` guard itself.

    ``script_brief`` is a parameter for the same reason: the fictional-outro
    branch passes it to ``compose_announcer_outro`` and it is bound far outside
    this range.

    Returns the ``_OTRLC.LineResult`` that was written. Every path through this
    body binds it -- there is a ``raise`` in the closed-vocabulary check, but no
    path REACHES the end without ``outro_res`` set.
    """
    # LATE IMPORTS, mirroring ``run()``'s own "no GPU / no model loads at module
    # import" contract at :3458. Hoisting these to module scope would break that
    # deliberately -- and would be the same defect the lazy-import test guards.
    from . import _otr_line_composer as _OTRLC
    from . import _otr_ledger as _OTRL

    # Named locals the extracted body already used, rebound here so the block
    # below stays byte-identical to what ran inline.
    _style_grammar_on = style_grammar_on
    _premise = str(premise or "")

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
    # WHICH FACT IS SPOKEN, AND BY WHICH ROUTE (2026-08-05).
    # A provenance-owned lane ALWAYS takes the deterministic append,
    # independent of _style_grammar_on. It cannot use the fictional-outro
    # route below: that one passes the note as LLM *context* and returns
    # whatever the model authors, so a receipt claiming the deterministic
    # coda was spoken would simply be untrue.
    if provenance_owned:
        _spoken_fact_for_coda = effective_spoken_fact
    elif _style_grammar_on:
        _spoken_fact_for_coda = nc_brief.strip()
    else:
        _spoken_fact_for_coda = ""

    if _spoken_fact_for_coda:
        _spoken_coda_source = (
            "provenance" if provenance_owned else "news_close_brief"
        )
        with slot_scheduler.helper_context("compose_news_coda"):
            outro_res = _OTRLC.compose_news_coda(
                creative_fn=creative_generate_fn,
                news_close_brief=_spoken_fact_for_coda,
                premise=_premise,
                intro_text=intro_text,
                creative_repo_id=resolved["creative_writing_model"],
                # The bank was never passed here, so every lane resolved
                # media_archive's coda_system prompt -- while the
                # fictional-outro call below has passed it since Stage 4.
                source_bank_id=resolved["source_bank"],
            )
        if not outro_res.text:
            # Pathological (brief cleaned to empty) -- never ship an empty
            # close. Deterministic news outro, LOUD.
            log.warning(
                "[OTR_LedgerScriptWriter] news coda produced no text "
                "(fact=%r); using the deterministic outro fallback",
                _spoken_fact_for_coda,
            )
            # The factual note remains in meta. Never route it through
            # the older character-truncating outro template, and do not
            # mark this generic close as a protected coda that later
            # repair could mistake for a fact-bearing row.
            #
            # This said "remains in meta/credits" unconditionally, and
            # stamped the deferred-to-credits flag to match -- but the
            # printed credit line can itself be empty, in which case both
            # asserted an attribution surface that does not exist. The
            # flag is now only raised when the credits really carry it.
            _deferred_to_credits = bool(
                str(meta.get("credits_source_line") or "").strip()
            )
            outro_res = _OTRLC.LineResult(
                text=_OTRLC.fallback_announcer_outro(""),
                compose_flags=(
                    ("announcer_outro_fallback",)
                    + (("source_note_deferred_to_credits",)
                       if _deferred_to_credits else ())
                    + ("spoken_surface_emergency",)
                ),
            )
            _spoken_coda_source = "none"
    elif provenance_owned:
        # Owned, but the normalizer produced no spoken line (unknown
        # status, or a licensed source with no label). The episode speaks
        # NO attribution and print carries it -- credits_source_line and
        # noncommercial_notice are untouched. Never fall back to the raw
        # interpreter note here: that is the leak, on the path most likely
        # to reach it. Neither composer is entered, which also avoids
        # compose_announcer_outro's empty-briefs RuntimeError.
        _spoken_coda_source = "none"
        # The "deferred to credits" flag is only TRUE when the credits
        # actually carry the line. Stamping it unconditionally would
        # assert an attribution surface that may not exist -- the flag
        # would say print has it while nothing does, which is the same
        # class of false receipt this whole item is correcting.
        _deferred_to_credits = bool(
            str(meta.get("credits_source_line") or "").strip()
        )
        log.warning(
            "[OTR_LedgerScriptWriter] provenance-owned lane has no spoken "
            "coda (status=%r); generic sign-off, printed credit %s",
            str((meta.get("provenance") or {}).get("status") or ""),
            "carries the attribution" if _deferred_to_credits
            else "is ALSO empty -- this episode attributes nowhere",
        )
        outro_res = _OTRLC.LineResult(
            text=_OTRLC.fallback_announcer_outro(""),
            compose_flags=(
                ("announcer_outro_fallback",)
                + (("source_note_deferred_to_credits",)
                   if _deferred_to_credits else ())
            ),
        )
    else:
        _spoken_coda_source = "none"
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
                source_bank_id=resolved["source_bank"],  # Stage 4
            )
        if _style_grammar_on:
            # On-flag but no news brief -> mark it (text unchanged; frozen).
            import dataclasses as _dc
            outro_res = _dc.replace(
                outro_res,
                compose_flags=outro_res.compose_flags + ("news_coda_no_brief",),
            )
    # WHICH FACT WAS SPOKEN -- the receipt a corpus audit can join on.
    # Closed vocabulary, validated at write time so a typo cannot drift
    # into the ledger and quietly widen what the audit accepts.
    if _spoken_coda_source not in _SPOKEN_CODA_SOURCES:
        raise ValueError(
            "OTR_LedgerScriptWriter: spoken_coda_source %r is not one of "
            "%s -- the receipt vocabulary is closed on purpose"
            % (_spoken_coda_source, sorted(_SPOKEN_CODA_SOURCES))
        )
    meta["spoken_coda_source"] = _spoken_coda_source
    # news_coda_emitted describes what was SPOKEN, not what the
    # interpreter wrote, and is stamped outside the style gate because a
    # provenance-owned lane bypasses that gate entirely.
    if provenance_owned or _style_grammar_on:
        meta["news_coda_emitted"] = _spoken_coda_source != "none"
        # DID THE NEWS CODA DEGRADE? Two routes count, and only two:
        #   news_coda_fact_only     -- the bridge failed validation, so the
        #                              fact shipped bare (compose_news_coda).
        #   spoken_surface_emergency -- the brief cleaned to empty, so the
        #                              deterministic outro shipped instead.
        # Deliberately EXCLUDED: news_coda_no_brief (there was no coda to
        # degrade) and announcer_outro_structural_fallback (the fictional-outro
        # route -- a different lane, and folding it in here would make one
        # boolean mean two different things depending on which route ran).
        # This tested for a "news_coda_fallback" string no composer has ever
        # emitted, so the receipt was permanently False.
        meta["news_coda_fallback"] = bool(
            {"news_coda_fact_only", "spoken_surface_emergency"}
            .intersection(outro_res.compose_flags)
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
    # `news_coda_spoken_reduction` was stamped here and is DELETED (2026-08-23).
    # It was gated on `news_coda_fact_reduced` / `news_coda_fact_deferred_to_credits`
    # -- two compose flags that are READ here and SET NOWHERE in the tree, so the
    # branch never fired and the receipt was never written on any episode. The
    # else-arm popped a key nothing had put there. Its reader count was zero too,
    # so no downstream consumer loses a field: this is the same shape as the
    # `news_coda_fallback` receipt above it, which tested for a string "no
    # composer has ever emitted".

    return outro_res

class OTR_LedgerScriptWriter(WriterTailMixin):
    """v2.0 LPL script writer with legacy-style widget surface.

    Wires the four shipped LPL modules (_otr_outline, _otr_canon,
    _otr_line_composer, _otr_model_loader) plus production_ledger
    into the legacy 4-slot output contract. Widget set restored 2026-05-10
    so users get back episode_title / target_words / num_characters /
    creativity / target_length / model controls. Style is no longer a
    widget -- it comes from the single deterministic engine call
    (style-engine consolidation, 2026-07-05).
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
        _google_slot_a_choices = _otr_model_catalog.google_api_catalog_dropdown_choices("a")
        _google_slot_b_choices = _otr_model_catalog.google_api_catalog_dropdown_choices("b")
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
                # `target_words` WAS WIDGET SLOT 2 AND WAS REMOVED
                # 2026-08-14 (operator directive). Episode length is an
                # OBSERVATION now, not an instruction: pick the number of
                # acts and the story is as long as it turns out to be.
                # `widgets_values` is POSITIONAL (BUG-LOCAL-097), so this
                # removal shifted every saved value after slot 2 --
                # `workflows/otr_canonical.json` and all variants were
                # regenerated in the SAME change. A graph saved before that
                # change must be re-saved.
                "num_characters": ("INT", {
                    "default": 2, "min": 1, "max": _FABLE2_MAX_CAST, "step": 1,
                    "tooltip": (
                        "REQUESTED number of speaking characters (plus "
                        "ANNOUNCER bookends). 1 = monologue/diary mode. This "
                        "is a request, not a cap: a story that genuinely needs "
                        "another voice may use one. The real ceiling is the "
                        "voice stock, because two characters never share a "
                        "voice."
                    ),
                }),
            },
            "optional": {
                # S30 B2a: single model_id widget replaced by two slots.
                # The catalog dropdown_choices() call scans the local HF
                # cache live and applies display-only suffixes such as
                # [LOCAL HF], [LOCAL GGUF], and [NOT DOWNLOADED]. Labels are
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
                            "Suffix tags like [LOCAL HF], [LOCAL GGUF], "
                            "and [NOT DOWNLOADED] are "
                            "stripped before HF lookup. To use a remote "
                            "OpenRouter model, set OPENROUTER_API_KEY and "
                            "pick OpenRouter A/B (see "
                            "https://github.com/jbrick2070/"
                            "ComfyUI-OldTimeRadio/blob/v2.0-alpha/docs/"
                            "openrouter-setup.md)."
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
                            "Profile/platform-owned baseline: your hardware "
                            "profile pins it, and a direct headless -Set "
                            "override is sanctioned and wins over the "
                            "profile when supplied. Pick a smaller model "
                            "here when you want Slot 1 != Slot 2 routing "
                            "for VRAM headroom. To use a remote OpenRouter "
                            "model, set OPENROUTER_API_KEY and pick "
                            "OpenRouter A/B (see "
                            "https://github.com/jbrick2070/"
                            "ComfyUI-OldTimeRadio/blob/v2.0-alpha/docs/"
                            "openrouter-setup.md)."
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
                # THE ONE LENGTH-SHAPED KNOB (operator directive
                # 2026-08-14). 'auto' was removed with target_words --
                # it meant "derive the act count from the word total" --
                # and the range grew from 1-7 to 1-8. Whatever is picked
                # here is honoured: there is no derived floor or ceiling
                # that can refuse it.
                # NARROWED 1-8 -> 1-6 (PBUG-20260825-01, operator decision,
                # 2026-08-25): 7 and 8 were reachable but mathematically
                # guaranteed to fail at Outline construction (36/41 beats
                # against Outline.beats' own max_length=32) -- not a
                # "refuse it" ceiling, a bound the schema already had that
                # this range simply now agrees with.
                "act_count": (
                    _ACT_COUNT_CHOICES,
                    {
                        "default": str(_DEFAULT_ACT_COUNT),
                        "tooltip": (
                            "Number of acts, 1-6. This is the only knob "
                            "that shapes episode length, and your pick "
                            "is always honoured.\n\n"
                            "More acts means a story with more turns in "
                            "it -- each act gets its own beat skeleton "
                            "and its own pass. The episode ends up as "
                            "long as the story needs; length is reported "
                            "afterwards, never requested up front.\n\n"
                            "  1 -> a single scene\n"
                            "  2 -> setup, resolution\n"
                            "  3 -> setup, complication, resolution\n"
                            "  6 -> the full arc, through crisis and climax"
                        ),
                    },
                ),
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
                        "DEPRECATED 2026-08-08 -- NO-OP sentinel. "
                        "Widget preserved to keep positional layout "
                        "stable (BUG-LOCAL-097). Formerly triggered "
                        "OTR_RTXUpscale's per-episode intermediate "
                        "cleanup, which was retired with the RTX-VSR "
                        "node (queue item 8; nodes/rtx_upscale.py "
                        "removed). Setting this True has no effect."
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
                        "Attempt-1 uses this cap directly; attempt-2 "
                        "retry uses the full cap. It is a per-CALL decode "
                        "budget, not a length target -- it was scaled from "
                        "target_words until 2026-08-14, which made it a "
                        "token ceiling derived from a word request."
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
                            "LEMMY easter-egg cameo -- the genial Cockney "
                            "communications officer who occasionally joins "
                            "the cast.\n\n"
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
                # Observes speaker leaks, banned phrases, length drift,
                # pronoun mismatches and on-beat misses on the rendered
                # text BEFORE the ledger is frozen, and stamps what it
                # finds on meta.lines[].validation_findings for audit.
                #
                # IT NEVER REGENERATES A LINE, and the widget said it did
                # until 2026-08-15. The promise was written in an era that
                # ended twice over: THE LAW (2026-07-22) forbids failing or
                # rerolling a story for length, language or style, which is
                # most of what these validators report; and the one finding
                # class that IS a real defect -- a character speaking
                # another's lines -- was built as an attribution repair,
                # lab-measured on 2026-08-14 at 3/6 then 1/6 recall on
                # identical fixtures, and shipped disabled for being too
                # unstable to hand a rewrite. Telemetry is the correct
                # behaviour here; the tooltip was the defect.
                #
                # Default OFF so the legacy PD1 byte-identity contract holds
                # out-of-the-box; flip ON for production smokes.
                "enable_production_stage3_validators": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "OFF (default) preserves PD1 byte-identity on the "
                        "legacy path -- no validators run. ON wires Stage 3 "
                        "validators (speaker-leak, banned-phrase, length, "
                        "pronoun, on-beat) into the production compose_line "
                        "for every character dialogue beat. TELEMETRY ONLY: "
                        "findings are stamped on "
                        "meta.lines[].validation_findings and NOTHING is "
                        "regenerated, rerolled or rejected -- an audit may "
                        "never fail a story for length, language, style or "
                        "quality. Costs no extra LLM call at any severity. "
                        "Flip ON for production smokes; OFF for the "
                        "byte-identity regression run."
                    ),
                }),
                # Sprint 2.2 (2026-05-28) -- Jeffrey 2026-05-27
                # directive: when build_news_briefs exhausts its
                # Keep the positional widget for workflow compatibility. The
                # bounded quality chain now handles malformed/rejected model
                # briefs and always stamps a validated source floor at its
                # ceiling. This switch governs only the legacy non-quality
                # SourceInterpretError branch.
                "news_briefs_required": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ON (default): typed non-quality source-interpreter "
                        "failures stay fail-loud. OFF: the legacy branch may "
                        "degrade to raw news_seed. Rejected/malformed LLM "
                        "briefs do not reach this switch: they rotate through "
                        "fresh technical/creative repair passes and end at a "
                        "validated bank-specific source floor."
                    ),
                }),
                # S2 (2026-06-01): the two OpenRouter slot-slug pickers,
                # APPENDED at the END of optional so the existing widget order
                # is untouched. They land at widgets_values[17]/[18] -- verified
                # against workflows/otr_canonical.json on 2026-08-07. (This
                # comment said [19]/[20] until then; a wrong positional claim in
                # a POSITIONAL widgets_values system is a trap, so it is stated
                # here only because it was re-verified.) PASSIVE: a pick here
                # binds a real slug to openrouter:slot-a/b but does NOT activate
                # remote -- it is used only when creative_writing_model /
                # technical_model selects that handle. Choices come from the S0
                # disk cache plus the curated aliases (network-free);
                # remote-disabled shows the "(enable OpenRouter)" sentinel.
                "openrouter_slot_a_model": (
                    _slot_a_choices,
                    {
                        "default": _slot_a_choices[0],
                        "tooltip": (
                            "OpenRouter model slug bound to the "
                            "'openrouter:slot-a' handle (the creative slot). "
                            "Passive: only used when creative_writing_model "
                            "is set to 'openrouter:slot-a'. Choices are the "
                            "curated '~family-latest' aliases (which resolve "
                            "upstream, so they never go stale) plus your "
                            "favourites from the cached catalog; run the "
                            "refresh script or set OTR_OPENROUTER_FULL_CATALOG=1 "
                            "to browse every cached slug. Shows "
                            "'(enable OpenRouter)' until OPENROUTER_API_KEY is "
                            "set. A saved slug is preserved even if absent "
                            "from a stale cache. See "
                            "https://github.com/jbrick2070/"
                            "ComfyUI-OldTimeRadio/blob/v2.0-alpha/docs/"
                            "openrouter-setup.md."
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
                            "to 'openrouter:slot-b'. Choices are the curated "
                            "'~family-latest' aliases plus your favourites "
                            "from the cached catalog; shows "
                            "'(enable OpenRouter)' until remote is enabled. "
                            "OTR_OPENROUTER_SLOT_B_REQUIRE_JSON=1 limits the "
                            "CACHED CATALOG rows to structured-output models; "
                            "the curated aliases are policy, not catalog "
                            "discovery, so they are still offered -- check the "
                            "family before binding one here. See "
                            "https://github.com/jbrick2070/"
                            "ComfyUI-OldTimeRadio/blob/v2.0-alpha/docs/"
                            "openrouter-setup.md."
                        ),
                    },
                ),
                # Comfy Credits slot-slug pickers (2026-06-01), APPENDED at the
                # END of optional so the existing widget order is untouched.
                # They land at widgets_values[19]/[20] -- re-verified against
                # workflows/otr_canonical.json on 2026-08-07. (This said
                # [21]/[22] until then: the SAME stale-positional trap the
                # OpenRouter comment above carried, caught in the same QA pass.) PASSIVE: a pick binds a real slug
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
                            "Credit-billed. See "
                            "https://github.com/jbrick2070/"
                            "ComfyUI-OldTimeRadio/blob/v2.0-alpha/docs/"
                            "comfy-credits-setup.md."
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
                            "Credit-billed. See "
                            "https://github.com/jbrick2070/"
                            "ComfyUI-OldTimeRadio/blob/v2.0-alpha/docs/"
                            "comfy-credits-setup.md."
                        ),
                    },
                ),
                # `refine_target_grade` was REMOVED 2026-08-28 -- a
                # DISHONEST CONTROL, not merely an unused one. Its tooltip
                # promised to "keep REWRITING the story ... until it
                # reaches this grade", offered five grades and a five-pass
                # cap, and named OTR_STORY_REFINE_BAR /
                # OTR_STORY_REFINE_PASSES as headless overrides.
                #
                # NONE OF THAT EXISTED. The refine machinery was deleted
                # 2026-07-24; neither env var is read anywhere in the repo,
                # and `_refine_loop` / `refine_pass` / `story_refine` return
                # zero hits. The widget spent a month promising a revision
                # loop that could not run, which is worse than absent -- a
                # reader reasonably believed the knob did something.
                #
                # Removed WITH its migration (operator ruling 2026-08-28 --
                # "that's being lazy not to remove an inert widget"): slot
                # 20 dropped and every later slot re-indexed across all 62
                # workflow files carrying this node. Every one stored 'Off'.
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
                # Stage 2C (multi-modal story schema, 2026-07-05) -- the
                # story-path source_bank selector, APPENDED at the END of
                # optional as combined widget slot 23, BUG-LOCAL-097. Choices
                # come LIVE from the lazy story-routing registry (stable bank
                # IDS as values; labels belong in tooltips only). NOTE: this
                # call may RAISE (StoryRoutingError) -- a DELIBERATE exception
                # to the "INPUT_TYPES must never raise" convention used by the
                # openrouter probe above (no-fallback law): a broken
                # banks.json must fail node registration LOUD, never boot
                # with a baked-in choice list. Non-runnable banks ARE listed
                # -- picking one raises a loud StoryBankNotRunnableError at
                # run() before any story work (honest error on use).
                # 2026-07-31: the ROLL SENTINEL is PREPENDED as choice 0. It
                # is a UI command, not a registry row -- no new widget, no
                # positional slot shift, and ZERO canonical-JSON diff (a
                # graph persists the selected VALUE, never the choice list).
                "source_bank": (
                    [_ROLLS.BANK_SENTINEL]
                    + list(_otr_story_routing.list_bank_ids()),
                    {
                        "default": "scifi_news_pro",
                        "tooltip": (
                            "Story-path SOURCE BANK (multi-modal story "
                            "schema). Selects which registered story pack "
                            "supplies the pack-routed creative prompts and "
                            "which lane the episode runs. scifi_news_pro = "
                            "the local default sci-fi bank, an LLM-first "
                            "multipass lane using the configured model "
                            "slots. "
                            "Each lane is an INDEPENDENT bank (own pack + "
                            "bank metadata). The only non-runnable row is '+ Add "
                            "Your Own' (custom_source_bank) -- picking it "
                            "FAILS LOUD before any story work (no fallback), "
                            "with its guide_ref naming the real path: author a "
                            "bundle under user_packs/source_banks/, run "
                            "'otr_check bank <path> --activate', restart, and "
                            "your bank joins this list as its own entry "
                            "(contract: docs/EXTENDING_OTR.md). A bank's own "
                            "default_story_model picks its story pack -- there "
                            "is no separate pack widget. "
                            "ROLL: pick 'roll (any eligible bank)' to let the "
                            "run choose for you, uniformly, from every "
                            "runnable bank whose lane can build the requested "
                            "shape. This is INDEPENDENT of the visual_style "
                            "roll -- rolling one does not roll the other. The "
                            "choice is recorded in the ledger at "
                            "meta.bank_roll (selected id, seed, and the exact "
                            "pool it drew from); set OTR_BANK_SEED to replay a "
                            "past roll. A pinned source_ref cannot be combined "
                            "with the roll -- a pinned source belongs to one "
                            "bank."
                        ),
                    },
                ),
                # Stage 3C (2026-07-06) -- the VISUAL STYLE selector, APPENDED
                # at the END as combined widget slot 24, BUG-LOCAL-097. Choices
                # LIVE from the lazy visual-style registry; may RAISE
                # (VisualStyleError) -- the same deliberate INPUT_TYPES
                # exception as source_bank above (no-fallback law; a broken
                # pack dir fails node registration LOUD). Unlike story banks,
                # every listed style is FULLY LIVE (styles rewrite prompt
                # tails only -- no execution lane needed).
                # 2026-07-31: this dropdown gets its OWN roll sentinel,
                # prepended as choice 0 -- the SECOND randomizer, switched
                # independently of the source_bank roll. Same UI-command
                # posture: no new widget, no slot shift, no canonical diff.
                "visual_style": (
                    [_ROLLS.STYLE_SENTINEL]
                    + list(_ROLLS.eligible_style_ids()),
                    {
                        "default": "sci_fi_radio",
                        "tooltip": (
                            "VISUAL STYLE (multi-modal story schema). "
                            "Rewrites ONLY the downstream still/video prompt "
                            "style language (tails); story content is "
                            "untouched. sci_fi_radio = the production look "
                            "(default, byte-identical). anime / cartoon / "
                            "paper_origami / archival_documentary / "
                            "recur_frac / shakespeare_stage_realism / "
                            "storybook_engraving / video_art are live "
                            "immediately. "
                            "Unknown id fails LOUD before any "
                            "story work. "
                            "ROLL: pick 'roll (any style)' to let the run "
                            "choose the look for you, uniformly, from every "
                            "registered style (they are all fully live, so "
                            "there is nothing to exclude). This is a SEPARATE "
                            "randomizer from the source_bank roll -- either, "
                            "both, or neither. The choice is recorded at "
                            "meta.style_roll; set OTR_VISUAL_STYLE_SEED to "
                            "replay a past roll (that is its own seed -- "
                            "OTR_STYLE_SEED is the narrative arc-shape seed "
                            "and is unrelated)."
                        ),
                    },
                ),
                # Google BYO API direct LLM slot pickers (2026-07-08),
                # APPENDED after source_bank/visual_style as combined widget
                # slots 25/26. Passive: these bind concrete Gemini model ids
                # only when creative_writing_model / technical_model selects
                # google_api:slot-a/b. Choices are network-free at INPUT_TYPES.
                "google_api_slot_a_model": (
                    _google_slot_a_choices,
                    {
                        "default": _google_slot_a_choices[0],
                        "tooltip": (
                            "Google Gemini API model bound to "
                            "'google_api:slot-a' (creative slot). Env-only "
                            "auth: OTR_GOOGLE_API_KEY, GEMINI_API_KEY, or "
                            "GOOGLE_API_KEY. Passive until the main model "
                            "dropdown selects google_api:slot-a. No local "
                            "fallback."
                        ),
                    },
                ),
                "google_api_slot_b_model": (
                    _google_slot_b_choices,
                    {
                        "default": _google_slot_b_choices[0],
                        "tooltip": (
                            "Google Gemini API model bound to "
                            "'google_api:slot-b' (technical slot). Use a "
                            "structured-output capable text model for JSON "
                            "passes. Env-only auth, no local fallback."
                        ),
                    },
                ),
                # Source Banks v2 source reference (2026-07-08), APPENDED
                # after the Google API pickers as combined widget slot 27.
                # Blank is inert; future bank-specific fetchers may consume a
                # URL/id/title here and must add their own fail-loud validators.
                "source_ref": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional source reference for source-bank lanes "
                            "(for example a public-domain URL/id/title). Blank "
                            "uses the bank's default source selection. This is "
                            "not a fallback; unsupported nonblank references "
                            "must fail loud in the consuming bank."
                        ),
                    },
                ),
                # S5 platform-portability (2026-07-10): the six EXPLICIT LLM
                # runtime-policy widgets, APPENDED after source_ref as
                # combined widget slots 28-33 (append-only; BUG-LOCAL-097).
                # Defaults = the nv50 16 GB baseline, so an old workflow with
                # a 28-slot vector resolves byte-identically. They feed
                # _resolve_inputs' LLMRuntimePolicy 1:1 (S1) and are
                # profile-managed via widget_mapping llm.* keys.
                "llm_device": (
                    ["cuda", "cpu", "mps"],
                    {"default": "cuda",
                     "tooltip": "EXPLICIT LLM device (platform profile "
                                "value). No auto-detect; an unavailable "
                                "device fails loud at load."},
                ),
                "llm_attn_impl": (
                    ["sdpa", "flash_attention_2", "eager"],
                    {"default": "sdpa",
                     "tooltip": "Attention implementation for the "
                                "transformers lane (the FA2 auto-probe is "
                                "gone). sdpa = the proven baseline."},
                ),
                "llm_quant_policy": (
                    ["bnb_nf4", "bnb_8bit", "none"],
                    {"default": "bnb_nf4",
                     "tooltip": "Quantization for the transformers lane. "
                                "bnb lanes are OFF on ROCm/MPS/CPU tiers "
                                "(missing bitsandbytes fails loud)."},
                ),
                "llm_vram_ceiling_gb": (
                    "FLOAT",
                    {"default": 14.5, "min": 0.0, "max": 96.0, "step": 0.1,
                     "tooltip": "Pre-download VRAM-fit ceiling (GB). 0 "
                                "DISABLES the gate (cpu tier only)."},
                ),
                "gguf_n_ctx": (
                    "INT",
                    {"default": 4096, "min": 512, "max": 32768, "step": 512,
                     "tooltip": "GGUF lane context window. NO silent "
                                "downgrade: a window that does not fit "
                                "free VRAM fails loud."},
                ),
                # 2026-08-25: Q6_K REMOVED from the operator-facing choices
                # (operator: "if it doesn't fit nicely ... rip it from the
                # dropdown"). Selecting it was a GUARANTEED failure: it is
                # unpinned in GGUF_ARTIFACTS -- ("...Q6_K.gguf", None, None) --
                # and since A6 (2026-07-27) an unpinned quant is REFUSED at
                # load, so the only reachable outcome was a raise. No Q6_K
                # artifact has ever been on this box either.
                # The table entry and _GGUF_QUANTS deliberately KEEP Q6_K: it
                # is the live fixture that proves the unpinned-refusal path
                # (tests/test_gguf_registry.py, tests/test_llm_runtime_policy.py).
                # Deleting it would have removed the operator's guaranteed
                # failure AND the only proof that the guard works. Re-add it
                # here the day it is on disk and pinned BY MEASUREMENT.
                "gguf_quant": (
                    ["Q8_0", "Q4_K_M"],
                    {"default": "Q8_0",
                     "tooltip": "GGUF artifact quant (filename + expected "
                                "size come from the artifact table). Only "
                                "PINNED quants are offered -- an unpinned "
                                "artifact is refused at load."},
                ),
                # S5 gate_in (2026-07-10, validation-order fix): the
                # OTR_WorkflowValidator report gates the WRITER now (link
                # 279), so an invalid variant fails BEFORE any LLM work
                # burns -- previously only OTR_VideoDirector was gated
                # (link 269) and a bad variant wasted the whole story
                # phase first. forceInput: consumes NO widgets_values slot.
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": "Ordering/validation signal (wire "
                               "OTR_WorkflowValidator.validation_report).",
                }),
            },
            # ComfyUI injects the configured Comfy API key into this hidden
            # input at execution time (the API-nodes auth convention). The
            # writer threads it to _otr_comfy_backend.set_auth() so the Comfy
            # Credits lane can make the credit-billed call. It is NOT a widget
            # (absent from widgets_values) and is never logged.
            #
            # The logged-in account's SESSION BEARER is deliberately not
            # requested here. The Comfy Registry security scan treats a
            # third-party pack that declares that hidden input as credential
            # access and marks the version critical / Flagged: alpha.13
            # through alpha.15 all carried exactly that finding on this dict
            # (PBUG-20260902-04). The API key is the credential ComfyUI
            # provides for this use, and the backend already preferred it.
            "hidden": {
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


    def run(
        self,
        episode_title="",
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
        # ComfyUI-injected hidden auth (API-nodes convention): the configured
        # Comfy API key. None when no key is configured / the Comfy Credits
        # lane is unused.
        api_key_comfy_org=None,
        # Story-scaffold UI toggle (2026-06-24): auto/on/off. Governs the whole
        # bundled scaffold via OTR_ENABLE_STYLE_GRAMMAR (see the resolver at the
        # top of the body). Default "auto" => env/default => byte-identical.
        story_scaffold="auto",
        # Stage 2C (2026-07-05): the story-path source_bank selector,
        # appended at the END of the widget surface (slot 23). Default
        # science_news = the production lane, byte-identical. Gated FIRST
        # in the body via require_runnable_bank (no fallback).
        source_bank="scifi_news_pro",
        # Stage 3C (2026-07-06): the visual-style selector, appended at the
        # END of the widget surface (slot 24). Default sci_fi_radio = the
        # production look, byte-identical. Validated fail-loud beside the
        # bank gate; stamped at meta["visual_style"] (the threading channel).
        visual_style="sci_fi_radio",
        # Google BYO API slot pickers (2026-07-08), appended after visual_style.
        # Default "" => unset; selecting google_api:slot-a/b with an unset slot
        # fails loud before the HTTP request.
        google_api_slot_a_model="",
        google_api_slot_b_model="",
        # Source Banks v2 reference widget (2026-07-08), appended after the
        # Google API pickers. Blank is inert for current science/media paths.
        source_ref="",
        # S5 platform-portability (2026-07-10): the six explicit LLM
        # runtime-policy widgets (slots 28-33). Defaults = nv50 baseline.
        llm_device="cuda",
        llm_attn_impl="sdpa",
        llm_quant_policy="bnb_nf4",
        llm_vram_ceiling_gb=14.5,
        gguf_n_ctx=4096,
        gguf_quant="Q8_0",
        # S5 gate_in (validation-order fix): opaque ordering signal from
        # OTR_WorkflowValidator -- never parsed, just sequenced.
        gate_in="",
    ):
        """Generate one accepted v2.0 LPL story artifact."""
        # Stage 2C run-intent gate -- the FIRST statement of the body, before
        # the story-scaffold env mutation, the refine gate, the budget resets,
        # and _resolve_inputs (RSS fetch): a non-runnable source_bank pick
        # fails ONCE, LOUD and CHEAP, with zero side effects. bank.runnable is
        # the ONLY runtime gate; unknown id = UnknownBankError (no fallback).
        # Chunk 3: the returned bank row is BOUND -- D.2.5 resolves the
        # bank's interpreter contract from it (r2 codex M1).
        # THE TWO RANDOMIZERS (2026-07-31), resolved before their gates.
        #
        # Each dropdown carries its OWN roll sentinel, so the two are
        # switched independently: roll the bank and pin the style, pin the
        # bank and roll the style, roll both, or roll neither. There is no
        # shared "randomize" flag and neither roll can enable the other.
        #
        # Both REBIND the local (`source_bank` / `visual_style`) to a
        # concrete id, so everything downstream -- the gates below,
        # _resolve_inputs, resolved[...], meta[...], pack routing, HUD and
        # credits -- sees an ordinary manual pick and needs no change.
        # A non-sentinel value returns unchanged with NO receipt, which is
        # what keeps the manual path byte-identical.
        #
        # HAZARD FOR WHOEVER REBUILDS THE REFINE LOOP: this resolves once
        # per run() entry. There is no refine re-entry at HEAD (the refine
        # machinery was removed; `refine_target_grade` is an inert widget),
        # so nothing re-rolls today. If a loop that re-enters run() ever
        # returns, it MUST carry these receipts back in and short-circuit --
        # otherwise every pass re-rolls and the shipped ledger records a
        # bank the episode never used. Both kibitz panelists found exactly
        # that bug in the r2 draft, independently.
        source_bank, _bank_roll = _ROLLS.resolve_bank_selection(
            source_bank,
            source_ref=source_ref,
        )
        visual_style, _style_roll = _ROLLS.resolve_style_selection(
            visual_style)
        _source_bank_row = _otr_story_routing.require_runnable_bank(source_bank)
        # Stage 3C visual-style gate -- beside the bank gate, same zero-side-
        # effect contract: an unknown visual_style id raises
        # UnknownVisualStyleError here, before ANY story work (no fallback).
        # Every REGISTERED style (and the dynamic visual_storybased style) is valid.
        if visual_style != _ROLLS.DYNAMIC_STYLE_ID:
            _otr_visual_styles.resolve_visual_style(visual_style)
        # The lane REQUEST GATE (2026-07-31) was removed 2026-08-14 with the
        # word authority. It sat here to let a dispatched lane refuse a
        # `target_words` outside its band -- one since-retired lane's
        # 30..900 was the only band ever declared. There is no target to
        # refuse, so there is nothing to gate: the act count is always
        # honoured, and every bank's topology accepts every act count.
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
        # LLM slot preflight (GGUF row registry, 2026-07-16): the single early
        # resolution of the ONE policy + normalized slot ids + immutable
        # per-slot GGUF load_config -- AFTER the bank / word-count / refine
        # gates, BEFORE the scaffold env mutation and any source fetch/rerank.
        # A gguf slot with a bad quant / out-of-range n_ctx / missing
        # OTR_GGUF_SEED fails LOUD here, cheaply, before any story work.
        _llm_preflight = _preflight_llm_selection(
            creative_writing_model=creative_writing_model,
            technical_model=technical_model,
            llm_device=llm_device,
            llm_attn_impl=llm_attn_impl,
            llm_quant_policy=llm_quant_policy,
            llm_vram_ceiling_gb=llm_vram_ceiling_gb,
            gguf_n_ctx=gguf_n_ctx,
            gguf_quant=gguf_quant,
        )
        import os
        _scaffold = _apply_story_scaffold_env(story_scaffold)
        if _scaffold in ("on", "off"):
            log.info(
                "[OTR_LedgerScriptWriter] story_scaffold=%s -> "
                "OTR_ENABLE_STYLE_GRAMMAR=%s (widget override)",
                _scaffold, os.environ.get("OTR_ENABLE_STYLE_GRAMMAR"),
            )

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
            _occ_budget.set_auth(api_key=api_key_comfy_org)
        except Exception:  # noqa: BLE001 -- budget/binding setup is best-effort
            pass

        try:
            from ._otr_google_api import models as _gai_models
            _gai_models.set_slot_bindings(
                slot_a=google_api_slot_a_model,
                slot_b=google_api_slot_b_model,
            )
        except Exception as exc:  # noqa: BLE001 -- non-Google runs stay unaffected
            log.warning(
                "[OTR_LedgerScriptWriter] Google API slot binding setup "
                "failed; a selected Google lane will fail closed: %r", exc,
            )

        # --- A. Resolve all widget inputs (RSS fetch happens here) -----
        resolved = _resolve_inputs(
            num_characters=num_characters,
            episode_title=episode_title,
            creative_writing_model=creative_writing_model,
            technical_model=technical_model,
            custom_premise=custom_premise,
            include_act_breaks=include_act_breaks,
            act_count=act_count,
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
            # Stage 2C: the source_bank widget selection (gated above).
            source_bank=source_bank,
            # Stage 3C: the visual_style widget selection (gated above).
            visual_style=visual_style,
            google_api_slot_a_model=google_api_slot_a_model,
            google_api_slot_b_model=google_api_slot_b_model,
            source_ref=source_ref,
            # S5: the explicit LLM runtime-policy widgets -> LLMRuntimePolicy.
            llm_device=llm_device,
            llm_attn_impl=llm_attn_impl,
            llm_quant_policy=llm_quant_policy,
            llm_vram_ceiling_gb=llm_vram_ceiling_gb,
            gguf_n_ctx=gguf_n_ctx,
            gguf_quant=gguf_quant,
            # GGUF row registry (2026-07-16): hand the RSS fetch/rerank the
            # preflight's ONE policy + the technical-slot load_config so a
            # gguf technical slot reranks under its real per-row config, not
            # the gemma env fallback.
            preflight_policy=_llm_preflight.policy,
            technical_load_config=_llm_preflight.load_config_by_slot.get("technical"),
            # THE FORWARDING WHOSE ABSENCE MADE THE WIDGET INERT. Without this
            # line the dispatched lane runners never saw the operator's cameo
            # choice at all -- only the legacy in-line path did, by reading the
            # run() local directly. A reach-test pins this arrival.
            lemmy_cameo=lemmy_cameo,
        )

        log.info(
            "[OTR_LedgerScriptWriter] start: creative_model=%r, "
            "technical_model=%r, act_count=%d, num_characters=%d, "
            "creativity=%r (temp=%.2f top_p=%.2f), seed_source=%s, "
            "episode_title=%r, perfect_run_spacesaver=%s",
            resolved["creative_writing_model"],
            resolved["technical_model"],
            resolved["act_count"],
            resolved["num_characters"],
            resolved["creativity"],
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
        from . import _otr_news_wiring as _OTRNW
        from . import production_ledger as _PL
        from . import _otr_continuity as _OTRCONT
        from . import _otr_config as _OTRCFG
        from . import _otr_style_catalog as _OTRSTYLE
        from . import _otr_source_world as _OTRWORLD
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
        #   Source interpreter  -> technical primary; a typed quality reject
        #                          alternates fresh creative/technical passes
        #                          before the bank-specific source floor.
        #
        # slot-interleave: when news_interpreter runs after the style
        # picker (creative -> technical) and before cast lock
        # (technical -> creative), one transition lands per direction.
        # Documented at the call sites below.
        # GGUF row registry (2026-07-16): attach the resolved episode sampling
        # to each per-slot GGUF load_config (from the early preflight) for the
        # receipt, then hand them to the scheduler. Sampling is generate-time
        # and does NOT shape the load / resident-reuse identity.
        _gguf_lc_by_slot = {
            _s: _lc.with_sampling(
                top_p=resolved["top_p"], min_p=resolved["min_p"],
                repeat_penalty=resolved["repetition_penalty"],
            )
            for _s, _lc in _llm_preflight.load_config_by_slot.items()
        }
        slot_scheduler = _SlotScheduler(
            creative_id=resolved["creative_writing_model"],
            technical_id=resolved["technical_model"],
            top_p=resolved["top_p"],
            # Phase 4 v4 (2026-05-11) sampling knobs.
            min_p=resolved["min_p"],
            repetition_penalty=resolved["repetition_penalty"],
            # S1 platform-portability: the explicit runtime policy rides
            # every request_slot call this scheduler makes.
            policy=resolved["llm_policy"],
            load_config_by_slot=_gguf_lc_by_slot,
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
        #   D.2  news interpretation (script_brief), THEN the style
        #        engine (build_story_contract, needs script_brief)
        #   D.3  lock_cast() -- ANNOUNCER first, LEMMY 11%, then
        #        per-character LLM call for description+gender+voice
        #   D.4  led.set_cast() + cast_status="locked"; the requested count
        #        lives at cast_contract.num_characters_request (the field
        #        cast_lock replays from -- the old top-level duplicates
        #        requested_num_characters/cast_locked retired 2026-08-28)
        #   D.5  generate_outline() consumes the locked character_cast
        # ---------------------------------------------------------------
        # D.1 Ledger up front. Subsequent stages stamp meta against it.
        # cast_status="building" is deliberately kept: frozen in a crashed
        # partial ledger it is the one signal that casting never finished.
        led = _PL.new_ledger(episode_id=None)
        episode_id = led.episode_id           # pending_<YYYYMMDD_HHMMSS>
        audio_dir = Path(led.out_dir)         # otr/episodes/<ep>/audio/
        episode_root = audio_dir.parent       # otr/episodes/<ep>/
        meta = led.data.setdefault("meta", {})
        meta["cast_status"] = "building"
        # Stage 2C: stamp the authoritative story-path selection (resolved
        # dict is the single source; run() gated it runnable already).
        meta["source_bank"] = resolved["source_bank"]
        # Randomizer receipt (2026-07-31). Written ONLY when the bank
        # actually rolled: on a manual pick the key is ABSENT -- not null,
        # not a stub -- so the frozen ledger answers "was this rolled?"
        # without a convention about empty values.
        if _bank_roll is not None:
            meta["bank_roll"] = _bank_roll.to_meta()
        # v4 campaign (2026-07-17): stamp the resolved VISUAL-STYLE pool class so
        # the style catalog (select_style) picks the right pool WITHOUT a family
        # base-map -- each _v4 bank is fully independent. Default 'generic'. This
        # is the visual-style axis, orthogonal to the source FEED / science floor.
        meta["style_pool_class"] = str(
            (_source_bank_row.defaults or {}).get("style_pool_class", "generic")
            or "generic"
        )
        # v4 P1(vi): opt-in header<->scene structural gate (deterministic G15).
        # Default False -> key absent -> inert for every current bank.
        if bool((_source_bank_row.defaults or {}).get("scene_coherence_check", False)):
            meta["scene_coherence_check"] = True
        meta["source_ref"] = resolved["source_ref"]
        meta["source_meta"] = dict(resolved["source_meta"])
        meta["source_rights"] = dict(resolved["source_rights"])
        # The announcer's work title, bound ONCE here -- immediately after
        # `source_meta` is stamped and UNCONDITIONALLY, on every lane. It feeds
        # the outline request and BOTH announcer-open producers further down.
        # `identity_from_meta` is the single bibliographic authority: it reads
        # `play_title` for shakespeare and `title` for public_domain, so no
        # per-lane branch is needed here; it never raises, and a lane that
        # adapts nothing yields "" and simply emits no WORK line.
        _identity = _OTRSID.identity_from_meta(meta)
        # GATED ON THE LANE, NOT ON TRUTHINESS. `work_title` means two
        # different things: the work being PERFORMED on the adaptation lanes,
        # and the PUBLICATION a post came from on media_archive, where 56 of 98
        # live ledgers carry a `source_label` like "Now See Hear!". Announcing
        # "a scene from Now See Hear!" would invent a play, which is a worse
        # fidelity defect than the wrong-play frame this change fixes.
        _work_title = (
            _identity.work_title
            if _identity.source_kind in _OTRSID.ADAPTATION_SOURCE_KINDS
            else ""
        )
        _stamp_news_seed_receipt(meta, resolved)
        # kibitz r2-r4 provenance: any bank whose defaults define
        # credits_source_line gets it stamped (data-driven -- the
        # original_radio row always defines it, so its credits line is
        # UNCONDITIONAL for that lane; science defines none and stays
        # byte-identical). The credits roll renders the stamp when
        # present -- no bank branch in the credits code.
        _credits_line = str(
            (_source_bank_row.defaults or {}).get("credits_source_line")
            or ""
        )
        if _credits_line:
            meta["credits_source_line"] = _credits_line
        # v4 P1(viii): opt-in source-provenance normalizer. Map source_rights ->
        # one normalized record; stamp the spoken coda line + fill
        # credits_source_line when the bank default did not. A research_only
        # source still BLOCKS publish, deterministically -- but at the
        # publication boundary now, not the freeze (2026-08-15, build contract
        # D5a): `blocks_publish` feeds the publication-eligibility receipt that
        # Phase 10 stamps and OTR_MasterAudioMux consumes, so the episode keeps
        # its archival final and only the OBS copy is withheld. G14 in the gap
        # audit reports the same fact as a warning.
        # ACTIVE on public_domain and shakespeare (nodes/story_packs/banks.json),
        # pinned by tests/test_provenance_v4.py. This comment read "inert for
        # every current bank" for a day after both banks opted in on 2026-08-04,
        # and that stale line is what produced a wrong spec for the citation fix.
        if bool((_source_bank_row.defaults or {}).get("provenance_normalize", False)):
            try:
                from . import _otr_provenance as _OTRPROV
            except ImportError:  # pragma: no cover -- flat load
                import _otr_provenance as _OTRPROV  # type: ignore
            _prov = _OTRPROV.normalize_provenance(meta.get("source_rights"))
            meta["provenance"] = _prov
            # STAMPED UNCONDITIONALLY, empty included. spoken_coda_line() returns
            # "" for unknown status and for a licensed source carrying no label,
            # and the old `if _coda:` guard left the KEY ABSENT in exactly those
            # cases -- so a reader testing the coda's presence would read an owned
            # lane as unowned and fall back to the raw LLM note, which is the leak
            # itself. Ownership is `"provenance" in meta`, stamped just above and
            # never conditional.
            # NAME WHAT WE ADAPTED (operator ruling 2026-08-15). Three live
            # episodes closed without naming their source: ghost_of_elsinore
            # said only "We've lost our signal.", and midnights_ticktock said
            # "a work in the public domain" while adapting Leacock's "Gertrude
            # the Governess". The mechanism was this call: spoken_coda_line
            # keyed on licence STATUS alone and took no per-work input, so it
            # could not have named a title even though `meta["source_meta"]`
            # (stamped just above) carries the title and the author.
            #
            # The identity is stamped as its own receipt because it carries
            # FIELD-LEVEL PROVENANCE -- which meta path each spoken name came
            # from -- so a wrong credit is fixable at its source instead of
            # being argued about downstream, and a DEGRADED identity (a lane
            # that could not name its work) is visible in the artifact rather
            # than silently becoming the generic sentence.
            # `_identity` is already bound above, unconditionally, from the
            # same `meta` -- re-deriving it here would parse the same mapping
            # twice and give this branch its own copy to drift from.
            meta["source_identity"] = _identity.as_receipt()
            meta["provenance_coda_line"] = _OTRPROV.spoken_coda_line(
                _prov, _identity)
            if _identity.is_degraded:
                log.warning(
                    "[OTR_LedgerScriptWriter] the closing announcer could not "
                    "name this source: kind=%r title=%r author=%r -- the coda "
                    "falls back to the generic sentence and "
                    "meta.source_identity records why",
                    _identity.source_kind, _identity.work_title,
                    _identity.author,
                )
            if not str(meta.get("credits_source_line") or "").strip():
                _pc = _OTRPROV.printed_credit_line(_prov)
                if _pc:
                    meta["credits_source_line"] = _pc
            # A NON-COMMERCIAL SOURCE HAS TO REACH A HUMAN (2026-08-04).
            # commercial_use_allowed was already validated, carried and
            # normalized -- and shown to nobody. An operator publishing a
            # Folger-sourced episode had no way to learn it must not be sold.
            # Stamped for downstream surfaces AND logged loudly here, because
            # the ledger is read by machines and the log is read by people.
            _nc = _OTRPROV.noncommercial_notice(_prov)
            if _nc:
                meta["noncommercial_notice"] = _nc
                log.warning("[OTR_LedgerScriptWriter] %s", _nc)
        meta["story_scaffold"] = _scaffold
        # Stage 3C: stamp the visual style -- THE threading channel: every
        # downstream visual composer reads meta["visual_style"] via
        # get_visual_style(meta) off the serialized ledger (stamp precedes
        # all of them by construction -- verified 2026-07-05).
        meta["visual_style"] = resolved["visual_style"]
        if resolved["visual_style"] == _ROLLS.DYNAMIC_STYLE_ID:
            meta["visual_style_receipt"] = {"status": "pending"}
        # Randomizer receipt for the OTHER surface -- separate key, separate
        # seed, separate presence. A run may carry one, both, or neither.
        if _style_roll is not None:
            meta["style_roll"] = _style_roll.to_meta()
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
        if not _skeleton_path:
            raise RuntimeError("failed to save skeleton ledger to disk")
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

        # --- scifi_news_pro S2: pipeline-runner dispatch (doc s11) --------
        # Consulted exactly ONCE, here: after the shared front (bank
        # resolve -> runnable gate -> science_rss fetch ->
        # validate_source_payload -> D.1 new_ledger + meta stamps +
        # skeleton save) and BEFORE the D.2 news-interpreter branch.
        # Hit -> the lane runner fills led/meta to the tail boundary and
        # the writer hands off to _run_writer_tail (the runner returns
        # plain NewsProTailParts; the WRITER builds WriterTailContext --
        # r4/M3 acyclic import graph). Miss (legacy_many_pass / the
        # original bank-shape branch) -> everything below runs
        # byte-identically. An unknown pipeline raises LOUD inside
        # _LANES.runner_for (r3/S1).
        _pipeline_id = str(getattr(
            _source_bank_row, "default_story_pipeline", "") or "")
        _lane_runner = _LANES.runner_for(_pipeline_id)
        if _lane_runner is not None:
            _parts = _lane_runner(
                payload=dict(resolved["news_article"]),
                pack=_otr_story_routing.resolve_story_pack(
                    _source_bank_row.source_bank_id),
                resolved=resolved,
                led=led,
                meta=meta,
                creative_fn=creative_generate_fn,
                technical_fn=technical_generate_fn,
                slot_scheduler=slot_scheduler,
                source_bank_row=_source_bank_row,
                episode_root=episode_root,
                episode_id=episode_id,
            )
            _tail_ctx = WriterTailContext(
                led=led,
                # Custom runners may save incrementally; Ledger.save()
                # replaces led.data with the merged on-disk payload, so the
                # pre-dispatch alias can be stale here. Hand the shared tail
                # the live mapping that owns the selected FinalDraft seals.
                meta=led.data.setdefault("meta", {}),
                resolved=resolved,
                outline_view=_parts.outline_view,
                canon=_parts.canon,
                episode_root=episode_root,
                episode_id=episode_id,
                contract=None,           # no style contract ("" slug path)
                style_grammar_on=False,  # receipt stamp honest
                source_bank_row=_source_bank_row,
                slot_scheduler=slot_scheduler,
                creative_fn=creative_generate_fn,
                technical_fn=technical_generate_fn,
                run_story_spine=_parts.run_story_spine,
                final_title_override=_parts.final_title_override,
                style_roll=_style_roll,
            )
            return self._run_writer_tail(
                _tail_ctx,
                tail_finalizer=getattr(_parts, "tail_finalizer", None),
            )

        # D.2 News interpretation. Read the full article (currently
        # discarded after RSS fetch -- see _fetch_rss_seed_or_die change
        # in this commit) and emit four purpose-specific briefs that
        # cast / outline / announcer / line-composer consume INSTEAD
        # of the mechanical 500-char slice of headline+summary.
        # ADR docs/news_interpreter_adr.md section 5 -- commit 3 of
        # the news_interpreter sprint.
        #
        # Source interpretation is bank-specific; liveness is shared. A typed
        # model-quality exhaustion alternates fresh technical/creative passes
        # with exact rejection feedback, then emits a validated source-derived
        # bank floor at the finite ceiling. Configuration, I/O, backend, and
        # contract failures remain fail-loud.
        article = resolved["news_article"]
        # Chunk 3 (2026-07-05): the interpretation routes through the bank's
        # declared interpreter contract (science_news -> news_interpreter ->
        # the verbatim build_news_briefs call, byte-identical). Resolution
        # sits OUTSIDE the try by design (a missing/unknown contract raises
        # SourceContractMissingError/UnknownInterpreterError LOUD -- never
        # caught by the degrade branch below).
        # kibitz r2-r4: the original lane swaps in an interpreter-SHAPED
        # adapter (same call contract) so the whole stamping path below --
        # validate_interpreter_result, meta["news"], the briefs unpack --
        # runs byte-identically for both lanes. Its failures
        # (OriginalBriefsError / StructuredCallFailedError) are NOT
        # SourceInterpretError, so the science degrade/halt branch below
        # never catches them: this lane hard-fails, no degrade, and the
        # news_briefs_required lever does not apply.
        _source_contract_lane = not _bank_has_no_source_contract(
            _source_bank_row)
        if not _source_contract_lane:
            _interp = _make_original_interpreter(
                creative_fn=creative_generate_fn,
                resolved=resolved,
                meta=meta,
            )
        else:
            # Wave 3: None for every shipped bank; a client bank resolves to
            # its OWN bundle's interpret_source, called with the identical
            # kwargs and validated by the identical validate_interpreter_result
            # below. Resolution stays OUTSIDE the try (AST-pinned).
            _interp_owner = _otr_story_routing.user_bank_bundle(
                _source_bank_row.source_bank_id)
            _interp = _otr_source_payload.resolve_interpreter(
                _source_bank_row, owner=_interp_owner)
        try:
            if _source_contract_lane:
                briefs = _run_source_interpreter(
                    interpreter=_interp,
                    bank=_source_bank_row,
                    payload=article,
                    source_meta=meta.get("source_meta") or {},
                    technical_fn=technical_generate_fn,
                    technical_model_id=str(resolved["technical_model"]),
                    slot_scheduler=slot_scheduler,
                    meta=meta,
                )
            else:
                # Original has a source-interpreter-shaped adapter, but its
                # creative front owns a separate bank-specific repair policy.
                with slot_scheduler.helper_context("build_news_briefs"):
                    briefs = _interp(
                        bank=_source_bank_row,
                        payload=article,
                        technical_fn=technical_generate_fn,
                        model_id=str(resolved["technical_model"]),
                    )
            # Contract enforcement: validates the direct attrs AND the dump
            # (single model_dump() call, inside the validator); a violation
            # raises SourcePayloadContractError which the except below does
            # NOT catch -- contract bugs propagate hard, never degrade.
            meta["news"] = _otr_source_payload.validate_interpreter_result(
                briefs,
                origin=(
                    f"run() D.2.5 interpret "
                    f"(bank={_source_bank_row.source_bank_id!r}, "
                    f"interpreter={_source_bank_row.interpreter!r})"
                ),
            )
            casting_brief = briefs.casting_brief
            script_brief = briefs.script_brief
            key_terms_tuple: tuple[str, ...] = tuple(briefs.key_terms)
            # ADAPTATION lanes: surface the SOURCE's real character names so the
            # cast preserves the play's people (MACBETH, BANQUO) instead of rolling
            # "QUASIMODO VAUGHN" from the random pool. public_domain provides them
            # via the brief's character_names field (LLM-extracted from prose);
            # shakespeare's manifest cast_hints ARE the real character names. Stamped
            # here where `briefs` is in scope; read at the lock_cast call below.
            # Invention lanes never set this key -> casting is byte-identical (C7).
            if (_source_bank_row.defaults or {}).get(
                    "propagate_adaptation_cast"):
                _adapt_names = list(
                    getattr(briefs, "character_names", None)
                    or (meta.get("source_meta") or {}).get("cast_hints")
                    or []
                )
                if _adapt_names:
                    meta["_adaptation_character_names"] = _adapt_names
                    # Join those names to the gender the SOURCE records, so the
                    # cast row stops being a 40/40/20 roll. This is the same gate
                    # that already decides an adaptation lane, so invention lanes
                    # never reach it and stay byte-identical.
                    _src_meta = meta.get("source_meta") or {}
                    try:
                        meta["_adaptation_character_genders"] = (
                            _otr_roster_gender.gender_map_for_names(
                                _adapt_names,
                                _src_meta.get("characters") or (),
                                play_code=str(_src_meta.get("play_code") or ""),
                                supplement=_otr_roster_gender
                                .load_gender_supplement(
                                    _otr_roster_gender.supplement_dir_for_bank(
                                        _source_bank_row)),
                            )
                        )
                    except _otr_roster_gender.RosterGenderError:
                        # A committed supplement contradicting a confirmed
                        # sidecar is an authoring fault in the data file, not a
                        # reason to kill an episode. Fall back to the roll and
                        # let the data test catch it.
                        log.exception(
                            "[OTR_LedgerScriptWriter] roster gender supplement "
                            "rejected; falling back to the gender roll")
                        meta["_adaptation_character_genders"] = {}
            log.info(
                "[OTR_LedgerScriptWriter] news_interpreter OK: "
                "%d key_terms in %d attempt(s)",
                len(briefs.key_terms), briefs.attempts,
            )
        except _otr_source_payload.SourceInterpretError as exc:
            # Quality/schema exhaustion never reaches this branch now: the
            # bounded cross-slot chain above either accepts a bank-specific
            # brief or builds a validated source floor. This legacy-required
            # branch remains only for a typed NON-quality interpreter failure
            # and for explicit back-compat callers using the widget/env escape.
            # Chunk 3: the contract wrapper chains the underlying failure as
            # __cause__ (science: NewsInterpreterError). Stamp AND re-raise
            # from the CAUSE so the science halt surface stays byte-identical
            # (r1 M1 + r2 codex M2).
            _halt_exc = exc.__cause__ if exc.__cause__ is not None else exc
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
                # what the failed brief was. Chunk 3: stamp + re-raise
                # the UNDERLYING failure (_halt_exc = __cause__ when
                # present) -- science stamps/surfaces
                # "NewsInterpreterError: ..." exactly as pre-chunk-3.
                meta["news"] = None
                meta["news_briefs_halt_reason"] = (
                    f"{type(_halt_exc).__name__}: {_halt_exc}"
                )
                try:
                    led.save()
                except Exception:  # noqa: BLE001
                    pass
                raise _halt_exc
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
        cast_seed, cast_seed_source = _resolve_cast_rng_seed()
        cast_rng = _random.Random(cast_seed)
        # BUG-LOCAL-269 decoupled the CAST from the fixed seed widget so the
        # characters stopped being identical every episode. Their VOICES were
        # left behind: every voice draw, the announcer draw and the music base
        # RNG are seeded on meta["episode_seed"], and on the inline lanes
        # (shakespeare / public_domain / original / media_archive) nothing ever
        # wrote that key. coerce_int_seed(None) folds to a single constant, so
        # measured across 14 published episodes the announcer was bf_emma 14
        # times and 18 cast rows used 5 distinct voices. Same bug, one layer
        # down.
        #
        # Stamped HERE, at the mint, rather than 150 lines later beside
        # meta["cast_contract"]: the consumers read this key, and any path that
        # does not reach the cast-contract stamp would otherwise leave them on
        # the frozen constant. The value is deliberately the SAME number as
        # cast_contract.cast_seed -- credits prefer that key while voices and
        # music read this one, and equal values are what keeps the displayed
        # receipt honest about what actually rendered.
        #
        # BLAST RADIUS, stated plainly: episode_seed is folded into
        # stable_line_seed and therefore into the resolved-request identity and
        # the AUDIO CACHE KEY, so this also varies per-line synthesis and rekeys
        # cached audio. That is the intent -- a per-episode seed is supposed to
        # produce a per-episode render -- but it means the first run after this
        # lands re-synthesizes rather than reusing cached lines.
        #
        # UNCONDITIONAL, deliberately. `meta` is created fresh a few hundred
        # lines up (new_ledger -> setdefault("meta", {})), so on this path there
        # is nothing to preserve -- and a conditional stamp would leave room for
        # the two keys to diverge, which is worse than clobbering: credits print
        # cast_contract.cast_seed while voices and music read this one, so a
        # mismatch would print a receipt for a render that did not happen.
        # Content-owned lanes never reach this line and keep their own stamp in
        # the writer tail. OTR_CAST_SEED pins the INLINE lanes for the C7 gate;
        # the content-owned scifi_news_pro runner has its own OTR_SCIFI_NEWS_PRO_SEED.
        meta["episode_seed"] = int(cast_seed)
        log.info(
            "[OTR_LedgerScriptWriter] cast RNG seed=%d (%s) -- cast + voices + "
            "music randomized per episode (BUG-LOCAL-269)",
            cast_seed, cast_seed_source,
        )

        # THE style engine (style-engine consolidation, 2026-07-05): build
        # ONE StoryContract here -- the earliest point where BOTH cast_seed
        # and script_brief exist -- so the SAME radio style steers casting,
        # the macro prompt, the climax shape, and the body. This call site
        # moved up from just before the outline build (was ~150 lines
        # later, after lock_cast) because lock_cast's casting prompt also
        # needs the style label; a single style source per episode, used
        # everywhere, replacing the old three-way manual/LLM-picker/
        # combo resolver. OFF (story_scaffold=off) => contract stays None
        # => no style anywhere => byte-identical. build_story_contract
        # never raises on a missing style, but the call is wrapped LOUD
        # per CLAUDE.md so a defect can never break the writer.
        # BANK-LEVEL SCAFFOLD GATE (operator definition, 2026-08-03). "Make a
        # random radio drama" IS the original bank: its spark deck supplies the
        # randomness, and a catalog premise injected beside the pitch fought it
        # inside one prompt, lost the story, and still dressed the cast -- the
        # tempests_chart specimen locked a "Mining Union Representative...
        # hard hat with a union pin" into a Cartographer's Guild tale because
        # lock_cast was told style="asteroid-mining labor dispute". The bank
        # row now decides whether the scaffold runs at all
        # (banks.json defaults.story_scaffold, validated by _otr_story_routing;
        # absent means "on"). Bank OFF outranks the widget/env: the definition
        # of the bank lives with the bank, not with a per-run switch.
        # The gate FOLDS INTO _style_grammar_on rather than living beside it.
        # Every scaffold branch downstream -- the OutlineRequest style fields,
        # the safe-open capture, the news coda, the ctx handoff to the writer
        # tail (rebound at the top of _run_writer_tail), BOTH style-receipt
        # stamps -- keys on this one
        # variable (KILL 2's whole point), and bank-off must take the exact
        # documented byte-identical OFF path everywhere at once. A separate
        # flag would leave eight branches on the env value with contract=None,
        # and the second stamp lives in the ctx-rebound tail scope where a new
        # local would simply not exist.
        _bank_scaffold = str(
            (_source_bank_row.defaults or {}).get("story_scaffold", "on")
        ).strip().lower()
        _style_grammar_on = _style_grammar_on and _bank_scaffold != "off"
        contract = None
        if _style_grammar_on:
            try:
                # Source-owned lanes derive their sound world from the WORK,
                # not from the cast-seed draw. The drawn palette put "a fire
                # in the grate, a mantel clock, a teacup" over a Wells time-
                # travel chapter, and it reached the listener three ways: the
                # prompt grammar, this meta stamp, and the canon's sound
                # palette derived from it. Passing the derived world into
                # build_story_contract replaces all three at once, because
                # the grammar is rendered inside that call. Gated on
                # style_pool_class exactly like the arc-shape roll at :4325.
                _source_sound_world = ""
                _sound_world_receipt = {}
                if str(meta.get("style_pool_class") or "") == "adaptation":
                    _sd = resolved.get("source_document")
                    if _sd is not None:
                        _source_sound_world = _OTRWORLD.derive_source_sound_world(
                            _sd.canonical_body)
                        _sound_world_receipt = _OTRWORLD.sound_world_receipt(
                            _sd.canonical_body,
                            source_ref=str(resolved.get("source_ref", "") or ""),
                        )
                contract = _OTRSTYLE.build_story_contract(
                    cast_seed,
                    script_brief,
                    str(resolved.get("news_seed", "") or ""),
                    meta,
                    source_sound_world=_source_sound_world,
                )
                meta["story_contract"] = {
                    "slug": contract.slug,
                    "label": contract.label,
                    "ending_tag": contract.ending_tag,
                    # 2026-06-25: carry the selected style's sound_world into the
                    # ledger meta. It was DROPPED here before, so the episode
                    # canon's sound_palette (derived from it) was always empty
                    # even though the style had a rich audio world.
                    "sound_world": contract.sound_world,
                    # Body-free receipt: which elements were heard in the
                    # source and whether the neutral default was used, so
                    # "why does this episode sound like that" is auditable
                    # without the source text. Present only on source-owned
                    # lanes; folded into THIS literal rather than assigned
                    # afterwards because the story contract is stamped
                    # exactly once (KILL 2) and a second statement would be
                    # a second stamp site.
                    **({"sound_world_source": _sound_world_receipt}
                       if _sound_world_receipt else {}),
                }
            except Exception as _contract_exc:  # noqa: BLE001 -- never break the writer
                log.warning(
                    "[OTR_LedgerScriptWriter] story-contract build failed (%s); "
                    "style stays unset for this episode.", _contract_exc,
                )
                contract = None
        # Canonical story-style receipt: meta.style is derived ONLY from the
        # story contract. If the scaffold is off, stamp an explicit status
        # receipt instead; never borrow meta.visual_style for story metadata.
        _stamp_story_style_receipt(
            meta, contract=contract, scaffold_enabled=_style_grammar_on)

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
        # Read from `resolved`, not from the run() local: ONE resolution site,
        # already validated, so this path and the dispatched lanes can never
        # disagree about what the operator asked for.
        lemmy_force = resolved["lemmy_force"]
        with slot_scheduler.helper_context("lock_cast"):
            cast_rows, cast_meta = _OTRCAST.lock_cast(
                creative_fn=creative_generate_fn,
                num_characters=resolved["num_characters"],
                news_seed=resolved["news_seed"],
                casting_brief=casting_brief,
                style=(contract.label if contract else ""),
                rng=cast_rng,
                cast_seed=cast_seed,
                force_lemmy=lemmy_force,
                # Adaptation lanes only; None everywhere else (byte-identical C7).
                source_character_names=meta.get("_adaptation_character_names"),
                source_bank_id=str(
                    getattr(_source_bank_row, "source_bank_id", "") or ""
                ),
                # The source's own gender for each of those names, with the
                # evidence that justified it. Empty on every invention lane.
                source_character_genders=meta.get(
                    "_adaptation_character_genders"),
                # Bug Bible 11.61: names an EARLIER pass invented for the people
                # in this story, which this cast is about to override. lock_cast
                # drops any the assembled roster owns. The helper reads only
                # lane-neutral structured identity surfaces; it never mines a
                # free-text brief for Title-Case guesses.
                upstream_identity_names=_upstream_identity_names(meta),
            )
        led.set_cast(cast_rows)
        meta["cast_status"]           = "locked"
        meta["cast_contract_version"] = "cast-v1"
        meta["cast_contract"] = {
            "lemmy_hit":              cast_meta["lemmy_hit"],
            # The policy string has ONE spelling, and it lives beside the
            # decision type in _otr_casting. `lock_cast` always sets the key, so
            # this default is unreachable -- which is exactly why it was free to
            # drift out of step with the producer had it stayed a literal.
            "lemmy_policy": cast_meta.get(
                "lemmy_policy", _OTRCAST.LEMMY_POLICY_OPERATOR_CAMEO),
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
        # Bug Bible 11.61: persist the name-authority guard's findings WITH the
        # episode identity. `lock_cast` reports rows and cannot do this itself --
        # it has no episode id -- and 11.61's verify clause wants the episode AND
        # the row, because a detector that only says "this episode is dirty"
        # cannot be used to repair. Always stamped, so "checked and clean" is
        # distinguishable from "never ran".
        # NO `episode_id` COPY HERE, DELIBERATELY. `lock_cast` cannot name the
        # episode, so the writer persists these events INTO the episode's own
        # ledger -- which already carries `episode_id` at the top level. A second
        # copy would be an episode-local durable pointer that `rename_episode`
        # does not rebase (it rebases publication_eligibility by name), so it
        # would still read `pending_<ts>` after the real slug is assigned. That
        # exact staleness cost two published episodes on 2026-08-15: the terminal
        # mux compared a receipt's `pending_...` against the live slug, read it as
        # a stale singleton, and withheld the OBS copy on EVERY episode. One
        # authority for the episode id; the events carry the row.
        _name_authority = cast_meta.get("name_authority") or {}
        meta["name_authority"] = {
            "upstream_identities": list(_name_authority.get("upstream_identities") or []),
            "events": list(_name_authority.get("events") or []),
            "unfenced_mode": _name_authority.get("unfenced_mode", ""),
        }
        if meta["name_authority"]["events"]:
            log.warning(
                "[OTR_LedgerScriptWriter] name-authority guard fired on %d cast "
                "row(s); see meta.name_authority for the row-level detail",
                len(meta["name_authority"]["events"]),
            )
        # VC chunk 3 (2026-06-22): carry the per-character voice-fit slots
        # (gender/timbre/role/age_band/speech_signature/description_digest) into
        # the frozen ledger meta so OTR_CastLock's bank caster can match on
        # timbre/age, not just gender. Free-form meta -> ledger schema unchanged.
        # THE CAMEO ROLL RECEIPT. This copy is REQUIRED for the same reason the
        # ones around it are: `lock_cast`'s meta is not merged wholesale, it is
        # copied key by key right here, so a key stamped upstream and not named
        # on this line never reaches the ledger.
        #
        # Measured on disk 2026-08-24, and it is why this line exists: across
        # 1853 shipped ledgers, `scifi_news_pro` (which calls the decision
        # function directly) carried this receipt 24 times, while
        # `media_archive` (153 episodes), `original` (103) and `science_news`
        # (157) carried it ZERO times. Those lanes recorded WHETHER the cameo
        # landed and never WHY -- so a forced-off harness run and a roll that
        # came up short were indistinguishable, and the shipped rate could not
        # be read at all.
        #
        # `.get(..., {})` rather than a fabricated default, matching the
        # fail-closed convention of `voice_cast_mode` below: an empty receipt
        # must read as "upstream did not stamp one", never as a decision
        # nobody made.
        meta["lemmy_roll_receipt"] = cast_meta.get("lemmy_roll_receipt") or {}
        meta["cast_voice_slots"] = cast_meta.get("cast_voice_slots") or {}
        # VC chunk 4 (2026-06-22): carry the HYBRID LLM voice-fit decision
        # (proposed/accepted voice_ref_id + reproducibility keys) into the frozen
        # ledger meta so OTR_CastLock can honour the accepted proposal (and fall
        # closed to the deterministic scorer otherwise). Free-form meta.
        meta["voice_cast_decision"] = cast_meta.get("voice_cast_decision") or {}
        # WHICH CASTER RAN -- "scorer" | "hybrid" | "hybrid_unavailable".
        # This copy is exactly as REQUIRED as the ones around it: without the
        # line, `lock_cast` stamps the marker and the writer drops it, so no
        # published episode can say which caster produced its voices.
        #
        # FAIL-CLOSED ON PURPOSE -- `.get(key, "")`, never `or "scorer"`. A
        # default here would FABRICATE the marker if the upstream stamp were ever
        # dropped, leaving every acceptance check green while the thing it
        # asserts is dead. An empty value must fail the gate loudly instead.
        meta["voice_cast_mode"] = str(cast_meta.get("voice_cast_mode", ""))
        # The gender pin's receipt: which source names were pinned, to what, and
        # on what evidence. This copy is REQUIRED -- lock_cast's meta is not
        # merged wholesale, it is copied key by key right here, so a key stamped
        # in lock_cast and not named on this line never reaches the ledger.
        # Always present, empty-shaped on the invention lanes.
        meta["cast_source_contract"] = cast_meta.get("cast_source_contract") or {
            "source_bank_id": "", "character_names": [],
            "gender_by_name": {}, "evidence": {},
        }
        # Item 8 chunk 6 (2026-08-06): declare that this episode was produced
        # under the voice/portrait consistency contract, so the corpus audit can
        # tell a policy-era ledger from one of the 1,587 frozen before it and
        # hold each to the right standard.
        #
        # Stamped HERE, by the producer, before the freeze -- deliberately NOT by
        # CastLock. CastLock can be re-run over an old ledger, and if it wrote
        # this stamp it would silently promote that ledger to a contract its
        # producer never honoured. Absence means legacy, permanently.
        #
        # cast_lock_revision cannot stand in: it counts EXECUTIONS, not contract
        # versions. Nor can "is presentation_gender present?" -- that cannot tell
        # POLICY ABSENT from FIELD DROPPED, which is precisely the ambiguity that
        # let the spoken-citation receipt regress unnoticed for thirty episodes.
        from ._otr_roster_gender import (
            VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY,
            VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION,
        )
        meta[VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY] = (
            VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION)
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
        # Build the act topology from (act_count, include_act_breaks,
        # num_characters). `target_words` left this call 2026-08-14 along
        # with every word-derived field it used to produce.
        episode_budget = _OTRB.compute_episode_budget(
            act_count=resolved["act_count"],
            include_act_breaks=resolved["include_act_breaks"],
            num_characters=resolved["num_characters"],
        )
        log.info(
            "[OTR_LedgerScriptWriter] act topology: act_count=%d, "
            "arc_phases=%s, per_phase_beats=%s, music_inter=%d",
            episode_budget.act_count, list(episode_budget.arc_phases),
            list(episode_budget.per_phase_beats),
            episode_budget.music_inter_count,
        )
        # The StoryContract (`contract`) was already built earlier in this
        # function, right after cast_seed -- before lock_cast -- so its
        # style-label threads into the casting prompt too (style-engine
        # consolidation, 2026-07-05). Reused here for the outline.
        outline_req = _OTRO.OutlineRequest(
            news_seed=resolved["news_seed"],
            style=(contract.label if contract else ""),
            character_cast=character_cast,
            script_brief=script_brief,
            key_terms=key_terms_tuple,
            cast_descriptions=cast_descriptions,
            include_act_breaks=bool(resolved.get("include_act_breaks", True)),
            budget=episode_budget,
            prior_macro="",
            prior_critique="",
            style_grammar=(contract.grammar if contract else ""),
            story_engine=(contract.story_engine if contract else ""),
            work_title=_work_title,
        )
        # The first structurally valid outline is authoritative.
        with slot_scheduler.helper_context("generate_outline"):
            outline = _OTRO.generate_outline(
                creative_generate_fn,
                outline_req,
                creative_repo_id=resolved["creative_writing_model"],
                source_bank_id=resolved["source_bank"],
            )

        # Length is an OBSERVATION (2026-08-14). Nothing was requested, so
        # nothing is stamped as a target and there is no planned per-beat
        # allocation to record -- `Beat.target_words` no longer exists.
        _OTRWD.stamp_contract(
            meta,
            owner=f"inline:{resolved['source_bank']}",
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
                    # (the stray `200` removed 2026-08-28: it predated the
                    # no-prose-gate retirement and enforced nothing.)
                    _open_status_quo = _OTRLC.clean_one_line(
                        str(getattr(_b, "intent", "") or ""),
                    )
                    break
            safe_open_brief = _OTRLC.SafeOpenBrief(
                setting=str(getattr(outline, "setting", "") or ""),
                time_of_day=str(getattr(outline, "time_of_day", "") or ""),
                opening_status_quo=_open_status_quo,
                cast=tuple(character_cast),
                era=str(meta.get("period", "") or ""),
                work_title=_work_title,
            )

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
        # Split on BOTH delimiters. The catalog's drawn worlds are flat comma
        # lists, but a SOURCE-DERIVED world joins per-element phrases with
        # "; " and each phrase carries its own internal commas -- so a
        # comma-only split fused the tail of one element onto the head of the
        # next ("a bell over open water; night quiet" as a single palette
        # entry). The consistency guard only checks the field is non-empty,
        # so the garbling passed review while corrupting essentially every
        # adaptation episode's canon.
        _canon_sound_palette: list = []
        if contract is not None and getattr(contract, "sound_world", ""):
            _canon_sound_palette = [
                part.strip()
                for part in re.split(r"\s*[;,]\s*", str(contract.sound_world))
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
        # Optional source terms are prompt context only. Preserve their authored
        # wording; Python performs no noun/entity/cast-name classification and
        # their presence or absence never affects acceptance.
        try:
            from . import _otr_specificity as _OTRSPEC
        except ImportError:  # pragma: no cover
            import _otr_specificity as _OTRSPEC  # type: ignore
        _spec_kts = (meta.get("news") or {}).get("key_terms") or ()
        if "specificity_anchors" not in meta:
            meta["specificity_anchors"] = _OTRSPEC.derive_specificity_anchors(
                _spec_kts)
        if (not meta.get("_specificity_anchors_injected")
                and meta.get("specificity_anchors")):
            canon_header = _OTRSPEC.inject_anchors_into_header(
                canon_header, meta["specificity_anchors"])
            meta["_specificity_anchors_injected"] = True
            log.info(
                "[OTR_LedgerScriptWriter] injected %d optional source term(s) "
                "into the composition prompt",
                len(meta["specificity_anchors"]),
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
            # ADAPTATION lanes do not roll an arc. The source already made that
            # choice, and a rolled shape actively steers the dramatic-state
            # prompt AWAY from it -- "ARC SHAPE: heist" was injected over a man
            # demonstrating a time machine, and "betrayal" over a courtship
            # comedy, both reaching the listener through the credits scroll.
            # Gated on style_pool_class rather than the scaffold switch: bank
            # `original` is scaffold-off precisely so it can invent freely, and
            # its genre variety is wanted. This class is stamped well before
            # this point and is true ONLY for shakespeare + public_domain.
            _arc_lane_rolls = str(meta.get("style_pool_class") or "") != "adaptation"
            _arc_shape = ""
            if _arc_lane_rolls:
                try:
                    _arc_style_seed = os.environ.get("OTR_STYLE_SEED", "").strip()
                    _arc_news_hash = str(
                        (meta.get("news") or {}).get("source_hash") or ""
                    )
                    _arc_seed = (
                        _arc_style_seed + "|" + _arc_news_hash
                        + "|"
                        + str((meta.get("news") or {}).get("script_brief") or "")[:64]
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
            # `slot_scheduler` is CONSTRUCTED unconditionally far above and
            # dereferenced several times before this point, so the old
            # `if ... is not None` guard could never be False -- and its
            # `else` re-called `_derive_news_ds` with byte-identical arguments,
            # differing only by dropping the helper_context attribution. An
            # unreachable branch that silently loses a receipt is worse than no
            # branch (removed 2026-08-28).
            with slot_scheduler.helper_context("dramatic_state"):
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

        # Retired constraint-editor scoring is fully removed; no write-only
        # quality receipt or dormant repair architecture remains here.

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
        # free-text fields (line_job, hidden_pressure), validate schema and
        # derived-source identity per slot, and stamp on
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
                policy=resolved["llm_policy"],
                load_config=slot_scheduler.load_config_by_slot.get("technical"),
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
            # audit then records zero pivots without rejecting the contracts,
            # rather than pinning the turn on announcer/music rows.
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
        # `script_text_parts` (and the per-beat `[VOICE: ...]` token that fed
        # it) were removed 2026-08-28: the list was appended to on every voiced
        # beat and NEVER read -- not here, not by WriterTailContext, not by the
        # tail. Three comments called it a live diagnostic; a diagnostic
        # nothing can observe is just work. Slot-0's transcript authority is
        # `assemble_script_text_from_ledger`, which reads the saved ledger.
        last_lines: list = []  # rolling window of LAST_LINES_WINDOW

        base_temp = resolved["temperature"]

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

        # Cast names and source terms are prompt context only.
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

        # Style descriptor for the composer's STATIC prefix -- prompt-
        # facing, so it uses the contract's prose label (style-engine
        # consolidation, 2026-07-05). Empty string flips the STYLE
        # block off in _build_user_prompt when scaffold is off / no
        # contract.
        style_descriptor = str(contract.label if contract else "").strip()

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
        # THE SPOKEN FACT IS NOT ALWAYS THE INTERPRETER'S NOTE (2026-08-05).
        # On the fidelity lanes the interpreter is handed the source URL and
        # asked for an attribution note in the same payload, and that reply was
        # appended VERBATIM as the announcer's last line -- 84 spoken lines
        # across 30 episodes read a URL or a licence identifier on air, and the
        # captions burn `lines[].text` into the video, so it reached the screen
        # too. `provenance_coda_line` is the deterministic replacement and it
        # already exists; it simply had no reader.
        #
        # `nc_brief` is deliberately NOT rewritten: the receipt block and the
        # post-assembly key_terms audit both still read the interpreter's own
        # brief, and it remains the treatment "Sign-off" line.
        provenance_owned = "provenance" in meta
        effective_spoken_fact = (
            str(meta.get("provenance_coda_line") or "").strip()
            if provenance_owned else nc_brief
        )
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

            # Derive this character beat's prompt tension.
            _a5_tension = (
                int(_otr_tension_by_beat.get(beat.beat_id, 0))
                if not is_announcer else 0
            )
            return _OTRLC.LineRequest(
                speaker=speaker,
                intent=beat.intent,
                mood=beat.mood,
                canon_header=canon_header,
                last_lines=list(last_lines),
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
            )

        # Stage 3 reads the real outline beat directly. It validates only
        # nonempty transport shape and the shared narrow safety policy; no
        # synthetic Stage1Plan or vocabulary/length model is constructed.
        _w1b_stage3_enabled: bool = bool(resolved.get(
            "enable_production_stage3_validators", False,
        ))

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
            # Lane-enablement chunk 2 (2026-07-05): the exchange STATIC
            # system prompt resolves from the bank's pack seam via the
            # router's repo=None lane. DELIBERATELY OUTSIDE the PD1
            # try/except below: a pack/seam failure (a bank without the
            # exchange_system seam) must FAIL THE EPISODE LOUD, never be
            # swallowed into a silent legacy fallback (no-fallback law;
            # Fable forward-note pattern from chunk 1). Science is
            # byte-identical (pack == EXCHANGE_SYSTEM_PROMPT, test-pinned).
            from ._otr_creative_prompt_router import (
                resolve_creative_system_prompt as _resolve_ex_prompt,
            )
            _ex_system: str = _resolve_ex_prompt(
                None, phase="exchange_system",
                source_bank_id=resolved["source_bank"],  # lane chunk 2
            )
            try:
                from ._otr_compose_exchange import (
                    run_exchange_prepass as _run_ex_prepass,
                    make_tier_a_adapter as _make_tier_a,
                )
                from ._otr_craft_floor import (
                    evaluate_tier_a as _eval_tier_a,
                    normalize_slot_line as _norm_slot_line,
                )
                _ex_tier_a = _make_tier_a(
                    _eval_tier_a, _norm_slot_line,
                )
                _ex_cast = getattr(outline, "cast", None) or cast_rows or []
                with slot_scheduler.helper_context("compose_line"):
                    _ex_lines_by_beat_id = _run_ex_prepass(
                        list(outline.beats),
                        meta.get("slot_drama_contracts") or {},
                        list(_ex_cast),
                        generate_fn=creative_generate_fn,
                        tier_a_check=_ex_tier_a,
                        system_prompt=_ex_system,  # lane chunk 2
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
                    # narrative pass. (There is no `enable_polish_pass`
                    # widget and there never was: the name appeared only in
                    # two comments, describing a control the operator could
                    # not set. Narration-leak cleanup lives in the composer's
                    # own repair path.)
                    # S32 B6: helper_context attribution. Per-beat
                    # invocation; the context-manager overhead is
                    # constant-time and negligible relative to the LLM
                    # call itself.
                    #
                    # Stage 3 observes the exact cleaned line against the real
                    # outline beat. Findings are telemetry; it never reauthors
                    # or retires the line.
                    _w1b_s3_kwargs = (
                        {
                            "enable_stage3_validators": True,
                            "stage3_beat": beat,
                        }
                        if _w1b_stage3_enabled else {}
                    )
                    # LineCompositionFailedError had NO handler anywhere in
                    # nodes/, so a beat the composer could not fill read as a
                    # dead render with a bare traceback. It is not: the ledger
                    # already owns this exact case. A voiced row with nothing
                    # sayable is marked an EXPLICIT skip, with its reason, by
                    # `_otr_ledger_cleanup` at the writer tail -- which is why
                    # this leaves the row EMPTY rather than writing a line.
                    # Python authors no prose here and the freeze contract
                    # ("text non-empty OR skip=True with a reason") is met by
                    # its existing owner, not by a second one invented here.
                    try:
                        with slot_scheduler.helper_context("compose_line"):
                            line_res = _OTRLC.compose_line(
                                creative_fn=creative_generate_fn,
                                req=line_req,
                                base_temperature=base_temp,
                                max_new_tokens_cap=resolved["max_new_tokens_cap"],
                                creative_repo_id=resolved["creative_writing_model"],
                                source_bank_id=resolved["source_bank"],  # 2C
                                **_w1b_s3_kwargs,
                            )
                    except _OTRLC.LineCompositionFailedError as _line_exc:
                        log.error(
                            "[OTR_LedgerScriptWriter] beat %s (%s) produced no "
                            "spoken line after every attempt; the row stays "
                            "empty and the ledger cleanup marks it an explicit "
                            "skip. The episode continues one beat shorter "
                            "(LOUD). Attempts: %s",
                            beat.beat_id, beat.speaker,
                            "; ".join(
                                f"{reason}" for _raw, reason
                                in getattr(_line_exc, "attempts", ()) or ()
                            ) or "(none recorded)",
                        )
                        line_res = _OTRLC.LineResult(
                            text="",
                            compose_flags=("line_composition_failed",),
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

                cid = char_id_by_name[beat.speaker]
                # An unfillable beat enters NO context window: an empty
                # "previous line" is worse than no previous line for every beat
                # composed after it.
                if cleaned:
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
                        try:
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
                                # QA F1 (2026-07-09): pack-routed intro seam.
                                source_bank_id=resolved["source_bank"],
                            )
                        except _OTRLC.AnnouncerBriefStarvedError as _open_exc:
                            # A STARVED BRIEF MUST NOT KILL A RENDER HERE. The
                            # rewrite caller can decline and keep an existing
                            # line; this one is composing that line for the
                            # first time, so declining means the episode has no
                            # opening at all. The guard's job is to stop a bare
                            # form reaching the model -- it has already done
                            # that by raising -- so take the deterministic open
                            # and record it, rather than trading a weak line
                            # for a dead episode.
                            #
                            # Reachable because the outline schema pins
                            # `setting` at min_length=1, which admits a single
                            # space that cleans away to nothing; pair that with
                            # an empty first-character-beat intent and the brief
                            # carries no scene context.
                            log.warning(
                                "[OTR_LedgerScriptWriter] safe-open brief was "
                                "starved (%s) -- deterministic open, and the "
                                "episode continues (LOUD).",
                                _open_exc.reason,
                            )
                            line_res = _OTRLC.LineResult(
                                text=_OTRLC.fallback_safe_open(safe_open_brief),
                                compose_flags=(
                                    "announcer_intro",
                                    "announcer_intro_structural_fallback",
                                ),
                            )
                    cleaned = line_res.text
                    beat_compose_flags = line_res.compose_flags
                    if _style_grammar_on:
                        # The composer emits announcer_intro_structural_fallback
                        # on this path; it has never emitted the string this
                        # once tested for, so the receipt was permanently False
                        # and reported a clean safe-open on episodes that fell
                        # back. Same defect class as the `hook` attribute.
                        meta["open_safe_fallback"] = (
                            "announcer_intro_structural_fallback"
                            in line_res.compose_flags
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
                    # Same disposition as the character branch above: the row
                    # stays empty, the ledger cleanup marks the explicit skip
                    # with its reason, and no Python sentence goes on air.
                    # This is a MID-EPISODE announcer beat -- the opening and
                    # closing bookends are authored by their own passes and
                    # keep their own structural floors.
                    try:
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
                                source_bank_id=resolved["source_bank"],  # 2C
                            )
                    except _OTRLC.LineCompositionFailedError as _line_exc:
                        log.error(
                            "[OTR_LedgerScriptWriter] mid-episode announcer "
                            "beat %s produced no spoken line after every "
                            "attempt; the row stays empty and the ledger "
                            "cleanup marks it an explicit skip. The episode "
                            "continues one beat shorter (LOUD). Attempts: %s",
                            beat.beat_id,
                            "; ".join(
                                f"{reason}" for _raw, reason
                                in getattr(_line_exc, "attempts", ()) or ()
                            ) or "(none recorded)",
                        )
                        line_res = _OTRLC.LineResult(
                            text="",
                            compose_flags=("line_composition_failed",),
                        )
                    cleaned = line_res.text
                    beat_compose_flags = line_res.compose_flags
                if cleaned:
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
            # beat. Fail loud.
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
            _line_fields = {
                "char_id":       cid,
                "traits":        traits,
                "compose_flags": list(beat_compose_flags),
            }
            _OTRL.patch_line_fields(led.data, beat.beat_id, _line_fields)
            led.save()

        # --- I.4.9. Post-composition announcer-intro REWRITE ----------
        # (INTRO_REWRITE_SPEC 2026-07-09, kibitz r2-r4, shape A.) The
        # in-loop intro was starved to the PRE-GEN SafeOpenBrief
        # (outline-derived, section F above); now the script exists,
        # derive a PRODUCED open brief from scene-1 rows + cast (input
        # starvation again -- the derive pass never sees past scene 1)
        # and recompose the intro through the SAME routed safe-open
        # composer. Runs BEFORE the I.5 outro pass so the outro
        # tone-echo (the ledger read below) sees the FINAL intro, and
        # BEFORE the J aggregates so flags/word counts stay fresh.
        # Failure posture: keep the in-loop intro (REAL composed
        # content, not a canned template -- same never-raise family as
        # K.5.5/K.5.6), stamp announcer_intro_rewrite_failed, log LOUD.
        # Only the derive + compose calls sit inside the try (kibitz r4
        # P1): row lookup/patch runs OUTSIDE it and RAISES on a missing
        # row (corruption, not a rewrite failure). Degenerate outline
        # (first == last announcer beat: the single bookend row) skips
        # the rewrite entirely -- mirror of the outro-pass guard.
        if (
            first_announcer_id is not None
            and last_announcer_id is not None
            and first_announcer_id != last_announcer_id
        ):
            _rw_text = None
            _rw_flag = "announcer_intro_rewrite_failed"
            _rw_compose_flags = ()
            _rw_reason = "compose_failed"
            try:
                # LLM slot: technical -- structured scene-1 derive.
                with slot_scheduler.helper_context(
                    "derive_produced_open_brief"
                ):
                    _rw_brief = derive_produced_open_brief(
                        led,
                        first_announcer_line_id=first_announcer_id,
                        technical_fn=technical_generate_fn,
                        technical_model_id=str(resolved["technical_model"]),
                    )
                # LLM slot: creative -- the SAME routed safe-open intro
                # composer the in-loop pass uses; story_scaffold=True
                # UNCONDITIONALLY (the rewrite is defined for all banks,
                # independent of the style-grammar lever). era threads
                # meta["period"] for parity with the in-loop
                # SafeOpenBrief construction (timeless lanes starve it).
                with slot_scheduler.helper_context(
                    "announcer_intro_rewrite"
                ):
                    _rw_res = _OTRLC.compose_announcer_intro(
                        creative_fn=creative_generate_fn,
                        script_brief="",
                        creative_repo_id=resolved["creative_writing_model"],
                        story_scaffold=True,
                        safe_open_brief=_OTRLC.SafeOpenBrief(
                            setting=_rw_brief.setting,
                            time_of_day=_rw_brief.time_of_day,
                            opening_status_quo=_rw_brief.opening_status_quo,
                            cast=tuple(_rw_brief.cast),
                            era=str(meta.get("period", "") or ""),
                            # From `meta`, exactly as `era` is -- NOT recovered
                            # from the scene-1 dialogue this rewrite reads. The
                            # title is an immutable value we already hold;
                            # parsing it back out of prose would be a
                            # non-deterministic route to a known fact. Without
                            # this line the rewrite silently restores the
                            # wrong-play frame the first producer just fixed.
                            work_title=_work_title,
                        ),
                        source_bank_id=resolved["source_bank"],
                    )
                if (
                    "announcer_intro_structural_fallback"
                    in _rw_res.compose_flags
                ):
                    # A STRUCTURAL FALLBACK IS NOT A REWRITE. The composer
                    # never returns empty text on failure -- it returns
                    # fallback_safe_open(), a deterministic template -- so
                    # treating "it returned something" as success replaced a
                    # real composed opening with a canned line and stamped it
                    # rewritten. The keep-the-in-loop-intro posture below only
                    # ever fired on a RAISE, which a model failure is not.
                    # Drop the flags too: extra_flags are appended even when
                    # the text is not, so the preserved row would otherwise
                    # collect the structural flag it did not earn.
                    _rw_reason = "structural_fallback"
                    log.warning(
                        "[OTR_LedgerScriptWriter] announcer intro REWRITE "
                        "returned a structural fallback -- keeping the "
                        "in-loop intro (LOUD).",
                    )
                else:
                    _rw_text = _rw_res.text
                    _rw_compose_flags = _rw_res.compose_flags
                    _rw_flag = "announcer_intro_rewritten"
                    _rw_reason = None
            except _OTRLC.AnnouncerBriefStarvedError as _rw_starved:
                # The derive produced a brief nothing can open on. Bounded
                # reason, read off the exception rather than parsed out of it.
                _rw_reason = _rw_starved.reason
                log.warning(
                    "[OTR_LedgerScriptWriter] announcer intro REWRITE "
                    "skipped (%s) -- keeping the in-loop intro (LOUD).",
                    _rw_reason,
                )
            except (StructuredCallFailedError, ValueError) as _rw_derive_exc:
                # Both are DELIBERATE derive outcomes: the ladder exhausting,
                # and _otr_story_brief raising ValueError when scene 1 has no
                # spoken rows. Classifying them as unexpected would file an
                # intended failure as a coding defect.
                _rw_reason = "derive_failed"
                log.warning(
                    "[OTR_LedgerScriptWriter] announcer intro REWRITE derive "
                    "failed (%s: %s) -- keeping the in-loop intro (LOUD).",
                    type(_rw_derive_exc).__name__, str(_rw_derive_exc)[:200],
                )
            except Exception as _rw_exc:  # noqa: BLE001 -- a polish pass
                # must never kill a finished episode; the in-loop intro
                # stands (it is real composed content, not a template).
                # Anything reaching HERE is unexpected -- an AttributeError or
                # TypeError is a bug in this code, not a starved brief, and it
                # gets a traceback so it cannot hide as routine starvation.
                _rw_reason = "uncaught_%s" % type(_rw_exc).__name__
                log.error(
                    "[OTR_LedgerScriptWriter] announcer intro REWRITE raised "
                    "an UNEXPECTED %s -- keeping the in-loop intro (LOUD).",
                    type(_rw_exc).__name__, exc_info=True,
                )
            _apply_intro_rewrite_result(
                led, first_announcer_id, _rw_text, _rw_flag,
                _rw_compose_flags,
            )
            meta.setdefault("announcer_intro_rewrite", {})["status"] = (
                _rw_flag
            )
            # The reason a later audit reads instead of inferring the cause
            # from logs that no longer exist. None on success; a closed
            # vocabulary otherwise.
            meta["announcer_intro_rewrite"]["reason"] = _rw_reason
            led.save()
            log.info(
                "[OTR_LedgerScriptWriter] announcer intro rewrite: %s "
                "(line %s)",
                _rw_flag, first_announcer_id,
            )

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
            outro_res = _compose_and_stamp_announcer_close(
                led,
                meta,
                first_announcer_id=first_announcer_id,
                last_announcer_id=last_announcer_id,
                provenance_owned=provenance_owned,
                style_grammar_on=_style_grammar_on,
                effective_spoken_fact=effective_spoken_fact,
                nc_brief=nc_brief,
                script_brief=script_brief,
                premise=str(getattr(outline, "premise", "") or ""),
                resolved=resolved,
                slot_scheduler=slot_scheduler,
                creative_generate_fn=creative_generate_fn,
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
                led.data["lines"], nc_key_terms,
            )
            meta["post_assembly_key_terms"] = {
                "status": "telemetry_only",
                "landed": landed,
                "missing": missing,
            }
            log.info(
                "[OTR_LedgerScriptWriter] optional post-assembly key-term "
                "telemetry: %d/%d landed (missing=%r)",
                len(landed), len(nc_key_terms), missing,
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

        # --- Tail handoff (scifi_news_pro S1a extraction) ----------------
        # Everything from J.5 to the M save lives in _run_writer_tail and
        # consumes ONLY the context below. The legacy path builds it from
        # its locals: final_title_override=None (LLM regen path as today)
        # and run_story_spine=True (the env-gated spine default decides
        # inside the tail) keep the behavior byte-identical.
        _tail_ctx = WriterTailContext(
            led=led,
            meta=meta,
            resolved=resolved,
            outline_view=outline,
            canon=canon,
            episode_root=episode_root,
            episode_id=episode_id,
            contract=contract,
            style_grammar_on=_style_grammar_on,
            source_bank_row=_source_bank_row,
            slot_scheduler=slot_scheduler,
            creative_fn=creative_generate_fn,
            technical_fn=technical_generate_fn,
            run_story_spine=True,
            final_title_override=None,
            style_roll=_style_roll,
        )
        return self._run_writer_tail(_tail_ctx)

# ---------------------------------------------------------------------------
# Self-test (no-model smoke)
# ---------------------------------------------------------------------------
