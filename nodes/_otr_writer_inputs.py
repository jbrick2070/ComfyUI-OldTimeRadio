"""``_resolve_inputs`` -- every widget value the writer is handed, resolved once.

Lean-mean order 9, slice 1. The writer module was 7,417 lines and this function
was 398 of them: the ONE place a raw widget string becomes a resolved decision.
It reads the operator's typed values, validates the ones with a legal range,
resolves the source bank and its fetcher, seeds the RSS lane, picks the models,
and returns the ``resolved`` dict that every downstream lane reads. Nothing in
here composes a line, calls an LLM, or touches the ledger.

Moved BYTE-IDENTICALLY, and the campaign asked for the proof rather than the
promise: every block was cut by line range, hashed, and re-hashed in place here.
The writer imports all of it back under its original names, so
``OTR_LedgerScriptWriter._resolve_inputs`` and every constant a test reaches for
still resolve exactly where they always did. This is a SEAM, not a rewrite.

The widget CHOICE LISTS came with it on purpose. ``_CREATIVITY_CHOICES``,
``_LEMMY_CAMEO_CHOICES``, ``_ACT_COUNT_CHOICES`` and ``_FABLE2_MAX_CAST`` are the
menus whose values this function is the sole interpreter of; leaving them behind
would have split one contract across two files and forced an import back into
the writer, which is the cycle this split exists to avoid. Their ORDER and
positional meaning are untouched -- the widget-schema epoch (order 10) is a
separate, workflow-atomic change and this is not it.

Stdlib plus the writer's own lazy sibling modules. It never imports the writer.
UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

# The same lazy, stdlib-only sibling modules the writer imports, in the same
# spelling. None of them import back into the writer or into this module.
from . import _otr_model_catalog as _otr_model_catalog
from . import _otr_source_payload as _otr_source_payload
from . import _otr_source_snapshot as _otr_source_snapshot
from . import _otr_story_routing as _otr_story_routing
from ._otr_shared import llm_policy as _llm_policy

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

try:                                            # package import
    from . import _otr_episode_budget as _OTRB
except ImportError:                              # flat/script import
    import _otr_episode_budget as _OTRB          # type: ignore

log = logging.getLogger("OTR")


#: Widget bound for `num_characters`, mirroring `_otr_scifi_news_pro`'s
#: `MAX_SPEAKING_CAST` (one speaking character per distinct voice in stock).
#:
#: DUPLICATED ON PURPOSE, not imported. `INPUT_TYPES` runs at node-registration
#: time, and reaching into the writer module from there would make widget
#: construction depend on import order -- a far worse failure than a number.
#: `tests/test_cast_size_is_a_request.py` asserts the two agree, so drift is
#: reported rather than silently shipped.
_FABLE2_MAX_CAST = 10


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


def _resolve_creativity(creativity: str) -> tuple[float, float]:
    """Map a creativity widget value to (temperature, top_p).

    Unknown values default to balanced (0.85 / 0.95). Returns floats.
    """
    temp = _CREATIVITY_TEMP_MAP.get(creativity, _CREATIVITY_TEMP_MAP["balanced"])
    top_p = _CREATIVITY_TOP_P_MAP.get(creativity, _CREATIVITY_TOP_P_MAP["balanced"])
    return (float(temp), float(top_p))


#: Act count used when the widget value is missing or out of range. Three
#: acts is the classic radio-drama shape (setup / complication / resolution)
#: and was the previous auto-derived default for a normal-length episode.
#: It is NOT derived from anything -- deriving it is what was removed.
_DEFAULT_ACT_COUNT: int = 3

#: Operator-facing act choices. Explicit 1..6 (narrowed from 1..8,
#: PBUG-20260825-01 -- 7 and 8 always overflowed Outline.beats' own
#: max_length=32); there is deliberately no 'auto' option, because 'auto'
#: meant "derive from target_words". DERIVED from MIN/MAX_ACT_COUNT, not a
#: second hardcoded range -- the two stay in lockstep by construction.
_ACT_COUNT_CHOICES: list[str] = [
    str(n) for n in range(_OTRB.MIN_ACT_COUNT, _OTRB.MAX_ACT_COUNT + 1)
]


def _resolve_inputs(
    episode_title: str = "",
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
    # Back-compat fail-loud lever. Model-quality exhaustion is now handled by
    # the bounded cross-slot chain + validated source floor before this lever;
    # True still prevents a typed non-quality interpreter failure from silently
    # degrading to meta["news"]=None.
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
    # Stage 2C (2026-07-05): the story-path source_bank widget selection.
    # Threaded into the resolved dict as the ONE authoritative value for
    # meta/ledger stamping + prompt threading. Already gated runnable by
    # run() (require_runnable_bank fires before this call).
    source_bank: str = "scifi_news_pro",
    # Stage 3C (2026-07-06): the visual_style widget selection; same
    # authoritative-value contract (gated by resolve_visual_style in run()).
    visual_style: str = "sci_fi_radio",
    # Google BYO API direct LLM lane (2026-07-08). Stable handles stay in
    # creative/technical_model; concrete Gemini model ids live here.
    google_api_slot_a_model: str = "",
    google_api_slot_b_model: str = "",
    # Source Banks v2 (2026-07-08): optional external source reference for
    # source-bank lanes. Blank is intentionally inert until a bank consumes it.
    source_ref: str = "",
    # S1 platform-portability (2026-07-10): the explicit LLM runtime policy
    # fields. Defaults EQUAL today's resolved nv50 16 GB baseline, so the
    # explicit policy reproduces current behavior exactly; the S5 writer
    # widgets feed these 1:1 (llm_device .. gguf_quant, append-only).
    llm_device: str = "cuda",
    llm_attn_impl: str = "sdpa",
    llm_quant_policy: str = "bnb_nf4",
    llm_vram_ceiling_gb: float = 14.5,
    gguf_n_ctx: int = 4096,
    gguf_quant: str = "Q8_0",
    # GGUF row registry (2026-07-16): the preflight-resolved technical-slot
    # load_config + the ONE validated policy, threaded into the RSS fetch/
    # rerank dispatch so a gguf technical slot reranks under its real per-row
    # load_config (path/quant/n_ctx) instead of the gemma env fallback. Both
    # None on a non-gguf run (request_slot then resolves from the policy).
    preflight_policy: Any = None,
    technical_load_config: Any = None,
    # The LEMMY cameo knob, resolved HERE so every lane reads one answer.
    # Defaulted from the choices list rather than a repeated literal: the two
    # spellings drifting apart is the failure this parameter exists to end.
    lemmy_cameo: str = _LEMMY_CAMEO_CHOICES[0],
) -> dict:
    """Resolve raw widget values into the effective set used by the run.

    Returns a single dict. Logs at INFO for branches that override the
    widget value (RSS fetch, smoke preset).

    Story: custom_premise verbatim > RSS auto-fetch.

    Style-engine consolidation (2026-07-05): there is no `style` widget
    input anymore. Every episode's style comes from exactly ONE call --
    ``_otr_style_catalog.build_story_contract()`` -- made later in
    ``run()`` once cast_seed and script_brief both exist. The old
    three-way style_custom/combo/LLM-picker resolver is gone.
    """
    # THE CAMEO KNOB IS VALIDATED FIRST, before any RSS fetch or source work.
    # EXACT membership, and a typo FAILS LOUD naming the three choices. The
    # legacy path used `_LEMMY_CAMEO_FORCE.get(...)`, which turns an unknown
    # string into None -- i.e. silently into the natural roll -- so a misspelled
    # "always include" read as "leave it to chance" and nobody could tell the
    # difference from the outside. Failing here costs nothing; failing after a
    # fetch costs a network round trip to learn a widget was mistyped.
    if lemmy_cameo not in _LEMMY_CAMEO_FORCE:
        raise ValueError(
            "lemmy_cameo=%r is not one of %s"
            % (lemmy_cameo, ", ".join(repr(c) for c in _LEMMY_CAMEO_CHOICES))
        )

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

    # A REQUEST, not a cap (operator directive 2026-08-12, all banks). The
    # only real ceiling is the voice stock, enforced in the writer against
    # `MAX_SPEAKING_CAST`, because two characters never share a voice.
    num_characters = max(1, min(_FABLE2_MAX_CAST, int(num_characters)))

    # ACT COUNT IS THE ONLY LENGTH-SHAPED KNOB (operator directive
    # 2026-08-14). The widget is an explicit 1..6 combo (narrowed from 1..8,
    # PBUG-20260825-01). There is no
    # 'auto' any more: 'auto' meant "derive the act count from
    # target_words", and target_words no longer exists. Nor is the pick
    # validated against a derived [default..max] band -- that band came
    # from the word total too, which meant a word count could REFUSE an
    # operator's act choice. An out-of-range value falls back to the
    # default rather than failing the render.
    try:
        act_count_int = int(str(act_count).strip())
    except (TypeError, ValueError):
        act_count_int = _DEFAULT_ACT_COUNT
    if not (_OTRB.MIN_ACT_COUNT <= act_count_int <= _OTRB.MAX_ACT_COUNT):
        log.warning(
            "[OTR_LedgerScriptWriter] act_count=%r out of range [%d, %d] "
            "-- using %d",
            act_count, _OTRB.MIN_ACT_COUNT, _OTRB.MAX_ACT_COUNT,
            _DEFAULT_ACT_COUNT,
        )
        act_count_int = _DEFAULT_ACT_COUNT
    temperature, top_p = _resolve_creativity(creativity)
    custom = (custom_premise or "").strip()

    # kibitz r2-r4: bank-shape dispatch BEFORE the custom check (r3 D7).
    # A source-contract-free bank (original_radio) has no fetch lane --
    # resolve_fetcher below raises SourceContractMissingError LOUD by
    # design -- and no source article exists yet: the creative front runs
    # at D.2. The A-branch draws the spark entropy (single entropy point)
    # and synthesizes the EXACT 7-key payload from the draw digest
    # (seed_text non-empty is the only content requirement). A typed
    # custom_premise on this lane is NOT a source article: it rides
    # source_meta["operator_hint"] into the concept pass as material
    # (kibitz r4 P2) -- never the payload.
    _rb_bank = _otr_story_routing.get_bank(source_bank or "scifi_news_pro")
    # Bake-off source-snapshot replay (r3 ruling B7). Loaded IMMEDIATELY after
    # bank resolution and BEFORE the three source branches so a frozen source
    # replays across the base/_v2/_v3 triplet -- the ONLY variable under test is
    # the pack, never a fresh RSS draw or a random spark. None => no snapshot for
    # this bank; the live path below runs unchanged. A mismatched/malformed
    # envelope raises SourceSnapshotError LOUD (never a silent fall-through to
    # live sourcing). The replayed source_meta sidecar carries the same fields a
    # live branch would (spark_atoms for the original lane, cast_hints for the
    # adaptation lanes), so every downstream owner is fed unchanged.
    _source_snapshot = _otr_source_snapshot.load_snapshot_for_bank(
        source_bank or "scifi_news_pro",
    )
    # Only the live fetch branch can produce one: the snapshot envelope is the
    # seven-key payload whose full_text is already the capped projection, and
    # the original / custom-premise lanes have no source work to ground in.
    # None means "no whole-body grounding available", which every consumer
    # must handle rather than assume.
    source_document = None
    if _source_snapshot is not None:
        news_article = _otr_source_payload.validate_source_payload(
            _source_snapshot.payload,
            origin="_resolve_inputs source_snapshot",
        )
        news_seed = _source_snapshot.seed_text or news_article["seed_text"]
        seed_source = _source_snapshot.seed_source
        source_meta = dict(_source_snapshot.source_meta)
        source_rights = dict(_source_snapshot.source_rights)
        log.info(
            "[OTR_LedgerScriptWriter] source-snapshot REPLAY: bank=%r base=%r "
            "seed_source=%r sha=%s",
            source_bank or "scifi_news_pro",
            _source_snapshot.base_source_bank_id,
            seed_source,
            _source_snapshot.payload_sha256[:12],
        )
        # The source is frozen, but cast/style still roll unless C7 pins them.
        # A replay leg meant as a controlled A/B (F2) needs OTR_C7=1; warn LOUD
        # if the seeds are unset so a mis-run does not masquerade as a control.
        if not (otr_env.get("OTR_CAST_SEED", "").strip()
                and otr_env.get("OTR_STYLE_SEED", "").strip()):
            log.warning(
                "[OTR_LedgerScriptWriter] source-snapshot REPLAY without C7 seed "
                "pinning (OTR_CAST_SEED/OTR_STYLE_SEED unset): the SOURCE is "
                "frozen but cast/style will roll fresh -- set OTR_C7=1 for a "
                "byte-stable replay leg.",
            )
    elif _bank_has_no_source_contract(_rb_bank):
        try:
            from . import _otr_original_radio as _OTROR
        except ImportError:  # pragma: no cover -- flat standalone load
            import _otr_original_radio as _OTROR  # type: ignore
        _spark = _OTROR.draw_spark_atoms()
        news_article = _otr_source_payload.validate_source_payload({
            "headline":  "Original Radio Drama - " + _spark.digest,
            "summary":   "",
            "full_text": _spark.digest_long,
            "source":    "Original (LLM)",
            "date":      datetime.now().date().isoformat(),
            "link":      "",
            "seed_text": _spark.digest,
        }, origin="_resolve_inputs original_radio")
        news_seed = _spark.digest
        seed_source = "original_llm"
        source_meta = {
            "kind": "original_llm",
            "spark_atoms": dict(_spark.atoms),
            "deck_version": _spark.deck_version,
            "deck_hash": _spark.deck_hash,
        }
        if custom:
            source_meta["operator_hint"] = custom
            log.info(
                "[OTR_LedgerScriptWriter] original lane: custom_premise "
                "riding as operator_hint (%d chars)", len(custom),
            )
        source_rights = {"license_label": "synthetic original"}
    elif custom:
        # Custom premise path: synthesize the same dict shape RSS
        # would produce so news_interpreter sees a uniform article
        # surface no matter how the story entered the writer.
        news_article = _otr_source_payload.validate_source_payload({
            "headline":  "",
            "summary":   "",
            "full_text": custom,
            "source":    "User Seed",
            "date":      "",
            "link":      "",
            "seed_text": custom,
        }, origin="_resolve_inputs custom_premise")
        news_seed = custom
        seed_source = "custom_premise"
        source_meta = {}
        source_rights = {}
    else:
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
        # Chunk 3 (2026-07-05): the fetch routes through the bank's declared
        # fetcher contract (science_news -> science_rss -> the verbatim
        # _fetch_rss_seed_or_die call, byte-identical). A bank without a
        # built fetcher lane raises SourceContractMissingError LOUD here --
        # never a silent slide into the science path. Resolution sits OUTSIDE
        # any try/except by design (no swallow).
        # Style-engine consolidation (2026-07-05): the fetch is
        # style-agnostic now -- there is no style value yet at this
        # pre-contract sourcing stage, and none is needed for rerank.
        # Independent source banks wave 3: a CLIENT bank may own its fetch
        # lane. `user_bank_bundle` is None for every shipped bank, so this is a
        # no-op on the six; for a client bank it hands resolution the ONE
        # bundle allowed to execute for that id. The result still flows through
        # normalize_fetch_result below -- client code never reaches the ledger.
        _fetch_bank = _otr_story_routing.get_bank(source_bank or "scifi_news_pro")
        _fetch_owner = _otr_story_routing.user_bank_bundle(
            _fetch_bank.source_bank_id)
        _fetch_entry = _otr_source_payload.resolve_fetcher(
            _fetch_bank, owner=_fetch_owner)
        _fetch_origin = (
            f"_resolve_inputs fetch (bank={_fetch_bank.source_bank_id!r}, "
            f"fetcher={_fetch_bank.fetcher!r})"
        )
        # The 4-value normalizer additionally carries the TRANSIENT source
        # document -- the complete uncapped body for source-owned lanes. The
        # payload's full_text is a 12,000-char projection, so the pre-outline
        # authors would otherwise read a prefix of a work that can run 25,000
        # words. The document is deliberately kept OUT of source_meta, which
        # is copied into durable ledger metadata at :3548.
        news_article, source_meta, source_rights, source_document = (
            _otr_source_payload.normalize_fetch_result_with_document(
                _fetch_entry.fetch(
                    bank=_fetch_bank,
                    technical_model=technical_model,
                    source_ref=source_ref,
                    load_config=technical_load_config,
                    policy=preflight_policy,
                ),
                origin=_fetch_origin,
            )
        )
        news_seed = news_article["seed_text"]
        seed_source = _fetch_entry.seed_source

    source_meta = dict(source_meta or {})
    news_seed_receipt = source_meta.pop("_news_seed_receipt", {})

    # Source identity is selected-source truth, not merely the requested
    # widget value. RSS lanes ignore/leave that widget blank and choose a real
    # item at fetch time; manifest lanes already return the resolved ref in
    # their sidecar. Preserve a differing operator request separately for
    # forensics, then expose the selected ref as the canonical ledger field.
    _requested_source_ref = str(source_ref or "").strip()
    _selected_source_ref = str(
        (source_meta or {}).get("source_ref")
        or (news_article or {}).get("link")
        or _requested_source_ref
        or ""
    ).strip()
    if _requested_source_ref and _requested_source_ref != _selected_source_ref:
        source_meta.setdefault("requested_source_ref", _requested_source_ref)

    return {
        "news_seed":            news_seed,
        "news_article":         news_article,
        "seed_source":          seed_source,
        "num_characters":       num_characters,
        "episode_title":        (episode_title or "").strip(),
        # S30 B2b: per-slot keys ONLY. The legacy `model_id` key is
        # deleted outright; consumers route via creative_writing_model
        # / technical_model. No "stamp both" hedge.
        "creative_writing_model": creative_writing_model,
        "technical_model":        technical_model,
        # S1 platform-portability: the ONE frozen policy object threaded
        # _SlotScheduler -> request_slot -> backend.load. Validated at
        # construction (LLMPolicyError on a bad enum -- fail loud here,
        # before any model work).
        "llm_policy": _llm_policy.LLMRuntimePolicy(
            device=str(llm_device),
            attn_impl=str(llm_attn_impl),
            quant_policy=str(llm_quant_policy),
            vram_ceiling_gb=float(llm_vram_ceiling_gb),
            gguf_n_ctx=int(gguf_n_ctx),
            gguf_quant=str(gguf_quant),
        ),
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
        # Stage 2C: the ONE authoritative source_bank value for prompt
        # threading + meta/ledger stamping (run() gated it runnable already).
        "source_bank": str(source_bank or "scifi_news_pro"),
        # Stage 3C: the ONE authoritative visual_style value (gated in run()).
        "visual_style": str(visual_style or "sci_fi_radio"),
        "google_api_slot_a_model": str(google_api_slot_a_model or ""),
        "google_api_slot_b_model": str(google_api_slot_b_model or ""),
        "source_ref": _selected_source_ref,
        "source_meta": dict(source_meta),
        "source_rights": dict(source_rights),
        "news_seed_receipt": dict(news_seed_receipt),
        # TRANSIENT -- deliberately its own key rather than a source_meta
        # field, because source_meta is copied wholesale into durable ledger
        # metadata and this holds the complete work. Nothing may stamp it.
        "source_document": source_document,
        # The cameo decision as a tri-state: None = roll, True/False = forced.
        # Every consumer -- the legacy block and every dispatched lane runner --
        # reads THIS key. The widget string is resolved exactly once, here.
        "lemmy_force": _LEMMY_CAMEO_FORCE[lemmy_cameo],
    }


def _bank_has_no_source_contract(bank) -> bool:
    """The original-lane runtime dispatch (kibitz r2-r4, r4 P8).

    A RUNNABLE bank with NEITHER fetcher NOR interpreter is the original
    lane: sweep rule 4a/4b guarantees every runnable source-contract bank
    declares both ids. The runnable conjunct keeps custom_source_bank
    (empty-empty but runnable:false) on its pinned LOUD path: in run() it
    dies at require_runnable_bank; a DIRECT _resolve_inputs call on it
    falls through to resolve_fetcher's SourceContractMissingError
    (test_source_payload_chunk3 pin). This is BANK-row data --
    pipeline.requires_source_contract stays validation-time-only per the
    registry law (_otr_story_routing.py:88-91)."""
    return (
        bool(getattr(bank, "runnable", False))
        and not getattr(bank, "fetcher", "")
        and not getattr(bank, "interpreter", "")
    )
