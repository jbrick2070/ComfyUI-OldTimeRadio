"""Source-payload fetcher/interpreter contracts -- lane-enablement chunk 3.

Makes the ``fetcher`` / ``interpreter`` ids on banks.json rows LIVE routing
coordinates behind a typed, fail-loud contract. Plan of record:
docs/multimodal-story-schema/CHUNK3_SOURCE_PAYLOAD_SUBPLAN.md (v5 FINAL,
kibitz r1-r4 converged 2026-07-05).

SCOPE: this contract is the ``legacy_many_pass`` ARTICLE ADAPTER -- the uniform
payload surface that pipeline's ``source_interpret`` pass consumes. It is NOT a
universal source packet: lane-specific provenance (public-domain author/rights,
media-archive URL/sha) rides each lane's enablement WITH its consumer, per the
Stage-2 "no fields without consumers" law. The simple_4 pipeline never touches
this contract.

Law: JSON owns content/config (banks.json carries the ids); Python owns
validation/routing/execution (this module carries the callables); NO fallbacks;
unknown/missing id = hard typed error -- a selected bank without a built lane
FAILS THE EPISODE LOUD, never a silent slide into the science path.

Import posture: stdlib-only at import time. This module imports NEITHER the
writer NOR news_interpreter NOR _otr_story_routing at module level (three-edge
cycle guard, test-pinned); the heavy imports happen LAZILY inside the wrapper
bodies. Zero file I/O at import. The ``bank`` parameter is DUCK-TYPED
(``.fetcher`` / ``.interpreter`` / ``.source_bank_id`` attributes) -- no
runtime routing import.

Registry metadata vs execution errors:
  - the routing SWEEP (in _otr_story_routing) validates ids at registry load
    and raises RegistryValidationError there;
  - this module's Unknown*/Missing errors are DEFENSE-IN-DEPTH for direct or
    synthetic callers -- both layers exist deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# typed errors (own hierarchy -- deliberately NOT StoryRoutingError subclasses:
# routing errors are registry-load problems, payload errors are execution
# problems)
# ---------------------------------------------------------------------------


class SourcePayloadError(Exception):
    """Base: any fail-loud source-payload contract problem."""


class UnknownFetcherError(SourcePayloadError):
    """A bank carries a non-empty fetcher id that is not registered."""


class UnknownInterpreterError(SourcePayloadError):
    """A bank carries a non-empty interpreter id that is not registered."""


class SourceContractMissingError(SourcePayloadError):
    """A bank with an EMPTY fetcher/interpreter id reached resolution.

    Its source-payload lane is not built yet (lane-enablement checklist,
    STAGE2_SUBPLAN.md section 4b item 3). There is no fallback."""


class SourcePayloadContractError(SourcePayloadError):
    """A payload dict or interpreter result violates the contract shape."""


class SourceInterpretError(SourcePayloadError):
    """The interpreter failed to produce briefs (execution failure).

    For the science lane this chains the underlying NewsInterpreterError as
    ``__cause__``; the writer's halt path stamps AND re-raises the cause so the
    science failure surface stays byte-identical."""


# ---------------------------------------------------------------------------
# payload contract (EXACT key set -- unknown key = hard error; the article
# surface of the legacy_many_pass source_interpret pass)
# ---------------------------------------------------------------------------

SOURCE_PAYLOAD_KEYS = frozenset({
    "headline", "summary", "full_text", "source", "date", "link", "seed_text",
})


@dataclass(frozen=True)
class SourceFetchResult:
    """Fetcher return with strict payload plus optional provenance sidecars.

    The payload remains the exact seven-key legacy contract. Rights/provenance
    metadata travels beside it so new banks can stamp source information
    without smuggling unknown keys through ``validate_source_payload``.
    """
    payload: dict
    source_meta: dict | None = None
    source_rights: dict | None = None


def validate_source_payload(payload, origin: str) -> dict:
    """Fail-loud shape check; returns a SHALLOW COPY of the payload.

    EXACT key set, every value str, seed_text non-empty after strip. No
    coercion. The copy keeps fetcher-owned dicts immutable post-validation.
    """
    if not isinstance(payload, dict):
        raise SourcePayloadContractError(
            f"{origin}: source payload must be a dict, got "
            f"{type(payload).__name__}"
        )
    keys = set(payload)
    missing = sorted(SOURCE_PAYLOAD_KEYS - keys)
    if missing:
        raise SourcePayloadContractError(
            f"{origin}: source payload missing key(s) {missing}"
        )
    unknown = sorted(keys - SOURCE_PAYLOAD_KEYS)
    if unknown:
        raise SourcePayloadContractError(
            f"{origin}: source payload has unknown key(s) {unknown}; the "
            f"contract is the EXACT set {sorted(SOURCE_PAYLOAD_KEYS)}"
        )
    for key in sorted(SOURCE_PAYLOAD_KEYS):
        val = payload[key]
        if not isinstance(val, str):
            raise SourcePayloadContractError(
                f"{origin}: source payload key {key!r} must be str, got "
                f"{type(val).__name__}"
            )
    if not payload["seed_text"].strip():
        raise SourcePayloadContractError(
            f"{origin}: source payload seed_text is empty -- a fetcher must "
            f"deliver a non-empty seed"
        )
    return dict(payload)


def _copy_sidecar(sidecar, *, origin: str, name: str) -> dict:
    if sidecar is None:
        return {}
    if not isinstance(sidecar, dict):
        raise SourcePayloadContractError(
            f"{origin}: {name} sidecar must be dict or None, got "
            f"{type(sidecar).__name__}"
        )
    return dict(sidecar)


def normalize_fetch_result(result, origin: str) -> tuple[dict, dict, dict]:
    """Validate a fetcher result and return payload/meta/rights copies.

    Legacy fetchers may still return a raw payload dict. Source Banks v2
    fetchers can return ``SourceFetchResult`` to carry attribution and rights
    sidecars without changing the exact payload key set.
    """
    if isinstance(result, SourceFetchResult):
        payload = validate_source_payload(result.payload, origin)
        source_meta = _copy_sidecar(
            result.source_meta, origin=origin, name="source_meta")
        source_rights = _copy_sidecar(
            result.source_rights, origin=origin, name="source_rights")
        return payload, source_meta, source_rights
    if isinstance(result, dict):
        return validate_source_payload(result, origin), {}, {}
    raise SourcePayloadContractError(
        f"{origin}: fetcher result must be a source payload dict or "
        f"SourceFetchResult, got {type(result).__name__}"
    )


# ---------------------------------------------------------------------------
# interpreter result contract (minimal duck-typed pin -- no Protocol class)
# ---------------------------------------------------------------------------

_DUMP_STR_KEYS = ("casting_brief", "script_brief", "news_close_brief")


def validate_interpreter_result(result, origin: str) -> dict:
    """Validate a briefs-like interpreter result; return its validated dump.

    Checks (kibitz r2 codex M3 + r3 codex M1/M2 + r4 codex M1):
      - direct attributes: model_dump()/casting_brief/script_brief/key_terms/
        attempts exist; key_terms is a NON-STRING iterable of non-empty
        strings (a bare str would char-split at tuple()); attempts is int;
      - dump values: casting_brief/script_brief/news_close_brief are str;
        key_terms is list[str] (ledger freeze expects a list);
      - dual-surface coherence: dump casting_brief/script_brief EQUAL the
        direct attrs; dump key_terms == list(direct key_terms).

    Returns the validated dump dict -- the writer assigns THAT object to
    meta["news"] (single validation point; model_dump() is called exactly
    once, here). Violations raise SourcePayloadContractError, which the
    writer's degrade branch never catches -- contract bugs propagate hard.
    """
    for attr in ("model_dump", "casting_brief", "script_brief", "key_terms",
                 "attempts"):
        if not hasattr(result, attr):
            raise SourcePayloadContractError(
                f"{origin}: interpreter result lacks required attribute "
                f"{attr!r}"
            )
    if isinstance(result.key_terms, (str, bytes)):
        raise SourcePayloadContractError(
            f"{origin}: interpreter result key_terms must be a NON-STRING "
            f"iterable of strings (a bare str would split into characters)"
        )
    try:
        direct_key_terms = list(result.key_terms)
    except TypeError as exc:
        raise SourcePayloadContractError(
            f"{origin}: interpreter result key_terms is not iterable"
        ) from exc
    if any(not isinstance(t, str) or not t.strip() for t in direct_key_terms):
        raise SourcePayloadContractError(
            f"{origin}: interpreter result key_terms must contain only "
            f"non-empty strings"
        )
    for attr in ("casting_brief", "script_brief"):
        if not isinstance(getattr(result, attr), str):
            raise SourcePayloadContractError(
                f"{origin}: interpreter result {attr} must be str"
            )
    if not isinstance(result.attempts, int) or isinstance(result.attempts, bool):
        raise SourcePayloadContractError(
            f"{origin}: interpreter result attempts must be int, got "
            f"{type(result.attempts).__name__}"
        )
    dump = result.model_dump()
    if not isinstance(dump, dict):
        raise SourcePayloadContractError(
            f"{origin}: interpreter result model_dump() must return a dict"
        )
    for key in _DUMP_STR_KEYS:
        if not isinstance(dump.get(key), str):
            raise SourcePayloadContractError(
                f"{origin}: interpreter dump key {key!r} must be present and "
                f"str (downstream reads it out of meta['news'])"
            )
    if not isinstance(dump.get("key_terms"), list) or any(
        not isinstance(t, str) for t in dump["key_terms"]
    ):
        raise SourcePayloadContractError(
            f"{origin}: interpreter dump key_terms must be list[str] "
            f"(ledger freeze expects a list)"
        )
    # Dual-surface coherence (r4 codex M1): a duck-typed interpreter must not
    # present different values on the attribute surface vs the dump surface.
    for key in ("casting_brief", "script_brief"):
        if dump[key] != getattr(result, key):
            raise SourcePayloadContractError(
                f"{origin}: interpreter dump {key!r} differs from the direct "
                f"attribute -- split-brain interpreter output"
            )
    if dump["key_terms"] != direct_key_terms:
        raise SourcePayloadContractError(
            f"{origin}: interpreter dump key_terms differs from the direct "
            f"attribute -- split-brain interpreter output"
        )
    return dump


# ---------------------------------------------------------------------------
# science lane wrappers (verbatim call-throughs; lazy heavy imports)
# ---------------------------------------------------------------------------


def _fetch_science_rss(*, bank, technical_model: str,
                       source_ref: str = "",
                       load_config=None, policy=None) -> dict:
    """science_rss: verbatim wrapper around the writer's RSS fetcher.

    Forwards technical_model POSITIONALLY -- the S31 B6 slot-label/id
    agreement invariant (technical model routes the technical re-rank
    slot) survives here, test-pinned. Style-engine consolidation
    (2026-07-05): style_slug removed -- the fetch/rerank chain is
    style-agnostic now, there is no style value yet at this pre-contract
    sourcing stage."""
    del source_ref  # science fetch ignores Source Banks v2 references
    try:
        from . import OTR_LedgerScriptWriter as _writer
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import OTR_LedgerScriptWriter as _writer  # type: ignore
    # Sci-fi v4 validates the fetched body itself before any model call. Keep
    # the legacy science_news forwarding surface byte-compatible, but make all
    # three v4 lanes ask the shared selector for an eligible article instead of
    # receiving the richest thin fallback and failing after model setup.
    # Independence refactor 2026-07-17: each strict-v4 lane by EXACT id (no
    # family base-map). scifi_sonnet survives only as its _v3.
    strict_v4_banks = {"scifi_codex", "scifi_codex_v3", "scifi_sonnet_v3"}
    if getattr(bank, "source_bank_id", "") in strict_v4_banks:
        return _writer._fetch_rss_seed_or_die(
            technical_model, require_science_floor=True,
            load_config=load_config, policy=policy,
        )
    return _writer._fetch_rss_seed_or_die(
        technical_model, load_config=load_config, policy=policy,
    )


def _interpret_news(*, bank, payload: dict, technical_fn,
                    model_id: str):
    """news_interpreter: verbatim wrapper around build_news_briefs.

    Kwarg mapping is the contract's RENAME foot-gun, pinned by test:
    payload["source"] -> outlet, payload["date"] -> pub_date. seed=0 keeps the
    news-interpreter cache key stable (the seed widget was removed,
    BUG-LOCAL-269/270). Translates ONLY NewsInterpreterError ->
    SourceInterpretError (chained); ANY other exception propagates untouched.

    Style-engine consolidation (2026-07-05): this stage runs BEFORE the
    single style engine (build_story_contract needs script_brief, which
    this call produces) -- there is no style value to feed it, and none
    is needed; news interpretation is style-agnostic by design now.
    """
    del bank  # science interpret needs no bank fields; contract signature only
    try:
        from . import news_interpreter as _ni
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import news_interpreter as _ni  # type: ignore
    try:
        return _ni.build_news_briefs(
            technical_fn=technical_fn,
            full_text=payload["full_text"],
            headline=payload["headline"],
            summary=payload["summary"],
            outlet=payload["source"],
            pub_date=payload["date"],
            seed=0,
            model_id=model_id,
        )
    except _ni.NewsInterpreterError as exc:
        raise SourceInterpretError(str(exc)) from exc


def _fetch_media_archive_rss(*, bank, technical_model: str,
                             source_ref: str = "",
                             load_config=None, policy=None) -> dict:
    """media_archive_rss: RSS/Atom media-history feed normalizer."""
    # load_config/policy accepted for uniform fetch dispatch; the media
    # archive lane has no in-fetch LLM rerank chain to thread them into.
    del load_config, policy
    try:
        from . import _otr_media_archive_sources as _mas
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_media_archive_sources as _mas  # type: ignore
    return _mas.fetch_media_archive_rss(
        bank=bank, technical_model=technical_model, source_ref=source_ref)


def _fetch_public_domain_source(*, bank, technical_model: str,
                                source_ref: str = "",
                                load_config=None, policy=None) -> SourceFetchResult:
    """public_domain_source: manifest-local public-domain source fetcher."""
    del technical_model  # source text selection is source_ref/default driven
    del load_config, policy  # accepted for uniform dispatch; no LLM rerank here
    try:
        from . import _otr_public_domain_sources as _pds
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_public_domain_sources as _pds  # type: ignore
    return _pds.fetch_public_domain_source(bank=bank, source_ref=source_ref)


def _fetch_shakespeare_folger(*, bank, technical_model: str,
                              source_ref: str = "",
                              load_config=None, policy=None) -> SourceFetchResult:
    """shakespeare_folger: manifest-local curated Shakespeare scene fetcher."""
    del technical_model  # scene selection is source_ref/default driven
    del load_config, policy  # accepted for uniform dispatch; no LLM rerank here
    try:
        from . import _otr_shakespeare_sources as _shx
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_shakespeare_sources as _shx  # type: ignore
    return _shx.fetch_shakespeare_scene(bank=bank, source_ref=source_ref)


def _interpret_media_archive(*, bank, payload: dict, technical_fn,
                             model_id: str):
    """media_archive_interpreter: archive source brain.

    Translates ONLY MediaArchiveInterpreterError -> SourceInterpretError.
    Contract/shape bugs and unexpected exceptions propagate just like the
    science wrapper's non-NewsInterpreterError path.
    """
    try:
        from . import _otr_media_archive_interpreter as _mai
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_media_archive_interpreter as _mai  # type: ignore
    try:
        return _mai.build_media_archive_briefs(
            technical_fn=technical_fn,
            payload=payload,
            model_id=model_id,
        )
    except _mai.MediaArchiveInterpreterError as exc:
        raise SourceInterpretError(str(exc)) from exc


def _interpret_public_domain(*, bank, payload: dict, technical_fn,
                             model_id: str):
    """public_domain_interpreter: faithful public-domain adaptation brain.

    Translates ONLY PublicDomainInterpreterError -> SourceInterpretError.
    Contract/shape bugs and unexpected exceptions propagate hard.
    """
    del bank
    try:
        from . import _otr_public_domain_sources as _pds
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_public_domain_sources as _pds  # type: ignore
    try:
        return _pds.build_public_domain_briefs(
            technical_fn=technical_fn,
            payload=payload,
            model_id=model_id,
        )
    except _pds.PublicDomainInterpreterError as exc:
        raise SourceInterpretError(str(exc)) from exc


def _interpret_shakespeare(*, bank, payload: dict, technical_fn,
                           model_id: str):
    """shakespeare_interpreter: compact Shakespeare scene adaptation brain.

    Translates ONLY ShakespeareInterpreterError -> SourceInterpretError.
    Contract/shape bugs and unexpected exceptions propagate hard.
    """
    del bank
    try:
        from . import _otr_shakespeare_sources as _shx
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_shakespeare_sources as _shx  # type: ignore
    try:
        return _shx.build_shakespeare_briefs(
            technical_fn=technical_fn,
            payload=payload,
            model_id=model_id,
        )
    except _shx.ShakespeareInterpreterError as exc:
        raise SourceInterpretError(str(exc)) from exc


# ---------------------------------------------------------------------------
# registries + resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetcherEntry:
    """A registered fetcher: the callable + its seed_source stamp label.

    seed_source is REGISTRY metadata (kibitz r1: the payload shape stays
    frozen; "rss_fetch" keeps the science ledger stamps byte-identical)."""
    fetch: object  # callable(*, bank, technical_model, source_ref="") -> result
    seed_source: str


_FETCHERS: "dict[str, FetcherEntry]" = {
    "science_rss": FetcherEntry(fetch=_fetch_science_rss,
                                seed_source="rss_fetch"),
    "media_archive_rss": FetcherEntry(fetch=_fetch_media_archive_rss,
                                      seed_source="media_archive_rss"),
    "public_domain_source": FetcherEntry(fetch=_fetch_public_domain_source,
                                         seed_source="public_domain_source"),
    "shakespeare_folger": FetcherEntry(fetch=_fetch_shakespeare_folger,
                                        seed_source="shakespeare_folger"),
}

_INTERPRETERS: "dict[str, object]" = {
    "news_interpreter": _interpret_news,
    "media_archive_interpreter": _interpret_media_archive,
    "public_domain_interpreter": _interpret_public_domain,
    "shakespeare_interpreter": _interpret_shakespeare,
}


def registered_fetcher_ids() -> "frozenset[str]":
    """Registered fetcher ids (for the routing sweep; ids only, no execution)."""
    return frozenset(_FETCHERS)


def registered_interpreter_ids() -> "frozenset[str]":
    """Registered interpreter ids (for the routing sweep)."""
    return frozenset(_INTERPRETERS)


def resolve_fetcher(bank) -> FetcherEntry:
    """Bank row -> its registered FetcherEntry. Empty id = the bank's
    source-payload lane is not built -- LOUD, never a slide into science."""
    fetcher_id = getattr(bank, "fetcher", "")
    bank_id = getattr(bank, "source_bank_id", "?")
    if not fetcher_id:
        raise SourceContractMissingError(
            f"source_bank {bank_id!r} declares no fetcher: its source-payload "
            f"lane is not built yet (lane-enablement checklist, STAGE2_SUBPLAN "
            f"section 4b item 3). There is no fallback."
        )
    entry = _FETCHERS.get(fetcher_id)
    if entry is None:
        raise UnknownFetcherError(
            f"source_bank {bank_id!r} declares unregistered fetcher "
            f"{fetcher_id!r}; registered: {sorted(_FETCHERS)}"
        )
    return entry


def resolve_interpreter(bank):
    """Bank row -> its registered interpreter callable. Empty id = LOUD."""
    interpreter_id = getattr(bank, "interpreter", "")
    bank_id = getattr(bank, "source_bank_id", "?")
    if not interpreter_id:
        raise SourceContractMissingError(
            f"source_bank {bank_id!r} declares no interpreter: its "
            f"source-payload lane is not built yet (lane-enablement checklist, "
            f"STAGE2_SUBPLAN section 4b item 3). There is no fallback."
        )
    fn = _INTERPRETERS.get(interpreter_id)
    if fn is None:
        raise UnknownInterpreterError(
            f"source_bank {bank_id!r} declares unregistered interpreter "
            f"{interpreter_id!r}; registered: {sorted(_INTERPRETERS)}"
        )
    return fn


__all__ = [
    "FetcherEntry",
    "SOURCE_PAYLOAD_KEYS",
    "SourceFetchResult",
    "SourceContractMissingError",
    "SourceInterpretError",
    "SourcePayloadContractError",
    "SourcePayloadError",
    "UnknownFetcherError",
    "UnknownInterpreterError",
    "normalize_fetch_result",
    "registered_fetcher_ids",
    "registered_interpreter_ids",
    "resolve_fetcher",
    "resolve_interpreter",
    "validate_interpreter_result",
    "validate_source_payload",
]
