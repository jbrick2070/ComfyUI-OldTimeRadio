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
validation/routing/execution (this module carries the callables); NO ROUTING
fallbacks. Unknown/missing id = hard typed error -- a selected bank without a
built lane FAILS THE EPISODE LOUD, never a silent slide into the science path.
The finite structural fallback below is not routing: it preserves the selected
bank, validated payload, source hash, and bank-specific truth/adaptation
contract without inventing semantic key terms.

Import posture: stdlib-only at import time. This module imports NEITHER the
writer NOR news_interpreter NOR _otr_story_routing at module level (three-edge
cycle guard, test-pinned); the heavy imports happen LAZILY inside the wrapper
bodies. Zero file I/O at import. The ``bank`` parameter is DUCK-TYPED
(``.fetcher`` / ``.interpreter`` / ``.source_bank_id`` attributes) -- no
runtime routing import. ``_otr_user_banks`` (client-owned entry points) is
likewise imported function-locally, so the import block alone proves the leaf.

Registry metadata vs execution errors:
  - the routing SWEEP (in _otr_story_routing) validates ids at registry load
    and raises RegistryValidationError there;
  - this module's Unknown*/Missing errors are DEFENSE-IN-DEPTH for direct or
    synthetic callers -- both layers exist deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

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


class ClientOwnerMissingError(SourcePayloadError):
    """A bank row routes an entry point to its own bundle, but resolution got
    no matching owner bundle.

    Always a CALL-SITE defect (the caller forgot `owner=`, or passed another
    bank's bundle), never client data -- an unowned or cross-owned `self` row
    could otherwise execute the wrong bundle's code."""


class SourcePayloadContractError(SourcePayloadError):
    """A payload dict or interpreter result violates the contract shape."""


class SourceInterpretError(SourcePayloadError):
    """The interpreter failed to produce briefs (execution failure).

    For the science lane this chains the underlying NewsInterpreterError as
    ``__cause__``; the writer's halt path stamps AND re-raises the cause so the
    science failure surface stays byte-identical."""


_EXHAUSTION_ATTEMPT_RE = re.compile(
    r"(?:all\s+)?(?P<count>\d+)\s+attempt(?:\(s\)|s)?\s+(?:failed|exhausted)",
    re.IGNORECASE,
)


def interpreter_exhaustion_attempts(exc: BaseException) -> int:
    """Return model calls consumed by a retryable structured interpreter exhaustion.

    The source wrappers deliberately translate only their typed interpreter
    exhaustion errors to :class:`SourceInterpretError`.  A few of those typed
    errors also represent pre-model configuration failures (for example a
    missing media-archive drama-seed deck), while the news wrapper can contain
    a backend/slot exception.  Only actual structured-output/content exhaustion
    may enter the liveness rescue chain; configuration, I/O, backend, and code
    failures keep their existing fail-loud behavior.

    ``0`` therefore means "not a structured/content exhaustion; do not retry or floor".
    """
    if not isinstance(exc, SourceInterpretError):
        return 0
    cause = exc.__cause__
    if cause is None:
        return 0
    reason = str(getattr(cause, "reason", "") or cause)
    if "slot fn raised" in reason.casefold():
        return 0
    attempts = getattr(cause, "attempts", None)
    if isinstance(attempts, int) and not isinstance(attempts, bool):
        return max(0, attempts)
    match = _EXHAUSTION_ATTEMPT_RE.search(reason)
    return int(match.group("count")) if match else 0


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
# deterministic source-interpreter fallback
# ---------------------------------------------------------------------------

SOURCE_INTERPRETER_FALLBACK_VERSION = "source_interpreter_fallback_v2"


def _fallback_one_line(value: object) -> str:
    """Collapse transport whitespace without truncating authored source text."""
    return " ".join(str(value or "").split()).strip()


def _fallback_error_excerpt(value: object, max_chars: int = 600) -> str:
    """Bound diagnostic metadata only; this never enters story authorship."""
    clean = _fallback_one_line(value)
    return clean if len(clean) <= max_chars else clean[:max_chars]



def _source_fallback_character_names(source_meta: dict) -> list[str]:
    raw = source_meta.get("cast_hints") or source_meta.get("character_names")
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        name = _fallback_one_line(value)
        folded = name.casefold()
        if not name or folded in seen:
            continue
        seen.add(folded)
        out.append(name)
    return out


@dataclass
class SourceInterpreterFallback:
    """Duck-typed briefs result for finite structured-output exhaustion.

    This is not a generic story author. ``build_source_interpreter_fallback``
    selects a bank-specific brief template and carries the validated source
    title/summary plus cast hints forward verbatim. It does not classify words
    or fabricate a key-term floor.
    """

    casting_brief: str
    script_brief: str
    news_close_brief: str
    key_terms: list[str]
    attempts: int
    source_hash: str
    character_names: list[str]
    fallback_reason: str
    model_id: str = "deterministic"
    prompt_version: str = SOURCE_INTERPRETER_FALLBACK_VERSION
    schema_version: str = SOURCE_INTERPRETER_FALLBACK_VERSION
    deterministic_fallback: bool = True

    def model_dump(self) -> dict:
        return {
            "casting_brief": self.casting_brief,
            "script_brief": self.script_brief,
            "news_close_brief": self.news_close_brief,
            "key_terms": list(self.key_terms),
            "attempts": self.attempts,
            "source_hash": self.source_hash,
            "character_names": list(self.character_names),
            "fallback_reason": self.fallback_reason,
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "schema_version": self.schema_version,
            "deterministic_fallback": self.deterministic_fallback,
        }


def build_source_interpreter_fallback(
    *, bank, payload: dict, source_meta: dict | None,
    attempts: int, failure_reason: str,
) -> SourceInterpreterFallback:
    """Build a validated, bank-specific minimum brief from source truth.

    The four registered source-interpreter families each get their own
    template.  A CLIENT-owned lane (the reserved ``self`` entry point) gets a
    source-grounded brief built from the bank's own identity, because there is
    no shipped dramaturgy to borrow and guessing one would author a genre the
    client never asked for.  Router mistakes -- an interpreter id that is
    neither registered nor the reserved client value -- and malformed source
    payloads remain hard contract failures.
    """
    clean_payload = validate_source_payload(
        payload, origin="build_source_interpreter_fallback",
    )
    if (not isinstance(attempts, int) or isinstance(attempts, bool)
            or attempts < 1):
        raise SourcePayloadContractError(
            "source interpreter structural fallback attempts must be a positive int"
        )
    if source_meta is None:
        clean_meta: dict = {}
    elif isinstance(source_meta, dict):
        clean_meta = dict(source_meta)
    else:
        raise SourcePayloadContractError(
            "source interpreter structural fallback source_meta must be dict or None"
        )

    interpreter_id = str(getattr(bank, "interpreter", "") or "").strip()
    headline = _fallback_one_line(clean_payload["headline"])
    summary = _fallback_one_line(
        clean_payload["summary"]
        or clean_payload["full_text"]
        or clean_payload["seed_text"]
    )
    source_context = " ".join(
        part for part in (
            f"Source title: {headline}." if headline else "",
            f"Source summary: {summary}." if summary else "",
        )
        if part
    )
    headline_suffix = f": {headline}" if headline else ""
    names = _source_fallback_character_names(clean_meta)
    names_clause = (
        f" Use the source's named characters: {', '.join(names)}."
        if names else " Use only people and roles supported by the source."
    )

    if interpreter_id == "media_archive_interpreter":
        casting = (
            "Cast an archivist, a preservation specialist, and a curious "
            "research partner with distinct, natural radio voices."
        )
        script = (
            "Create a compact SFW fictional archive drama grounded in the "
            "selected media-history item. Make the conflict about identifying, "
            "restoring, researching, or preserving that item, and resolve with "
            f"credible preservation progress. {source_context} "
            "Treat the RSS payload as the only authority for factual claims."
        )
        close = (
            "This fictional archive drama was inspired by the selected media "
            f"source{headline_suffix}; all added dramatic "
            "events are fictional."
        )
    elif interpreter_id == "public_domain_interpreter":
        casting = (
            "Preserve the configured public-domain unit's actual people, roles, "
            "and relationships; give each speaker a distinct radio voice."
            + names_clause
        )
        script = (
            "Create a compact, faithful, SFW radio adaptation of the configured "
            "public-domain unit. Preserve its central dramatic turn and ending; "
            "do not add an unrelated framing story or change the resolution. "
            f"{source_context}"
        )
        close = (
            "This broadcast is a dramatization of the configured public-domain "
            f"source unit{headline_suffix}."
        )
    elif interpreter_id == "shakespeare_interpreter":
        casting = (
            "Use the selected Shakespeare scene's actual characters and "
            "play-world relationships with clear, distinct radio voices."
            + names_clause
        )
        script = (
            "Create a compact, faithful, SFW radio adaptation of the selected "
            "Shakespeare scene. Preserve its characters, dramatic turn, "
            "play-world stakes, and ending without an unrelated modern frame. "
            f"{source_context}"
        )
        close = (
            "This noncommercial broadcast adapts the configured Shakespeare "
            f"scene under its recorded source and rights terms{headline_suffix}."
        )
    elif interpreter_id == "news_interpreter":
        casting = (
            "Cast credible people whose roles are supported by the supplied "
            "article, with distinct and natural radio voices."
        )
        script = (
            "Create a compact SFW dramatization grounded only in the supplied "
            f"article. {source_context} "
            "Clearly separate fictional dramatic events from reported facts."
        )
        close = (
            "The preceding dramatization was inspired by the supplied report; "
            f"the reported source was {headline or 'the selected article'}."
        )
    elif interpreter_id == _user_banks().SELF_ENTRY_POINT:
        # A CLIENT bundle owns this lane.  `_otr_story_routing` unlocks the
        # reserved id only on an `is_client` row and never teaches it to the
        # shipped registry, so reaching this branch proves the bank is
        # client-owned.  There is no shipped template to borrow: the brief is
        # built from the bank's OWN identity plus the validated payload, and
        # asserts nothing about genre, period, or form that the client did not
        # supply.  Without this branch a client interpreter that exhausted its
        # structured-output ladder died on `UnknownInterpreterError` naming
        # 'self' -- a message about OUR router for a failure that was theirs.
        bank_label = (
            str(getattr(bank, "label", "") or "").strip()
            or str(getattr(bank, "source_bank_id", "") or "").strip()
            or "this source bank"
        )
        casting = (
            "Cast speakers whose roles are supported by the supplied source "
            "material, with distinct, natural radio voices." + names_clause
        )
        script = (
            "Create a compact SFW radio drama grounded only in the supplied "
            f"source material from {bank_label}. Treat that source as the "
            "only authority for factual claims, keep the dramatic turn it "
            f"supports, and resolve it there. {source_context}"
        )
        close = (
            f"This broadcast was drawn from {bank_label}{headline_suffix}; "
            "all added dramatic events are fictional."
        )
    else:
        raise UnknownInterpreterError(
            f"source interpreter fallback has no bank-specific builder "
            f"for interpreter {interpreter_id!r}"
        )

    digest = hashlib.sha256()
    for key in sorted(SOURCE_PAYLOAD_KEYS):
        digest.update(clean_payload[key].encode("utf-8", "replace"))
        digest.update(b"\0")
    fallback = SourceInterpreterFallback(
        casting_brief=_fallback_one_line(casting),
        script_brief=_fallback_one_line(script),
        news_close_brief=_fallback_one_line(close),
        key_terms=[],
        attempts=attempts,
        source_hash=digest.hexdigest()[:16],
        character_names=names,
        fallback_reason=_fallback_error_excerpt(failure_reason),
    )
    validate_interpreter_result(
        fallback,
        origin=f"source deterministic fallback ({interpreter_id})",
    )
    return fallback


# ---------------------------------------------------------------------------
# science lane wrappers (verbatim call-throughs; lazy heavy imports)
# ---------------------------------------------------------------------------


def _rss_source_fetch_result(
    payload: dict,
    *,
    fetcher_kind: str,
    news_seed_receipt: Mapping[str, Any] | None = None,
) -> SourceFetchResult:
    """Wrap a selected RSS item without changing the seven-key payload.

    RSS fetchers historically returned the raw payload dict, so their selected
    article URL and outlet disappeared at ``normalize_fetch_result`` while the
    manifest-backed banks carried typed provenance sidecars.  The selected item
    -- not the blank/ignored widget request -- owns RSS provenance.  Rights stay
    explicitly unknown; selecting an item is not evidence of a license.
    """
    clean_payload = validate_source_payload(
        payload, origin=f"{fetcher_kind} selected RSS item")
    link = str(clean_payload.get("link") or "").strip()
    label = str(clean_payload.get("source") or "").strip()
    date = str(clean_payload.get("date") or "").strip()
    source_meta = {
        "kind": str(fetcher_kind),
        "source_ref": link,
        "source_url": link,
        "source_label": label,
        "source_date": date,
    }
    if news_seed_receipt:
        source_meta["_news_seed_receipt"] = dict(news_seed_receipt)
    return SourceFetchResult(
        payload=clean_payload,
        source_meta=source_meta,
        source_rights={
            "license_status": "unknown",
            "source_url": link,
            "source_label": label,
        },
    )


def _fetch_science_rss(*, bank, technical_model: str,
                       source_ref: str = "",
                       load_config=None, policy=None) -> SourceFetchResult:
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
    receipt: dict[str, Any] = {}
    payload = _writer._fetch_rss_seed_or_die(
        technical_model, load_config=load_config, policy=policy,
        receipt_sink=receipt,
    )
    return _rss_source_fetch_result(
        payload,
        fetcher_kind="science_rss",
        news_seed_receipt=receipt,
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
                             load_config=None, policy=None) -> SourceFetchResult:
    """media_archive_rss: RSS/Atom media-history feed normalizer."""
    # load_config/policy accepted for uniform fetch dispatch; the media
    # archive lane has no in-fetch LLM rerank chain to thread them into.
    del load_config, policy
    try:
        from . import _otr_media_archive_sources as _mas
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_media_archive_sources as _mas  # type: ignore
    payload = _mas.fetch_media_archive_rss(
        bank=bank, technical_model=technical_model, source_ref=source_ref)
    return _rss_source_fetch_result(
        payload, fetcher_kind="media_archive_rss")


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
    """Registered fetcher ids (for the routing sweep; ids only, no execution).

    SHIPPED ids only, forever. Client-owned lanes resolve through their own
    bundle (see `resolve_fetcher`) and never enter this registry -- a client
    bank cannot shadow, replace, or extend a shipped entry point."""
    return frozenset(_FETCHERS)


def registered_interpreter_ids() -> "frozenset[str]":
    """Registered interpreter ids (for the routing sweep). Shipped only."""
    return frozenset(_INTERPRETERS)


# Client-owned lanes get a namespaced seed_source stamp so a ledger never
# confuses "a client bundle fetched this" with any shipped fetcher's provenance.
USER_BANK_SEED_SOURCE_PREFIX = "user_bank:"


def _user_banks():
    """The client-bundle module, imported FUNCTION-LOCALLY.

    `_otr_user_banks` is another stdlib-only, zero-I/O leaf, so this could be a
    module-level import -- it stays local so this module's documented
    "stdlib-only at import time" posture remains provable by inspecting the
    import block, and the leaf can never grow an edge into the authority."""
    try:
        from . import _otr_user_banks as _oub
    except ImportError:  # pragma: no cover -- flat-import test harnesses
        import _otr_user_banks as _oub  # type: ignore
    return _oub


def client_entry_point_id() -> str:
    """The reserved entry-point value meaning "my own bundle owns this lane"."""
    return _user_banks().SELF_ENTRY_POINT


def _require_owner(owner, *, bank_id: str, field: str):
    """The admitted bundle that may execute for `bank_id`, or LOUD.

    Identity is checked, not merely presence: resolving bank A's `self` lane
    against bank B's bundle would silently run the wrong client's code."""
    _oub = _user_banks()
    self_id = _oub.SELF_ENTRY_POINT
    if owner is None:
        raise ClientOwnerMissingError(
            f"source_bank {bank_id!r} routes its {field} to its own bundle "
            f"({self_id!r}) but no owner bundle reached resolution; pass "
            f"owner=_otr_story_routing.user_bank_bundle({bank_id!r})"
        )
    if not isinstance(owner, _oub.UserBankBundle):
        raise ClientOwnerMissingError(
            f"source_bank {bank_id!r}: owner must be an admitted "
            f"UserBankBundle, got {type(owner).__name__}"
        )
    if owner.bank_id != bank_id:
        raise ClientOwnerMissingError(
            f"owner bundle {owner.bank_id!r} does not own source_bank "
            f"{bank_id!r}; a bank may only execute its OWN bundle"
        )
    return owner


def resolve_fetcher(bank, *, owner=None) -> FetcherEntry:
    """Bank row -> its FetcherEntry. Empty id = the bank's source-payload lane
    is not built -- LOUD, never a slide into science.

    `owner` is the admitted client bundle behind this bank
    (`_otr_story_routing.user_bank_bundle(bank_id)`), or None for a shipped
    bank. A row whose fetcher is the reserved client id resolves to its OWN
    bundle's `fetch_source`, which obeys the identical keyword contract as a
    shipped fetcher and whose result still passes `normalize_fetch_result`.
    A client bank may equally declare a SHIPPED fetcher id and get the shipped
    wrapper -- reuse is allowed, replacement is not."""
    fetcher_id = getattr(bank, "fetcher", "")
    bank_id = getattr(bank, "source_bank_id", "?")
    if not fetcher_id:
        raise SourceContractMissingError(
            f"source_bank {bank_id!r} declares no fetcher: its source-payload "
            f"lane is not built yet (lane-enablement checklist, STAGE2_SUBPLAN "
            f"section 4b item 3). There is no fallback."
        )
    _oub = _user_banks()
    if fetcher_id == _oub.SELF_ENTRY_POINT:
        bundle = _require_owner(owner, bank_id=bank_id, field="fetcher")
        return FetcherEntry(
            fetch=_oub.bundle_entry_point(bundle, _oub.FETCH_ENTRY_ATTR),
            seed_source=f"{USER_BANK_SEED_SOURCE_PREFIX}{bank_id}",
        )
    entry = _FETCHERS.get(fetcher_id)
    if entry is None:
        raise UnknownFetcherError(
            f"source_bank {bank_id!r} declares unregistered fetcher "
            f"{fetcher_id!r}; registered: {sorted(_FETCHERS)}"
        )
    return entry


def resolve_interpreter(bank, *, owner=None):
    """Bank row -> its interpreter callable. Empty id = LOUD.

    Same owner contract as `resolve_fetcher`: the reserved client id resolves
    to the bundle's `interpret_source`, called with the identical keyword
    contract and validated by the identical `validate_interpreter_result`."""
    interpreter_id = getattr(bank, "interpreter", "")
    bank_id = getattr(bank, "source_bank_id", "?")
    if not interpreter_id:
        raise SourceContractMissingError(
            f"source_bank {bank_id!r} declares no interpreter: its "
            f"source-payload lane is not built yet (lane-enablement checklist, "
            f"STAGE2_SUBPLAN section 4b item 3). There is no fallback."
        )
    _oub = _user_banks()
    if interpreter_id == _oub.SELF_ENTRY_POINT:
        bundle = _require_owner(owner, bank_id=bank_id, field="interpreter")
        return _oub.bundle_entry_point(bundle, _oub.INTERPRET_ENTRY_ATTR)
    fn = _INTERPRETERS.get(interpreter_id)
    if fn is None:
        raise UnknownInterpreterError(
            f"source_bank {bank_id!r} declares unregistered interpreter "
            f"{interpreter_id!r}; registered: {sorted(_INTERPRETERS)}"
        )
    return fn


__all__ = [
    "ClientOwnerMissingError",
    "FetcherEntry",
    "SOURCE_PAYLOAD_KEYS",
    "SOURCE_INTERPRETER_FALLBACK_VERSION",
    "USER_BANK_SEED_SOURCE_PREFIX",
    "SourceFetchResult",
    "SourceContractMissingError",
    "SourceInterpretError",
    "SourceInterpreterFallback",
    "SourcePayloadContractError",
    "SourcePayloadError",
    "UnknownFetcherError",
    "UnknownInterpreterError",
    "build_source_interpreter_fallback",
    "client_entry_point_id",
    "normalize_fetch_result",
    "interpreter_exhaustion_attempts",
    "registered_fetcher_ids",
    "registered_interpreter_ids",
    "resolve_fetcher",
    "resolve_interpreter",
    "validate_interpreter_result",
    "validate_source_payload",
]
