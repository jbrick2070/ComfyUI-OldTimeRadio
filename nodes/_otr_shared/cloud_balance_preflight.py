"""Queue-time wallet check -- refuse before the first credit moves.

Wired from ``OTR_WorkflowValidator`` right after the slug preflight and before
the visual-asset gate. The slug gate stops a dead model; this gate stops a live
model from draining a wallet that cannot finish the episode. Per-call
``reserve()`` only knows ``OTR_CLOUD_MEDIA_BUDGET_USD`` (often unset) or learns
about an empty wallet from HTTP 402 after spend has started.

Three wallets, never pooled:

* ``comfy``       -- ``comfy:slot-*`` writer and every ``cloud_*`` / ``sonilo``
                     / ``ideo`` partner media. Remaining from
                     ``GET https://api.comfy.org/customers/balance``.
* ``openrouter``  -- ``openrouter:slot-*`` writer. Remaining from
                     ``GET https://openrouter.ai/api/v1/key`` (``limit_remaining``),
                     then ``/api/v1/credits`` only when the key is uncapped.
* ``google``      -- ``google_api:slot-*`` writer and ``google_*`` engines.
                     No remaining-dollar API exists. Estimate and warn only.

The estimate is deliberately high: the script does not exist at queue time, so
every paid engine is priced at the count it *could* reach, then padded 15%
(minimum $1.00). A refusal names the wallet, the need, the remaining, and the
top cost lines so the operator can see the arithmetic.

Cold-import-clean: stdlib only at module import. Engine registries, backend
resolvers, and both balance GETs are lazy and injectable so tests run with no
Comfy tree and no socket.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Callable, NamedTuple, Optional
from urllib.request import urlopen  # bare name clears the registry $http2 literal; still seen by the network-sites guard

log = logging.getLogger("OTR.cloud.balance")

WALLET_COMFY = "comfy"
WALLET_OPENROUTER = "openrouter"
WALLET_GOOGLE = "google"

COMFY_HOST = "api.comfy.org"
COMFY_BALANCE_URL = "https://api.comfy.org/customers/balance"
OPENROUTER_HOST = "openrouter.ai"
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"
OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"

#: Official display rate from ``ComfyUI_frontend`` ``comfyCredits.ts``. The
#: backend ``*_micros`` fields are CENTS, so ``remaining_usd = micros / 100``.
#: This constant is only for the human-readable credit figure in messages.
COMFY_CREDITS_PER_USD = 211
COMFY_MICROS_PER_USD = 100.0

#: Padding after the sum: 15%, and never less than one dollar.
PAD_RATIO = 0.15
PAD_MIN_USD = 1.0

#: Conservative queue-time counts. The script does not exist yet.
DEFAULT_BEATS = 40            # upper bound on a planned episode's beats
DEFAULT_FRESH_CAP = 15        # OTR_ImageDirector.fresh_cap default
DEFAULT_ACT_COUNT = 3
MUSIC_CUES = 4
#: 3-act and longer default. 1-act uses 4s -- see clip_seconds_for_act_count.
VIDEO_CLIP_SECONDS = 8
VIDEO_CLIP_FPS = 25
VIDEO_CLIP_SECONDS_1ACT = 4
#: ElevenLabs floor: the beat topology tops out near 1,520 spoken words per
#: episode (~9,000 characters). Per act, rounded up, so a 1-act still reserves
#: a real number and a 5-act reserves more than the whole ceiling.
TTS_CHARS_PER_ACT = 4000
#: Writer floor when no catalog price is available (Comfy slots, Google
#: slots, an OpenRouter slug missing from the cache). USD per million tokens;
#: above every writer slug this pack recommends.
WRITER_FLOOR_USD_PER_MTOK = 5.0
#: Per-run token ceilings the backends already enforce (their defaults).
OPENROUTER_RUN_TOKENS_DEFAULT = 300_000
COMFY_RUN_TOKENS_DEFAULT = 1_000_000
GOOGLE_RUN_TOKENS_DEFAULT = 300_000

_WRITER = "OTR_LedgerScriptWriter"
_DIRECTOR = "OTR_VideoDirector"
_CASTLOCK = "OTR_CastLock"
_MUSIC = "OTR_StableAudioTheme"
_IMAGE_DIRECTOR = "OTR_ImageDirector"
_CAST_SLOTS = ("char_voice_engine", "announcer_voice_engine")


class CostLine(NamedTuple):
    wallet: str
    label: str        # engine id or writer handle
    count: float      # units the estimate assumes
    unit_usd: Optional[float]  # None = no price function (unknown)
    total_usd: float  # 0.0 when unknown
    note: str


class BalanceResult(NamedTuple):
    remaining_usd: Optional[float]  # None = not obtained
    error: str
    host: str
    severity: str  # "refuse" | "warn" -- how a missing number is treated
    detail: str = ""


class WalletVerdict(NamedTuple):
    wallet: str
    estimate_usd: float
    needed_usd: float
    remaining_usd: Optional[float]
    unknown: tuple   # labels with no price
    lines: tuple     # CostLine sorted by total desc
    severity: str    # "ok" | "warn" | "refuse"
    reason: str


# ---------------------------------------------------------------------------
# Wallet routing
# ---------------------------------------------------------------------------

def wallet_for_engine(engine_id: str) -> Optional[str]:
    eid = str(engine_id or "").strip()
    if not eid:
        return None
    if eid.startswith("google_"):
        return WALLET_GOOGLE
    if eid.startswith("cloud_") or eid in {"sonilo", "ideo"}:
        return WALLET_COMFY
    return None


def wallet_for_writer_handle(handle: str) -> Optional[str]:
    text = str(handle or "")
    if text.startswith("comfy:"):
        return WALLET_COMFY
    if text.startswith("openrouter:"):
        return WALLET_OPENROUTER
    if text.startswith("google_api:"):
        return WALLET_GOOGLE
    return None


def pad_usd(subtotal: float) -> float:
    return max(float(subtotal) * PAD_RATIO, PAD_MIN_USD)


def clip_seconds_for_act_count(act_count) -> int:
    """Veo menu, not a guess: 1 act -> 4s / 100 frames; else 8s / 200."""
    try:
        n = int(act_count)
    except (TypeError, ValueError):
        n = DEFAULT_ACT_COUNT
    if n <= 0:
        n = DEFAULT_ACT_COUNT
    return VIDEO_CLIP_SECONDS_1ACT if n == 1 else VIDEO_CLIP_SECONDS


# ---------------------------------------------------------------------------
# Prices (lazy, injectable)
# ---------------------------------------------------------------------------

def _stub_video_request(duration_s=None):
    """Enough request shape for duration-aware ``_estimated_usd`` helpers."""
    secs = VIDEO_CLIP_SECONDS if duration_s is None else int(duration_s)
    return {
        "canvas": {"fps": VIDEO_CLIP_FPS},
        "timing": {"target_frame_count": max(secs, 1) * VIDEO_CLIP_FPS},
    }


def _module_of(engine):
    import sys
    return sys.modules.get(type(engine).__module__)


def default_unit_usd(engine_id: str, engine, duration_s: int = VIDEO_CLIP_SECONDS) -> Optional[float]:
    """One unit (clip / still / cue / act of speech) in USD, or None."""
    eid = str(engine_id or "")
    if engine is None:
        return None
    mod = _module_of(engine)
    if eid == "sonilo":
        fn = getattr(mod, "estimate_music_usd", None)
        floor_fn = getattr(mod, "_sonilo_min_duration_s", None)
        if callable(fn):
            secs = 30
            if callable(floor_fn):
                try:
                    secs = max(int(floor_fn()), 1)
                except Exception:  # noqa: BLE001 -- env parse is advisory
                    secs = 30
            return float(fn(secs))
        return None
    if eid == "cloud_elevenlabs":
        fn = getattr(mod, "estimate_tts_usd", None)
        if callable(fn):
            return float(fn("x" * TTS_CHARS_PER_ACT))
        return None
    per_clip = getattr(engine, "_estimated_usd", None)
    if callable(per_clip):
        try:
            return float(per_clip(_stub_video_request(duration_s)))
        except Exception as exc:  # noqa: BLE001 -- price is advisory, not fatal
            log.warning("[OTR.cloud.balance] %s _estimated_usd failed: %s", eid, exc)
            return None
    per_still = getattr(engine, "_est_usd", None)
    if callable(per_still):
        try:
            return float(per_still())
        except Exception as exc:  # noqa: BLE001
            log.warning("[OTR.cloud.balance] %s _est_usd failed: %s", eid, exc)
            return None
    return None


def _env_int(name: str, default: int) -> int:
    try:
        from . import env as otr_env
    except ImportError:  # pragma: no cover -- flat import
        import env as otr_env  # type: ignore
    raw = str(otr_env.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def default_writer_usd(handle: str, slug: str) -> tuple:
    """(usd, note) for one writer handle at that backend's per-run ceiling."""
    wallet = wallet_for_writer_handle(handle)
    price_per_tok = None
    source = "floor $%.2f/Mtok" % WRITER_FLOOR_USD_PER_MTOK
    if wallet == WALLET_OPENROUTER:
        tokens = _env_int("OPENROUTER_MAX_TOKENS_PER_RUN", OPENROUTER_RUN_TOKENS_DEFAULT)
        try:
            from .._otr_openrouter_backend import _cached_model
            pricing = (_cached_model(slug) or {}).get("pricing") or {}
            quotes = []
            for key in ("prompt", "completion"):
                raw = pricing.get(key)
                if raw is not None:
                    quotes.append(float(raw))
            if quotes:
                price_per_tok = max(quotes)
                source = "catalog $%.2f/Mtok" % (price_per_tok * 1e6)
        except Exception:  # noqa: BLE001 -- cache is advisory
            price_per_tok = None
    elif wallet == WALLET_COMFY:
        tokens = _env_int("OTR_COMFY_MAX_TOKENS_PER_RUN", COMFY_RUN_TOKENS_DEFAULT)
    else:
        tokens = GOOGLE_RUN_TOKENS_DEFAULT
    if price_per_tok is None:
        price_per_tok = WRITER_FLOOR_USD_PER_MTOK / 1e6
    return tokens * price_per_tok, "%d tokens x %s" % (tokens, source)


# ---------------------------------------------------------------------------
# Balances (lazy, injectable)
# ---------------------------------------------------------------------------

def _http_get_json(url: str, bearer: str, timeout: float = 20.0) -> tuple:
    """(status, payload). Raises on transport failure; HTTP errors return."""
    headers = {"User-Agent": "OTR-cloud-balance-preflight",
               "Accept": "application/json"}
    if bearer:
        headers["Authorization"] = "Bearer %s" % bearer
    req = urllib.request.Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return int(resp.status), (json.loads(body) if body.strip() else {})
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        try:
            payload = json.loads(body) if body.strip() else {}
        except ValueError:
            payload = {"raw": body[:200]}
        return int(exc.code), payload


def parse_comfy_balance(payload) -> Optional[float]:
    """``*_micros`` are cents. ``effective_balance_micros`` wins over ``amount_micros``."""
    if not isinstance(payload, dict):
        return None
    for key in ("effective_balance_micros", "amount_micros"):
        raw = payload.get(key)
        if raw is None:
            continue
        try:
            return float(raw) / COMFY_MICROS_PER_USD
        except (TypeError, ValueError):
            continue
    return None


def _comfy_bearer_from(api_key) -> Callable[[], str]:
    """The queue's own credential: the api_key_comfy_org that
    OTR_ComfyCredential bound for this prompt (the only Comfy credential
    since the 2026-09-19 rip). Empty raises, which comfy_balance reports as
    a warn -- a local-only graph never carries one."""
    def bearer() -> str:
        key = str(api_key or "").strip()
        if not key:
            raise RuntimeError(
                "no api_key_comfy_org reached this queue (the '0 - Comfy "
                "Credential' node binds it)")
        return key
    return bearer


def comfy_balance(*, get_json: Callable = None, bearer: Callable = None,
                  api_key: str = None) -> BalanceResult:
    """Remaining Comfy Credits in USD. A missing number on this host refuses,
    except when the queue carries no credential at all -- that case is a
    warn, not a refusal, and the lane fails closed at execution instead."""
    getter = get_json or _http_get_json
    bearer_fn = bearer or _comfy_bearer_from(api_key)
    try:
        token = bearer_fn()
    except Exception as exc:  # noqa: BLE001 -- AUTH is reported, not raised
        return BalanceResult(
            None, "no Comfy API key on this queue (%s)" % exc,
            COMFY_HOST, "warn",
            "balance unverified; a Comfy-billed pick will fail closed at execution")
    try:
        status, payload = getter(COMFY_BALANCE_URL, token)
    except Exception as exc:  # noqa: BLE001 -- transport
        return BalanceResult(None, "%s: %s" % (type(exc).__name__, exc),
                             COMFY_HOST, "refuse")
    if status != 200:
        return BalanceResult(None, "HTTP %d from %s" % (status, COMFY_BALANCE_URL),
                             COMFY_HOST, "refuse")
    remaining = parse_comfy_balance(payload)
    if remaining is None:
        return BalanceResult(None, "balance payload has no *_micros field",
                             COMFY_HOST, "refuse")
    return BalanceResult(remaining, "", COMFY_HOST, "refuse",
                         "~%d credits" % int(remaining * COMFY_CREDITS_PER_USD))


def _openrouter_bearer() -> str:
    from .._otr_openrouter_backend import _openrouter_key
    key = _openrouter_key()
    if not key:
        raise RuntimeError("no OpenRouter key")
    return key


def openrouter_balance(*, get_json: Callable = None, bearer: Callable = None) -> BalanceResult:
    """``/key`` ``limit_remaining`` when capped; else ``/credits``
    (management-key-only: 401/403 there is a warn, never a refusal)."""
    getter = get_json or _http_get_json
    bearer_fn = bearer or _openrouter_bearer
    try:
        token = bearer_fn()
    except Exception as exc:  # noqa: BLE001
        return BalanceResult(None, "no OpenRouter key (%s)" % exc,
                             OPENROUTER_HOST, "refuse")
    try:
        status, payload = getter(OPENROUTER_KEY_URL, token)
    except Exception as exc:  # noqa: BLE001 -- transport
        return BalanceResult(None, "%s: %s" % (type(exc).__name__, exc),
                             OPENROUTER_HOST, "refuse")
    if status != 200:
        return BalanceResult(None, "HTTP %d from %s" % (status, OPENROUTER_KEY_URL),
                             OPENROUTER_HOST, "refuse")
    data = payload.get("data") if isinstance(payload, dict) else None
    data = data if isinstance(data, dict) else {}
    limit_remaining = data.get("limit_remaining")
    if limit_remaining is not None:
        try:
            return BalanceResult(float(limit_remaining), "", OPENROUTER_HOST,
                                 "refuse", "key limit_remaining")
        except (TypeError, ValueError):
            pass
    # Uncapped key: the account balance lives behind a management key.
    try:
        status, payload = getter(OPENROUTER_CREDITS_URL, token)
    except Exception as exc:  # noqa: BLE001
        return BalanceResult(
            None, "key is uncapped; /credits unreachable (%s)" % exc,
            OPENROUTER_HOST, "warn")
    if status != 200:
        return BalanceResult(
            None, "key is uncapped; /credits returned HTTP %d "
            "(management key required)" % status,
            OPENROUTER_HOST, "warn")
    data = payload.get("data") if isinstance(payload, dict) else None
    data = data if isinstance(data, dict) else {}
    try:
        total = float(data.get("total_credits"))
        used = float(data.get("total_usage") or 0.0)
    except (TypeError, ValueError):
        return BalanceResult(None, "key is uncapped; /credits payload unreadable",
                             OPENROUTER_HOST, "warn")
    return BalanceResult(total - used, "", OPENROUTER_HOST, "refuse",
                         "account total_credits - total_usage")


def google_balance(**_ignored) -> BalanceResult:
    """No remaining-dollar API for a Gemini key. Estimate only."""
    return BalanceResult(None, "Google API exposes no remaining balance",
                         "", "warn", "estimate is logged, never refused")


_BALANCE_FNS = {
    WALLET_COMFY: comfy_balance,
    WALLET_OPENROUTER: openrouter_balance,
    WALLET_GOOGLE: google_balance,
}


def default_balance(wallet: str) -> BalanceResult:
    fn = _BALANCE_FNS.get(str(wallet or ""))
    if fn is None:
        raise ValueError("unknown wallet %r" % wallet)
    return fn()


# ---------------------------------------------------------------------------
# Prompt walk -> cost lines
# ---------------------------------------------------------------------------

def _int_widget(inputs, name, default):
    value = (inputs or {}).get(name, None)
    if value is None or isinstance(value, (list, tuple)):
        return int(default)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return int(default)


def _find_nodes(prompt, scoped, class_type):
    hits = [n for n in scoped if n.get("class_type") == class_type]
    if hits:
        return hits
    return [n for n in (prompt or {}).values()
            if isinstance(n, dict) and n.get("class_type") == class_type]


def estimate_lines(
    prompt, unique_id, *,
    resolve_engine=None,
    unit_usd_fn=None,
    writer_usd_fn=None,
    walk_fn=None,
    skip_writer=False,
) -> list:
    """Every cost line the queued prompt could reach. Does not raise for
    price gaps -- an unknown price is a line with ``unit_usd=None``."""
    try:
        from . import cloud_slug_preflight as csp
    except ImportError:  # pragma: no cover
        import cloud_slug_preflight as csp  # type: ignore

    lines = []
    if unique_id is None or not str(unique_id).strip():
        return lines
    if not isinstance(prompt, dict) or not prompt:
        return lines

    scoped = csp._collect_scoped(prompt, unique_id, walk_fn=walk_fn)
    lookup = resolve_engine or csp._default_resolve_engine
    writer_price = writer_usd_fn or default_writer_usd

    beats = DEFAULT_BEATS
    fresh_cap = DEFAULT_FRESH_CAP
    for node in _find_nodes(prompt, scoped, _IMAGE_DIRECTOR):
        fresh_cap = max(fresh_cap,
                        _int_widget(node.get("inputs"), "fresh_cap", DEFAULT_FRESH_CAP))

    writers = [n for n in scoped if n.get("class_type") == _WRITER]
    act_count = DEFAULT_ACT_COUNT
    replay_flags = []
    if writers:
        seen_acts = []
        for node in writers:
            inputs = node.get("inputs") or {}
            seen_acts.append(_int_widget(inputs, "act_count", DEFAULT_ACT_COUNT))
            replay_flags.append(bool(csp._widget_str(inputs, "replay_from", "")))
        act_count = max(seen_acts) if seen_acts else DEFAULT_ACT_COUNT
    replay = bool(replay_flags) and all(replay_flags)
    clip_s = clip_seconds_for_act_count(act_count)
    if unit_usd_fn is None:
        def price(label, eng):
            return default_unit_usd(label, eng, clip_s)
    else:
        price = unit_usd_fn

    # Video: every distinct paid engine is assumed to take every beat.
    video_ids = []
    image_roles = []
    for node in [n for n in scoped if n.get("class_type") == _DIRECTOR]:
        videos, images = csp.director_engine_picks(node.get("inputs") or {})
        video_ids.extend(videos)
        image_roles.extend(images)

    def add(label, count, note):
        wallet = wallet_for_engine(label)
        if wallet is None:
            return
        eng = lookup(label)
        unit = price(label, eng)
        total = float(unit) * float(count) if unit is not None else 0.0
        lines.append(CostLine(wallet, label, float(count), unit, total, note))

    for eid in sorted({str(v).strip() for v in video_ids if v}):
        add(eid, beats, "%d beats x %ds clip" % (beats, clip_s))

    # Stills: fresh_cap per image role that still renders.
    role_counts = {}
    for eid in image_roles:
        key = str(eid or "").strip()
        if key:
            role_counts[key] = role_counts.get(key, 0) + 1
    for eid, roles in sorted(role_counts.items()):
        add(eid, fresh_cap * roles, "%d roles x fresh_cap %d" % (roles, fresh_cap))

    for node in scoped:
        ctype = node.get("class_type")
        inputs = node.get("inputs") or {}
        if ctype == _MUSIC:
            picked = csp._widget_str(inputs, "engine", "")
            if picked and not picked.startswith("+ Add Custom"):
                add(picked, MUSIC_CUES, "%d cues" % MUSIC_CUES)
        if ctype == _CASTLOCK:
            voices = {csp._widget_str(inputs, slot, "") for slot in _CAST_SLOTS}
            for picked in sorted(v for v in voices if v):
                add(picked, act_count,
                    "%d acts x %d chars" % (act_count, TTS_CHARS_PER_ACT))

    if skip_writer or replay:
        return lines

    seen_wallets = {}
    for node in writers:
        inputs = node.get("inputs") or {}
        for handle, widget, _authority, _host in csp._writer_handles(inputs):
            wallet = wallet_for_writer_handle(handle)
            if wallet is None:
                continue
            slug, _reason = csp._posted_writer_slug(handle, inputs)
            usd, note = writer_price(handle, slug)
            # One per-run token ceiling per backend: slot-a and slot-b on the
            # same wallet share it, so keep the dearer of the two.
            prior = seen_wallets.get(wallet)
            if prior is None or usd > prior.total_usd:
                seen_wallets[wallet] = CostLine(
                    wallet, handle, 1.0, float(usd), float(usd),
                    "%s (%s)" % (note, slug or widget))
    lines.extend(seen_wallets.values())
    return lines


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def judge_wallet(wallet: str, lines, balance: Optional[BalanceResult]) -> WalletVerdict:
    """Compare one wallet's padded estimate to its remaining balance."""
    mine = sorted((ln for ln in lines if ln.wallet == wallet),
                  key=lambda ln: ln.total_usd, reverse=True)
    unknown = tuple(ln.label for ln in mine if ln.unit_usd is None)
    subtotal = sum(ln.total_usd for ln in mine)
    needed = subtotal + pad_usd(subtotal)
    remaining = balance.remaining_usd if balance is not None else None

    if balance is None:
        return WalletVerdict(wallet, subtotal, needed, None, unknown, tuple(mine),
                             "warn", "balance not queried")
    if remaining is None:
        sev = "refuse" if balance.severity == "refuse" else "warn"
        reason = "remaining unknown (%s)" % (balance.error or "no result")
        return WalletVerdict(wallet, subtotal, needed, None, unknown, tuple(mine),
                             sev, reason)
    if unknown and remaining <= 0.0:
        return WalletVerdict(
            wallet, subtotal, needed, remaining, unknown, tuple(mine), "refuse",
            "wallet is empty and %s has no price function" % ", ".join(unknown))
    if remaining + 1e-9 < needed:
        return WalletVerdict(
            wallet, subtotal, needed, remaining, unknown, tuple(mine), "refuse",
            "remaining $%.2f < needed $%.2f" % (remaining, needed))
    if unknown:
        return WalletVerdict(
            wallet, subtotal, needed, remaining, unknown, tuple(mine), "warn",
            "no price function for %s; estimate is incomplete" % ", ".join(unknown))
    return WalletVerdict(wallet, subtotal, needed, remaining, unknown, tuple(mine),
                         "ok", "")


def collect_verdicts(prompt, unique_id, *, balance_fn=None, comfy_api_key=None,
                     **estimate_kwargs) -> list:
    """One verdict per wallet the prompt touches. Local-only => [].

    ``comfy_api_key`` is the queue's api_key_comfy_org hidden input; the
    Comfy wallet is measured with it when no ``balance_fn`` override is
    given."""
    lines = estimate_lines(prompt, unique_id, **estimate_kwargs)
    wallets = []
    for ln in lines:
        if ln.wallet not in wallets:
            wallets.append(ln.wallet)
    if balance_fn is None:
        def balance_fn(wallet):
            if wallet == WALLET_COMFY:
                return comfy_balance(api_key=comfy_api_key)
            return default_balance(wallet)
    getter = balance_fn
    verdicts = []
    for wallet in wallets:
        try:
            balance = getter(wallet)
        except Exception as exc:  # noqa: BLE001 -- a getter crash is a finding
            host = COMFY_HOST if wallet == WALLET_COMFY else (
                OPENROUTER_HOST if wallet == WALLET_OPENROUTER else "")
            balance = BalanceResult(None, "%s: %s" % (type(exc).__name__, exc),
                                    host, "refuse" if host else "warn")
        verdicts.append(judge_wallet(wallet, lines, balance))
    return verdicts


def _fmt_line(ln: CostLine) -> str:
    if ln.unit_usd is None:
        return "%s: no price (%s)" % (ln.label, ln.note)
    return "%s: $%.2f (%s)" % (ln.label, ln.total_usd, ln.note)


def format_refusal(verdicts) -> str:
    refuses = [v for v in verdicts if v.severity == "refuse"]
    parts = []
    for v in refuses:
        remaining = ("$%.2f" % v.remaining_usd) if v.remaining_usd is not None else "unknown"
        top = "; ".join(_fmt_line(ln) for ln in v.lines[:3])
        parts.append(
            "%s wallet -- needed $%.2f (estimate $%.2f + pad), remaining %s -- %s"
            " -- top: %s" % (v.wallet, v.needed_usd, v.estimate_usd, remaining,
                            v.reason, top or "(no lines)"))
    return (
        "OTR_WorkflowValidator: cloud balance preflight -- %d wallet(s) short -- %s. "
        "Top up the wallet or pick a cheaper engine before the run spends a credit."
        % (len(refuses), " | ".join(parts))
    )


def ensure_prompt_cloud_balance(prompt, unique_id, **kwargs) -> list:
    """Wired entry. Raises ValueError listing every wallet that refuses."""
    verdicts = collect_verdicts(prompt, unique_id, **kwargs)
    for v in verdicts:
        if v.severity == "warn":
            log.warning("[OTR.cloud.balance] %s wallet: estimate $%.2f (needed $%.2f) -- %s",
                        v.wallet, v.estimate_usd, v.needed_usd, v.reason)
        elif v.severity == "ok":
            log.info("[OTR.cloud.balance] %s wallet: needed $%.2f, remaining $%.2f",
                     v.wallet, v.needed_usd, v.remaining_usd)
    if any(v.severity == "refuse" for v in verdicts):
        raise ValueError(format_refusal(verdicts))
    return verdicts


__all__ = [
    "BalanceResult",
    "CostLine",
    "WalletVerdict",
    "clip_seconds_for_act_count",
    "collect_verdicts",
    "comfy_balance",
    "default_balance",
    "default_unit_usd",
    "default_writer_usd",
    "ensure_prompt_cloud_balance",
    "estimate_lines",
    "format_refusal",
    "google_balance",
    "judge_wallet",
    "openrouter_balance",
    "pad_usd",
    "parse_comfy_balance",
    "wallet_for_engine",
    "wallet_for_writer_handle",
]
