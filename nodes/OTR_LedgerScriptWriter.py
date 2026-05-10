"""OTR_LedgerScriptWriter — v2.0 LPL writer with legacy-style widget surface restored 2026-05-10.

Pipeline (unchanged from v2.0 LPL):

    1. Validate + normalize inputs (legacy widget set restored).
    2. Resolve effective values:
       - news_seed = custom_premise verbatim if non-empty,
         else RSS auto-fetch via story_orchestrator._fetch_science_news.
       - style = style_custom if non-empty, else style combo (with
         "auto (LLM generates)" sentinel deferred to a model call
         once the LLM is loaded, see _generate_style_via_llm).
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

Widget surface (2026-05-10 restoration):
    required:
        target_words      INT   (canonical length unit; radio ~140 wpm
                                 conversion is only for the est_minutes
                                 output, never for story planning)
        num_characters    INT   (replaces cast_size — kept legacy name for UX)
    optional:
        episode_title     STRING       (stamped into ledger.meta.episode_title)
        model_id          combo        (HF model — story LLM)
        cleanup_model_id  combo        [v2.0 MVP no-op]  — kept for UX parity
        custom_premise    STRING       (RSS override; empty triggers feed fetch)
        include_act_breaks BOOLEAN     [v2.0 MVP no-op]  — outline drives structure
        self_critique     BOOLEAN      [v2.0 MVP no-op]  — script_critic node handles this now
        target_length     combo        (smoke presets force target_words override)
        style             combo        (tonal preset)
        style_custom      STRING       (free-text override; empty falls back to style)
        creativity        combo        (maps to temperature + top_p preset)
        arc_enhancer      BOOLEAN      [v2.0 MVP no-op]
        optimization_profile combo     [v2.0 MVP no-op forward-compat] — plumbed to model loader
        perfect_run_spacesaver BOOLEAN (stamped on ledger.meta for RTXUpscale spacesaver)

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

LAST_LINES_WINDOW = 3
"""Rolling context window size for compose_line. Each character /
announcer beat appends to the window; non-voiced beats do not."""

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


# ---------------------------------------------------------------------------
# target_length preset map. Only smoke presets force a target_words
# override; longer presets are UI labels (the outline schema, not act
# count, drives v2 LPL structure).
# ---------------------------------------------------------------------------

_TARGET_LENGTH_CHOICES = [
    "30 words (smoke, 1 act)",
    "tiny (smoke, 1 act)",
    "short (3 acts)",
    "medium (5 acts)",
    "long (7-8 acts)",
    "epic (10+ acts)",
]

_TARGET_LENGTH_FORCE_WORDS = {
    "30 words (smoke, 1 act)": 30,
    "tiny (smoke, 1 act)":     100,
}


# ---------------------------------------------------------------------------
# Style widget surface — three-way (Jeffrey 2026-05-10):
#   1. Free-text override (`style_custom`) wins when non-empty.
#   2. `style` combo set to "auto (LLM generates)" -> LLM proposes a
#      tonal descriptor from the resolved news_seed.
#   3. Any other combo entry -> used verbatim.
# Both axes — story (custom_premise/RSS) AND style (combo/auto/custom) —
# drive story content; the user wants both selectable.
# ---------------------------------------------------------------------------

_STYLE_AUTO_SENTINEL = "auto (LLM generates)"

_STYLE_CHOICES = [
    _STYLE_AUTO_SENTINEL,
    "tense claustrophobic",
    "space opera epic",
    "psychological slow-burn",
    "hard-sci-fi procedural",
    "noir mystery",
    "chaotic black-mirror",
    "pulp adventure",
    "rust-belt cyber-noir",
    "paranoid procedural",
]

_LLM_STYLE_FALLBACK = "psychological slow-burn"
"""Hardcoded last-resort if the style-generation LLM call fails or
returns empty / unusable text. Kept short so the outline+critic
downstream don't get a degenerate prompt."""


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

_CLEANUP_MODEL_CHOICES = ["auto (use story model)"] + _MODEL_CHOICES

_OPTIMIZATION_PROFILE_CHOICES = [
    "Standard",
    "Pro (Ultra Quality)",
    "Obsidian (UNSTABLE/4GB)",
]


# ---------------------------------------------------------------------------
# Truncating generate_fn wrapper (top_p parametrized 2026-05-10)
# ---------------------------------------------------------------------------


def _build_truncating_generate_fn(cache_entry: dict, *, top_p: float = 0.92):
    """Return a generate_fn (messages, *, temperature, max_new_tokens) that
    left-truncates oversized prompts before model.generate.

    `top_p` is captured in the closure so the creativity widget can
    override it (legacy parity at _otr_legacy_writer.py:768). The
    `temperature` arg per call is whatever the line composer / outline
    passes (compose_line bumps it on retry).

    Cap math: max_input_tokens = max(64, context_cap - max_new_tokens).
    Truncation is left-side (drops oldest tokens, preserves most
    recent context).
    """
    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]
    context_cap = int(cache_entry.get("context_cap") or 8192)
    active_top_p = float(top_p)

    def generate_fn(messages, *, temperature, max_new_tokens):
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

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(temperature),
                top_p=active_top_p,
                max_new_tokens=int(max_new_tokens),
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            out[0][prompt_len:], skip_special_tokens=True,
        )

    return generate_fn


# ---------------------------------------------------------------------------
# Pure helpers (testable without model load)
# ---------------------------------------------------------------------------


def _generate_style_via_llm(
    generate_fn,
    news_seed: str,
    *,
    temperature: float = 0.85,
) -> str:
    """Ask the loaded LLM to suggest a tonal style based on the news_seed.

    Used when the user leaves the `style` widget empty: instead of a
    hardcoded combo selection, the LLM proposes a 3-6 word phrase that
    matches the article's content. Per Jeffrey 2026-05-10: "every other
    style type needs to be generated via LLM not pre-baked in the
    widget".

    Falls back to ``_LLM_STYLE_FALLBACK`` if the LLM is unreachable,
    returns empty text, or returns something obviously off-spec.
    Capped at ~60 chars so downstream prompt budgets stay sane.

    `generate_fn` matches the (messages, *, temperature, max_new_tokens)
    contract returned by ``_build_truncating_generate_fn``.
    """
    seed_excerpt = (news_seed or "").strip()
    if not seed_excerpt:
        return _LLM_STYLE_FALLBACK
    # Cap the excerpt so the prompt doesn't blow the model's context;
    # a 1-2 sentence summary is plenty of grounding for the style call.
    excerpt = seed_excerpt[:600]
    sys_msg = (
        "You are a 1940s sci-fi radio drama showrunner. Given a real "
        "science article, you propose a short tonal style descriptor "
        "for the episode adaptation. Examples of valid output: "
        "'tense claustrophobic', 'noir mystery', 'pulp adventure', "
        "'rust-belt cyber-noir', 'paranoid procedural'."
    )
    user_msg = (
        f"Article seed:\n{excerpt}\n\n"
        f"Reply with ONLY a 3-6 word lowercase tonal style descriptor. "
        f"No quotes, no commentary, no period at the end."
    )
    try:
        out = generate_fn(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=float(temperature),
            max_new_tokens=32,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_LedgerScriptWriter] style LLM-suggest failed (%s); "
            "falling back to %r",
            exc, _LLM_STYLE_FALLBACK,
        )
        return _LLM_STYLE_FALLBACK
    text = (out or "").strip().strip('"\'')
    # Strip the model's usual leading "Style:" / "Suggested:" preamble.
    for prefix in ("style:", "suggested:", "tone:", "descriptor:"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    # Take only the first line and cap length.
    text = text.splitlines()[0].strip() if text else ""
    text = text[:60].rstrip(".,;: ")
    if not text or len(text) < 3:
        log.warning(
            "[OTR_LedgerScriptWriter] style LLM-suggest returned "
            "empty/short %r; falling back to %r",
            text, _LLM_STYLE_FALLBACK,
        )
        return _LLM_STYLE_FALLBACK
    log.info(
        "[OTR_LedgerScriptWriter] style LLM-suggest -> %r (from seed "
        "len=%d)",
        text, len(seed_excerpt),
    )
    return text


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
        "You are titling a single episode of a 1940s sci-fi radio drama. "
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


def _resolve_target_words(target_words, target_length: str) -> int:
    """Apply smoke target_length presets that force a target_words override.

    Non-smoke presets (short/medium/long/epic) are UI labels only — the
    outline schema, not act count, drives v2 LPL structure.
    """
    forced = _TARGET_LENGTH_FORCE_WORDS.get(target_length)
    if forced is not None:
        log.info(
            "[OTR_LedgerScriptWriter] target_length=%r forces target_words=%d "
            "(widget value %r overridden)",
            target_length, forced, target_words,
        )
        return int(forced)
    return max(5, int(target_words))


def _fetch_rss_seed_or_die(style: str, model_id: str) -> str:
    """Run the story_orchestrator RSS fetcher and return a news_seed string.

    Lifts the exact path the legacy writer used. `style` is mapped to
    the closest legacy slug for the LLM re-rank step; if the fetcher
    returns None (every feed failed) we raise loudly — the legacy
    writer behaved the same.
    """
    try:
        try:
            from . import story_orchestrator as _so
        except ImportError:
            import story_orchestrator as _so  # type: ignore
        # Style normalization: re-ranker expects a slug like "hard_sci_fi";
        # use the closest match or fall back to the canonical default.
        slug = (style or "").lower().replace(" ", "_").replace("-", "_")
        if slug not in {"hard_sci_fi", "noir", "psychological_slow_burn",
                        "space_opera_epic", "tense_claustrophobic",
                        "chaotic_black_mirror"}:
            slug = "hard_sci_fi"
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
        return seed_text
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
    cleanup_model_id: str = "auto (use story model)",
    custom_premise: str = "",
    include_act_breaks: bool = True,
    self_critique: bool = True,
    target_length: str = "short (3 acts)",
    style: str = _STYLE_AUTO_SENTINEL,
    style_custom: str = "",
    creativity: str = "balanced",
    arc_enhancer: bool = True,
    optimization_profile: str = "Standard",
    perfect_run_spacesaver: bool = False,
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
    target_words = _resolve_target_words(target_words, target_length)
    num_characters = max(1, min(6, int(num_characters)))
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
        news_seed = custom
        seed_source = "custom_premise"
    else:
        # Best-effort RSS re-rank slug. If style is still pending
        # (auto/LLM), use the hardcoded fallback only for the slug --
        # the writer's final style still gets LLM-proposed from the
        # ACTUAL fetched article below.
        rss_style_slug = resolved_style or _LLM_STYLE_FALLBACK
        news_seed = _fetch_rss_seed_or_die(rss_style_slug, model_id)
        seed_source = "rss_fetch"

    return {
        "news_seed":            news_seed,
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
        "cleanup_model_id":     str(cleanup_model_id or "auto (use story model)"),
        "include_act_breaks":   bool(include_act_breaks),
        "self_critique":        bool(self_critique),
        "target_length":        str(target_length),
        "creativity":           str(creativity),
        "temperature":          float(temperature),
        "top_p":                float(top_p),
        "arc_enhancer":         bool(arc_enhancer),
        "optimization_profile": str(optimization_profile),
        "perfect_run_spacesaver": bool(perfect_run_spacesaver),
    }


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
            "char_id":      cid,
            "name":         name,
            "description":  None,
            "gender":       None,
            "voice_preset": None,
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
        # Widget order matches the legacy OTR_LLMScriptWriter widget
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
                "model_id": (_MODEL_CHOICES, {
                    "default": DEFAULT_MODEL_ID,
                    "tooltip": (
                        "Hugging Face model ID. Mistral-Nemo is the "
                        "validated production default. Suffix tags "
                        "([ALPHA], (EXPERIMENTAL)) are stripped by "
                        "the loader before HF lookup."
                    ),
                }),
                "cleanup_model_id": (_CLEANUP_MODEL_CHOICES, {
                    "default": "auto (use story model)",
                    "tooltip": (
                        "[v2.0 MVP no-op] Legacy two-LLM split widget "
                        "for cleanup phases. v2 LPL uses one model for "
                        "outline + line composition; the structured "
                        "cleanup phases the legacy writer ran are no "
                        "longer in the pipeline. Kept here for UI parity."
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
                        "[v2.0 MVP no-op] Legacy act-break + sponsor-"
                        "message generation. v2 LPL structure is driven "
                        "by the outline schema's beat list; act breaks "
                        "aren't a separate widget-controlled phase."
                    ),
                }),
                "self_critique": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "[v2.0 MVP no-op] Legacy Draft -> Critique -> "
                        "Revise loop. v2 pipeline does this in a "
                        "downstream OTR_LLMScriptCritic node instead "
                        "of inside the writer."
                    ),
                }),
                "target_length": (_TARGET_LENGTH_CHOICES, {
                    "default": "short (3 acts)",
                    "tooltip": (
                        "Length preset. '30 words (smoke)' and 'tiny "
                        "(smoke)' FORCE target_words=30 / 100 (the "
                        "widget value is overridden). short / medium / "
                        "long / epic are UI labels only — v2 LPL "
                        "structure is driven by outline schema, not "
                        "act count."
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
                "arc_enhancer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "[v2.0 MVP no-op] Legacy arc-structure "
                        "enhancer. v2 LPL outline schema enforces a "
                        "narrative arc via its beat list directly."
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
        model_id=DEFAULT_MODEL_ID,
        cleanup_model_id="auto (use story model)",
        custom_premise="",
        include_act_breaks=True,
        self_critique=True,
        target_length="short (3 acts)",
        style=_STYLE_AUTO_SENTINEL,
        style_custom="",
        creativity="balanced",
        arc_enhancer=True,
        optimization_profile="Standard",
        perfect_run_spacesaver=False,
    ):
        """Generate a v2.0 LPL script. See module docstring for pipeline."""

        # --- A. Resolve all widget inputs (RSS fetch happens here) -----
        resolved = _resolve_inputs(
            target_words=target_words,
            num_characters=num_characters,
            episode_title=episode_title,
            model_id=model_id,
            cleanup_model_id=cleanup_model_id,
            custom_premise=custom_premise,
            include_act_breaks=include_act_breaks,
            self_critique=self_critique,
            target_length=target_length,
            style=style,
            style_custom=style_custom,
            creativity=creativity,
            arc_enhancer=arc_enhancer,
            optimization_profile=optimization_profile,
            perfect_run_spacesaver=perfect_run_spacesaver,
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
        from . import _otr_outline as _OTRO
        from . import _otr_canon as _OTRC
        from . import _otr_line_composer as _OTRLC
        from . import _otr_model_loader as _OTRML
        from . import _otr_ledger as _OTRL
        from . import production_ledger as _PL

        # --- C. Load LLM + build truncating generate_fn ---------------
        cache_entry = _OTRML.load_llm(
            resolved["model_id"], device="cuda",
            optimization_profile=resolved["optimization_profile"],
        )
        generate_fn = _build_truncating_generate_fn(
            cache_entry, top_p=resolved["top_p"],
        )

        # --- C2. Style LLM-suggest path (auto sentinel / empty combo) ---
        # When the user picked "auto (LLM generates)" or left the combo
        # blank AND no style_custom override, ask the model to propose
        # a tonal style descriptor based on the resolved news_seed.
        # The widget-typed style_custom and the verbatim combo entries
        # bypass this branch.
        if resolved["style_pending"]:
            generated_style = _generate_style_via_llm(
                generate_fn,
                resolved["news_seed"],
                temperature=resolved["temperature"],
            )
            resolved["style"] = generated_style
            log.info(
                "[OTR_LedgerScriptWriter] style auto-resolved: %r "
                "(LLM proposal from news_seed, fallback was %r)",
                generated_style, _LLM_STYLE_FALLBACK,
            )

        # --- D. Generate validated outline ----------------------------
        outline_req = _OTRO.OutlineRequest(
            news_seed=resolved["news_seed"],
            style_hint=resolved["style"],
            cast_size=resolved["num_characters"],
            target_words=resolved["target_words"],
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

        # --- F. Create per-episode workspace via new_ledger -----------
        led = _PL.new_ledger(episode_id=None)
        episode_id = led.episode_id           # pending_<YYYYMMDD_HHMMSS>
        audio_dir = Path(led.out_dir)         # otr/episodes/<ep>/audio/
        episode_root = audio_dir.parent       # otr/episodes/<ep>/

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

        # --- H. Build cast rows + char_id index ------------------------
        cast_rows, char_id_by_name = _build_cast_rows(outline.cast)
        led.set_cast(cast_rows)

        # --- I. Per-beat loop ------------------------------------------
        line_rows: list = []
        script_text_parts: list = []
        last_lines: list = []  # rolling window of LAST_LINES_WINDOW

        base_temp = resolved["temperature"]

        for beat in outline.beats:
            traits = (beat.mood or "").strip() or DEFAULT_TRAITS
            cleaned: str
            cid: str
            token: str

            if beat.speaker_role == "character":
                line_req = _OTRLC.LineRequest(
                    speaker=beat.speaker,
                    intent=beat.intent,
                    mood=beat.mood,
                    target_words=beat.target_words,
                    canon_header=canon_header,
                    last_lines=list(last_lines),
                )
                cleaned = _OTRLC.compose_line(
                    generate_fn, line_req, base_temperature=base_temp,
                )
                cid = char_id_by_name[beat.speaker]
                token = f"[VOICE: {beat.speaker}, {traits}] {cleaned}"

                last_lines.append((beat.speaker, cleaned))
                if len(last_lines) > LAST_LINES_WINDOW:
                    last_lines.pop(0)

            elif beat.speaker_role == "announcer":
                line_req = _OTRLC.LineRequest(
                    speaker="ANNOUNCER",
                    intent=beat.intent,
                    mood=beat.mood,
                    target_words=beat.target_words,
                    canon_header=canon_header,
                    last_lines=list(last_lines),
                )
                cleaned = _OTRLC.compose_line(
                    generate_fn, line_req, base_temperature=base_temp,
                )
                cid = "announcer"
                token = f"[VOICE: ANNOUNCER, {traits}] {cleaned}"

                last_lines.append(("ANNOUNCER", cleaned))
                if len(last_lines) > LAST_LINES_WINDOW:
                    last_lines.pop(0)

            elif beat.speaker_role in NON_VOICED_ROLES:
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

            line_rows.append({
                "line_id":       beat.beat_id,
                "shot_id":       None,
                "beat_id":       beat.beat_id,
                "char_id":       cid,
                "text":          cleaned,
                "traits":        traits,
                "boundary":      None,
                "bark_wav_path": None,
                "start_s":       None,
                "dur_s":         None,
                "_speaker_role": beat.speaker_role,
            })
            script_text_parts.append(token)

        # --- J. set_lines + post-patch additive speaker_role ----------
        led.set_lines([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in line_rows
        ])
        for r in line_rows:
            _OTRL.patch_line_fields(
                led.data, r["line_id"],
                {"speaker_role": r["_speaker_role"]},
            )

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
        _title_sub_meta = None
        if (
            final_title
            and old_title_in_lines
            and final_title != old_title_in_lines
        ):
            n_lines_patched = 0
            n_subs_total = 0
            for r in line_rows:
                new_text, n_subs = _substitute_title_in_text(
                    r["text"], old_title_in_lines, final_title,
                )
                if n_subs > 0:
                    r["text"] = new_text
                    _OTRL.patch_line_text(
                        led.data, r["line_id"], new_text,
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
        # Mirrors the legacy writer's gen_params_initial stamp so the
        # critic (`script_critic._coerce_params`) can keep reading the
        # same field names. Also stamps episode_title (forward-compat
        # title chain slot) and perfect_run_spacesaver.
        meta = led.data.setdefault("meta", {})
        meta["gen_params_initial"] = {
            "target_words":         resolved["target_words"],
            "num_characters":       resolved["num_characters"],
            "model_id":              resolved["model_id"],
            "cleanup_model_id":      resolved["cleanup_model_id"],
            "style":                 resolved["style"],
            "style_combo":           resolved["style_combo"],
            "style_custom":          resolved["style_custom"],
            "style_source":          resolved["style_source"],
            "creativity":            resolved["creativity"],
            "temperature":           resolved["temperature"],
            "top_p":                 resolved["top_p"],
            "target_length":         resolved["target_length"],
            "include_act_breaks":    resolved["include_act_breaks"],
            "self_critique":         resolved["self_critique"],
            "arc_enhancer":          resolved["arc_enhancer"],
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

        # --- L. Assemble return values --------------------------------
        script_text = "\n\n".join(script_text_parts)
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
    try:
        spec = cls.INPUT_TYPES()
        assert "required" in spec, "missing required block"
        assert "optional" in spec, "missing optional block"
        # Required block: legacy widget order — episode_title, target_words, num_characters.
        req_keys = list(spec["required"].keys())
        assert req_keys == ["episode_title", "target_words", "num_characters"], \
            f"required widget order drift: {req_keys}"
        for k in ("model_id", "cleanup_model_id",
                  "custom_premise", "include_act_breaks", "self_critique",
                  "target_length", "style", "style_custom", "creativity",
                  "arc_enhancer", "optimization_profile",
                  "perfect_run_spacesaver"):
            assert k in spec["optional"], f"optional missing key: {k}"
        # open_close MUST be absent (dropped 2026-05-10).
        assert "open_close" not in spec["required"]
        assert "open_close" not in spec["optional"]
        # episode_title is a STRING (default empty).
        et_type, et_meta = spec["required"]["episode_title"]
        assert et_type == "STRING"
        assert et_meta.get("default") == ""
        # target_words INT clamps + default (locked to 350 per Jeffrey 2026-05-10)
        tw_type, tw_meta = spec["required"]["target_words"]
        assert tw_type == "INT"
        assert tw_meta["min"] == 30 and tw_meta["max"] == 10000
        assert tw_meta["default"] == 350, \
            f"target_words default drift: {tw_meta['default']!r} (must be 350)"
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
        # target_length first two are smoke presets
        tl_choices, _ = spec["optional"]["target_length"]
        assert tl_choices[0].startswith("30 words")
        assert tl_choices[1].startswith("tiny")
        # style combo: first entry is the LLM-auto sentinel; remaining
        # entries are the baked-in tonal presets the user can pick.
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
        print("[2/9] PASS: INPUT_TYPES schema (15 widgets, open_close absent, style 3-way)")
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

        # 5c. target_length forces target_words for smoke presets.
        assert _resolve_target_words(350, "30 words (smoke, 1 act)") == 30
        assert _resolve_target_words(350, "tiny (smoke, 1 act)") == 100
        # 5d. Non-smoke presets pass widget value through.
        assert _resolve_target_words(350, "short (3 acts)") == 350
        assert _resolve_target_words(1400, "epic (10+ acts)") == 1400

        print("[5/9] PASS: resolver helpers (creativity + target_length)")
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

    # 7. _generate_style_via_llm fallback behavior (no actual LLM call).
    try:
        # 7a. Empty seed -> hardcoded fallback (no LLM dereference).
        assert _generate_style_via_llm(lambda *a, **kw: "x", "") == _LLM_STYLE_FALLBACK

        # 7b. LLM raises -> fallback.
        def _raises(*a, **kw):
            raise RuntimeError("LLM offline")
        assert _generate_style_via_llm(_raises, "some seed text") == _LLM_STYLE_FALLBACK

        # 7c. LLM returns clean phrase -> stripped + returned verbatim.
        def _good(*a, **kw):
            return "  bleak industrial noir  \n"
        assert _generate_style_via_llm(_good, "seed") == "bleak industrial noir"

        # 7d. LLM returns labeled preamble -> preamble stripped.
        def _preamble(*a, **kw):
            return "Style: claustrophobic procedural"
        assert _generate_style_via_llm(_preamble, "seed") == "claustrophobic procedural"

        # 7e. LLM returns empty -> fallback.
        def _empty(*a, **kw):
            return ""
        assert _generate_style_via_llm(_empty, "seed") == _LLM_STYLE_FALLBACK

        # 7f. LLM returns too-short -> fallback.
        def _short(*a, **kw):
            return "ok"
        assert _generate_style_via_llm(_short, "seed") == _LLM_STYLE_FALLBACK

        print("[7/9] PASS: _generate_style_via_llm (5 paths + fallback chain)")
    except Exception:
        failures.append(("7/9 _generate_style_via_llm", traceback.format_exc()))
        print("[7/9] FAIL: _generate_style_via_llm")

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
