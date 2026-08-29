"""The writer's TAIL -- everything from the title regen to the ledger save.

Lean-mean order 9, slice 2. ``_run_writer_tail`` is 911 lines, it is the last
third of every episode the pack has ever produced, and it was the largest single
thing left in a 7,417-line file. It is now here, with the ten helpers and two
contracts that only it owns.

IT IS A MIXIN AND THAT IS THE WHOLE POINT. Measured before it was touched: the
tail reads ``self`` ZERO times -- it is a module function wearing a method's
clothes, exactly as ``WriterTailContext`` has always claimed ("the tail consumes
ONLY this context"). Turning it into a free function would have meant
re-indenting 911 lines, and a reflowed body cannot be hash-checked against the
one that shipped. As a mixin method the body moves BYTE FOR BYTE, its signature
is unchanged, ``self._run_writer_tail(ctx, ...)`` still reads the same at the
call site, and every test that reaches for
``OTR_LedgerScriptWriter._run_writer_tail`` still finds it.

WHAT CAME WITH IT, and why nothing else did. The helpers here are the tail's
transitive closure, measured rather than guessed: title generation and its
excerpt set, the cast RNG seed, the intro-rewrite application, the final slot
telemetry, the news payload, the story-style receipt, and the custom-lane title
provenance. Seven of them have no other caller in the pack. THREE do --
``_resolve_cast_rng_seed``, ``_apply_intro_rewrite_result`` and
``_stamp_story_style_receipt`` are also called from ``run()`` -- and they still
moved, because the alternative is the writer importing from here while this
module imports back from the writer. The dependency runs ONE WAY: the writer
imports this module; this module never imports the writer.

Every name is re-exported from ``OTR_LedgerScriptWriter`` under its original
spelling, so no caller and no test had to learn a new address.

UTF-8, no BOM, ASCII source.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Protocol

# The same lazy, stdlib-only sibling modules the writer imports, in the same
# spelling. None of them import back into the writer or into this module.
from . import _otr_lane_specs as _LANES
from . import _otr_model_catalog as _otr_model_catalog
from . import _otr_rolls as _ROLLS
from . import _otr_visual_styles as _otr_visual_styles
from . import _otr_word_delivery as _OTRWD
from ._otr_story_brief import (
    REJECT_JSON_PARSE as _STORY_BRIEF_REJECT_JSON_PARSE,
    REJECT_SCHEMA as _STORY_BRIEF_REJECT_SCHEMA,
    run_produced_story_summary,
    run_story_brief_reflection,
)

log = logging.getLogger("OTR")


WORDS_PER_MINUTE_ESTIMATE = 140
"""Word-per-minute estimate for the est_minutes output socket only.
Story planning is words-only; this constant is never used to derive
a target_seconds input to the LLM. Mirrors legacy at
story_orchestrator.py:6584."""


STORY_STYLE_STATUS_SCAFFOLD_OFF = "story_scaffold_off"
"""Durable receipt for intentional no-story-style runs.

This is deliberately NOT written to ``meta.style``. ``meta.style`` remains the
story-grammar contract slug; ``meta.visual_style`` remains the visual prompt
pack selector. When the scaffold is off, credits can prove the absence of
``meta.style`` is intentional without ever borrowing the visual style id.
"""


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
    # QA F1 (2026-07-09): bank-aware title framing. The system prompt used to
    # hardcode "sci-fi radio drama" for EVERY bank; the caller now threads the
    # bank's banks.json `title_form_label` (first live consumer of that
    # field). Default keeps legacy callers/self-tests byte-identical.
    title_form_label: str = "sci-fi radio drama",
    # PBUG-20260815-05 (2026-08-19): the work this episode adapts, already
    # lane-gated by the caller. Default "" keeps every legacy caller and
    # self-test byte-identical, and the original lane (which adapts nothing)
    # renders exactly the prompt it always did.
    work_title: str = "",
) -> str:
    """Generate an episode title via a forced scratchpad pass.

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

    Returns the cleaned authored title, or an empty string on an LLM
    failure, missing `TITLE:` line, or wrapper-only output. The caller
    falls back to outline.title on an empty result.

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

    # PBUG-20260815-05: this pass saw dialogue excerpts and a generic bank
    # label, and nothing anywhere in its context named the work it was
    # adapting. A Macbeth scene therefore titled itself after The Tempest --
    # a SIBLING PLAY in the same curated manifest -- free-associated off the
    # scene's genuine storm sound-world.
    #
    # The work title enters as a DISAMBIGUATING ANCHOR, never as title
    # material, and that distinction is the whole design. Told only "this is
    # Macbeth", a model happily answers "The Macbeth Prophecy" on every
    # adaptation episode -- trading a rare fidelity defect for a constant
    # blandness one, which THE LAW does not license either. So the anchor
    # comes with the rule that keeps the name OUT of the title.
    #
    # No sibling title is ever named. The craft rule this bug's own log entry
    # cites -- never put the feared failure in the model's context -- is
    # respected: the model is told what it IS adapting, never what it must
    # not say.
    #
    # Capped like every other field in this prompt (excerpts 1200, premise
    # 600, arc 300). A manifest row is DATA, and an uncapped field is how a
    # malformed one reaches the composer's token budget. The longest title in
    # the shipped corpus is 45 chars ("The Surprising Adventures of Baron
    # Munchausen"), so this never binds in practice -- it is a floor under a
    # bad row, not a policy about titles.
    work_str = (work_title or "").strip()[:120].strip()
    anchor_block = f"THIS EPISODE ADAPTS: {work_str}\n\n" if work_str else ""
    anchor_rule = (
        f" - this episode adapts {work_str}; keep that name OUT of the "
        "title, and never name a different work\n"
    ) if work_str else ""

    _form = (title_form_label or "").strip() or "sci-fi radio drama"
    sys_msg = (
        f"You are titling a single episode of a {_form}. "
        "You receive the finished story material and propose an "
        "specific, evocative episode title. You work on a scratchpad "
        "first, then commit to a final answer."
    )
    user_msg = (
        f"{story_block}\n\n"
        f"{anchor_block}"
        "Title this episode. Work through these steps in order:\n\n"
        "DETAILS: list 3 concrete physical details actually present "
        "in the story above -- a specific object, place, sound, or "
        "image, one per line.\n"
        "CANDIDATES: draft 3 distinct candidate episode titles, each "
        "drawing on one of those details, one per line.\n"
        "TITLE: on the final line, write the single best title from "
        "your candidates.\n\n"
        "Rules for the final title:\n"
        " - use a non-empty authored title\n"
        " - draw from a vivid image, important object, character, or "
        "thematic tension actually present in the story\n"
        " - feel specific and memorable, not generic\n"
        " - avoid cliches like \"The Beginning\", \"Final Chapter\", "
        "\"Untitled\", or \"Episode X\"\n"
        f"{anchor_rule}"
        "\n"
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

    # Any non-empty authored title is valid. Python phrase lists and word
    # ceilings are not story-quality judges and must not discard model output.

    log.info(
        "[OTR_LedgerScriptWriter] title regen -> %r (scratchpad pass, "
        "from %d-char script)",
        candidate, len(text),
    )
    return candidate


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
        # OTR_CAST_SEED WITHOUT OTR_C7 IS ALMOST ALWAYS A LEAK, not a choice.
        # The soak launcher only ever sets this variable inside its C7 branch,
        # and it sets OTR_STYLE_SEED alongside it. A lone CAST_SEED therefore
        # means an inherited value from some earlier shell -- which is exactly
        # what happened on 2026-08-22, when all four legs of a motion-module
        # bake-off cast GULLIVER REEVES and the operator caught it by watching
        # the episodes rather than by reading a log. Say so at WARNING, name
        # the cast it will produce, and name the variable to clear.
        if not os.environ.get("OTR_C7", "").strip():
            log.warning(
                "CAST SEED IS PINNED TO %s BUT OTR_C7 IS NOT SET. Every "
                "episode from this server will open with the SAME cast (seed "
                "42 rolls HAYES VANCE / GULLIVER REEVES / JIMBO BLACK). This "
                "is a leaked environment variable unless you meant it -- a "
                "real byte-identity run sets OTR_C7=1 as well. Clear it with "
                "`set OTR_CAST_SEED=` and reboot the server to get fresh "
                "per-episode casting back (BUG-LOCAL-269).", env)
        return int(env), "OTR_CAST_SEED override"
    return random.SystemRandom().getrandbits(32), "OS entropy"


def _apply_intro_rewrite_result(
    led, first_announcer_id, new_text, flag, extra_flags=(),
):
    """Apply the intro-rewrite outcome to the ledger intro row.

    Sits OUTSIDE the rewrite try/except BY DESIGN (kibitz r4 P1): a
    missing intro row is ledger corruption -> RuntimeError, never
    swallowed as a "rewrite failure". Flags are READ-EXTEND-PATCH
    (patch_line_fields REPLACES the field; the row carries in-loop
    telemetry -- announcer_intro / open_safe_fallback -- that must
    survive the rewrite). ``new_text`` None = the rewrite failed:
    the in-loop text stands and only the failure flag lands.
    """
    # Lazy dual import (repo import-isolation convention -- the ledger
    # helpers are run()-local everywhere else in this module).
    try:
        from . import _otr_ledger as _OTRL
    except ImportError:  # pragma: no cover -- flat test/standalone load
        import _otr_ledger as _OTRL  # type: ignore

    row = None
    for _ln in led.data.get("lines") or []:
        if isinstance(_ln, dict) and _ln.get("line_id") == first_announcer_id:
            row = _ln
            break
    if row is None:
        raise RuntimeError(
            f"[OTR_LedgerScriptWriter] intro rewrite: no ledger row with "
            f"line_id={first_announcer_id!r} -- ledger skeleton and outline "
            f"have drifted apart (corruption, not a rewrite failure)"
        )
    flags = list(row.get("compose_flags") or [])
    if new_text:
        if not _OTRL.patch_line_text(led.data, first_announcer_id, new_text):
            raise RuntimeError(
                f"[OTR_LedgerScriptWriter] intro rewrite: patch_line_text "
                f"returned False for line_id={first_announcer_id!r}"
            )
    flags.extend(str(value) for value in (extra_flags or ()) if str(value))
    flags.append(flag)
    flags = list(dict.fromkeys(flags))
    if not _OTRL.patch_line_fields(
        led.data, first_announcer_id, {"compose_flags": flags},
    ):
        raise RuntimeError(
            f"[OTR_LedgerScriptWriter] intro rewrite: patch_line_fields "
            f"returned False for line_id={first_announcer_id!r}"
        )


def _stamp_final_slot_telemetry(
    *, meta, resolved, slot_scheduler, pipeline_id: str,
    title_source: str,
) -> None:
    """Stamp the authoritative slot receipt after the final writer LLM call."""
    meta["slot_transitions"] = int(slot_scheduler.transitions)
    meta["slot_calls_by_slot"] = dict(slot_scheduler.calls_by_slot)
    meta["slot_calls_by_helper"] = {
        helper: dict(buckets)
        for helper, buckets in slot_scheduler.slot_calls_by_helper.items()
    }
    meta["slot_transitions_by_phase"] = [
        dict(record) for record in slot_scheduler.slot_transitions_by_phase
    ]
    if _LANES.is_dispatched(pipeline_id):
        # Custom runners name every structured pass through helper_context.
        # Derive rows from that executed journal; never claim legacy phases.
        params = {}
        for helper, buckets in slot_scheduler.slot_calls_by_helper.items():
            active = [slot for slot, count in buckets.items() if int(count) > 0]
            if len(active) != 1:
                continue
            slot = active[0]
            params[str(helper)] = {
                "slot": slot,
                "model": resolved[
                    "creative_writing_model" if slot == "creative"
                    else "technical_model"
                ],
            }
        meta["gen_params_by_phase"] = params
        return
    params = {}
    if meta.get("news") is not None:
        source_receipt = meta.get("source_interpreter")
        source_slot = "technical"
        source_model = resolved["technical_model"]
        if isinstance(source_receipt, dict):
            source_model = str(
                source_receipt.get("model") or source_model)
            if source_model == "deterministic":
                source_slot = "deterministic"
        params["news_interpreter"] = {
            "slot": source_slot, "model": source_model,
        }
    for phase in ("cast_lock", "outline", "dialogue_composer"):
        params[phase] = {
            "slot": "creative", "model": resolved["creative_writing_model"],
        }
    if title_source == "llm_post_composition":
        params["title_regen"] = {
            "slot": "creative", "model": resolved["creative_writing_model"],
        }
    meta["gen_params_by_phase"] = params


def _build_news_payload(
    outline,
    news_seed: str,
    seed_source: str,
    *,
    source_label: str = "",
    origin_label: str = "",
    headline_override: str = "",
) -> str:
    """Build the slot-2 news_used JSON string.

    1-element JSON array matching legacy article shape. seed_source flags
    whether the body came from a user-typed custom_premise or from the
    RSS fetcher. (The old story_orchestrator:5141 pointer is stale --
    kibitz r4 P5: real consumers are the FreezeCascade passthrough +
    video_engine's HUD/treatment readers.)

    kibitz r2-r4 provenance surface: the three keyword args are
    DATA-DRIVEN extensions resolved by the caller (bank defaults +
    final title). All default "" -> legacy lanes byte-identical.
    origin_label, when present, rides the entry dict; the video HUD
    reads it with "NEWS SEED" as the legacy default.
    """
    source = source_label or (
        "User Seed" if seed_source == "custom_premise" else "RSS Auto-Fetch"
    )
    entry = {
        "headline":  headline_override or outline.title,
        "summary":   outline.premise[:500],
        "full_text": news_seed,
        "source":    source,
        "date":      datetime.now().date().isoformat(),
        "link":      "",
    }
    if origin_label:
        entry["origin_label"] = origin_label
    return json.dumps([entry], indent=2, ensure_ascii=False)


def _stamp_story_style_receipt(meta: dict, *, contract,
                               scaffold_enabled: bool) -> None:
    """Stamp the story-style receipt without crossing visual/story channels.

    ``meta.style`` is only the story-grammar slug from ``contract``. If story
    grammar is intentionally disabled, stamp a separate status receipt so late
    consumers can distinguish "no story style by design" from "style build
    failed." Never fill ``meta.style`` from ``meta.visual_style``.
    """
    meta["story_scaffold_enabled"] = bool(scaffold_enabled)
    if contract is not None:
        meta["style"] = contract.slug
        meta.pop("story_style_status", None)
    elif not scaffold_enabled:
        meta.pop("style", None)
        meta["story_style_status"] = STORY_STYLE_STATUS_SCAFFOLD_OFF
    else:
        meta.pop("story_style_status", None)


def _title_source_for_custom_override(source_bank_row: Any) -> str:
    """Return truthful custom-lane title provenance without changing ctx."""
    bank_id = str(getattr(source_bank_row, "source_bank_id", "") or "").strip()
    # The special case that used to sit here returned the LEGACY literal
    # `fable2_script_title` for this one lane (PBUG-20260712-05 preserved it
    # when every custom runner was wrongly stamped with that lane's value).
    # The 2026-08-16 rename made the legacy literal equal to what the generic
    # branch already derives, so the branch was dead and is gone. Frozen
    # ledgers keep their old `fable2_script_title` string on disk; nothing
    # branches on this value, it is provenance telemetry.
    if bank_id:
        return f"{bank_id}_script_title"
    return "custom_pipeline_script_title"


@dataclass
class WriterTailContext:
    """Everything the writer's tail consumes -- scifi_news_pro S1a (r1/C2,
    fields PINNED r2 by direct read of the tail body; one name only).

    The tail (`OTR_LedgerScriptWriter._run_writer_tail`) spans, in order:
    J.5 title regen -> canon write -> K meta stamps -> K.5 visual_plan ->
    K.5.5 story-brief reflection -> K.5.6 produced-story summary ->
    Wave-2 story-spine orchestrator (or the writer-LLM unload) ->
    REJECT gate -> provenance stamps -> L return assembly -> M save.

    The legacy path BUILDS this from its run() locals (byte-identical
    behavior: final_title_override=None, run_story_spine=True keeps the
    env-gated spine default). The scifi_news_pro lane (S1b+) builds it from its
    parsed artifacts. The tail consumes ONLY this context -- no closure
    over run() locals.
    """

    led: Any
    meta: dict
    resolved: dict
    outline_view: Any          # needs: .premise, .title (regen grounding +
                               # fallback + consistency guard + news payload).
                               # scifi_news_pro: premise = treatment.dramatic_question
                               # line, title = treatment.title
    canon: Any                 # episode canon object; the tail is the only
                               # canon WRITER (J.5 re-titles + writes it)
    episode_root: Any
    episode_id: str
    contract: Any | None       # style contract; scifi_news_pro = None ("" slug path)
    style_grammar_on: bool     # scifi_news_pro = False (receipt stamp honest)
    source_bank_row: Any       # defaults: title_form_label, hud_origin_label
    slot_scheduler: Any
    creative_fn: Any
    technical_fn: Any
    run_story_spine: bool      # legacy True (env-gated as today); scifi_news_pro
                               # FALSE -- its P4/P5/P8 loop is the lane's
                               # equivalent; revisit post-S3
    final_title_override: str | None
                               # r3/M3: scifi_news_pro sets the play's parsed TITLE
                               # here (title_source="scifi_news_pro_script_title").
                               # Tail precedence: user-typed episode_title >
                               # override > LLM regen. Legacy passes None ->
                               # byte-identical behavior.
    style_roll: Any = None
    """The visual-style roll receipt, or None for a direct pick.

    ADDED 2026-08-23 TO FIX A LIVE NameError, and the docstring above is why it
    had to be a FIELD. The tail's dynamic-style FLOOR FALLBACK -- the branch
    that runs when the dynamic visual-style reflection fails -- read a bare
    `_style_roll`, which is a LOCAL OF `run()`. Two sibling methods share no
    scope, so that compiled to a global load with nothing behind it: reaching
    that branch raised `NameError: name '_style_roll' is not defined` and took
    the whole tail down instead of falling back to a floor style.

    It survived because the invariant was tested with `co_freevars == ()`, and
    a sibling method's local can NEVER appear as a free variable -- it compiles
    to LOAD_GLOBAL. The test was checking a thing that could not fail. The
    replacement walks the bytecode for global loads the module does not define,
    which is what actually catches this.

    Defaulted so the scifi_news_pro lane and every existing caller build the
    context exactly as before; a direct visual-style pick has no roll receipt
    and None is what that branch already handles.
    """


class TailFinalizer(Protocol):
    """Optional lane-owned proof hook executed around the writer save.

    Existing lanes pass ``None`` and retain their byte-for-byte tail path.
    New source-bank lanes use this narrow protocol to prove their receipts,
    run the freeze audits after all writer metadata mutations, and verify the
    persisted JSON without changing any spoken text.
    """

    def before_save(self, *, ctx: WriterTailContext) -> None: ...

    def after_save(
        self, *, saved_path: str, ledger_data: Mapping[str, Any]
    ) -> None: ...


class WriterTailMixin:
    """Carries ``_run_writer_tail`` onto the writer node.

    A mixin rather than a free function so the 911-line body below is the SAME
    BYTES that shipped -- see the module docstring. It holds no state and reads
    no ``self``; the tail's only input is its context object.
    """

    def _run_writer_tail(
        self, ctx: "WriterTailContext", *,
        tail_finalizer: "TailFinalizer | None" = None,
    ) -> tuple[str, str, str, float, str]:
        """The writer's tail: J.5 title regen -> canon write -> K meta
        stamps -> K.5 visual_plan -> K.5.5/K.5.6 reflections -> Wave-2
        story-spine orchestrator (or writer-LLM unload) -> REJECT gate ->
        provenance stamps -> L return assembly -> M save.

        Consumes ONLY ``ctx`` (scifi_news_pro S1a extraction -- no closure
        over run() locals). Returns
        ``(script_text, script_json, news_json, est_minutes,
        technical_model)`` -- the writer's output tuple.
        """
        # Late imports (pure modules -- same no-load-at-import law as
        # run()'s section B; these two were run() locals pre-extraction).
        from . import _otr_canon as _OTRC
        from . import production_ledger as _PL

        led = ctx.led
        meta = ctx.meta  # same object as led.data["meta"]; K re-derives it
        resolved = ctx.resolved
        outline = ctx.outline_view
        canon = ctx.canon
        episode_root = ctx.episode_root
        episode_id = ctx.episode_id
        contract = ctx.contract
        _style_grammar_on = ctx.style_grammar_on
        _source_bank_row = ctx.source_bank_row
        slot_scheduler = ctx.slot_scheduler
        creative_generate_fn = ctx.creative_fn
        technical_generate_fn = ctx.technical_fn

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
        elif ctx.final_title_override is not None:
            # Custom lanes supply an accepted authored TITLE via ctx; title
            # regen never runs because it would discard that lane-owned title.
            # Legacy passes None -> this branch never fires there.
            final_title = ctx.final_title_override
            title_source = _title_source_for_custom_override(
                ctx.source_bank_row
            )
        else:
            # kibitz r3 D4 (2026-07-09) ROOT-CAUSE FIX: assemble from the
            # CANONICAL ledger. The writer's old in-loop token list never saw
            # the I.5 outro overwrite (title regen was reading the
            # deterministic PLACEHOLDER close) and would never have seen the
            # I.4.9 intro rewrite. Same authority the slot-0 output uses
            # (section L below). That list was removed outright 2026-08-28 --
            # nothing ever read it, so "diagnostic-only" was generous.
            assembled_script = _PL.assemble_script_text_from_ledger(led.data)
            # PBUG-20260815-05: anchor the title pass to the work it is
            # actually adapting. `identity_from_meta` is the SINGLE
            # bibliographic authority -- it reads `play_title` for shakespeare
            # and `title` for public_domain -- so no per-lane branch is needed
            # here and no second reader is grown.
            #
            # THREE INVARIANTS, each one a defect that already shipped once at
            # the sibling identity read further down this same method:
            #  - the import is METHOD-LOCAL. `_run_writer_tail` is a separate
            #    METHOD, not a closure over run(), so a name bound in run()
            #    raises NameError here on EVERY episode -- and an enclosing
            #    `except Exception` swallows it, leaving a fix that is dead
            #    code while the suite stays green.
            #  - the read is INSIDE the try. A synthetic or partial `meta`
            #    makes `identity_from_meta` raise, and nothing about naming
            #    the work may ever be able to fail an episode.
            #  - the LANE GATE is applied, never truthiness. `work_title`
            #    holds the PUBLICATION on media_archive (56 of 98 live ledgers
            #    carry a `source_label` like "Now See Hear!"), so an ungated
            #    read would anchor a feed post's title to a magazine name --
            #    inventing a work instead of naming one, which is a worse
            #    fidelity defect than the wrong-play title being fixed.
            _title_work = ""
            try:
                try:
                    from . import _otr_source_identity as _OTRSID_TITLE
                except ImportError:  # pragma: no cover -- flat/standalone load
                    import _otr_source_identity as _OTRSID_TITLE  # type: ignore
                _title_identity = _OTRSID_TITLE.identity_from_meta(meta)
                if (
                    _title_identity.source_kind
                    in _OTRSID_TITLE.ADAPTATION_SOURCE_KINDS
                ):
                    _title_work = str(_title_identity.work_title or "")
                # A positive receipt, so a PUBLISHED episode can prove which
                # way this went without a re-run. Stamped ONLY on a successful
                # read -- the same convention `meta["bank_roll"]` uses a few
                # hundred lines up: an ABSENT key means the read raised, an
                # empty string means the read succeeded and the lane
                # legitimately adapts nothing. Collapsing those two into one
                # falsy value is the `voice_cast_decision == {}` ambiguity
                # that cost a whole arc to diagnose.
                meta["title_work_anchor"] = _title_work
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_LedgerScriptWriter] title work-anchor unavailable "
                    "(%s); titling without it",
                    exc,
                )
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
                    # QA F1 (2026-07-09): bank-aware framing via banks.json
                    # title_form_label (science value == the old hardcode).
                    title_form_label=str(
                        (getattr(_source_bank_row, "defaults", {}) or {})
                        .get("title_form_label") or "sci-fi radio drama"
                    ),
                    work_title=_title_work,
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
            "act_count":            resolved["act_count"],
            "num_characters":       resolved["num_characters"],
            # S30 B2b: the legacy `model_id` key is DELETED outright.
            # Every consumer that previously read meta.gen_params_initial.
            # model_id now reads creative_writing_model + technical_model
            # explicitly (B3 onward).
            "creative_writing_model": resolved["creative_writing_model"],
            "technical_model":        resolved["technical_model"],
            "creativity":            resolved["creativity"],
            "temperature":           resolved["temperature"],
            "top_p":                 resolved["top_p"],
            "act_count":             resolved["act_count"],
            "include_act_breaks":    resolved["include_act_breaks"],
            "optimization_profile":  resolved["optimization_profile"],
            "seed_source":           resolved["seed_source"],
            "source_ref":            resolved["source_ref"],
        }
        # Actual word delivery is intentionally NOT stamped here. Story-spine,
        # final hygiene, and producer-owned lane work may still change rows.
        # The exact current surface is fitted and hash-stamped at the final
        # inline/content boundary below, immediately before reflections and
        # readiness consumers.
        # Post-ship audit fix (2026-07-10): stamp the resolved runtime
        # policy into the ledger so DOWNSTREAM LLM consumers (freeze
        # cascade reviewer, shot-lock derivation) run under the SAME
        # policy -- not a silent nv50-baseline fallback on other tiers.
        _pol = resolved["llm_policy"]
        meta["llm_policy"] = {
            "device": _pol.device,
            "attn_impl": _pol.attn_impl,
            "quant_policy": _pol.quant_policy,
            "vram_ceiling_gb": _pol.vram_ceiling_gb,
            "gguf_n_ctx": _pol.gguf_n_ctx,
            "gguf_quant": _pol.gguf_quant,
            "lane_allowlist": list(_pol.lane_allowlist),
        }
        # GGUF row registry (2026-07-16): serialize the immutable per-slot GGUF
        # load_config the writer actually loaded under (resolved path / quant /
        # n_ctx / n_batch / n_gpu_layers / kv / seed + algo / pinned top_k /
        # sampling / stop / think). Downstream consumers read THIS, not the env.
        # Empty dict for a non-GGUF run.
        meta["llm_gguf_load_config"] = {
            _slot: _lc.as_receipt()
            for _slot, _lc in slot_scheduler.load_config_by_slot.items()
        }
        # S30 B2b: top-level slot stamps + per-phase routing trace.
        # `gen_params_by_phase` records the slot + resolved model for
        # each writer-level LLM phase that fired. Critic / cascade
        # phases that live in B3+ nodes stamp their own entries when
        # they land.
        meta["creative_writing_model"] = resolved["creative_writing_model"]
        meta["technical_model"]        = resolved["technical_model"]
        # Slot/helper/phase receipts are stamped once, after all shared-tail
        # reflection and story-spine calls. An earlier snapshot omitted those
        # calls and falsely described custom runners with legacy phase names.
        # Always stamp the resolved final title (user / LLM regen / outline
        # fallback). title_source records which branch won so downstream
        # consumers and BUG_LOG forensics can tell user-typed from
        # LLM-regenerated runs without inspecting widget state.
        meta["episode_title"] = final_title
        meta["title_source"] = title_source

        # --- J.7. The announcer's WORK phrase becomes Python-owned ---------
        # (J.6 is a TOMBSTONE -- the retired post-hoc title-substitution
        # section, pinned removed by test_post_hoc_title_substitution_is_
        # removed. Do not revive that label for new work.)
        # PBUG-20260817-04. The model was handed `WORK: a scene from Nonsense
        # Novels` -- by a seam that literally says "Use ONLY the WORK title
        # ...; invent none" -- and announced "The Adventure of the Purloined
        # Paper", a work that does not exist. The closing coda in the SAME
        # episode named the source correctly, because the coda is a TEMPLATE.
        # So the work-title half stops being composed and becomes rendered,
        # exactly as `compose_news_coda` already does with its fact.
        #
        # HERE, AND NOT AT EITHER COMPOSE SITE, because the subtitle form
        # needs `meta["episode_title"]`, which binds directly above at J.5 --
        # AFTER both the in-loop intro and the rewrite. A splice at either
        # call site ships the subtitle branch reading a title that does not
        # exist yet.
        #
        # `_work_title` is the LANE-GATED value (ADAPTATION_SOURCE_KINDS); a
        # fresh `identity_from_meta` read here would reopen the media_archive
        # collision, where `work_title` holds the PUBLICATION and 56 of 98
        # live ledgers carry one. Empty work title -> no phrase, untouched line.
        # The lane gate is RE-APPLIED here, not bypassed. `run()`'s
        # `_work_title` local is not in the tail's scope, and the one thing
        # this recompute may not do is drop the gate: `identity_from_meta`
        # maps media_archive's `source_label` onto the SAME field, so an
        # ungated read announces "a scene from Now See Hear!" on 57% of that
        # lane -- a worse fidelity defect than the one being fixed.
        # THE WHOLE BLOCK IS GUARDED, identity read included. An earlier
        # version computed the identity OUTSIDE the try and took ten tail
        # tests red: a synthetic or partial `meta` makes `identity_from_meta`
        # raise, and an unprotected read there kills the writer tail on lanes
        # that were never going to get a frame at all. Nothing about naming
        # the work may ever be able to fail an episode.
        try:
            # BOTH imports are local because `_run_writer_tail` is a SEPARATE
            # METHOD, not a closure over run() -- its docstring says it
            # consumes only `ctx`. `_OTRLC` is bound inside run(), so reaching
            # for it here raised NameError on EVERY episode, and this block's
            # own `except Exception` swallowed it: the fix was dead code that
            # logged a warning and left the invented title standing. Green
            # tests proved only that it failed safely, which is not the same
            # as working.
            try:
                from . import _otr_line_composer as _OTRLC_TAIL
                from . import _otr_source_identity as _OTRSID_TAIL
            except ImportError:  # pragma: no cover -- flat/standalone load
                import _otr_line_composer as _OTRLC_TAIL  # type: ignore
                import _otr_source_identity as _OTRSID_TAIL  # type: ignore
            _tail_identity = _OTRSID_TAIL.identity_from_meta(meta)
            _tail_work_title = (
                _tail_identity.work_title
                if _tail_identity.source_kind
                in _OTRSID_TAIL.ADAPTATION_SOURCE_KINDS
                else ""
            )
            _src_meta = meta.get("source_meta") or {}
            if not isinstance(_src_meta, dict):
                _src_meta = {}
            _work_frame = _OTRLC_TAIL.build_work_frame(
                work_title=_tail_work_title,
                author=str(_tail_identity.author or ""),
                act=_src_meta.get("act"),
                scene=_src_meta.get("scene"),
                episode_title=final_title,
            )
            # The intro row is found HERE rather than threaded: the tail does
            # not carry `first_announcer_id` (it is a compose-loop local), and
            # reaching for it raises NameError on every adaptation episode.
            # The FIRST announcer row is the opening by construction -- music
            # rows speak as RADIO, and the outro is the LAST announcer row.
            _intro_row = next(
                (r for r in (led.data.get("lines") or [])
                 if isinstance(r, dict)
                 and str(r.get("speaker") or "").upper() == "ANNOUNCER"),
                None,
            ) if _work_frame else None
            if _intro_row is not None and _intro_row.get("line_id"):
                _intro_id = _intro_row["line_id"]
                _spliced = _OTRLC_TAIL.splice_work_frame(
                    str(_intro_row.get("text") or ""), _work_frame,
                )
                if _spliced:
                    # PROTECTED_FACT_COMPONENT_FLAG or the clean stage
                    # rewrites the span we just fixed -- the shipped row
                    # already carries `unclean_spoken_text`, and an
                    # unprotected Python-owned span is how PBUG-20260815-01
                    # deleted the coda attribution on 9 of 14 voiced rows.
                    _apply_intro_rewrite_result(
                        led, _intro_id, _spliced,
                        "announcer_work_frame_rendered",
                        (_OTRLC_TAIL.PROTECTED_FACT_COMPONENT_FLAG,),
                    )
                    led.save()
                    meta["announcer_work_frame"] = _work_frame
        except Exception:
                # THE LAW: an audit may never fail an episode. A frame that
                # cannot be spliced leaves the composed opening standing --
                # that is the pre-fix behaviour, not a broken render.
                log.warning(
                    "[OTR_LedgerScriptWriter] work-frame splice failed; "
                    "keeping the composed announcer opening (LOUD).",
                    exc_info=True,
                )
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
            # Ledger-facing: the controlled slug (style-engine
            # consolidation, 2026-07-05), consistent with the canonical
            # story-style receipt stamped below.
            "style":      (contract.slug if contract else ""),
        }
        # scaffold_enabled is the EFFECTIVE flag: the bank gate is folded into
        # _style_grammar_on at the contract site and rides here through
        # ctx.style_grammar_on, so a bank-off run stamps enabled=False and
        # reads as "off by the bank's own definition", never as a style-build
        # failure.
        _stamp_story_style_receipt(
            meta, contract=contract, scaffold_enabled=_style_grammar_on)

        # Shared inline banks run one mandatory structural/safety handoff.
        # Producer-owned banks already performed their fixed tail and only need
        # the writer-model unload here. Neither path judges story length or
        # quality and neither can author a replacement story.
        if ctx.run_story_spine:
            try:
                from . import _otr_story_spine as _OTRSPINE
            except ImportError:  # pragma: no cover
                import _otr_story_spine as _OTRSPINE  # type: ignore
            _OTRSPINE.run_post_script_spine(led, meta)
        else:
            try:
                from . import _otr_writer_vram as _OTRVRAM
            except ImportError:  # pragma: no cover
                import _otr_writer_vram as _OTRVRAM  # type: ignore
            meta["writer_llm_unload"] = (
                _OTRVRAM.unload_writer_llm_after_script()
            )

        # The first structurally complete inline ledger is authoritative.
        # stamp_actual files the receipt at word_budget.actual_receipts[stage]
        # AND merges it onto word_budget top-level; the old top-level
        # meta.writer_word_delivery alias was a byte-equal duplicate and was
        # retired 2026-08-28 (V4 finding 12).
        _OTRWD.stamp_actual(
            led.data,
            stage="writer_final_rows",
        )


        # --- K.5.5/K.5.6 final-row reflections ------------------------
        # Both reflections must describe the exact delivered story, not the
        # pre-spine/pre-fit draft. They mutate meta only and use the technical
        # slot; failures retain the established non-raising sentinel contract.
        # LLM slot: technical
        # Read the durable meta stamp rather than `resolved`: run() writes
        # `meta["visual_style"]` at :3925 before dispatch, and other lanes
        # (scifi_news_pro, tests) build their own `resolved` dicts that need not
        # carry a `visual_style` key. Meta is the tail's declared field on
        # WriterTailContext; the resolver dict is not.
        _is_dynamic_style = (
            meta.get("visual_style") == _ROLLS.DYNAMIC_STYLE_ID
        )
        with slot_scheduler.helper_context("story_brief_reflection"):
            _brief_delta = run_story_brief_reflection(
                led,
                technical_generate_fn,
                technical_model_id=resolved["technical_model"],
                is_visual_storybased=_is_dynamic_style,
            )
        # POP THE MODEL BEFORE MERGING (PBUG-20260812-04). `visual_card` is the
        # only value in this delta that is not JSON -- it is a live
        # `VisualStyleCardModel`, added by `run_story_brief_reflection` at
        # `_otr_story_brief.py:643`. `meta.update()` put it straight into the
        # ledger, and although the serialized copy is written below as
        # `meta["visual_style_card"]`, the RAW MODEL stayed alongside it. The
        # very next `led.save()` then died:
        #
        #   [Ledger] save failed: Object of type VisualStyleCardModel is not
        #   JSON serializable
        #   RuntimeError: failed to save ledger after visual_style pack embedding
        #
        # `meta` is the ledger. Nothing that is not JSON may enter it -- the
        # model is a WORKING VALUE for the composer below, not ledger content.
        # Nothing reads `meta["visual_card"]`; the only reader took it from this
        # delta, and now takes it from this local instead.
        _card = _brief_delta.pop("visual_card", None)
        meta.update(_brief_delta)

        if _is_dynamic_style:
            if _card is not None:
                # Dynamic reflection succeeded -> compose pack from card.
                # Code/composer defects raise LOUD per Section 5 (never floored).
                _composed_pack = _otr_visual_styles.compose_pack_from_card(_card)
                _otr_visual_styles.validate_pack(_composed_pack, _ROLLS.DYNAMIC_STYLE_ID)

                _canonical_bytes = json.dumps(
                    _composed_pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                _sha256 = hashlib.sha256(_canonical_bytes).hexdigest()
                _attempts = _brief_delta.get("story_brief_attempts", 1)

                meta["embedded_visual_style_pack"] = _composed_pack
                meta["visual_style_card"] = _card.model_dump() if hasattr(_card, "model_dump") else dict(_card)
                meta["visual_style_receipt"] = {
                    "status": "dynamic",
                    "floor_style_id": None,
                    "failure_class": None,
                    "sha256": _sha256,
                    "schema_version": "v2",
                    "composer_version": 1,
                    "technical_model_id": resolved["technical_model"],
                    "attempts": _attempts,
                    "floor_roll": None,
                }
                meta["visual_style"] = _ROLLS.DYNAMIC_STYLE_ID
            else:
                # Dynamic reflection failed (MODEL or TRANSPORT) -> floor fallback
                _err = _brief_delta.get("story_brief_error") or ""
                if _err in (_STORY_BRIEF_REJECT_JSON_PARSE, _STORY_BRIEF_REJECT_SCHEMA):
                    _fail_class = "model"
                else:
                    _fail_class = "transport"

                if ctx.style_roll is not None:
                    _f_seed = ctx.style_roll.seed
                    _f_source = ctx.style_roll.seed_source
                else:
                    # Direct pick has no roll receipt: resolve a floor seed at
                    # failure time from process env (OTR_VISUAL_STYLE_SEED
                    # override or OS entropy), matching the rolled path.
                    _f_seed, _f_source = _ROLLS.resolve_seed(_ROLLS.STYLE_SEED_ENV)

                _f_order = _ROLLS.floor_style_ids()
                _f_selected = _ROLLS.draw(_f_order, _f_seed, random.Random)
                _f_roll_dict = {
                    "surface": "visual_style_floor",
                    "requested": _ROLLS.DYNAMIC_STYLE_ID,
                    "eligible_order": list(_f_order),
                    "selected": _f_selected,
                    "seed": _f_seed,
                    "seed_source": _f_source,
                }

                _raw_floor = json.loads(
                    (_otr_visual_styles._VISUAL_STYLES_ROOT / f"{_f_selected}.json").read_text(encoding="utf-8")
                )
                _floor_pack = dict(_raw_floor)
                _floor_pack["style_id"] = _ROLLS.DYNAMIC_STYLE_ID
                _floor_pack["label"] = f"Visual Story-Based (Floor: {_f_selected})"

                _canonical_bytes = json.dumps(
                    _floor_pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
                _sha256 = hashlib.sha256(_canonical_bytes).hexdigest()
                _attempts = _brief_delta.get("story_brief_attempts", 1)

                meta["embedded_visual_style_pack"] = _floor_pack
                meta["visual_style_receipt"] = {
                    "status": "floor",
                    "floor_style_id": _f_selected,
                    "failure_class": _fail_class,
                    "sha256": _sha256,
                    "schema_version": "v2",
                    "composer_version": 1,
                    "technical_model_id": resolved["technical_model"],
                    "attempts": _attempts,
                    "floor_roll": _f_roll_dict,
                }
                meta["visual_style"] = _ROLLS.DYNAMIC_STYLE_ID

            # Transaction: Require truthy led.save() BEFORE run_produced_story_summary
            _reflection_save = led.save()
            if not _reflection_save:
                raise RuntimeError("failed to save ledger after visual_style pack embedding")

        # LLM slot: technical
        with slot_scheduler.helper_context("produced_story_summary"):
            _story_delta = run_produced_story_summary(
                led,
                technical_generate_fn,
                technical_model_id=resolved["technical_model"],
            )
        meta.update(_story_delta)

        # The metadata reflections may have reloaded either writer slot.
        # Reclaim it unconditionally before TTS/image/video consumers.
        from . import _otr_writer_vram as _OTRVRAM_FINAL
        meta["writer_llm_unload"] = (
            _OTRVRAM_FINAL.unload_writer_llm_after_script()
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

        try:
            from ._otr_google_api import models as _gai_models
            meta.update(_gai_models.google_api_run_meta(
                resolved["creative_writing_model"],
                resolved["technical_model"],
            ))
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
        # CANONICAL ledger rows. Post-loop mutations (the news_close_brief
        # announcer override in I.5) write to led.data["lines"] and were not
        # mirrored into the writer's old in-loop token list, which is why the
        # ledger is the source of truth for the slot-0 STRING output. That
        # list is gone as of 2026-08-28; nothing read it.
        # Sprint 3E (2026-05-25): the former J.6 post-hoc title
        # substitution -- another such ledger-only mutation -- is gone
        # (late title binding means no provisional title in dialogue).
        # What follows is the one final producer boundary shared by every
        # source bank: after every writer-side text mutation, before the lane
        # finalizer's Phase-10 freeze.
        #
        # Independent source banks wave 6: the LEDGER CLEANUP PASS. Every
        # downstream consumer reads FIELDS, so this boundary owes them a
        # COMPLETE ledger -- especially for a client-authored bank, whose own
        # code may never touch the ledger and whose source material may have
        # been thin. The pass completes what the writer owns deterministically,
        # REPAIRS unsafe spoken language in place (content is never a
        # story-fail; the freeze gate's G9 stays the last-resort backstop),
        # fills the one required prose field, and raises only when a required
        # field has no owner and no value. It runs BEFORE the delivery stamp
        # below: sanitizing after that stamp would leave text_for_tts carrying
        # language the canonical row no longer has.
        #
        # The pass does NOT own the episode seed. That receipt has one owner
        # per lane family -- the seeded cast picker upstream for legacy lanes,
        # the content-owned block just below for lanes that never run it --
        # and a freshly minted seed is not derivable from the inputs, so a
        # pass that minted one for every lane would make this tail
        # irreproducible (tests/test_scifi_news_pro_tail_context.py pins that).
        # THE CLEAN STAGE, and it runs FIRST at this boundary. A MODEL reads
        # every spoken row and names anything in it that is not speech; a
        # MODEL then rewrites it. Every sealed line becomes TTS audio, so a
        # stage direction left inside one gets read aloud on air -- measured
        # at 11-40% of spoken rows on every bank (2026-08-14).
        #
        # COST, STATED HONESTLY: it is NOT detector-gated. One judge call per
        # voiced row, so a clean 16-row episode still spends 16 small calls
        # here. That is deliberate and it is the whole reason the pass works
        # -- gating on a pattern list is what let "The door closes behind
        # him" through, since no verb list contains every verb. A dirty row
        # then costs up to two repair calls, each re-read by the judge.
        # Bounded: after that budget the row SHIPS carrying a compose flag
        # and the log says so. It never stops a render.
        #
        # It uses the CREATIVE slot, not the technical one: it is rewriting
        # dialogue, so the tier that wrote the line rewrites it and the
        # repaired line still sounds like its neighbours (operator ruling
        # 2026-08-14).
        #
        # BEFORE run_ledger_cleanup, deliberately -- that pass re-stamps text
        # metrics, so a row rewritten here is measured after the rewrite
        # rather than before it.
        # BOTH PASSES BELOW ARE ONE AUTHORIZED WINDOW, AND IT IS A
        # TRANSACTION. They run AFTER a content-owned lane stamped its
        # acceptance receipt and BEFORE the freeze cascade re-validates it, so
        # every row they legitimately rewrite invalidates that receipt --
        # which is how `scifi_news` came to die `line receipt mismatch for
        # l004` 13.6 minutes in, with the script already finished. The window
        # is captured ONCE around both passes (one transition per pass is
        # impossible -- the second's pre-state is the first's output, so the
        # chain could never start at the acceptance) and reconciled once at
        # the end. If the reseal cannot be PROVED, the transaction restores
        # the accepted ledger, stamps a degradation receipt and the episode
        # ships without the repairs: Law 7, a render must not die. A lane with
        # no acceptance receipt has nothing to protect and gets no
        # transaction.
        from . import _otr_clean_transaction as _OTRTXN
        _clean_window = _OTRTXN.open_transaction(
            led, finalizer=tail_finalizer)

        from . import _otr_ledger_clean as _OTRLCLN
        with slot_scheduler.helper_context("ledger_clean"):
            _OTRLCLN.run_ledger_clean(
                led.data,
                slot_fn=creative_generate_fn,
                bank_id=str(meta.get("source_bank") or ""),
            )

        from . import _otr_ledger_cleanup as _OTRLCLEAN
        with slot_scheduler.helper_context("ledger_cleanup"):
            _OTRLCLEAN.run_ledger_cleanup(
                led.data,
                slot_fn=technical_generate_fn,
                bank_id=str(meta.get("source_bank") or ""),
            )

        if _clean_window is not None:
            _clean_window.reconcile()

        # PBUG-20260802-02 repair, right here and nowhere else: this is the
        # LAST point before the freeze cascade node runs, and (per the
        # comment below) the last thing that touches canonical `text` --
        # exactly why a repaired line's word receipt, stamped a few lines
        # down, already reflects the post-repair state instead of needing a
        # second restamp. Content-owned lanes (scifi_news_pro today) are
        # skipped: their own earlier gate (stamp_receipt/require_voice_
        # coverage) already runs before this tail is ever reached for them,
        # and a deliberately silent non-speaking cast entity (a Relay, not a
        # person) must never be forced to speak by a generic repair pass.
        from . import _otr_freeze_cascade as _OTRFC
        if _OTRFC.resolve_freeze_policy(meta).run_inline_safety_cleanup:
            from . import _otr_cast_coverage_repair as _OTRCCR
            meta["cast_coverage_repair"] = _OTRCCR.repair_zero_coverage_cast(
                led,
                creative_fn=creative_generate_fn,
                canon_header=_OTRC.render_episode_canon_header(canon) if canon else "",
                style_descriptor=str(contract.label if contract else "").strip(),
                source_bank_id=str(meta.get("source_bank") or ""),
                meta=meta,
                creative_repo_id=str(resolved.get("creative_writing_model") or "") or None,
                slot_scheduler=slot_scheduler,
            )
            if meta["cast_coverage_repair"].get("repaired"):
                _PL.refresh_ledger_text_metrics(led)

        # THE DELIVERED WORD RECEIPT IS RESTAMPED HERE, on every lane, because
        # the window above is the last thing that touches canonical `text`:
        # the clean stage rewrites rows and the cleanup re-stamps their
        # metrics, so the receipt taken at `writer_final_rows` describes a
        # draft that no longer exists. A SEPARATE stage name, not a re-use:
        # `stamp_actual` files each receipt under its stage in
        # `word_budget.actual_receipts`, so restamping under the old name
        # would overwrite the pre-clean record instead of adding the
        # post-clean one. Counts are telemetry, never a gate (THE LAW).
        _OTRWD.stamp_actual(
            led.data,
            stage="writer_final_rows_post_clean",
        )

        from ._otr_readiness import stamp_text_for_tts_delivery
        from ._otr_text_delivery import CONTENT_OWNED, delivery_mode_for_meta
        if delivery_mode_for_meta(meta) == CONTENT_OWNED:
            # Content-owned lane runners construct their own cast rows and
            # stamp their own voice presets, so they never run the writer's
            # seeded cast picker.  They still owe the downstream credits
            # contract a durable seed receipt, and this shared tail is the one
            # producer boundary upstream of CastLock, freeze, and CreditsRoll.
            #
            # Stamp meta.episode_seed -- the receipt otr_credits_roll accepts --
            # and NOT meta.cast_contract.cast_seed.  cast_seed is not a generic
            # episode seed: it is a claim that the writer's picker produced this
            # cast from that seed and can be REPLAYED from it.  CastLock replays
            # the picker whenever it sees cast_seed, and a lane-owned cast has no
            # num_characters_request to replay with, so claiming it detonates the
            # replay (BUG: "num_characters must be 1-6, got 0").
            if meta.get("episode_seed") is None:
                _episode_seed, _episode_seed_source = _resolve_cast_rng_seed()
                meta["episode_seed"] = int(_episode_seed)
                log.info(
                    "[OTR_LedgerScriptWriter] content-owned episode seed=%d "
                    "(%s) stamped before freeze",
                    _episode_seed, _episode_seed_source,
                )
            # Canonical ``text`` is sealed before this shared tail, so the
            # pronunciation-safe delivery string is stamped here -- after the
            # cleanup pass above, the last thing that may touch that text.
            stamp_text_for_tts_delivery(led)

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
        _stamp_final_slot_telemetry(
            meta=meta,
            resolved=resolved,
            slot_scheduler=slot_scheduler,
            pipeline_id=str(getattr(
                _source_bank_row, "default_story_pipeline", "") or ""),
            title_source=title_source,
        )
        # The lane finalizer is the true last mutation boundary: every shared
        # writer metadata stamp (including consistency_status) is complete
        # before Phase 10 seals the ledger and authorship receipt.
        if tail_finalizer is not None:
            tail_finalizer.before_save(ctx=ctx)
        script_text = _PL.assemble_script_text_from_ledger(led.data)
        script_json = json.dumps(led.data, indent=2, ensure_ascii=False)
        # kibitz r2-r4 provenance: bank-defaults-driven labels; the
        # original lane stamps source "Original (LLM)" (via seed_source
        # mapping below), headline = the FINAL title (codex r3 catch:
        # outline.title predates J.5 regen), and the HUD origin label.
        # Legacy lanes pass "" everywhere -> byte-identical payload.
        _bank_defaults = dict(_source_bank_row.defaults or {})
        news_json = _build_news_payload(
            outline, resolved["news_seed"], resolved["seed_source"],
            source_label=(
                "Original (LLM)"
                if resolved["seed_source"] == "original_llm" else ""
            ),
            origin_label=str(_bank_defaults.get("hud_origin_label") or ""),
            headline_override=(
                final_title
                if resolved["seed_source"] == "original_llm" else ""
            ),
        )

        actual_word_count = sum(
            int(r.get("word_count") or 0) for r in led.data["lines"]
        )
        est_minutes = max(
            1, round(actual_word_count / WORDS_PER_MINUTE_ESTIMATE, 1),
        )

        # --- M-pre. Resolved remote models (provenance) -----------------
        # WHY THIS IS HERE AND NOT ONLY IN THE CREDITS SHEET. A `~latest`
        # OpenRouter alias resolves to a CONCRETE model server-side, and the
        # entire safety argument for shipping aliases instead of pinned slugs
        # is "replay is unaffected because we record what actually served the
        # run". That record existed in exactly ONE place --
        # video_engine.py's "RESOLVED (OPENROUTER)" credits section -- which is
        # built on the VIDEO path. Proven live 2026-08-09: a story-only leg on
        # `~anthropic/claude-opus-latest` made real remote calls and finished
        # with NO provenance anywhere in its ledger, because no video node ever
        # ran. Any writer-only or scoring run silently lost the answer to "which
        # model wrote this".
        # Stamped just before the terminal save so it captures every call in the
        # window opened by reset_run_budget() earlier in this run, including the
        # reflection pass. Written ONLY when non-empty: a purely local run adds
        # no key, so today's ledgers stay byte-identical.
        try:
            from ._otr_openrouter_backend import (
                resolved_models_snapshot as _resolved_snapshot)
            _resolved_now = _resolved_snapshot() or {}
        except Exception:  # noqa: BLE001 -- provenance must never fail a render
            _resolved_now = {}
        if _resolved_now:
            meta["resolved_models"] = _resolved_now
            log.info(
                "[OTR_LedgerScriptWriter] resolved remote model(s): %s",
                ", ".join(
                    "%s -> %s (%d call(s))" % (
                        slug,
                        (_resolved_now[slug] or {}).get("resolved")
                        or "(unreported)",
                        int((_resolved_now[slug] or {}).get("calls") or 0))
                    for slug in sorted(_resolved_now)))

        # --- M. Save ledger -------------------------------------------
        # Spec r3/final.md section 6: terminal saves MUST be truthy-required.
        # `Ledger.save()` returns None rather than raising on failure
        # (production_ledger.py:1423-1492); accepting None silently would
        # leave downstream consumers reading a stale ledger from disk while
        # the writer logs "DONE".
        saved_path = led.save()
        if not saved_path:
            raise RuntimeError(
                "terminal ledger save returned no path -- ledger not persisted"
            )
        if tail_finalizer is not None:
            tail_finalizer.after_save(
                saved_path=str(saved_path), ledger_data=led.data,
            )
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
        return (
            script_text,
            script_json,
            news_json,
            est_minutes,
            resolved["technical_model"],
        )
