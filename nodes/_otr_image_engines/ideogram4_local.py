"""Ideogram 4 (local weights) image adapter -- the typography-first still engine.

An OPT-IN peer (``default_roles=()``): ``z_image_turbo`` remains the shipped
default and is not displaced. Ideogram 4's competence is TYPOGRAPHY, so the lane
it earns is ``still_word`` -- the card whose whole job is to put the script's own
words on screen legibly.

**Honest value statement, measured 2026-08-21/22.** `z_image_turbo` already
renders word cards well: correct spelling, frame-filling type, ~12 s per card
against this engine's ~95 s, from 4.5 GB of weights against 11 GB. Ideogram
brings richer display typography and a flatter, more "set" look -- and on long
card lines it is NOT uniformly reliable (measured line-break and letter faults).
That is why this ships opt-in and why the operator's eye, not a pixel metric, is
the acceptance gate.

WHY THE PROMPT IS REBUILT RATHER THAN PASSED THROUGH
----------------------------------------------------
This checkpoint was trained on structured JSON captions. Handing it OTR's prose
made it REFUSE 6 of 6 real production card lines -- returning a flat placeholder
card at the right dimensions with host status SUCCESS. Rebuilding the same lines
in the vendor's three-key caption schema took that to 0 of 6. So the adapter
always emits the schema; no request shape reaches the model as raw prose.

This is also why this engine is the ONLY one that owes a translation layer. The
other six image adapters take OTR's prose verbatim, and that is correct for
them; nothing here argues they should change.

THE LENS ROUTES ON METADATA, NOT ON PROSE TEXT (2026-08-26)
-----------------------------------------------------------
Until this date the caption transform chose its route by searching the prose for
two literal anchors. Both are minted by ONE composer entry point,
``otr_meta_brief_image_prompt.compose_still_word_prompt``, which serves ONE lane
-- ``still_word``. Every other still lane (``still_flat``, ``still_pan``,
``still_motion``) hands this adapter a prose SCENE that matches neither anchor,
so all of them landed in the same unrouted fallthrough -- the one whose own
docstring conceded that raw prose is what this model refuses. The lens was
routing blind on three lanes out of four.

It did not have to be. The dispatcher has been putting ``kind``, ``role``,
``beat_id`` and ``char_id`` on every image request since the still spine was
built (``otr_image_gen_dispatcher``); this adapter simply never read them. It
reads ``kind`` and ``role`` now, and an object kind the table does not know
FAILS LOUD rather than sliding into the scene route -- a silent misroute is the
exact defect being repaired here, and leaving the door open for the next new
kind would be choosing the same bug a second time.

THE PROHIBITION TRAP (Bug Bible 12.126)
---------------------------------------
The official topology wires the guider's negative from ``ConditioningZeroOut`` of
the POSITIVE -- a zero tensor. **There is no negative channel at all, so every
token is positive conditioning and a prohibition can only ADD its own nouns.**
OTR's composer appends ``only the quoted words, no other text, no logos, no
captions`` to every word card, and all nine style packs append ``no lettering``
to the music card. A card came back with ``NO MISCOS`` painted across it -- a
mangled render of our own "no logos". The adapter therefore STRIPS prohibition
clauses and states intent positively, which is the only form this topology can
act on. The shared composer is untouched: every other engine has a real negative
channel and still needs its guard (IMAGE_GEN_PREFLIGHT IG5.1 -- the ENGINE owns
its own shape adaptation).

THE TEXT BOX IS LOAD-BEARING, AND ITS SIZE IS A TRADE
-----------------------------------------------------
Measured 8 cells at a seed known to misbehave, two lines x four framings: a
two-thirds-height text bbox produced invented "page furniture" (fake catalog
numbers, a fake copyright, a page number) on 4 of 4 designed-artifact cards,
while a moderate bbox produced none on 4 of 4. Removing the bbox entirely is
WORSE than either -- without that anchor the type collapses to a small line and
the spelling breaks (``Pompei fits.`` for ``Pompeii.``). So the moderate box is
pinned: it is the only setting measured clean on BOTH axes, spelling and
furniture.

Cold-import clean (V-12): module scope imports the dep-free registry, the role
vocabulary and stdlib only. torch / comfy / the weights are reached lazily inside
``render_image`` via ``wrapper_bridge``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re

from .registry import register, EngineUnusable, EngineUsabilityReason
from .._otr_shared.role_compat import ROLES

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Weights. Env overrides exist for a mirror/quant swap; each resolved basename
# feeds the effective engine version so a swap cannot serve stale cached stills.
# --------------------------------------------------------------------------- #
COND_UNET_ENV = "OTR_IDEOGRAM4_COND_UNET"
UNCOND_UNET_ENV = "OTR_IDEOGRAM4_UNCOND_UNET"
CLIP_ENV = "OTR_IDEOGRAM4_CLIP"
VAE_ENV = "OTR_IDEOGRAM4_VAE"

#: PRECISION LADDERS, NOT ONE HARDCODED BASENAME (2026-09-03).
#:
#: This engine used to demand exactly `ideogram4_nvfp4_mixed.safetensors`, and
#: nvfp4 is a Blackwell format -- so an AMD, Mac or 3060 box that had installed
#: the perfectly good fp8 or int8 build was still refused with
#: "missing: ideogram4_nvfp4_mix...", a dead end unless the operator happened to
#: know the four env overrides existed. The engine was never Blackwell-only;
#: only its default was, and the refusal read as though the lane were.
#:
#: `Comfy-Org/Ideogram-4` is UNGATED. It publishes the two DIFFUSION slots in
#: three precisions and the text encoder in two; the VAE is single. So the
#: ladders below are 3 / 3 / 2 / 1, NOT "three precisions across four slots" --
#: an earlier draft of this comment said that and it was never true.
#:
#: THE CONSEQUENCE IS A REAL MIXED-PRECISION REQUIREMENT: there is no int8 text
#: encoder, so a box holding only the int8 diffusion pair must still fetch an
#: nvfp4 or fp8 `qwen3vl_8b`. That is legal -- the encoder and the UNET are
#: loaded separately -- but it is not obvious, and the refusal message does not
#: spell it out. Worth stating in the fetch docs if int8 ever becomes a
#: recommended row rather than a fallback.
#:
#: The published layout and naming:
#:
#:   ideogram4_nvfp4_mixed        5.49 GB   Blackwell (sm_120), smallest
#:   ideogram4_fp8_scaled         9.28 GB   universal
#:   ideogram4_int8_convrot       9.58 GB   universal (same family as the
#:                                          proven minimax_h3 int8_convrot)
#:
#: So the ladder is ordered SMALLEST-FIRST rather than best-first: on a
#: Blackwell card nvfp4 is both the fastest and the lightest, and on anything
#: else it simply will not be installed and the walk falls through. Whatever
#: resolves is what the effective engine version hashes (line ~590), so a
#: precision swap busts the still cache on its own -- a machine that changes
#: precision cannot serve a still rendered by the other one.
#:
#: An env override still wins absolutely and is used verbatim; the ladder is
#: only the default walk.
_COND_UNET_CANDIDATES = (
    "ideogram4_nvfp4_mixed.safetensors",
    "ideogram4_fp8_scaled.safetensors",
    "ideogram4_int8_convrot.safetensors",
)
_UNCOND_UNET_CANDIDATES = (
    "ideogram4_unconditional_nvfp4_mixed.safetensors",
    "ideogram4_unconditional_fp8_scaled.safetensors",
    "ideogram4_unconditional_int8_convrot.safetensors",
)
_CLIP_CANDIDATES = (
    "qwen3vl_8b_nvfp4.safetensors",
    "qwen3vl_8b_fp8_scaled.safetensors",
)
_VAE_CANDIDATES = ("flux2-vae.safetensors",)

#: (env var, candidate basenames, folder_paths category) per required artifact.
#:
#: (Four `_DEFAULT_*` aliases used to sit here, justified as "the name the error
#: message leads with". That was false -- the refusal text is literal and the
#: params path resolves through `resolve_all_artifacts()`, so nothing read them.
#: Removed 2026-09-04, the same session that added them. `candidates[0]` is the
#: lead name, and it needs no second spelling.)
_ARTIFACTS = (
    (COND_UNET_ENV, _COND_UNET_CANDIDATES, "diffusion_models"),
    (UNCOND_UNET_ENV, _UNCOND_UNET_CANDIDATES, "diffusion_models"),
    (CLIP_ENV, _CLIP_CANDIDATES, "text_encoders"),
    (VAE_ENV, _VAE_CANDIDATES, "vae"),
)

# --------------------------------------------------------------------------- #
# Recipe. Read off the official template, with one deliberate deviation.
# --------------------------------------------------------------------------- #
STEPS = 20
STD = 1.75
CFG = 7.0
CFG_OVERRIDE, CFG_OVERRIDE_START, CFG_OVERRIDE_END = 3.0, 0.7, 1.0
SAMPLER = "euler"

#: DELIBERATE DEVIATION, RECORDED RATHER THAN HIDDEN. The template's preset table
#: gives Default = {steps 20, mu 0.0, std 1.75} and wires mu from that table, so
#: the scheduler node's 0.5 widget default is dead and 0.5 is in fact the TURBO
#: value. Every card in this campaign that came back with perfect spelling was
#: rendered at 0.5, and a hard-won recipe is not changed on theory -- so 0.5
#: stays, named as a deviation, until a matched A/B says otherwise.
MU = 0.5

#: Canvas must be /16-legal with a 256 floor -- the template does this in a
#: ComfyMathExpression; we do the same arithmetic in Python.
_CANVAS_MULTIPLE = 16
_CANVAS_FLOOR = 256

#: The text box, normalized 0-1000 as [y1, x1, y2, x2]. See the module docstring:
#: this is the measured sweet spot between "type too small / spelling breaks" and
#: "type so large the model adds page furniture".
TEXT_BBOX = [200, 60, 700, 940]

#: The music card's OBJECT bbox, and note the ORDER IS DIFFERENT from
#: ``TEXT_BBOX`` above: an ``obj`` element is ``[x, y, w, h]`` while a ``text``
#: element is ``[y1, x1, y2, x2]`` (template caption contract, read 2026-08-21).
#: Getting that backwards misplaces the subject silently instead of failing.
#:
#: Wide, centred, and generous on purpose. This exists to give the model ONE
#: concrete thing to place on a card that may carry no words -- not to compose
#: the shot. A tight box would turn a mood still into a product photograph.
MUSIC_OBJECT_BBOX = [180, 240, 640, 520]

#: Ratios the caption schema was trained on. Exact reduction is wrong here --
#: 1472x832 reduces to 23:13, which is not one of them.
_STANDARD_RATIOS = ((1, 1), (16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3),
                    (4, 5), (5, 4), (21, 9), (3, 1), (1, 3))

#: Prohibition clauses OTR splices in. Composer-owned, verified live against all
#: nine style packs. Stripped from the ATMOSPHERE ONLY -- never from the card
#: text, which is script.
_PROHIBITION_RE = re.compile(
    r",?\s*(?:only the quoted words|no other text|no logos|no captions"
    r"|no lettering|no on-screen text|no subtitle line|no studio name"
    r"|no genre label)", re.I)

#: The composer's own anchors. `_fold_inner_dquotes` guarantees the template's
#: pair is the ONLY double-quote pair in the card (pinned by
#: tests/test_still_word.py), which is what makes this extraction safe.
_WORD_RE = re.compile(r'a title card displaying the words "([^"]*)"')
_TITLE_RE = re.compile(r'an abstract picture evoking "([^"]*)"')

#: The four caption routes. The first two are decided by the composer's anchors
#: ABOVE -- evidence carried in the prose itself -- and the last two by the
#: request's object ``kind``, which the dispatcher has always supplied.
ROUTE_WORD = "word"
ROUTE_TITLE = "title"
ROUTE_PORTRAIT = "portrait"
ROUTE_SCENE = "scene"

#: THE LIVE OBJECT-KIND VOCABULARY, mapped EXPLICITLY -- every kind names its
#: route, so adding one upstream becomes a loud failure here instead of a silent
#: demotion into the scene caption.
#:
#: Six of the seven are the closed enum ``VALID_KINDS`` in
#: ``nodes/_otr_shared/still_plan_helpers.py``; the seventh is
#: ``_otr_video_engines.coverage_plan.JUMP_STILL_KIND``, which is deliberately
#: NOT a ``scene_*`` token so segment stills stay invisible to the beat-indexed
#: consumers. The literals are REPEATED rather than imported because this module
#: is cold-import clean (V-12) and reaching into the video package would run its
#: self-registration at image-engine import time.
#: ``test_the_kind_table_covers_the_live_vocabulary`` pays that import cost in
#: the suite instead, and fails the moment the two definitions drift apart.
OBJECT_KIND_ROUTES = {
    "portrait": ROUTE_PORTRAIT,
    "scene_open": ROUTE_SCENE,
    "scene_beat": ROUTE_SCENE,
    "scene_character": ROUTE_SCENE,
    "mesh_fodder": ROUTE_SCENE,
    "scene_background_plate": ROUTE_SCENE,
    "jump_segment": ROUTE_SCENE,
}

#: AN ABSENT KIND IS NOT AN UNKNOWN KIND. ``build_caption(prose, w, h)`` remains
#: a valid positional call -- the refusal repro script and the caption tests use
#: it that way, and the dispatcher itself defaults a kind-less object to
#: ``portrait`` long before the request is built. A caller that hands over no
#: metadata gets the scene route, which is precisely where it already was. Only
#: a NON-EMPTY kind outside the table above is a caller bug.
KIND_UNSPECIFIED = ""

#: Refusal card statistics, DERIVED from captured artifacts (never guessed):
#: refusals measured min 68-87 / std 9.9-10.7; real renders min 0-1 / std 27-41.
REFUSAL_MIN_FLOOR = 50.0
REFUSAL_MAX_STD = 15.0


class Ideogram4RefusalError(RuntimeError):
    """The model returned a safety-refusal placeholder instead of a card.

    Deliberately a LOCAL error rather than the dispatcher's ``ImageRenderError``.
    ``otr_image_gen_dispatcher`` imports the image-engine package at module
    scope, so importing it back would be a cycle -- and because this adapter's
    own import in ``__init__`` is GUARDED, that cycle would be swallowed and the
    engine would silently fail to register. The dispatcher wraps any adapter
    exception into a named ``ImageRenderError`` already.
    """

    #: A MODEL VERDICT IS NOT AN ENGINE FAULT, and this flag is how the
    #: dispatcher tells the two apart WITHOUT importing this class (see above --
    #: the import would be a cycle, swallowed by a guarded import, and the
    #: engine would silently fail to register). The attribute travels on the
    #: exception instance instead.
    #:
    #: An OOM, a missing wrapper node or a decode failure means the engine is
    #: BROKEN and must hard-fail the episode (NO FALLBACKS, operator
    #: 2026-06-18). A safety refusal means the engine worked perfectly and the
    #: model declined this one card: it returned valid decoded pixels at the
    #: exact requested dimensions with the graph completing. Killing eight
    #: finished beats over one declined card is the asymmetry the operator
    #: named on 2026-08-22 -- *"why is refusing card killing the episode, i
    #: dont think thats good feature"*, *"its an experimental stack its not
    #: perfect"*, and *"i didnt want any fail on this or that"*.
    #:
    #: DECLARED, so a future engine opts in by SAYING so rather than by being
    #: recognised from its class name.
    is_model_refusal = True


def _snap(value: int) -> int:
    """/16-legal with a 256 floor -- the template's own canvas arithmetic."""
    value = int(value or 0)
    snapped = ((value + _CANVAS_MULTIPLE - 1) // _CANVAS_MULTIPLE) * _CANVAS_MULTIPLE
    return max(snapped, _CANVAS_FLOOR)


def canonical_aspect(width: int, height: int) -> str:
    """Nearest STANDARD ratio string. Sending raw pixels was a measured defect."""
    if not width or not height:
        return "16:9"
    target = float(width) / float(height)
    w, h = min(_STANDARD_RATIOS, key=lambda r: abs((r[0] / r[1]) - target))
    return f"{w}:{h}"


def _resolve_artifact(env_var: str, candidates, category: str):
    """``(basename, verified)`` for one artifact, walking a precision ladder.

    ONE resolver shared by ``assert_usable`` and the params path, so the
    usability gate and the render can never disagree -- the 2026-07-05 landmine
    where a gate required an env var while render fell back to an absent default
    and died deep in a FileNotFoundError instead of greying out early.

    An ENV OVERRIDE WINS ABSOLUTELY and is used verbatim: it is the operator
    naming a mirror or a quant we have never heard of, and second-guessing it
    against a ladder would defeat the reason it exists. Only the default walk
    consults ``candidates``, taking the first one actually INSTALLED so a box
    that has fp8 but not nvfp4 resolves instead of being refused.
    """
    override = os.path.basename((os.environ.get(env_var, "") or "").strip())
    names = [override] if override else [os.path.basename(c) for c in candidates]
    try:
        import folder_paths  # ComfyUI runtime; absent in the CPU suite
        installed = {os.path.basename(n)
                     for n in (folder_paths.get_filename_list(category) or [])}
        for name in names:
            if name in installed or folder_paths.get_full_path(category, name):
                return name, True
        # Nothing on the shelf: report the FIRST candidate, which is what the
        # refusal should name and what an operator most likely wants to fetch.
        return names[0], False
    except Exception:  # noqa: BLE001 -- no folder_paths -> nothing discoverable
        return names[0], False


def resolve_all_artifacts():
    """``[(basename, verified, category), ...]`` in the fixed artifact order."""
    return [(*_resolve_artifact(env, candidates, cat), cat)
            for env, candidates, cat in _ARTIFACTS]


def _tidy(text: str) -> str:
    """Close the punctuation seam a removed clause leaves behind.

    Cutting the card clause out of the middle of a comma-joined prose string
    strands its neighbours: the style prefix ends in a period and the next
    fragment starts with a comma, giving `anime style. , huge high-contrast`.
    Every token is positive conditioning here, so stray punctuation is not
    merely untidy -- it is noise the encoder reads.

    BOTH CAPTURING RULES BELOW USED TO REPLACE WITH A BARE U+0001 INSTEAD OF
    THE ``\\1`` BACKREFERENCE (found 2026-08-26). Every match therefore DELETED
    the captured punctuation and injected a C0 control character into the
    prompt -- `"a warm revelation., sepia"` became
    `"a warm revelation\\x01 sepia"`. It reached the model on EVERY route, and
    the prompt refused six times in the 2026-08-26 sweep ends in exactly that
    shape. Pinned by
    ``test_tidy_preserves_punctuation_instead_of_injecting_a_control_character``.

    The first rule also EATS the following whitespace on purpose and re-emits a
    single space, because `"style. ,huge"` would otherwise close up to
    `"style.huge"` and weld two words together.
    """
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"([.;:])\s*,\s*", r"\1 ", text)  # "style. ,huge" -> "style. huge"
    text = re.sub(r",\s*,+", ",", text)            # ", ,"       -> ","
    text = re.sub(r"\s+([,.;:])", r"\1", text)     # " ,"        -> ","
    # Strip dangling separators at BOTH ends: an empty atmosphere leaves the
    # composed sentence opening with "; " or closing with ",".
    return text.strip().strip(",;:").strip()


def _route_for_kind(kind: str, role: str) -> str:
    """The object route for one request's ``kind``. Fails loud on an unknown.

    ``role`` is carried here only to NAME THE LANE in that failure. It is
    deliberately not a routing key: the three live roles (``announcer_visual``,
    ``music_visual``, ``character_video``) never disagree with ``kind`` about
    which route applies, and standing up a second, weaker authority over the
    same decision is how two routers drift apart. What an operator actually
    needs when this raises is which lane minted the row, and that is what
    ``role`` supplies.
    """
    token = str(kind or "")
    if token == KIND_UNSPECIFIED:
        return ROUTE_SCENE
    try:
        return OBJECT_KIND_ROUTES[token]
    except KeyError:
        raise ValueError(
            "ideogram4_local: unknown still kind %r (role %r). Every object "
            "kind must name its caption route in OBJECT_KIND_ROUTES; the "
            "mapped kinds are %s. NO FALLBACK -- routing a kind nobody mapped "
            "into the scene caption is the silent misroute this table exists "
            "to replace." % (token, str(role or ""),
                             ", ".join(sorted(OBJECT_KIND_ROUTES)))
        ) from None


def caption_route(prose: str, *, kind: str = "", role: str = "") -> str:
    """Which of the four routes this request takes. Pure.

    THE SINGLE AUTHORITY: :func:`build_caption` calls exactly this, so a test
    that pins a lane's route is pinning the real decision rather than a parallel
    copy of the priority order.

    Priority is WORD -> TITLE -> PORTRAIT -> SCENE. The two prose anchors outrank
    ``kind`` because they are the stronger evidence: a word card's ledger row
    wears the shared cheap-family ``scene_character`` kind -- it inherits face
    framing while actually minting typography from the spoken line -- so routing
    on kind first would send every word card to the scene caption and ship cards
    with no words on them.
    """
    # Resolved FIRST, and its result often discarded, ON PURPOSE: an unmapped
    # kind then fails on every lane rather than only on the two that consult it.
    # A hole in the table is a hole whether or not this particular prose happened
    # to carry a composer anchor, and a hole that surfaces on only some lanes is
    # the kind that ships.
    object_route = _route_for_kind(kind, role)
    if _WORD_RE.search(prose or ""):
        return ROUTE_WORD
    if _TITLE_RE.search(prose or ""):
        return ROUTE_TITLE
    return object_route


def _wrapped_caption(aspect: str, description: str) -> dict:
    """The minimal caption: the vendor's keys, and nothing invented.

    ``background`` IS EMPTY ON PURPOSE, and that is the second half of the
    2026-08-26 fix. The old fallthrough put the IDENTICAL string into
    ``high_level_description`` and ``background`` -- the input pasted into two
    fields, which is not a deconstruction and merely told the model the same
    thing twice. The honest alternative is not a richer guess: the composer emits
    a comma-joined five-layer string behind a style prefix, which is a
    convention, not a grammar, so any attempt to re-extract subject / setting /
    elements from it mis-fires. Leaving the slot empty invents no setting.
    """
    return {
        "aspect_ratio": aspect,
        "high_level_description": description,
        "compositional_deconstruction": {"background": "", "elements": []},
    }


def build_caption(prose: str, width: int = 0, height: int = 0, *,
                  kind: str = "", role: str = "") -> dict:
    """OTR's composed prose -> the vendor's three-key caption schema.

    ORDER IS LOAD-BEARING: the quoted card text is extracted and removed BEFORE
    any scrubbing, so the scrub can only ever touch composer-owned atmosphere.
    Scrubbing first would corrupt a spoken line that legitimately contains a
    guard phrase -- a card reading "No captions, no excuses." is SCRIPT, and
    script is never edited.

    Four routes (see :func:`caption_route` for the priority and why), and none of
    them passes prose through raw:
      WORD     -> the quoted line becomes the single text element;
      TITLE    -> ``elements: []``, because the music card is contractually
                  WORDLESS and its own guard would otherwise ask for words;
      PORTRAIT -> the request's kind says a face, wrapped in the schema;
      SCENE    -> every other mapped kind, and a caller that supplied none.

    PORTRAIT AND SCENE EMIT THE SAME MINIMAL SHAPE, DELIBERATELY. The vendor
    schema on disk carries exactly ``background`` and ``elements`` under the
    deconstruction, the only element type ever observed in it is ``text``, and
    there is no portrait-specific key to fill. Writing one would be inventing
    detail this adapter was never handed. What splitting the route buys is that
    the choice is now RECORDED and auditable, and that a kind nobody mapped can
    no longer reach either route by accident.
    """
    aspect = canonical_aspect(width, height)
    route = caption_route(prose, kind=kind, role=role)

    def _atmosphere(match) -> str:
        rest = prose[:match.start()] + prose[match.end():]
        rest = _PROHIBITION_RE.sub("", rest)
        return _tidy(rest)

    # The anchor is re-searched rather than threaded out of `caption_route`,
    # because only the two anchor routes need the capture group and the route
    # decision must live in exactly one place. Re-running one short regex on one
    # prompt per still is not a cost worth a second copy of the priority order.
    if route == ROUTE_WORD:
        word = _WORD_RE.search(prose)
        card = word.group(1)
        atmosphere = _atmosphere(word)
        return {
            "aspect_ratio": aspect,
            "high_level_description": _tidy(
                f"A title card showing one block of lettering reading "
                f"'{card}', set as large as the frame allows, above an "
                f"unbroken bare strip of ground, {atmosphere}"),
            "compositional_deconstruction": {
                "background": _tidy(
                    f"{atmosphere}; the lower part of the frame is "
                    f"the same continuous featureless ground"),
                "elements": [{
                    "type": "text",
                    "bbox": list(TEXT_BBOX),
                    "text": card,
                    "desc": ("the single block of lettering, spelled exactly, "
                             "set as large as will fit, centred"),
                }],
            },
        }
    if route == ROUTE_TITLE:
        title = _TITLE_RE.search(prose)
        evoked = title.group(1)
        atmosphere = _atmosphere(title)
        return {
            "aspect_ratio": aspect,
            # The captured title is the SUBJECT. Dropping it would render an
            # unrelated abstract image.
            "high_level_description": _tidy(
                f"A vintage tabletop radio receiver, its dial lit, standing in "
                f"a composition evoking '{evoked}', {atmosphere}"),
            "compositional_deconstruction": {
                "background": atmosphere,
                # ONE OBJECT ELEMENT, AND NO TEXT ELEMENT (2026-08-26).
                #
                # THIS ROUTE IS THE MUSIC CARD, and only the music card: the
                # anchor phrase `_TITLE_RE` matches is minted at exactly ONE
                # site, `otr_meta_brief_image_prompt.py:1240`, inside
                # `if _role == _STILL_WORD_MUSIC_ROLE`. So no role check is
                # needed here and no route has to be inserted ahead of this one.
                #
                # WHY IT CHANGED. Measured on a live leg, 2026-08-26: this
                # engine rendered 6 of 8 stills in one episode -- every card
                # carrying words -- and REFUSED both music bookends, on
                # different seeds, at min 78/80 std 10.5 against a real card's
                # min~0 std 27-41. The captions were verified free of control
                # characters, face language, prohibition clauses and duplicated
                # fields, so none of the defects fixed earlier that day
                # explained it. The one structural difference left was
                # `elements: []`.
                #
                # A display-typography model handed an abstract with nothing to
                # anchor on is being asked for the one thing it is not for. The
                # music card is contractually WORDLESS (operator 2026-07-04), so
                # it cannot be given lettering -- but the vendor schema's OTHER
                # element type costs no words at all. One `obj` gives the model
                # a concrete subject to place while the mood still comes from
                # `background`.
                #
                # BBOX ORDER IS PER-TYPE AND GETTING IT BACKWARDS FAILS SILENTLY
                # (r2 2026-08-21): an `obj` bbox is [x, y, w, h]; a `text` bbox
                # is [y1, x1, y2, x2]. This is an obj, so x first. It is left
                # deliberately WIDE and centred rather than tight -- the point is
                # to anchor the composition, not to lock it, and the mood image
                # should still be free to be a mood image.
                #
                # The operator's ruling that made this necessary: ideogram must
                # stay selectable for the music role and must PRODUCE AN OUTPUT.
                # *"It could be a horrible card but it needs to produce an
                # output."* A refusal is the one result that is not allowed.
                "elements": [{
                    "type": "obj",
                    "bbox": list(MUSIC_OBJECT_BBOX),
                    "desc": ("a period tabletop radio receiver, warm dial glow, "
                             "the single concrete subject of the frame"),
                }]},
        }
    # PORTRAIT and SCENE. Both wrap the scrubbed prose in the vendor shape; the
    # difference between them is the recorded routing decision above, not an
    # invented caption. See :func:`build_caption`'s docstring.
    return _wrapped_caption(aspect, _tidy(_PROHIBITION_RE.sub("", prose)))


def caption_json(prose: str, width: int = 0, height: int = 0, *,
                 kind: str = "", role: str = "") -> str:
    """The caption as the SINGLE-LINE MINIFIED JSON the schema specifies.

    The metadata is threaded straight through: changing only
    :func:`build_caption`'s signature would leave the lens dead, since this is
    what ``_params`` actually calls.
    """
    return json.dumps(build_caption(prose, width, height, kind=kind, role=role),
                      ensure_ascii=False, separators=(",", ":"))


def classify_refusal(frame) -> tuple:
    """``(is_refusal, minimum, std)`` for a decoded ``(H, W, 3)`` uint8 frame.

    A refusal is not an exception and not a black frame -- it is a flat pale
    placeholder card at the exact requested dimensions, delivered with host
    status SUCCESS, so every generic guard passes it (Bible 12.125).
    """
    minimum = float(frame.min())
    std = float(frame.std())
    return (minimum > REFUSAL_MIN_FLOOR and std < REFUSAL_MAX_STD), minimum, std


def _role_of(profile) -> str:
    if isinstance(profile, dict):
        return str(profile.get("role") or "")
    return str(getattr(profile, "role", "") or "")


@register
class Ideogram4LocalEngine:
    """Registered as ``ideogram4_local``. Local Ideogram 4, dual-expert."""

    name = "ideogram4_local"
    roles = ROLES
    #: OPT-IN. z_image_turbo stays the shipped default; no model is "primary".
    default_roles = ()
    #: Non-commercial model agreement -- docs/IDEOGRAM4_LICENSE_ATTESTATION.md.
    #: The code ships; the weights never do.
    commercial_clean = False
    requires_flag = None            # vestigial: the registry IS the menu
    required_inputs = ("text_prompt",)
    #: This topology has no reference-image path at all.
    accepts_reference_image = False
    #: `native` and `node_key` are deliberately ABSENT: this is a LOCAL engine
    #: and must take the GPU lease like the other local engines.

    #: Base version. `engine_version` is a PROPERTY below so that repointing any
    #: weight env var changes the still cache key -- otherwise a quant swap would
    #: silently serve stale cached stills.
    #:
    #: BUMPED "1" -> "2" ON 2026-08-26 for the metadata routing above, and the
    #: bump is REQUIRED rather than bookkeeping. The dispatcher's still cache key
    #: is ``(role, object_id, prompt_hash, seed, engine_id, engine_version)``,
    #: and ``prompt_hash`` is computed from OTR's PROSE before this adapter ever
    #: runs -- so a change that alters only the caption this adapter builds is
    #: invisible to every other term. Without the bump, every still minted under
    #: the old blind fallthrough would be served from cache forever and the fix
    #: would never reach a rendered frame.
    base_engine_version = "2"

    #: Terminal graph node: its IMAGE output is the still.
    _TERMINAL = "decode"

    @property
    def engine_version(self) -> str:
        """Base version + the resolved artifact identities.

        The dispatcher's still cache key is
        ``(role, object_id, prompt_hash, seed, engine_id, engine_version)``.
        Model overrides change the rendered pixels without touching any other
        term, so the resolved basenames belong in the version or a swap serves
        yesterday's images forever.
        """
        names = "|".join(name for name, _verified, _cat in resolve_all_artifacts())
        digest = hashlib.sha256(names.encode("utf-8")).hexdigest()
        return f"{self.base_engine_version}.{digest[:8]}"

    # ---- residency (classes resolve lazily; loader nodes own the weights) ----
    def load(self) -> None:  # pragma: no cover -- resolved lazily in render
        return None

    def unload(self) -> None:  # pragma: no cover
        self._classes = None

    def assert_usable(self, host_caps, profile, request_template=None):  # noqa: ARG002
        """FAIL CLOSED until every artifact is installed (BUG-046): greyed out,
        never a stub. Shares :func:`_resolve_artifact` with the render path."""
        missing = [f"{name} ({cat})"
                   for name, verified, cat in resolve_all_artifacts()
                   if not verified]
        if missing:
            # NAME THE ALTERNATIVES. The refusal used to quote only the nvfp4
            # basename, which reads as "this lane needs a Blackwell card" -- and
            # that is false. Every precision below lives in the same UNGATED
            # Comfy-Org/Ideogram-4 repo and any of them satisfies this engine.
            raise EngineUnusable(
                self.name, _role_of(profile),
                EngineUsabilityReason.MISSING_MODEL,
                f"ideogram4_local requires all four artifacts; missing: "
                f"{', '.join(missing)}. NOT Blackwell-only -- fetch ANY one "
                f"precision from the ungated Comfy-Org/Ideogram-4: "
                f"nvfp4_mixed (5.5 GB, Blackwell), fp8_scaled (9.3 GB, any "
                f"GPU) or int8_convrot (9.6 GB, any GPU), plus the matching "
                f"qwen3vl_8b text encoder and flux2-vae. Or point "
                f"{COND_UNET_ENV} / {UNCOND_UNET_ENV} / {CLIP_ENV} / {VAE_ENV} "
                f"at installed files.",
                kind="image",
            )
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # noqa: ARG002
        return {"engine_id": self.name}

    def teardown(self, prepared) -> None:  # noqa: ARG002
        return None

    # ---- params / graph (pure; CPU-testable) --------------------------------
    def _params(self, request) -> dict:
        """Pure: resolve every render parameter from the request + env."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        # width/height may be None; w/h carry the raw values.
        width = _snap(get("width") or get("w") or 1472)
        height = _snap(get("height") or get("h") or 832)
        # ONE resolution pass: each call re-probes all four artifacts through
        # folder_paths, so calling it per-name meant 16 filesystem probes to read
        # four basenames.
        cond, uncond, clip, vae = (a[0] for a in resolve_all_artifacts())
        return {
            "cond_unet": cond, "uncond_unet": uncond,
            "clip_name": clip, "vae_name": vae,
            # kind/role reach the caption transform HERE or nowhere: the
            # dispatcher has put them on every request since the still spine was
            # built, and this line reading only `prompt` is what left the lens
            # routing blind on three of the four still lanes.
            "prompt": caption_json(str(get("prompt") or ""), width, height,
                                   kind=str(get("kind") or ""),
                                   role=str(get("role") or "")),
            "seed": int(get("seed") or 0),
            "width": width, "height": height,
            "steps": STEPS, "mu": MU, "std": STD, "cfg": CFG,
            "object_id": str(get("object_id") or ""),
        }

    def _node_candidates(self, params=None):  # noqa: ARG002
        """Ordered ComfyUI node-class candidates per graph node."""
        return {
            "unet_cond": ("UNETLoader",),
            "unet_uncond": ("UNETLoader",),
            "clip": ("CLIPLoader",),
            "vae": ("VAELoader",),
            "pos": ("CLIPTextEncode",),
            "zero": ("ConditioningZeroOut",),
            "cfg_override": ("CFGOverride",),
            "guider": ("DualModelGuider",),
            "latent": ("EmptyFlux2LatentImage",),
            "sigmas": ("Ideogram4Scheduler",),
            "sampler": ("KSamplerSelect",),
            "noise": ("RandomNoise",),
            "sample": ("SamplerCustomAdvanced",),
            "decode": ("VAEDecode",),
        }

    def _build_graph(self, params, wire):
        """Pure: the official Ideogram 4 dual-expert topology.

        The negative branch is the ZEROED POSITIVE (`ConditioningZeroOut`), not a
        text negative and not an empty socket -- which is precisely why no
        prohibition text can act here (see the module docstring).
        """
        W = wire
        return {
            "unet_cond": {"class": "unet_cond",
                          "inputs": {"unet_name": params["cond_unet"],
                                     "weight_dtype": "default"}},
            "unet_uncond": {"class": "unet_uncond",
                            "inputs": {"unet_name": params["uncond_unet"],
                                       "weight_dtype": "default"}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": params["clip_name"],
                                "type": "ideogram4", "device": "default"}},
            "vae": {"class": "vae", "inputs": {"vae_name": params["vae_name"]}},
            "pos": {"class": "pos",
                    "inputs": {"text": params["prompt"], "clip": W("clip", 0)}},
            "zero": {"class": "zero",
                     "inputs": {"conditioning": W("pos", 0)}},
            "cfg_override": {"class": "cfg_override",
                             "inputs": {"model": W("unet_cond", 0),
                                        "cfg": CFG_OVERRIDE,
                                        "start_percent": CFG_OVERRIDE_START,
                                        "end_percent": CFG_OVERRIDE_END}},
            "guider": {"class": "guider",
                       "inputs": {"model": W("cfg_override", 0),
                                  "positive": W("pos", 0),
                                  "negative": W("zero", 0),
                                  "cfg": float(params["cfg"]),
                                  "model_negative": W("unet_uncond", 0)}},
            "latent": {"class": "latent",
                       "inputs": {"width": int(params["width"]),
                                  "height": int(params["height"]),
                                  "batch_size": 1}},
            "sigmas": {"class": "sigmas",
                       "inputs": {"steps": int(params["steps"]),
                                  "width": int(params["width"]),
                                  "height": int(params["height"]),
                                  "mu": float(params["mu"]),
                                  "std": float(params["std"])}},
            "sampler": {"class": "sampler",
                        "inputs": {"sampler_name": SAMPLER}},
            "noise": {"class": "noise",
                      "inputs": {"noise_seed": int(params["seed"])}},
            "sample": {"class": "sample",
                       "inputs": {"noise": W("noise", 0),
                                  "guider": W("guider", 0),
                                  "sampler": W("sampler", 0),
                                  "sigmas": W("sigmas", 0),
                                  "latent_image": W("latent", 0)}},
            "decode": {"class": "decode",
                       "inputs": {"samples": W("sample", 0),
                                  "vae": W("vae", 0)}},
        }

    def render_image(self, request, prepared=None):  # noqa: ARG002
        """Mint ONE still and return it as a decoded uint8 (H, W, 3) RGB array.

        ONE render per card. No re-roll, by operator directive: *"I don't wanna
        be running extra stills. I accept some errors. I don't want it burning
        extra GPU cycles."* The invented-furniture defect is addressed in the
        PROMPT, where it costs nothing, and any residual is accepted.

        No footer detector runs here. Calibrated on near-black word cards it
        flagged 32 of 35 real production stills, because style packs
        legitimately light the bottom of the frame -- a guard that wrong is
        worse than none. It survives as an offline instrument for the word-card
        corpus.
        """
        from .._otr_video_engines import wrapper_bridge as _wb

        params = self._params(request)
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates(params))
        self._classes = classes
        graph = self._build_graph(params, _wb.Wire)
        try:
            images = _wb.run_graph(graph, classes, terminal=self._TERMINAL)[0]
            frames = _wb.images_to_uint8(images)      # (B,H,W,3) uint8
        finally:
            # Single-resident discipline: 11 GB of experts must not sit on the
            # lease while the next engine waits.
            _wb.reclaim_idle_models(reason="ideogram4_local post-decode")

        frame = frames[0]
        refused, minimum, std = classify_refusal(frame)
        if refused:
            raise Ideogram4RefusalError(
                f"{params['object_id'] or '<object>'}: ideogram4_local returned a "
                f"safety-refusal placeholder (min={minimum:.1f}, std={std:.1f}; "
                f"a real card measures min~0, std~27-41). The card text or its "
                f"styling was refused by the model, not by OTR.")
        log.info(
            "[OTR.image.ideogram4_local] minted still %dx%d seed=%d steps=%d "
            "mu=%.2f cfg=%.1f min=%.1f std=%.1f", params["width"],
            params["height"], params["seed"], params["steps"], params["mu"],
            params["cfg"], minimum, std)
        return frame
