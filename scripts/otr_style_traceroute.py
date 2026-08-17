"""OTR style traceroute -- where every style identity reaches a prompt.

ONE STYLE AUTHORITY (2026-08-17, PBUG-20260817-01). The operator's requirement,
in his words: *"it touches all still packs and all video packs so will need a
good traceroute."* He is right about the blast radius -- roughly twenty
style-bearing surfaces per pack across two prompt families and ten style
identities -- and that is not something anyone can eyeball.

READ-ONLY. It renders nothing, loads no model, spends no GPU, is wired into no
node, and is not part of the canonical workflow. It prints a report and exits 0.
It is NOT a classifier and NOT a gate: it inspects our own CONFIG against
itself, never anyone's story, so THE LAW is untouched.

THE QUESTION IT EXISTS TO ANSWER is THE FIGHT: does a pack's own negative
contradict its own positive? Before this build the engine-side negative was
style-blind and applied to every pack -- it said "cartoon, illustration" while a
cartoon episode's positive said "bright cartoon illustration".

By the literal phrase match this tool uses, that historical negative fought
FOUR of the nine packs: `anime`, `cartoon`, `sci_fi_radio` and
`storybook_engraving`. (`recur_frac` and `video_art` also read as mis-served on
JUDGMENT -- "clean digital" against "pristine raster geometry", "oversaturated,
glossy" against "luminous ... phosphor bloom" -- but neither is a literal
phrase collision, so this tool does not and should not count them. An earlier
draft of the docs asserted "six of nine" by mixing the judgment call into the
measured number; the measured number is four.)

After the build the eight non-default packs are CLEAN.

Two columns are reported per pack, and the difference between them matters:

  AUTHORED  -- the raw `negative_tail` in the pack JSON. A conflict here means
               the pack was authored asking to suppress something it also asks
               for. `sci_fi_radio` has two: its negative says "cartoon,
               illustration" while its own surfaces ask for
               `announcer_subject_ltx_mouth` -> "a living cartoon appliance
               face" and `still_word_title_mood_style` -> "atmospheric period
               illustration".
  EFFECTIVE -- what actually conditions a mint, after
               `_otr_visual_styles.effective_negative` drops any phrase the
               pack's own positive asks for.

Operator ruling 2026-08-17: *"we can't have negative prompts conflicting with
any visual style."* So EFFECTIVE conflicts must be ZERO, always, for every
identity including the dynamic pack -- that is the assertion worth wiring into
CI (`--strict` fails only on effective conflicts). An AUTHORED conflict is a
housekeeping note, not a defect: the runtime resolves it, and the raw string is
kept so the default lane's historical negative stays legible.

    python scripts/otr_style_traceroute.py
    python scripts/otr_style_traceroute.py --style cartoon
    python scripts/otr_style_traceroute.py --fights-only

Exit code is 0 in all cases -- including when fights are found. This reports;
it never fails a build. Use --strict if you want a non-zero exit for CI.
UTF-8, no BOM, ASCII output.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_REPO, "nodes") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO, "nodes"))

os.environ.setdefault("OTR_TEST_MODE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import _otr_visual_styles as vs  # noqa: E402

#: Pack fields that reach a STILL prompt, grouped the way an operator thinks
#: about them rather than the way the dataclass is ordered.
STILL_SURFACES = (
    ("tails", ("positive_tail", "image_grade_tail", "broadcast_tail",
               "era_tail")),
    ("character", ("portrait_look", "portrait_look_talking",
                   "portrait_instruction_look")),
    ("scene", ("scene_instruction_look", "plate_look")),
    ("announcer", ("announcer_subject_face", "announcer_subject_ltx_mouth",
                   "announcer_subject_object", "radio_object_look")),
    ("cards", ("still_word_title_mood_style",)),
    ("fallback", ("non_character_emblem_fallback",)),
)
STILL_DICT_SURFACES = ("open_subjects", "still_word_typography",
                       "still_word_backdrop")
VIDEO_DICT_SURFACES = ("motion_registers",)

#: Engines that actually apply a negative, with the CFG that makes it live.
#: A negative at cfg 1.0 is inert -- Flux takes its adherence from a
#: FluxGuidance embedding -- so listing the number keeps the report honest.
NEGATIVE_ENGINES = (
    ("z_image_turbo", 2.0, "pack negative + request, hygiene fallback"),
    ("lumina_image", 4.0, "request negative (env override only)"),
    ("flux_gen1", 1.0, "request negative -- INERT at cfg 1.0"),
)
NO_NEGATIVE_ENGINES = ("sd35_large", "flux2_klein", "hidream_i1",
                       "eng_google_image", "eng_cloud_image")

_WORD = re.compile(r"[a-z][a-z0-9-]+")
#: Terms that are anti-ARTIFACT, not anti-STYLE. A pack may legitimately carry
#: these in its negative even if the word appears in its positive text.
_ARTIFACT_TERMS = frozenset({"text", "watermark"})
#: Words too generic for a contradiction to mean anything.
_STOPWORDS = frozenset({
    "and", "the", "with", "clean", "soft", "deep", "bold", "flat", "warm",
    "cool", "dark", "light", "lighting", "color", "colour", "style", "shot",
    "look", "even", "low", "high", "subtle", "grade", "tone", "toned",
})


def _terms(text):
    return {w for w in _WORD.findall(str(text or "").lower())
            if w not in _STOPWORDS}


def _negative_phrases(style):
    """The negative as the COMMA-SEPARATED PHRASES it is actually written in.

    Phrases, not bare words, because these negatives are qualified: "plastic
    skin" and "waxy skin" do not contradict a portrait that wants realistic
    skin, they are the specific failure modes being excluded. Splitting them
    into words reported `skin` as a fight on two packs that have none.
    """
    out = []
    for raw in str(getattr(style, "negative_tail", "") or "").split(","):
        phrase = raw.strip().lower()
        if phrase and phrase not in _ARTIFACT_TERMS:
            out.append(phrase)
    return out


def _fights_in(negative_text, style):
    """Phrases in ``negative_text`` that ``style``'s own positive asks for.

    Compares against the positive surfaces an image engine actually sees, not
    just `positive_tail` -- the character portrait that flipped photoreal was
    driven by `portrait_look`, so a fight there is as real as one in the tail.
    """
    phrases = [c.strip().lower() for c in str(negative_text or "").split(",")
               if c.strip() and c.strip().lower() not in _ARTIFACT_TERMS]
    if not phrases:
        return []
    parts = [str(getattr(style, f, "") or "")
             for _, fields in STILL_SURFACES for f in fields]
    # The dict surfaces reach prompt text too (the still-word card joins
    # typography + backdrop straight in), so they count for conflicts.
    for f in STILL_DICT_SURFACES:
        values = getattr(style, f, None) or {}
        try:
            parts.extend(str(v or "") for v in values.values())
        except AttributeError:
            pass
    positive = " ".join(parts).lower()
    return [p for p in phrases if re.search(r"\b%s\b" % re.escape(p), positive)]


def find_fights(style):
    """AUTHORED conflicts -- what the pack JSON declares."""
    return _fights_in(getattr(style, "negative_tail", ""), style)


def find_effective_fights(style):
    """EFFECTIVE conflicts -- what survives to condition a real mint.

    This is the one that must always be empty (operator ruling 2026-08-17).
    """
    return _fights_in(vs.effective_negative(style), style)


def describe(style, verbose=True):
    lines = []
    sid = style.style_id
    cue = vs.compact_style_cue(style)
    lines.append("=" * 72)
    lines.append("STYLE: %s (%s)" % (sid, style.label))
    lines.append("  front-anchored token : %s"
                 % (repr(cue) if cue else "(none -- default pack, exempt)"))
    neg = str(getattr(style, "negative_tail", "") or "")
    lines.append("  pack negative        : %s"
                 % (neg if neg else "(empty -> engine hygiene fallback)"))
    if not verbose:
        return lines

    lines.append("  -- STILL surfaces --")
    for group, fields in STILL_SURFACES:
        for f in fields:
            val = str(getattr(style, f, "") or "")
            mark = "     " if val else "  (-) "
            lines.append("%s%-14s %-30s %s"
                         % (mark, group, f, (val[:60] if val
                                             else "(empty by design)")))
    for f in STILL_DICT_SURFACES:
        d = getattr(style, f, {}) or {}
        lines.append("     %-14s %-30s %d keys" % ("dict", f, len(d)))
    lines.append("  -- VIDEO surfaces --")
    for f in VIDEO_DICT_SURFACES:
        d = getattr(style, f, {}) or {}
        longest = max((len(v) for v in d.values()), default=0)
        lines.append("     %-14s %-30s %d keys, longest %d/%d chars"
                     % ("dict", f, len(d), longest,
                        vs._MOTION_REGISTER_MAX_CHARS))
    lines.append("     %-14s %-30s %s"
                 % ("prepend", "_prefix_video_style_cue",
                    "render_driver, 5 call sites (shared token)"))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--style", help="inspect one style id")
    ap.add_argument("--fights-only", action="store_true",
                    help="print only the negative-vs-positive contradictions")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any fight is found (CI use)")
    args = ap.parse_args(argv)

    ids = ([args.style] if args.style
           else sorted(vs.resolve_visual_style(i).style_id
                       for i in vs.list_style_ids())
           if hasattr(vs, "list_style_ids") else None)
    if ids is None:
        ids = sorted(os.path.splitext(f)[0]
                     for f in os.listdir(vs._VISUAL_STYLES_ROOT)
                     if f.endswith(".json"))

    fights_total = 0
    authored_total = 0
    out = []
    for sid in ids:
        try:
            style = vs.resolve_visual_style(sid)
        except Exception as exc:  # noqa: BLE001
            out.append("STYLE: %s -- UNRESOLVABLE (%s)" % (sid, exc))
            continue
        if not args.fights_only:
            out.extend(describe(style))
        authored = find_fights(style)
        effective = find_effective_fights(style)
        fights_total += len(effective)
        authored_total += len(authored)
        if effective:
            out.append("  !! EFFECTIVE FIGHT %-20s suppresses its own positive "
                       "AT MINT TIME: %s" % (sid, ", ".join(effective)))
        if authored:
            out.append("  ~  authored conflict %-19s %s -> dropped by "
                       "effective_negative, never applied"
                       % (sid, ", ".join(authored)))
            for term in authored:
                for group, fields in STILL_SURFACES:
                    for f in fields:
                        if re.search(r"\b%s\b" % re.escape(term),
                                     str(getattr(style, f, "") or "").lower()):
                            out.append("       '%s' wanted by %s.%s"
                                       % (term, group, f))
        if not (authored or effective) and not args.fights_only:
            out.append("  ok: negative does not contradict this pack")

    out.append("=" * 72)
    out.append("NEGATIVE-CAPABLE ENGINES")
    for name, cfg, note in NEGATIVE_ENGINES:
        live = "LIVE" if cfg > 1.0 else "inert"
        out.append("  %-16s cfg %-4s %-6s %s" % (name, cfg, live, note))
    out.append("  no negative conditioning at all: %s"
               % ", ".join(NO_NEGATIVE_ENGINES))
    out.append("=" * 72)
    out.append("EFFECTIVE FIGHTS: %d   <- MUST BE 0. A negative may never "
               "conflict with a visual style" % fights_total)
    out.append("                        (operator ruling 2026-08-17). Any "
               "non-zero value here is a defect.")
    out.append("AUTHORED conflicts: %d  <- housekeeping only; resolved at mint "
               "by effective_negative()." % authored_total)

    print("\n".join(out))
    return 1 if (args.strict and fights_total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
