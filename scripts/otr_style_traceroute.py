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

THE VIDEO SIDE (added 2026-08-17, GO_FORWARD D-BIS finding 1). Everything above
audits STILL surfaces. `VIDEO_DICT_SURFACES` had existed here from the start but
was only ever DISPLAYED -- `_fights_in` never read it -- so a video engine's
negative contradicting a pack's own `motion_registers` was structurally
invisible to this tool. It is now measured and REPORTED. What it found:

  eng_cloud_video conflicts with FIVE packs. It bans "whip pans" while the
  `music_open` register of `anime`, `cartoon`, `paper_origami` AND the DEFAULT
  `sci_fi_radio` all ask for a dial that "whip-pans"; it bans "flicker" while
  `shakespeare_stage_realism`'s announcer asks for candlelight that flickers.
  Every LOCAL video engine (ltx_av, humo, ltx_8gb, ltx_video) is clean -- their
  negatives are pure quality/artifact terms, so the exposure is confined to the
  opt-in cloud lane.

  This is a STATIC-AUDIT finding, not a PBUG: the cloud engine is opt-in,
  requires credentials, and its provider-side behaviour is unverifiable from
  this repo. It needs one live observation before promotion.

  `--strict` IS DELIBERATELY NOT EXTENDED TO IT. The video negatives are FROZEN
  RECIPE ("the recipes are not on the table"), so failing CI on a string this
  repo may not edit would be a build-breaker by construction. Resolving these --
  by dropping a term at COMPOSE time, never by editing a recipe string -- is an
  operator ruling, not a driver decision.

    python scripts/otr_style_traceroute.py
    python scripts/otr_style_traceroute.py --style cartoon
    python scripts/otr_style_traceroute.py --fights-only

Exit code is 0 in all cases -- including when fights are found. This reports;
it never fails a build. Use --strict if you want a non-zero exit for CI.
UTF-8, no BOM, ASCII output.
"""
from __future__ import annotations

import argparse
import ast
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

#: THE VIDEO SIDE (2026-08-17, D-BIS finding 1). `VIDEO_DICT_SURFACES` has
#: existed since this tool was written but was only ever DISPLAYED -- `_fights_in`
#: reads STILL surfaces alone, so a video negative contradicting a pack's own
#: `motion_registers` was structurally invisible here. These are the shipped
#: per-engine video negatives, by (label, module path, constant name).
#:
#: They are read STATICALLY, by AST, rather than imported. This tool's promise is
#: that it "renders nothing, loads no model, spends no GPU"; importing a video
#: engine drags in the render stack and breaks that on the spot.
VIDEO_NEGATIVE_SOURCES = (
    ("eng_cloud_video", "nodes/_otr_video_engines/eng_cloud_video.py",
     "_WAN_NEGATIVE_DEFAULT"),
    ("eng_ltx_av", "nodes/_otr_video_engines/eng_ltx_av.py",
     "_LTX_DEFAULT_NEGATIVE"),
    ("eng_humo", "nodes/_otr_video_engines/eng_humo.py",
     "_HUMO_DEFAULT_NEGATIVE"),
    ("eng_ltx_8gb", "nodes/_otr_video_engines/eng_ltx_8gb.py",
     "_LTX8_DEFAULT_NEGATIVE"),
    ("eng_ltx_video", "nodes/_otr_video_engines/eng_ltx_video.py",
     "_LTX_DEFAULT_NEGATIVE"),
    # The word-card animator (Pixverse) is a SECOND negative-bearing engine in
    # eng_cloud_video.py. It was missed on the first pass because this list is
    # hand-curated -- see the coverage guard below, which is the real fix.
    ("eng_cloud_word_razzle", "nodes/_otr_video_engines/eng_cloud_video.py",
     "_RAZZLE_NEG_DEFAULT"),
    # Shared by eng_wan_i2v + eng_wan_ti2v, which import it rather than
    # defining their own. Found BY the coverage guard, not by hand -- which is
    # the argument for the guard existing.
    ("wan_shared (i2v + ti2v)", "nodes/_otr_video_engines/wan_shared.py",
     "_WAN_DEFAULT_NEGATIVE"),
)


def discover_video_negative_constants():
    """Every module-level `*NEGATIVE*` string constant in the video engines.

    `VIDEO_NEGATIVE_SOURCES` above is hand-curated, which makes it wrong by
    construction the moment an engine grows a negative -- exactly what happened
    with `_RAZZLE_NEG_DEFAULT`. This discovers them instead, so the report can
    say out loud when it is auditing fewer negatives than exist. Returns
    ``{(relpath, const_name)}``.
    """
    found = set()
    root = os.path.join(_REPO, "nodes", "_otr_video_engines")
    try:
        names = sorted(f for f in os.listdir(root) if f.endswith(".py"))
    except OSError:
        return found
    for fname in names:
        path = os.path.join(root, fname)
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                name = getattr(target, "id", "")
                if "NEGATIVE" not in name.upper() or name.endswith("_ENV"):
                    continue
                value = node.value
                if isinstance(value, ast.Call) and value.args:
                    value = value.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    found.add(("nodes/_otr_video_engines/" + fname, name))
    return found

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


def _literal_assign(path, name):
    """The string a module assigns to ``name``, read by AST -- no import.

    Handles the two shapes the video engines actually use: a plain (possibly
    implicitly-concatenated) string literal, and a literal wrapped in a call --
    `_WAN_NEGATIVE_DEFAULT = visual_safety_negative("...")`. The wrapper only
    merges in `VISUAL_SAFETY_NEGATIVE_PROMPT`, which is "", so the literal is
    the effective value. Returns "" when the name is absent or not a literal,
    because this tool reports and must never fail a build.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
    except (OSError, SyntaxError):
        return ""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == name
                   for t in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return ""


def _motion_text(style):
    """Every `motion_registers` value joined -- the pack's VIDEO positive."""
    values = getattr(style, "motion_registers", None) or {}
    try:
        return " ".join(str(v or "") for v in values.values()).lower()
    except AttributeError:
        return ""


def find_video_fights(negative_text, style):
    """Phrases in a VIDEO negative that this pack's `motion_registers` ask for.

    Same phrase discipline and the same artifact-term exclusion as
    `_fights_in`, pointed at the video surface. It additionally matches simple
    hyphen/plural variants, because the packs write "Dial whip-pans" where the
    engine negative writes "whip pans" -- a strict literal test misses the
    single widest conflict in the tree.
    """
    phrases = [c.strip().lower() for c in str(negative_text or "").split(",")
               if c.strip() and c.strip().lower() not in _ARTIFACT_TERMS]
    positive = _motion_text(style)
    if not positive or not phrases:
        return []
    hits = []
    for phrase in phrases:
        stem = phrase.rstrip("s")
        variants = {phrase, stem, phrase.replace(" ", "-"),
                    phrase.replace("-", " "), stem.replace(" ", "-")}
        for variant in variants:
            if variant and re.search(r"\b%s" % re.escape(variant), positive):
                hits.append(phrase)
                break
    return hits


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
    # ---- THE VIDEO SIDE (D-BIS finding 1) --------------------------------
    # Reported, NEVER gated. `--strict` stays scoped to STILL effective fights
    # on purpose: the video negatives are FROZEN RECIPE ("the recipes are not
    # on the table"), so failing CI on a string this repo may not edit would be
    # a build-breaker by construction. Whether these conflicts should be
    # resolved -- and by dropping a term at compose time, never by editing a
    # recipe -- is an operator call, not a driver decision.
    out.append("=" * 72)
    out.append("VIDEO-SIDE: engine negative vs the pack's own motion_registers")
    out.append("  (reported only -- --strict is deliberately NOT extended here)")
    video_total = 0
    for label, relpath, const in VIDEO_NEGATIVE_SOURCES:
        negative = _literal_assign(os.path.join(_REPO, relpath), const)
        if not negative:
            out.append("  %-18s (negative not readable -- skipped)" % label)
            continue
        hit_packs = []
        for sid in ids:
            try:
                style = vs.resolve_visual_style(sid)
            except Exception:  # noqa: BLE001 -- already reported above
                continue
            hits = find_video_fights(negative, style)
            if hits:
                hit_packs.append("%s -> %s" % (sid, ", ".join(sorted(set(hits)))))
                video_total += len(set(hits))
        if hit_packs:
            out.append("  %-18s CONFLICTS on %d pack(s):"
                       % (label, len(hit_packs)))
            for hp in hit_packs:
                out.append("       %s" % hp)
        else:
            out.append("  %-18s clean against every pack" % label)

    # COVERAGE GUARD: say out loud when a negative exists that this report did
    # not audit. A hand-curated source list that silently omits an engine reads
    # exactly like an engine with no conflicts.
    audited = {(relpath, const) for _, relpath, const in VIDEO_NEGATIVE_SOURCES}
    missed = sorted(discover_video_negative_constants() - audited)
    if missed:
        out.append("  !! NOT AUDITED -- %d video negative(s) exist that this "
                   "report does not cover:" % len(missed))
        for relpath, const in missed:
            out.append("       %s :: %s" % (relpath, const))
        out.append("     Add them to VIDEO_NEGATIVE_SOURCES.")
    else:
        out.append("  coverage: every discoverable video negative is audited")

    out.append("=" * 72)
    out.append("EFFECTIVE FIGHTS: %d   <- MUST BE 0. A negative may never "
               "conflict with a visual style" % fights_total)
    out.append("                        (operator ruling 2026-08-17). Any "
               "non-zero value here is a defect.")
    out.append("AUTHORED conflicts: %d  <- housekeeping only; resolved at mint "
               "by effective_negative()." % authored_total)
    out.append("VIDEO conflicts:    %d  <- reported, not gated. Frozen recipe: "
               "needs an operator ruling," % video_total)
    out.append("                        never a driver edit to a recipe string.")

    print("\n".join(out))
    return 1 if (args.strict and fights_total) else 0


if __name__ == "__main__":
    raise SystemExit(main())
