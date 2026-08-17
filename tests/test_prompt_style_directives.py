"""The per-engine prompt-style overlays -- STORED, NOT WIRED (queue item C).

Ten engines each carry two constants, per the schema decided 2026-08-17 in
`docs/2026-08-17-per-engine-prompt-style-guide-RESEARCH.md`:

  * `PROMPT_STYLE_DIRECTIVE` -- 240 chars HARD. The only half that may ever
    reach an LLM or a prompt.
  * `PROMPT_STYLE_NOTES`     -- uncapped, humans only, never injected.

Nothing in the repo reads either one yet; acting on them is a separate, measured
change gated on `scripts/otr_talking_radio_probe_eval.py` at a fixed seed. These
tests therefore pin the CONTRACT, not behaviour -- the cap, the completeness, and
the two prohibitions that stop a future window shipping a known defect one layer
up.

WHY THIS READS BY AST AND NEVER IMPORTS AN ENGINE. `scripts/otr_style_traceroute.py`
established the pattern: it pulls each engine's negative without importing the
module so the tool keeps its "loads no model, spends no GPU" promise. The same
applies here, and it also keeps the test honest about engines whose module scope
reaches for optional runtime deps. One contract difference: the traceroute returns
"" on a read failure because it REPORTS and must never fail a build. A test has
the opposite duty, so every reader here raises instead.
"""
import ast
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]

#: 240 mirrors `render_driver._LTX_MOTION_PROMPT_MAX` and is roughly 4-6 concrete
#: rules. The RESEARCH doc's justification is measured, not aesthetic: the kinetic
#: motion clause (bounded instruction, validated output, deterministic fallback)
#: ran `generated: 6, invalid: 0` on a live leg, while F2 -- a many-rule judgment
#: task on the same class of local model -- swung 3/6 then 1/6 on IDENTICAL
#: fixtures and ships disabled. Bounded and validated works; open-ended does not.
_HARD_CAP = 240

#: The ten owning modules, engine slug -> path. `ltx_8gb` is deliberately ABSENT:
#: the RESEARCH doc treats "ltx_video / ltx_8gb" as ONE block, so the LTX family
#: pair lives on `eng_ltx_video` and the 8GB tier carries a pointer, not a copy.
_OWNERS = {
    "z_image_turbo": "nodes/_otr_image_engines/z_image_turbo.py",
    "flux_gen1": "nodes/_otr_image_engines/flux_gen1.py",
    "lumina_image": "nodes/_otr_image_engines/lumina_image.py",
    "ltx_video": "nodes/_otr_video_engines/eng_ltx_video.py",
    "ltx_av": "nodes/_otr_video_engines/eng_ltx_av.py",
    "wan_i2v": "nodes/_otr_video_engines/eng_wan_i2v.py",
    "wan_ti2v": "nodes/_otr_video_engines/eng_wan_ti2v.py",
    "fastwan_8gb": "nodes/_otr_video_engines/eng_fastwan_8gb.py",
    "humo": "nodes/_otr_video_engines/eng_humo.py",
    "minimax_h3": "nodes/_otr_video_engines/eng_minimax_h3.py",
    # The three the RESEARCH doc never enumerated, added 2026-08-17 on the
    # operator's scope call. Their directives are HIS drafts from public docs,
    # explicitly "NOT yet validated", stored verbatim. `flux2_klein` is the one
    # that matters most: `requires_flag = None`, so it is selectable as shipped
    # while its two siblings are default-OFF opt-ins.
    "flux2_klein": "nodes/_otr_image_engines/flux2_klein.py",
    "hidream_i1": "nodes/_otr_image_engines/hidream_i1.py",
    "sd35_large": "nodes/_otr_image_engines/sd35_large.py",
}

#: The sibling that must NOT grow its own copy of the LTX family pair.
_LTX_POINTER_ONLY = "nodes/_otr_video_engines/eng_ltx_8gb.py"

#: A directive may STATE that a negative is inert or absent -- that is a phrasing
#: fact a writer needs, and `minimax_h3` legitimately opens with it. It may never
#: instruct a model to AUTHOR avoidance content. Struck from z_image on operator
#: instruction 2026-08-17 and generalized here, because the reason generalizes:
#: PBUG-20260817-01 was engine-side negative authoring vetoing the style the
#: episode selected, and teaching a writer to author negatives rebuilds that
#: defect one layer up. The pack owns the style negative, the engine owns hygiene,
#: and video negatives are frozen recipe -- there is no fourth owner.
#:
#: THIS IS A TRIPWIRE, NOT A PROOF -- and its first version was far too weak to
#: be worth the confidence its name implied. A Sonnet QA pass ran 16 plausible
#: authoring instructions past a blocklist of nine phrasings and FOURTEEN walked
#: through: "name what to avoid", "describe what to exclude", "add banned terms",
#: and even "add the negative prompt terms here" -- which the old list missed
#: because it required "negative" and "terms" to sit adjacent. Blocklisting
#: invented phrasings is guessing at an open set. So the rule is INVERTED for the
#: word that matters: any mention of "negative" fails UNLESS an approved
#: has-no-effect hedge is also present. Ways to say "this channel is inert" are a
#: CLOSED set; ways to say "author one" are not.
#:
#: Deliberately narrow to stay free of false positives. Four real directives say
#: "exclusions have no effect", so the avoidance list matches "to exclude" and
#: never the bare stem "exclu"; and it does NOT match a bare "avoid", because
#: "avoid tag lists" is a legitimate PHRASING rule about how to write, not an
#: instruction to author scene content that must be absent.
#: A mention of "negative" is allowed only alongside one of these. Two shapes
#: qualify, and the SECOND was missed on the first pass: saying the channel has no
#: effect, and PROHIBITING the writer from emitting one. The operator's own
#: `flux2_klein` draft closes with "No tags, weights, or negatives" -- the strongest
#: possible form of the strike -- and the first version of this guard would have
#: rejected it for containing the word. A rule that fails maximum compliance is a
#: broken rule, so the prohibition shapes are listed too.
_NEGATIVE_HEDGES = (
    "no negative",
    "or negatives",
    "no negatives",
    "without negatives",
    "negative is inert",
    "negative is absent",
    "negative has no effect",
    "negative is not consulted",
)

_AVOIDANCE_IMPERATIVES = (
    "what to avoid",
    "elements to avoid",
    "things to avoid",
    "to exclude",
    "banned terms",
    "unwanted",
    "undesired",
    "never depict",
    "do not depict",
    "should not appear",
    "steer away",
    "negative keywords",
)

#: Rule 2 of the schema: no adjectives about quality -- they are unactionable and
#: uncheckable, which is exactly what a directive may not be.
_QUALITY_ADJECTIVES = (
    "masterpiece",
    "best quality",
    "high quality",
    "highly detailed",
    "beautiful",
    "stunning",
    "gorgeous",
    "award-winning",
)


#: The two names this module exists to police.
_WANTED = ("PROMPT_STYLE_DIRECTIVE", "PROMPT_STYLE_NOTES")


def _module_constants(rel_path):
    """The overlay constants in a module, read WITHOUT importing it.

    Adjacent string literals inside parentheses are folded by the parser into one
    `ast.Constant`, so a multi-line implicitly-concatenated directive reads back
    as the single joined string the engine will actually expose.

    A NON-LITERAL assignment to either name is a HARD FAILURE, not a miss. The
    first version of this reader skipped anything that was not a plain `ast.Assign`
    of an `ast.Constant`, so a `BinOp` (`"a" + "b"`), an f-string, a `.join()`, or
    an annotated `PROMPT_STYLE_DIRECTIVE: str = "..."` all read back as ABSENT --
    and three downstream checks then passed vacuously on the empty string while
    the author could see the constant sitting right there in the file. A
    self-diagnosing failure beats an invisible one.
    """
    path = _REPO / rel_path
    if not path.is_file():
        pytest.fail("engine module is missing: %s" % rel_path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.Assign):
            targets = node.targets
        else:
            continue
        wanted = [t.id for t in targets
                  if isinstance(t, ast.Name) and t.id in _WANTED]
        if not wanted:
            continue
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            got = type(value).__name__ if value is not None else "no value"
            pytest.fail(
                "%s assigns %s from something other than a plain string literal "
                "(got %s at line %d). This reader cannot evaluate that, and a "
                "silent miss would make the cap and prohibition checks pass on an "
                "empty string. Use a bare literal or implicit concatenation."
                % (rel_path, ", ".join(wanted), got, getattr(node, "lineno", 0)))
        for name in wanted:
            found[name] = value.value
    return found


@pytest.fixture(scope="module")
def overlays():
    """slug -> (directive, notes) for all ten owners."""
    out = {}
    for slug, rel_path in _OWNERS.items():
        consts = _module_constants(rel_path)
        out[slug] = (consts.get("PROMPT_STYLE_DIRECTIVE"),
                     consts.get("PROMPT_STYLE_NOTES"))
    return out


@pytest.mark.parametrize("slug", sorted(_OWNERS))
def test_engine_carries_both_fields(overlays, slug):
    directive, notes = overlays[slug]
    assert directive is not None, (
        "%s has no PROMPT_STYLE_DIRECTIVE -- the schema is two fields per engine"
        % slug)
    assert notes is not None, (
        "%s has no PROMPT_STYLE_NOTES -- the reasoning is half the deliverable"
        % slug)


@pytest.mark.parametrize("slug", sorted(_OWNERS))
def test_directive_within_hard_cap(overlays, slug):
    directive = overlays[slug][0]
    assert directive is not None, (
        "%s has no directive, so a cap check would pass vacuously on \"\"" % slug)
    assert len(directive) <= _HARD_CAP, (
        "%s directive is %d chars, over the %d hard cap: %r"
        % (slug, len(directive), _HARD_CAP, directive))


@pytest.mark.parametrize("slug", sorted(_OWNERS))
def test_directive_is_one_nonempty_block(overlays, slug):
    directive = overlays[slug][0] or ""
    assert directive.strip(), "%s directive is empty" % slug
    assert "\n" not in directive, (
        "%s directive carries a newline -- it is injected into a prompt, so it "
        "must be one flat block" % slug)
    assert directive == directive.strip(), (
        "%s directive has leading/trailing whitespace, which spends cap on "
        "nothing" % slug)


@pytest.mark.parametrize("slug", sorted(_OWNERS))
def test_directive_never_instructs_negative_authoring(overlays, slug):
    """The operator's 2026-08-17 strike, made structural.

    Stating that a negative is inert or absent is allowed and useful. Instructing
    a model to author avoidance content is not -- see the reasoning above
    `_NEGATIVE_HEDGES`.
    """
    directive = overlays[slug][0]
    assert directive is not None, (
        "%s has no directive, so this check would pass vacuously -- fix the "
        "missing field first" % slug)
    low = directive.lower()

    if "negative" in low:
        assert any(hedge in low for hedge in _NEGATIVE_HEDGES), (
            "%s directive mentions the negative with no approved has-no-effect "
            "hedge %s. Saying a negative is inert or absent is legal and useful; "
            "instructing a writer to author one is not -- the pack owns the style "
            "negative, the engine owns hygiene, and video negatives are frozen "
            "recipe. Directive: %r" % (slug, list(_NEGATIVE_HEDGES), directive))

    hits = [phrase for phrase in _AVOIDANCE_IMPERATIVES if phrase in low]
    assert not hits, (
        "%s directive tells the writer to author avoidance content %s -- that is "
        "a negative in everything but name, and it rebuilds PBUG-20260817-01 one "
        "layer up" % (slug, hits))


@pytest.mark.parametrize("slug", sorted(_OWNERS))
def test_directive_has_no_quality_adjectives(overlays, slug):
    directive = overlays[slug][0]
    assert directive is not None, (
        "%s has no directive, so this check would pass vacuously" % slug)
    directive = directive.lower()
    hits = [word for word in _QUALITY_ADJECTIVES if word in directive]
    assert not hits, (
        "%s directive carries unactionable quality adjectives %s (schema rule 2)"
        % (slug, hits))


@pytest.mark.parametrize("slug", sorted(_OWNERS))
def test_notes_carry_a_provenance_stamp(overlays, slug):
    """A stored string must never be mistakable for a measured finding.

    The adoption gate is explicit in the RESEARCH doc -- a directive is a
    hypothesis until a before/after on the probe at a fixed seed. The stamp is
    what stops the next reader treating an authored string as evidence.
    """
    notes = overlays[slug][1] or ""
    assert "PROVENANCE" in notes, (
        "%s notes carry no PROVENANCE stamp" % slug)


def test_directives_are_all_distinct(overlays):
    """Ten engines, ten answers. Identical text means someone pasted rather than
    derived -- the configurations genuinely differ (cfg 1.0 through 5.0, live
    negatives and absent ones, LLM encoders and T5)."""
    seen = {}
    for slug, (directive, _notes) in sorted(overlays.items()):
        if directive in seen:
            pytest.fail("%s and %s share a directive verbatim" % (seen[directive], slug))
        seen[directive] = slug


def test_ltx_8gb_carries_a_pointer_not_a_copy():
    """One authority for the LTX family.

    D-BIS finding 2 is the standing example of what two copies become: the same
    7-term negative boilerplate exists in four copies and two silently diverged
    with no recorded reason. A deliberate DEPARTURE for the 8GB tier is allowed
    later -- it just has to be a real pair with a recorded reason, which is what
    makes it a departure rather than a drift. Until then, no copy.
    """
    consts = _module_constants(_LTX_POINTER_ONLY)
    assert "PROMPT_STYLE_DIRECTIVE" not in consts, (
        "eng_ltx_8gb defines its own PROMPT_STYLE_DIRECTIVE. The LTX family pair "
        "lives on eng_ltx_video; a byte-identical copy here is the duplicate-drift "
        "shape D-BIS finding 2 already flags. If this IS a deliberate departure, "
        "add it to _OWNERS with the reason recorded in its notes.")
    assert "PROMPT_STYLE_NOTES" not in consts, (
        "eng_ltx_8gb defines its own PROMPT_STYLE_NOTES -- see above.")
    body = (_REPO / _LTX_POINTER_ONLY).read_text(encoding="utf-8")
    # Loosened deliberately: requiring the exact dotted string made a purely
    # cosmetic reword or an aliased import fail a test whose real invariant --
    # no second copy -- the two checks above already prove.
    assert "eng_ltx_video" in body and "PROMPT_STYLE_DIRECTIVE" in body, (
        "eng_ltx_8gb must name where its overlay actually lives, or a reader "
        "concludes the engine was simply missed.")


def test_the_overlays_are_not_wired_anywhere():
    """The premise of this whole change, enforced instead of merely asserted.

    Every engine comment and this module's docstring claim nothing reads these
    constants yet, and that acting on them is a separate change gated on a probe
    A/B at a fixed seed. Nothing pinned it, so the gate was honour-system: one
    `directive = z_image_turbo.PROMPT_STYLE_DIRECTIVE` in a dispatcher would wire
    them silently and no other assertion in this file would notice.

    That gate is not ceremony. Queue item D is BLOCKED on a real unknown -- the
    still-prompt writer does not know its target engine, because binding happens
    at dispatch and roles drift under `OTR_FORCE_ENGINE_MAP`. Wiring a per-engine
    directive before that is settled targets an engine the writer cannot see.
    """
    owners = {(_REPO / p).resolve() for p in _OWNERS.values()}
    owners.add((_REPO / _LTX_POINTER_ONLY).resolve())
    skip = {".git", ".claude", "__pycache__", "tests", "node_modules",
            "kibitz-runs", "kibitz", "otr", "output", "docs", "site-packages"}
    offenders = []
    for path in _REPO.rglob("*.py"):
        if any(part in skip for part in path.parts):
            continue
        if path.resolve() in owners:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name in _WANTED:
            if name in text:
                offenders.append("%s reads %s"
                                 % (path.relative_to(_REPO).as_posix(), name))
    assert not offenders, (
        "the overlays are STORED, NOT WIRED. Acting on them is a separate, "
        "measured change -- a before/after on "
        "scripts/otr_talking_radio_probe_eval.py at a fixed seed -- and the "
        "engine-binding question (queue item D) is still BLOCKED. New readers "
        "found:\n  %s" % "\n  ".join(offenders))


def test_the_owner_map_still_points_at_real_files():
    """The map cannot rot silently.

    SCOPE, stated plainly because this test's first name ("no engine module was
    missed") promised a completeness it does not deliver: it checks the TEN the
    RESEARCH doc enumerated and cannot discover an engine nobody listed.

    A Sonnet QA pass found three registered local image engines outside the
    RESEARCH doc's original ten -- `flux2_klein` (`requires_flag = None`, so it is
    NOT gated and sits live in the menu), `hidream_i1` and `sd35_large` (both
    default-OFF opt-ins). **The operator closed that scope call on 2026-08-17 and
    supplied all three directives himself**, drafted from public docs and then
    validated in a v2 pass, so the map is now THIRTEEN. His three are stored
    verbatim; the original ten remain driver-derived. Neither set is measured.
    """
    for slug, rel_path in sorted(_OWNERS.items()):
        assert (_REPO / rel_path).is_file(), (
            "%s: %s no longer exists -- the overlay moved or the engine was "
            "renamed, and this map is now lying" % (slug, rel_path))
    assert len(_OWNERS) == 13, (
        "this map has %d. It was TEN (the RESEARCH doc's blocks) until the "
        "operator's 2026-08-17 scope call added flux2_klein, hidream_i1 and "
        "sd35_large. Adding or removing one is a deliberate act, so update this "
        "count in the same commit -- this assertion firing is the guard working, "
        "not a bug." % len(_OWNERS))
