"""Cross-lane structural guards for the sci-fi source banks.

These are CLASS-level regressions, not one-off fixes: each guard here pins a
defect that killed a live canonical roll on ONE lane and could equally kill the
others. Codex and Sonnet each hand a strict Pydantic model a blob of
model-emitted JSON, so a contract that JSON cannot express is not a style
problem -- it is an artifact that can never validate, no matter what the model
writes.

Live case (2026-07-11, Gemini P1, first roll to reach it):
    pitches: Input should be a valid tuple
      [type=tuple_type, input_value=[{'premise': ...}], input_type=list]
`PitchSlateV4.pitches` was annotated `tuple[PitchV4, PitchV4, PitchV4]`. JSON has
no tuple -- a model can only ever emit an array -- and the lane's `_Strict` config
(`strict=True`) refuses to coerce a list into a tuple. The pass was unsatisfiable
by construction. The fix expresses the same "exactly three" contract as a
length-pinned list.
"""
from __future__ import annotations

import importlib
import json
import pathlib
import typing

import pytest
from pydantic import BaseModel

LANE_MODULES = (
    "nodes._otr_scifi_codex",
)

# Types JSON cannot represent. A JSON document has exactly one sequence type
# (array) and one mapping type (object); it has no tuple, set, or frozenset.
JSON_IMPOSSIBLE = (tuple, set, frozenset)


def _strict_models(module):
    for name in dir(module):
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and (obj.model_config or {}).get("strict") is True
        ):
            yield name, obj


def _offending_types(annotation) -> list:
    """Every JSON-impossible origin anywhere inside a (possibly nested) annotation."""
    found = []
    origin = typing.get_origin(annotation)
    if origin in JSON_IMPOSSIBLE:
        found.append(origin)
    if annotation in JSON_IMPOSSIBLE:
        found.append(annotation)
    for arg in typing.get_args(annotation):
        found.extend(_offending_types(arg))
    return found


@pytest.mark.parametrize("module_name", LANE_MODULES)
def test_no_strict_lane_field_is_unrepresentable_in_json(module_name):
    """A strict lane model may not require a type JSON cannot produce.

    tuple / set / frozenset are unsatisfiable under `strict=True` when the value
    arrives from `json.loads` -- the pass can never validate. Use a length-pinned
    `list` (min_length == max_length) to express a fixed-arity contract instead.
    """
    module = importlib.import_module(module_name)
    violations = []
    for model_name, model in _strict_models(module):
        for field_name, field in model.model_fields.items():
            bad = _offending_types(field.annotation)
            if bad:
                violations.append(
                    f"{module_name}.{model_name}.{field_name}: "
                    f"{field.annotation!r} requires {bad[0].__name__}, "
                    f"which JSON cannot express"
                )
    assert not violations, (
        "strict lane model fields must be expressible in JSON -- a model can only "
        "ever emit an array, and strict mode will not coerce it:\n  "
        + "\n  ".join(violations)
    )


# A model-authored collection whose ITEM has no declared shape is a coin flip: the
# model must guess whether to return a list of things or an object grouping those
# things. Both readings satisfy `list[dict[str, str]]`-shaped prose, so which one you
# get depends on the model, and the seam's wording, and the day.
#
# Live case (2026-07-11, Codex P6, gemma-4-E4B on the creative slot):
#     issues: Input should be a valid list
#       [type=list_type, input_value={'blur_of_causality': [{...}]}, input_type=dict]
# `ListenerReviewV4.issues` was `list[dict[str, str]]` and the seam named six
# diagnostic lenses. The model grouped its issues under those lenses -- a completely
# reasonable reading of what we wrote -- and P6 exhausted its ladder and killed the
# roll. Mistral had been guessing the other way and we mistook that for a contract.
#
# Each entry below is a field allowed to stay shapeless, with the reason it is NOT a
# coin flip. Anything else must nest a real model. Add to this list only with a reason
# that a NEW model, reading only the schema, could not misread.
ALLOWED_SHAPELESS = {
    # justification: guarded. _has_forbidden_script_scene_keys pins the container to a
    # list of dicts and rejects score-shaped echoes, and a deterministic repair covers
    # the rest. The free keys inside are deliberate room for the script's scene prose.
    "nodes._otr_scifi_codex.ScriptArtifactV4.scenes",
}


def test_no_lane_asks_a_model_to_count_words():
    """No strict lane model may carry a model-reported word count.

    Word count is advisory and never a gate (operator law). A field that asks the model
    to report counts invites a seam that audits them -- and an LLM cannot count words,
    so the gate fails on a measurement the model cannot make and the writer cannot fix.
    Python measures the real count objectively at assembly (`word_receipt`).

    Live case (2026-07-11): `FinalAuditV4.observed_word_counts` was REQUIRED, and
    `codex_final_audit_system` told the auditor to check the "exact word count" and
    "pass only if all checks are true" -- a word-count gate that could demand a full
    P9 rewrite. Nothing in Python ever read the field.
    """
    offenders = []
    for module_name in LANE_MODULES:
        module = importlib.import_module(module_name)
        for model_name, model in _strict_models(module):
            for field_name in model.model_fields:
                lowered = field_name.lower()
                if "word" in lowered and ("count" in lowered or "len" in lowered):
                    offenders.append(f"{module_name}.{model_name}.{field_name}")
    assert not offenders, (
        "these fields ask a model to report a word count -- Python measures words, "
        "models write them:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("pack_path", sorted(
    pathlib.Path(__file__).resolve().parents[1].joinpath("nodes", "story_packs").glob("*/*_v1.json")
))
def test_no_seam_audits_word_count(pack_path):
    """A seam may steer toward a length. It may never make length a pass/fail check."""
    import json

    text = json.dumps(json.loads(pack_path.read_text(encoding="utf-8"))).lower()
    forbidden = ("exact word count", "verify the word count", "correct word count")
    hits = [phrase for phrase in forbidden if phrase in text]
    assert not hits, (
        f"{pack_path.name} makes word count auditable: {hits}. Word count is advisory; "
        "gating on it asks a model to enforce something it cannot measure."
    )


def _is_shapeless(annotation) -> bool:
    """True when a container's items have no declared shape (dict/Any, not a model)."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if annotation is typing.Any:
        return True
    if origin is dict or annotation is dict:
        return True
    if origin in (list, set, frozenset, tuple):
        return any(_is_shapeless(arg) for arg in args)
    if origin is typing.Union or str(origin) == "types.UnionType":
        return any(_is_shapeless(a) for a in args if a is not type(None))
    return False


@pytest.mark.parametrize("module_name", LANE_MODULES)
def test_no_model_authored_collection_is_shapeless(module_name):
    """A collection the model FILLS must declare what one item looks like.

    Otherwise the model has to guess between `[{...}, {...}]` and
    `{"category": [{...}]}` -- and a schema that admits two readings will eventually
    be read the other way. Nest a real model (see ListenerIssueV4) instead.
    """
    module = importlib.import_module(module_name)
    violations = []
    for model_name, model in _strict_models(module):
        for field_name, field in model.model_fields.items():
            path = f"{module_name}.{model_name}.{field_name}"
            if path in ALLOWED_SHAPELESS:
                continue
            if _is_shapeless(field.annotation):
                violations.append(f"{path}: {field.annotation!r}")
    assert not violations, (
        "these model-authored fields have no declared item shape, so the model must "
        "guess the container -- nest a real model, or justify it in ALLOWED_SHAPELESS:"
        "\n  " + "\n  ".join(violations)
    )


def test_listener_issue_is_typed_and_keeps_category_open():
    """The P6 fix pins the SHAPE without pinning the listener's vocabulary."""
    from nodes import _otr_scifi_codex as lane

    issues = lane.ListenerReviewV4.model_fields["issues"]
    assert typing.get_args(issues.annotation)[0] is lane.ListenerIssueV4, (
        "issues must be a list of typed ListenerIssueV4, not a shapeless container"
    )
    # A listener that coins a better word for the flaw than our six is doing its job.
    # Pinning `category` to an enum would reject it for that.
    assert lane.ListenerIssueV4.model_fields["category"].annotation is str


def test_listener_review_repair_flattens_grouped_issues_verbatim():
    """The exact payload that killed the 2026-07-11 roll must now normalize.

    Python re-homes the model's sentences into the right container. It must never
    write one: every word in the repaired review has to appear in the raw output.
    """
    from nodes import _otr_scifi_codex as lane

    raw = """{
      "strengths": ["The cold open earns its silence."],
      "issues": {
        "blurred_causality": [
          {"line_id": "l004", "direction": "Show the relay failing before Vesh names it."}
        ],
        "stalled_pacing": [
          {"direction": "Cut the second corridor beat; it repeats the first."}
        ]
      },
      "require_full_retake": true
    }"""

    review = lane.repair_listener_review_shape(raw)
    assert review is not None
    assert [i.category for i in review.issues] == ["blurred_causality", "stalled_pacing"]
    assert review.issues[0].line_id == "l004"
    assert review.issues[1].line_id is None
    assert review.strengths == ["The cold open earns its silence."]
    assert review.require_full_retake is True
    for issue in review.issues:
        assert issue.direction in raw, "Python may re-home the model's words, never write them"


def test_listener_review_repair_fails_closed_on_an_unmappable_issue():
    """Fail closed rather than silently drop a diagnosis.

    If we cannot tell which of several strings is the direction, guessing would either
    invent a critique or quietly lose one. Return None and let the typed repair ask.
    """
    from nodes import _otr_scifi_codex as lane

    raw = '{"issues": {"pacing": [{"where": "l002", "why": "slow", "how": "trim"}]}}'
    assert lane.repair_listener_review_shape(raw) is None


def test_no_lane_demands_frozen_clean():
    """`frozen_with_warns` is a CLEAN freeze and must never block a finished episode.

    It means the reviewer read the ledger and made NO edits; the warns are soft gaps --
    notes. The structural verdicts (frozen_with_doctor_edits, too_many_edits,
    needs_full_rerun) are the ones that block, because those are defects.

    Live case (2026-07-11, Codex, gemma-4-E4B): the pre-tail audit demanded
    `frozen_clean` outright while the saved-ledger audit ten lines below accepted
    `frozen_with_warns` -- so the IDENTICAL ledger was illegal before the save and legal
    after it. "frozen_with_warns -- 2 soft gap(s)" killed a roll that had passed P0-P8.
    Gemini and Sonnet had already been fixed; Codex was the last lane holding the stale
    gate, and nothing was watching for the divergence.
    """
    import pathlib
    import re

    nodes = pathlib.Path(__file__).resolve().parents[1] / "nodes"
    offenders = []
    for lane in sorted(nodes.glob("_otr_scifi_*.py")):
        for lineno, line in enumerate(lane.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            # An equality test against frozen_clean alone -- the membership form
            # `not in ("frozen_clean", "frozen_with_warns")` is the correct gate.
            if re.search(r'(?:!=|==)\s*["\']frozen_clean["\']', line):
                offenders.append(f"{lane.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "a freeze gate compares against frozen_clean alone, so frozen_with_warns -- a "
        "clean freeze with no reviewer edits -- would kill a finished episode:\n  "
        + "\n  ".join(offenders)
    )


def test_source_grounded_p0_disables_generic_string_clamping():
    """P0 may repair literal metadata, never silently truncate authored fields."""
    import ast
    import pathlib

    codex_source = pathlib.Path("nodes/_otr_scifi_codex.py")
    codex_tree = ast.parse(codex_source.read_text(encoding="utf-8"))
    for node in ast.walk(codex_tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "invoke_codex_structured":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        pass_node = keywords.get("pass_id")
        if not (isinstance(pass_node, ast.Constant) and pass_node.value == "P0"):
            continue
        clamp = keywords.get("clamp_overlong_strings")
        assert isinstance(clamp, ast.Constant) and clamp.value is False
        break
    else:
        pytest.fail("no Codex P0 call found")


def test_codex_audit_passes_never_fail_a_story_over_a_critic_note():
    """THE LAW: an audit may improve a story, it may never fail one.

    Codex's audit passes (P4 structure review, P6 listening room, P8 final
    audit) return CRITIQUE artifacts -- verdicts, issue notes, rationales. Not
    one character of any of them is ever spoken aloud, and the only fields that
    steer anything downstream are Literals that a length clamp cannot touch. So
    an over-long critic NOTE must never be able to abort the episode.

    This class has now killed two live legs:
      * 2026-07-14, 30w  -- P2 died on a 140-char `cast.2.role_in_conflict`.
      * 2026-07-14, 420w -- P4 died on a 240+ char `rationale` (f6c42c5f), the
        model's own footnote on a story it had just reviewed.

    P6 and P8 are protected by the clamp DEFAULT (True). P4 was the lone audit
    explicitly opted out, which is exactly how it got to kill a leg. Pin all
    three: an audit pass must never be fail-closed on string length.
    """
    import ast
    import pathlib

    tree = ast.parse(
        pathlib.Path("nodes/_otr_scifi_codex.py").read_text(encoding="utf-8")
    )
    audit_passes = {"P4", "P6", "P8"}
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "invoke_codex_structured":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        pass_node = keywords.get("pass_id")
        if not (isinstance(pass_node, ast.Constant) and pass_node.value in audit_passes):
            continue
        seen.add(pass_node.value)
        clamp = keywords.get("clamp_overlong_strings")
        # Either omitted (inherits the clamp=True default) or explicitly True.
        # Explicit False on an audit is the bug this test exists to prevent.
        assert clamp is None or (
            isinstance(clamp, ast.Constant) and clamp.value is True
        ), (
            f"Codex audit pass {pass_node.value} is fail-closed on string "
            f"length. An audit may improve a story, it may never fail one."
        )

    assert seen == audit_passes, f"missing Codex audit passes: {audit_passes - seen}"


@pytest.mark.parametrize(
    "module_name,root_name,facts_key,entities_key,numbers_key,fact_def",
    (
        ("nodes._otr_scifi_codex", "FactIndexV4", "facts", "entities", "numbers", "FactV4"),
    ),
)
def test_source_grounded_p0_has_a_finite_shared_output_envelope(
    module_name, root_name, facts_key, entities_key, numbers_key, fact_def,
):
    """BUG-11.50: a capped P0 needs a schema-bounded artifact surface."""
    module = importlib.import_module(module_name)
    root_schema = getattr(module, root_name).model_json_schema()
    properties = root_schema["properties"]
    assert properties[facts_key]["maxItems"] == 6
    assert properties[entities_key]["maxItems"] == 4
    assert properties[numbers_key]["maxItems"] == 4
    assert properties["tone"]["minLength"] == 1
    assert properties["tone"]["maxLength"] == 80
    assert module.SourceSpanV4.model_json_schema()["properties"]["quote"]["maxLength"] == 240
    fact_schema = root_schema["$defs"][fact_def]
    assert fact_schema["properties"]["source_spans"]["maxItems"] == 1
    assert fact_schema["properties"]["claim"]["maxLength"] == 240
    assert module.p0_output_token_budget() == 2800


def test_no_lane_seam_treats_the_word_target_as_a_quota():
    """The requested word count is a SCALE REQUEST, not a quota.

    Live kill (2026-07-11, Gemini): the scene-critique seam ordered the critic to
    "Ensure the total word count of the lines equals the scene's target word limit."
    At 30 words over 6 beats that is ~5 words a beat -- exact equality is
    unreachable, so the critic failed the scene, the bounded rewrite missed too, and
    the run died with "scene scene_01 failed its bounded rewrite". The model was
    obeying us.

    Operator law: word count never causes trimming, padding, culling, or a rewrite.
    It is a statistic recorded after the fact. Guard the seams so it cannot creep
    back in as a gate.
    """
    import glob
    import json as _json
    import re

    repo = pathlib.Path(__file__).resolve().parent.parent
    # A sentence that talks about the word target AND commands an exact match is a
    # quota. A sentence that explicitly frames it as advisory is the cure, not the
    # disease -- so it is not an offender even though it says the word "quota".
    mentions = re.compile(r"word count|word target|target word", re.IGNORECASE)
    commands = re.compile(
        r"\b(must equal|equals|meet it exactly|match(es)? exactly|must match)\b",
        re.IGNORECASE,
    )
    cures = re.compile(r"advisory|not a quota|never pad|never fail", re.IGNORECASE)

    offenders = []
    for path in glob.glob(str(repo / "nodes" / "story_packs" / "scifi_*" / "*.json")):
        # The Codex pack is owned by a concurrent workstream; guard the lanes we own.
        if "scifi_codex" in path:
            continue
        pack = _json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        for seam, text in (pack.get("prompt_stages") or {}).items():
            if not isinstance(text, str):
                continue
            for sentence in re.split(r"(?<=\.)\s+|\n", text):
                if (
                    mentions.search(sentence)
                    and commands.search(sentence)
                    and not cures.search(sentence)
                ):
                    offenders.append(
                        f"{pathlib.Path(path).name}::{seam}: {sentence.strip()[:120]}"
                    )
    assert not offenders, (
        "a lane seam is treating the advisory word target as a hard quota:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------- #
# The guard below exists because the lessons were NOT automatically carried
# from one lane to the next. Codex and Sonnet were each written
# independently, so each one independently re-made the same mistakes: a seam that
# asks for a field its schema forbids, and Python putting words in a character's
# mouth. A lesson only survives as an executable guard that runs across ALL lanes.
# --------------------------------------------------------------------------- #

LANE_SOURCES = (
    "nodes/_otr_scifi_codex.py",
)


@pytest.mark.parametrize("source", LANE_SOURCES)
def test_python_never_puts_words_in_a_characters_mouth(source):
    """No lane may hand a spoken-text field a string Python wrote.

    Live (Sonnet): the Warden's closing ruling was hardcoded as
    `DraftLineV4(text="The record holds now.", ...)` -- Python speaking for a
    character. It did not even need to: RewriteResultV4 already carried
    `vesh_resolution`, the model's own line, and the lane was throwing it away.

    A literal assigned to `text=` is dialogue no model authored. If a line must be
    spoken, a model field must supply it. Python judges; the LLM writes.
    """
    import ast
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    tree = ast.parse((repo / source).read_text(encoding="utf-8"))

    spoken = {"text", "premise", "title", "claim", "spoken_claim"}
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg not in spoken:
                continue
            value = kw.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.strip():
                offenders.append(
                    f"{source}:{value.lineno}: {kw.arg}={value.value!r} -- Python authored this"
                )
            if isinstance(value, ast.JoinedStr):  # an f-string built into spoken text
                offenders.append(
                    f"{source}:{value.lineno}: {kw.arg}=<f-string> -- Python composed this"
                )
    assert not offenders, (
        "Python is authoring story text; a model field must supply it:\n  "
        + "\n  ".join(offenders)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
