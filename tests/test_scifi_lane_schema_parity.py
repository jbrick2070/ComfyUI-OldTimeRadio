"""Cross-lane structural guards for the sci-fi source banks.

These are CLASS-level regressions, not one-off fixes: each guard here pins a
defect that killed a live canonical roll on ONE lane and could equally kill the
others. Codex, Gemini, and Sonnet each hand a strict Pydantic model a blob of
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
import typing

import pytest
from pydantic import BaseModel

LANE_MODULES = (
    "nodes._otr_scifi_codex",
    "nodes._otr_scifi_gemini",
    "nodes._otr_scifi_sonnet",
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


def test_gemini_pitch_slate_still_pins_exactly_three_pitches():
    """The JSON-native fix must not loosen the three-pitch contract."""
    from nodes import _otr_scifi_gemini as lane

    field = lane.PitchSlateV4.model_fields["pitches"]
    constraints = {
        type(meta).__name__: getattr(meta, "min_length", None) or getattr(meta, "max_length", None)
        for meta in field.metadata
    }
    assert constraints, "pitches must stay length-pinned"

    def pitch(n):
        return {"premise": f"p{n}", "setting": f"s{n}", "tonal_palette": f"t{n}"}

    # Exactly three validates -- from a LIST, the only thing JSON can deliver.
    slate = lane.PitchSlateV4.model_validate({"pitches": [pitch(0), pitch(1), pitch(2)]})
    assert len(slate.pitches) == 3
    # Two or four still fail closed.
    for count in (2, 4):
        with pytest.raises(Exception):
            lane.PitchSlateV4.model_validate(
                {"pitches": [pitch(i) for i in range(count)]}
            )


@pytest.mark.parametrize(
    "module_name,invoke_name",
    (
        ("nodes._otr_scifi_gemini", "invoke_gemini_structured"),
        ("nodes._otr_scifi_sonnet", "invoke_sonnet_structured"),
    ),
)
def test_source_grounded_p0_refuses_to_be_left_truncated(module_name, invoke_name):
    """P0 carries the source payload: it must fail loud, not lose its prefix.

    Parity with the Codex lane, which already pins this. A provenance prompt that
    is silently left-truncated drops the system/schema prefix and yields a
    confidently wrong artifact instead of an honest failure.
    """
    import ast
    import pathlib

    source = pathlib.Path(module_name.replace(".", "/") + ".py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != invoke_name:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        pass_node = keywords.get("pass_id")
        if not (isinstance(pass_node, ast.Constant) and pass_node.value == "P0"):
            continue
        must_fit = keywords.get("prompt_must_fit")
        assert isinstance(must_fit, ast.Constant) and must_fit.value is True, (
            f"{module_name} P0 is source-grounded and must pass prompt_must_fit=True"
        )
        return
    pytest.fail(f"no P0 call found in {module_name}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
