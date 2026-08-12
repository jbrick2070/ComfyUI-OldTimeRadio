"""PBUG-20260812-02 -- a Pydantic field name that shadows a BaseModel attribute.

FOUND BY A LIVE RUN, not by review: the first leg of the 45-word
every-visual-path campaign (2026-08-12) died in `OTR_LedgerScriptWriter` with
`TypeError: Object of type method is not JSON serializable`, 78 s in, before any
video work. The message names neither the model nor the field.

THE MECHANISM. Pydantic's `ModelMetaclass` inherits `ABCMeta`, so
`BaseModel.register` is a bound metaclass method. `CastShape` declares a FIELD
called `register`. Pydantic does not reject that -- the clash is on the
metaclass, not the class body -- and a fully validated instance is fine, because
the value lives in the instance `__dict__`. **But any instance whose `register`
is absent from `__dict__` falls through to the class attribute, and
`model_dump()` then hands back the bound method.** The next `json.dumps` dies.

TWO TESTS, DELIBERATELY DIFFERENT IN KIND:

* the CHARACTERIZATION test passes TODAY and pins the trap itself, so the
  knowledge is executable even while the defect is open;
* the CONTRACT test is `xfail(strict=True)` -- it states the rule the writer
  models should obey, fails today because `CastShape.register` breaks it, and
  FLIPS THE SUITE RED the moment someone fixes the field, which is the signal to
  delete the marker. Same progressive-ledger shape as the lane matrix's
  `EXPECTED_RED` and its strict unexpected-pass gate.

Fixing it is the writer lane's call -- rename (root) or default (containment) --
because both touch the structured-output contract the model is prompted against.
See `docs/PROD_BUG_LOG.md` PBUG-20260812-02.
"""
from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from nodes import _otr_scifi_fable2 as fable2


#: The writer models on the path that failed live.
WRITER_MODELS = (fable2.Treatment, fable2.CastShape)


def shadowing_fields(model_cls):
    """Field names on ``model_cls`` that also resolve on ``BaseModel``.

    The check is `hasattr(BaseModel, name)` rather than a hand-list, so it
    catches metaclass attributes (`register`) as well as BaseModel's own
    (`copy`, `json`, `schema`, `dict`) -- and it keeps catching whatever a
    future Pydantic adds.
    """
    return sorted(n for n in model_cls.model_fields if hasattr(BaseModel, n))


# ---------------------------------------------------------------------------
# CHARACTERIZATION -- passes today, and pins the trap
# ---------------------------------------------------------------------------
def test_a_validated_cast_shape_dumps_its_register_as_a_STRING():
    """The normal path is fine, which is exactly why this hid."""
    shape = fable2.CastShape(name="Ada", role="lead", want="w", pressure="p",
                             register="dry")
    dumped = shape.model_dump()
    assert dumped["register"] == "dry"
    json.dumps(dumped)          # serializes cleanly


def test_an_UNSET_register_leaks_a_BOUND_METHOD_into_model_dump():
    """THE DEFECT, reproduced. This is the live failure in three lines.

    `model_construct` skips validation and applies defaults -- and `register`
    has no default, so the field never enters `__dict__` and the dump falls
    through to `ModelMetaclass.register`.
    """
    shape = fable2.CastShape.model_construct(
        name="Ada", role="lead", want="w", pressure="p")
    dumped = shape.model_dump()
    assert callable(dumped["register"]), (
        "the trap is gone -- if `register` no longer resolves to a method, the "
        "field was renamed or given a default and the xfail below should now "
        "be failing too")
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(dumped)


def test_the_shadowed_name_is_on_the_METACLASS_not_the_class_body():
    """Why Pydantic did not refuse the field: it only guards its own class
    attributes, and `register` arrives via `ABCMeta`."""
    assert hasattr(BaseModel, "register")
    assert "register" not in vars(BaseModel)
    assert callable(BaseModel.register)


# ---------------------------------------------------------------------------
# THE CONTRACT -- strict xfail, so fixing the field turns the suite RED
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="PBUG-20260812-02: CastShape.register shadows BaseModel.register "
           "(via ABCMeta on Pydantic's metaclass), so an instance built "
           "without it dumps a bound method and the writer dies on json.dumps. "
           "OPEN: the fix is the writer lane's call (rename = root, default = "
           "containment) because both change the structured-output contract. "
           "When it is fixed this test PASSES and strict=True fails the suite, "
           "which is the signal to delete this marker.")
def test_no_writer_model_field_shadows_a_BaseModel_attribute():
    """The rule that would have caught this the day the field was added.

    Structural and general: it names the whole CLASS of defect rather than the
    one field, so a future model that adds `copy`, `json`, `schema` or `dict`
    is caught before it reaches a live render.
    """
    offenders = {cls.__name__: shadowing_fields(cls)
                 for cls in WRITER_MODELS if shadowing_fields(cls)}
    assert not offenders, (
        "these writer model fields shadow a BaseModel/metaclass attribute and "
        "will serialize as a bound method whenever the instance is built "
        "without them: %s" % offenders)


def test_the_rule_finds_exactly_the_known_offender_and_no_other():
    """Scopes the open defect, so a NEW shadowing field cannot hide behind the
    xfail above -- this one is green and would break on a second offender."""
    assert shadowing_fields(fable2.Treatment) == []
    assert shadowing_fields(fable2.CastShape) == ["register"]
