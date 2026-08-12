"""PBUG-20260812-02 -- a pydantic field name that shadows a BaseModel attribute.

FOUND BY A LIVE RUN, not by review: the first leg of the 45-word
every-visual-path campaign (2026-08-12) died in `OTR_LedgerScriptWriter` with
`TypeError: Object of type method is not JSON serializable`, 78 s in, before any
video work. The message names neither the model nor the field.

THE MECHANISM -- and it is worse than the first diagnosis recorded. Pydantic's
`ModelMetaclass` inherits `ABCMeta`, so `register` resolves on `BaseModel` as a
bound method. `CastShape` declared `register: str` with no default, which reads
as "required". **Pydantic instead took the inherited attribute as the field's
DEFAULT**, and three things followed silently:

* the field went OPTIONAL;
* the JSON SCHEMA HANDED TO THE WRITER stopped listing `register` in
  `required`, so the model was never obliged to produce a documented,
  load-bearing contract field (doc s5: HOW a character speaks);
* any cast shape that omitted it carried a bound method -- which reaches the
  prompt text as `register: <bound method ModelMetaclass.register of ...>` and
  kills the node on the next `json.dumps`.

So the crash was the *lucky* outcome. The quiet one is a writer prompt carrying
a repr of a method where a character's speaking register belongs. The first
diagnosis claimed only `model_construct` could trigger it; in fact ORDINARY
VALIDATION did, which is why it reached production.

THE FIX IS AT THE FIELD: `register: str = Field(...)`. The name is contract
vocabulary and cannot change, so the class body shadows the inherited attribute
with an explicit required marker. That restores exactly what the bare
annotation was always meant to say.

THESE TESTS ARE THE REASON THE FIX CANNOT SILENTLY REGRESS. Deleting the
`Field(...)` restores a *syntactically fine* `register: str` -- the defect has
no visible symptom in the source, so only an executable check catches it.
"""
from __future__ import annotations

import importlib
import inspect
import json
import pkgutil

import pytest
from pydantic import BaseModel

from nodes import _otr_scifi_fable2 as fable2


#: The writer models on the path that failed live.
WRITER_MODELS = (fable2.Treatment, fable2.CastShape)


def shadowing_fields(model_cls):
    """Field names on ``model_cls`` that also resolve on ``BaseModel``.

    `hasattr(BaseModel, name)` rather than a hand-list, so it catches metaclass
    attributes (`register`) as well as BaseModel's own (`copy`, `json`,
    `schema`, `dict`) -- and keeps catching whatever a future pydantic adds.

    A hit is NOT a defect by itself: `CastShape.register` is still a legitimate
    hit and always will be, because the name is contract vocabulary. What makes
    a hit safe is that the field declares its own default (or `Field(...)`), so
    the inherited attribute never becomes the default. That is the invariant
    `test_no_pydantic_field_defaults_to_a_non_serializable_value` enforces.
    """
    return sorted(n for n in model_cls.model_fields if hasattr(BaseModel, n))


# ---------------------------------------------------------------------------
# THE FIX -- the field the live failure came through
# ---------------------------------------------------------------------------
def test_register_is_REQUIRED_and_has_no_inherited_default():
    """The whole defect in one assertion: `is_required()` was False."""
    field = fable2.CastShape.model_fields["register"]
    assert field.is_required(), (
        "CastShape.register has gone optional again -- the `Field(...)` was "
        "removed and pydantic is using the inherited ModelMetaclass.register "
        "as the default. The writer will emit a bound method into a prompt.")


def test_the_schema_handed_to_the_writer_REQUIRES_register():
    """The quiet half of the defect. The crash was survivable; a contract
    field the model is never asked for is a corrupted script."""
    schema = fable2.CastShape.model_json_schema()
    assert "register" in schema["required"], (
        "the writer's structured-output schema no longer demands `register`, "
        "so the model may omit a documented load-bearing field")


def test_omitting_register_now_REFUSES_instead_of_carrying_a_method():
    """Honest and retryable beats silent and wrong: a ValidationError goes to
    the repair ladder, a bound method goes into the episode."""
    with pytest.raises(Exception) as caught:
        fable2.CastShape(name="Ada", role="lead", want="w", pressure="p")
    assert "register" in str(caught.value)


def test_a_validated_cast_shape_serializes_cleanly():
    """The normal path, which was always fine -- which is exactly why this hid
    for as long as it did."""
    shape = fable2.CastShape(name="Ada", role="lead", want="w", pressure="p",
                             register="dry")
    dumped = shape.model_dump()
    assert dumped["register"] == "dry"
    assert json.loads(json.dumps(dumped))["register"] == "dry"


def test_the_contract_field_is_still_spelled_register():
    """The fix must not have renamed it. `register` is prompt vocabulary the
    writer is instructed in (doc s5); renaming the field would silently change
    the contract the model is answering."""
    assert "register" in fable2.CastShape.model_fields
    assert "register" in fable2.CastShape.model_json_schema()["properties"]


def test_model_construct_can_no_longer_hand_back_a_bound_method():
    """`model_construct` skips validation, so it is the last way a shape could
    reach `model_dump` without the field. With no default there is nothing to
    fall through TO, so the attribute is simply absent -- loud, not silent."""
    shape = fable2.CastShape.model_construct(
        name="Ada", role="lead", want="w", pressure="p")
    with pytest.raises(AttributeError):
        shape.register


# ---------------------------------------------------------------------------
# THE GENERAL RULE -- the check that would have caught this the day it landed
# ---------------------------------------------------------------------------
def all_pydantic_models():
    """Every pydantic model reachable under `nodes/`, deduplicated.

    Whole-package rather than a hand-list: the defect is a naming accident, so
    the next instance will be in whichever module nobody thought to enumerate.
    Modules that fail to import are skipped -- import health is other suites'
    job, and this one must not go red for an unrelated reason.
    """
    import nodes as _nodes

    seen, models = set(), []
    for found in pkgutil.walk_packages(_nodes.__path__, "nodes."):
        try:
            module = importlib.import_module(found.name)
        except Exception:
            continue
        for obj in vars(module).values():
            if not (inspect.isclass(obj) and issubclass(obj, BaseModel)
                    and obj is not BaseModel):
                continue
            key = (obj.__module__, obj.__name__)
            if key not in seen:
                seen.add(key)
                models.append(obj)
    return models


def test_the_sweep_actually_sees_the_models_it_claims_to_check():
    """Guards the rule below against becoming vacuous. A broken import or a
    renamed package would make the sweep pass by finding nothing."""
    models = all_pydantic_models()
    assert len(models) > 50, "the model sweep collapsed -- it found %d" % len(models)
    assert fable2.CastShape in models


def test_no_pydantic_field_defaults_to_a_non_serializable_value():
    """THE RULE. Every field default under `nodes/` must survive `json.dumps`.

    Stated as the CLASS of defect rather than as `register`, because the danger
    is not the word: it is any field whose name happens to collide with a
    pydantic/metaclass attribute, which then becomes a live default nobody
    wrote. `copy`, `json`, `schema`, `dict`, `validate` and `construct` are all
    waiting to do the same thing.
    """
    offenders = []
    for model in all_pydantic_models():
        for name, field in model.model_fields.items():
            default = field.default
            if default is None or repr(default) == "PydanticUndefined":
                continue
            try:
                json.dumps(default)
            except TypeError:
                offenders.append("%s.%s.%s = %r"
                                 % (model.__module__, model.__name__, name,
                                    default))
    assert not offenders, (
        "these pydantic fields default to something that cannot be serialized "
        "-- almost certainly an inherited attribute pydantic adopted because "
        "the field name collides with it: %s" % offenders)


def test_the_known_shadowed_name_is_still_only_CastShape_register():
    """Scopes the collision itself, separately from the default rule above. A
    NEW shadowing field is worth a human look even when it is safely
    defaulted."""
    assert shadowing_fields(fable2.Treatment) == []
    assert shadowing_fields(fable2.CastShape) == ["register"]


def test_the_shadowed_name_arrives_via_the_METACLASS_not_the_class_body():
    """Why pydantic never refused the field, and why a `vars(BaseModel)` check
    would have missed it: `register` comes from `ABCMeta`."""
    assert hasattr(BaseModel, "register")
    assert "register" not in vars(BaseModel)
    assert callable(BaseModel.register)
