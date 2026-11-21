"""Slot wrappers forward the schema-binding hook by re-wrapping.

planning/schema-binding/DECISION.md, option D, step 1. A local writer slot
exposes ``_otr_bind_schema``; the attempt-counting wrappers in the SciFi lane
and the four brief interpreters used to drop it, so a pass that bound behind
one would silently get the unbound slot. Forwarding must RE-WRAP, not copy:
``bind`` returns the raw scheduler closure, and a copied hook would let bound
calls skip the counter. Behaviour-neutral today -- no pass binds through these
wrappers yet. CPU only, no model. UTF-8 no BOM.
"""
from __future__ import annotations

import inspect

from pydantic import BaseModel

from nodes import _otr_structured_call as SC


class _Schema(BaseModel):
    text: str


class _LocalSlot:
    """A local-lane slot double: callable, with a binding hook whose bound
    closure records the schema it was bound to."""

    def __init__(self, label="unbound"):
        self.label = label
        self.calls = []

    def __call__(self, msgs, *, temperature, max_new_tokens):
        self.calls.append((msgs, temperature, max_new_tokens))
        return self.label

    def _otr_bind_schema(self, schema):
        bound = _LocalSlot(label="bound:" + schema.__name__)
        bound._otr_bound_schema_model = schema
        return bound


def test_the_helper_rebinds_through_the_wrapper():
    inner = _LocalSlot()
    counted = []

    def rewrap(fn):
        def wrapper(msgs, *, temperature, max_new_tokens):
            counted.append(1)
            return fn(msgs, temperature=temperature, max_new_tokens=max_new_tokens)
        return SC.forward_schema_binding(wrapper, fn, rewrap)

    outer = rewrap(inner)
    assert callable(getattr(outer, "_otr_bind_schema", None))
    bound = outer._otr_bind_schema(_Schema)
    assert bound("m", temperature=0.1, max_new_tokens=8) == "bound:_Schema"
    assert counted == [1], "a bound call must still pass through the wrapper"
    assert bound._otr_bound_schema_model is _Schema
    assert inner.calls == [], "the unbound slot is not called once bound"


def test_a_slot_without_the_hook_leaves_the_wrapper_alone():
    def remote(msgs, *, temperature, max_new_tokens):
        return "remote"

    def wrapper(msgs, *, temperature, max_new_tokens):
        return remote(msgs, temperature=temperature, max_new_tokens=max_new_tokens)

    assert SC.forward_schema_binding(wrapper, remote, lambda f: f) is wrapper
    assert not hasattr(wrapper, "_otr_bind_schema")
    assert not hasattr(wrapper, "_otr_bound_schema_model")


def test_scifi_counting_wrapper_counts_bound_calls_in_the_same_box():
    from nodes import _otr_scifi_news_pro as SNP
    fn, box = SNP._counting(_LocalSlot())
    fn("a", temperature=0.5, max_new_tokens=4)
    bound = fn._otr_bind_schema(_Schema)
    assert bound("b", temperature=0.5, max_new_tokens=4) == "bound:_Schema"
    assert box == {"calls": 2}
    assert bound._otr_bound_schema_model is _Schema


def test_every_brief_interpreter_wrapper_forwards_the_hook():
    """The four `_counting_slot_fn` closures live inside their builders, so
    the wiring is pinned at its real site."""
    from nodes import (_otr_media_archive_interpreter, _otr_public_domain_sources,
                       _otr_shakespeare_sources, news_interpreter)
    for module in (_otr_media_archive_interpreter, _otr_public_domain_sources,
                   _otr_shakespeare_sources, news_interpreter):
        src = inspect.getsource(module)
        assert "forward_schema_binding(_counting_slot_fn, slot_fn, _counted)" in src, (
            module.__name__)
        assert src.count("def _counting_slot_fn(") == 1, module.__name__
