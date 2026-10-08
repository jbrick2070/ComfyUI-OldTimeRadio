"""The writer's schema request scans the HF cache once, not once per widget.

``creative_writing_model`` and ``technical_model`` offer the same list, and
each called ``dropdown_choices()`` -- a live walk of the local HF hub cache --
so every INPUT_TYPES request paid for two identical scans.
"""
from nodes import _otr_model_catalog as cat
from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W


def _widget(spec, name):
    for section in ("required", "optional"):
        if name in spec.get(section, {}):
            return spec[section][name]
    raise KeyError(name)


def test_one_scan_per_request_with_independent_copies(monkeypatch):
    calls = []
    real = cat.dropdown_choices

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cat, "dropdown_choices", counted)
    spec = W.INPUT_TYPES()
    assert len(calls) == 1
    creative = _widget(spec, "creative_writing_model")[0]
    technical = _widget(spec, "technical_model")[0]
    assert list(creative) == list(technical) == list(real())
    assert creative is not technical


def test_each_request_still_refreshes(monkeypatch):
    calls = []
    real = cat.dropdown_choices
    monkeypatch.setattr(cat, "dropdown_choices",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    W.INPUT_TYPES()
    W.INPUT_TYPES()
    assert len(calls) == 2
