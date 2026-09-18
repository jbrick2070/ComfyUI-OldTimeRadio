"""razzle_ltx_8gb -- local LTX 0.9.8 i2v with the raised razzle motion clause.

ADDITIVE. ``ltx_8gb`` stays the generic 8 GB i2v row. This sibling shares the
whole 0.9.8 substrate (canvas, frame ladder, T5, session identity, weights)
and overrides only prompt composition. Image engine stays independent: this
adapter consumes the still it is handed.

A hold belongs on still_word / still_flat, not here.
"""
from __future__ import annotations

from . import eng_ltx_8gb as _LX
from . import razzle_prompt as _RP
from .registry import register


@register
class RazzleLtx8gbEngine(_LX.Ltx8gbEngine):
    """Local razzle: animate a still with a full decisive action on ltx_8gb."""

    name = "razzle_ltx_8gb"
    default_roles = ()

    def _compose_positive(self, request) -> str:
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        return _RP.compose_positive(get("text_prompt"))

    def _negative_prompt(self):
        """LTX recipe negative plus the shared razzle no-hold / artifact extras.

        Goes through ``_negative_prompt`` (not a parallel ``_negative_for``)
        so the prequal tripwire still compares one frozen string to itself.
        Session identity is unchanged -- it never carried the negative text.
        """
        return _RP.compose_negative(super()._negative_prompt())
