# OTR Roadmap Ideas (parked, not scheduled)

Ideas the operator wants captured but not built yet. Each is a seed, not a spec.

## Stage-direction beats -> a dedicated non-voice media engine
**Parked 2026-07-03** (from the no-fallbacks rip judgment call).

Context: the ledger can carry "stage-direction-only" beats — a line like
"(flips the switch)" that cleans to empty text with no spoken words. Today the
voice path emitted 0.3s of silence for these; the no-fallbacks rip removed that
(it now fails loud so silence never silently ships and such lines can't creep
into dialogue).

The idea: instead of silence OR a hard fail, route a stage-direction beat to a
NEW media engine that renders the action visually while the audio bed continues —
options floated: an overlay video, a procgen clip, a 3D-model beat, or a still.
Tricky because it needs its own ledger role + a real consumer.

Until then: stage-direction-only beats are NOT generated (the writer should never
emit them; the voice gate fails loud if one appears). Re-add the role to the
ledger when this media engine exists.
