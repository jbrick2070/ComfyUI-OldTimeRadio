#!/usr/bin/env bash
# test_cast.sh -- cast name<->gender<->voice coherence regression harness (S0).
#
# Wraps the R1-R9 invariants (tests/test_cast_invariants.py) + the cast-relevant
# unit suite + the audio-baseline integrity check. Green here is the per-sprint
# loop gate AND the wave-merge gate. The known-fail guard in tests/conftest.py
# hard-exits 2 on ANY failure in the collected set, so a clean run exits 0.
#
# Usage (from repo root):
#   ./test_cast.sh                  # full cast bar
#   ./test_cast.sh -k something     # extra pytest args are forwarded
#   OTR_NAME_MODE=llm_slot_fill ./test_cast.sh    # mode-matrix (S9)
#
# Override the interpreter with OTR_PY if your venv lives elsewhere.
set -u
cd "$(dirname "$0")"

PY="${OTR_PY:-C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe}"

exec "$PY" -m pytest \
  tests/test_cast_invariants.py \
  tests/test_otr_casting.py \
  tests/test_lock_cast_routing.py \
  tests/test_writer_cast_lock_voice_preset.py \
  tests/test_writer_paired_wiring.py \
  tests/test_helper_paired_signatures.py \
  tests/test_meta_slot_transitions.py \
  tests/test_voice_resolver.py \
  tests/test_narrative_face.py \
  tests/test_freeze_cascade_g6.py \
  tests/test_audio_byte_identical.py \
  -q -p no:cacheprovider "$@"
