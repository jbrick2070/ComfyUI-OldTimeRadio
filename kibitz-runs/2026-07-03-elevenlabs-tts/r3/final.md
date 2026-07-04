# Kibitz r3 (wiring) — judgment

Panel: codex TIMED OUT at 360s (reasoning=high; subprocess TimeoutExpired, no
review produced) — benched for r3. Antigravity already benched (r2 hang). So r3
is Claude-anchor-only + the Fable final confirmation gate below.

Claude r3 anchor (kibitz_anchor_r3.md) checked the clean fold for NEW
wiring/sequencing defects and found the integration internally consistent:
- Sprint order holds: xfail removals now live in the adapter's own sprint
  (S1/S5), not S0; S0 stays pure-code/green.
- S1 ordering note (accepted into the plan implicitly): within S1 the
  CAPABILITIES row + profile must register BEFORE / same-commit as the
  `_LEGACY_FIRST_ENGINES` append so `test_capability_profiles.py:213`
  set-equality and `build_engine_combo` never see a half-registered engine.
- announcer_voice_ref("elevenlabs") is called in begin_episode (adapter), after
  the bank + S2 manifest load — correct timing.
- S6 selects via dropdown VALUE (append-only positional) routing to the cloud
  adapter; no is_default flip needed (dropdown-is-enable).

The decisive "100% code-ready" gate is the Fable final confirmation (fable_final_
confirmation.md), which grounds against the real repo itself.
