VERDICT: yes-with-verification. Operator explicitly wants clean audio. Retire synthesized noise and make the existing canonical audio node clean by default; preserve its real rate/channel/ledger work and public optional deterministic FX.

MUST-FIX
1. CONFIRMED: scene_sequencer.py:634-730 produces noise in CPU/CUDA branches and its sole production caller at 1193 adds it to supplied voices. Remove helper plus its state/caller/test patch as a feature retirement, not a dead-code claim.
2. CONFIRMED: AudioEnhance public optional effects have explicit settings. Keep schema identity/order; set six canonical values to clean, defaults likewise, and allow LPF zero in schema. Do not delete a registered useful rate/channel node or leave runtime noise for tests.
3. CONFIRMED: current fresh harness seeds actual noise. Clean implementation must replace that with unseeded actual silence/identity checks and preserve chirp placement proof. Old/new sound hashes intentionally differ.

SHOULD-FIX: independently challenge silent assembly behavior, zero-width meaning, no-op 48k identity and exact tests. Cap 15 primary files and list uncovered work. No extra workers, model upgrades, code edits, tests, servers or GPU. Current HEAD 1f27e4123a9567400ab39ef698d2a8a394601284.

This is a late bounded addition to the existing cleanup campaign. Same two lanes perform its R1-R3, then shared R4 with C1-C3; no claim earlier cleanup rounds reviewed it.
