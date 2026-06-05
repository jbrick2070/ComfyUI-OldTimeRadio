# OTR Cast System — Go-Forward Sprint Plan
**2026-05-30 · branch `v2.0-alpha` · execution doc (drives code→wire→regress→commit until all sprints green)**

## Decision
Python owns cast structure; the LLM only names + textures; a validator referees; deterministic
pool is the fallback, not the creative center. Ship the deterministic layer first (Phase 1), then
layer the schema-locked LLM system on top (Phase 2). `OTR_NAME_MODE=pool` reproduces today's
behavior exactly; `OTR_NAME_MODE=llm_slot_fill` adds episode-specific life with a guaranteed backstop.

## Hard constraints (every sprint respects these — non-negotiable)
- **C7 byte-identity.** A known-coherent historical seed must yield the *identical* cast and identical
  audio hash in `pool` mode. If a change breaks C7, it is wrong.
- **One `rng.choice` per slot.** `python_assign_voice_preset` draws exactly once per slot (`:785`).
  Gender/timbre/age filtering uses **no** RNG. Adding axes must not change draw count or order.
- **Isolated RNG for repair/naming.** Any name swap uses `random.Random(f"{cast_seed}:{char_id}")`.
  The main `cast_rng` sequence is never perturbed.
- **Name is decided before the script exists.** Cast is frozen *before* the writer generates dialogue,
  so a fallback swap can never desync a line that already used the name. (Two LLM passes, not one.)
- **No LLM retry.** Validation failure → deterministic pool repair. Terminates, stays reproducible.
- **Env-only config.** No new ComfyUI widgets. All knobs are env vars.
- **One file = one agent per wave.** Parallelism is along file-ownership boundaries (see §Waves).

## Frozen interface contracts — defined ONCE in S0, then never renegotiated
> This is what makes the subagent parallelism safe. Agents build against these stubs, not against
> each other's live code. Freeze them in S0 before any wave starts.

```python
# config/cast_pools.py
FIRST_NAMES_BY_GENDER: dict[str, list[str]]            # keys: "male" | "female" | "unisex"
FIRST_NAMES: list[str]                                  # compat alias = flat union, ORDER UNCHANGED
FIRST_NAMES_BY_GENRE: dict[str, dict[str, list[str]]]   # genre -> gender -> names
def gender_of_first_name(name: str) -> str:             # -> "male"|"female"|"unisex"|"unknown"
```

```text
# env vars (all optional; defaults reproduce current behavior)
OTR_NAME_MODE            = pool | llm_slot_fill          (default: pool)
OTR_CAST_GENRE           = scifi_1950s | noir | space_opera | auto   (default: auto)
OTR_NAME_CROSS_GENDER_RATE = 0.0                         (float; 0.0 = strict repair)
OTR_OTHER_NAME_POLICY    = unisex | all                  (default: unisex)
```

```jsonc
// CastPlanner slot — Python -> writer. IMMUTABLE. The LLM may not alter these.
{ "char_id": "C02", "gender": "female", "age_band": "middle_adult",
  "voice_preset": "fixed_voice_id", "dramatic_role": "skeptical mission doctor" }

// LLM Pass-1 return — LLM -> validator. NAME + TEXTURE ONLY.
{ "char_id": "C02", "name": "Dr. Mara Venn",
  "one_line_presence": "Precise, tired, morally alert.",
  "dialogue_style": "short clinical sentences with buried warmth" }
```
- `age_band` enum: `young_adult | middle_adult | older_adult | ageless`
- Validator **rejects**: any key outside the Pass-1 set; any unknown/missing `char_id`; wrong count;
  duplicate names; any attempt to carry gender/voice/role back from the LLM (not accepted at all).

## The per-sprint loop
```text
per sprint (on its own branch):
  code     implement against frozen S0 contracts
  wire     connect into node/graph/flow + read env var(s)
  regress  ./test_cast.sh  →  MUST be green (golden invariants + this sprint's new tests)
  commit   only when regress is green; one commit per sprint, message = "S<n>: <goal>"

per wave:
  merge all wave branches  →  run FULL ./test_cast.sh on merged main  →  gate
  next wave starts only on a green bar. never build on red.
```

## Regression suite — the green bar (`./test_cast.sh`)
- [ ] **R1 determinism** — same `OTR_CAST_SEED` twice → identical cast.
- [ ] **R2 C7 byte-identity** — known-coherent seed → identical cast + identical audio hash vs `golden/` (pool mode).
- [ ] **R3 coherence** — every binary-gender slot's name tag == slot gender at `CROSS_GENDER_RATE=0.0`.
- [ ] **R4 quota** — 40/40/20 gender split unchanged.
- [ ] **R5 voice-uniqueness** — existing invariant still passes.
- [ ] **R6 rng-draw count** — assert exactly one `rng.choice` per slot in `python_assign_voice_preset`.
- [ ] **R7 schema conformance** — Pass-1 output: allowed keys only, correct `char_id`s, correct count, no dup names (llm mode).
- [ ] **R8 fallback** — forced LLM failure → deterministic pool repair, terminates, zero retries.
- [ ] **R9 freeze-order** — assert cast frozen before writer Pass-2; no name string mutated post-script.

## Sprint grid

| ID | Phase | Goal | Primary file (owner) | Depends | Wave | New tests |
|----|-------|------|----------------------|---------|------|-----------|
| **S0** | 0 | Freeze all contracts + golden baseline + harness | `test_cast.sh`, `golden/` (Lead) | — | W0 (solo) | R1–R6 scaffolds |
| **S1** | 1 | Tag pools by gender, add genre buckets, `gender_of_first_name` | `config/cast_pools.py` (A) | S0 | W1 | R3 tagger |
| **S2** | 1 | Deterministic name-repair after gender-bound loop (`:601-608`) + `CROSS_GENDER_RATE` | `nodes/_otr_casting.py` (B) | S0 (S1 stub) | W1 | R2, R3 |
| **S3** | 1 | Unload writer LLM after script, **before** TTS phase (orders 3–8), not at order 17 | graph / node lifecycle (C) | S0 | W1 | VRAM smoke |
| — | — | **PHASE 1 SHIPS** — merge W1 → full regress → tag `v2.0` | | | gate | R1–R6 green |
| **S4** | 2 | CastPlanner: build immutable slots (role/gender/age/voice) | `nodes/_otr_castplanner.py` **(new, D)** | S1 | W2 | slot schema |
| **S5** | 2 | Voice mapping `(gender × age)`, **keep one `rng.choice`/slot** (`:785`) | `nodes/_otr_casting.py` (B) | S0 (S4 schema) | W2 | R6 |
| **S6** | 2 | Pass-1 LLM name+texture call, schema-locked, isolated-RNG | `OTR_LedgerScriptWriter` (A) | S0 (S4 schema) | W2 | R7 |
| **S7** | 2 | Validator + fallback routing → S2 repair (no retry) | `nodes/_otr_cast_validator.py` **(new, E)** | S1 (S6 schema) | W2 | R7, R8 |
| **S8** | 2 | Pass-2 writer vs **frozen** cast + texture conditioning | `OTR_LedgerScriptWriter` (A) | S6, S7 | W3 (solo) | R9 |
| **S9** | 2 | Full mode-matrix regress (`pool` + `llm_slot_fill`) + sign-off | `test_cast.sh` (Lead) | all | W3 (solo) | R1–R9 |

## Parallel execution — subagent waves
Dependencies are satisfied by the **frozen S0 contracts**, so logically-downstream sprints still build
in parallel: they code against the schema, not the implementation. Integration is proven in W3.

```text
W0  [solo · Lead]   S0  freeze contracts + golden baseline
                          │  (nothing proceeds until contracts are frozen — this is what de-risks ∥)
                          ▼
W1  [3 agents ∥]    A: S1  cast_pools.py        ┐
                    B: S2  _otr_casting.py repair │  disjoint files · all vs S0 contracts
                    C: S3  VRAM unload (lifecycle)┘
                          │  merge W1 → full regress → ── PHASE 1 SHIPS (tag v2.0) ──
                          ▼
W2  [4 agents ∥]    D: S4  _otr_castplanner.py  (new) ┐
                    E: S7  _otr_cast_validator.py (new)│  2 new files = zero-conflict
                    A: S6  writer Pass-1               │  2 shared files, 1 owner each
                    B: S5  voice × age (_otr_casting)  ┘
                          │  merge W2 → full regress → gate
                          ▼
W3  [solo · Lead]   S8  Pass-2 integration  →  S9  mode-matrix regress + sign-off
```

**Why this ordering parallelizes cleanly**
- New-file sprints (S4 CastPlanner, S7 Validator) can never collide — assign them freely.
- Shared edit-files have exactly one owner per wave: `cast_pools.py`→A, `_otr_casting.py`→B,
  `OTR_LedgerScriptWriter`→A. No two agents touch the same file concurrently.
- S5/S6/S7 logically depend on S4's slots, but S4's **output schema is frozen in S0**, so they build
  against the contract in parallel; S8 (W3) is where the real wiring meets and gets regressed.
- The `gender_of_first_name` function is the single shared primitive — it tags the pool (S1) *and*
  referees LLM output (S7). Both consume the same S0 contract; neither blocks the other.

## Subagent rules (read before dispatching)
1. **One file, one agent, per wave.** If two sprints touch the same file, they go in different waves
   or merge into one agent's task. The grid's "owner" column is the assignment.
2. **Build against frozen contracts + stubs**, never against another agent's live branch.
3. Each agent runs its own `code → wire → regress` on its branch and must be green before merge.
4. **Merge the whole wave, then run full `./test_cast.sh` on merged main.** Never start the next wave
   on a red bar. The regress gate is the serialization point; everything else is parallel.
5. New-file sprints are always safe to parallelize; shared-file sprints are the only constraint.

## Open decision still owned by Jeffrey (set before S6)
`OTR_NAME_CROSS_GENDER_RATE` semantics — pick the lane, because the validator contract depends on it:
- **Strict (rate 0.0):** validator auto-repairs every name/gender mismatch; LLM is a flavor generator
  on a leash. Simplest, ships clean.
- **LLM owns intent (rate > 0):** validator only catches hard errors (dupes, wrong count, schema), and
  the LLM is allowed deliberate non-stereotypical names. Requires the LLM to *declare* intent so a
  deliberate mismatch isn't mistaken for an error — heavier contract, defer unless you want it for v1.

Default ships strict; flip later without touching the structure.

*Sits alongside `…__{01_chatgpt,02_gemini,03_nvidia,05_claude_synthesis}.md`. Rename to `__06_*` if you want it in the round-robin sequence.*
