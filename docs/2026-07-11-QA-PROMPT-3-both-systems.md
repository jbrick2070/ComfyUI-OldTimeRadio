# OTR QA SWEEP #3 -- paste into agy AND into codex

REVIEWER ONLY. Read anything; do NOT edit source, do NOT git add/commit/push.
Write to `qa3_<yourname>.md` in the repo root and stop.
Pull first -- HEAD moves every few minutes. Label every claim CONFIRMED (you opened it /
ran the number) or [ASSUMPTION]. Show arithmetic. Retract anything you got wrong.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha

## Scorecard -- you both earned this one

You independently converged on the same Sonnet design, and it matched what I had already
written: merge corrections by `line_ref`, cap cites at 3 (do not widen the line schema),
and settle facts as ONE-BASED with `cites=[] + non_fact=True` for lines that state no
fact. Two agents and a judge agreeing from three directions is the strongest signal we
have had all day.

And the `fact_0` finding was bigger than either of you framed it. `DraftLineV4.cites` had
`min_length=1`, so a line that states no fact COULD NOT VALIDATE -- and the lane satisfied
the schema by citing a sentinel `fact_0` that the one-based P0 contract can never
produce. Every ceremonial line in the episode carried a FALSE citation, and the seal and
sign-off borrowed the attestation's real fact id for a claim they never make. The schema
was forcing the lane to lie. Now fixed: a line that cites nothing is allowed to cite
nothing, and must say so.

I also found, while implementing your plan, something neither of you flagged:
`RewriteResultV4` already carries `vesh_resolution` -- the Warden's on-air acknowledgment
-- and the lane was THROWING IT AWAY and hardcoding `text="The record holds now."` in
Python. Python was speaking for a character while the model's line sat unused. That is a
direct violation of the law, and it was invisible to both of you.

**Standing lesson: hunt for Python that AUTHORS. A hardcoded string in a `text=` field is
the worst bug class in this codebase, and it hides in plain sight.**

## JOB 1 -- attack my Sonnet implementation (adversarial)

Read `nodes/_otr_scifi_sonnet.py` at HEAD: `DraftLineV4` (+ its `non_fact` validator),
`_audited_line_indices`, `_apply_rewrite_corrections`, the P5 loop, and the P6 event
assembly. Try to break it:
- `_audited_line_indices` excludes `non_fact` lines so the model's `line_ref` matches the
  list it was actually shown. Is that index stable across BOTH rewrite rounds, given the
  Warden lines appended between rounds? Walk round 0 and round 1 concretely.
- `_apply_rewrite_corrections` rejects out-of-range, duplicate, and unflagged refs. Is
  rejecting an UNFLAGGED correction right, or should the doctor be allowed to fix a line
  the audit missed? Argue it.
- A corrected line is forced to `non_fact=False` and must therefore cite >= 1. Can the
  doctor legitimately need to turn a factual line into a non-factual one? What breaks?
- Does anything downstream (ledger assembly, coverage, freeze cascade, the fact-coverage
  math) still assume every line has >= 1 cite, now that ceremonial lines have none?
  **This is the highest-risk question in this document.** Grep for consumers of `cites`.

## JOB 2 -- the 720-word gate: CONVERGE and give me the exact edit set

You now agree `resolve_context_cap` is the LIVE path and `compute_effective_context_limit`
is DEAD CODE (that is also a hit for the standing dead-code rip). Finish the job:
1. The exact minimal edit set to make the effective writer cap 16384 -- every file:line,
   including the per-row `CuratedModel.context_window` AND `CURATED_CONTEXT_OVERRIDES`
   AND `HARD_VRAM_CONTEXT_LIMIT`, and which of them actually bind. Plus every test that
   pins the current 8192 and what it asserts.
2. You disagree on VRAM. agy says +1.25 GiB KV at 16k (NF4 double-quant, full GPU
   device_map); earlier numbers said ~3.1 GiB. Settle it: read HOW the loader actually
   loads Mistral-Nemo (quantization config, device_map, dtype), state the KV formula with
   the model's real config (layers, kv heads, head_dim), and give the number at 8k/16k/24k
   with a GO/NO-GO on a 16 GB RTX 5080 alongside the rest of the pipeline.
3. The 8192 default reportedly pins an audio byte-identity baseline. Prove or disprove
   that an env-var opt-in leaves it untouched -- name the test.
4. Which passes set `prompt_must_fit=True`, and what is the blast radius when a 720w
   prompt then fails LOUD instead of silently truncating? Is there any pass where a loud
   failure is worse than a degraded output? Argue it.

## JOB 3 -- the dead-code rip (standing operator directive)

Dead levers cost live rolls: a reviewer patches the dead one. `compute_effective_context_limit`
is the case study. Give me the inventory for the model/loader/context family only:
`_otr_model_catalog.py`, `_otr_loader_backends.py`, `_otr_model_loader.py`,
`_otr_model_runtime.py`.
For each dead item: file:line, what it was for, what superseded it, and whether a test
pins it -- and whether that test asserts RUNTIME BEHAVIOR or merely the value of a
constant (a test that only pins a dead constant dies with it).
Flag every place where TWO levers exist for one behavior and neither is marked live.

## JOB 4 -- hunt for Python that authors

The `vesh_resolution` bug is a class, not an incident. Sweep ALL FOUR lanes (codex,
gemini, sonnet, fable2) plus the shared writer tail for Python that WRITES story text
rather than judging it:
- any literal string assigned to a `text=` / dialogue / premise / title / description
  field
- any f-string or `.join()` / concatenation that builds spoken text
- any place Python trims, pads, truncates, or reflows a line
- any model field that IS authored but gets overwritten, defaulted, or dropped by Python
Rank by "would a listener hear it". Each with file:line and the model field that SHOULD
be supplying it.

## Output (`qa3_<yourname>.md`)

JOB 1 ATTACKS ON THE SONNET FIX (esp. downstream consumers of `cites`)
JOB 2 720W EDIT SET + settled VRAM number + GO/NO-GO
JOB 3 DEAD-CODE INVENTORY
JOB 4 PYTHON-THAT-AUTHORS, ranked
CONFIDENCE on every line.
