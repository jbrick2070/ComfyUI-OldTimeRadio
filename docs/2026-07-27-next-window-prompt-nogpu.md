# NEXT WINDOW PROMPT -- NO GPU AVAILABLE

Paste the block at the bottom into a fresh window. Everything above is context
for a human. Two variants are provided: **A (recommended)** runs a kibitz arc to
settle an open architecture fork and then codes the winner; **B** skips the arc
and goes straight at the largest CPU-only coding block.

Both are **100% CPU**. Nothing here needs the GPU, and nothing here is the live
render slice (that is 7d, and it is explicitly OUT until a GPU is free).

---

## State

`HEAD == origin == 4b5e04fe` on `v2.0-alpha`. Suite **6913 passed / 27 skipped /
1 xfailed**. Canonical workflow `5377914B` byte-identical.

Multi-clip beat coverage: chunks **1-6 and 7a are DONE**, plus 7b slice 1.
Eleven QA rounds behind it. Read `docs/GO_FORWARD_PLAN.md` CURRENT STEP (
rewritten 2026-07-26 -- older text about "the first adapter opt-in" is dead) and
`docs/2026-07-26-chunk-7b-window-prompt.md` (the fifteen-variable env audit
table and the OPEN DECISION write-up).

**Nothing has rendered through this machine yet.** That is fine and expected --
7d is the live leg and it is not this window's job.

---

## THE OPEN FORK -- this is what variant A settles

`otr_shot_lock._stamp_coverage_plan` calls `frame_contract_for(engine)`, gets a
STATIC literal ceiling, partitions the beat, and mints one still per jump
segment. The renderer *then* reads the environment (and the capability profile)
and can get a DIFFERENT ceiling. Both halves are internally consistent, which is
exactly why the divergence is silent.

An audit found **fifteen** environment variables that can move a declared
contract fact. A blanket "refuse on disagreement" was built and **reverted**: it
broke six tests that assert an operator CAN lower a cap, which is documented,
production-used behaviour (`eng_ltx_8gb.py`'s docstring lists
`OTR_LTX_8GB_MAX_FRAMES` as a normal override for VRAM-constrained boxes, and
`test_floor_max_override_is_an_absolute_hard_cap` was written for the
2026-07-24 WAN 8GB launch contract after an 8GB leg died in the cost model).

**(A) REFUSE** -- env != declaration is terminal. Simple; literal reading of "it
either works or it fails". Costs the 8GB tier its knob, and cannot reach the
`OTR_ACTIVE_PROFILE` / `profile_max_render_frames()` case at all.

**(B) RESOLVE ONCE AT PLAN TIME AND STAMP IT** -- the env and the profile are
read exactly once, where the plan is built, and the resolved ceiling is frozen
into the coverage plan the same way `render_frames` already is. Both phases then
read the SAME number, so the divergence is unconstructible rather than detected.
Every operator knob survives. Subsumes the profile case.

The coder window that wrote this leans hard toward **(B)** -- it is the same
"make them the same call, not two calls that happen to match" fix this build has
applied five times -- but (B) changes what a `FrameContract` MEANS and touches
the stamped-plan schema, so it wants grounding rather than a coder's preference.

**That is exactly what an r1-r4 arc is for**, and the repo's own doctrine says
so: *"every remaining big block must be RE-GROUNDED by a kibitz arc before it
executes -- r3+r4 by default, a full r2->r3->r4 for both."*

---

## Variant A -- the arc, then the code (RECOMMENDED)

Cost is near zero: agy is $0, codex is rung 3, and the arc replaces a decision
the operator would otherwise have to make cold.

Ordering that matters: **write the problem statement BEFORE running any round.**
An arc run against a vague prompt produces vague grounding, and this fork has a
precise shape -- two named options, a known cost on each side, and six specific
tests that constrain the answer. Put those in the doc.

After the arc: judge every claim against real code yourself, pick the branch,
record WHY in the commit, then build it with the standing cadence -- suite green
-> mutation-prove -> fan out (agy + Sonnet lenses + codex) -> judge -> fix
survivors -> re-mutate -> push.

---

## Variant B -- skip the arc, code 7c

7c is the largest CPU-only block left and does not strictly need the fork
settled, though two of its items touch it. **Rip the fallbacks.** The operator
has explicitly accepted temporary breakage:

> "we have to rip out the dead code. There's no fallbacks. All video models must
> obey these new paths... I realize it's going to temporarily not work until you
> build the code. If that's accepted. And that's why I asked you before that we
> record the requirements for each path."

The requirements ARE now recorded (`docs/ENGINE_MATRIX.md`, generated, with a
`--check` drift gate), which was the stated precondition. The list:

| target | where |
|---|---|
| ping-pong `extend_frames_to_target` | `eng_wan_ti2v.py:521-533`, `eng_ltx_8gb.py:426-437` |
| composite loop-fill | `otr_silent_composite._should_loop_fill` |
| held-last-frame | composite |
| **provider-side clamps** (added by 7a's audit) | `_CloudVideoBase._duration_seconds` ends `max(min_s, min(max_s, secs))`; `word_razzle` does `8 if secs > 5 else 5`; Veo's `_duration_s` discards the requested length at 1080p/4k |
| **`trim_tail` computed and never applied** | `render_beat_coverage` early-returns to the historical path. PRE-EXISTING drift, not a regression -- `wan_i2v` already quantized 50 to 53 and shipped 53, and the composite absorbed it. Wiring the trim and removing the absorption belong together |
| **`OTR_LTX_LOOP_VIA_REVERSE`** | ON by default, returns `2N-1` frames -- a 169-frame ask comes back as 193 because the loop floor is 97. `ltx_video` violates its own declared 169 ceiling TODAY with no env set |
| adapter-side half of chunk 5 (r4's shape, still owed) | each segment graph takes the prepared handles as LITERALS and omits its loader nodes |

Expect the suite to go red here and that is the point -- but land it in slices,
each one green and pushed, not as one large red commit.

---

## Standing rules that have bitten before

* **Never blanket-kill Python.** Selective kill by CommandLine via CIM only -- a
  blanket kill severs the MCP servers and, in a remote window, the bridge you
  are watching through. CLAUDE.md section 4.
* **Preserve other windows' dirty `tmp/` paths** -- `_chain_720.ps1`,
  `_rearm_gate.ps1`, `_status_bake.ps1`. Never reset/stash/checkout them away.
* **Pathspec-only commits.** Commit AND push every green slice; verify
  `HEAD == origin`. One push attempt max, then hand over a PowerShell block.
* **Never read `/mnt/user-data/uploads/`** for this repo -- lagging snapshot.
  All reads via Desktop Commander on the Windows path.
* The KNOWN-FAIL-GUARD conftest hook **suppresses pytest tracebacks**. To see
  one, write a temp `.py` under `tmp/` that calls the code directly and prints
  `traceback.print_exc()`.
* Test venv: `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe`. The
  repo's own `.venv` has no pytest.
* UTF-8 no BOM, ASCII, SFW, never the word "dummy". $0 external spend.
* **THE LAW:** an audit may improve a story, never fail one for length,
  language, style, visual vocabulary, or quality.
* Two strikes then `/kibitz` before a third fix attempt.
* **Mutation-prove every fix.** A test that has not been shown to fail for the
  right reason is decorative -- a mutation run caught a vacuous assertion in 7a
  that two humans and a green suite had missed.

## Model budget

Rung 1 local Qwen -> rung 2 **agy (Gemini 3.6 Flash High, $0)** -> rung 3
**codex `gpt-5.6-sol` high** -> rung 4 Claude -> rung 5 OpenRouter -> rung 6
Fable. The operator has cleared fanning out to Sonnet subagents, agy and codex.
Between them they have found real defects in already-green code six times.

Kibitz: `python kibitz/scripts/kibitz.py --doc <plan>.md --round r3 --only agy`
(`--only codex`, `--only claude`). `KIBITZ_AGY_MODEL` defaults to
"Gemini 3.6 Flash (High)".

---

# VARIANT A -- paste from here down (RECOMMENDED)

```
Resume the OTR build -- run the otr-handoff skill; you are CODER WINDOW A per
GO_FORWARD "Window packing". NO GPU IS AVAILABLE this window: the 7d live slice
is OUT, everything below is CPU-only.

Read docs/GO_FORWARD_PLAN.md IN FULL (CURRENT STEP is authoritative, rewritten
2026-07-26 -- older text about "the first adapter opt-in" is dead), the last few
docs/HANDOFF_LOG.md entries, docs/2026-07-26-chunk-7b-window-prompt.md (the
15-variable env audit table and the OPEN DECISION), and git log/status.

State your MODEL & CREDIT BUDGET rung, then the current step, then a 5-line
scope/acceptance summary, then STOP until I confirm.

CURRENT STEP: settle the 7b architecture fork with a kibitz arc, then code the
winner. The fork: otr_shot_lock._stamp_coverage_plan reads a STATIC declared
ceiling, partitions the beat and mints its per-segment stills -- and only then
does the renderer read the environment (and the capability profile) and possibly
get a DIFFERENT ceiling. Both halves are internally consistent, so the
divergence is silent. Option A is REFUSE on disagreement, which is simple but
breaks the documented 8GB operator knob (a blanket refusal was already built and
REVERTED for exactly that reason -- six tests assert an operator can lower a cap,
and they are production behaviour, not stale debt). Option B is RESOLVE ONCE AT
PLAN TIME AND STAMP IT, so both phases read the same frozen number and the
divergence becomes unconstructible rather than detected; it keeps every knob and
subsumes the OTR_ACTIVE_PROFILE / profile_max_render_frames() case that A cannot
reach at all.

FIRST write docs/2026-07-27-multiclip-7b-fork-problem-statement.md -- the two
options, the cost on each side, the six constraining tests by nodeid, and what
"stamping a resolved contract" would do to the plan schema. A vague arc prompt
buys vague grounding, and this fork has a precise shape.

THEN run the arc: r2 -> r3 -> r4 via kibitz, agy first ($0), then codex
gpt-5.6-sol high. Judge every claim against real code YOURSELF -- do not adopt a
recommendation you have not verified. Write the judgment to
docs/2026-07-27-multiclip-7b-fork-judgment.md, pick the branch, and record WHY
in the commit that implements it.

THEN build it, in green pushed slices. Mutation-prove every fix -- a test that
has not been shown to fail for the right reason is decorative; a mutation run
caught a vacuous assertion in chunk 7a that a green suite and two QA panels had
missed. Fan out for QA before each push: agy via kibitz, codex, and Sonnet
subagent lenses. Two panels found six real defects in 7a's already-green,
already-mutation-proven code and the stubs passed all six.

If the arc converges early and there is window left, start 7c (rip the
fallbacks) -- the list is in docs/2026-07-27-next-window-prompt-nogpu.md.

Two strikes then /kibitz. THE LAW holds. Never blanket-kill Python -- selective
CIM kill by CommandLine only. Preserve other windows' dirty tmp/ paths
(tmp/_chain_720.ps1, tmp/_rearm_gate.ps1, tmp/_status_bake.ps1). Never read
/mnt/user-data/uploads/ for this repo; all reads via Desktop Commander on
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Test venv is
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe. Pathspec-only commits,
commit AND push every green slice, verify HEAD == origin. UTF-8 no BOM, ASCII,
SFW, never the word "dummy". $0 external spend.
```

---

# VARIANT B -- paste from here down (skip the arc, go straight at 7c)

```
Resume the OTR build -- run the otr-handoff skill; you are CODER WINDOW A per
GO_FORWARD "Window packing". NO GPU IS AVAILABLE this window: the 7d live slice
is OUT, everything below is CPU-only.

Read docs/GO_FORWARD_PLAN.md IN FULL (CURRENT STEP is authoritative, rewritten
2026-07-26), the last few docs/HANDOFF_LOG.md entries,
docs/2026-07-27-next-window-prompt-nogpu.md, and git log/status.

State your MODEL & CREDIT BUDGET rung, then the current step, then a 5-line
scope/acceptance summary, then STOP until I confirm.

CURRENT STEP IS CHUNK 7c: rip the fallbacks. No fallbacks; all video models obey
the coverage-plan paths. The operator has explicitly accepted that the build
temporarily does not work, on the condition that the per-path requirements were
recorded first -- they were, in docs/ENGINE_MATRIX.md, which is generated from
the live registry with a --check drift gate in the suite.

The list: ping-pong extend_frames_to_target (eng_wan_ti2v.py:521-533,
eng_ltx_8gb.py:426-437); composite loop-fill
(otr_silent_composite._should_loop_fill); held-last-frame; the provider-side
clamps found by 7a's audit (_CloudVideoBase._duration_seconds ends
max(min_s, min(max_s, secs)); word_razzle does 8 if secs > 5 else 5; Veo's
_duration_s discards the requested length at 1080p/4k); the trim_tail that
render_beat_coverage computes and never applies because it early-returns to the
historical path (PRE-EXISTING drift, not a regression -- wan_i2v already
quantized 50 to 53 and shipped 53 and the composite absorbed it, so wiring the
trim and removing the absorption belong together); OTR_LTX_LOOP_VIA_REVERSE,
which is ON by default and returns 2N-1 frames so ltx_video breaks its own
declared 169 ceiling today with no env set; and the adapter-side half of chunk 5
(each segment graph takes the prepared handles as LITERALS and omits its loader
nodes).

Expect red and land it in SLICES -- each slice green and pushed on its own, never
one large red commit. Where a fallback's removal would break a real lane, say so
and stop rather than guessing; a blanket env refusal was already built and
reverted in 7b for exactly that reason.

Mutation-prove every fix. Fan out for QA before each push: agy via kibitz ($0),
codex gpt-5.6-sol, and Sonnet subagent lenses; judge every claim against real
code YOURSELF before acting on it.

Two strikes then /kibitz. THE LAW holds. Never blanket-kill Python -- selective
CIM kill by CommandLine only. Preserve other windows' dirty tmp/ paths
(tmp/_chain_720.ps1, tmp/_rearm_gate.ps1, tmp/_status_bake.ps1). Never read
/mnt/user-data/uploads/ for this repo; all reads via Desktop Commander on
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio. Test venv is
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe. Pathspec-only commits,
commit AND push every green slice, verify HEAD == origin. UTF-8 no BOM, ASCII,
SFW, never the word "dummy". $0 external spend.
```
