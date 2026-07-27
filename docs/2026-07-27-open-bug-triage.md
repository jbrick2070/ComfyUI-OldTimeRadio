# Open-bug triage -- codex + agy + Fable, 2026-07-27

Operator-directed. HEAD at entry `43ee48d9`; one fix landed at `54b3626b`.

Panel: **codex `gpt-5.6-sol` high** and **agy `Gemini 3.6 Flash (High)`** via
kibitz r1 (run: `kibitz-runs/2026-07-27-bug-triage/r1/`), then a **Fable**
consult under CLAUDE.md section 9's reality exception. Claude wrote the anchor
triage first and grounded every panel claim against the real Windows files
before acting on any of it.

**The codex seat was verified as `gpt-5.6-sol` for this run.** That file has
drifted to `gpt-5.5` on past arcs; the pin held here.

## The headline: the panels disagreed with me more than with each other

My anchor triage had five A-rows. The panel corrected three of them, cut one
entirely, and added one I had not found. Grounding confirmed every correction.

| row | my call | after grounding |
|---|---|---|
| A1 GGUF ceiling decorative | fix in preflight | **fix shape was INCOMPLETE** -- see below |
| A2 override echo hides `llm` | add `llm` to the echo | **wrong causal chain** -- see below |
| A3 `provider_side` regression | write the test | **ALREADY COVERED** -- cut |
| A4 `_use_i2v` contradiction | probably a misread | **CONFIRMED and reachable** |
| A5 encode boundary | maybe defer | **cut as a live bug, keep a dtype assert** |
| A6 | -- | **NEW: the Q4 artifact has no size and no SHA** |

**A1 -- the cache-hit bypass I missed.** I had "enforce the policy ceiling in
GGUF preflight". Codex found that a resident model returns at
`_otr_model_loader.py:982-992` *without entering preflight at all*, and
`GGUFLoadConfig.reuse_key()` (`_otr_gguf_backend.py:435-439`) excludes the
ceiling. So a permissive-policy load satisfies a stricter-policy request by
cache hit and my fix would have missed it entirely. The correct shape is ONE
policy-admission calculation before BOTH cache reuse and loading, with a test
for permissive-cache -> stricter-request at the same load identity.

**A2 -- my causal chain was wrong.** The override does not come from the
validator's `OTR_ACTIVE_PROFILE` export; it happens at submission via
`scripts/otr_canonical_api_run.py:157` -> `apply_profile_to_workflow`. And the
real applier (`nodes/_otr_workflow_apply.py:492-540`) ALREADY flattens `llm`.
Only the printed echo (`scripts/otr_api.py:816-825`) is stale. So the fix is to
generate the echo FROM the applier's flattened map -- adding `llm` by hand, as I
proposed, would leave the next drift intact.

**A3 -- cut.** Both seats plus my own read agree: covered by
`test_video_render_driver_perbeat_audio.py:319-325` (the redirect preserves
`cloud_kling_avatar`), `test_video_platform_aseam.py:903-920` (picked route) and
`test_still_plan_parity.py:114-116` (forced route). I had checked the CODE and
not the TESTS, and would have written a duplicate.

**A6 -- NEW, and the highest-value defect absent from GO_FORWARD.** The shipped
8 GB profile selects Gemma `Q4_K_M`, but `GGUF_ARTIFACTS`
(`_otr_gguf_backend.py:56-60`) gives that quant size `None` and `GGUF_ROWS`
(`:226-233`) gives sha `None`. Size/SHA checks are conditional, so a truncated
or partial Q4 download passes readiness.

## Fable resolved both splits the mechanical panels left open

- **A5 (codex: fix at the shared boundary / agy: cut).** Fable: **cut as a live
  bug, keep Codex's location at a fraction of his scope.** Every live producer
  feeds the encoder an exact-size uint8 buffer, ffmpeg raises on a short write,
  and chunk 6 already put a decode-count at the boundary that matters
  (`assemble_beat_segments`, `wan_shared.py:224-232`). The residual is latent: a
  future float32 caller would pipe 4x the bytes and get a clean receipt. One
  `dtype == uint8` assert closes it.
- **B4 `ShotRow` (mine: operator ruling / agy: coder fix).** Fable: **coder
  fix.** ShotLock demonstrably stamps `role`, `char_id`, `start_s`/`dur_s`,
  `coverage_plan` and `coverage_contract`, none of which exist on a model
  declaring `extra="forbid"` -- so `ShotRow(**real_row)` raises on every real
  ledger, and the "live safety net" other docs cite cannot validate a single
  shipped episode. The repo has its own precedent (`observability`,
  `requires_mesh_portrait`) for fixing exactly this as code. There is no
  product question left.
- **agy's import finding: REAL imports, NOT a build-breaker.** Fable verified
  all four files and found eight more, then killed it: the enforced gate
  (`test_capability_profiles.py:481-503`) excludes the audio lane BY DESIGN and
  says so in its own docstring; ComfyUI imports torch/PIL/numpy before any
  custom node loads; and `__init__.py` wraps every node import so a broken dep
  skips one node loudly. **Do not file.**

## What Fable found that nobody had filed -- and what shipped

**LANDED @ `54b3626b`.** Both defects live in `OTR_MasterAudioMux`, the LAST
node of the graph, where everything raises AFTER the whole episode has rendered.

1. **A FATAL env knob at the terminal node.**
   `float(os.environ.get("OTR_MAX_CREDITS_TAIL_S", "45"))` was unguarded, so
   `OTR_MAX_CREDITS_TAIL_S=45s` in a launch environment killed a finished
   episode with an uncaught `ValueError` -- over a value that only widens a
   sanity ceiling. The `PBUG-20260723-02` shape, at the opposite end of the
   pipeline from where this build usually pays for it. Now IGNORED and NAMED.
   The sibling knob in the same file was already guarded; this was the one that
   was not.
2. **The duration gate fails open and the receipt says OK anyway.**
   `_probe_float` returns `-1.0` when ffprobe is absent or a duration is
   unparsable, which skips the only video-longer-than-audio guard -- and the
   report still appended `duration_check v=-1.000s a=-1.000s ... OK`. Now
   `UNPROVEN`, with the gate named as SKIPPED. Not made fatal: it is the final
   sanity ceiling, not the primary correctness guard, and refusing would lose a
   finished episode on a box that merely lacks ffprobe.

Still open from Fable, not yet fixed: `CanonicalClip.frame_count` -- "the
integer timing authority" -- is decode-counted truth for assembled multi-segment
beats but self-declared input length for every single-render beat, and
`eng_humo` / `eng_ltx_av` return self-declared dicts with no M7 probe while
`wan_i2v`, `wan_ti2v`, `ltx_8gb` and `ltx_video` all probe. The two derivations
agree today only because every producer pipes exact bytes.

## A defect in the bug list itself

**Every line cite I checked has moved.** `_is_cloud_video_engine` is at
`render_driver.py:1599`, not `1274-1295`; the "NO FALLBACK to text-only" refusal
is at `:2148`, not `1801-1817`; `_use_i2v` is at `eng_ltx_video.py:583`, not
`559-572`. The defects are mostly still real; their coordinates are not. Re-pin
a row's cite when you touch it.

## Ranked queue for the next window

Codex's sequencing point stands: **B5 is a dependency, not a peer.** Whether the
profile family is retained or retired changes the value and the acceptance
target of A1, A2 and A6. Get that ruling first.

1. **A1** -- one policy-admission calculation before both cache reuse and load.
2. **A6** -- pin the Q4 size and SHA; reject a non-zero truncated file.
3. **A2** -- generate the echo from the applier's flattened map.
4. **A4** -- make the adapter refuse the missing/stale init image; replace the
   fallback assertions in `tests/test_video_motion.py:340-344`.
5. **B4** -- complete `ShotRow` (now a coder fix, per Fable).
6. **A5-lite** -- one `dtype == uint8` assert at the encoder boundary.
7. Fable's `frame_count` asymmetry: copy the four siblings' M7 probe line into
   `eng_humo` and `eng_ltx_av`.

**Cut and not to be re-derived:** A3 (already covered), the heavy-import finding
(not a violation of the gate as this build defines it), B1 the WAN knob rename
(default: leave), B2 the style-tail enum (default: ratify the exemption).

## Process note

r2/r3/r4 of the kibitz arc were NOT run. The arc hardens a *plan* across four
lenses; what was asked for was a triage plus fixes, and r1 plus the Fable
consult answered it. If the next window wants the full arc on the ranked queue
above, it starts at r2 with this document as input.
