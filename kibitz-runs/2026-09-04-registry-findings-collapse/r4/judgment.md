# r4 judgment -- registry-findings-collapse (driver: Claude / Opus 5)

Roster: Sonnet 5 (subagent seat), Antigravity Gemini 3.8 Flash (High),
Antigravity Gemini 3.1 Pro (High) -- two distinct agy models run as two lanes
against the same r3 `final.md`. Codex had no seat in r3 or r4 (standard tier
out of credits until 2026-09-07; the Spark model overflowed its context and
then hit its own limit). Cursor did not review r4. Driver anchor written
before the fan-out (`docs/2026-09-04-registry-findings-collapse/driver_anchor_r4.md`).

## THE ROUND'S OWN LESSON: two lanes reviewed a tree that had moved

Both agy lanes opened with the same MUST-FIX -- that Phase 0 names symbols
which "do not exist", 3.1 Pro adding that they were "hallucinated in prior
rounds" and concluding **"Delete Phase 0 entirely."** Grounded: the symbols
are absent because Phase 0 SHIPPED while r4 was running (commit `2f12a696`,
twelve files, 149 lines, full suite 13,490 green, an independent QA pass
clean). The lanes read the post-rip tree against a pre-rip plan. Acting on
that MUST-FIX would have reverted a green, reviewed, pushed commit on the
strength of a review that was right about the file and wrong about the world.

3.1 Pro's supporting claim is independently false: it says the
`_stamp_durable` line at `otr_shot_lock.py:3406` "is a live CALL, not an
import, so ripping it would break the ledger stamp." What Phase 0 removed was
the DUPLICATE IMPORT at `:3407-3409`; the call at `:3406` stands, bound by
the first import in the same method at `:3216-3219`. The suite agrees.

Kept as the rule for the rest of this arc: a review of a moving tree must be
grounded against the COMMIT it read, and a reviewer's "this does not exist"
is a claim about a timestamp before it is a claim about the code.

## Antigravity, Gemini 3.8 Flash (High), grounded

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | six Phase 0 symbols do not exist; `episodes_for_obs_dir` also needs its `__all__` entry removed | STALE INPUT (they were ripped in `2f12a696`); the `__all__` half was ALREADY done in that commit | no action; the plan's Phase 0 stands as shipped |
| MF2 | `popen([])` raises `IndexError` before the named error; name the exception class | ALREADY SATISFIED: `proc.py` validates argv in `_check`, which BOTH `run` and `popen` call, and the class is `ExecutableNotAllowed(RuntimeError)` | no change; the plan text now names the class |
| MF3 | `env.get`'s `default: str \| None` is wrong -- `otr_image_gen_dispatcher.py:206` passes the int `4242` | CONFIRMED (verified: exactly one such site in the tree) | `get(name, default: Any = None) -> Any`, with the call site named in the docstring. A type hint must not describe a contract the function does not enforce |
| MF4 | name the five network-guard files and the AST predicate | CONFIRMED as a gap -- but its list is wrong: it swaps the RSS fetcher's SSRF-hardened socket for `_otr_kokoro_voice_prefetch`'s `hf_hub_download`, which is not in the scan's network findings | the five FILES are the payload's: `_otr_comfy_backend.py:380`, `_otr_feed_fetch.py:249`, `_otr_openrouter_backend.py:1006`, `_otr_google_api/client.py:187` (+`:235`), `_otr_shared/cloud_media_invoke.py:571` |
| SF1 | a literal `"python*"` dict key cannot match `python3.12` | ALREADY SATISFIED: the interpreter rule is a separate `startswith` predicate, and `_normalized` calls `str()` first (so a `Path` argv[0] is fine) | no change |
| SF2 | `test_terminal_frame.py:498` also asserts `wrapper_bridge.py::encode_frames_to_silent_mp4` | CONFIRMED | both `:495` and `:498` are the receipt; `wrapper_bridge` migrates in b-video |
| SF3 | "twelve rewritten tests" contradicts the ten in-scope files | CONFIRMED | wording fixed |
| SF4 | the psutil test must also `monkeypatch` `os.name = "nt"` or it passes vacuously off Windows | CONFIRMED | folded into D's test |
| OPT/CUT | snapshot stays a plain dict; identity re-exports; cut the six phantoms | first two already so; the cut is the stale-input item | |

## Antigravity, Gemini 3.1 Pro (High), grounded

| # | claim | verdict | disposition |
|---|---|---|---|
| MF1 | every Phase 0 symbol is hallucinated; `_stamp_durable:3406` is a live call; delete Phase 0 | REJECTED on both halves (see above). The strongest recommendation of the round was the one to act on least | none |
| MF2 | the "three-form recipe `ffmpeg.py:28-34`" cannot be copy-pasted: that file uses `from .ffprobe` because it LIVES in `_otr_shared`; a top-level node needs `from ._otr_shared import`, an engine `from .._otr_shared import` | CONFIRMED, and the best catch of the round -- the r3 wording would have failed at the first migrated node | the plan now states the recipe BY DEPTH, with the flat fallback each level needs |
| SF1 | mirror the stdlib signature exactly | CONFIRMED (same as Flash MF3) | done |
| SF2 | the psutil warning should say lease reclamation is disabled | CONFIRMED | the warning names the consequence, not just the cause |
| OPT | say that no drift fix may ride in the migration commits | already the plan's A-0 | restated |

## Sonnet 5 seat, grounded

Its MUST-FIX (that `content_oracle.py` has ZERO production importers, against
the driver's claim of two) was CONFIRMED and acted on before r4 closed: the
two "importers" are comment mentions, `git grep -nE "^[^#]*\b(import|from)\b.*content_oracle"`
returns only eight test sites, and the r3 final now records three defensible
options with the decision left to the operator. Its two receipt nits are also
confirmed: the argv receipt has 21 distinct files to the scanner's 20 (the
extra is `production_ledger.py`, whose `check_output` the scan counted under
its own rule), and the driver note's per-binary tally summed to 33, not 35 --
the corrected tally is ffmpeg 23, ffprobe 2 (including
`otr_master_audio_mux.py:144`), python 3, git 2, nvidia-smi 1, blender 1,
plus 3 rows the note's shorthand had folded together = 35. The
`eng_spandrel_esrgan.py` sha256 is at `:259`, as the plan says (verified by
grep; the seat's `:258` is one line high).

## Convergence

No new must-fix survives grounding except the two folded above (the env
signature and the import depth), both of which are wording in the plan and
one line in an owner that is not yet committed. The architecture has not
moved since r1. The arc is CLOSED; the build proceeds from `final.md`.
