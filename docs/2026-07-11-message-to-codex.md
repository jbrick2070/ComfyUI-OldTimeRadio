# MESSAGE TO CODEX (paste this whole file into codex)

Good work on `codex_framing.md`. The verdict is accepted in principle: do not bolt a
billboard onto the script; separate the announcer from the conflict cast, give the
score an explicit listener-facing frame plane, let the LLM author every syllable, and
have Python reject only an invalid frame topology. That is the right shape and it
obeys the law (Python judges, the LLM writes). Same for treating the requested word
count as a creative scale request and a post-hoc statistic -- never a trimmer, padder,
or rewriter.

Before you write a line of code, read this. It is not optional.

## 1. YOU ARE NOT ALONE IN THIS REPO RIGHT NOW

Another agent (Claude, in Cowork) is actively editing and pushing to `v2.0-alpha`
while you work. HEAD is moving every few minutes -- it went `220066ef` -> `a5e44d2c`
-> `ea149ab0` -> `88888583` during your last task. The project rule is ONE coder
window in the code at a time, and we are temporarily breaking it, so we serialize by
FILE OWNERSHIP instead. Respect it exactly.

**DO NOT TOUCH these files. They are being edited live:**
- `nodes/_otr_scifi_gemini.py`
- `nodes/_otr_scifi_sonnet.py`
- `nodes/cast_lock.py`
- `tests/test_scifi_lane_schema_parity.py`
- `tests/test_scifi_gemini_lane.py`, `tests/test_scifi_sonnet_lane.py`
- `tests/test_cast_lock.py`, `tests/test_fable2_tail_context.py`

**YOU OWN** the beat/word-count planning and the video render path: the multi-clip
splitter, the render plan, the video engines, and the frame-plane work in
`nodes/_otr_scifi_codex.py` + its pack seams + `workflows/otr_canonical.json`.

If you believe you MUST change a file on the do-not-touch list, STOP and say so
instead of doing it. We will hand the baton over explicitly.

Always `git pull --rebase origin v2.0-alpha` before you start and again before you
push. If you hit a conflict in a file you do not own, do not resolve it -- stop and
report.

## 2. The rules you must follow (these are the operator's, not mine)

- Fix at the ROOT CAUSE. No shims, no band-aids, no post-hoc moves.
- **Python judges, the LLM writes.** Never have Python author, rewrite, trim, pad, or
  template story text. A "fix" that invents dialogue or prose is an automatic reject.
  Deterministic repair is allowed ONLY for mechanical metadata that is already implied
  by an accepted upstream artifact (ids, ordering, enums, a fixed role label,
  forbidden extra keys). If it is ambiguous, FAIL CLOSED -- never guess.
- **Any node / wiring / widget change MUST land in `workflows/otr_canonical.json` in
  the SAME change as the code.** Code that is not wired into that JSON is dead. Its
  `widgets_values` is POSITIONAL: only ever APPEND a new optional widget at the END --
  inserting mid-list silently shifts every saved value.
- Run the FULL suite + the Bug Bible after EVERY code change, without being asked.
  Suite is green at 7597 passed / 31 skipped / 1 xfailed; the Bug Bible (separate repo,
  `comfyui-custom-node-survival-guide`) is 17 passed. Do not regress either.
- COMMIT AND PUSH every green chunk to `v2.0-alpha` in the same session. Local-only
  commits are the failure mode we guard against.
- Never the word "dummy" -- use "placeholder" or "stub". SFW. UTF-8, no BOM.
- Rendered assets go straight to `otr\episodes\<ep>\`, final to `otr\obs\`. Never stage
  in tmp "to move later." Confirm the asset exists (Test-Path) before declaring success.

## 3. What I need from you next, in order

**(a) The multi-clip restoration.** You found the 9s-target/10s-cap splitter was
tombstoned and ShotLock never replaced its multi-clip behavior. Restore it as designed:
each beat gets multiple editorial clips, provider-native tiling applied AFTER engine
selection for minimums/maximums/discrete duration tiers, across all 30 registered
engines. The ledger records the parent line once and tracks every visual child exactly.
Master audio and story text stay untouched.

**(b) The frame plane.** Then implement your own framing proposal: the announcer leaves
the conflict cast, the score gains an explicit frame plane, and a validator fails closed
on an invalid frame TOPOLOGY only (never on the words). Do not let this ship 6x worse at
720 words -- your scaling table (30 / 120 / 300 / 720) is the part I care most about,
because at 30 words a full billboard would eat the entire episode.

**(c) One thing you must tell me before you start (a):** does removing per-beat word
count chasing change what a "beat" IS, or how many there are? Gemini's new P3 output
reservation is sized off the advisory band count (`outline_output_token_budget(words,
len(bands))`), and Codex's whole-script budget is sized off the accepted LINE COUNT. If
you change the beat/line structure, those reservations under-reserve and the passes will
silently truncate -- which is exactly the bug that just cost us four consecutive rolls
(`PROMPT_GUARD: Truncated 5408 -> 4592 tokens`). Say what changes and I will re-derive
them. Do not change them yourself; they are in files you do not own.
