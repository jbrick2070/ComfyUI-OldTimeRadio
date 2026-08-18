# Fix-decision prompt (paste to agy)

The trace is DONE and confirmed twice independently, so this round is not blind:
the reviewer gets the facts and is asked to decide the fix. Paste everything
between the lines.

---

## DO NOT WRITE CODE YET

Decide and specify. No patch, no diff, no implementation. If you start writing
code, stop. I want a decision with its contract and its risks, which I will
review before anything is built.

## THE SETTING

Repo: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
(real Windows files -- read them directly).

It generates old-time-radio drama episodes as videos. Each opens with a hero
TITLE CARD showing the episode title. Episodes land in
`C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes\<episode_id>\`, and
finished ones are published to `otr\obs\`.

## THE CONFIRMED DIAGNOSIS -- verified independently twice, do not re-derive

* `OTR_LedgerScriptWriter.INPUT_TYPES` exposes an optional widget
  `"episode_title"`. `_resolve_inputs` strips it into
  `resolved["episode_title"]`.
* In `OTR_LedgerScriptWriter._run_writer_tail` the precedence ladder is:
  `if resolved["episode_title"]:` -> use it **verbatim**,
  `title_source = "user"`, and the LLM title pass never runs;
  `elif ctx.final_title_override is not None:` -> the lane's own authored title;
  `else:` -> `_generate_title_from_script` on the finished script
  (`title_source = "llm_post_composition"`), falling back to `outline.title`.
* The check is only "is the box non-empty". **It cannot tell a human's real
  title from an automated run label.** Nothing is malfunctioning; every step
  does what it was designed to do.
* **The title is an IDENTITY, not a caption.** The same string reaches at least
  eight surfaces: the burned title card; the mp4 filename and episode directory
  (via `safe_title`); the ledger and treatment sidecar filenames; the
  treatment's `Title :` line; `episode_canon.json`; the credits-roll hero text;
  the telemetry HUD; and the published artifact name in `otr\obs\`.
* **Automated runs systematically inject labels here.**
  `scripts/otr_gpu_soak_matrix.py` builds `f"SOAK{index:02d} {bank} {style}
  {short}"` and passes it as `--title` on **every leg, unconditionally**;
  `scripts/otr_canonical_api_run.py` (`_apply_writer_shortcuts`) maps `--title`
  onto the widget. `scripts/otr_writer_bank_gate.py` passes no title at all.
* Result: 17 of the 65 finished videos in `otr\obs\` carry a run label as their
  on-screen title, filename and credits.

## THE DECISION WANTED

**Where does the fix belong, and what is the exact contract afterwards?**
Argue for ONE and say why the others are worse:

* **(a) At the automated runners** -- stop putting bookkeeping labels in a
  production field. Give them a separate run label that never becomes the title.
  Repairs all eight surfaces at once. What breaks? How do those runs stay
  distinguishable to the operator afterwards?
* **(b) At the widget** -- keep accepting the label but record it as a distinct
  source (e.g. `"harness_label"` instead of `"user"`). Does that fix anything
  the operator SEES, or only the bookkeeping?
* **(c) At the precedence ladder** -- let a supplied title coexist with the
  lane's authored title instead of short-circuiting it. Note the wrong episode's
  lane HAD an authored title ready and was never asked.
* **(d) Something better.**

**Then specify the contract:** what the box means, who may write it, what
happens when both a supplied label and a lane-authored title exist, and **what
the title card should display for an automated run** -- it still has to publish,
and the operator still needs to tell those runs apart at a glance.

## HARD CONSTRAINTS -- breaking one kills the proposal

1. **PUBLISHING TO `otr\obs\` MUST NOT BE REDUCED, GATED OR RELOCATED.** The
   operator reads success by seeing episodes appear there: *"a test is not
   complete unless published to obs... if I see it in obs then it's somewhat a
   success"*, and *"if I don't see it in obs and it took more than 5 minutes
   it's a fail."* Automated runs BELONG there. A previous attempt to tidy them
   into a subfolder was reverted within minutes.
2. **Priority, in his words:** *"long term yeah I want the title right, but for
   my dailies it keeps me going to see episodes."* Fix the label at its source;
   never by suppressing or gating the publish. **Any proposal that reduces how
   many episodes appear in `otr\obs\` tomorrow is wrong regardless of how tidy
   it is.**
3. **AN AUTOMATED RUN MUST BE A REAL EPISODE, START TO FINISH.** Two operator
   statements, this session:
   * *"the code changes all the time, so a harness should always be from live
     code."*
   * *"it should mimic the entire workflow most of the time unless I say
     otherwise."*

   So the DEFAULT for an automated run is the **whole canonical workflow, end to
   end, down the identical road a real episode takes, finishing with a published
   artifact in `otr\obs\`.** Testing one isolated part is the EXPLICIT EXCEPTION
   he names and asks for by hand -- it is not something a fix may impose. This
   also matches the standing repo rule that every headless/API run loads the real
   `workflows/otr_canonical.json`, never a stale copy, a generated variant, or an
   ad-hoc graph.

   **DISQUALIFIED, therefore:** any fix that gives automated runs a separate code
   path, a forked writer node, a shortened pipeline, a stub, or a special-case
   branch that real episodes never take. That would mean the tests stop testing
   what actually ships -- which is the entire reason these runs exist. If your
   proposal makes an automated run behave *differently* from a real episode
   anywhere except the label it carries, say so explicitly and justify it.
4. **The human override must survive.** A person typing a real title into that
   box is a wanted feature. Do not remove it.
5. **An audit may never FAIL an episode** for length, language, style or
   quality, and a render must **degrade, never raise**.
6. **`workflows/otr_canonical.json` is the source of truth.** Any widget change
   must be made in that JSON in the same change, and `widgets_values` is
   POSITIONAL -- only ever APPEND a new optional widget at the END. Inserting
   mid-list shifts every saved value and causes silent drift.

## SECOND QUESTION -- the drift receipt

The operator wants the title tagged so future drift is catchable: *"tag
episode_title and title in an episode run ledger... along with a timestamp of
when it was stamped and telemetry so we catch future drift."*

Draft spec: `docs/2026-08-17-title-provenance-SPEC.md`. Its argument: today's
`meta.title_source` is ONE slot, so it records only the LAST writer and cannot
show a title changing hands -- which is what drift is. Proposed shape is an
append-only chain of stamps (value, source, stage, symbol, UTC timestamp,
replaced-value). Precedent already in this repo:
`_otr_source_identity.SourceIdentity` carries a `provenance` map for exactly
this reason.

**Review that spec.** Right-sized, overbuilt, or underbuilt? And what is the
CHEAPEST check that would have caught these 17 episodes the day they landed?

## OUTPUT FORMAT

```
RECOMMENDATION: <a|b|c|d> + why the others are worse
THE CONTRACT:
  what the box means:
  who may write it:
  conflict rule (supplied label vs lane-authored title):
  what an automated run's title card displays:
  how automated runs stay distinguishable to the operator:
CONSTRAINT CHECK: <address all six explicitly, especially 1 and 3>
MUST-FIX BEFORE BUILD: <numbered, defect + concrete fix>
SHOULD-FIX: <numbered>
CUT THESE: <what to drop, why it is safe>
PROVENANCE SPEC: <right-sized|overbuilt|underbuilt> + the cheapest catch
[ASSUMPTION] <anything unverified>
```

Cite by SYMBOL with short verbatim quotes -- **never line numbers**, they go
stale here within the hour. Write "not proven" rather than inferring. If any
claim above is wrong, say so: it was written by the driver, who has been
corrected several times this week.
