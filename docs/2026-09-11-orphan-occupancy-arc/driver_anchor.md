# Driver anchor -- row 3.3, orphan generation occupancy

Written by the driver (Claude Opus 5, Cowork, 5080) BEFORE any panel round, grounded
in the real Windows files. The panel proposes; the driver disposes and verifies every
claim against these files.

**Round shape (operator directive): ONE second opinion per round.** r1 codex, r2
cursor, r3 codex, r4 sonnet. **An arc IS coding** (operator, 2026-09-11), so this arc
is the work, not a prelude to it.

**Why this row first:** the operator's bar is *"as long as it doesn't crash when it's
not supposed to."* This is the crash. Everything else in the design backlog is
correctness or taste; this one loses whole episodes to OOM.

---

## 1. The mechanism, grounded

1. **A timeout abandons a live GPU worker.** `nodes/_otr_model_loader.py:130-131`,
   verbatim: *"story_orchestrator._run_with_timeout abandons a worker thread on a
   wall-clock timeout (generation is not cancellable mid-token)."* The thread keeps
   decoding on the GPU. Nothing can stop it, and that is not a bug -- HF generation
   genuinely has no mid-token cancel.

2. **The cache is invalidated immediately.** `_detach_and_invalidate_locked`
   (`:185`) clears the singleton entry under `_CACHE_EPOCH_LOCK`.

3. **So the process now believes the GPU is free, and it is not.**
   `has_local_resident_llm()` (`:1688`) says so in its own docstring: *"This does NOT
   make orphan GPU occupancy visible -- it still reads 'nothing resident' the instant
   a timeout invalidates the cache dict, even if the orphan worker is still actively
   on GPU. That is the deferred orphan-occupancy registry (PBUG-20260825-04), not
   this function's job."*

4. **The next stage loads a second model into VRAM the orphan still holds.** On an
   8 GB card that is an OOM; on 16 GB it is an OOM on the bigger models. The
   PBUG-20260825-04 narrative shows the shape: *"two upstream stages timed out first
   ('NewsCuration phase exceeded 65s', 'NewsCurationDeep exceeded ...')"* and then a
   quantized load failed to materialise.

**The defect is not the abandonment and not the invalidation. It is that the two
happen without anything recording that VRAM is still occupied.** Residency is tracked
by a cache dict that means "do I have a handle", when the question that matters is
"is the device busy".

## 2. Why this needs an arc and not a sixth patch

Several narrow fixes have landed here and each surfaced a NEW race. The most recent
(r4 kibitz, Cursor) moved the read inside `_CACHE_EPOCH_LOCK` because
`_detach_and_invalidate_locked`'s `clear()` and `update()` are two statements, so an
unlocked read could observe a momentarily-cleared dict. That fix is correct and it
closed a torn read -- and it explicitly declined the larger question.

Per CLAUDE.md's two-strikes floor, a defect that survives repeated fixes means the
MODEL of the problem is wrong, not the patch. That is why this is an arc.

## 3. Questions for the reviewer -- answer these, do not restate the above

1. **Is the mechanism above complete and correct?** Walk `_run_with_timeout` in
   `nodes/story_orchestrator.py` and the invalidation path yourself. Is there a
   FOURTH step the driver missed -- something that already partially mitigates this,
   or a second way an orphan is created that is not a timeout?
2. **What is the smallest honest way to know the device is still busy?** Candidates
   to weigh, and name others: the abandoned future/thread itself as the registry (it
   is still referenced somewhere -- find out where); a torch allocator reading; an
   epoch counter that a worker decrements on exit; an explicit occupancy registry as
   PBUG-20260825-04 deferred. Which survives the fact that the worker is
   uncancellable and may never return?
3. **What should the next loader DO when it learns the device is busy?** Wait with a
   deadline, load smaller, refuse the load, or proceed and let the OOM happen? Note
   the project rule: an OOM is explicitly *"the only acceptable killer"*, and a
   truthful failure is preferred over a silent degradation. Argue from that.
4. **Blast radius.** This is shared code on both machines (CLAUDE.md 0B). The 8 GB
   box hits it far more often; the 16 GB box must be provably unchanged when no
   orphan exists. What measurement proves that?
5. **Is any part render-inert enough to ship before a test wave, or must all of it
   wait?** Say plainly.

## 4. Hard constraints -- a proposal violating any is rejected on sight

- **No new content gates, story-rejection gates, word/duration/cast limits,
  standalone checkers, chunkers, or recursive retry loops.**
- **No prompt changes.** Prompts are hand-crafted per model and character-budgeted.
- **Do not make an OOM silent.** An OOM is the only acceptable killer and must stay
  truthful; the goal is to stop CAUSING avoidable ones, not to hide them.
- **Do not reduce how many episodes reach `otr/obs/`.** A fix that refuses more loads
  than it saves is a worse fix.
- **Never blanket-kill Python processes** -- that kills the agent's own tooling too.
- **Do not revive deleted code, add a second workflow graph, or add a new
  output-path owner.**
- Style: no curse words, never the name "dummy".
