# Opus: start priority #1 cleanup

Work in `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`
on `v2.0-alpha`.

Read AGENTS.md, CLAUDE.md, docs/GO_FORWARD_PLAN.md, and
docs/2026-09-10-cleanup-opus/HANDOFF.md plus PLAN.md and qualification.json.
The latest GO_FORWARD operator instructions control scope and ownership.

You own cleanup sprints 1-2. Implement these three chunks in order:

1. P1: remove automatic roomtone/tape hiss, set clean enhancement defaults in
   code and the real canonical workflow together, and remove the seven unused
   SceneSequencer assignments.
2. P2: give G8 sole ownership of duplicate-ID diagnostics while preserving
   validation failures.
3. P3: remove freeze's unused model acquisition while preserving required
   validation, final unloading and recovery behavior.

Verify the preparation commit a3551ff5 is on origin and recheck preconditions
at the current Windows HEAD. Follow the plan's focused/full/Bug Bible and fresh
canonical CPU checks. Compare with a fresh baseline: the preparation recorded
54 existing unexpected failures, not a green suite. Allow no new unexpected
failures or quarantine additions. Use the real workflows/otr_canonical.json,
never a stale copy or old harness graph.

Obtain one grounded finished-diff review per chunk. Commit and push each
qualified chunk immediately, verify HEAD equals origin/v2.0-alpha, and preserve
unrelated changes. No GPU/server run, model upgrade, artifact/cache deletion,
release change, speaker-placement implementation or replay-system work.
Do not rerun the completed design campaign or expand the cleanup hunt.

After P3, update the handoff and GO_FORWARD with the exact resulting HEAD,
qualification evidence and remaining limitations. Stop at that boundary:
Codex owns My Story sprints 3-5. Do not begin its shared-file changes.
