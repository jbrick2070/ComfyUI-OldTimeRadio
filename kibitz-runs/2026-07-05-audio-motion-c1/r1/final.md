# Kibitz r1 -- judgment + hardened C1 wiring plan

Panel this round: codex (grounded, strong). Antigravity HUNG at 0 bytes for
~6 min (agy credit/quota bug, consistent with the 2026-07-02+ pattern) -> treated
as tapped, driver stopped. Claude = anchor + judge.

## Grounding of codex's claims (verified against the REAL tree)
- MUST-FIX 1 (insertion point) -- **CONFIRMED**. Probe of the real JSON: link
  256 = node90 OTR_ShotLock[out0] -> node91 OTR_ImageGenDispatcher[in0]; link
  260 = node91[out0] -> node92 OTR_VideoRenderBatch[in0]; link 267 = node91[out1
  image_done] -> node92[in2]; link 264 = node7 EpisodeAssembler[out1] ->
  node92[in1 master_audio_path]. So node 92 consumes node 91's patched ledger,
  NOT ShotLock. My design doc's "ShotLock->[96]->VideoRenderBatch" was WRONG.
  Corrected Option A = **91 -> [96 OTR_AudioMotionProfile] -> 92**, re-point link
  260 through 96, preserve node 91's image_done (link 267) straight to 92.
- MUST-FIX 2 (durable-write authority undefined) -- **CONFIRMED**. The shipped
  core only mutates an in-memory dict; C1 says "ledger stamp" = must survive to
  disk. FIX: the producer resolves the active ledger path
  (`_otr_ledger.in_flight_ledger_path()`, the same discovery node 93/85 use) and
  writes via `save_ledger_safe(path, led)` -- best-effort, fail-soft, never
  blocks the render. Option A node ALSO returns the mutated wire JSON downstream.
- MUST-FIX 3 (Option B slice not observable) -- **CONFIRMED-per-codex /
  verify-at-build**. run_episode returns ledger/clips/trace/vram_peak, not the
  per-beat slice paths, so Option B would re-slice via the render_driver slicer
  (a cache hit) rather than "reuse" the slice for free. Weakens B; not my pick.
- MUST-FIX 4 (row universe undefined) -- **CONFIRMED**. FIX: profile is **one
  row per VIDEO SHOT** (`ledger["video"]["shots"]`, what the render path
  iterates), timing from the shot's start_s/dur_s; skip (ok=False,
  reason="no timing") when timing is missing. resolver =
  `render_driver._slice_master_audio(master_path, start_s, dur_s, master_hash)`
  (standalone, READ-ONLY, cache-keyed on master_audio_sha256).
- SHOULD-FIX 1 (goal/method mismatch) -- **ACCEPTED**. C1 literally says
  "producer node". Honoring the spec => Option A is the faithful build; Option B
  is explicitly the "opportunistic stamp" alternative, not the spec.
- SHOULD-FIX 2 (A "always runs" needs a live dep path) -- **ACCEPTED/nuanced**.
  node 92 is OUTPUT_NODE=True and always in the graph, so node 96 on the
  91->96->92 path always executes. (The real A-vs-B difference is loop-level: B's
  per-beat loop is empty on a procgen-only episode -> no profiles; A's node
  iterates video.shots regardless.)
- SHOULD-FIX 3 (B zero-JSON only if no INPUT_TYPES change) -- **ACCEPTED**. B
  adds no inputs to node 92 (it already has both) -> truly zero-JSON.
- CUT 1 (custom IS_CHANGED premature) -- **ACCEPTED**. v1 relies on input-hash
  caching + the existing slice cache key; custom IS_CHANGED is a follow-up.
- CUT 2 (operator eyeball is process not architecture) -- **PARTIAL**. Agreed
  the ACCEPTANCE gates are validator + JSON round-trip + link/widget audit; but
  in THIS repo the operator DOES gate frozen-JSON graph edits (hard rule), so
  Option A's JSON edit still needs his eyeball -- that is a repo rule, not
  over-engineering.

## Converged position after r1
Option A (dedicated producer node, faithful to the C1 spec) is the correct
target, with: insertion 91->[96]->92; durable save via save_ledger_safe on the
in-flight ledger path; one profile per video shot; resolver = the existing
read-only master slicer; NO custom IS_CHANGED in v1; acceptance = validator +
round-trip + link/widget audit. Option A edits the FROZEN production JSON, which
is operator-gated here, and the sole consumer (C2) is DEFERRED -> no urgency to
touch the frozen graph while the operator is away.

DECISION: extraction core is SHIPPED (@d60bf371). The producer-node wiring
(Option A, corrected + hardened above) is BUILD-READY and handed to the operator
for the frozen-JSON eyeball -- this is the "real fork + needs operator" stopping
condition the operator's own baton defined. r2-r4 (coding/wiring/convergence
detail) run when the operator greenlights the JSON edit.
