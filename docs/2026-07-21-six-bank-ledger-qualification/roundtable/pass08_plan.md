# Pass 08 plan -- live bug 8 patch, budget, timing, and music convergence

Reviewer lane: exact `Gemini 3.5 Flash (High)` through Antigravity,
review-only. Driver/coder/judge: Sol only. Audit workspace: clean detached
throwaway worktree at commit
`a58ff25b9706b3df63de2f4cbdbc89bf6ab035c3`.

The broad R2 runner twice parked on reviewer-created background searches and
hit Antigravity's print timeout. Sol therefore split the same scope into four
evidence-only R2 slices (schema, word ownership, zero timing, music topology),
still on the exact lane and in the same clean worktree. Sol grounded every
claim against the real Windows checkout before writing this R3 implementation
anchor.

## Grounded R2 findings

1. **LMFE patch schemas -- confirmed blocker and sibling.** Both
   `_SpokenLineRepairRowV4.line_id` and `_ScriptQualityPatchRowV4.line_id`
   combine `pattern=r"^l\d{3}$"` with redundant min/max length constraints.
   LM Format Enforcer rejects that schema before generation; reuse can then hit
   the same cached incomplete token state as `None.allowed_tokens`. No other
   Python string field has this shape.
2. **Word ownership -- confirmed blocker.** Scifi Codex assembled and sealed a
   143-character-word artifact for a 180-word request without producer-stamped
   `meta.word_budget.target_words/band`. The shared tail also ignores a present
   producer band and tests drift only against global `0.7..1.3` constants.
   Character words are the authority; announcer overhead stays separate.
3. **Title timing -- confirmed blocker.** A known first-dialogue frame of zero
   is treated as missing by a `> 0` check. Zero means no opening-card gap, not
   missing timing, and must not trigger the BUG-404 envelope warning.
4. **Music vocabulary/topology -- confirmed six-bank consumer gap.** Scifi
   Codex writes noncanonical IDs/placements and anchors its interstitial to an
   ordinary spoken row; Fable2 has canonical IDs and music sentinels but omits
   cue anchors/placements; the four inline banks synthesize an unanchored
   interstitial. Live inline masters contain opening/closing audio but their
   ledgers contain no music timing/mirrors, while the interstitial is dropped.
   Live scifi classified all three cues as interstitial and consumed none.
5. **Render plan -- intentional/discard.** `meta.render_plan` is passive legacy
   story-QA telemetry. The real renderer must continue rendering every ShotLock
   shot.

## Proposed R3 implementation

### A. Structured patches and final word fit

- Remove only the redundant min/max constraints from the two regex-locked
  `line_id` fields. Add LMFE character-by-character parser coverage for both
  schemas.
- Stamp a target-relative Scifi Codex character-word contract before content
  mutation. Convert an open lower tolerance of 90% and a closed upper tolerance
  of 111% to inclusive integer bounds, then stamp the exact integer-derived
  ratios. For target 180 this is 163--200; no absolute campaign values are
  hard-coded.
- After P6/P7, P8/P9, and the final spoken-hygiene scour, deterministically
  recount non-skipped `speaker_role=character` words. If outside the band,
  select a bounded subset of existing character rows and run a compact P10
  line-text patch through creative then technical slots. Every accepted merge
  reuses the full graph/spoken-hygiene validator and gets a fresh deterministic
  recount. Retry under a finite dynamic budget, retain the closest valid
  artifact, and truthfully stamp `pass`, `two_slot_quality_floor`, or
  `quality_floor`; count/taste never kills liveness.
- Only after that loop: final validation, ledger assembly, word counts,
  authorship receipt, accepted-line hashes, and shared-tail seals/readiness.
  The shared tail honors a valid producer-stamped band and falls back to the
  legacy constants only when no valid band exists.

### B. Canonical music contract and post-audio join

- Keep internal Scifi Codex model vocabulary unchanged, but translate its
  ledger output to canonical `opening/inter_01/closing` IDs and
  `opening/interstitial/closing` placements. Fable2 stamps the same placements
  plus each music sentinel's `anchor_line_id`. StableAudioTheme retains a
  defensive alias map for stale inputs.
- StableAudioTheme binds an otherwise unanchored synthesized interstitial to
  the ordered `music_inter` sentinel and stamps a deterministic cue-spec hash
  over prompt, requested duration, placement, and anchor. The manifest remains
  the batch-row/output-path authority.
- SceneSequencer checks an interstitial anchor before dispatching every ledger
  row. A sentinel anchor inserts and passes through; an ordinary spoken anchor
  inserts before the dialogue and then still consumes/stamps that dialogue.
  It writes interstitial timing and WAV path back by cue identity.
- EpisodeAssembler reconciles manifest rows into `ledger.music[]` (including
  legacy synthesized cues), identity-gates authored rows, stamps WAV paths,
  places bookends by canonical placement, and remains the sole owner of timed
  mirrored music `lines[]` rows.
- ShotLock's same-episode join identity-merges disk-owned music render fields
  and appends/replaces only assembler-owned `mirrored_from=music` rows tied to
  matched cue identities. This is the bridge that makes image/video/OBS see the
  post-audio music timeline; no workflow socket/link change is required.

### C. Timing, tests, and admission records

- Treat any non-`None` first-dialogue frame as known; zero produces no open
  window and no missing-timing warning. Preserve the zero receipt.
- Add focused tests for both LMFE schemas; target-relative word bounds,
  extend/compress/recount/floor behavior, producer-band drift ownership;
  canonical cue translation and aliases; Fable2/legacy anchors; dialogue-anchor
  insertion without dialogue loss; manifest materialization/bookend timing;
  ShotLock music identity/mirror transfer; and zero-onset title behavior.
- Record only the live-admitted failures in `PROD_BUG_LOG.md` and promote the
  reusable contracts to the Bug Bible with executable coverage.
- Canonical workflow JSON is expected to remain byte-identical; nevertheless
  run validator, JSON round-trip, link/input audit, and widget-vector audit.
