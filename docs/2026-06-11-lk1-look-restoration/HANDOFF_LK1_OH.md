# LK-1 + OH Session Handoff — 2026-06-11

## Session summary

Coder window. Two parallel tracks completed.

---

## Part 1: LK-1 — LTX look restoration

### What was done

- **OH-0..3 first** (output-tree consolidation) per operator directive.
- Acceptance contact sheets (A/B: text-only vs still-conditioned) produced for b000/b001/b005.
- Operator: "they look ok but no animation."
- Bug log research → BUG-LOCAL-095 (LTXVImgToVideo freeze-frame) + BUG-LOCAL-112 (prompt dilution kills motion).

### Bugs fixed @ aba0c5a (2026-06-11)

**BUG-LOCAL-095** — `LTXVImgToVideo` with `strength=1.0` encodes the init image into ALL latent frames, freezing every frame. Fix: switched to `LTXVImgToVideoConditionOnly` in `_node_candidates_i2v()`. That node outputs `(CONDITIONING, CONDITIONING)` only; no latent produced. The existing `EmptyLTXVLatentVideo` latent node is KEPT and feeds the sampler as pure noise → real motion denoising.

**BUG-LOCAL-112** — `finish_visual_prompt(max_chars=240)` was too long; LTX-2B's motion signal dilutes past ~188 chars. Fix: `max_chars=188` in `render_driver.py`.

**Files changed:**
- `nodes/_otr_video_engines/eng_ltx_video.py` — `_node_candidates_i2v()`, `_build_graph_i2v()`
- `nodes/_otr_video_engines/render_driver.py` — max_chars 240 → 188
- `tests/test_video_motion.py` — topology assertions updated

Suite: 4128 pass / 0 fail. Bug Bible green (16 pass).

### Still pending for Part 1

- **LTX motion eyeball render** — need ComfyUI Desktop RESTART to load new code, then run a 30w headless smoke with `OTR_ENABLE_LTX_VIDEO=1` to confirm real motion renders. Contact sheets for operator eyeball.

---

## Part 2: OH — Output-tree consolidation

### What was done (OH-0..3)

| Ticket | Status |
|--------|--------|
| OH-0: _otr_paths.py shared helpers | DONE — contract enforced in capstone |
| OH-1: episodes/_shared/ cache tier | DONE — portrait_ledger.py uses _shared/cache |
| OH-2: hygiene gate in capstone | DONE — 29 contract tests pass |
| OH-3: janitor module | DONE — wired at boot + post-publish |

### Still pending

| Ticket | Status |
|--------|--------|
| OH-4: live migration (stills + state → _shared, attic) | **PENDING OPERATOR GO** — 14 entries, ~8.2 GB; dry-run table produced |
| OH-5: docs + tracker + section-0 ticks | After OH-4 |

To proceed with OH-4: operator says "go" → run `scripts/_otr_migrate_output_tree.py --live` → verify, commit+push.

---

## Git state

| Commit | Contents |
|--------|----------|
| aba0c5a | BUG-LOCAL-095+112 fixes (ConditionOnly + max_chars 188) |
| 68c6982 | BUG-LOG entries for BUG-LOCAL-095+112 |

HEAD == origin/v2.0-alpha. No BOM, no 0-byte, AST clean.

---

## Next actions (operator)

1. **RESTART ComfyUI Desktop** to load the new ConditionOnly code.
2. **LTX motion verify**: run a 30w headless smoke with `OTR_ENABLE_LTX_VIDEO=1`, eyeball the clips for real motion.
3. **OH-4 GO**: say "go OH-4" → coder runs the live migration.
4. After OH-4 + eyeball pass: run a full 30w episode render end-to-end with still-conditioned LTX clips.
