# Output-tree consolidation — OPERATOR DIRECTIVE: episodes/ + obs/ ONLY

**Operator (2026-06-11 night):** "all output assets should be stored either in
otr/episodes or otr/obs. That's it. We need to fix this first." The live tree has 16
top-level dirs under `<comfy output>/otr/`; two were written DURING tonight's render.
Panel: harden the consolidation contract + migration into build-ready tickets.

## Live inventory (measured 2026-06-11 20:15; newest-file per folder)

| Folder | Files | MB | Newest | Attribution (code-grounded) |
|---|---|---|---|---|
| episodes | 2,157 | 20,345 | TODAY | CORRECT — per-episode tree; videos/composited/upscaled/stills already live UNDER episodes/<id>/ (nodes/_otr_paths.py) |
| obs | 100 | 3,848 | TODAY | CORRECT — flat final-deliverable dir, OBS-watched |
| stills | 377 | 420 | TODAY (mid-render) | ST-3 content-addressed stills CACHE (dispatcher cache-hit materialization; otr_save_to_episode_workspace falls back to otr_stills_dir("") = top level when episode_id empty) |
| tmp | 720 | 352 | TODAY (mid-render) | OTR_TMP sanctioned scratch (launch cmd sets TEMP/TMP/OTR_GPU_LEASE_DIR here; ffmpeg children + atomic-publish staging). 720 files = stale-cleanup insufficient |
| state | 7 | 1 | TODAY | BUG-LOCAL-090 deliberate per-machine state tier (news_history.json etc.; otr_state_dir()) |
| aship / aship_test / _lane1 | 11 | ~0 | 06-07..09 | soak/probe debris (June) |
| audio | 893 | 7,667 | 05-06 | DEAD — pre-v2-audio-lane tree |
| videos | 247 | 356 | 05-02 | DEAD — pre per-episode videos/ move |
| script_gates / blend_test / _legacy_stills / qa_waveforms / qa_frames / portraits | ~73 | ~9 | 04-26..05-10 | DEAD legacy debris |

Code authority: `nodes/_otr_paths.py` is the SINGLE path module (every helper routes
through it; OTR_OUTPUT_DIR env override is the headless pin). Per-episode placement is
already right; the violations are the three SYSTEM tiers (cache/tmp/state) + dead debris.

## Hard constraints

Frozen audio untouched (V-1); determinism; LOUD failures; UTF-8 no BOM; OBS watches
`otr/obs` FLAT (don't break the operator's OBS source); `OTR_OUTPUT_DIR` env override
keeps working; Windows atomic publish needs tmp on the SAME VOLUME as episodes/
(os.replace); the capstone/sweep HYGIENE GATE counts files under server output —
contract changes must update that gate in the same chunk; the content-addressed stills
cache exists to REUSE stills across legs/episodes (the sweep's cache-reuse design) —
killing cross-episode reuse is a real cost, weigh it.

## Questions for the panel (adversarial, concrete)

1. **Placement of the 3 live system tiers under a 2-folder contract.** Options:
   (A) ONE sanctioned dot-dir `otr/.system/{cache,tmp,state}` — assets rule stays
   "episodes|obs only", system internals get exactly one hidden home;
   (B) strict-2: cache -> `episodes/_shared/cache/`, tmp -> `episodes/_shared/tmp/`,
   state -> `episodes/_shared/state/` (inside episodes/, underscore-prefixed);
   (C) per-episode only: cache materializes INTO episodes/<id>/stills (no global cache;
   accept re-render cost), tmp per-episode, state -> .system or config.
   Pick one, defend it against: OBS, hygiene gate simplicity, same-volume rename,
   cross-episode reuse, "the operator opens the folder and SEES only episodes+obs".
2. **Enforcement.** Spec the fail-LOUD guard: a pytest + a soak-gate check that any
   top-level entry under otr/ outside the sanctioned set FAILS the run; plus a
   _otr_paths.py-level assert (no helper may return a path outside the contract).
   Audit requirement: prove no writer bypasses _otr_paths.py (grep/AST sweep for
   hardcoded "otr/<dir>" strings).
3. **Migration + debris.** Order of operations for moving live tiers without breaking
   in-flight episodes; disposition of the ~8.4 GB dead debris (archive-then-delete
   list for the OPERATOR to approve — nothing auto-deletes); a janitor (stale-tmp
   sweep) with age threshold.
4. **Sequencing.** Operator wants it FIRST, but a 7-leg sweep + wan batch + 0-E Phase B
   are queued on the CURRENT tree (hygiene baselines included). Recommend: land as
   ticket OH-1 immediately after the queue drains (hours) vs interrupt now. Justify.

Deliver: the chosen placement option + the enforcement spec + a 4-6 ticket cut
(OH-1..n) a coder window can run, honoring the one-plan rule (tickets live in
3D_TOOLKIT_PLAN.md section 0 LIVE STATUS).

## OPERATOR CLARIFICATION (2026-06-11, supersedes panel preferences)

"That includes ALL ledgers, treatments, stills, video pieces, 3D assets, everything —
in logical subfolders (portraits or whatever they're called) — all in episodes/,
EXCEPT the final video [which goes to obs/]." Binding reading:
- The ASSET OF RECORD for anything episode-scoped lives under `episodes/<id>/<logical
  subfolder>/` (ledger, script/treatment, stills, portraits, per-line video pieces,
  composited/upscaled intermediates, meshes/3D assets, captions, manifests).
- `obs/` = FINAL deliverable videos ONLY (flat, OBS-watched).
- System plumbing (scratch tmp, per-machine state, any content-addressed cache) is
  NOT an asset home: it must be invisible to the operator (hidden), hold only copies
  or transients (never the only copy of an asset), and the panel/judge picks its ONE
  location. Cross-episode reuse (e.g. a stable-cast mesh) is implemented as cache
  COPIES, with the original of record in the minting episode's folder.
