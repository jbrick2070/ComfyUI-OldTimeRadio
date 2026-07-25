# OTR Go-Forward Plan

**Updated:** 2026-07-25 -- CODER A still-plans: r3 panel (codex
`gpt-5.6-sol` high + agy Gemini 3.6 Flash High) + Claude grounding.
Chunk order corrected to S1b -> S0b-core -> S2 -> S3/S0c -> S5 -> S4.
HEAD `79fe4d3f`; no production code touched this session.

**Updated:** 2026-07-25 -- **WAN 8-GB LOW-VRAM LAUNCH CONTRACT IS DONE @
`f914f0a4`** (first item of the rescoped order, CODER A). The 8-GB tier's
render ceiling now travels profile -> `OTR_VideoDirector.max_render_frames`
(appended widget, canonical ships 0 = unpinned) -> v2 policy -> ShotLock ledger
-> `render_driver.build_episode_render_policy` -> `MotionEngineBase.prepare` ->
`eng_wan_ti2v._floor_length`, because `launch.env` can never reach a leg
submitted to an already-booted server. Only `otr_8gb_wan` declares the new
OPTIONAL key `video.max_render_frames`, so every other tier -- including the
qualified 16-GB WAN lane -- is unchanged. Canonical `A66A416B` -> `5377914B`;
11 variants + 4 paired `.env.json` hashes regenerated. Record:
`PBUG-20260723-02`. Live 8-GB requalification leg still OWED.
**OPERATOR BLOCK -- PER-MODEL STILL PLANS (CODER A, live).** Spec of
record: `docs/2026-07-25-still-plans-locked-build-spec.md`; grounding:
`docs/STILL_PLAN_SEED_INVENTORY.md`. S0a / S0a-b / S1 are LANDED, S0b is
FILED-BLOCKED, and the build is at the S2 gate -- per-chunk detail is in
`docs/HANDOFF_LOG.md`, not here. The 2026-07-25 r3 panel (codex
`gpt-5.6-sol` high + agy Gemini 3.6 Flash High, both pins verified) plus
Claude's grounding CORRECTED the plan in four ways; see CURRENT STEP.

**Updated:** 2026-07-24 -- **INDEPENDENT SOURCE BANKS v1 IS DONE. All seven
waves LANDED @ `30358ad1`; the CODER E slot is FREE.** A client authors a
bundle under `user_packs/source_banks/<id>/`, runs
`scripts/otr_check.py bank <path> --activate`, restarts ComfyUI, and their bank
is a selectable peer of the shipped six: discovered and integrity-checked,
admitted in the ONE routing authority, free to own `fetch_source` /
`interpret_source` via the reserved `"self"` entry point, reaching the network
only through the bounded `nodes/_otr_feed_fetch.py` seam, and handed a COMPLETE
ledger by the shared tail's cleanup pass (`nodes/_otr_ledger_cleanup.py`).
Wave 7 was ASSESSED, not built: the `source_bank` COMBO already reads its
choices live from the registry and a bank's own `default_story_model` picks its
pack, so no node, widget, link or canonical-JSON change was ever needed --
canonical is still byte-identical at `A66A416B` after all seven waves. Suite
6403 / Bible 17. `docs/EXTENDING_OTR.md` is the client contract of record.
**OPERATOR RESCOPE 2026-07-24 (supersedes the older queue everywhere in this
file):** the 45-word scene matrix, the 54-case visual-style sweep and the
WHOLE quick-wins block are CUT -- the operator wants coding, not matrices, and
will triage bugs as a batch later. The order is now **WAN 8-GB contract ->
LEAN-MEAN FRONT -> Randomizer A -> `dynamic_story` -> LEAN-MEAN TAIL -> SFX ->
re-observe the parked story bugs.** ENGINE_MATRIX survives the cut as a W6
sub-step, not a standalone chunk. Two story-shaped defects are PARKED, not
closed (see OPEN BUGS). **And every remaining big block must be RE-GROUNDED by
a kibitz arc before it executes -- r3+r4 by default, a full r2->r3->r4 for both
LEAN-MEAN blocks, dropping to r2 anywhere the coding plan itself proves stale.
See STANDING RE-GROUND GATE. These plans are two weeks old and the tree moved
under all of them.**

This file contains only go-forward work, open bugs, and standing operator
contracts. Completed work is NEVER re-described here -- it moves to
`docs/HANDOFF_LOG.md` (history) and `docs/PROD_BUG_LOG.md` (bugs) the session
it ships. Doctrine lives in `docs/PRODUCTION_SPRINT_LESSONS.md`.

## CURRENT VERIFIED HANDOFF -- 2026-07-24

This block is the current source of truth for the overnight qualification.
Nothing in this file is an instruction to reset, stash, delete, or overwrite
user changes.

- Branch: `v2.0-alpha`; HEAD and origin are `30358ad1` (the seven extensibility
  waves are `66e214ec`, `cc69e683`, `84945bc4`, `c97a0e91` + `8c45172d`,
  `1504bb4c` + `3d97a130`, `30358ad1`; the six-bank no-prose-gate retirement
  chunk is `314dd481` below). Per-wave detail lives in
  `docs/HANDOFF_LOG.md`. The worktree is
  CLEAN of task-owned changes -- what remains is `tmp/` scratch (including
  another window's modified `tmp/_chain_720.ps1`, `tmp/_rearm_gate.ps1`,
  `tmp/_status_bake.ps1` -- PRESERVE) and untracked campaign receipts (plus
  `config/profiles/otr_sbcov_1..6.json`, intentionally untracked
  coverage-campaign scratch; nothing in-repo references them, and untracked
  `docs/_bakeoff_*.log.err` + `docs/otr-*.pdf` from an earlier window).
- LANDED @ `30358ad1` (2026-07-24; suite 6403 passed / 27 skipped / 1 xfailed;
  Bible 17; AST/JSON/BOM/zero-byte/UTF-8 clean; canonical byte-identical;
  pushed, HEAD == origin): wave 7, CLOSED BY ASSESSMENT. The `source_bank`
  COMBO on `OTR_LedgerScriptWriter` already reads its choices live from the
  routing registry, and waves 1-3 already made that registry own pack
  directories BY BANK (`_Registry.pack_dirs`), so an activated client bank is
  selectable with no new node, widget, link or canonical change. The chunk
  shipped the missing PROOF (three pins that an activated bundle reaches
  `INPUT_TYPES()["optional"]["source_bank"]`, that its widget value resolves to
  a pack inside its own bundle, and that admitting a bank leaves the canonical
  34-slot positional widget vector untouched) and fixed the one real defect:
  `guide_ref` had NO runtime consumer, so the `+ Add Your Own` row answered a
  click with a dead end while its own text still claimed "the simple_4 pass
  runner does not exist yet". `require_runnable_bank` now appends the row's
  `guide_ref`, and that text names the folder, the CLI, the restart and
  `docs/EXTENDING_OTR.md` (new section 6: what the client sees in ComfyUI).
- All seven extensibility waves and their per-wave findings are recorded in
  `docs/HANDOFF_LOG.md`; `docs/EXTENDING_OTR.md` is the landed client contract.
- Prior root fix at `f150213f`: `nodes/_otr_video_engines/render_driver.py` requires
  an authoritative scene-target manifest only for scene/mesh-consuming shots;
  visualizer-only `viz_mxc_cpu`, `viz_mxc_mandala`, and `viz_camera` lanes may
  execute without one. Regression coverage:
  `tests/test_ledger_cleanup_contracts.py`.
- Verification: full Windows OTR suite `6403 passed, 27 skipped, 1 xfailed`;
  Bug Bible `17 passed, 24 skipped, 3 xfailed`.
- Canonical workflow byte-identical at SHA-256
  `A66A416BFBCAD127356047043C8C07637BC50CACE2CD7D4E0436C7CD80B09CB4`.
- Live media proof: isolated `media_archive@120w` passed with `RESULT SUCCESS`,
  `obs_publish OK`, and non-zero episode/OBS assets. In the monitored run
  `tmp/six_bank_sweep_20260723_205002_331`, `original`, `public_domain`,
  `shakespeare`, and `scifi_news_pro` passed at 120 words. `scifi_news` failed
  closed on provider/context capacity and produced no publish artifact. The
  `scifi_news_pro@120w` pass does not clear its known `requested_output=2800`
  versus provider cap `512` blocker.
- WAN is already canonically qualified and remains closed. LTX remains
  untouched/unqualified until its explicit cases run.
- Overnight monitoring automation is active in the Codex app as
  `otr-overnight-qualification-monitor`. It must continue from the live logs,
  preserve canonical assets, and report terminal receipts or reproduced bugs.
- LANDED @ `314dd481` (2026-07-24; suite 6182 passed / 27 skipped / 1
  xfailed; Bible 17; AST/BOM/zero-byte/canonical-hash gates passed; pushed,
  HEAD == origin): word-fit ceilings /
  candidate ownership retired (length = non-gating telemetry on all six
  routes); provider-capacity whole-artifact output contracts with preserved
  list-subclass markers; `scifi_news` P1/P2/P3/P5 + `scifi_news_pro`
  pitch/treatment/news/script/casting migrated to provider-capacity output (no
  target-derived cap, no +25% missing-END branch); `scifi_news_pro` markup
  acceptance now structural delimiter/order/roster only; placeholder G13 fully
  retired; campaign receipt truth hardened (no PASS without canonical
  `RESULT SUCCESS`); the repair-first plan (explicit P0 slice identity, bounded
  tagged repair context, one direct alternate owner, original post-validator
  reuse, journaled owner/backend/rung/nonce/disposition).

### Immediate next actions

1. Preserve the completed run artifacts and record its 4/5 120-word receipt
   result; do not rerun the known provider-capacity failure as a workaround.
2. Open a coder window on the WAN 8-GB low-VRAM launch contract. It is the
   first item of the rescoped order and needs no GPU to write.
3. For any reproduced failure, fix the owning producer/receipt boundary,
   re-run focused tests, the full Windows suite, and Bug Bible, then commit and
   push the green code chunk to `v2.0-alpha` and verify `HEAD == origin`.
4. Never add fallback assets, truncation, silent resizing, arbitrary provider
   caps, or prose-quality rejection.

## MODEL & CREDIT BUDGET (operator, 2026-07-24 -- read this EVERY window)

Every window states, in its first reply, which rung of this ladder it is on
and why. Pick the cheapest tool that can win; escalate only when the cheaper
rung cannot decide.

**Reset state 2026-07-24: Claude weekly credits FRESH; Codex credits FRESH
(reset taken today). Both pools reset weekly -- front-load heavy coder windows
and the big Codex spends early in the credit week; late-week, drop to the $0
rungs instead of grinding a paid pool dry.**

| Rung | Model / tool | Cost | Use for | Never for |
|---:|---|---|---|---|
| 1 | Local Qwen on the 4060 (`10.55.0.2:1234`, LM Studio/ACPX): `qwen3-coder-30b-a3b-instruct` now; `Qwen2.5-Coder-14B Q4_K_M` as the fast tier once installed | $0 | Read-only FIRST-PASS triage of failures, logs, diffs before any credit spend | Final diagnosis, patches, tests, live qualification (Codex/Claude own those); NEVER loaded on the 5080 (ComfyUI renders only) |
| 2 | agy / Antigravity, `KIBITZ_AGY_MODEL="Gemini 3.6 Flash (High)"` (operator 2026-07-24: 3.6 > 3.5; DISPLAY name exactly -- a wrong id silently kills agy and the arc runs codex-only; check antigravity.log per round) | $0 | Default grounded reviewer for ALL mechanical review; second panelist on every kibitz | -- |
| 3 | Codex CLI `gpt-5.6-sol` (high) | weekly credits | The second opinion of record: two-strikes law (mandatory 3rd-attempt panel), sec-16 + r5 extensibility confirm, pre-execution grounding of big blocks, live-failure kibitz, HANDOFF_CODEX grind delegation | Mechanical review agy can do alone. Verify `codex_model_selected.txt` every arc (stale skill cache once drifted to gpt-5.5 mid-arc unnoticed) |
| 4 | Claude (Cowork, this) | weekly credits | The actual work: planner + coder windows, anchor/judge on every panel, live-run drive | Babysitting renders (the Codex-app overnight monitor owns that); single-small-item windows (batch per the Window packing rules) |
| 5 | Cloud roundtable (OpenRouter) | real $ | Genuine R1 ideas passes only; <$20 autonomy rule applies | Mechanical/grounding review (that is rungs 2-3) |
| 6 | Fable | scarce | Single final gate on a lean-mean epoch commit only (section-9 reality exception) | Anything else |

Production (in-pipeline, all $0/local, offline-first): writers = Mistral-Nemo
(ctx cap 16384) + `gemma-4-12b` (saved runtime-qualified local default);
stills/video-init = `z_image_turbo` (Qwen-Image engine is REMOVED -- keep
Qwen3/Qwen2.5 LLM support and Z-Image's `CLIPLoader(type="qwen_image")`
encoder, unrelated). Cloud writers (Sonnet-4.5 etc.) stay opt-in bake-off
arms, never the default.

Per-window model mapping:

- RENDER / qualification windows: local production models + the Codex-app
  monitor; Claude only to launch and wrap.
- CODER windows (quick-wins, lean-mean): Claude codes; rung-1 Qwen triages
  every failure first; Codex only via the two-strikes law.
- PLANNER window: Claude; the sec-16 + r5 kibitz (codex + agy) is THIS WEEK's
  scheduled Codex spend while both pools are fresh -- it is the operator
  bottleneck on the critical path.
- CODER E extensibility (21-31 d): spans multiple credit weeks -- plan wave
  boundaries at the weekly resets; mid-build Codex only via two-strikes.

## THE LAW (operator, 2026-07-22 -- supersedes anything that disagrees)

> **AN AUDIT MAY IMPROVE A STORY. IT MAY NEVER FAIL ONE FOR LENGTH, LANGUAGE,
> STYLE, VISUAL VOCABULARY, OR QUALITY.**

The sole terminal spoken-prose policy is the shared whole-word safety
authority: profanity, explicit guns/knives/weapons, and explicit
sexual/nudity content. Smoking and benign substrings such as `begun` pass.
Structural JSON/schema/IDs/roster/source-proof/rights/graph/markup/nonempty/
provider-integrity failures remain fail-closed because they protect a usable
ledger rather than judge prose. Across all six banks, requested word length,
actual word count, drift, one-breath estimates, visual/world vocabulary,
noun/POS heuristics, casing/title/honorific style, craft, and quality are
guidance or telemetry only -- they may never reject, reroll, retire, replace,
or block an episode. Same-story LLM cleanup is allowed.

## CURRENT STEP -- PER-MODEL STILL PLANS: S1b, then S0b-core, then S2

The r1->r5 arc converged and S1 landed, but a 2026-07-25 r3 panel + Claude
grounding found FOUR corrections that must be made before any wiring. The
chunk order is now:

**S1b (NEW, do first -- pure parity, blocked on nothing).** S1's
`framing_geometry` strings are PARAPHRASES, not transplants, and spec section
5 makes that field the layer-2 prompt TEXT ("authored TEXT -- the ONE free
field"). Wiring S1 as it stands would silently degrade every prompt in the
system. Concretely lost against `docs/STILL_PLAN_SEED_INVENTORY.md`:
`mesh_fodder` dropped the entire clay-blob clause (unbroken silhouette, plain
seamless mid-grey backdrop, no props, even diffuse frontal light);
`scene_background_plate` dropped "no people, no subject, no characters";
`portrait` dropped "never crop the top of the head" and is the EMPTY STRING on
19 engines. S1b restores every clause VERBATIM from the inventory. The S0a
fixture must stay green -- this is transcription, not design.

**S0b-core (corrected).** Land the routing freeze atomically. THREE
corrections to `docs/S0b_KIBITZ_NEEDED.md` before it is built:
  1. The closed `engine_facts` descriptor `{engine_id, family, provider_side}`
     (spec:230) has NO aspect field, but `resolve_row_aspect`
     (`still_plan_helpers.py:177-189`) needs `engine_render_aspect` /
     `render_aspect` and SILENTLY RETURNS PORTRAIT when absent -- so every
     `inherit_engine` row would resolve portrait, including `cloud_kling_avatar`
     and both wide `_169` HuMos. Add a canonical `render_aspect` field and
     reject missing values instead of falling back.
  2. The frozen-routing prepass as specified does NOT close the defect it is
     named for. `apply_engine_override` (`render_driver.py:2784`) applies only
     `OTR_FORCE_ENGINE_MAP`; the radio-host redirect is a SEPARATE mutation at
     `:1413-1513`. The prepass must freeze each role's FINAL effective engine,
     redirect included, before `validate_and_repair_still_spine`.
  3. The test-literal inventory is stale: ~35 `policy_version=2` sites, not 31
     (`test_hybrid_voice_fit` has none; `test_still_plan_parity` adds five).
     Derive the list mechanically.
  SCOPE (judge call on a panel split): adopt agy's S0b-core / S0c relief, but
  keep `ltx_resolved` FROZEN inside S0b-core -- that answers codex's objection
  that deferring it desynchronizes `when_engine_talking`. Only the
  `eng_ltx_av.assert_usable` mismatch ASSERTION defers to S0c.

**S2 (cutover).** OPERATOR EYEBALL RESOLVED 2026-07-25 -- and it is far
narrower than three docs claimed. There are FOUR HuMo engines; only `humo` and
`humo_1.7B` ship `render_aspect="portrait"`, and `humo_1.7B_169` /
`humo_14B_169` are ALREADY wide (the ComfyUI dropdown labels this to the
operator as "(portrait)" / "(16:9)" -- a visible product contract). Nothing
about HuMo "flips". The S2 delta is FOUR ROLE-CELLS: two portrait HuMo picks x
announcer/music, under the hosts-off DEFAULT, where `_enforce_radio_is_host`
redirects the beat to the WIDE `ltx_audio_in` that actually renders it. With
`OTR_ENABLE_HUMO_HOSTS=1` a portrait HuMo keeps its portrait still. Confirmed
three ways: operator, codex, and agy independently. The old "via the `_169`
siblings' render_aspect" framing in `docs/S2_EYEBALL_REQUEST.md` is WRONG on
mechanism and must be corrected along with the S0a fixture's special_cases
rows.

**S3** shim + stale-prose deletion. **S0c** the ltx_av mismatch gate.

**S5 (NEW, the operator's actual directive).** Operator 2026-07-25: "ensure
that each video path has its own customized still operations." It is NOT met
today. Driving the live registry over all 31 engines yields 14 shared plan
objects but only SIX distinct signatures and SIX distinct structures -- meaning
the framing prose adds ZERO per-engine differentiation, and 19 engines
(`wan_ti2v`, `google_*`, `still_*`, `word_razzle`, `cloud_*`, `ltx_8gb`) share
one identical signature whose portrait row is empty. S5 diverges the restored
clauses per engine so an i2v engine whose still IS the init frame, a t2v engine
whose still is optional, and a Ken Burns pan stop receiving identical
instructions. S5 CHANGES PROMPTS: it needs its own acceptance and must land
after the wiring, never inside a parity chunk.

**S4** two fresh-boot live legs (default route + forced HuMo bookend).

Gate: r4 convergence at CURRENT HEAD on the corrected plan before code. Both
r3 panelists explicitly rejected Path B (S2-first against live env); do not
revive it.

Operator rescope 2026-07-24 -- the rest of the live order:

1. ~~**WAN 8-GB low-VRAM launch contract**~~ -- DONE @ `f914f0a4`
   (`PBUG-20260723-02`). The live 8-GB requalification leg is still owed and
   belongs to a render window.
2. **LEAN-MEAN FRONT** (`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 ->
   W5+SW4 -> C1-C5`), with `docs/ENGINE_MATRIX.md` folded in as a W6 sub-step.
3. **Randomizer A -> `dynamic_story`.**
4. **LEAN-MEAN TAIL** (`SW1/SW2/SW3 -> C6 -> C7 -> W8`).
5. **SFX** (still behind the Timeline Cue Ledger C0/C1 gate).
6. **Re-observe the parked story bugs** -- after SFX, see whether they still
   occur at that HEAD (see OPEN BUGS).

Standing constraints, unchanged by the rescope: keep the RTX 5080 free for
ComfyUI; the 4060 Qwen endpoint is a read-only QA reviewer, not a production
ComfyUI slot; six-bank requalification (canonical `RESULT SUCCESS`,
`obs_publish OK`, exact episode/OBS assets, and the archival final's parent
equal to the ledger-owned episode root -- PBUG-20260720-05 acceptance) is
still owed whenever a render window next opens, and was NOT cut.

## OPERATOR CAMPAIGN QUEUE -- 2026-07-23 (PAUSED)

The overnight media qualification was aborted after the WAN lane and the LTX
visual-style sweep stalled at case 6/54. No new GPU run is authorized while
confirmed bugs are being closed. Failure inventory / staging record:
`docs/2026-07-23-video-failure-inventory.md`.

Bug-first order before resuming:

1. Requalify receipt truth against the captured six-bank stdout and confirm
   the old false PASS is now a terminal FAIL (fix LANDED @ `314dd481`;
   needs live confirmation only).
2. Make the image phase own every required scene-still, mesh-fodder, and
   opening-still target, with a complete target/path receipt before video
   dispatch; no text-only or dark-floor degradation for a missing required
   still. (`f150213f` fixed the no-still visualizer spine handoff; the
   scene/mesh-consuming ownership contract is the remaining piece.)
3. Make the WAN 8-GB profile carry its actual 832x480/17-frame low-VRAM
   launch contract instead of falling back to the 177-frame default.
4. Then provider-capacity and SciFi News markup-repair residuals.

Remaining media qualification (CUT DOWN by the operator rescope 2026-07-24 --
the 45-word model-coverage matrix and the 54-case visual-style sweep are
DELETED, not deferred; reviving either is a new operator decision):

1. Six 120-word canonical runs in bank order `media_archive`, `original`,
   `public_domain`, `shakespeare`, `scifi_news`, `scifi_news_pro`:
   `google/gemma-4-12b-it` both writer slots, `viz_mxc_cpu` /
   `viz_mxc_mandala` / `viz_camera` video slots, `z_image_turbo` all three
   image slots. (4/5 of the 120w receipts are already banked from
   `tmp/six_bank_sweep_20260723_205002_331`; `scifi_news` is the open FAIL.)
   This is the ONLY surviving matrix.

The coordinator keeps one canonical API prompt active at a time, reloads
`workflows/otr_canonical.json` for every case, and records each prompt and
receipt under `tmp/`.

## OPEN BUGS / DEFECTS (live, not yet closed)

MECHANICAL defects survive story-engine churn; STORY-QUALITY judgments do not.
That split is why the two eyeball-era entries below are PARKED rather than
listed as live.

- **`scifi_news` P0 convergence defect** -- both 120w and 320w legs fail in P0
  after two attempts on non-literal fact source spans; provider/model
  convergence, extends BUG-11.35. NOT a word/length gate. Blocks the last 120w
  receipt and the `scifi_news` live reverify (PBUGs 20260712-22/23/24/25, fixed
  in tree, reverify still owed).
- **`scifi_news_pro` provider capacity** -- `requested_output=2800` vs
  provider cap `512`; the whole-artifact retry contracts LANDED @ `314dd481`
  are the base; the residual fix is now unblocked. Related independent items: the P9 8K
  structured-capacity follow-up + the GGUF structured-enforcement NEWBUG. Do
  not raise the minimum word target as a capacity workaround.
- **WAN 8-GB low-VRAM launch contract** -- FIRST item of the rescoped order.
- **Image-phase still ownership** -- bug-first item 2 above.
- **`eng_ltx_video._use_i2v` contradicts fail-closed** (found 2026-07-25, r3
  panel, grounded). With I2V enabled and the init image missing it LOGS and
  degrades to the text-to-video path (`eng_ltx_video.py:559-572`), while
  `render_driver.py:1801-1817` RAISES `RenderError` "NO FALLBACK to text-only
  rendering" on that same state. Two contradictory policies; whichever fires
  first wins. Static finding at HEAD -- needs a live reproduction before it
  becomes a PBUG row.
- **Four env-read sites missing from the S0b inventory** (r3 panel, grounded):
  `eng_ltx_video.py:541-564` (`OTR_ENABLE_LTX_I2V`), `render_driver.py:1176-1203`
  and `otr_meta_brief_image_prompt.py:297-300` (`OTR_ENABLE_HUMO_HOSTS`), and
  `eng_ltx_av.py:352-353,403-432` (recipe/UNET re-read outside `assert_usable`).

**PARKED -- unverified at HEAD, re-observe AFTER SFX (operator 2026-07-24).**
Both were eyeball observations against a story engine that has since had its
LLM vetoes ripped, THE LAW imposed (2026-07-22), six banks renamed onto new
packs, word-fit ceilings retired, the repair-first plan landed, and a ledger
cleanup pass added. Neither has a reproduction at current HEAD, and under the
standing rule a finding with no reproduction is not a row. Do NOT schedule
coder time against either. They are settled by the operator eyeballing a real
render leg after SFX: still there -> re-admit as a FRESH dated row with that
leg as evidence; gone -> the LAW-era work already fixed it, tombstone it.

- **Announcer framing defect** (`docs/2026-07-11-announcer-framing-defect.md`)
  -- PARKED. Episodes START a story instead of admitting you into one; the
  announcer takes debate turns instead of framing. Operator eyeball
  2026-07-11. If it survives re-observation the fix is still seam + score
  contract + fail-closed validator, never Python authorship.
- **Name-splice defect #2** -- PARKED. v4-campaign Phase 0 record in
  HANDOFF_LOG; its timebox predates THE LAW.

- **PBUG-20260710-07** -- root fix shipped; stays ROOT-OPEN in the log until
  ratified at the next operator fan-out (green codex leg `c1f3891f` is the
  retire candidate).
- **Phase-2 de-naming** (module filenames, `meta[]` ledger keys, wire-schema
  `.v4` literals) -- DEFERRED, operator-flagged, from the keep-6 rename.

## Coder queue (re-grounded 2026-07-24)

One coder window at a time; every chunk = focused tests + full suite + Bug
Bible + commit AND push + `HEAD == origin/v2.0-alpha`.

```text
WAN 8-GB low-VRAM launch contract        (no re-ground needed: a live 2026-07-23
                                          defect, not an old plan)
  -> [r2->r3->r4] LEAN-MEAN FRONT (W0->W1->W2->W3->W4a->W4b->W7->W6->W5+SW4->C1-C5)
       (operator 2026-07-24: lean-mean starts at r2, NOT r3 -- its coding plan
        is a file-and-line kill list, the most perishable thing a plan can be;
        ENGINE_MATRIX.md is a W6 SUB-STEP now, not a standalone chunk)
  -> [r3+r4] Randomizer A
  -> [r3+r4] dynamic_story           (wiring only -- rev-5 DESIGN stays FINAL)
  -> [r2->r3->r4] LEAN-MEAN TAIL (SW1/SW2/SW3 -> C6 -> C7 -> W8)
       (same doc, same r2 rule; run its arc when the TAIL opens, not before --
        every block above edits the very writer this block splits)
  -> [R4.1 refit = its re-ground] SFX campaign (after Timeline Cue Ledger C0/C1)
  -> re-observe the PARKED story bugs; batch-triage whatever is left
```

The bracket is the STANDING RE-GROUND GATE below. Every one of these plans was
written against a tree that no longer exists. Default entry is r3 (wiring);
drop to r2 if the coding plan itself is wrong; if in doubt, start at r2. **Both
LEAN-MEAN blocks are pinned to a full r2 -> r3 -> r4 by operator decision --
their doubt is already settled, do not re-argue it down to r3.** No block
executes without an r4 convergence at current HEAD.

CUT by the operator 2026-07-24 and NOT to be re-derived by a later window: the
45-word scene matrix, the 54-case visual-style sweep, and the entire
quick-wins block. Image-phase still ownership and the six-bank requalification
were not cut -- they stay in OPEN BUGS / the campaign queue and get picked up
whenever a render window opens.

### Quick-wins block -- CUT 2026-07-24 (operator)

The whole block is gone. The operator's call, verbatim in intent: "we will
triage more bugs later" -- the block was a schedule, and ripping a schedule
does not rip the underlying defects. Everything in it that was a real bug
still lives in OPEN BUGS above; everything in it that was a nice-to-have is
simply not being built. Do NOT re-derive this table from git history.

ONE item survived the cut, folded into LEAN-MEAN W6 as a sub-step rather than
kept as a standalone chunk:

- **`docs/ENGINE_MATRIX.md`** -- emit from the three live CAPABILITIES
  registries per the existing generator pattern (`build_variants.py`
  ~:276-338): write during `--all` / explicit emit; `--check` regenerates in
  memory and FAILS on drift without writing. Columns + stable ordering; link
  from README. The lean-mean doc (`:301-304`) only needs W6's README policy
  line to link it, so this is an ordering preference the operator set on
  2026-07-10 -- NOT a hard technical dependency. W6 executes without it; the
  README link is what suffers. Estimate 0.5-1 d.

Also recorded so a later window does not re-open them: quick-win 6
(`scifi_news_pro` C5 consumers) was already CLOSED IN CODE under
PBUG-20260720-04. The `scifi_news` live reverify (PBUGs 20260712-22/23/24/25)
is not lost either -- it moved into the `scifi_news` P0 convergence row in
OPEN BUGS, which is what actually blocks it.

### STANDING RE-GROUND GATE -- r3/r4 before ANY remaining block (operator 2026-07-24)

Every remaining big block was planned on a tree that no longer exists. Since
those docs were written the LLM vetoes were ripped, THE LAW landed, six banks
were renamed onto new packs, word-fit ceilings were retired, the whole
extensibility build shipped (seven waves, a new routing authority, a new
network seam, a new ledger-cleanup pass in the writer tail), and the suite grew
past 6,400. A plan's line cites, seam names and file inventories are the FIRST
things to rot, and every one of these blocks is a rip or a rewire that acts on
exactly those.

**THE GATE, in the operator's words: run an r3-r4 for all remaining blocks; if
issues turn up go back to r2; if in doubt, restart at r2.** Concretely:

- **Default entry point is `r3` (wiring).** These plans already have an r1
  (arc) and an r2 (coding plan) on record, so the cheap re-ground is the wiring
  round run against CURRENT code, followed by `r4` (convergence). Use the local
  panel (`/kibitz`: codex `gpt-5.6-sol` high + agy) -- it crawls the REAL repo,
  which is the whole point here.
- **Drop to `r2` when r3 finds the CODING PLAN wrong, not just the line
  numbers.** Stale cites are an r3 fix. A seam that no longer exists, an
  authority that moved, a step whose precondition another build already
  satisfied or destroyed -- that invalidates the coding plan itself, and
  patching an r2 from inside an r3 produces a plan nobody reviewed.
- **If in doubt, start at r2.** A wasted r2 costs one panel round; executing a
  stale coding plan costs a day of rips against the wrong file list, and the
  rips are the hard kind to unwind.
- **No block executes without an r4 convergence at current HEAD.** Record the
  run under `kibitz-runs/<date>-<block>-r<N>/` and cite it in the block entry
  below when it lands, so the next window can see how fresh the grounding is.
- **OPERATOR PIN: both LEAN-MEAN blocks run a FULL `r2 -> r3 -> r4`**, not the
  r3 default. Their doubt is already settled -- a later window must not
  re-argue them down to r3 to save a round.
- **Credit note:** this is rung 2-3 work (agy is $0; codex is weekly credits).
  Roughly ten panel rounds across the remaining blocks (2 lean-mean arcs at
  three rounds each, plus r3+r4 for randomizer and dynamic_story) is a real
  Codex spend -- front-load it early in a credit week, run the blocks' arcs
  when their block opens rather than all at once, and never let a stale
  `codex_model_selected.txt` silently drop an arc to codex-only or to the wrong
  model.

### Big blocks (in ROADMAP-ratified order)

1. **LEAN-MEAN FRONT** (`W0 -> W1 -> W2 -> W3 -> W4a -> W4b -> W7 -> W6 ->
   W5+SW4 -> C1-C5`) -- `docs/2026-07-10-lean-mean-rip-final.md`, D-1..D-6
   RATIFIED. **RE-GROUND: FULL `r2 -> r3 -> r4`, pinned by the operator
   2026-07-24 -- do NOT enter at r3.** The reasoning is the block's own nature:
   it is a deletion campaign whose entire value IS its file-and-line kill
   inventory, which is the most perishable thing a plan can carry. Its own
   header already declares five stale areas, and since it was written the
   extensibility build added modules, moved the writer tail, and grew the suite
   past 6,400 -- so the question is not "do the line numbers still point at the
   right code", it is "is this still the right code to delete". That is an r2
   question. Existing drift-check items fold into the r2 brief (SW-3
   news_ingest re-survey, W6 keep-list adds, W7 tombstone re-triage, R-7
   re-grep; SW-1 writer re-survey waits for the TAIL). ENGINE_MATRIX is now a
   W6 SUB-STEP, not a separate precondition. Dedicated window; multi-day.
   SECOND ITEM in the rescoped order, after the WAN 8-GB contract.
2. **Randomizer Rolls Design A** --
   `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`. NO LONGER GATED --
   extensibility landed, and its `_otr_lane_specs` authority was ABSORBED by
   that build, so this shrinks to `_otr_bank_roll` + eligibility. **RE-GROUND:
   r3 + r4 REQUIRED.** Note what the doc's own filename admits -- it is an r2
   coding plan that NEVER got an r3 or r4, so this is the arc completing, not
   repeating. Its r3 brief must carry two known deltas: the absorbed
   `_otr_lane_specs` authority, and that the bank list is now a LIVE registry
   read (`list_bank_ids()` can return a CLIENT bank; eligibility must treat one
   as an ordinary peer) rather than a six-row literal. 1-2 d + 1 GPU day.
3. **`dynamic_story` visual direction** -- rev-5 FINAL; roster-agnostic;
   re-derive IDs at build. After the randomizer. **RE-GROUND: r3 + r4
   REQUIRED, and the "do not rerun panels" rule still holds -- these are not in
   conflict.** That rule protects the DESIGN (the r1 arc: what the feature
   should be, already settled over five revisions). r3 asks a different
   question -- does this design still WIRE to the code that exists today -- and
   the roster, the routing authority and the writer tail have all moved since
   rev-5. Re-litigating the design is forbidden; re-grounding the wiring is
   mandatory. 5-9 coder-days + 2-4 GPU days.
4. **LEAN-MEAN TAIL** (`SW1/SW2/SW3 -> C6 -> C7 -> W8`) -- the writer/widget
   structural split, REQUIRED by ROADMAP to come after blocks 2-3.
   **RE-GROUND: FULL `r2 -> r3 -> r4` (same doc as the FRONT, same operator
   pin), run WHEN THE TAIL OPENS -- not now.** Blocks 1-3 all edit the very
   writer this block then splits, so an arc run today would ground against a
   writer that will not exist by the time it executes; running it early is
   worse than not running it, because it produces a confident stale plan. SW-1's
   full seam re-survey is part of that arc, against the then-current writer.
5. **SFX campaign** (after the Timeline Cue Ledger C0/C1 gate) -- **RE-GROUND:
   the R4.1 refit already IS this gate.** The generated-SFX R4 candidate stays
   local/ignored evidence until it is re-grounded into a tracked current-HEAD
   R4.1 plan; treat that refit as the r3/r4 pass for this block rather than
   scheduling a second one. Sequencing + scope contract live in `ROADMAP.md`
   (no second SFX queue, no library fallback).

Open judgment question (render-window, not coder-slot): the LOCAL
mistral/gemma writer matrix -- the Sonnet arm of the creative-writer question
is answered (record: `docs/2026-07-17-model-bakeoff-scoreboard.md`); the local
roster comparison never ran.

## Window packing (credit discipline -- one line starts any window)

Starting any window costs the same boot context, so BATCH chunks per window
and never open one for a single small item. Every window starts by pasting
its one-line kickoff -- the `otr-handoff` skill reads this file + git and
states the current step. No manual context handoff, ever. This planner window
keeps GO_FORWARD + HANDOFF_LOG current; coder windows never write plans
(window-roles rule).

| Window | Scope | Model rung (see MODEL & CREDIT BUDGET) | Gate | Size |
|---|---|---|---|---|
| RENDER | finish the six-bank 120w wrap ONLY (the 45w matrix and 54-case sweep are CUT); fillers: cpu-tier smoke + nv50 re-soak | local production + Codex-app monitor | opens whenever the operator wants a live leg | GPU days |
| CODER A "still-plans" | WAN 8-GB contract DONE @ `f914f0a4`; the r1->r5 still-plan arc DONE and CONVERGED @ `84328aa1`. **NEXT: execute `docs/2026-07-25-still-plans-locked-build-spec.md` in order -- S0a, S0b, S1, S2, S3, then S4 live.** No further panel round is owed. | Claude codes, Qwen triages, codex on 3rd strike | UNGATED (S2 needs the operator eyeball) | multi-day |
| ~~CODER B~~ | quick-wins harness window -- **DISSOLVED** by the 2026-07-24 rescope (its whole scope was quick-wins) | -- | -- | -- |
| ~~CODER C~~ | quick-wins foundations window -- **DISSOLVED** by the 2026-07-24 rescope; ENGINE_MATRIX moved into CODER D's W6 | -- | -- | -- |
| CODER D "lean-mean front" | **FULL `r2 -> r3 -> r4` kibitz arc FIRST** (operator pin), then W0 .. C1-C5 with ENGINE_MATRIX as a W6 sub-step. The arc is the window's first job, not a formality -- if r2 says the kill list is wrong, the window's output is a new r2, not a rip. | Claude codes + judges; kibitz = codex `gpt-5.6-sol` high + agy | after A; NO rip before r4 converges at HEAD | multi-day |
| PLANNER | extensibility hardening + `docs/EXTENDING_OTR.md` DONE 2026-07-24; NEXT = Bug Bible operator fan-out + the `check_compatibility` fork; plan upkeep | rungs 2-4 | parallel with D | docs |
| ~~CODER E~~ | independent client-authored source banks v1 -- **ALL SEVEN WAVES DONE @ `30358ad1`**; slot RETIRED, do not reopen (deferred power-user tiers are a NEW block, not this one) | -- | -- | -- |
| CODER F | **r3 + r4 arc per block FIRST**, then Randomizer A -> `dynamic_story`. For `dynamic_story` the arc is WIRING ONLY -- rev-5's design stays FINAL, do not rerun the design panels. | Claude codes + judges; kibitz = codex + agy | after D; NO code before r4 converges at HEAD | ~6-11 d |
| CODER G "lean-mean tail" | **FULL `r2 -> r3 -> r4` arc FIRST, run HERE and not earlier** (every block before this one edits the writer this block splits), then SW1-SW3, C6, C7, W8 | Claude; kibitz = codex + agy; Fable single final epoch gate | after F; NO split before r4 converges at HEAD | multi-day |

Kickoff lines (paste as the FIRST message of the new window; swap the letter):

> resume the OTR build -- you are CODER WINDOW A per GO_FORWARD "Window
> packing"; execute your scope in order, one green pushed chunk at a time,
> and state your MODEL & CREDIT BUDGET rung first.

## Parallel lane -- no coder slot required

- **Bug Bible operator fan-out** -- 9+ closed candidates + the
  duplicate-legacy_id cleanup waiting on one fan-out session.
- **Render-window fillers:** cpu-tier smoke (needs the google image lane or
  stills) + nv50 re-soak -- the two open portability remainders; release QA
  validation time, not coding.
- **SFX R4.1 re-ground** (0.5-1 docs day): re-ground the local generated-SFX
  R4 candidate into a tracked current-HEAD R4.1 plan. Sequencing + scope
  contract live in `ROADMAP.md` (Timeline Cue Ledger C0/C1 gate first; no
  second SFX queue, no library fallback).
- **Operator-promotable option:** SFX C0 (per-line WAV stems + transcript
  drift report) is independently shippable per ROADMAP but stays parked
  unless explicitly promoted.

## Bug Bible promotion field -- pending actions only

| Record | Pending action |
|---|---|
| `PBUG-20260712-22/23/24/25` | Live reverify -- blocked by the `scifi_news` P0 convergence defect, then fan-out |
| `PBUG-20260712-18/19/26` + `PBUG-20260713-15..18` + `-20` | Awaiting the next operator Bible fan-out (overlap check + approval) |
| `PBUG-20260713-19` | Live requalification pending (promoted BUG-05.11) |
| duplicate-id cleanup | Same fan-out: BUG-11.54 legacy_id -> `PBUG-20260713-21`; verify the acronym-union rule's legacy_id (both Bible rows cite `-10`; see the log's renumber note) |
| historical `PBUG-20260711-18` | Keep as a standing context/cap engineering risk (its quick-win-9 home was cut 2026-07-24); never eligible from static evidence |
| `PBUG-20260710-07` | Ratify retirement at the next fan-out (green codex leg `c1f3891f`) |

The active production-fix owner updates `docs/PROD_BUG_LOG.md`; the approval
queue is `docs/BUG_BIBLE_PROMOTION_QUEUE.md`; no plan review or invented
fixture creates a row.

## Validation and handoff law

- Current whole-tree receipt (2026-07-24): full Windows suite `6403 passed /
  27 skipped / 1 xfailed`; Bug Bible `17 passed / 24 skipped / 3 xfailed`.
  Detail in HANDOFF_LOG.
- Every code chunk: focused tests, full Windows suite, Bug Bible,
  AST/JSON/BOM/zero-byte checks, commit, push, verify
  `HEAD == origin/v2.0-alpha`.
- Every node/widget/link/schema change edits `workflows/otr_canonical.json`
  in the same commit and runs `OTR_WorkflowValidator`, JSON round-trip,
  strict link/input, live widget-vector, and generated-variant audits.
- Reset selectively before every headless run; never blanket-kill Python.
  Every run loads the canonical workflow and writes directly to canonical
  episode/OBS paths. Asset existence, not resident VRAM, proves completion.
- One coder edits code or `workflows/otr_canonical.json` at a time; read-only
  audits and documentation may run in parallel. HANDOFF_LOG + this file are
  the only tracking surfaces (the otr-build-tracker artifact is RETIRED).

## Open risks

- Extensibility v1 is DONE, so it no longer constrains randomizer /
  dynamic_story sequencing. Deferred power-user tiers (client own-runner +
  staging, dependency manifest, standalone story_rules) are explicitly OUT of
  v1 and are a NEW block if the operator ever wants them -- not a reopening of
  CODER E. NO CLIENT BANK HAS EVER RUN LIVE: every wave is proven by the suite
  and by contract tests, and the first real client bundle is still an unproven
  path end to end (fetch -> interpret -> writer -> cleanup -> tail -> publish).
  Treat the first live client-bank leg as a qualification, not a formality.
- CLIENT-AUTHORED PYTHON executes in-process (wave 3). The posture that must
  hold in every future change: `--activate` is the consent act; the seam fails
  LOUD (`UserBankExecutionError`) and never substitutes; client code never
  touches the canonical ledger; owner IDENTITY is verified so a bank can only
  run its OWN bundle; the shipped fetcher/interpreter registries are never
  widened to admit a client id. Do not relax any of these for convenience.
- The client-facing surface is now LIVE TEXT, not just docs: the
  `custom_source_bank` row's `guide_ref` is raised to the operator by
  `require_runnable_bank`, and the `source_bank` tooltip repeats it. Any future
  change to the activation path (folder name, CLI verb, restart behaviour) must
  update `nodes/story_packs/banks.json`, that tooltip and
  `docs/EXTENDING_OTR.md` together, or the product will confidently instruct
  clients to do the wrong thing.
- **`check_compatibility` is RESERVED, not wired (wave-4 decision, kibitz
  r3 codex `gpt-5.6-sol` high + r4 agy Gemini 3.6 Flash High, Claude judge).**
  No request type, no decision type, no runtime consumer exists, so activation
  does not inspect it -- not even for callability -- and `EXTENDING_OTR.md`
  now calls it a reserved name instead of "NOT YET WIRED". `COMPAT_ENTRY_ATTR`
  is left INERT in `BUNDLE_ENTRY_ATTRS` with a comment saying so. **Operator /
  planner decision flagged, NOW WITH A 2-of-2 RECOMMENDATION TO RIP
  (2026-07-24, operator-directed consult; codex `gpt-5.6-sol` high and Fable,
  independently, no shared context; Claude grounded both against the tree):**
  the argument that decided it is that Option A's stated benefit is FALSE --
  `BUNDLE_ENTRY_ATTRS` constrains what OTR-side code may request from
  `bundle_entry_point()`, it reserves nothing against clients, and activation
  provably ignores whatever a client puts under that name
  (`tests/test_otr_check_cli.py:335` asserts a bundle whose
  `check_compatibility` is a plain integer activates). The only artifact that
  reserves the name is the `EXTENDING_OTR.md` paragraph, which exists either
  way; the constant's sole executable effect is to legalize a call nobody
  makes. Verified blast radius if ripped: ~5 code sites, 2 test files, 3 docs;
  no workflow JSON, no routing, no source-payload consumer. Case AGAINST,
  stated by both: churn on landed green code for zero behaviour change, the
  constant is loudly commented inert and a test documents the inertness, and
  the plan of record already names the future consumer (randomizer
  eligibility), so it may be re-added within a wave or two. STILL NOT A CODER
  CHUNK -- the rip touches landed wave-3/4 code and the plan of record's
  "fetch_source + interpret_source + check_compatibility" line. Either ratify
  the inert constant or schedule the rip as a planner chunk. (The one piece
  already fixed @ `8c45172d`, correct under either answer: the `missing_module`
  quarantine message demanded a `check_compatibility` the code has never
  required. Both panelists found it independently. Proposed doctrine line: a
  name published to clients before its consumer exists lives in the
  client-facing DOC as "reserved, no contract, ignored if defined" and nowhere
  in executable code, because code that names an interface is read as
  enforcing it.)
- **The ledger-cleanup pass now runs on EVERY bank, not just client banks**
  (wave 6, `3d97a130`). It is a no-op on a complete ledger and costs no LLM
  call there, but two shipped-lane behaviours did change and are worth watching
  on the next live legs: (a) unsafe spoken language on a
  `content_owned_readonly` bank is now REPAIRED at the writer tail instead of
  reaching G9 untouched, so a leg that used to die at freeze may now ship a
  sanitized line; (b) a blank `meta.episode_title` is now filled at the tail
  instead of exploding later in `otr_credits_roll`. Both are the intended
  direction under THE LAW; neither has a live receipt yet.
- Lean-mean front/tail drift: the tail's SW-1 re-survey is mandatory against
  the then-current writer. Never interleave the two campaigns in one window.
- No code lands mid-sweep of an active qualification campaign (uniform-code
  confound -- the 420-rung lesson).
- The active campaigns may surface new lane defects; the campaign window owns
  admitting PBUGs (new-bug problem-statement rule applies).
- `dynamic_story` touches the writer, the visual-style authority and the
  canonical workflow; it re-derives the live JSON at build. It is now the only
  claimant on those surfaces (extensibility has released them).
- Generated-SFX R4 stays local/ignored evidence until the tracked R4.1 refit
  lands; it is not an executable queue.

## Tombstones (do not re-derive; records in HANDOFF_LOG + PROD_BUG_LOG)

Keep-6 bank rename (six de-versioned banks; default `scifi_news`,
local/offline-first) -- LLM veto rip + THE LAW -- roster trim + Sonnet-bake-off
rip (science_news family, `_v2` lanes, scifi_sonnet retired) -- v4 improvement
campaign banks #2-#5 PARKED (superseded by the rename + THE LAW; revive only
by operator decision; plan of record `docs/2026-07-17-v4-campaign/final.md`) --
codex56sol attempt telemetry + PBUG-20260712-17 root fix -- fresh two-matrix
bakeoff -- Qwen-Image still engine (removed 2026-07-23) -- word-fit ceilings /
candidate campaigns -- style-dropdown four-surfaces -- otr-build-tracker
artifact -- `tencent/hy3:free` panel seat (expired 2026-07-21) --
**the 45-word scene matrix, the 54-case visual-style sweep, and the entire
quick-wins block (CUT by the operator 2026-07-24: coding over matrices, bugs
triaged as a batch later; ENGINE_MATRIX survived as a Lean-Mean W6 sub-step,
CODER B and CODER C dissolved with the block)** --
**independent client-authored source banks v1 (all seven waves, CODER E,
2026-07-24 @ `30358ad1`; contract `docs/EXTENDING_OTR.md`; w7 closed by
assessment -- no widget was needed and none was invented)** -- the retired
Path-A/B user-source-lanes architecture.

## Pointers

- `ROADMAP.md` (dependency edges; lean-mean pin self-declares stale cites)
- `docs/PRODUCTION_SPRINT_LESSONS.md` (incl. lesson 24 lost-anchor; 25 bank-teardown)
- `docs/SOURCE_BANK_PREFLIGHT.md` -- add-a-bank gate + the Teardown protocol
- `docs/PROD_BUG_LOG.md` / `docs/BUG_BIBLE_PROMOTION_QUEUE.md`
- `docs/HANDOFF_LOG.md` (all completed-work history, newest at top)
- `docs/2026-07-23-video-failure-inventory.md` (campaign staging record)
- `docs/2026-07-15-720-bakeoff-verdict.md` (KEEP/IMPROVE + open items)
- `docs/2026-07-17-model-bakeoff-scoreboard.md` (writer-model verdict)
- `docs/EXTENDING_OTR.md` (LANDED client contract: add your own source bank)
- `docs/2026-07-24-independent-source-banks-v1-plan.md` (extensibility plan -- DELIVERED)
- `docs/2026-07-12-user-source-lanes-architecture.md` (SUPERSEDED -- Path-A/B decision log)
- `docs/2026-07-10-lean-mean-rip-final.md` (drift-check header 2026-07-15)
- `docs/2026-07-12-randomizer-rolls-r2-coding-plan.md`
- `docs/2026-07-12-dynamic-story-visual-scope.md`
- `docs/2026-07-10-llm-first-story-edit-pass.md` (X1-X4 live remainder)
- `docs/2026-07-11-announcer-framing-defect.md` (OPEN)
- `docs/2026-07-11-timeline-cue-ledger.md`
- `workflows/otr_canonical.json`
