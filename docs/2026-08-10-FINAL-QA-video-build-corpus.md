# FINAL QA PASS - OTR video-build corpus

Date: 2026-08-10

Status: **QA-ONLY; NOT IMPLEMENTATION-READY; AWAITING REVIEW**

Mutation boundary: this report is the only intended deliverable from this pass.
The four corpus documents, source code, tests, profiles, Bug Bible, production bug
log, and `workflows/otr_canonical.json` were not edited.

VERDICT: ready for the implementation prompt? **no**.

## Entry gate

The requested final pass stops at the entry gate. The master status is
`r2-HARDENED`, not `r4-HARDENED`
(`docs/2026-08-09-SPEC-lab-findings-into-otr.md:4`). The same file nevertheless
refers to conclusions that "r3 proved" and to an "r3 MUST-FIX"
(`docs/2026-08-09-SPEC-lab-findings-into-otr.md:714-715,755`). Those references
do not establish that r3 was completed, and neither establishes r4 convergence.

Everything below is a separately authorized **correction-planning audit**. It
does not waive the stop gate, silently relabel the master, or authorize changes
to the corpus or code.

## Audited snapshot

OTR HEAD: `d91ff4508eed00c79967197565074ceef5d2e217`

Lab HEAD: `4d87cfa3278c39cbdde6f3cb8b16f241aeb58c02`

The four corpus files were untracked in the OTR working tree at audit time, so
the hashes below, not OTR HEAD, identify the reviewed bytes:

| File | SHA-256 |
|---|---|
| `docs/2026-08-09-SPEC-lab-findings-into-otr.md` | `7E395C7A8AACF9EFA2147768A455ECFF730D0D6743DF0E43919080E1D4B252E3` |
| `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md` | `89E5ED7B8ABDAA1AEA00FB5DF0924C50C7C0E0AAB1848A46CB9F2841A017540B` |
| `docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md` | `AB27CCCBBA2D65AE906E7D663B2C07E9DB7111D026601422223B4078DC84EFC7` |
| `docs/VIDEO_LANE_PREFLIGHT.md` | `CAB61CD7CB2AED5BFD3395DD2E7AF1314E4D0164E501DA7EBCF492C22E379F47` |

Lab narrative hashes used by this audit:

| File | SHA-256 |
|---|---|
| `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\ENVELOPE_LADDERS.md` | `A312D938C3868B75378F0F758CEB792DB080ADAB1D764FBE97C3BE92C0922FDC` |
| `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\HUMO_DIET.md` | `15418E58F955C28E3483EE7E96FB4CCB407A386F50784DEF0E1275A119B08004` |
| `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\HUMO_BAKEOFF.md` | `D48F715C960721DF7506FF6FB8B740A645167A719B0A173040E6C368BB9B67E6` |
| `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\H3_MUSIC_FOLLOWUP.md` | `79ACEF32EAB107D6134693C98687C5DAE40D4A8D26842B6D1FA95DBDE1B8F1B1` |
| `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\WAN_RETENTION_FINDINGS.md` | `3C59C3046F4D1FFA50E800DCC0AC9F80677030FF93A7E5308366DCFE7A7E6607` |

## Corrections to the attached QA

1. **Retract the claimed transplant-vs-follow-up dropdown contradiction.** The
   attached report calls the standalone runner and the future dropdown-overrule
   build contradictory
   (`C:\Users\jeffr\.codex\attachments\f87f6e90-860b-4105-afc8-ac89bab13dbf\pasted-text.txt:21-24`).
   The master explicitly sequences the runner in this build and the dropdown
   overrule in the next spec
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:709-754`), while the design
   brief calls itself the follow-up/NEXT spec
   (`docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:19-21`). These are sequential
   scopes, not conflicting delivery claims.

2. **Retract the claimed H3 continuity contradiction.** The attached report
   treats `CONTINUITY_NONE` and `SINGLE` as different behavior
   (`C:\Users\jeffr\.codex\attachments\f87f6e90-860b-4105-afc8-ac89bab13dbf\pasted-text.txt:36-39`).
   Both mean a standalone, unchained render
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:241`;
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:213`). The identifiers are
   vocabulary drift that should be normalized, not a behavioral contradiction.

3. **Retract the unchanged Veo docstring as a stale-corpus finding.** The corpus
   describes that current source text as work to be changed
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:243-246`;
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:55-57`). The unchanged source at
   `nodes/_otr_video_engines/frame_contract.py:108` is expected before
   implementation.

4. **Reclassify three alleged missing owners.** HuMo short-render refusal is
   owned by the plan's HuMo sibling step
   (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:304-307`); the `ltx_8gb` Sage and
   node gates are in its LTX work and build step
   (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:106-117,319-323`); dead-code
   deletion is in plan step 9
   (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:336-339`). Their absence from
   the master's canonical sequence is still a sequence/ownership ambiguity, but
   they are not absent from the entire corpus.

5. **Retract `n/a` enforcement as a missing transplant owner.** It belongs to
   the explicitly later mime-overrule spec
   (`docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:19-36`), not the current
   transplant.

6. **Replace the `WIDE_DIMS` location-only finding with the actual behavior
   failure.** `eng_humo.py:625-635` does use the aspect helper, but it also
   accepts `OTR_HUMO_WIDTH`/`OTR_HUMO_HEIGHT` overrides. The literal default is
   in `nodes/_otr_shared/aspect.py:31`. Therefore the master's statement that
   the graph simply "renders WIDE_DIMS = (832,480)" is not a fixed runtime
   guarantee (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:493-499`).

7. **Do not treat the old f107 receipt as proof of the corrected minimum.** It
   is a measured below-range failure
   (`C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\ENVELOPE_LADDERS.md:69-75`).
   The installed node supplies the actual trained-range declaration at
   `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_minimax_h3.py:90,116`.

8. **The attached QA did not obey its own stop rule.** It triggers STOP at line
   6 and continues with findings at line 14
   (`C:\Users\jeffr\.codex\attachments\f87f6e90-860b-4105-afc8-ac89bab13dbf\pasted-text.txt:6-14`).
   This report separates the failed final gate from the later, explicitly
   authorized correction-planning audit.

## Contradictions

1. **Hardening status is internally impossible to audit as written.** The
   master says `Status: r2-HARDENED`
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:4`) but later says "r3
   proved" and "r3 MUST-FIX 1"
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:714-715,755`). Neither quote
   supplies the missing r3 review artifacts or r4 convergence.

2. **The two canonical build orders disagree.** The master orders S9, S1, S8,
   S2, S7, H3+S6, LTX, then P1
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:765-784`). The lane plan puts
   hero/HuMo second and third, LTX sixth, H3 seventh, naming eighth, and cleanup
   ninth (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:285-341`). Both cannot be
   the exact commit sequence for one build.

3. **H3 registration and naming have incompatible ownership.** The master
   combines H3 registration with S6 naming because registration changes
   `INPUT_TYPES` (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:780-782`). The
   lane plan builds H3 first and moves all names to a later commit
   (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:325-334`).

4. **Retired HuMo public names remain live-looking in the plan.** The master
   rules `humo17_high_audio_in_portrait`, `humo14_high_audio_in_portrait`, and
   `humo14_high_audio_in_wide`
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:374-381`). The plan still
   calls build steps `humo17_high_face`, `humo14_high_face`, and
   `humo14_high_face_wide`
   (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:167-168,298,304`), although its
   own window table uses the corrected names at lines 220-221.

5. **The HuMo 14B parity ruling is closed and pending at once.** The master says
   "CLOSED 2026-08-10" and "Cast is final"
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:659-664`), then says parity
   "is still pending" at lines 670-671. The plan repeats pending at
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:189-194`, and master commit 9
   still waits for measurement/operator confirmation at line 784. The receipt
   says **PARITY**, ruled 2026-08-10
   (`C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\ENVELOPE_LADDERS.md:20-23`).

6. **`still_plan` is mandatory and deleted in the same build.** S8c and G7.4
   require it declared and audit-clean
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:616-619`;
   `docs/VIDEO_LANE_PREFLIGHT.md:90-92`). The plan deletes it because production
   reads nothing (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:336-339`). A
   single authority must be selected before code begins.

7. **The mime design status and its body disagree.** The header says R1 is
   complete, candidate B is cut, the role slot is selected, and episode mime is
   capped at 200 canvas frames
   (`docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:3-18`). Lines 71-119 still
   present candidate B and ask R1 to decide binding, failure, role, and runner
   questions. That section must be labeled historical input or replaced by the
   decided design.

8. **The mime insertion seam is both corrected and reasserted.** The design
   header says `scene_sequencer.py:1296-1320` is not the line-level insertion
   boundary (`docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:5-8`). The master
   calls it the insertion point
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:739-742`), and the design body
   repeats that at lines 66-69. Current source shows opening theme, one already
   assembled `main_waveform`, and closing theme at
   `nodes/scene_sequencer.py:1296-1320`; line-level assembly occurs earlier at
   `nodes/scene_sequencer.py:958-1051,1109-1128`.

9. **The preflight origin claim exceeds the admission evidence.** The preflight
   says every gate exists because "a real lane failed it"
   (`docs/VIDEO_LANE_PREFLIGHT.md:13-14`). The master says the sixteen issues
   were found by audit and "would surface" as failed runs
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:470-473`). Under the repo's
   admission rule, a static audit is not a live production failure. The
   preflight must distinguish audit-derived gates from live-artifact failures.

10. **The H3 mime window conflates model legality, machine qualification, and
    episode policy.** The plan advertises 129..377 canvas frames / 15.08 s
    (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:213`). The master says f277
    reached 14.72 GiB and lengths above f192 require headroom work
    (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:702-707`), while the follow-up
    caps episode mime at 200 canvas frames / 8 s
    (`docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:10-12`). The full lattice may
    remain model-legal, but it is not all machine-qualified or episode-legal.

11. **The public measurement table's headings do not describe its rows.** It is
    headed "Measured warm" / "Longest 1 render"
    (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:364-376`) but mixes cold-only
    H3 data, OTR-side non-gated measurements, a whole-child chained diagnostic,
    and theoretical maximum durations. These are different evidence surfaces.

12. **The stated 7.5-to-12.5 GiB empty gap is no longer true.** The naming
    rationale claims an empty measured gap
    (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:351-359`). OTR-wrapper WAN and
    FastWan warm absolute peaks occupy roughly 8.05-9.38 GiB
    (`C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\ENVELOPE_LADDERS.md:49-58`).

13. **The master's LTX HQ lane disappears from the per-lane plan.** The master
    requires 1024x576x193 and assigns it to its LTX commit
    (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:146-156,783`). The plan's LTX
    work list and build group omit it
    (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:72-117,319-323`).

## Stale facts

1. **Retired H3 minimum remains in acceptance.** The master still says "H3's
   minimum (107)" (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:805-808`). The
   installed node declares a trained range of approximately 124..362 model
   frames on the 17k+5 grid
   (`C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_minimax_h3.py:90,116`),
   which the corpus converts to canvas menu 129..377
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:196-204`;
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:37-39`).

2. **Retired `*_high_face*` names are still written as current build IDs.** See
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:167-168,298,304`. Current live IDs
   are the three `*_high_audio_in_*` names at
   `docs/2026-08-09-SPEC-lab-findings-into-otr.md:374-381`; old names may exist
   only as legacy aliases.

3. **`PENDING_HUMAN` remains after the dated parity ruling.** See
   `docs/2026-08-09-SPEC-lab-findings-into-otr.md:670-671` and
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:193-194`; current truth is the
   2026-08-10 PARITY ruling at
   `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\ENVELOPE_LADDERS.md:20-23`.

4. **Resolved R1 alternatives remain written as live choices.** See
   `docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:71-119`; the resolved status is
   at lines 3-18.

5. **H3 is labeled warm from cold-only receipts.** The public table reports
   6.5-6.7 GiB under "Measured warm"
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:364-367`). The seed 42/43
   Ref2VA receipts are cold gates at 864x480x124
   (`C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\HUMO_BAKEOFF.md:63-72`).

Negative sweep:

- The corpus does not say the admission guard has zero call sites. It correctly
  describes a wired but inert guard and the empty qualification set at
  `docs/2026-08-09-SPEC-lab-findings-into-otr.md:71-84`; current code is
  `nodes/_otr_video_engines/motion_common.py:332-367` and
  `nodes/_otr_video_engines/render_driver.py:3191-3268`.
- The corpus does not apply "agents never push" to OTR. It states commit and
  push per green chunk at
  `docs/2026-08-09-SPEC-lab-findings-into-otr.md:10-14`.
- The 14.5 GiB hardware gate remains a gate, not a claim that the selected hero
  consumes 14 GiB. The stale content is the parity-pending language, not the
  existence of the 14.5 GiB gate
  (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:20,659-671`).

## Operator rulings

| Required ruling | Location | QA result |
|---|---|---|
| Equal-partner dropdown; one master JSON; lanes independent; sage-free H3 episodes; lab-first new loads | `docs/2026-08-09-SPEC-lab-findings-into-otr.md:3,23-27` | Present under the 2026-08-09 file date. |
| Low/high public naming with `audio_in` and portrait stated in HuMo IDs | `docs/2026-08-09-SPEC-lab-findings-into-otr.md:351-382` | Present and dated; undercut by stale `*_face*` IDs in the lane plan. |
| Hero = `humo_14B_169` under `humo_diet` | `docs/2026-08-09-SPEC-lab-findings-into-otr.md:659-674` | Present and dated 2026-08-10; undercut by pending text. |
| Workhorse = H3 seed 43 fixed via profile | `docs/2026-08-09-SPEC-lab-findings-into-otr.md:653-676` | Present with the 2026-08-09 human verdict. |
| Mime overrules TTS/music; runner-first sequence | `docs/2026-08-09-SPEC-lab-findings-into-otr.md:709-755`; `docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:42-53` | Present and dated; the two specs are sequential, not contradictory. Insertion-seam text is stale. |
| Cheap `n/a` entries; label carries guidance; tooltips secondary; no frontend JS | `docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:23-40` | Present and dated 2026-08-10. Correctly belongs to the follow-up build. |
| Describe the scene; no restraint vocabulary | `docs/2026-08-09-SPEC-lab-findings-into-otr.md:267-295`; `docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:48-50` | Present under the dated documents. |

## Citation failures

Only wrong-file/wrong-behavior failures are listed here. Drift appears in the
next section.

1. **Declared-canvas precedence cites the resolver, not the precedence seam.**
   The master cites `render_driver.py:227-267` for "applied last"
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:42-43`). That range
   defines/validates the declaration. Application precedence is at
   `nodes/_otr_video_engines/render_driver.py:2554-2570`.

2. **HuMo wide size is not fixed as claimed.** The master says the graph
   "renders WIDE_DIMS = (832,480)" and cites `eng_humo.py:625-635`
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:493-496`). Current code uses
   `humo_dims_for_aspect` and accepts width/height environment overrides at
   `nodes/_otr_video_engines/eng_humo.py:625-635`; the default literal is at
   `nodes/_otr_shared/aspect.py:31`.

3. **The cited mime insertion seam has the wrong behavior.** The master and
   design body cite `scene_sequencer.py:1296-1320`
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:739-742`;
   `docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:66-69`). That range joins
   opening theme, already assembled main waveform, and closing theme. The
   line-level assembly is at `nodes/scene_sequencer.py:958-1051,1109-1128`.

4. **The still-spine description overstates an engine-ID list.** The plan says
   `_still_spine_requires_scene` is engine-ID-hardcoded and every JUMP lane must
   be added (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:234-237`). Current
   `nodes/_otr_video_engines/render_driver.py:767-780` also handles families,
   required `init_image`, and provider-side `accepts_still`.

5. **The plan says H3 mouth beats refuse multi-clip, but the cited policy does
   not.** The claim is at
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:212,230-233`. Current
   `nodes/_otr_video_engines/mouth_policy.py:158-174,197-200` warns and records
   `long_takes`; it does not refuse. The master also says longer hero beats
   JUMP-chain at `docs/2026-08-09-SPEC-lab-findings-into-otr.md:668-670`.

## Citation spot audit and drift

Twenty-five citations were sampled. `PASS` means the cited behavior exists;
`DRIFT` means behavior is right but the current line moved; `FAIL` is carried
into the section above.

| Corpus citation | Current source | Result |
|---|---|---|
| master `:33-34` | `nodes/_otr_video_engines/eng_wan_i2v.py:235-242` | PASS |
| master `:42-48` | `render_driver.py:227-303`; precedence actually `:2554-2570` | FAIL |
| master `:50-53` | `eng_wan_i2v.py:245-250`; `eng_wan_ti2v.py:331-339` | PASS |
| master `:57-60` | `scripts/_otr_w45_boot.ps1:42-48` | PASS |
| master `:68-69` | `eng_wan_i2v.py:214` | PASS |
| master `:73-81` | `motion_common.py:332-367`; `render_driver.py:3191-3268` | PASS |
| master `:88-92` | `tests/test_vram_admission_boundary.py:69-72`; `eng_fastwan_8gb.py:295-296` | PASS |
| master `:94-102` | `motion_common.py:319,425-482` | PASS |
| master `:107-114` | `render_driver.py:3249,3697`; `nodes/otr_video_render_batch.py:128-172` | PASS |
| master `:148-156` | `eng_ltx_av.py:108-115,239-240,294-331` | PASS |
| master `:160-168` | `registry.py:137-214`; `public_engines.py:68-72`; `otr_video_director.py:158-195,233-250` | PASS |
| master `:175-182` | `eng_google_veo_video.py:60-68,516-521`; `cloud_media_canonical.py:387-392`; `wrapper_bridge.py:616-621` | PASS |
| master `:243-246` | `frame_contract.py:108`; `tests/test_engine_contract_roster.py:204-228` | PASS |
| master `:250-265` | `render_driver.py:1546-1550`; `otr_shot_lock.py:1025-1027`; `mouth_policy.py:104-117`; `content_oracle.py:33-38`; `schemas.py:177-181` | PASS |
| master `:299-306` | `render_driver.py:2998-3024`; `_otr_shared/capability_profiles.py:118-123` | PASS |
| master `:397-401` | `tests/test_public_engines.py:40` | DRIFT: cited `:42` |
| master `:477-486` | `config/profiles/otr_8gb_wan.json:56,83` | DRIFT: env cited `:82` |
| master `:493-499` | `eng_humo.py:625-635`; `_otr_shared/aspect.py:31` | FAIL |
| plan `:75-80` | `eng_ltx_av.py:177`; `tests/test_ltx_av_env_import_safety.py:32-41` | PASS |
| plan `:81-85` | `eng_ltx_av.py:819`; `render_driver.py:2542`; test `:62-63` | DRIFT: driver cited `:2541` |
| plan `:125-132` | `eng_wan_ti2v.py:825-833`; profile `:56,83` | DRIFT: env cited `:82` |
| plan `:145-150` | `eng_wan_i2v.py:245-250`; `eng_wan_ti2v.py:331-339` | PASS |
| design `:59-64` | `scene_sequencer.py:1401-1422`; `otr_master_audio_mux.py:264-294` | PASS |
| design `:66-69` | `scene_sequencer.py:1296-1320`; actual line assembly `:958-1051,1109-1128` | FAIL |
| preflight `:9-10` | `tests/test_lane_preflight_matrix.py` | MISSING PLANNED ARTIFACT |

## Numbers vs receipts

### Evidence-baseline failure

The master says lab commit `4d87cfa` contains `PROMOTION_BRIEF`, `HUMO_BAKEOFF`,
`HUMO_DIET`, `WAN_RETENTION_FINDINGS`, and receipts
(`docs/2026-08-09-SPEC-lab-findings-into-otr.md:16-18`). At that Git object:

- `docs/PROMOTION_BRIEF.md`, `docs/HUMO_BAKEOFF.md`, and `docs/HUMO_DIET.md`
  exist, although their working copies are now modified.
- `docs/ENVELOPE_LADDERS.md`, `docs/H3_MUSIC_FOLLOWUP.md`,
  `docs/WAN_RETENTION_FINDINGS.md`,
  `results/humo_diet/humo_14b_diet_envelope.json`,
  `results/otr_side/wan_cost_ladder/fits/attempt-004.json`, and
  `results/otr_side/wan_retention/comparison.json` are absent and currently
  untracked in the lab repo.

This makes the stated baseline unreproducible and invalidates the proposed
manifest linkage at
`docs/2026-08-09-SPEC-lab-findings-into-otr.md:628-634`. Every retained receipt
must pass `git cat-file -e <evidence-commit>:<receipt-path>`.

### Master envelope table

Source table: `docs/2026-08-09-SPEC-lab-findings-into-otr.md:120-134`.

| Row | QA result | Receipt/current truth |
|---|---|---|
| LTX AV GGUF Q3, 832x480x97, 7.2-7.5 GiB | PARTIAL | Exact cold/warm values are 7.47/7.41 GiB in `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\results\ltx_audio_gguf_run4.json:4,6,8,12,14` and `...\ltx_audio_gguf_run5.json:4,6,8,12,14`; no receipt supports the 7.2 lower endpoint. Workload identity is `...\docs\VIDEO_RECIPE_ATTEMPTS.md:10,22-25`. |
| LTX AV HQ, 1024x576x193, 7.36 GiB / 585.3 s | SUPPORTED | `...\results\ltx_audio_hq_h3_1024x576_193f_run2.json:32,36-38,271,325,338`. |
| WAN TI2V 5B Q5, 832x480, 12.5-13.2 GiB warm | UNSUPPORTED AS WRITTEN | Exact 832x480x193 Q5 warm receipt is 12.1 GiB: `...\results\wan_ti2v_5b_cmp_832x480_f193_run4.json:30,34-36,207,261,274`; recipe at `...\recipes\wan_ti2v_5b_cmp_832x480_f193.json:155,200,205,210`. OTR-wrapper f25-f177 candidate peaks are 8,246-9,606 MiB at `...\docs\ENVELOPE_LADDERS.md:47-65`. The row omits rung and measurement surface. |
| WAN I2V, 13.93 warm / 14.05 cold | PARTIAL | Supported only at 832x480x33 by `...\results\wan_i2v_14b_exoneration_832x480_f33_run1.json:30,34-36,198,252,265` and run2 `:30,34-36,207,261,274`. The table omits f33, violating the key requirement at `docs/VIDEO_LANE_PREFLIGHT.md:56-57`. |
| HuMo 1.7B default, 15.12-15.23 GiB | SUPPORTED WITH QUALIFIER | OTR-side portrait f129 only: `...\results\otr_side\humo_1_7b_bakeoff_take1.json:21,116,219` and take2 `:21,110,213,219`; workload at `...\docs\HUMO_BAKEOFF.md:30-37,63-64`. |
| HuMo 1.7B diet, 12.84 GiB warm | SUPPORTED | Portrait 480x832x129 in `...\results\humo_1p7b_diet_run2.json:26,28-30,224,275,283`; ruling at `...\docs\HUMO_DIET.md:123-127`. |
| HuMo 14B default, 14.98 GiB | SUPPORTED WITH QUALIFIER | OTR-side portrait 480x832x97: `...\results\otr_side\humo_14b_fp8_bakeoff_take1.json:21,110,213`; workload at `...\docs\HUMO_BAKEOFF.md:30-37,65`. |
| HuMo 14B diet landscape, 13.06 warm / 13.17 cold | NUMERICALLY SUPPORTED; BASELINE FAIL | `...\results\humo_14b_diet_landscape_832x480_f97_run1.json:26,28-30,226,277,285` and run2 `:26,28-30,239,290,298`; parity at `...\docs\ENVELOPE_LADDERS.md:20-23`. Files are absent from `4d87cfa`. |
| HuMo 14B diet portrait, 13.22 warm / 13.14 cold | NUMERICALLY SUPPORTED; BASELINE FAIL | `...\results\humo_14b_diet_portrait_480x832_f97_run1.json:26,28-30,226,277,285` and run2 `:26,28-30,239,290,298`. Files are absent from `4d87cfa`. |
| H3 Ref2VA, 6.51-6.71 GiB | PARTIAL / WRONG CLASSIFICATION SURFACE | Cold 864x480x124 only in `...\results\h3_r2v_refaudio_tts_lipsync_exact_seed42_run1.json:26,28-30,142,185,193` and seed43 at the same fields. No warm pass and no 832x480 qualification exist. Ref2VA cannot classify H3 I2V or score/mime. |
| WAN chained 12.43 peak / +5.11 retained | SUPPORTED DIAGNOSTIC ONLY | Whole-child chained diagnostic: `...\results\otr_side\wan_retention\phase1_wan_ti2v_long_first.json:111,127,287`; summary `...\docs\WAN_RETENTION_FINDINGS.md:62-64`. |
| FastWan chained 12.57 / +5.33 retained | SUPPORTED DIAGNOSTIC ONLY | `...\results\otr_side\wan_retention\phase3_fastwan_8gb_long_first.json:110,126,286`; summary `...\docs\WAN_RETENTION_FINDINGS.md:65`. |
| LTX chained 14.59 / +3.06 retained | SUPPORTED FAILED DIAGNOSTIC | `...\results\otr_side\wan_retention\phase3_ltx_video_long_first.json:163,179,244`; `...\docs\WAN_RETENTION_FINDINGS.md:66,69-77`. It used 832x448, not 832x480. |
| 29.5x speed, 13.8 s vs 407.5 s | SUPPORTED | `...\results\comparisons\general_video_speed_pair.json:28,35,51,58,63`. |

### Plan clip-window table

Source table: `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:209-223`.

| Lane | QA result | Evidence boundary |
|---|---|---|
| `h3_low_video` | MODEL MATH CORRECT; GATE UNSUPPORTED | Installed 124..362 step 17 at `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_minimax_h3.py:90`; conversion at plan `:37-39`. No valid canonical f124+ I2V envelope exists; canonical 832x480x107 failed at 15.390 GiB in `...\results\h3_i2v_canonical_832x480_f107_run1.json:55,60-62,2231,10061,10082`. |
| `h3_low_audio_in` | MODEL MATH CORRECT; WINDOW UNQUALIFIED | Only f124 cold at 864x480 is measured. That does not qualify 832x480, warm cache, or 15.08 s. |
| `h3_low_mime` runner | MODEL-LEGAL WINDOW OVERSTATED AS QUALIFIED | f192 score is 11.063 GiB cold; f277 is 14.722 GiB and fails: `...\results\h3_music_followup_score_seed42_f192_run1.json:2,49,2225,10105,10121` and f277 equivalent; summary `...\docs\H3_MUSIC_FOLLOWUP.md:94-106`. Episode policy caps canvas f200 / 8 s at design `:10-12`. |
| `ltx23_low_audio_in` | SOURCE WINDOW CORRECT; RECEIPT COVERAGE PARTIAL | 9..497 q8 at `nodes/_otr_video_engines/eng_ltx_av.py:1350-1357`; measured receipt stops at f97 (`...\docs\VIDEO_RECIPE_ATTEMPTS.md:10,22-25`). 19.88 s is model-legal, not envelope-qualified. |
| `ltx23_high_video` | SOURCE WINDOW CORRECT; NOT QUALIFIED | Fixed f169 at `nodes/_otr_video_engines/eng_ltx_video.py:483-490`. Existing evidence is a failed chained diagnostic at 832x448. |
| `ltx098_low_video` | SOURCE WINDOW CORRECT; UNMEASURED | 9..161 q8 at `nodes/_otr_video_engines/eng_ltx_8gb.py:538-545`; master admits UNMEASURED at `docs/2026-08-09-SPEC-lab-findings-into-otr.md:369`. |
| `wan22_high_video` | SOURCE WINDOW CORRECT; PROFILE/RECEIPT MISMATCH | 17..177 q4 at `nodes/_otr_video_engines/eng_wan_ti2v.py:105-107,288-300`; current profile pins 17 at `config/profiles/otr_8gb_wan.json:56`. f177 candidate peak is 9,606 MiB, not 12.5-13.2 GiB (`...\docs\ENVELOPE_LADDERS.md:49-65`). |
| `wan22_high_fast` | SOURCE WINDOW INHERITED; PIN UNMEASURED | Profile pin81 is `config/profiles/otr_8gb_fastwan.json:62`; no receipt measures f81. Candidate f177 is 8,641 MiB at `...\results\otr_side\wan_cost_ladder\fits\attempt-004.json:1014,1038-1041`. |
| `wan21_high_i2v` | SOURCE WINDOW CORRECT; MAX UNQUALIFIED | 33..177 q4 at `nodes/_otr_video_engines/eng_wan_i2v.py:96-97,220-233`; only f33 has warm evidence. |
| `humo17_high_audio_in_portrait` | SOURCE WINDOW CORRECT; MAX UNQUALIFIED | 33..177 q4 at `nodes/_otr_video_engines/eng_humo.py:67-68,1029-1035`; diet evidence is f129 only. |
| HuMo14 portrait/wide | SOURCE WINDOW AND MAX RUNG SUPPORTED | 33..97 q4 at `nodes/_otr_video_engines/eng_humo.py:106,268-274,1151-1157`; both f97 orientation receipts are cited above. |
| Viz/still rows | CODE CONTRACT, NOT ENVELOPE EVIDENCE | `unbounded` / `any` has no VRAM or wall-time receipt and must not be presented as machine qualification (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:222-223`). |
| Approximate 750-frame segment counts | ILLUSTRATIVE ONLY | The plan itself requires real `partition_beat` literals at `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:203-207`. |

Required evidence key for every numeric claim: engine/adapter, recipe and quant,
canvas, measured model-frame rung, delivered/canvas frame count, boot lane,
cache state, measurement surface (`absolute`, `net`, `adapter`, `whole-child`,
or retained), wall-time boundary, receipt path, receipt SHA-256, and a Git commit
that contains the receipt.

## Missing or ambiguous owners

1. **Preflight infrastructure is consumed before it exists.** The plan requires
   every lane to read `docs/LANE_BUILD_LESSONS.md` and run
   `tests/test_lane_preflight_matrix.py` before coding
   (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:343-359`). Both paths are absent.
   S8c says the suite ships early with the evidence manifest
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:582-626`), but master commit 1
   names only S9 (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:765-769`).

2. **The all-engine preflight suite cannot be green on commit 1 as specified.**
   S8c requires assertions over all engines while S8b already enumerates sixteen
   red defects (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:470-626`). A
   progressive expected-red mechanism or separate evaluation/enforcement mode is
   required for the repo suite to remain green at each chunk. A silent skip is
   not acceptable under `docs/VIDEO_LANE_PREFLIGHT.md:3-10`.

3. **The evidence manifest has no update owner.** It is called immutable in
   commit 1 (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:628-634`), but commit
   4 later creates qualification receipts that G4.1 requires reachable from the
   manifest (`docs/VIDEO_LANE_PREFLIGHT.md:50-57`). The manifest must be
   versioned/append-only, and every receipt-producing commit must own an update.

4. **The promised standalone H3 runner has no explicit commit.** It "ships THIS
   build" (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:709-730`) but appears in
   neither master commits 1-9 nor plan steps 0-10
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:765-784`;
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:285-341`). No matching runner
   exists under `scripts/`.

5. **LTX HQ has no lane-plan owner.** See contradiction 13.

6. **S8b item 12 has no owner.** Still-lane ffmpeg preflight and dark-floor
   refusal are at `docs/2026-08-09-SPEC-lab-findings-into-otr.md:558-563` and do
   not appear in the plan build order at
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:285-341`.

7. **S8b item 14 has no root-fix owner.** The fixed-169 `ltx_video` cost problem
   is at `docs/2026-08-09-SPEC-lab-findings-into-otr.md:568-572`; the plan
   explicitly defers re-deriving the floor at
   `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:97-101`.

8. **S8b item 16 has no owner.** The missing Hy3D graph gate and pycairo
   dependency are at
   `docs/2026-08-09-SPEC-lab-findings-into-otr.md:577-580`; neither is in the
   plan build order.

9. **S8b item 15 has mutually exclusive owners.** Preflight requires
   `still_plan`; cleanup deletes it. See contradiction 6.

10. **Master S8b ownership is mostly implicit, not exact.** Against master
    commits 1-9 (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:765-784`), item 1
    is exact; items 2, 6, and 8 are only inferable; items 3-5, 7, and 9-16 are
    not explicitly named. The lane plan supplies some of those homes, proving
    that the two documents are not one executable sequence.

11. **G7.3 is not owned by every affected commit.** Preflight requires
    `ENGINE_MATRIX.md` regeneration with every canvas, contract, or registration
    change (`docs/VIDEO_LANE_PREFLIGHT.md:88-89`). The master names regeneration
    only with its naming commit
    (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:638-647,780-782`), although
    earlier commits change canvases/contracts.

12. **G7.2 has no owner on profile-changing commits.** It requires variant and
    node-87 workflow strings regenerated in the same commit
    (`docs/VIDEO_LANE_PREFLIGHT.md:86-89`), but the master profile step does not
    name those artifacts (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:783-784`).

13. **G8.1 solo smoke coverage is incomplete.** G8.1 applies to every changed
    lane (`docs/VIDEO_LANE_PREFLIGHT.md:94-98`). Master acceptance names wan_i2v,
    LTX HQ, two H3 adapters, and HuMo 1.7B only
    (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:803-815`), not every lane
    changed by S8b, naming, profiles, and procedural fixes.

14. **The lane-first sequence leaves G6 red until cleanup.** Lane 1 must pass
    preflight before commit (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:343-357`),
    but its dead Sage-sidecar claim is deferred to step 9
    (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:336-339`).

## Correct forward plan - no corpus or code changes authorized yet

### Gate A - reviewer accepts or amends this QA

1. Review this report against the hash snapshot above.
2. Resolve only disputed QA findings in this file.
3. Do not edit the four corpus documents, source, tests, profiles, or workflow
   until this QA is accepted.

### Gate B - stabilize evidence before rewriting claims

1. Commit the selected lab narratives and receipts to a new, actual lab evidence
   commit. Do not continue calling `4d87cfa` the baseline unless every retained
   path passes `git cat-file -e 4d87cfa:<path>`.
2. Record a receipt index with the full evidence key specified above. Copy the
   required immutable evidence subset into OTR or ship sufficient receipt data
   with the manifest; the master already states that a digest of an unreachable
   file is insufficient
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:628-634`).
3. Split each duration claim into three columns:
   `model-legal window`, `machine-qualified window`, and `episode-policy cap`.
   Do not infer one from another.
4. Run lab-first measurements for `ltx098_low_video` and
   `ltx23_high_video` before assigning their final low/high public names
   (`docs/2026-08-09-SPEC-lab-findings-into-otr.md:384-389`).
5. Do not use Ref2VA receipts to classify H3 I2V or score/mime. Each adapter and
   measurement surface receives its own evidence row.

### Gate C - after QA approval, reconcile the documents only

1. Make the master the single normative sequence and make the lane plan a
   derived lane view. Put one commit ID beside every S8b item, every preflight
   gate, every profile/workflow mutation, every receipt update, and every smoke.
2. Remove stale 107/111 live minima; retain historical f107 receipts only when
   explicitly labeled below trained range. Keep model 124..362 and canvas
   129..377 as the model lattice, separate from qualified/policy caps.
3. Replace every live-looking `*_high_face*` occurrence with the ruled
   `*_high_audio_in_*` ID or explicitly label it a legacy alias.
4. Remove all parity-pending text and the commit dependency on a ruling that is
   already closed.
5. Correct the SceneSequencer seam and turn design lines 71-119 into a clearly
   labeled historical R1 input, or replace them with the decided architecture.
6. Correct the preflight origin sentence so static audits are not represented as
   production failures.
7. Resolve `still_plan` once. Default recommendation: preserve the later G7.4
   ruling and make `still_plan` a runtime-consumed authority; delete it only if a
   new dated operator ruling also removes/replaces G7.4 in the same document
   change.
8. Add an ownership table that proves no artifact is consumed before its
   producing commit and no gate has zero or multiple owners.
9. Record the operator's reaffirmed ruling from 2026-08-10: **one lane is open
   at a time; close its QA before touching the next lane.** Remove family-wide
   and global naming sweeps from the normative sequence. A lane's registration,
   public ID, alias, node-87 strings, profile/variant, `ENGINE_MATRIX.md`, and
   canonical-workflow delta land atomically with that lane.

### Gate D - proposed implementation sequence after the documents pass review

The corrected corpus must state the 2026-08-10 operator ruling literally: **one
lane at a time, QA it, then move to the next.** The earlier plan already records
the same build law at
`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:253-260`; its grouped build steps at
lines 298-339 fail to execute that law.

#### Lane lock and close protocol

Only one lane may be `OPEN`. A lane can take several commits when measurement
must separate instrumentation from a root fix, but no other lane starts between
those commits.

For every lane, in order:

1. **Open exactly one row.** Read the complete lessons ledger, run the preflight
   evaluator for that lane, and bind every red item to this lane's work packet.
2. **Implement only that lane.** A necessary shared-code change is allowed, but
   it does not mark any sibling lane green. Add sibling non-regression coverage
   in the same chunk.
3. **Wire atomically.** If the lane changes registration, selection, a profile,
   a widget, or a node-87 string, update `workflows/otr_canonical.json`, that
   lane's public ID/alias, profile/variant, and `ENGINE_MATRIX.md` in the same
   lane change. There is no later global naming sweep.
4. **Regress.** Run targeted tests, AST/dead-reference/import checks, the full
   Windows suite, Bug Bible regression, and the workflow JSON round-trip plus
   validator/link/widget audit when applicable.
5. **Smoke only the open lane.** Reset the server, run its declared boot/profile,
   and receipt exact canvas, frames, audio law, VRAM surface, trim behavior, and
   artifact path.
6. **Close with QA.** The lane's preflight row must be green, its expected-red
   entries removed, evidence manifest version updated, receipt hash recorded,
   solo-smoke receipt present, and QA verdict `PASS`. Commit and push the green
   chunk and verify HEAD==origin.
7. **Learn, then unlock.** Append only live lessons from that lane to
   `docs/LANE_BUILD_LESSONS.md`. The next lane remains locked until all six
   prior conditions are complete.

#### Foundation before lane 1

0. **Evidence and progressive preflight foundation.** Land a versioned evidence
   manifest, create `docs/LANE_BUILD_LESSONS.md`, and create
   `tests/test_lane_preflight_matrix.py`. Pending lanes use an explicit,
   defect-ID-bound expected-red ledger with strict unexpected-pass behavior; no
   skip and no false all-green claim. This foundation changes no lane from red
   to green and introduces no unused runtime mechanism.

#### Exact one-lane order

1. **`wan21_high_i2v` (`wan_i2v`).** Own S1/S8b-1, weight resolution,
   declared-canvas precedence, this lane's Sage/isolation truth, exact f33
   evidence qualifier, public surface, workflow/profile, and solo smoke. It is
   first because it cannot start today
   (`docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:145-163`).
2. **`humo14_high_audio_in_wide` (`humo_14B_169`).** Build the boot-contract
   mechanism with its first real consumer, never as unused infrastructure. Own
   S8b-4, this lane's S8b-6 manifest fields, S8b-8 boot-field resolution, the
   ruled `humo_diet` hero cast, exact public ID, workflow/profile, and f97 smoke.
3. **`humo17_high_audio_in_portrait` (`humo_1.7B`).** Own S8b-3, this lane's
   S8b-6 fields, its S8b-7 stale text, diet/default compatibility, and its own
   profile/workflow/receipt/smoke. Do not alter the portrait 14B lane here.
4. **`humo14_high_audio_in_portrait` (`humo`).** Close the remaining HuMo lane
   independently with its exact public ID, manifest fields, diet profile,
   workflow delta, and portrait f97 smoke.
5. **`wan22_high_video` (`wan_ti2v`).** Own S2 for this row, S8b-2, this lane's
   S8b-7 text, profile-pin correction, and S8b-5 Sage/isolation truth. Keep the
   lane open across two subchunks if needed: instrument retention first, collect
   its telemetry, then land the justified root response. Close only after the
   exact OTR-lifecycle cost row, manifest, matrix, workflow/profile, chained
   smoke, and retention QA are green.
6. **`wan22_high_fast` (`fastwan_8gb`).** Qualify its own injected cost row and
   pin, verify inherited retention behavior, and close its public surface,
   profile/workflow, manifest, matrix, and solo smoke without reopening WAN
   TI2V.
7. **`ltx23_low_audio_in` (`ltx_audio_in`).** Own S3, S8b-9, S8b-10, the
   1024x576x193 HQ profile, legal stage-A canvas, import/env refusal, exact
   evidence qualifiers, public surface, workflow, and solo smoke.
8. **`ltx098_low_video` (`ltx_8gb`).** Run its lab-first measurement before
   final naming. Own its part of S8b-11, S8b-13, profile-canvas reconciliation,
   Sage/node gates, public surface, workflow, manifest, matrix, and solo smoke.
9. **`ltx23_high_video` (`ltx_video`).** Run its lab-first single-render
   measurement. Own its part of S8b-11 and a root resolution of S8b-14 rather
   than deferred logging; then close its name, profile/workflow, evidence,
   matrix, and solo smoke.
10. **`mesh_stage`.** Own the Hy3D graph half of S8b-16 and this lane's dead
    profile-canvas channel. Gate all required node classes before load, then
    close its matrix row and solo smoke.
11. **`viz_green`.** Reconcile only this visualizer's profile/canvas contract,
    ffmpeg gates, matrix row, and solo smoke.
12. **`viz_camera`.** Repeat the learned visualizer checks for this lane only;
    close its own profile/canvas, matrix row, and solo smoke.
13. **`viz_mxc_cpu`.** Close this lane independently with its profile/canvas,
    dependencies, matrix row, and solo smoke.
14. **`viz_mxc_mandala`.** Own the pycairo half of S8b-16, then close this
    lane's profile/canvas, named dependency refusal, matrix row, and solo smoke.
15. **`still_motion`.** Resolve G7.4/S8b-15 for this lane and own its S8b-12
    ffmpeg and missing-still refusal. Close its profile/canvas, still authority,
    matrix row, and solo smoke before touching another still lane.
16. **`still_pan`.** Apply the now-proven still-lane rules, but change and smoke
    only this lane.
17. **`still_flat`.** Apply the same checklist independently and close only its
    row and smoke.
18. **`still_word`.** Preserve its existing missing-still refusal, add/verify
    the ffmpeg and single-authority contract, and close its own row and smoke.
19. **`h3_low_video` (`minimax_h3_video`).** Add the shared H3 implementation
    with this first registered adapter only; do not register audio-in yet. Own
    corrected 124..362 model / 129..377 canvas math, 24-to-25 delivery,
    continuity, Sage-free boot, silent-file self-probe, this lane's public ID,
    profile/workflow, matrix, evidence qualifiers, and solo smoke.
20. **`h3_low_audio_in` (`minimax_h3_audio_in`).** Add the second adapter in a
    separate lane change. Own its mouth policy, soft-reference/JUMP behavior,
    seed-43 workhorse profile, still authority, public ID, workflow, matrix,
    adapter-specific evidence, and solo smoke. Ref2VA evidence does not qualify
    the preceding I2V lane.
21. **Standalone `h3_low_mime` runner.** Treat the promised runner as its own
    lane-sized work packet even though it stays out of episode registration.
    Own the G5.2 keeps-audio exemption, clip/stem receipts, durable output path,
    model-legal vs machine-qualified range, and solo-runner QA. Do not claim
    f277+ qualified.
22. **All-row and episode gate.** After lane 21 closes, require every preflight
    row green, every expected-red entry removed, every solo-smoke receipt
    present, evidence manifest current, and workflow validation green. Only then
    run the final end-to-end episode with H3 and same-engine chaining.

The mime dropdown-overrule implementation remains a **separate follow-up spec**
after this transplant. Its `n/a` options, role-slot overrule, pre-audio runner,
and self-scored audio ownership do not move into steps 0-12 merely because the
design brief already exists
(`docs/2026-08-10-DESIGN-BRIEF-mime-overrule.md:19-40`).

### Gate E - r3/r4 hardening and re-entry to final QA

1. Run r3 as a wiring/dependency review against the canonical workflow and the
   exact owner table.
2. Correct every r3 must-fix and rerun affected citation and receipt checks.
3. Run r4 convergence. Only when r4 records no new must-fix may the master status
   become `r4-HARDENED`.
4. Re-run the original final-QA prompt against new hashes. A status-label-only
   edit does not pass this gate.

## Verify markers

- `verify: pin the installed MiniMax H3 source to a package/Git revision`; the
  local file proves the current tooltip at
  `C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\comfy_extras\nodes_minimax_h3.py:90,116`,
  but the corpus does not record that source revision.
- `verify: produce actual partition_beat literals for every advertised 750-frame
  example`; the plan marks current counts illustrative at
  `docs/2026-08-09-TRANSPLANT-PLAN-per-lane.md:203-207`.
- `verify: measure ltx098_low_video and ltx23_high_video before final low/high
  naming`; the master marks them unmeasured at
  `docs/2026-08-09-SPEC-lab-findings-into-otr.md:384-389`.
- `verify: qualify or refuse H3 score/mime requests above canvas f200 under the
  14.5 GiB gate`; f277 failed at
  `C:\Users\jeffr\Documents\ComfyUI\vram-recipe-lab\docs\H3_MUSIC_FOLLOWUP.md:105-106`.
