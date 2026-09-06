# Physical 4060 video/audio qualification and final clean-start test

Updated September6, 2026,05:22PDT. This is the user's expanded campaign,
not a claim that every menu item works. Local GPU: RTX4060 Laptop,8GB,Ada.
The [drill log](4060_DRILL_LOG.md) owns chronology and the
[portability report](4060_PORTABILITY_REPORT_2026-09-06.md) owns verdicts.

## Current checkpoint and source work

The sole active GUI Run is September6,04:50:36.149PDT. Five automatic visual
downloads verified36,818,738,352bytes and READY05:03:09.811. Writer episode
`pending_20260906_050309` started05:03:09.837; first357-token concept completed
05:18:20.239, followed by `original_select`05:18:20.259. Follow an explicitly
logged later rename only. Do not submit another Run or alter installed code.
Current development run is hand-patched, not clean-install qualification.

User now requests ONE ACT in the shipped default canonical. DEV canonical
writer act_count is changed from string3 to string1, with no other canonical
widget/model/link change. The existing story-only builder was run; it also
synchronized its pre-existing cnr_id and visual_style drift to canonical
(`sci_fi_radio` becomes `roll (any style)` in that derived diagnostic graph).
Generic newly added writer-node and legacy missing/invalid-input defaults
remain3; this request is the shipped-template default, not a runtime migration.

**Pending, not ready to ship:**92 generated graph JSONs still inherit3 and must
be regenerated with the normal generator after the active run is idle. Its
schema discovery imports the node package and is deliberately not run against
the active workload. No hand editing of generated variants or invented schema
fixtures. Do not commit/publish a partially synchronized default bundle.

Two selected stdlib canonical/story checks plus84 existing focused checks
passed (86total). The third new inheritance test is pending regeneration.
Independent finished-diff review found no new blocker. Full pytest and the
configured Bug Bible remain unavailable in this installation. After idle:
regenerate, run generator --check, all template tests, widget/link audits,
AST/JSON/diff checks, review the actual generated diff and push the scoped
green bundle. Preserve the unrelated historical untracked review-plan file.

## Candidate matrix: bounded serial coverage, not all combinations

One act per cell, normal app-managed downloads, no manual model placement,
cache seeding, launch-command hacks or additional packs. Hold non-target
choices fixed in a separately named JSON derived from canonical. Test the
selected engine in every compatible role present in that act; verify from
render receipts that it actually ran. An unused selection is NOT a pass.

| Cell | Selection to exercise | Current disposition |
| --- | --- | --- |
| 01 | LTX0.9.8 low16:9 x3; Kokoro both voices; MusicGen | RUNNING baseline; availability verified, full result pending |
| 02 | Bark, both compatible voice roles | NOT TESTED; in-process candidate |
| 03 | Stable Audio3 music | NOT TESTED; ordinary weight provisioning is missing in current integrated preflight |
| 04-11 | still_motion, still_pan, still_flat, still_word, viz_green, viz_camera, viz_mxc_cpu, viz_mxc_mandala | NOT TESTED; eight separate procedural/still controls, only compatible roles |
| 12 | h3_low_video | CONDITIONAL; native weights and normal approved boot route required |
| 13 | h3_low_audio_in | CONDITIONAL; same shared weights, audio-conditioned role required |

Maximum13 presently identified cells including baseline; no endless repeats or
Cartesian product. Procedural controls have no video weights to download, but
still-based choices retain the selected image model/assets and mandala may
need its declared Cairo dependency. Do not silently substitute engines.

Kokoro uses hexgrad/Kokoro-82M; the Python3.13 path additionally uses the
onnx-community/Kokoro-82M-v1.0-ONNX backend. These are backend assets, not
separate offered model choices. MusicGen adapter fixes facebook/musicgen-small;
medium/large are not menu cells. Bark uses suno/bark and baked-in voice presets.
Stable Audio3 uses stable_audio_3_small_music.safetensors plus
t5gemma_b_b_ul2.safetensors. File size is not measured VRAM. Missing ordinary
provisioning is a product finding to diagnose/fix in source, not permission
for a hand download into the installation.

H3 has historical isolated physical4060 clip receipts around7.15GiB cold and
6.79GiB warm, not a complete OTR episode. Its90-frame lab recipe differs from
the canonical adapter's normal frame floor. The five-file manifest totals
63,440,965,087bytes (59.084GiB); do not start it without measured disk capacity
and an app-supported path. Source requests a Sage-free named boot contract.
Do not change the current launch by hand. Its special encoder's NVFP4 filename
does not by itself establish Blackwell-only behavior; nor does its historical
Ada receipt generalize to other NVFP4 weights. Revalidate current artifacts and
upstream prerequisites before any later download.

## Complete source-menu disposition

This inventory is source-declared, not a fresh GUI capture establishing that
every guarded adapter registered. UI suffixes such as16:9/portrait/audio-reactive
are generated; select the exact visible label at test time.

| Offered video tokens not listed as candidates above | Disposition/reason |
| --- | --- |
| wan22_high_video, wan22_high_fast | NOT TESTED here; repository production footprint approximately12.5-13.2GiB, outside this no-tuning8GB campaign |
| ltx23_low_audio_in, ltx23_high_video | BLOCKED by GGUF/LTXVideo pack dependencies; do not install |
| ltx25_high_video, ltx25_high_foley_plus, ltx25_high_mime | BLOCKED by additional runtime/pack prerequisites; no completed physical4060 qualification located |
| humo14_high_audio_in_portrait, humo14_high_audio_in_wide, humo17_high_audio_in_portrait, humo17_high_audio_in_wide | NOT TESTED at shipped recipes; even1.7B portrait documents12.84GiB warm,14B approximately13GiB |
| animatediff15_video, animatediff15_v3_video, animatediff15_v3_haunted_video, animatediff15_v3_stillin_lab_video | BLOCKED by explicit no-extra-pack rule; prior successes do not waive it |
| mesh_stage | PREREQUISITE-BLOCKED/UNKNOWN: Hunyuan3D plus pinned portable Blender; not a weight-only lane |
| cloud_kling_avatar, cloud_seedance_2, cloud_wan_i2v, cloud_wan_i2v_audio, cloud_vidu_q2_pro_fast_720p, word_razzle, google_omni_video, google_veo_video | OUT OF SCOPE: remote services, not physicalGPU downloadable models; no paid cloud run authorized |
| + Add Custom Model | Not a bounded predefined model; no invented extra cell |

Retired wan22_high_i2v is an alias, not a registered current cell. No current
H3 mime adapter was found. The already-known haunted-label defect remains
logged; do not re-debug it as a new canonical failure.

Audio role menus:

- Characters: indextts2, chatterbox, dia, bark, kokoro, elevenlabs, google_tts.
- Announcer: kokoro, chatterbox, dia, elevenlabs, google_tts, bark.
- Music: stable_audio_3, musicgen, stable_audio_music, sonilo, google_lyria.

| Non-candidate audio choice | Disposition/reason |
| --- | --- |
| chatterbox, dia | PREREQUISITE-BLOCKED/UNKNOWN: isolated sidecar venv/worker and reference audio; hardware fit not established. Diagnose normal product setup; do not manually fabricate it |
| indextts2 | OFFERED-BUT-UNSHIPPED: hardcoded character menu includes it but its adapter is excluded from registry bundle. Product packaging/menu finding, not permission to sideload |
| stable_audio_music | BLOCKED under no-token rule: stabilityai/stable-audio-open-1.0 configuration declares HF authentication |
| elevenlabs, google_tts, sonilo, google_lyria | OUT OF SCOPE: remote voice/music services |

Blocked prerequisite rows stay visible and unresolved until a normal product
path is implemented and tested or their exclusion is documented. Do not call
the campaign exhaustive-success while unresolved local candidates remain.
Old mocked tests/zero VRAM measurements from runner errors are not inference
evidence; historical receipts above are repository evidence, not fresh trials.

Source owners: [video menu](../nodes/otr_video_director.py),
[video registry imports](../nodes/_otr_video_engines/__init__.py),
[public labels](../nodes/_otr_shared/public_engines.py),
[audio menus](../nodes/_otr_engine_profiles.py),
[audio adapters](../nodes/_otr_audio_engines/__init__.py),
[audio prerequisites](../config/audio_engine_profiles.yaml),
[H3 manifest](../scripts/otr_fetch_lane_weights.py),
[H3 adapter](../nodes/_otr_video_engines/eng_minimax_h3.py),
[machine receipts](MACHINE_MATRIX.md),
[three-engine plan](2026-09-01-three-engines-portability-PLAN.md).

## Per-cell evidence and failure rules

Record exact software/source version, JSON identity and visible choices,
timestamped clicks/screenshots, every transfer's exact observed byte size,
start/end/wait time, source/cache/native destination, errors verbatim, any
considered hand step, episode identity and terminal artifacts. If the app does
not expose a size or timestamp, mark UNKNOWN; never infer it from a menu label.
Private raw evidence stays separate from public Git summaries.

OOM ends that cell as FAIL at its exact settings: no tuning around it. Writer
401 ends that trial: no token acquisition. An extra-pack request is a stop
defect. Healthy slow generation is not a reason to cancel/requeue. Preserve
failed evidence before a separately labeled source-fix/retest cycle.

Full development cell PASS requires its RESULT SUCCESS, obs_publish OK and
matching final episode file on disk, plus proof of the chosen engine's use.
Writing, intermediate media, availability READY or component clips alone are
not enough. Keep development success distinct from the final no-hand-step gate.

## End-of-campaign reset, release and human test

Do not delete anything during the active run or pending model campaign.

1. Finish the candidate matrix with explicit results/exclusions and reconcile
   unresolved local prerequisites. Freeze/push reviewed source and preserve
   all logs, screenshots, successful/failed receipts and final deliverables.
2. Read-only inventory exact test-instance, shared models, HF/alternate caches,
   saved test workflows and reparse points. Verify ownership and canonical
   absolute paths. A symlink and its target are separate deletion decisions.
   Exclude source repo, evidence, unrelated installs/data and preserved results.
3. Perform only the scoped authorized clean-start reset. Prefer recoverable
   cleanup where feasible; never recursively follow links or delete a broad
   workspace root. Report exactly what was removed and its recoverability.
   UI deletion and software installation require action-time confirmation under
   the computer-use safety policy, even with standing authorization. If needed
   while the user is asleep, stop before that action and preserve the blocker.
4. Publish the tested new registry version at the end, not during development.
   pyproject is a publication trigger. Verify actual distributed contents,
   dependencies and version identity, not only the source tree or upload log.
5. Reinstall that explicit version through Manager's normal visible GUI,
   load shipped default canonical ONE ACT and Run once. Pending is not Banned;
   do not mistake registry No nodes found for local import failure. Verify
   `[OldTimeRadio] OK - All 25 nodes loaded successfully` in the console.
6. No source patches, cache seeding, file copying, hidden commands or manual
   repair in this fresh human trial. Record all friction. PASS requires
   RESULT SUCCESS + obs_publish OK + episode file + ZERO HAND STEPS.
   Anything less is FAIL WITH FINDINGS; do not erase earlier failures.

The existing five-minute heartbeat continues this campaign, quiet on unchanged
state, and pauses when done or an actual user-action blocker prevents progress.
