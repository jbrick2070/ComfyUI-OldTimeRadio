# Lane build lessons -- the ledger every lane reads before it writes code

Companion to `2026-08-09-TRANSPLANT-PLAN-per-lane.md` (the per-lane loop) and
`VIDEO_LANE_PREFLIGHT.md` (the gates). This file is the MECHANISM that makes
one-lane-at-a-time pay for itself: after every lane closes, what actually bit
gets written here as a CHECK SOMEONE CAN RUN, and the next lane starts by
running the whole ledger against itself before any code is written.

**A lesson is not recorded until it is phrased as a check.** "Watch out for
canvas drift" is not a lesson; "does `render_canvas` equal what `_build_graph`
emits?" is.

**Every check that can be automated gets a twin assertion in
`tests/test_lane_preflight_matrix.py` in the same change.** The ledger is the
prose; the suite is the enforcement. A lesson with no twin assertion must say
in its own entry why it cannot have one.

## How to use this file

1. Read every entry top to bottom before writing a line of the lane.
2. Note which entries this lane already fails -- that is part of the work list.
3. Run `tests/test_lane_preflight_matrix.py`; the lane's red rows are the rest
   of the work list.
4. When the lane closes, append what bit you. If nothing bit, write that too --
   a lane that sailed through is evidence the ledger is working.

---

## L1 -- Weight resolution

**Check:** does the lane resolve every declared weight through `folder_paths`
(or a documented env pin), or does it hardcode a path that happens to be wrong
on this box?

**Symptom:** `assert_usable` raises `EngineUnusable(MISSING_MODEL)` before the
first forward, so the lane is dead on arrival and the failure names a file the
operator never installed under that name.

**Root cause:** a weight-existence helper written as a bare
`os.path.exists(<hardcoded default>)`. `folder_paths` is ComfyUI's resolver and
knows every configured model root and category; a hardcoded default knows one.

**Origin:** `wan_i2v` shipped dead. `_ckpt_path()` defaulted to
`<comfy_root>/models/checkpoints/wan2.2-i2v.safetensors` and `_installed()` was
a bare `os.path.exists` that never consulted `folder_paths`
(`eng_wan_i2v.py:245-250`) -- while the installed weight on this box is
`C:\ComfyUI-Models\diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`:
different name, different category. The sibling `eng_wan_ti2v.py:331-339` had
the fallback all along.

**Runnable check:** preflight gate G1. Every local GPU lane declaring
`model_requirements` must reference `folder_paths` in its weight-resolution
path, and a missing weight must produce a NAMED `EngineUnusable` from
`assert_usable` -- never a swallowed import.

**Twin assertion:** `test_lane_preflight_matrix.py::test_g1_weights_resolve`.

---

## L2 -- Canvas truth

**Check:** does the lane DECLARE `render_canvas`, is it /32-legal on both axes,
and does it equal what the graph actually renders? And does every profile
canvas either match that declaration or is the profile channel documented dead
for that lane?

**Symptom:** silent. The request says one size, the render emits another, and
nothing compares them -- until admission, still sizing, or composite scaling
starts trusting `request.canvas`, at which point it becomes a real error with
no obvious author.

**Root cause:** `declared_render_canvas` is applied LAST and overrules ledger
and env, so a lane with no declaration falls through to the 1472x832 landscape
default no matter what its profile says.

**Origin:** `humo_14B_169` rewrites its request to 1472x832
(`render_driver.py:2501-2509`) while the graph renders 832x480 -- a 3.07x pixel
disagreement. Separately, nine lanes (mesh_stage, four viz, four still) carry
`render.canvas_w/h` in their profiles that nothing reads.

**Precision that cost a QA round:** "the graph renders 832x480" was not a fixed
runtime guarantee -- `_native_dims` (`eng_humo.py:625-635`) resolves through
`humo_dims_for_aspect` (literal at `_otr_shared/aspect.py:31`) but ALSO honours
`OTR_HUMO_WIDTH`/`OTR_HUMO_HEIGHT`. A canvas declaration must therefore either
agree with those overrides or the lane must state that they are unsupported
once a canvas is declared. Check the OVERRIDE PATH, not just the default.

**Runnable check:** preflight gate G2, all four sub-gates -- declared,
/32-legal, equals the graph, derived/intermediate canvases legal too.

**Twin assertion:** `test_lane_preflight_matrix.py::test_g2_canvas_truth`.

---

## L3 -- Contract versus runtime, at the CANVAS rate

**Check:** does the declared frame contract match what the adapter actually
emits, AT THE CANVAS RATE (25 fps)? Is the discrete menu declared in FRAMES,
derived from the installed node's real limits rather than a doc's rounded
seconds? Is continuity declared explicitly rather than defaulted?

**Symptom:** a 4% accumulating drift with no assert anywhere. 192 model frames
generated at 24 fps and LABELLED 25 fps play as 7.68 s against an 8.00 s audio
window -- 320 ms of mouth drift over one beat, silent.

**Root cause:** the local encoder only LABELS the rate (`-r` before `-i pipe:0`,
`wrapper_bridge.py:620`); it cannot resample. Only the cloud path gets a
duration-preserving ffmpeg resample for free (`cloud_media_canonical.py:387-388`).
A local 24 fps model must convert in numpy immediately before the encoder.

**Second half of the same lesson:** a contract that overstates costs GPU work
silently. `ltx_video` declares `min_frames == max_frames == 169`, so a 50-frame
beat renders 169 and trims 119 -- ~3.4x the work, untracked, and the 169 floor
was measured at 1472x832 rather than the 832x480 the lane now declares.

**Third half:** a discrete menu derived from prose drifts. The H3 grid was
drafted 107-345 from a problem statement's rounded "4-15 s"; the installed node
(`comfy_extras/nodes_minimax_h3.py:90,116`) declares `step 17` with a trained
range of ~124-362. Pin the ladder literal against the node's real min/max at
build, never against a doc.

**Runnable check:** preflight gate G3. `native_fps == target_fps == 25`;
discrete menus in frames with both boundaries pinned; continuity explicit;
multi-clip partition literals derived by running the real `partition_beat`, not
by hand.

**Twin assertion:** `test_lane_preflight_matrix.py::test_g3_contract_matches_runtime`.

---

## L4 -- Receipt completeness, and PROVED silence

**Check:** does the clip dict carry `vram_peak_mb`, `recipe`, `quant` and
`render_canvas`? And is silence PROVED on the emitted file rather than
declared?

**Symptom (receipts):** the driver falls back to an instantaneous VRAM read
(`render_driver.py:3298-3302`), so every envelope built on those numbers is
built on a sample taken at an arbitrary moment rather than the peak.

**Symptom (silence):** `has_audio: False` is a hand-written literal in every
adapter and nothing probes the per-beat FILE. For a joint-AV model that ships a
receipt which lies.

**Root cause:** a manifest field with no owner defaults to `None` and no test
notices, because the test that "covers audio" (`test_audio_byte_identical`)
only re-hashes a stored fixture -- it proves the fixture is self-consistent,
never that this render was silent.

**Origin:** `eng_humo.py:900-902, :983-997` return no `vram_peak_mb`/`recipe`/
`quant`/`render_canvas`; the WAN lanes already fixed this at
`eng_wan_ti2v.py:1171-1194`. And no adapter probed its own emitted clip until
the V-1 self-probe was made a gate.

**Corollary that is its own rule:** ripping an LLM pass or a stamp is allowed;
leaving a ledger field with no owner is not. Before removing anything that
writes a field: enumerate every field it wrote, give each field exactly one new
owner, then delete the call, then prove it on a LIVE leg.

**Runnable check:** preflight gates G4 (admission honesty -- a qualified row or
the words "admission NOT enforced" on disk in the manifest) and G5 (the
canonicalize path runs `validate_silent_clip_contract` on its OWN emitted
file).

**Twin assertion:** `test_lane_preflight_matrix.py::test_g4_admission_honesty`
and `::test_g5_audio_law_self_probe`.

---

## L5 -- A lane can vanish from the menu without a single line in the log

**Check:** does anything in the lane's import path -- module scope, class body,
the `@register` decorator -- run code that can raise? Module-scope env reads in
particular: do they go through the guarded numeric parser?

**Symptom:** the engine is simply absent from the dropdown. No traceback, no
warning. `audit_engine_roster()` is the only thing that can see it, because the
missing engine is missing from both the registry and any registry walk.

**Root cause:** every adapter import in `_otr_video_engines/__init__.py` is
wrapped in `try: ... except Exception: pass` so a packaging quirk can never
break the namespace import. The cost is that a BROKEN adapter fails silently.

**Origin:** `eng_ltx_av.py:177` is a bare module-scope `float()` on
`OTR_LTX_AV_RESERVE_VRAM_GB` -- the one env read the guarded `_env_num`
(`:59-90`) was not applied to. A malformed value raises at import, the guarded
import swallows it, and the lane vanishes (reproduced live: registry 27 -> 26).
`tests/test_ltx_av_env_import_safety.py:33-42` claims to cover every
module-scope env read and omits exactly this one.

**The wider shape:** a MODULE-SCOPE assert can be worse than a raise. Two public
ids mapping to one internal id trips the bijection assert in
`public_engines.py:68-72` at IMPORT time, and because the director and the
shared profile/driver modules import it unguarded, the blast radius is most of
OTR silently vanishing from the node menu. That is why an engine rename MOVES
the old public id into `_LEGACY_ENGINE_ALIASES` and never ADDS a second row.

**Runnable check:** preflight gate G6.3 plus the standing roster count.

**Twin assertion:** `test_lane_preflight_matrix.py::test_g6_guards_fire_early`
and `::test_registry_roster_is_intact`.

---

## L6 -- A configured knob that reaches nothing

**Check:** for every boot/profile field this lane depends on, trace it to the
argv or env that actually consumes it. Then enforce it by probing the RUNNING
server, never the profile text.

**Symptom:** a "configured" contract that clamps nothing, and a check that
passes because it read the same config file that was already wrong.

**Root cause:** `profile.launch.extra_args` is written only into a markdown
documentation string (`build_variants.py:180,211`); no launcher ever turns it
into argv, and `--disable-pinned-memory` appears in ZERO non-doc files
repo-wide. The live channel is `launch.env`, consumed at
`_otr_soak_server_launch.cmd:120`.

**Same class, second instance:** `launch.sage_attention` is read only by the
schema validator and a docs generator. No boot script passes
`--use-sage-attention`. A profile field that looks like a boot control and
controls nothing is how the next reader gets this wrong.

**Same class, third instance:** the `sidecar_optional -> sidecar_required` Sage
escalation. `resolve_isolation()` has no production caller -- only tests -- and
there is no video sidecar runner in `nodes/`. Either delete the claim or build
the escape; do not leave a docstring describing protection that is not there.

**Runnable check:** boot contracts ride `launch.env`; enforcement probes
`comfy.cli_args.args` on the running server; a dead channel is documented dead
in the lane's row rather than silently trusted.

**Twin assertion:** `test_lane_preflight_matrix.py::test_g6_guards_fire_early`
(declaration half). The running-server half is a smoke-time receipt check --
it cannot be asserted CPU-side, and that is stated here deliberately.

---

## L7 -- Evidence has a shape, and a number without it is not evidence

**Check:** does every numeric claim this lane makes carry engine/adapter,
recipe and quant, canvas, measured model-frame rung, delivered/canvas frame
count, boot lane, cache state, measurement surface (absolute / net / adapter /
whole-child / retained), wall-time boundary, receipt path, receipt SHA-256, and
a Git commit that CONTAINS the receipt?

**Symptom:** a table headed "Measured warm" that mixes cold-only data,
OTR-side non-gated measurements, a whole-child chained diagnostic, and
theoretical maxima. Every row looks equally authoritative and none of them can
be reproduced.

**Root cause:** a digest of a file that is not shipped proves nothing to a
reader without it. `git cat-file -e <evidence-commit>:<receipt-path>` is the
only check that distinguishes a baseline from a claim.

**Origin:** the corpus named lab commit `4d87cfa` as its baseline; at that
object, `ENVELOPE_LADDERS.md`, `H3_MUSIC_FOLLOWUP.md`,
`WAN_RETENTION_FINDINGS.md` and three results files are ABSENT and currently
untracked in the lab repo. Also: a Ref2VA cold receipt at 864x480x124 was being
used to classify H3 I2V and score/mime, which are different measurement
surfaces entirely.

**And the separation that keeps a window honest:** three different columns,
never inferred from one another -- `model-legal window`, `machine-qualified
window`, `episode-policy cap`. The full H3 lattice is model-legal; it is not
all machine-qualified (f277 hit 14.72 GiB, over the gate) and not all
episode-legal.

**Runnable check:** every lane's evidence rows land in
`docs/evidence/video_evidence_manifest.json` with the full key, and gate G4
reads that manifest rather than a comment.

**Twin assertion:** `test_lane_preflight_matrix.py::test_evidence_manifest_is_well_formed`.

---

## Per-lane log

Append one section per closed lane: what bit, the root cause, the check that
would have caught it, and the twin assertion added. A lane that hit nothing new
still gets a line saying so.

<!-- LANE LOG BEGINS -->

## Lane 1 -- `wan22_high_i2v` (`wan_i2v`), closed 2026-08-11

Three things bit. Two were the seed lessons doing their job; the third was new
and is now L8.

**What bit (1): the wrong default was only HALF the L1 defect.** The audit
named `_installed()`'s bare `os.path.exists` as the killer, and it was -- but
fixing only that still left the lane dead off the ComfyUI runtime, because the
`folder_paths` fallback's last resort was `<comfy_root>/models/<category>` and
this box keeps its weights in `C:\ComfyUI-Models`. Inside a live server
`folder_paths` reads `extra_model_paths.yaml` and finds them; in the CPU suite,
in the preflight matrix, in any tool that asks "is this lane installed?", the
import fails and the answer was a confident, wrong NO.

**Root cause:** two different questions were being answered by one probe --
"where would the LOADER find this?" (folder_paths, live) and "is this weight on
this box?" (any configured root, always). The second had no answer off the
runtime.

**Runnable check:** does the lane resolve its weight with NO environment
variables set at all? Not "does it resolve on my machine", where a leftover
`OTR_*_CKPT` export in the shell can make a dead lane look alive -- that is
exactly what masked this: `wan_ti2v` read as installed only because an env pin
happened to be exported, so the two WAN lanes looked different for a reason
that had nothing to do with their code.

**Fix:** `wan_shared.configured_models_root()` -- one spelling of "where this
box keeps its models", the same override chain `_otr_gguf_backend._models_root`
already used -- probed LAST in `_resolve_model_file_by_token`. Additive by
construction: every earlier probe still wins, so it can only turn a false
negative into the truth. It fixed all three WAN lanes at once, which is why it
belongs in shared code and why the sibling lanes got non-regression coverage in
the same chunk without being marked green.

**Twin assertion:** `tests/test_wan_i2v.py::test_a_models_root_override_is_honoured`
(behavioural, against a staged temp root) and
`::test_weight_resolution_does_not_stop_at_one_hardcoded_location`.

**What bit (2): a declaration moves a control that other tests lean on.**
Declaring `render_canvas` on wan_i2v broke two tests in
`tests/test_ltx_8gb_canonical_canvas.py` that used this lane as their
"declares NOTHING" differential control, plus the `ENGINE_MATRIX.md` drift
gate. All three were CORRECT failures -- the suite noticing that the world
changed. The control has now moved twice (wan_ti2v 2026-08-02, wan_i2v
2026-08-11) and is on `mesh_stage`.

**Runnable check:** before declaring a canvas, grep the test tree for the
lane's id used as a NEGATIVE control (`declared_render_canvas(x) is None`,
"takes the landscape default", "declares nothing"). Move the control to a lane
that still declares nothing and say in the test WHY it moved -- the invariant
outlives every occupant, so the test is edited, never deleted.

**What bit (3):** see L8 below -- the naming table said 2.1 and the weight says
2.2.

**What did NOT bite, worth recording:** the module-scope bijection assert
(L5's blast radius) never fired, because the rename ADDED a public row for a
lane that had none rather than adding a second row for a lane that already had
one. The live menu was checked on the running server after the change -- 27
options, `wan22_high_i2v (16:9)` present -- rather than only in the CPU suite,
since an empty ComfyUI menu is precisely the failure a CPU suite cannot see.

---

## Lane 2 -- `humo14_high_audio_in_wide` (`humo_14B_169`), closed 2026-08-11

**THE LEDGER PAID FOR ITSELF ON THE FIRST TRY.** L1 says to check weight
resolution before writing code. HuMo had the identical defect wan_i2v died of --
both `_ckpt_path` implementations stopped at
`<comfy_root>/models/diffusion_models`, so off the ComfyUI runtime a correctly
installed HuMo read as MISSING. Found by reading the ledger, not by a failed
render. Two copies of the chain were also two places to fix it; there is now one
resolver shared by all four tiers.

**What bit (1): a mechanism with no consumer is a mechanism that does not
work.** The `humo_diet` boot contract had been "configured" in the corpus for
days, and `--reserve-vram` had a launcher hook while `--disable-pinned-memory`
had none -- so a profile selecting the diet would have clamped exactly one of
its two knobs and the other was a markdown string. The knob only became real
when a lane needed it.

**Runnable check:** for every boot/profile field a lane depends on, trace it to
the argv or env that consumes it, and then verify against the RUNNING process
rather than the config. `Select-String` the server log for the flag, or read
`comfy.cli_args.args`. A check that reads the same config the launcher was meant
to honour cannot tell "applied" from "written down".

**Twin assertion:**
`tests/test_boot_contracts.py::test_the_launcher_turns_both_diet_knobs_into_argv`
-- which asserts the hook exists AND that the variable reaches the command line,
because declared is not applied.

**What bit (2): the smoke's number was not the corpus's number, and that is
L7 rather than a contradiction.** The live cold render peaked at 14,604 MB
absolute against a headline 13.06 GiB warm. Different cache state (cold, model
load inside the window) and different measurement surface (device-total,
including the ~1,940 MB idle baseline; net of it, roughly 12.66 GiB). Both are
true; neither is the other.

**Runnable check:** before comparing any two VRAM numbers, state the surface and
the cache state of each. If either is unknown, they are not comparable, and a
receipt that puts them in one column is manufacturing agreement.

---

## L9 -- Fix a defect at its root and the gate that watched it may go blind

**Check:** after refactoring a resolver, a stamp or a guard into a shared
helper, re-run the checks that watched the OLD shape. A gate that looks for a
token inside a named method stops seeing it the moment the token moves one call
away.

**Symptom:** a gate flips RED on four lanes that just got BETTER, or -- far
worse in the other direction -- stays green while the thing it watched is gone.

**Origin (lane 2):** factoring both HuMo `_ckpt_path` chains into one
`_resolve_unet` made preflight G1 report all four HuMo tiers as resolving a
hardcoded default, because the gate searched a fixed list of resolver method
names and `_resolve_unet` was not on it. The lanes were strictly more correct
than before. The gate was right to be narrow -- a whole-module search would let
an unrelated `folder_paths` mention launder a hardcoded path -- so the fix was
to teach it the new name, not to widen it.

**Runnable check:** when a lane's preflight row changes state in a commit that
was not about that gate, the gate is the suspect, not the lane.

**Twin assertion:** `WEIGHT_RESOLVER_METHODS` in the preflight suite is a named,
commented list rather than a heuristic, so adding a resolver is an explicit act.

---

## L8 -- A public id is a claim about the model, and claims go stale

**Check:** does every user-facing id state the model version the lane actually
LOADS? Check it against the weight basename and the frozen recipe id, not
against the naming table in a spec.

**Symptom:** none, ever, from the code's side. The lane runs perfectly while
telling every user the wrong thing, and the id is durable -- it becomes a saved
value in workflow graphs, so correcting it later costs an alias forever.

**Root cause:** naming tables are written from memory at plan time, and a
version number is exactly the kind of detail that survives a review because it
looks like a typo rather than a claim.

**Origin (lane 1):** the transplant spec's naming table prints
`wan21_high_i2v` for this lane. The lane is Wan **2.2**: the installed weight is
`wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`, the frozen recipe id is
`wan22_14b_i2v_single_pass_v1`, and `registry.CAPABILITIES` carries a dated
comment recording that this row was corrected FROM a stale `wan2.1` label TO
`wan2.2-i2v` once already. The same mislabel came back through a doc.

**What was done, and why it is not a spec override:** the live menu id states
`wan22_high_i2v`, and the spec's `wan21_high_i2v` was registered as a LEGACY
ALIAS -- so it resolved rather than being a dead end, and neither spelling ever
stopped working. Flagged for the operator rather than silently chosen, and not
a spec edit.

**OUTCOME, 2026-08-11: ruled `wan22_high_i2v`, and the diagnosis was better
than the lesson's.** The operator's answer was that the naming had been decided
all along -- `wan21` was ONE mistyped version number in the spec, and every
downstream document inherited it. So the failure mode is narrower and more
worrying than "claims go stale": a single typo at the source of a reviewed
corpus propagates silently through every review round, because reviewers check
consistency WITH the spec rather than the spec against the artifact. Seven
review passes did not catch it; reading the weight filename did. The check
below is unchanged and is exactly the one that works -- assert the version
token against the lane's own weight basename and recipe id, never against
another document.

**Runnable check:** for every public id, assert the version token against the
lane's own weight basename and recipe id.

**Twin assertion:**
`tests/test_public_engines.py::test_the_naming_convention_rows_state_the_model_they_load`
and `tests/test_wan_i2v.py::test_the_lane_is_named_wan_2_2_because_that_is_what_it_loads`.

## Lane 3 -- `humo17_high_audio_in_portrait` + `humo17_high_audio_in_wide`, closed 2026-08-11

Two lanes in one packet because they are one checkpoint at two aspects. Almost
everything came free from lane 2, which is the ledger working as designed. Two
things were new.

**What bit (1): a public rename breaks tests that hardcode the aspect suffix.**
Three tests asserted `"%s (16:9)" % public`, which was true only because every
public row so far happened to be landscape. The first PORTRAIT public row made
a correct rename look like a bug in three places at once.

**Runnable check:** before adding a public id for a lane, grep the test tree for
`(16:9)` next to the naming tables and for the lane's bare internal id used as
an expected widget VALUE. Derive the suffix from `_aspect_suffix(internal)`
rather than writing it.

**Twin assertion:** `tests/test_public_engines.py::_expected_label` is now the
one place the label shape is built, and
`test_still_aspect_and_labels.py::test_label_suffix_is_aspect_derived` pins a
portrait public row explicitly.

**What bit (2): a renamed lane changes what the APPLIER writes into node 87.**
`test_slot_matrix_soak` asserted the saved widget value equalled the bare
internal id. A lane with a public id gets its generated menu LABEL instead, so
the assertion failed on behaviour that is correct. But asserting the generated
label for EVERY engine would have been wrong the other way -- an engine with no
public row still gets its bare id.

**Runnable check:** the contract on a saved widget value is that it RESOLVES,
not that it is spelled a particular way. Assert
`resolve_engine_id(saved) == engine`, and assert the exact generated label only
for lanes that actually carry a public row (read from `_INTERNAL_TO_PUBLIC`, so
the next rename does not have to remember the file exists).

**What did NOT bite:** the weight resolver, the boot contract, the manifest
fields and the canvas-override refusal all came from lane 2 unchanged. Lane 3
wrote no new resolution code at all.

---

## L10 -- A correct test can become a bug's bodyguard without anyone touching it

**Check:** when this lane depends on a number pinned in a profile, a variant or
a test, ask what OTHER module would have to change for that number to stop
being true -- and assert against the SOURCE OF TRUTH rather than the literal.

**Symptom:** the fix looks like the regression. Three green tests fail the
moment the real defect is corrected, and the suite argues -- confidently, in
triplicate -- for the broken behaviour.

**Root cause:** a pin written as a correct low-VRAM launch contract goes stale
by REMOTE action. `config/profiles/otr_8gb_wan.json` pinned
`video.max_render_frames: 17` and its env twin `OTR_WAN_TI2V_MAX_FRAMES: "17"`;
both were right until 2026-08-02, when `wan_ti2v` was added to
`frame_contract.PLANNING_CAP_ENGINES` and the adapter-side compensation that
made a render cap harmless was ripped the same day. From that moment the pin
narrowed the PLANNER instead of the render, so every beat on that profile
became a chain of 0.68-second segments -- while the adapter's own comment still
claimed WAN was "deliberately excluded from PLANNING_CAP_ENGINES". The comment
described protection that had already been deleted.

**Origin (lane 5):** `test_otr_8gb_wan_profile_pins_low_vram_contract`,
`test_applied_8gb_variant_pins_17_and_other_tiers_stay_unpinned` and
`test_the_wan_8gb_variant_still_carries_its_real_17_frame_ceiling` all asserted
17 as correct.

**Runnable check:** a test that pins a tunable number asserts it against the
profile or the declaration it comes from, never against a literal copy. Then
the number can move without any test lying in either direction.

**Twin assertion:** two of the three now read the profile. The third keeps its
literal deliberately, because its whole job is to pin the SHIPPED variant, and
that one is named so the next reader knows the difference.

**Second half, same lane, different mechanism:** `workflows/variants/*.env.json`
is **not** generated by `scripts/build_variants.py` -- only four exist and they
are kept BY HAND. Regenerating a variant rewrites the graph and the launch
recipe and silently leaves the env recipe behind, still carrying the old number
and a `master_hash` for a graph that no longer exists. Guards now exist
(`test_the_hand_kept_env_recipe_cannot_drift_from_its_profile`,
`test_the_hand_kept_env_recipe_carries_the_LIVE_master_hash`), so the next
mover is told rather than finding out in a leg.

---

## Lane 4 -- `humo14_high_audio_in_portrait` (`humo`), closed 2026-08-11

The 2026-06-09 keystone, and the tier that closed the family: four tiers, four
canvas declarations, four boot-contract lists. Only two things were this lane's
own work -- the canvas declaration and the public id. The weight resolver, the
boot-contract mechanism, the manifest fields, the unconditional exact-fit guard
and the override refusal all arrived in lanes 2 and 3 and applied unchanged.
That is the ledger's whole argument, stated as a measurement rather than a hope.

**What bit: BOTH its profiles were claiming landscape**, on the tier whose
entire job is the pillarbox talking head. `otr_w45_humo.json` said 832x480 and
so did `otr_g4_humo.json`. The w45 one is the one a human would think to check;
the g4 one surfaced only because G2.3 reads EVERY profile that selects the
engine, not just the one being edited.

**Runnable check:** when declaring a canvas, enumerate every profile whose
engine selection resolves to this lane -- not the profile you are editing. A
by-hand fix leaves half the lie in place.

**Twin assertion:** `test_lane_preflight_matrix.py::gate_g2_canvas` G2.3,
already generic; the lane needed no new assertion, only the gate it already had.

**A control with no occupant left gets REWRITTEN, not deleted.** `humo` held
the "declares NOTHING" differential control in two test files, and with the
family closed there is no HuMo tier left to hold it. `test_boot_contracts.py`
stopped parking the invariant on whichever tier had not been done yet and now
asserts the SCOPING RULE directly -- strip the declaration, the overrides go
back to winning. `test_ltx_8gb_canonical_canvas.py`'s list simply lost its
third occupant; `mesh_stage`, `ltx_audio_in`, `still_pan` and `viz_mxc_cpu`
remain, and each leaves when its own packet runs. (Lane 7 takes `ltx_audio_in`
out of that list -- read this paragraph before you do.)

**Cold absolute peaks, the family in a readable order** (all COLD, device
total, ~1,890 MB idle baseline included -- state the surface, per L7):

| Tier | Rung | Canvas | Peak |
|---|---:|---|---:|
| `humo` (14B portrait) | f97 | 480x832 | 13,800 MB |
| `humo_14B_169` (14B wide) | f97 | 832x480 | 14,604 MB |
| `humo_1.7B` (1.7B portrait) | f129 | 480x832 | 15,261 MB |

The 1.7B is the most expensive of the three because it renders a third more
frames, not because it is a heavier model. **Frame count dominates on this
family** -- the single most useful thing these legs say to any admission work.

---

## Lane 5 -- `wan22_high_video` (`wan_ti2v`), closed 2026-08-11

The lane that carried a LIVE production bug shipping since 2026-08-02, and the
first naming MOVE. Both are written up as L10 above and in the lane receipt;
what belongs here is the third thing.

**What bit: the first MOVE, and why it could not be an ADD.** `wan_8gb` did not
stay as a second public row -- it moved into `_LEGACY_ENGINE_ALIASES`. Two
public ids on one internal id collapses `_INTERNAL_TO_PUBLIC` and trips the
module-scope bijection assert at IMPORT time; because the director imports that
module unguarded, the blast radius is most of OTR vanishing from the ComfyUI
node menu rather than one lane failing cleanly (L5's wider shape, now proved on
a real rename).

**Runnable check:** after any rename, verify on the RUNNING server that the old
id, the old id with its aspect suffix, the new id and the bare internal id all
resolve, that the old id appears in NO menu option, and that the menu row count
is unchanged. A CPU suite cannot see an empty ComfyUI menu.

**What was deliberately NOT done, and is not an omission:** the cost row stays
DISQUALIFIED per standing default Q3. `QUALIFIED_COST_ROWS` is empty and the
manifest says "admission NOT enforced" for this lane, in words.

---

## Lane 6 -- `wan22_high_fast` (`fastwan_8gb`), closed 2026-08-11

**The healthiest lane in the audit: 7/7 green before the packet started and 7/7
after.** Canvas declared and tested, ceiling pinned at a measured rung, LoRA
absence already failing preflight closed, seed trap already pinned. Nothing was
broken, so nothing was fixed. A lane that sails through is evidence the ledger
is working, and it is recorded as such.

**The one wrinkle worth a check: an IDENTITY public row needs no alias on the
way out.** `fastwan_8gb`'s public id WAS its internal id, so unlike `wan_8gb` it
required no `_LEGACY_ENGINE_ALIASES` entry -- a bare internal id already passes
through `resolve_engine_id` step 3. Adding one would have been harmless but
misleading, implying a rename that never happened at the internal level.

**Runnable check:** before writing an alias row for a renamed lane, ask whether
the OLD public id and the internal id are the same string. If they are, the
alias is noise.

**The throughput claim, proved live rather than quoted:** same boot, same still,
same canvas, same rung as lane 5 -- 81 frames at 832x480 in **70.5 s** against
`wan_ti2v`'s **171.2 s**, 2.43x. That is one cold pair, not a benchmark; the
label's "~2.7x" comes from the lab and the label sells throughput and only
throughput, with a test forbidding the words "better", "hq" and "high quality"
in it.

**An unrelated flake, recorded so the next window does not chase it:**
`tests/test_feed_fetch_seam.py::TestBoundedRequest::test_a_redirect_into_the_private_network_is_refused`
failed once during this lane's suite run and passes in isolation. It coincided
with the Wi-Fi DNS drop that also killed a `git push` -- a network-dependent
test caught in a network outage, not a code defect.

---

## L11 -- A derived canvas is a canvas, and the grid applies to it

**Check:** does this lane compute any INTERMEDIATE resolution from the render
canvas -- a half-canvas pass, a tile, a base latent, an upsample target? If so,
run the same legality check on the derived value that you run on the full one,
and derive the constraint on the FULL canvas from it.

**Symptom:** none from the code's side, for weeks. The full canvas passes every
gate, the render succeeds, and an illegal intermediate rides underneath it. The
suite can even PIN the illegal value as correct.

**Root cause:** `assert_ltx_dims` was called on the full canvas at two sites and
never on the derived one. Nothing in the codebase knew the derived value
existed except the graph builder that computed it.

**Origin (lane 7):** `_build_graph_ia2v` halves the canvas for its stage-A
motion pass (`base_w, base_h = width // 2, height // 2`). At the lane's live
832x480 that is 416x240, and `240 % 32 == 16`. Three things asserted it was
fine: the driver comment said "base 416x240 (all /32)",
`tests/test_ltx_av_ia2v_canonical.py` pinned 416 and 240 as expected, and the
full-canvas gate passed because 832x480 IS /32.

**The arithmetic worth keeping, because it generalises:** stage B feeds that
latent to `LTXVLatentUpsampler`, whose installed schema
(`comfy_extras/nodes_lt_upsampler.py`) takes `samples` / `upscale_model` / `vae`
and NO target size -- it "upsamples a video latent by a factor of 2", full stop.
So the DELIVERED canvas is 2x the stage-A base, and therefore:

* the stage-A latent is /32-legal **iff the full canvas is /64 on both axes**;
* snapping the base up to fix it is NOT a fix -- base 416x256 delivers 832x512
  against a declared 832x480, trading an illegal latent for a canvas lie.

/64 rungs, with the aspect stated so nobody re-derives them: 1024x576 (exact
16:9), 896x512 (7:4), 1280x704 (1.818, and the one that breached the ceiling).
Only 1024x576 is both /64 and exact 16:9.

**Runnable check:** grep the lane's graph builder for `// 2`, `* 2`, `//`, and
any `width`/`height` arithmetic, then assert the result against the model's
spatial multiple at the point it is computed -- not at the caller.

**Twin assertion:** `eng_ltx_av._build_graph_ia2v` now calls
`_AVD.assert_ltx_dims(base_w, base_h, length)` on the derived latent, and
`tests/test_ltx_av_ia2v_canonical.py::test_two_stage_topology` pins the legal
512x288 with the reasoning in the test.

---

## L12 -- Check a contract-bearing env var RAW, not after its own fallback

**Check:** does the refusal that compares environment to declaration read the
RAW `os.environ` string, or a module constant that a crash-guard has already
normalised? If the latter, the refusal cannot see the disagreement it exists to
catch.

**Symptom:** the check passes, the operator's variable is ignored, and nothing
says so. Strictly worse than having no check, because the check's existence
reads as proof the case is handled.

**Root cause:** two correct rules composed into a wrong one. `_env_num` exists
so a typo cannot take the IMPORT down (it warns and returns the declared
default) -- that is right. `assert_env_matches_contract` exists so an env value
that disagrees with a declaration is a refusal -- also right. But comparing
`_LTX_AV_MAX_FRAMES` (already fallen back to 497) against `max_frames=497`
reports AGREEMENT for `OTR_LTX_AV_MAX_FRAMES=garbage`.

**Origin (lane 7):** found by the kibitz r1 panel (codex, MUST-FIX 4) in the
first draft of this lane's own refusal, before it shipped.

**Runnable check:** for every contract-bearing variable, assert that BOTH a
parseable disagreement AND an unparseable value raise. If the test only covers
the parseable case, the fallback hole is still open.

**Twin assertion:**
`tests/test_ltx_av_driver_wiring.py::test_a_contract_bearing_env_var_is_checked_RAW_not_after_the_fallback`,
parameterised over both shapes.

---

## Lane 7 -- `ltx23_low_audio_in` (`ltx_audio_in`), closed 2026-08-11

Four owned defects, and the two biggest turned out to be one defect wearing two
spec numbers. Full write-up in the receipt; what the NEXT lane must check:

**What bit (1): S3 and S8b-10 were the same defect.** The spec listed them
separately -- "declare render_canvas = (1024, 576)" and "the ia2v stage-A base
416x240 is not /32-legal" -- and they read like a preference and a bug. They are
one bug: the canvas was decided by an inline RECIPE-DEPENDENT branch in the
driver that `declared_render_canvas` would have overruled anyway, and the value
that branch chose has no legal stage A. Declaring 1024x576 fixes both, and the
reason is arithmetic (L11), not spec obedience.

**Runnable check:** when a spec lists a "declare X" item and a "Y is illegal"
item on the same lane, check whether declaring X makes Y legal before treating
them as two jobs.

**What bit (2): the panel found a hole in my own fix before it shipped.** See
L12. Worth recording because the fix LOOKED complete and had a passing test.

**What bit (3): lane 5's rename never reached four other variants.** Five
committed variants still carried `wan_8gb (16:9)` in node 87 --
`otr_amd16_rocm`, `otr_amd8_rocm`, `otr_nv40_12gb`, `otr_upscale_ship`,
`otr_sbcov_5`. Lane 5 regenerated only the variant whose profile it edited, so
`scripts/build_variants.py --check` had been RED since that lane closed, and
lane 7 could not tell its own drift from inherited drift.

**Runnable check:** a rename regenerates **every** variant, not the lane's own
-- node 87 carries an engine string in variants that have nothing to do with
the renamed lane's profile. Run `scripts/build_variants.py --check` BEFORE
starting a lane, so an inherited red is attributed to the lane that caused it.

**What bit (4): the LTX boot lane enabled only one of the two LTX engines.**
`_otr_soak_server_launch.cmd`'s `LTX` token set `OTR_ENABLE_LTX_VIDEO=1` and not
`OTR_ENABLE_LTX_AV=1`, so this lane could not be smoked on the boot it declares
without exporting a flag by hand. A boot lane you have to supplement by hand is
not a boot lane; the token now enables both.

**Runnable check:** before smoking, confirm the lane's engine is ENABLED by the
boot token it declares -- read the launcher, then confirm on the running server
that the engine appears in `/object_info`. A fail-closed `EngineUnusable` and a
missing enable flag look identical from the smoke's exit code.

**What did NOT bite:** the weight resolver, the manifest fields, the V-1
self-probe and the admission honesty gate were all already green on this lane
before the packet started (G1, G3, G4, G5, G7 never went red).

---

## A process defect this ledger caught about ITSELF (2026-08-11, lane 7)

Lanes 4, 5 and 6 closed green and pushed **without appending to this file**.
Step 9 of the per-lane loop is "append what bit you to the lessons ledger", and
three lanes in a row skipped it; the sections above were reconstructed from
`docs/evidence/lane_receipts/lane0{4,5,6}-*.md` at the start of lane 7.

**Runnable check:** the commit that closes a lane touches
`docs/LANE_BUILD_LESSONS.md`. If `git show --stat <lane-commit>` does not list
this file, the lane is not closed -- the receipt records what happened, the
ledger records what the NEXT lane must check, and they are not the same
document.

---

## L10 -- A lane's peak must reach disk from the PROBE, not from a sample

**Check:** after a solo smoke, is the lane's VRAM peak readable from an artifact
on disk, and did it come from `VramPeakProbe`? If the only number you have is
one you read off `nvidia-smi` while watching, you have a LOWER BOUND, not a
peak, and it must not seed a cost row.

**Symptom:** none at smoke time. The render passes, the receipt looks complete,
and the missing number is only noticed later -- when someone tries to calibrate
an admission row and finds the field null on every clip the harness produced.
The tempting repair is to re-render, which pays GPU time for data the first run
already computed.

**Root cause:** the engine measured it correctly and something downstream
dropped it. `eng_wan_ti2v` and `eng_fastwan_8gb` both run `VramPeakProbe` and
thread the maximum into their clip dict; `render_driver._clip_summary` returned
a six-key summary that did not include it, so the single-engine smoke wrote a
report with no peak in it. Measured, threaded, and discarded one function short
of disk.

**Two rules that came out of it, both binding (operator, 2026-08-11):**

- **PROVENANCE.** A cost row may be seeded ONLY from a true `VramPeakProbe`
  maximum. A single `nvidia-smi` reading is a lower bound and never seeds a row.
  Record it if you have it, with `seeds_cost_row: false`, so the number stays
  readable without becoming usable.
- **NET, NOT ABSOLUTE.** `free_vram_mb()` returns `mem_get_info()` FREE bytes
  and `compute_real_frame_budget` compares against `free * 0.85`. FREE already
  excludes the resident desktop baseline, so an overhead derived from an
  ABSOLUTE peak double-charges that baseline on every prediction -- which is
  exactly how the shipped WAN row came to refuse every segment length the
  planner produces. `net = absolute_peak - the leg's own pre-queue baseline`,
  and the 0.85 margin is where conservatism belongs.

**Better still, once S7.1 lands:** measure in the units admission compares in.
Record `free_vram_mb()` at render start and its MINIMUM during the window; the
difference IS the render's demand, with no baseline arithmetic to get wrong.
Baseline subtraction is a first-order correction and the baseline is not
constant across a render, because ComfyUI evicts and reloads. Re-derive then,
and treat any disagreement with the subtracted figures as a FINDING rather than
a correction.

**Runnable check:** after a solo smoke, assert the durable report carries a
non-null `vram_peak_mb`. If it is null, do not re-render -- find what dropped
it between the adapter and the file.

**Twin assertion:** the passthrough in `render_driver._clip_summary`, plus
`seeds_cost_row` in `docs/evidence/video_evidence_manifest.json` so an
unqualified number cannot be picked up by accident.
