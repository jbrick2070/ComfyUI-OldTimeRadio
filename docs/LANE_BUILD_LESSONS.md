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

## L14 -- "I could not check" is not "it passed"

**Check:** for every gate that reads external state -- a running server, a
probe, an import that may not exist -- ask what it returns when it CANNOT LOOK.
If that answer is indistinguishable from "satisfied", the gate is decorative on
exactly the machines where it matters most.

**Symptom:** a green check on a box that never verified anything. Nothing in
the log, because from the caller's side an empty problem list IS success.

**Root cause:** the unknown case was written as `return []` with a comment
saying "the caller decides" -- and no caller decided. `assert_running_server`
treats an empty list as met and raises nothing, so a contract constraining real
VRAM clamps evaluated as COMPLIANT wherever `comfy.cli_args` could not be
imported. The same shape appeared twice in one file: `running_server_boot_state`
recorded `sage_probe_error` when the Sage probe raised, and NOTHING ever read
that field, so a failed probe left `sage_attention = None` and the comparison
skipped -- silently passing a Sage-constrained contract on the exact lanes Sage
silently corrupts. **Recording an error nobody reads is swallowing it.**

**And the test that should have caught it pinned the bug instead.** It was
named `test_unknowable_is_not_the_same_as_satisfied` and its body asserted
`check_running_server(HUMO_DIET) == []` -- the name stated the invariant, the
body enforced its violation. That is why this survived six lanes of green runs.

**The distinction that makes the fix correct:** a contract that constrains
NOTHING (`default`) is genuinely satisfied by an unreadable server -- there is
nothing to violate. A contract that constrains something is not. Draw the line
on what the SPEC asks for, never on whether the check happened to be able to
look.

**Origin:** retro bug hunt r1 on lanes 0-6 (2026-08-11). Both reviewers reached
it independently, which is itself the signal -- a defect two harnesses find
without conferring is not a matter of taste.

**Runnable check:** for each contract, assert that an `{"available": False}`
state produces a REFUSAL when the contract constrains a knob, and an empty list
when it does not. Assert a failed probe refuses too.

**Twin assertion:** `tests/test_boot_contracts.py::test_unknowable_is_not_the_same_as_satisfied`
(rewritten to assert its own name) and `::test_a_failed_sage_probe_is_not_a_pass`.

---

## L13 -- A defect found in one adapter is a defect in every adapter sharing the mechanism

**Check:** when a lane's defect is caused by a SHARED mechanism -- a common
helper, a node class both graphs use, an idiom copied between adapters -- sweep
every other adapter that shares it BEFORE the lane closes. Do not record the
defect against the lane you happened to find it in.

**Symptom:** the ledger says the defect is fixed, the preflight row is green,
and an identical instance of it is still live one file away. Worse, the second
instance now has a written record saying the class of bug was dealt with.

**Origin (lane 9, operator ruling 2026-08-11):** S8b-10 was recorded as an
`ltx_audio_in` defect -- the ia2v stage-A latent being non-/32. It is nothing of
the kind. It is a property of **any lane that halves its canvas and upsamples
with a fixed-x2 `LTXVLatentUpsampler`**, because that node takes no target size.
`eng_ltx_video`'s HQ two-stage path does exactly that and had the identical
416x240 base at its declared 832x480. Lane 7 fixed one instance and the corpus
never suggested there was a second.

**Why it hid so long, which is the useful half:** `1472x832` is ALSO /64 on both
axes. The two-stage path was legal at the old landscape default and became
illegal the moment the lane moved to 832x480 -- and nobody rechecked the
two-stage geometry against the new canvas. A canvas change can break a graph
that has not changed.

**The rule, stated so it can be run:** both axes must be **/64** for a
halve-then-fixed-x2 recipe to have a legal stage A. 832x480 fails (480/64 =
7.5); 1024x576 passes (16 and 9) and is true 16:9 besides.

**Runnable check:** grep every adapter for `// 2` on a canvas dimension and for
`LTXVLatentUpsampler`; every hit's declaring lane must have a /64 canvas.

**Twin assertion:**
`test_lane_preflight_matrix.py::test_a_halving_two_stage_lane_declares_a_64_legal_canvas`
-- generic over the registry, so a future adapter adopting the idiom is covered
on arrival rather than when someone remembers.

---

## Lane 8 -- `ltx098_low_video` (`ltx_8gb`), closed 2026-08-11

**What bit (1): adding a gate makes every test that used the gated function as
a PROXY go red.** `assert_usable` gained a node-class gate, and six "CONTROL
... still passes" checks in two files went red at once. They were not testing
node classes -- their subjects are loader tokens, DIR overrides and the
integrity floor -- but they call the real `assert_usable` on a CPU box with an
empty ComfyUI registry, so the new gate refused before their subject was ever
reached.

**Runnable check:** before adding a gate to a widely-called preflight function,
grep for tests that assert that function SUCCEEDS. Each one is now also
asserting your new precondition, whether it means to or not. Fix them at the
fixture -- give them what the new gate wants -- never by weakening the gate.

**Twin assertion:** an autouse `_node_classes_present` fixture in both files,
with the reason in its docstring, plus three tests that exercise the gates
directly (`..._refused_BEFORE_any_weight_is_resolved`,
`..._refused_at_PREFLIGHT_not_at_load`, `..._reads_the_ACTIVE_candidate_set`).
A gate with no coverage is the hole one level up from the one you just closed.

**What bit (2): a hole is invisible next door for a reason worth knowing.**
`eng_ltx_video` has had the identical node gate for a while, and the CPU suite
never calls its `assert_usable` at all -- so the gate is never exercised there
and nobody learned what it does off the runtime. `ltx_8gb` has a dedicated
`assert_usable` suite, which is the only reason this surfaced.

**Runnable check:** "the sibling does it this way and its tests are green" is
not evidence the pattern is CPU-safe. Check whether the sibling's tests
actually call the function.

**What bit (3): an ORDERING claim needs a test that can fail.** The Sage gate is
first "so a refusal costs nothing rather than a checkpoint load" -- a claim that
passes trivially if the gate is anywhere in the function. The test makes weight
resolution raise `RuntimeError`, so a mis-ordered gate fails with the wrong
exception type instead of quietly passing.

**What did NOT bite: the measurement-before-naming order paid.** The corpus told
this lane to measure before final naming and the manifest said the marker was
provisional because nothing had ever measured the lane on this box. Smoking
first turned `low` from an inherited guess into 6,835 MB net -- and the label
states the measured COST rather than "runs on an 8 GB card", which is the claim
the retired `8gb` token made without evidence.

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

---

## L15 -- A fix lands on the path you TESTED, not the path that RUNS

**Check:** when an adapter has more than one render path, ask which one THIS
BOX actually takes, and check the fix is on that one. Selection is usually
implicit -- a recipe detected from a weight's filename, a family probe, an env
default -- so the running path is a property of the installed models, not of
the code you are reading.

**Symptom:** none from the CPU suite, which exercises both paths equally
happily with fakes. The gap only appears on a live leg, and only if something
downstream fails LOUD. If the missing thing has a fallback, nothing appears at
all.

**Root cause:** `eng_ltx_video` has two render paths. `render_clip` got a
`VramPeakProbe` and a `_clip_telemetry` call; `_render_clip_hq` got neither.
`_detect_recipe` routes a `ltx-2.3-22b-dev-*` unet to `hq_two_stage`, and that
is the unet installed here -- so the probed path never runs and the running
path was never probed. The commit that added the receipt fields said, in its
own message, that it added them "because lane 9 cannot measure without them".

**And the fallback is what made it invisible AND dangerous.** `render_driver`
reads `clip.get("vram_peak_mb") or _mc.vram_used_mb()`, so a missing peak is
replaced by an instantaneous post-render sample that is shaped exactly like a
peak. Measured on the live leg: the substitute read **4,124 MB** where the true
probe maximum was **15,916 MB** -- a 3.9x understatement that would have seeded
a cost row, and a cost row built on a lower bound admits renders that then OOM.
An `or` that reaches for a weaker source is how a lane reports a number it
never measured.

**Origin (lane 9, 2026-08-11):** found by reading this ledger before writing
code, which is step 1 of the per-lane loop. No GPU time was spent discovering it.

**Runnable check:** for each adapter, list its render paths, resolve which one
the INSTALLED weights select, and assert the receipt fields on THAT one. Then
assert that a raw return with no peak canonicalizes to `None` rather than to a
substitute -- "I could not measure" must not be spellable as a number (L14, one
layer down).

**Twin assertion:** `tests/test_ltx_video_receipt_seam.py` -- the peak test pins
a probe SPY's sentinel rather than `is not None`, because `is not None` passes
on the GPU box against the very fallback it exists to forbid.

---

## L16 -- A constant measured at one canvas is not a fact about the engine

**Check:** for every magic number an adapter enforces -- a decode floor, a tile
size, a frame band -- ask WHAT IT WAS MEASURED AGAINST. If the answer is a
canvas, a rung or a boot the lane no longer uses, the number is unknown at
today's configuration and must be re-measured rather than inherited.

**Symptom:** silent, expensive, and it looks like caution. `ltx_video` forced
EVERY beat to 169 frames because 169 was the only length that decoded at
1472x832. At the declared 1024x576 a 2 s beat rendered 169 frames and the
composite discarded 119 of them -- ~3.4x the GPU work and 147.5 s instead of
75.4 s -- to satisfy a constraint that did not exist at that canvas.

**Root cause:** the number was correct when written and nothing re-derived it
when the canvas moved. Constants do not carry their own provenance, so a
measured constraint and an arbitrary constant are indistinguishable six months
later -- and the conservative-looking one never gets challenged.

**Origin (lane 9, 2026-08-11):** swept the ladder at 1024x576 under the consent
act. f9, f49, f97, f121 and f137 all decode clean -- including f121 and f137,
the exact pair that FAILS at 1472x832. The floor moved to the bottom of the
ladder and the contract became the honest `min_frames=9, quantum=8`.

**This is L13 one layer up.** L13 says a defect in a shared mechanism is a
defect in every adapter sharing it. L16 says a MEASUREMENT is a fact about the
configuration it was taken at, and a configuration change silently invalidates
it. Both are "a change over there broke something over here that nobody
edited"; the /64 stage-A defect and this floor are the same shape.

**The corollary that keeps it honest:** a moved constant needs its own re-sweep
trigger written down, or the next canvas move repeats this exactly. Both the
constant and the `render_canvas` declaration now say so in their comments, and
`assert_env_matches_contract` already REFUSES an env canvas that disagrees with
the declaration for this reason.

**Runnable check:** grep the adapter for enforced numeric constants; for each,
the comment must name the canvas/rung/boot it was measured at. A constant with
no stated provenance is a constant nobody can safely move OR keep.

**Twin assertion:**
`tests/test_engine_contract_roster.py::test_the_local_ladders_match_their_adapters_named_constants`
now asserts the declaration against `_LTX_DECODE_FLOOR_DEFAULT` /
`_LTX_MAX_FRAMES_DEFAULT` rather than against literals, so the next measurement
moves ONE number in the engine instead of two numbers in two files.

---

## Lane 9 -- `ltx23_high_video` (`ltx_video`), closed 2026-08-11

Preflight was 7/7 green before the lane opened, so this was never gate-flipping:
two measurements the corpus forbade guessing at, and what they then obliged.
Both new lessons above are this lane's. Three more things worth the next lane's
attention.

**What bit (1): six tests went red on the floor move and every one was
correct.** `test_look_qa_round5.py`'s three `TestLtxFrameCap` cases, two roster
tests and the ENGINE_MATRIX drift gate. They had all pinned 169 as a LITERAL,
and two of them were asserting nothing at all once the default moved:
`test_cap_below_floor_wins` needs a floor ABOVE the cap to mean anything and had
been riding the old default, and a roster test set
`OTR_LTX_MIN_DECODE_FRAMES = min_frames - 72`, which was a valid disagreeing 97
at floor 169 and became **-63** at floor 9 -- clamped back up by `_env_int`, so
the "disagreement" agreed and the refusal correctly did not fire.

**Runnable check:** a test that derives a value by ARITHMETIC on a tunable
(`min - 72`, `cap + 80`) must stay in range at every value that tunable can
take. Prefer moving UP from a floor and DOWN from a ceiling; a derived value
that falls outside the parser's range gets clamped and the test silently stops
testing. This is L10 wearing a different hat: the literal was not the only thing
that went stale, the ARITHMETIC AROUND it did.

**What bit (2): the hand-kept env recipe drifted exactly as L10 predicted.**
Regenerating the variants after the rename moved
`otr_16gb_ltx_video.json`'s `master_hash`, and
`workflows/variants/*.env.json` is NOT generated. The guard lane 5 wrote
(`test_the_hand_kept_env_recipe_carries_the_LIVE_master_hash`) caught it. A
guard written by an earlier lane catching a later lane's drift is the ledger
working exactly as designed, and it is recorded here as evidence of that.

**What did NOT bite:** the weight resolver, the canvas declaration, the Sage
gate, the node gate, the V-1 self-probe and the admission honesty gate were all
green before the packet started and stayed green. The only reason this lane had
work at all is that its two numbers had never been measured on this box.

**A harness trap that is NOT this lane's code, recorded so nobody re-hits it:**
handing a smoke an init still that already lives in ComfyUI's `input/` directory
makes `wrapper_bridge.stage_into_comfy_input` copy the file onto itself, which
on Windows is `PermissionError: [WinError 32]` and kills the render at the
staging seam before any GPU work, with an error that names nothing about init
images. Keep a lane's still in the lane's own directory. Flagged for its own
fix; every in-process adapter shares that helper.

---

## L17 -- A gate that reads a DECLARATION cannot be proof, whatever it is named

**Check:** for every validator this lane relies on, ask WHERE THE FIELD IT
CHECKS CAME FROM. If the answer is "the adapter under validation wrote it", the
check is a declaration agreeing with itself and proves nothing -- no matter how
strict it looks or how green it reports. Then ask what artifact on disk could
answer the same question, and check THAT.

**Symptom:** none, ever, from inside. The validator raises correctly on
malformed dicts, its tests pass, the gate is green, and the property it exists
to establish has never once been established.

**Root cause:** the two halves of a contract -- what the clip CLAIMS and what
the bytes ARE -- were both routed through the same dict. `has_audio is not
False` in `validate_directory_clip` read the literal `has_audio: False` that
`_directory_clip` had hand-written four lines earlier. Meanwhile
`list_directory_frames` selected frames by FILENAME EXTENSION, so a file named
`0001.png` containing an mp4 -- or a WAV, which is the case that makes this an
AUDIO defect and not a tidiness one -- counted as a frame and shipped as
evidence of silence.

**Origin (lane 10, 2026-08-11):** `mesh_stage` is the only directory-clip lane
in the roster, so its G5 row could not be closed by copying what nine mp4 lanes
do. Bolting an ffprobe onto a PNG directory would have satisfied the gate's
string match and proved nothing about anything -- which is the trap worth
recording, because it is the cheap move and it looks like compliance.

**The fix shape, which generalises:** find the STRUCTURAL fact that makes the
claim true and prove that instead. A PNG/EXR is a still-image format with no
audio stream to carry, so proving each frame really is one -- from its magic
bytes, not its name -- makes silence a fact about the bytes. Put the proof in
the SHARED listing helper, not the strict validator, so the tolerant read path
(`frame_dir_summary`, which the manifests and `_clip_summary` use and which
never raises) inherits it too; otherwise a receipt can still call an impostor
directory real while the validator would have refused it.

**And the half that keeps the gate honest.** Teaching a LEXICAL gate a new
function name buys a green row for a STRING, so the named function needs a test
that fails when the proof behind it rots. Teach the name per lane, never widen
the gate to "any validator" -- that would let a future lane launder a missing
proof past it (L9, same shape, opposite direction).

**Runnable check:** for each field a validator enforces, name the writer. If the
writer is the thing being validated, the check is decorative. And for every gate
taught a new name, assert the named function REFUSES the case it exists to
catch -- here, a file with a frame extension whose bytes are not an image.

**Twin assertion:**
`test_lane_preflight_matrix.py::test_the_directory_clip_audio_law_really_proves_the_frames`
(the guard on the teaching) and
`tests/test_video_directory_clip.py::test_a_frame_is_proved_by_its_MAGIC_BYTES_not_its_extension`.

---

## L18 -- "Read by nothing" and "read, then overruled" are different bugs

**Check:** before writing off a configured knob as dead, TRACE IT FORWARD one
step at a time and say where it stops. If it reaches a real consumer and is then
overwritten downstream, it is not dead -- it is a TRAP, and the fix is different.

**Symptom:** identical to a dead channel from the render's side (the configured
number never decides anything), and the opposite from the OPERATOR's side. Edit
the profile, regenerate the variant, and the widget visibly changes -- so the
change looks applied. Nothing anywhere says it was discarded.

**Root cause:** a chain audited by grepping for readers of the FIELD NAME.
`render.canvas_w` has no driver that reads it, which is true and answers the
wrong question. `_otr_workflow_apply` flattens it into the node-87
`OTR_VideoDirector` widgets, and `otr_video_director` turns those widgets into
`request["canvas"]` -- the field changes NAME at the seam, so a name-keyed grep
stops one step early and reports a dead end that is a live wire.

**Origin (lane 11's opening check, 2026-08-11, folded back into lane 10):** the
video corpus and lane 10's own first draft both called the profile canvas
channel "read by NO driver" and "read by nothing". Measured on lane 10's own
regenerated variant, node 87 moved from `25, 832, 480` to `25, 1472, 832` when
the profile changed. What actually discards it is `build_request_from_shot`,
which overwrites the request canvas to the 1472x832 landscape default for every
NON-FACE family -- and then a `render_canvas` declaration overrules that too.
So the number is read, carried into the request, and twice overruled.

**Why it matters and is not pedantry:** the two diagnoses prescribe opposite
fixes. "Dead" says document the channel and move on. "Read, then overruled"
says the config must be kept TRUTHFUL, because an operator will act on it --
which is why lanes 11-18 declare or reconcile rather than annotate.

**Runnable check:** for any field you are about to call dead, follow it through
every rename. A profile field ends at a WIDGET at least as often as it ends at a
driver, and `git grep <field>` cannot see past the assignment that renames it.
Then confirm the terminus by MEASURING -- change the value, regenerate, and diff
the artifact.

**Twin assertion:**
`tests/test_video_mesh_stage.py::test_the_profile_canvas_agrees_with_the_declaration`
(the drift guard the accurate diagnosis obliges), and the corrected wording in
`PROFILE_CANVAS_DOCUMENTED_DEAD` + G2.3's failure message, which is what every
later lane reads when it decides how to close its own G2 row.

---

## L19 -- A `render_canvas` declaration is a LEVER REMOVAL, so a lane with no native canvas must not declare one

**Check:** before declaring a canvas, ask what that number IS. If it is a
property of the lane -- a latent grid, a trained input size, a canvas-dependent
constant the lane enforces -- declare it. If it is the value of an operator
knob the lane merely receives, do NOT: `declared_render_canvas` is applied LAST
and overrules every earlier channel, so declaring silently disables that knob
for this lane alone.

**Symptom:** none, and it reads as tidying up. Every gate goes green, the smoke
agrees with production for the first time, and the change is described in the
commit as a declaration of existing behaviour. What actually shipped is that one
lane stopped honouring an env var its siblings still honour, decided by which
lane's packet happened to run first.

**Root cause:** two different facts wear the same number. `build_request_from_shot`
hands every non-face family `OTR_VIDEO_LANDSCAPE_CANVAS`'s default of 1472x832,
and `render_single` hands wide lanes `OTR_VIDEO_RENDER_CANVAS`'s 832x480. A lane
that renders at "1472x832 in production" may be stating a fact about the DRIVER's
default, not about itself -- and the difference is invisible until someone moves
the knob.

**Origin (lane 11, 2026-08-11):** `viz_green`'s first draft declared
(1472, 832) on exactly the measured argument above -- the two builders disagreed,
so the smoke was validating a size production never uses, which is lane 7's
finding almost word for word. But this engine paints and encodes at whatever the
request carries: no latent grid, no fixed model input, nothing canvas-dependent.
Found by the Codex consult, which read the override path the draft had not.

**Lesson L2 ALREADY CONTAINED THIS CHECK** and it was walked past: "a canvas
declaration must therefore either agree with those overrides or the lane must
state that they are unsupported once a canvas is declared. Check the OVERRIDE
PATH, not just the default." L19 exists because a check buried inside another
lesson's supporting paragraph did not fire when it mattered. If a check keeps
getting missed, promote it to its own entry.

**And the measurement that settles it, worth copying as a technique:** the same
smoke was run twice -- once with the declaration, once with the declaration
removed and `OTR_VIDEO_RENDER_CANVAS=1472x832` instead -- and the two mp4s have
the SAME sha256. Byte-identical output proves the declaration was purely
canvas-SELECTION and bought nothing the lever did not already do. When arguing
about whether a change is behavioural, render it both ways and diff the digest.

**Runnable check:** for every lane that declares a canvas, assert the lane has a
canvas-dependent property that justifies pinning (a grid multiple it enforces, a
measured constant, a graph that would break). For every lane that does NOT
declare one, assert the operator lever still reaches it through the REAL request
builder -- so a future declaration cannot be added without a test saying what it
cost.

**Twin assertion:**
`tests/test_video_visualizer.py::test_the_landscape_lever_still_reaches_this_lane`
(drives `build_request_from_shot` with `OTR_VIDEO_LANDSCAPE_CANVAS` moved) and
`::test_the_lane_DECLARES_NO_canvas_and_honours_ANY_request_size`.

---

## Lane 11 -- `viz_green`, closed 2026-08-11

Two red gates, and the lane's own lesson is that **its first draft closed the
harder one the wrong way and the fix was to delete the code it had added.**
Full write-up in the receipt; what the NEXT lane must check:

**What bit (1): "make the smoke agree with production" is not always a reason
to declare.** See L19. The draft's reasoning was lane 7's, correctly recalled
and wrongly applied -- lane 7's lane had a canvas-dependent graph; this one has
none.

**Runnable check:** when reusing a previous lane's argument, check the PREMISE
that made it true there, not just the shape of the conclusion.

**What bit (2): the consult earned its keep, and it earned it by reading a file
I never opened.** `tmp/_gen_profiles.py` -- a July generator script sitting in
`tmp/` -- states in its own docstring that the six `otr_sbcov_*` profiles are
"throwaway smoke config ... DELETED after the sweep. NOT committed." That
inverts the whole sbcov question from "six missing build inputs, should we adopt
them?" to "twelve LEAKED artifacts of a finished sweep, should they be deleted?"
-- and I had spent the analysis reasoning from git metadata alone.

**Runnable check:** when a file's provenance is the question, look for the thing
that WROTE it before reasoning from `git log`. A generator left in `tmp/` is
evidence, and `git log --all` on an untracked file is silent by construction.

**What bit (3): the pre-lane gate number is workstation-dependent.**
`build_variants.py --check` reports "46 variants" here; `git ls-files` counts
**45**. The 46th is another window's untracked `otr_upscale_ltx_probe.json`. The
gate globs the DIRECTORY, so its headline count silently includes files the repo
has never seen -- which is the same defect class as the sbcov crash one level
up.

**Runnable check:** a count quoted as a baseline must say whether it counts
TRACKED files or on-disk files. `git ls-files <dir> | wc -l` and a directory
glob are different numbers on any working tree, and only one of them is
reproducible on a fresh clone.

**What did NOT bite:** G1, G5 and G7 were green before the packet and stayed
green -- this lane already probed its own emitted mp4 for the audio law, which
is the gate lane 10 had to build from scratch.

---

## L20 -- Documenting a rule can DISABLE the check that enforces it

**Check:** for every LEXICAL gate -- one that greps source text for a token --
ask whether the PROSE explaining the rule contains that same token. If it does,
the gate is now satisfiable by its own documentation, and the better the
comments get the weaker the gate gets.

**Symptom:** perfect. Every lane green, every lane commented, and the check
enforcing nothing. There is no failure to notice because the gate never fires;
the only observable is that a deliberate mutation stops being caught.

**Root cause:** G3.3 asked `"continuity=" not in _mro_source(type(eng))` -- a
substring search over the class's source, comments included. That was sound
while nobody wrote about continuity. Then lanes 10, 11 and 12 each added a
comment explaining WHY their lane's value is `CONTINUITY_NONE`, and every one of
those comments contains the literal `continuity=`. From that moment the gate
would have gone green for a lane whose real declaration had been deleted,
satisfied by the paragraph explaining the declaration it no longer had.

**And the value cannot rescue it, which is what makes this class nasty.**
`CONTINUITY_NONE` is the dataclass DEFAULT, so a lane that never considered
chaining and a lane that concluded NONE after reading its own render path are
byte-identical at runtime. "Assert the resolved value" -- the usual answer to a
weak lexical check -- proves nothing here. The only readable difference is
whether the keyword was PASSED, which is a fact about syntax, not about values.

**Origin (lane 12, 2026-08-11):** found by the post-coding QA pass, on a test
written minutes earlier in the same session, after the same weak assertion had
already shipped in lanes 10 and 11. Three lanes wrote the identical tautology
without noticing, which is the argument for the QA pass existing.

**The fix, and why it is not just "use a regex":** the reader is now
`frame_contract.declares_continuity_kwarg`, which parses the AST and looks for a
`FrameContract(...)` call with a `continuity` keyword. Comments are not AST
nodes, so prose cannot satisfy it. It lives in the ENGINE module, not the
preflight suite, because the gate and every lane's own test must ask the SAME
question -- two readers of one invariant is how they drift apart.

**Runnable check:** every lexical gate needs a test that feeds it something
which TALKS about the rule without following it, and asserts refusal. If you
cannot write that test, the gate is reading text when it should be reading
structure.

**Twin assertion:**
`test_lane_preflight_matrix.py::test_g3_cannot_be_satisfied_by_a_COMMENT_about_continuity`
-- two synthetic classes, one that documents `continuity=` in a comment and a
docstring while passing nothing, one that actually passes it, with an explicit
assertion that their resolved VALUES are identical so the reason a value check
could never have caught this is stated in the test itself.

---

## L21 -- A fallback is only safe to remove once you prove nothing still falls back

**Check:** before turning a silent degrade into a refusal, enumerate what still
routes to the thing you are hardening. Not "is the fallback chain documented as
removed" -- grep for the machinery by NAME and confirm only comments remain,
then check whether anything ELSE reaches this lane automatically.

**Symptom of getting it wrong:** an episode that used to ship watchable turns
into a hard failure mid-render, and the failure is correct in principle and
catastrophic in practice because it fires on a path nobody knew was live.

**Why it is worth a lesson rather than a shrug:** the argument AGAINST removing
a fallback is usually a stale comment. `still_motion` is described in several
files as "the fallback-chain terminus the A-S6 chain humo -> humo_1.7B ->
still_motion converges on" -- including in its own render docstring. That chain
was RIPPED on 2026-07-02. So the strongest-sounding reason to keep the dark
floor was a sentence describing a mechanism that had not existed for six weeks.

**The check that resolved it (lane 15, 2026-08-11), in order:**

1. grep the machinery by name -- `UNIVERSAL_FLOOR`, `FLOOR_NAMES`,
   `make_fallback_of`. Only comments recording the rip came back.
2. `default_roles` is empty, so no role auto-selects it.
3. grep for anything else routing TO the lane. One hit,
   `check_ltx_open_health`, and reading it showed a POST-HOC manifest detector
   about engine SELECTION -- not a router, and not about a missing still.
4. Only then: what does a missing input actually MEAN here? With
   `accepts_still = True` the dispatcher mints the still, so absent means
   MINTING FAILED -- the exact case that must not ship quietly.

**And scope it, then pin the scope.** The refusal went on `StillMotionFamily`
alone; the shared base default stays `False` so the two sibling lanes are
byte-identical until their own packets decide. That is pinned by a test naming
all four families, because otherwise the next lane cannot tell whether its
family already refuses -- and a behaviour change spreading silently across three
lanes is what one-lane-at-a-time exists to prevent.

**Runnable check:** a commit that removes a fallback cites, in its message, the
grep that proves nothing still uses it. "The docs say the chain was removed" is
not that grep.

**Twin assertion:**
`tests/test_video_cheap_render.py::test_still_motion_REFUSES_a_missing_still_instead_of_a_black_beat`
(both shapes: no still declared, and a declared-but-absent path -- the common
shape of a failed mint) and
`::test_the_other_still_families_are_UNCHANGED_by_that_refusal`.

---

## L22 -- Unoccupied is not dead, and a kept branch owes two proofs

**Check:** when the last caller of a branch goes away, ask whether the branch
serves a DOCUMENTED CAPABILITY that merely has no user right now, or whether
nothing could ever reach it. Delete only the second. And if you keep it, prove
BOTH halves -- that nothing reaches it today, and that it still works when
something does.

**Symptom of getting it wrong in either direction:** delete an unoccupied
control and the next family that needs it silently gets a different behaviour
(and the helpers it called become orphans). Keep dead code and it rots
untested, which is how a "safety net" turns out to be broken the one time it is
finally reached.

**Origin (lane 17, 2026-08-11):** lanes 15-17 gave all four still families the
missing-still refusal, which emptied the last caller of `render_clip`'s
synthesised dark-floor `else:` branch. Lane 16 had written down, confidently,
that the branch should then be DELETED as dead code. On reaching it that was
wrong: the branch serves `uses_still = False`, and the attribute's own comment
documents that capability ("False families always synthesize a procedural
floor"). No such family is registered today -- so it is a control with no
occupant, exactly the shape lane 4 ruled on when the HuMo family closed and the
"declares NOTHING" canvas control lost its last holder.

**The two proofs, because "kept deliberately" is otherwise indistinguishable
from "left behind":**

1. A registry-walking test that FAILS if any engine can reach the branch, whose
   message tells that engine's author to record the ruling rather than letting
   the old behaviour quietly become reachable again.
2. A test that renders THROUGH the branch using a minimal stand-in for the
   future occupant, so the capability is proved functional rather than assumed
   from the fact that it still compiles.

**Runnable check:** for every branch you are about to delete because "nothing
calls it", grep the field or flag that selects it for a comment DESCRIBING when
it should be used. A documented capability with no occupant is a control; an
undocumented one with no occupant is dead code.

**Twin assertion:**
`tests/test_video_cheap_render.py::test_the_synthesised_floor_is_UNREACHABLE_from_every_registered_engine`
and `::test_the_synthesised_floor_STILL_WORKS_for_a_uses_still_False_family`.

---

## L23 -- The environment a module is IMPORTED under is not the one it is TESTED under

**Check:** does any module in `nodes/_otr_shared/` or `nodes/_otr_video_engines/`
reach a SIBLING package by an absolute `from nodes.<pkg> import ...`? If so it
works in the CPU suite and raises in production, and no test can see it.

**Symptom, and it has three shapes -- this is what makes it worth an entry.**
`nodes` resolves against `sys.path`. Under pytest the repo root is on the path,
so `nodes` IS this package and every such import succeeds. Inside a running
ComfyUI server it is NOT: ComfyUI has its own top-level `nodes` module (the node
registry), OTR lives under `custom_nodes/ComfyUI-OldTimeRadio`, and the same line
raises `ModuleNotFoundError` on every boot. All three live instances behaved
differently:

* **Raised and was CAUGHT, leaving a field UNKNOWN** -- the Sage probe in
  `boot_contracts`. UNKNOWN is correctly not a pass, so the lane refused on every
  server for a reason that was about an import. Dormant for as long as no
  contract constrained Sage.
* **Raised and was SWALLOWED into a stale fallback** -- the worst.
  `content_oracle.family_for_engine` fell into `_FAMILY_FALLBACK` on every call,
  so the registry was "the source of truth when present" only OFF the runtime.
  That table stops at 2026-07-05, so five engines resolved to family `""` in
  production -- not in `MOTION_FAMILIES` -- and were **silently motion-exempt**.
* **Raised outright** -- `slot_matrix.eligible_engines_for_role`.

**Root cause:** an absolute import used for what is structurally a relative one.
The fix is one character class: `from .._otr_video_engines import ...`, which
resolves through the package's own `__name__` and is correct under both names.

**Why no test caught it:** every test runs under the name that makes it work.
The bug is *in the difference between the two environments*, so the only
instruments that can see it are a live run or a check on the import ITSELF.

**Runnable check:** a live smoke is what found this. Cheaper, and now automated:
AST-walk both shared packages for `Import`/`ImportFrom` nodes whose module
starts with `nodes.`. Do NOT grep the source text -- the comment explaining the
fix necessarily quotes the broken line (see L20).

**Twin assertion:**
`tests/test_minimax_h3_video.py::test_no_shared_module_reaches_a_sibling_by_an_absolute_nodes_import`,
plus `::test_the_family_oracle_answers_from_the_REGISTRY_not_the_stale_table`
for the soft-failure half, which asserts the CONSEQUENCE rather than the import.

---

## L24 -- A contract drafted from prose clamps the knobs someone wrote down

**Check:** before shipping a boot contract (or any "run it this way" record),
find the measurement legs that actually PASSED and diff their real boot lane
against the contract. Every knob in the passing lane and not in the contract is
a knob the contract is silently relying on.

**Symptom:** the contract names real knobs, passes its own verification, and the
lane still blows the ceiling -- because the knob that decides the outcome was
never in it.

**Root cause:** `h3` was written from the spec's sentence ("sage_attention: false
plus `--disable-pinned-memory`") before any H3 measurement was read back. The lab
receipts said something else: **every** passing H3 leg on this box held
`--reserve-vram 12`, including the trained 1344x768 canvas at 9.15 GiB, while the
one I2V leg without it peaked **15.39 GiB and FAILED** on a canvas 2.6x SMALLER
and a length SHORTER. A smaller, shorter render peaking 70% higher is not a
canvas effect -- and the mechanism is plain, since reserving 12 GiB away from
model loading is what forces a 21 GB DiT to stream rather than attempt residency.

**The generalisation:** a contract assembled from what a document SAYS is a
hypothesis. The passing runs are the evidence, and the difference between them
and the failing run is the contract. Read the receipts, not the prose.

**Twin assertion:**
`tests/test_minimax_h3_video.py::test_the_h3_contract_carries_the_MEASURED_reserve_clamp`
and `::test_the_reserve_clamp_REACHES_the_launcher_and_the_profile_carries_it`
(the second is L6's rule: a boot pin no launcher turns into argv clamps nothing).

---

## L26 -- A LEXICAL test does not just fail on comments; it PASSES on wrong code

**Check:** does any test prove a behaviour by reading SOURCE TEXT --
`inspect.getsource`, a grep, a substring -- rather than by calling the thing?
If so it can pass while the behaviour is the opposite of what it claims.

**This is L20's other half, and it took three strikes in one session to see it.**
L20 records that prose containing a gate's token can SATISFY the gate. Lane 19
found the inverse: a comment can BREAK a gate (a test grepping the adapter for
`VAEDecodeAudio` failed on the docstring explaining its absence), and a source
grep for an import failed on the comment quoting the broken line. Both were
merely annoying -- they went red on correct code.

**Lane 20's was the expensive one: it went GREEN on wrong code.** The lane
claimed in a comment that it kept a portrait spine, and the test written to
prove it asserted that the engine id did not appear in the first 400 characters
of a function's source. It passed. The function returned `True` anyway through
a generic rule further down, and a second mechanism then OVERWROTE the lane's
`init_image` with a wide scene still -- on a lane whose `init_image` IS the
reference the model lip-syncs. Nothing raises; it renders the wrong identity on
every beat. Found by the post-coding QA pass, which CALLED the function.

**Runnable check:** for every assertion about behaviour, ask "what does this
test do if the function returns the opposite?" If the answer is "still passes,
because it never called it", the test is prose about prose. Call the function,
compare the value, and if the value depends on a branch in another module,
evaluate that branch's real condition rather than describing it.

**And the corollary that actually catches these:** a comment asserting what the
code below it does is a claim that needs the same proof as the code. When you
write "this lane is deliberately NOT in this list, so it gets X", run it.

**Twin assertion:** `tests/test_minimax_h3_audio_in.py::
test_the_scene_still_never_OVERWRITES_the_reference_this_lane_lip_syncs`, paired
with `::test_this_lane_DOES_take_the_scene_still_SPINE_like_its_incumbent` --
kept as a PAIR because the original error was conflating two questions (which
stills get minted, and which one becomes `init_image`) into one wrong sentence.

---

## L25 -- A guard that fires when its condition changes must be UPDATED, not deleted

**Check:** when a scope guard you wrote goes red because the thing it forbade
has legitimately arrived, ask what it was guarding FOR. Delete it only if the
answer is nothing.

**Where it came from:** lane 19 wrote
`test_lane_20s_adapter_is_NOT_registered_by_this_lane` -- scope discipline while
the sibling was somebody else's packet. Lane 20 is the commit that legitimately
retires that assertion, and deleting it would have thrown away the invariant it
existed to protect. What lane 19 actually cared about was never "the sibling
does not exist"; it was "the sibling never shares this lane's internal id",
because two public ids on one internal id trips a module-scope bijection assert
at IMPORT time and empties most of the ComfyUI menu (L5).

So the test was REWRITTEN to assert the surviving invariant: two public ids, two
separate internal engines, bijection intact. The guard outlived its own trigger.

**The generalisation:** a temporal guard ("not yet") and a structural guard
("never like this") look identical while the temporal condition holds. When it
stops holding, the structural half is what you keep -- and it is usually the
half worth having.

**Twin assertion:** `tests/test_minimax_h3_video.py::
test_lane_20s_adapter_arrived_as_a_SEPARATE_internal_engine`.

---

## Lane 20 -- `h3_low_audio_in` / `minimax_h3_audio_in`, closed 2026-08-12

Full receipt: `docs/evidence/lane_receipts/lane20-h3_low_audio_in.md`. The
cheapest lane in the campaign because lane 19 paid for it, and it rendered live
on the FIRST attempt against lane 19's three.

**What did NOT bite, and why that is the finding.** The boot contract, the
staging, the canvas guard, the length arithmetic, the 24->25 conversion and the
V-1 probe were all inherited and all correct on the first try. A shared
implementation module is worth building when the second lane is a REGISTRATION
plus a conditioner rather than an engine.

**What bit (1): the ONE value that must not be inherited is the one a reader
would assume is shared.** Both lanes are the same model at the same canvas on
the same grid -- so `continuity` looks like base-class material, and it is not.
Lane 19 earns `strict_first_frame` because `MiniMaxH3ImageToVideo` pins a
keyframe at index 0; `MiniMaxH3ReferenceToVideo` has **no `first_frame` input at
all**, so lane 20 is `soft_reference`. Putting continuity on the base with a
subclass override would have made STRICT the silent default for whichever lane
forgot -- the exact shape L19 warns about (copy the reasoning, not the shape).

**Runnable check:** before hoisting a value into a shared base, ask whether the
two lanes would both be CORRECT if one of them forgot to override it. If not,
it is not base material, and the base should refuse to supply it at all.
`_UNET_DEFAULT = None` on the base does exactly that -- `_weight_rows()` raises
a named refusal rather than guessing which 21 GB DiT this adapter meant.

**What bit (2): a plan-time gate fails LOUD and EARLY, which makes it easy to
under-test.** Extending the mouth policy is a one-line membership change, so the
tempting test asserts the mapping. The test that earns its keep asserts the
CONSEQUENCE: `mouth_owner_for_beat` RAISES `MouthPolicyError` for this
engine/family/role when `is_character_face` is False. That is what would have
happened on every H3 character beat -- before a single weight loaded -- had the
line not been extended in the same commit.

**What bit (4): the refactor introduced exactly one defect, and it was the one
this lane's own docstring warns about.** `session_identity` read the
module-level `H3_RECIPE_RECEIPT` -- lane 19's receipt -- so the moment a second
adapter shared that method, lane 20's identity described lane 19's recipe. It
stayed DISTINCT between the lanes anyway (the DiT token in the same tuple
differs), so no session would have been wrongly reused; it would simply have
been a receipt that lies. Written by the same hand, in the same change, as the
paragraph forbidding it.

**Runnable check, and it is the reusable part:** after hoisting a method into a
shared base, CALL IT ON BOTH SUBCLASSES AND DIFF THE RESULTS. Every per-lane
value in the output must actually differ. Neither lane's own tests would have
caught this -- both pass in isolation, because each one only ever looks at
itself. A hardcoded constant in a shared method is invisible until two callers
are compared side by side.

**Runnable check:** for a registration that joins an existing FAMILY, grep the
family's name and the incumbent lane's id across the driver and judge each hit:
which are correctly lane-specific and which are membership tests waiting to
happen. `_still_spine_requires_scene` was the one that correctly stayed
lane-specific here -- `ltx_audio_in` consumes a wide SCENE still, while this
lane presents a portrait as `<Picture 1>`.

**What bit (3): the API serialization is not the in-process call.** The lab's
`COMFY_AUTOGROW_V3` reference sockets serialize DOTTED
(`"ref_images.ref_image_0"`) because ComfyUI's prompt EXECUTOR flattens the
schema and reassembles the dict. Calling the node class directly bypasses the
executor, and a V3 node's `FUNCTION` is `EXECUTE_NORMALIZED` -- a plain
passthrough -- so `execute` must receive the dict it iterates. Reading the
runtime settled in minutes what a live experiment would have spent a 4-minute
render on.

**Runnable check:** when copying a graph from an API/JSON recipe into an
in-process `run_graph` call, resolve the node's `FUNCTION` first and read what
it does with kwargs. A dotted key that the executor owns is not a key the node
accepts, and the failure mode is a socket that looks connected and resolves to
nothing -- which renders successfully, with no references.

---

## Lane 19 -- `h3_low_video` / `minimax_h3_video`, closed 2026-08-12 -- THE FIRST NEW ENGINE

Full receipt: `docs/evidence/lane_receipts/lane19-h3_low_video.md`. Lanes 10-18
repaired lanes that already existed; this one ADDS a 33.1B packed AV DiT, and the
shelf's defaults inverted in two places.

**What bit (1): the smoke found two production bugs and the test suite found
neither.** Both are written up as L23 (the import) and L24 (the contract), and
both had SHIPPED. The pattern connecting them is that each was invisible to
every instrument except a live run: one lived in the difference between the test
environment and the server, the other in the difference between a spec sentence
and a measurement. **A lane's smoke is not a formality that confirms the tests.**

**What bit (2): the lexical-gate lesson fired in the INVERSE direction, twice in
one lane.** L20 records that prose containing a gate's token can SATISFY the
gate. The same instrument can also BREAK it: a test greping the adapter source
for `VAEDecodeAudio` failed on the `_build_graph` docstring that explains why the
audio decoder is absent. Both times the fix was to ask the STRUCTURE -- the built
graph, the AST -- instead of the text. **L20's check should be read in both
directions**, and its runnable form is simply: never grep source text for a token
that the correct code is likely to discuss.

**What bit (3): a new engine makes every ROSTER notice, and that is the cost of
admission.** Ten tests went red on registration -- boot-contract fixtures whose
H3 state no longer satisfied the contract, the public-id table, the still-plan
parity fixture, the layer-2 geometry equality check (my `scene_character` row
dropped "16:9" and was caught), and the session-identity roster, which correctly
refused a splittable lane holding local handles that declared no
`session_identity`. Every one was fixed at the FIXTURE or by writing the missing
declaration -- never by weakening the gate. Same shape as lane 8's six red tests.

**Runnable check:** after `@register`, run the roster suites BEFORE the lane
suite -- `test_public_engines`, `test_still_plan_parity`,
`test_still_plan_layer2_parity`, `test_multiclip_session_identity_roster`,
`test_boot_contracts`. They enumerate what a new engine owes.

**What bit (4): `render_single` is missing whatever the newest lane needs.**
Lane 7 taught it to read `declared_render_canvas`; lane 19 had to teach it to
select a boot contract, because it invents its own request and passes
`profile=None`, which means `default`. This is now a PATTERN, not an incident:
every solo lane smoke runs through that one function, so anything production
carries and it does not invent is absent from every smoke. **Before smoking a
lane that declares something new, check whether `render_single` asks for it.**

**What did NOT bite:** the frame contract, the still plan, the public surface and
the ENGINE_MATRIX row were mechanical once the grid was derived rather than
transcribed; deriving `H3_CANVAS_RUNGS` from the node's own `align_frame_count`
at import meant the published menu could not disagree with what the node snaps
to, and it reproduced the spec's tuple exactly without ever being compared to it.

**An OPEN gap this lane deliberately did not close:** no NET VRAM figure. The
leg reports 6,315 MB ABSOLUTE and no pre-queue baseline was sampled, so no net is
claimed and **nothing in this receipt may seed a cost row** (L7: name the
surface, and the cost-row surface is NET).

---

## Lane 18 -- `still_word`, closed 2026-08-11 -- AND THE CHEAP SHELF IS FINISHED

Two of this lane's three assigned items were already closed. The lane's value
was finding that out FIRST and then making it hold.

**What paid off: lane 14's rule, applied a second time.** The corpus assigned
"preserve the missing-still refusal, add/verify the ffmpeg contract". Running
the acceptance check before writing anything showed both were done -- the
refusal since Sprint B (this family was FIRST), the ffmpeg gate inherited from
lane 15's shared-base fix. Nothing was implemented.

**But "already done" is only safe once something holds it that way.** Before
this lane, `still_word` had NO test asserting the ffmpeg gate -- it was
inherited behaviour nobody checked here. The lane's deliverable is therefore the
acceptance check itself, kept as a test on the lane that owns the contracts, and
exercising both (firing the refusal through `render_clip`, firing the gate with
`find_ffmpeg` stubbed empty) rather than reading flags.

**Runnable check:** when a packet turns out to be already-done, ask what would
tell you if it stopped being done. If the answer is "nothing on this lane",
the packet's real work is the test, not the fix.

**The evidence trick that paid off three lanes running -- a builder partition.**
All four still lanes were smoked from the same still at the same canvas and
frame count:

| lanes | builder | sha256 (16) |
|---|---|---|
| 15, 16 | `ffmpeg_still_motion_cmd` (cover+crop) | `3692f155b93b5f87` |
| 17, 18 | `ffmpeg_still_static_cmd` (fit+pad) | `56d48f215d58868c` |

Two builders, two digests, exactly partitioned. Every one of those lanes' G2
arguments rests on "these lanes share a render path and that path has no native
canvas" -- now demonstrated in BOTH directions (shared paths match, distinct
paths differ) instead of asserted. It cost nothing; the renders already existed.

**Runnable check:** when several lanes' reasoning depends on them sharing (or
not sharing) a code path, hash their artifacts from identical inputs. A claim
that predicts both a match and a mismatch is falsifiable; one that predicts only
matches is not.

**The shelf-wide finding, worth stating once:** eight cheap lanes, eight INERT
G2 rows. Not one of them has a native canvas, and every one would have lost the
`OTR_VIDEO_LANDSCAPE_CANVAS` operator lever by declaring one. That makes INERT
the DEFAULT for a procedural/CPU lane and a declaration the anomaly -- a future
lane that wants to declare owes the L19 argument explicitly.

---

## Lane 17 -- `still_flat`, closed 2026-08-11 -- AND THE STILL SHELF IS FINISHED

Third still lane; all four families now refuse a missing still. Its own lesson
is L22 above. Two more things.

**What paid off: a different builder was the reason to re-check, and it was the
only lane where that mattered.** `still_flat` sets `_still_motion = False`, so
it renders through `ffmpeg_still_static_cmd` (fit+pad) rather than the pan
builder lanes 15-16 verified. Same answer, but arrived at rather than assumed --
and the smoke corroborated it: lanes 15/16 hash IDENTICALLY (shared pan
builder), this lane hashes DIFFERENTLY. **Same-builder lanes match,
different-builder lanes do not**, so the "they share a builder" claim is now
falsifiable in both directions rather than just one.

**What bit: a second scope guard fired, from a DIFFERENT sprint.** Lane 16
fired lane 15's guard; lane 17 fired Sprint B's (2026-07-03), which had asserted
since `still_word` shipped that its `still_flat` sibling still had the
always-renders floor. Rewritten, not deleted: Sprint B's reasoning -- "a silent
black floor would swallow a mint failure exactly where it matters" -- was never
wrong, it was NARROW. It matters everywhere.

**Runnable check:** when your change fires a guard written by an older sprint,
read that sprint's REASONING before editing its test. If the reasoning
generalises, you are completing it and the test should say so; if it does not,
you are violating it and should stop.

**What bit (3), caught by QA: the test that PROVES the kept branch is
unreachable could itself have produced a FALSE PASS.** Its first draft skipped
any subclass whose `render_clip` override did not mention `_require_still` in
its source. A future family that overrides `render_clip` but DELEGATES with
`super().render_clip(...)` satisfies that -- short body, no mention, different
method object -- and would have been skipped while actually running the base's
floor logic.

**Runnable check:** a test that proves a NEGATIVE ("nothing reaches this") must
be FAIL-CLOSED on the unknown case. If it cannot tell statically whether a
subject qualifies, it must demand an explicit declaration and fail without one
-- never `continue`. Source-sniffing a method body is a guess; requiring the
override to be named in a table with its reason is a decision.

**And a stale-text pattern worth naming, since lanes 15-17 all hit it:** a
behaviour change on a shared shelf leaves stale prose in roughly a DOZEN places
-- class docstrings, attribute comments, a helper's docstring in another module,
a slot-matrix constant, a test module header, and (worst) a RUNTIME LOG MESSAGE
that misleads the operator at the exact moment they are debugging the failure
you just introduced. Grep for the CLAIM, not the file: "always renders",
"terminus", "floor", "fails LOUD". The log message is the one to fix first --
a comment misleads a reader who chose to look, a log misleads one who did not.

---

## Lane 16 -- `still_pan`, closed 2026-08-11

Lane 15's two answers, taken on this lane's own evidence. Nothing new bit; two
things are worth the next lane's attention.

**What paid off (1): the scope guard fired, which is why lane 15 wrote it.**
Lane 15 pinned `still_pan._require_still is False`. Lane 16 flipped it and that
test went RED -- exactly its job on a SHARED base, where a behaviour change can
otherwise spread by inheritance instead of by decision. It is rewritten as a
LEDGER of who has ruled (`still_word` always, `still_motion` 15, `still_pan` 16,
`still_flat` not yet, base False) so the next lane reads its own status in one
line rather than inferring it.

**Runnable check:** when a packet changes a flag on a shared base's subclass,
the test that pins WHO HAS THE FLAG must list every sibling and name the lane
that ruled. "Assert my lane is True" tells the next lane nothing.

**What paid off (2): a byte-identical smoke proved a claim I would otherwise
have asserted.** The G2 justification says `still_pan` and `still_motion` share
one ffmpeg builder, so neither has a native canvas. Their two smokes -- distinct
files, same still, same canvas, same frame count -- have the SAME sha256
(`3692f155...`). That is the shared builder demonstrated rather than described,
and it cost nothing because both renders already existed.

**Runnable check:** when two lanes are claimed to share a render path, hash
their artifacts. Identical inputs through a genuinely shared deterministic
builder produce identical bytes; if they do not, the "shared" claim has a
qualifier nobody has written down.

**A cleanup rule this lane started, for lane 17 to finish:** the dark-floor
test's parameter list is now the LIVE SCOPE of that branch. If lane 17 gives
`still_flat` the refusal too, that list empties -- and an empty list with the
`else:` branch still in `render_clip` means DEAD CODE. Delete the branch then,
rather than leaving a floor nothing can reach.

> **REVISED BY LANE 17 (2026-08-11), and the revision is the lesson.** That
> instruction was wrong. When the list emptied, the branch turned out to be a
> CONTROL WITH NO OCCUPANT, not dead code: it serves `uses_still = False`, a
> capability this shelf's own attribute comment documents ("False families
> always synthesize a procedural floor"), which simply has no registered family
> today. Lane 4 already ruled on that shape -- rewrite with the reason, never
> delete, because the invariant outlives every occupant -- and deleting would
> have stranded `_lavfi_source` and `ffmpeg_lavfi_floor_cmd` too.
>
> **The distinction to carry:** code no caller can reach is dead; code no
> caller *currently* reaches, for a documented capability, is unoccupied. Test
> the difference rather than guessing it -- lane 17 asserts BOTH that nothing
> reaches the branch today and that it still works when something does.

**What bit (3): a shared base has inheritors you did not picture.** Making the
scope guard GENERIC over the registry (rather than four names) immediately
failed -- because **`mesh_stage` also extends `_CheapFamilyBase`**. It takes the
frame contract and the canvas/still helpers but OVERRIDES `render_clip`
entirely, so `_require_still` is INERT there; it refuses a missing still with
its own `FileNotFoundError`. Nothing was broken, and the four-name version of
the guard would have gone on passing forever without ever mentioning the fifth
inheritor.

**Runnable check:** write shelf-scope guards as `set(registry) - set(ruled)`,
never as a list of names you can picture. The difference is the inheritor you
forgot -- and when one turns up, record whether the shared field APPLIES to it
(here it does not) and what does the job instead, because "the flag does not
apply here" is only safe paired with what does.

---

## Lane 15 -- `still_motion`, closed 2026-08-11

The first still lane and the first packet in this run to change BEHAVIOUR rather
than declarations. Both new items are above (L21) or already known; three things
for the next lane.

**What bit (1): the assigned defect had two halves with different blast radii.**
S8b-12 is one spec item but the ffmpeg preflight gate lives on the SHARED base
(so it sweeps all four still lanes, per L13) while the missing-still refusal is
per-family. Fixing both "as one item" would have widened a behaviour change to
three lanes that never asked for it.

**Runnable check:** when a spec item names a defect on "the four X lanes", ask
for EACH half whether it lives on the shared base or the subclass. The answer
decides whether the fix is a sweep or a one-liner, and they can differ inside a
single numbered item.

**What bit (2): a proxy test again, and the fixture fix again.** Adding
`_require_still` turned `test_each_family_renders_silent_clip[still_motion]`
red -- it called `render_clip` with no `asset_refs`, so it had quietly become an
assertion that this family renders from nothing, which was never its subject
(the clip CONTRACT was). Fixed by handing every still-consuming family a real
still. This is the third time in six lanes that a gate exposed a proxy test;
lane 8 wrote the rule and it keeps paying.

**What bit (3): a green gate row can certify the wrong thing.** G7.4 is GREEN
for `still_plan` -- declared and audit-clean -- while `still_plan` is read by
NOTHING in production (S8b-15, re-verified at `eb3f8412`). The row is honest
about what it checks and misleading about what a reader assumes, so the receipt
says in words that a green G7 does not mean the plan is wired. It was NOT wired
by this lane: giving it a consumer is a design change across every adapter that
declares one.

**Runnable check:** for every gate a lane leaves green, ask what a reader would
ASSUME it proves, and write down the gap where the assumption is wider than the
assertion.

**What did NOT bite:** G3 was already green -- inherited free from lane 10's
`_CheapFamilyBase` fix. The ledger's whole argument, collected five lanes later.

---

## Lane 14 -- `viz_mxc_mandala`, closed 2026-08-11 -- AND THE FAMILY IS CLOSED

The last visualizer. **All four closed the same way: profile canvas channel
INERT, continuity declared per lane.** Four separate one-line contract fixes in
four modules, because this family shares no base -- lane 10's `_CheapFamilyBase`
fix reached the still shelf instead.

**What bit (1): the lane that LOOKED stateful and was not.** This is the only
visualizer that reuses a drawing context across frames -- one
`cairo.ImageSurface` + `Context` allocated before the loop. That reads exactly
like carried state, and carried state is what `CONTINUITY_NONE` would be lying
about. It is not: `paint_mandala` repaints the full field every frame from that
frame's own audio analysis, and nothing reads a predecessor frame's PIXELS. The
reuse is an allocation optimisation.

**Runnable check:** when a render loop hoists an object out of the frame loop,
ask whether the loop READS it or only WRITES it. A reused buffer that is fully
overwritten each iteration carries no continuity; one that is read before being
written does, and then `CONTINUITY_NONE` would be a false declaration.

**What bit (2): the assigned defect was already fixed, and saying so is the
work.** The corpus assigns this lane "the pycairo half of S8b-16 (a NAMED
dependency refusal)". It was already in `assert_usable` AND `load()`, with
separate messages for cairo and ffmpeg, and already covered by a test that
forces the ImportError via `monkeypatch.setitem(sys.modules, "cairo", None)` so
it runs even where pycairo IS installed. The lane's job there was to VERIFY and
record it as already-green -- not to rebuild it, and not to claim it.

**Runnable check:** before implementing a defect a corpus assigns you, run its
own acceptance check first. A spec written weeks earlier may describe a hole
someone has since filled, and re-implementing it is how a second, divergent
copy of a guard gets born.

**What did NOT bite:** L19 held again even here -- an `ImageSurface` is whatever
size you ask for, so a graphics library imposes no canvas. That was the last
plausible candidate for a procedural lane with a native canvas, which is why the
family's four-for-four record is now a useful default: **if a future procedural
lane wants to declare a canvas, that is the anomaly and it owes the L19 argument
explicitly.**

---

## Lane 13 -- `viz_mxc_cpu`, closed 2026-08-11

**Nothing new bit.** Third visualizer, same two answers, premise re-derived
against this engine's own painter before either was reused (`ring_geom(w, h)`,
the scanline/vignette/font tables and the encoder are all request-derived).

**One thing this lane adds to L19, and it is a sharper version of the rule:**
the argument against declaring is strongest exactly where the temptation is
weakest to notice. This tier's stated purpose is running "on ANY box (AMD / Mac
/ Intel), no GPU, no shaders", and seven profiles select it including
`otr_amd16_rocm`, `otr_amd8_rocm` and `otr_mac_mps`. Pinning a canvas on the
lane that exists to be PORTABLE would have been the single worst place to do
it -- and the diff would have looked identical to the other two.

**Runnable check:** before declaring anything on a lane, read its module
docstring for what the lane is FOR. A declaration that contradicts the lane's
stated purpose is a design error even when every gate goes green.

**What the QA pass corrected, and it is a naming trap worth inheriting:**
the first draft of this receipt said the "declares NOTHING" differential control
"stays on this lane". It was never on this lane.
`test_ltx_8gb_canonical_canvas.py` holds TWO things that read alike:
`test_a_SIBLING_lane_still_takes_the_landscape_default` is the real control
(it drives `build_request_from_shot`) and is pinned to **`still_pan` alone**;
`test_engines_that_declare_NOTHING_are_left_alone` is a weaker list assertion
(`declared_render_canvas(x) is None`) that does include `viz_mxc_cpu`. Nothing
had to move because nothing was there -- not because it was already right.

**Runnable check:** when a receipt claims a test "did not need to change",
open the test and confirm WHICH assertion it is. Two tests in one file, both
about lanes that declare nothing, are one careless sentence apart -- and the
sentence would have told lane 16 the control had already been handed over.

---

## Lane 12 -- `viz_camera`, closed 2026-08-11

**Nothing new bit, and that is the entry.** Both red gates were the two lane 11
had just solved, both answers transferred unchanged, and the lane cost one
import, one keyword, two gate-table entries and three tests. A lane that sails
through is evidence the ledger is working and is recorded as such.

The one thing worth stating, because it is the difference between reuse and
cargo-culting: **L19's transfer was EARNED, not assumed.** L19 says copy the
reasoning, not the shape, so `eng_viz_camera.render_clip` was read to confirm
the premise holds here -- every painter call, the scanline table, the vignette
and the encoder are built from the request's own `w, h`, with no latent grid, no
trained input size and no canvas-dependent constant. It does hold, so the lane
declared no canvas and documented its profile channel INERT, exactly as lane 11
did. Had the premise failed, the same two lines would have been the wrong fix.

**Runnable check for lanes 13 and 14:** before reusing lane 11/12's G2 answer,
grep your engine's render path for any dimension that is NOT derived from the
request -- a hardcoded tile, a fixed table size, a constant the painter enforces.
One hit means your lane may have a native canvas and the L19 reasoning does not
transfer.

**A GPU-time judgment recorded so it is not read as a gap:** lane 11 smoked two
legs to prove the `OTR_VIDEO_LANDSCAPE_CANVAS` lever reaches a declaration-free
lane live. Lane 12 smoked ONE, because that property is about the DRIVER, not
the engine, and it is now pinned CPU-side per lane by a test driving the real
`build_request_from_shot`. Re-rendering to re-prove a driver fact already in
evidence is not diligence, it is duplication.

---

## Lane 10 -- `mesh_stage`, closed 2026-08-11

Four red gates, the most defective lane left. Both new lessons above are this
lane's -- L17 is G5, and G1 turned out to be L1's third instance. Three more
things worth the next lane's attention.

**What bit (1): I CALLED IT AN OUTAGE AND IT WAS NOT -- the operator caught it,
and the mistake is more useful than the finding.** The first version of this
entry said "the lane was DEAD ON THIS BOX and nine months of green tests never
said so". Then the operator said 3D had been working before the refactor, which
is checkable in one command, and he was right:

* `_ckpt_path` is **byte-identical** to `37254f39`, where mesh_stage was
  rendering in June -- nothing regressed.
* Under the soak launcher, `HF_HOME=C:\ComfyUI-Models\huggingface`, so the old
  resolver's second probe is `dirname(HF_HOME) + "checkpoints"` =
  `C:\ComfyUI-Models\checkpoints` -- **exactly where the weight is.** It
  resolved every time.

What my probe measured was a BARE SHELL, where `HF_HOME` is unset because the
LAUNCHER sets it, not the User env. That is still a genuine defect -- it is Bug
Bible **12.88**: "where would the LOADER find this?" and "is this weight on this
box?" were sharing one probe, so every off-runtime caller (CPU suite, preflight
matrix, any doctor tool) got a confident wrong NO, and a weight registered only
through `extra_model_paths.yaml` was invisible in-process too. The fix stands
unchanged. The SEVERITY claim was invented by the environment I happened to test
in.

**Runnable check, corrected -- and this is the version that would have caught
me:** resolving with NO env pins tells you about the OFF-RUNTIME question, and
that is the only question it answers. Before calling a lane BROKEN, resolve it
the way PRODUCTION resolves it -- read the launcher and reproduce its env
(`HF_HOME`, `*_DIR`, hydrated User vars). Two probes, two questions:

    bare shell   -> "can anything off-runtime answer honestly?"   (12.88)
    launcher env -> "is the lane actually broken for the operator?" (severity)

Asking only the first and reporting the second is how a resolver defect gets
written up as an outage. **And when the operator says "that used to work",
diff it before arguing** -- `git log --follow` on the one file settled this in
about a minute, and the receipt had already claimed otherwise in writing.

**What bit (2): the differential control moved for the FOURTH time, and the
handover is now the interesting part.** `mesh_stage` held the "declares
NOTHING" canvas control in `test_ltx_8gb_canonical_canvas.py`; it passes to
`still_pan` (lane 16). Four handovers in ten days, each because the occupant
gained a declaration of its own. The list is down to two lanes.

**Runnable check:** unchanged from lane 1 -- before declaring a canvas, grep the
test tree for the lane's id used as a NEGATIVE control. What lane 10 adds: when
the control list is nearly empty, say in the test WHERE it goes next, because
the last occupant leaving is what turns "move the control" into "delete the
test", and the invariant outlives every occupant.

**What bit (3): a shared-mechanism fix flips OTHER lanes green, and the ledger
insists you say so in the same commit.** Adding `continuity=` to
`_CheapFamilyBase` flipped the four still lanes' G3 rows, and the strict
unexpected-pass gate FAILED until their `EXPECTED_RED` entries were deleted --
which is the gate working exactly as designed. The discipline that matters is
the other half: the four visualizers and `google_omni_video` have the identical
defect through their OWN contracts, were NOT reached by the base fix, and stay
red. A sweep that reports lanes it did not touch is worse than no sweep, because
the next window reads the ledger and believes it.

**What did NOT bite:** the boot lane (this lane needs no token -- the registry
IS the menu), the still plan, the public surface, the admission honesty row, and
the atomic-publish/stale-tmp machinery were all green before the packet started
and stayed green. The E-1..E-4 chain -- cache key, mesher, VRAM barrier, cube
self-test, validate, atomic publish -- ran end to end on the first live attempt.

**An OPEN gap this lane deliberately did not close, so the next reader does not
assume it did:** `_directory_clip` returns no `vram_peak_mb` / `recipe` /
`quant` / `render_canvas`, so the smoke report carries a null peak and this lane
has NO measured VRAM number. G4 is green because the manifest records it as
admission-unenforced in words, which is honest. But a peak here needs a probe
threaded through a torch mesher AND a Blender subprocess -- two different
measurement surfaces (L7) -- so it is a measurement design question, not a
passthrough like lane 9's. **Nothing in lane 10's receipt may seed a cost row.**
