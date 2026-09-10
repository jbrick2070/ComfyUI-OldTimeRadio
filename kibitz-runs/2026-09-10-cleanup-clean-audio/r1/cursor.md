VERDICT: yes-with-fixes. Node 3→4→7 really is the production coloring route, but the operator sentence ("no random DSP/tape hiss") and C4's widget list are different products, and P1 tells the implementer to land an unnamed C1 in the same audible chunk.

MUST-FIX BEFORE BUILD:
1. [C4 lead + C4.3 + P1 enhance equality] Goal and method diverge. Operator ask is random hiss/roomtone gone. C4 then defines "Clean" as also killing Haas, mid-side width, bass warmth, the 16 kHz LPF, and canonical tape (sat+wow), via canonical widgets `[48000, 0.3, 0.8, 0.15, 16000, "subtle"]` → `[48000, 0.0, 0.0, 0.0, 0.0, "off"]` (`workflows/otr_canonical.json:754-760`). Those last five are not RNG. LPF is the worst: `audio_enhance.py:345-347,433-437` exists to "kill Bark chirp artifacts"; P1 itself admits turning it off "can expose artifacts already in upstream voices." Fix (smallest): pick one sentence and make the widget list match it. Narrow = delete `_generate_room_tone` (`scene_sequencer.py:634-730,1182-1198`) + tape hiss (`audio_enhance.py:258-260,276-286`) only; leave canonical LPF/spatial/tape-sat. Wide = keep the zeroed widgets, but rewrite "Clean" to say the default route is dry and Bark chirps are accepted. Do not ship the wide edit under the narrow slogan.
2. [header / P1 "C4+C1"] C1 is not specified in this document (not in `docs/DEAD_CODE_EXECUTION_PLAN.md` or `docs/OTR_STANDING_RULINGS.md` either). "Include C1 stores/prose" is an un-reviewable second change riding an audible default-route edit. Fix: paste C1's exact line list here, or drop C1 from this chunk. C4.1 already deletes `current_env` / `env_timeline` / the bed (`scene_sequencer.py:989-990,1137,1182-1198`); do not invent extra stores under that label.
3. [P1 silence→master WAV] The all-zero master claim is only true because `_master_loudness` returns unchanged when `peak < 1e-8` (`scene_sequencer.py:380-381`) and PCM write is `* 32767` (`:1757`). State that as the contract. Also state `_trim_trailing_silence` (`:289-304,882`) will shrink all-zero clips to 100 samples; zeros stay zeros, duration does not. If either guard is treated as accidental, the four-case proof is a landmine, not a spec.

SHOULD-FIX:
1. [P1 four opening/silence combos] Silent+opening and silent+no-opening are both all-zero PCM, so they do not prove the 1.50 s opening placement the chirp path measures (`scripts/otr_canonical_audio_check.py:216-220`; 2.0 s cue − 500 ms crossfade). Keep both only if you assert a sample-count delta ≈ 1.50 s; otherwise one silent case is the additive-noise contract.
2. [C4.3 vs schema] Today three disagree: INPUT_TYPES Haas 0.4 / bass 0.1 / tape `"off"` (`audio_enhance.py:333-357,374-376`), canonical Haas 0.8 / bass 0.15 / tape `"subtle"` (`otr_canonical.json:754-760`). Collapsing them is right only after Must-Fix 1. LPF `min: 8000` with tooltip `0=off` is already a UI lie; lowering min is required if and only if canonical 0.0 stays.
3. [C4.4 / P1 SHA] "Only node 4's six values changed" dies if canonical is json-load/dumped. Surgical edit of `widgets_values`. 23 node ids confirmed (1,3,4,7,12,62,63,80-95). verify: `len(graph["links"])==62` (`links` at `:2984`; `last_link_id` is 290, not a count).
4. [P1 "listening qualification is outside"] This is an audible default-route change. CLAUDE.md treats `otr/obs/` as the success signal. Smallest close: A/B the existing chirp master WAVs (pre vs post). Not a GPU episode; do not call synthetic chirps a listen.
5. [output fans] Node 7 `episode_audio` still fans to video + scopes (`otr_canonical.json:874-877`, links 15/272). A drier mix changes those consumers. Name them out of scope or the chunk is lying about blast radius.

OPTIONAL / NICE-TO-HAVE:
- Tape-on-zeros (`P1` supplemental) is the right hiss probe: `tanh(0)=0` and wow `fill_value=0.0` (`audio_enhance.py:271-305`). A chirp+tape energy check is optional.
- `__init__.py:8-13` and canonical `extra.description` (`otr_canonical.json:3495`) still advertise Haas/spatial/room tone as the product. Capability vs default; tidy in this chunk or the next prose pass.
- `tests/test_segment_loudnorm.py:88-91` name "room_tone" is clip-gate behavior, not `_generate_room_tone`. Leave it.

CUT THESE:
1. Payoff "128/158 runtime lines" (C4 Compatibility/Payoff, P1). Line-count is not the goal; it will drag C2/C3 metrics into a behavior change.
2. Hitchhiking C1 (and any C2/C3 SHA pledge) into this chunk. Safe: C2/C3 are already declared out of scope; C4's JSON delta is node 4 widgets only.
3. Second silent case if Must-Fix 3 + Should-Fix 1 duration check are refused. One all-zero run through 3→4→7 already proves no additive noise.
4. New DSP/offset/resampler work (already banned). Do not grow the check into a second harness; extend `scripts/otr_canonical_audio_check.py` in place as written.

[ASSUMPTION] The superseded roomtone KEEP is not in `docs/OTR_STANDING_RULINGS.md`; treat the brief as the only KEEP override.
[ASSUMPTION] Foley stays a later mux mix (`scene_sequencer.py:1658-1675`); removing the 0.01 bed (`:1188-1197`) dries dialogue under that bed. verify: live foley-route policy still stacks.
[ASSUMPTION] After the two RNG sites die (`scene_sequencer.py:655,678-726`; `audio_enhance.py:278`), 3→4→7 at tape `"off"` is seed-free. verify: no other `np.random`/`torch.randn` on this route, including assembler.
[ASSUMPTION] `scripts/build_variants.py` consumers still exist even though `workflows/` only has `otr_canonical.json`. verify: `--check` after the widget edit; regenerate if variants copy node 4 values.
UNTRACED: `nodes/production_ledger.py` body (gates still written at `audio_enhance.py:476-516`); Comfy execution cache; parent cleanup plan's actual C1 text.

Grounded route (confirmed): clips → node 3 (`sequence` resample/level/concat + current auto bed) → link 5 → node 4 (`enhance`) → link 6 → node 7 (`assemble` + `_master_loudness` + WAV); music enters node 7 on links 282/283 from node 83 (`otr_canonical.json:840-851,3410-3423`), not node 4. `current_env` is never updated (`scene_sequencer.py:989,1137` only), so descriptor textures are already dead; default hiss/hum/crackle at 0.01 still run. `_stereo_decorrelate` docstring `0.0 = mono` is false (`:117-136`); skipping at `spatial_width > 0` (`:450`) matches the plan's tooltip fix. `OTR_AudioEnhance` is a live registered public node (`__init__.py:173`). No Python test imports AudioEnhance; `tests/test_sequencer_ledger.py:66-84` is the one `_generate_room_tone` patch. Independent two-process WAV match without seeds is the right post-RNG proof (`tests/test_canonical_audio_check.py:37-47`); deleting `rng_seed` there is required, not extra.

Uncovered: variant builder, foley mux, ledger gate consumers, C1 source doc.
