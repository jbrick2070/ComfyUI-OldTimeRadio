# VIDEO_LANE_PREFLIGHT receipt -- lane 4, `humo14_high_audio_in_portrait` (`humo`)

`VIDEO_LANE_PREFLIGHT receipt: humo | 2026-08-11 | smoke receipt
output/otr/episodes/_lane_smokes/lane04_humo14_portrait/ | verdict PASS`

The last HuMo tier, and the 2026-06-09 keystone. **The HuMo family is now
closed: four tiers, four canvas declarations, four boot-contract lists.**

## Matrix row

7/7 GREEN. Its `EXPECTED_RED` G2 entry deleted -- the last one the family held.

## What it needed, and what came free

Only two things were this lane's own work: the canvas declaration and the
public id. The weight resolver, the boot-contract mechanism, the manifest
fields, the unconditional exact-fit guard and the override refusal all arrived
in lanes 2 and 3 and applied unchanged. That is the ledger's whole argument.

**BOTH its profiles were claiming landscape.** `otr_w45_humo.json` said
832x480, and so did `otr_g4_humo.json` -- on the tier whose entire job is the
pillarbox talking head. The w45 one is the one I would have thought to check;
the g4 one only surfaced because G2.3 reads EVERY profile that selects the
engine, not just the one being edited. That is the gate earning its keep: a
by-hand fix would have left half the lie in place.

## G8.1 solo smoke

| Item | Value |
|---|---|
| Boot contract | `humo_diet`, flags confirmed on the command line |
| Harness | `_otr_single_engine_smoke.py --engine humo --frames 97` |
| Prompt id | `5e684ea9-f891-4035-9154-0f9b1f6deb79` |
| Wall time | 271.3 s |
| Canvas PROBED | **480x832** -- portrait, equals the declaration |
| Frames PROBED | **97** counted, duration 3.880 s = 97/25 exactly |
| Rate | 25/1 |
| Audio | **zero audio streams** |
| Trim | none |
| Peak | **13,800 MB absolute, COLD** (net of the ~1,890 MB idle baseline, roughly 11.6 GiB) |
| Artifact | `.../lane04_humo14_portrait/humo14_480x832_f97_diet_smoke.mp4` |
| sha256 | `5ac6df8219b57017f04ed3d364deabb45b7521c2aa41aacb2b1568ef974aaf81` |

The four HuMo cold absolute peaks now sit in a readable order:

| Tier | Rung | Canvas | Peak (absolute, cold) |
|---|---:|---|---:|
| `humo` (14B portrait) | f97 | 480x832 | 13,800 MB |
| `humo_14B_169` (14B wide) | f97 | 832x480 | 14,604 MB |
| `humo_1.7B` (1.7B portrait) | f129 | 480x832 | 15,261 MB |

The 1.7B is the most expensive of the three because it renders a third more
frames, not because it is a heavier model. Frame count dominates on this
family, which is the single most useful thing these three legs say to lane 5's
admission work.

## Controls that moved again

`humo` held the "declares NOTHING" differential control in TWO test files. With
the family closed there is no HuMo tier left to hold it, so
`test_boot_contracts.py` stopped parking the invariant on whichever tier had
not been done yet and asserts the SCOPING RULE directly -- strip the
declaration, the overrides go back to winning. A control with no occupant left
gets rewritten, not deleted. `test_ltx_8gb_canonical_canvas.py`'s list simply
lost its third occupant; `mesh_stage`, `ltx_audio_in`, `still_pan` and
`viz_mxc_cpu` remain, and each will leave when its own packet runs.
