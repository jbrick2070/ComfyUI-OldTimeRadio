# VIDEO_LANE_PREFLIGHT receipt -- lane 6, `wan22_high_fast` (`fastwan_8gb`)

`VIDEO_LANE_PREFLIGHT receipt: fastwan_8gb | 2026-08-11 | suite 9920 passed |
smoke receipt output/otr/episodes/_lane_smokes/lane06_wan22_high_fast/ |
verdict PASS`

The healthiest lane in the whole audit, and it stayed that way: 7/7 green before
this packet started and 7/7 after. Its canvas was declared and tested, its
ceiling pinned at a measured rung, LoRA absence already failed preflight closed,
its seed trap already pinned. Nothing was broken here, so nothing was fixed --
this lane's work was its public surface and a live proof.

## The naming MOVE, with a wrinkle worth recording

`fastwan_8gb` was an **identity row**: its public id WAS its internal id. So
unlike `wan_8gb` it needed no entry in `_LEGACY_ENGINE_ALIASES` on the way out
-- a bare internal id already passes through `resolve_engine_id` step 3. Adding
one would have been harmless but misleading, implying a rename that never
happened at the internal level.

Verified live: `fastwan_8gb`, `fastwan_8gb (16:9)` and `wan22_high_fast` all
resolve to `fastwan_8gb`; the old token appears in NO menu option; the menu is
still 27 rows.

## G8.1 solo smoke -- and the throughput claim, proved

Same boot (`default`), same still, same canvas, same rung as lane 5's
`wan_ti2v` smoke, deliberately -- so the two are directly comparable.

| | `wan22_high_video` | `wan22_high_fast` |
|---|---:|---:|
| Frames | 81 | 81 |
| Canvas | 832x480 | 832x480 |
| Wall time | 171.2 s | **70.5 s** |

**2.43x sooner for the same frames at the same canvas.** The label sells
throughput and only throughput -- "the SAME motion at the SAME canvas for the
SAME VRAM as wan22_high_video, ~2.7x sooner" -- and a test forbids the words
"better", "hq" and "high quality" appearing in it. The measured 2.43x here is a
single cold pair, not a benchmark; the label's ~2.7x comes from the lab.

| Item | Value |
|---|---|
| Prompt id | `d168c034-f596-4c72-85eb-e64f6102129a` |
| Canvas PROBED | **832x480** |
| Frames PROBED | **81** counted, duration 3.240 s = 81/25 exactly |
| Rate | 25/1 |
| Audio | **zero audio streams** |
| Trim | none |
| Artifact | `.../lane06_wan22_high_fast/fastwan_832x480_f81_default_smoke.mp4` |
| sha256 | `5a35436d35316e90d6c6b84390e476722da31051c0225ac2f71910a646219673` |

## Deliberately unchanged

The injected cost row (`FRAME_COST_MODEL["fastwan_8gb"] = (7000.0, 185.0)`) and
motion floor stay exactly as they are. S2 says recalibrate this row in the SAME
commit as `wan_ti2v` or it goes silently stale -- and standing default Q3 says
`wan_ti2v` ships DISQUALIFIED rather than qualified from lab numbers. So both
rows stay put together, which is the consistent answer. The manifest still says
"admission NOT enforced" for this lane, in words.

Inherited retention behaviour belongs to row 5b's measurement campaign, not
here.

## An unrelated flake, recorded so it is not mistaken for a regression

`tests/test_feed_fetch_seam.py::TestBoundedRequest::test_a_redirect_into_the_private_network_is_refused`
failed once during this lane's suite run and passes in isolation. It coincided
with the Wi-Fi DNS drop that also killed a `git push` -- a network-dependent
test caught in a network outage, not a code defect.
