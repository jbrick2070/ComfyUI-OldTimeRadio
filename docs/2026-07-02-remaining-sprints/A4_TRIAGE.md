# Sprint A / A4 -- fallback-consumer test triage (grep-verified 2026-07-02)

Grep basis: `make_fallback_of | EXPECTED_OOM_TRAIL | fallback_engine | resolve_fallback_chain |
escalate_to_fallback | build_fallback_decision | restamp_shot_row | append_runtime_fallback_decision |
format_swap_log | allow_auto_fallback` over tracked *.py at HEAD 53a8775d. All PLAN.md A4 sites
confirmed; additional consumers found and triaged below.

## DELETE (file or test is fallback-machinery-only)

- `tests/test_video_fallback_chain_additive.py` -- entire file (unit tests for
  resolve_fallback_chain, which A1 deletes).
- `tests/test_video_render_driver.py:24/33/41` chain-resolution tests + `:84`
  EXPECTED_OOM_TRAIL fixture use -- delete these tests; REST of file survives.
- `tests/test_video_render_driver_additive.py:77-82`
  (test_make_fallback_of_overlay_and_terminus) -- delete; `:415` run_episode call
  KEEP-REWRITE (drop fallback_of kwarg); `:94` local `fallback_engine = None` stub KEEP.
- `tests/test_video_humo.py:216-247` chain-convergence tests -- delete.
- `tests/test_video_still_parallax.py:177-186` "dangling fallback self-heals to floor"
  -- delete; replace w/ LOUD-failure contract.
- `tests/test_video_character_3d.py:360-369` make_fallback_of/FLOOR_NAMES resolution test -- delete.
- `tests/test_video_retry_taxonomy.py:144-229` fallback-action API tests
  (build_fallback_decision / restamp_shot_row / append_runtime_fallback_decision /
  format_swap_log / chain) -- delete.
- `tests/test_video_retry_taxonomy_additive.py:100-148` same API -- delete (NOT in the
  plan's list; found by grep).
- `tests/test_video_survival_guide_vectors.py:63-76` ghost-fallback-reference + chain
  vector tests -- delete; `:103/:128/:134` escalate_to_fallback assertions -- rewrite to
  classification-only asserts (is_hard / retries); REST of file survives.

## KEEP-REWRITE (pins valid non-fallback behavior that must survive)

- `tests/test_cs3_inter_beat_reclaim.py:55/65/74/84` -- reclaim behavior is the point;
  drop `fallback_of=rd.make_fallback_of()` kwargs once run_episode loses the parameter.
- `tests/test_ltx_av_driver_wiring.py:25-28` -- wiring is the point; drop the fb use.
- `tests/test_video_soak_fixture.py:82/128` -- rewrite to the NO-TRAIL LOUD contract:
  a forced OOM RAISES a named RenderError and the soak asserts the raise (no trail match).
- `tests/test_video_humo.py:53/:57` -- rewrite to `fallback_engine is None` (both tiers).
- `tests/test_video_mesh_stage.py:68/320` -- `:68` rewrite to None; `:320` drop fb use.
- `tests/test_video_still_parallax.py:57` -- rewrite to None.
- `tests/test_video_triposr.py:35` -- rewrite to None (NOT in the plan's list; found by grep).
- `tests/test_video_character_3d.py:354-356` -- rewrite to None x3 + named RenderError
  on OOM (never chain-to-humo).
- `tests/test_video_retry_taxonomy.py` classification tests (:67/:76/:82/:88/:96) --
  rewrite escalate_to_fallback asserts to classification-only once the field is deleted.
- `tests/test_video_retry_taxonomy_additive.py:64/:67` -- same rewrite.
- `scripts/otr_video_gpu_smoke.py:36/92-107/183` -- rewrite: no chain walk, no
  decision/swap-log; failure = LOUD raise.
- A3 pinned-True sites: `test_route_a_14b_promotion.py:132`,
  `test_still_aspect_and_labels.py:208`, `test_video_platform_aseam.py:316/:341`
  (+ `:260` schema list) -- flip with A3a/c/d/e.
- `scripts/run_otr_30word_smoke.py:212-216` False-patch -- becomes a harmless no-op; KEEP.

## SURVIVES UNCHANGED (already pins the None contract)

- `tests/test_ltx_audio_in_engine.py:97`, `tests/test_video_visualizer.py:40`,
  `tests/test_video_viz_mandala.py:41`, `tests/test_video_viz_rainbow.py:35`.

## NEW (per-family LOUD-failure contract tests, added in A1+A2)

One per adapter family (humo x2, ltx_video, mesh_stage, triposr, still_parallax,
character_3d x3): render failure raises a NAMED RenderError; no engine swap, no restamp,
audio untouched.

## Non-test rip surface confirmed by grep (A1)

- `nodes/_otr_video_engines/render_driver.py` :17 docstring, :35 import, :52/:56/:63
  FLOOR/SYNTH tables, :64 comment, :117 EXPECTED_OOM_TRAIL, :153 make_fallback_of,
  :165 chain getattr, :1035-1037/:1287-1289/:1880-1882 stale comments, :2003
  append_runtime_fallback_decision consumer, :2101 run_episode default, :2527/:2529/:2589
  soak verifier + fixture, :2654-2657 __all__.
- `scripts/otr_video_soak.py` :40 import, :57 map comment, :150 make_fallback_of,
  :165 getattr, :206 chain, :219-227 decision/restamp/swap-log, :245, :259/:294-296 trail.
- `nodes/_otr_shared/fallback.py` -- DELETE whole module.
- `nodes/_otr_shared/retry_taxonomy.py` -- delete the action API (:136 field, :251/:282/:300/:323
  fns, :361-364 __all__) + docstring :8; KEEP classification.
- Adapters (A2): eng_humo.py:130/:514, eng_ltx_video.py:340, eng_mesh_stage.py:329 (+:328
  comment), eng_triposr.py:121 (+:151 msg), eng_still_parallax.py:187,
  eng_character_3d.py:258/:327/:398 (+:53/:292/:365/:436 docs/msgs), eng_humo.py:19 docstring,
  cheap_families.py:172/:182 comments.
- `nodes/otr_video_director.py:241/:295/:345` + `nodes/_otr_video_engines/schemas.py:130/:331` (A3).
