# still_word / word_razzle -- BUILD-READY plan (kibitz r2 + roundtable converged)

Hardened by: my code-grounded anchor + kibitz codex (local, file-reading) + a
roundtable frontier pass (Grok + Gemini; GPT failed on reasoning-token empty).
Antigravity dropped (credit bug, hangs silent -- operator ok). Strong convergence
across all sources; every claim below GROUNDED by me against the real repo.

Panel/judge: Cowork Claude anchor + judge. Spend this pass: ~$0.20.
Hard rules: NO fallbacks / dropdown-only defaults; workflow-JSON same-change rule
(BUG-LOCAL-097 positional widgets); UTF-8 no BOM; SFW; single resident heavy
<= 14.5 GB; audio spine byte-identical; suite + Bug Bible + B7 + push per green
chunk; prod/main GATED.

## What still_word IS (operator-confirmed)

A NEW model-agnostic VIDEO engine (a still_flat sibling in
`nodes/_otr_video_engines/cheap_families.py`; flat hold). Selected per-role in the
VIDEO dropdown; the IMAGE model that mints the base still is chosen INDEPENDENTLY
(never coupled into the video options). The delta vs still_flat is the PROMPT the
base still is generated from, branched by role/mode:
- character / announcer (unique beats): prompt GENERATED FROM the beat's script
  line (the spoken words drive the image). Per-beat unique, NEVER pooled.
- music: an ABSTRACT picture of `meta["episode_title"]` (NO words) + Meta brief +
  visual style.
- pooled char beats: DEFERRED (pooling was removed 2026-07-01; not buildable --
  see below). Named-only.

`word_razzle` = the ANIMATED variant. NAME CONSTANT ONLY in v1 -- do NOT register
a non-rendering engine (registry-is-the-menu would make it a selectable dark row).
Full build Phase-gated per docs/GO_FORWARD_NEXT/2026-07-02-ideo-word-razzle-vid.md.

## GROUNDED build sites (exact -- verified against the code)

1. **cheap_families.py** -- new `StillWordFamily` (subclass the still-hold base
   like StillFlatFamily :227): name "still_word", family "static_image_gen",
   flat hold (`_still_motion=False`), consumes the provided base still,
   EMPTY default_roles. It does NOT mint. `@register`.
   MUST fail LOUD if the base still is absent -- NEVER the dark lavfi floor
   (cheap_families.py:129-140 is the floor path to avoid; NO FALLBACKS).
2. **_otr_video_engines/__init__.py** -- import so it self-registers.
3. **registry.py CAPABILITIES["still_word"]** row (cpu/registered) -- the
   consistency invariant (tests/test_capability_profiles.py:215) requires it.
4. **render_driver.py ENGINE_FAMILY** (:51-64) -- add
   `"still_word": "static_image_gen"`.
5. **render_driver.py still-init tuple** (:1044) -- add "still_word" to
   `("still_pan","still_flat","ltx_audio_in")` so the beat's minted scene still is
   routed to its init. (static_image_gen is NOT in `_SCENE_INIT_FAMILIES` -- the
   :1044 explicit branch is the correct site; do NOT touch _SCENE_INIT_FAMILIES or
   _PROFILES -- those were roundtable near-misses, and _PROFILES is defaults-only
   which violates selectable-not-default.)
6. **Composer** (`nodes/otr_meta_brief_image_prompt.py`) -- the per-role video
   engine is ALREADY forwarded in `image_policy_json["video_models"]`
   (otr_image_director.py:380). Add a pure `_still_word_roles_from_policy(
   image_policy_json)` reader (resolve via `role_slots.engine_id_for_role`) exactly
   like `_mesh_fodder_roles_from_policy` / `_talking_roles_from_policy`
   (:1388-1390), pass it into `derive_image_prompts`, and in the object-build loop
   switch the prompt when the beat's role is a still_word role. (Grok's "add a
   video_policy_json input" is REJECTED -- the data is already in the image policy.)
7. **New pure helper** `compose_still_word_prompt(meta, role, beat_line)` (do NOT
   reuse `_compose_char_scene_prompt` -- it calls the writer LLM, breaking PURE +
   deterministic). Word mode: turn the beat line into an image prompt. Music mode:
   abstract-title prompt from meta["episode_title"]. FAIL LOUD (raise) on blank
   line (word) / blank title (music) BEFORE prompt_hash. Tests: quotes, ellipsis,
   em-dash, stage direction, empty; blank-title raise; determinism for fixed seed.
8. **Workflow JSON** -- still_word joins the dynamic VIDEO combo (all_engine_names);
   selectable, no widget-value change (defaults stay). Add a test that "still_word"
   is in the OTR_VideoDirector video combo. If any saved widget value must move,
   do it in the SAME commit (hard rule 0).

## DEFERRED (named, not built in v1)

- **pooled-char mood SET** -- pooling was deleted 2026-07-01 (rip-sfx-broll):
  otr_image_director.py:383-384 "every beat is per-beat now"; schemas.py:112-119;
  otr_shot_lock.py:723-725. The mode targets a pipeline concept that no longer
  exists. Ship word-driven + music-title only; reopen pool-mood if pooling returns.
- **word_razzle** animated path (name constant only).

## Verify-at-build

- Registration in ALL sites (1-5) -- else still_word never receives a still.
- compose_still_word_prompt PURE + deterministic + the edge-case + fail-LOUD tests.
- Model-agnostic: a lightweight UNIT assertion that the composed still_word prompt
  is engine-INDEPENDENT (same prompt regardless of which image engine renders it)
  -- kept (operator emphasized model-agnostic) though a full 2-model GPU render is
  cut as heavy.
- No fallbacks; missing still / blank line / blank title all fail LOUD.
- Audio byte-identical; single resident heavy <= 14.5 GB.
- Suite + Bug Bible + B7 green; push per green chunk; HEAD==origin.

## Judgment log (accepted / rejected)

- ACCEPTED (all sources): render_driver ENGINE_FAMILY + :1044 tuple; pool-mode
  deferred; pure compose_still_word_prompt + fail-LOUD; fail-loud no-floor; cut
  word_razzle stub to a name constant; register in all sites.
- REJECTED (grounded misreads): grok's "add video_policy_json input" (video_models
  already in image_policy_json); grok's _SCENE_INIT_FAMILIES/_PROFILES sites
  (static_image_gen uses the :1044 branch; _PROFILES is defaults-only).
- KEPT over a CUT: the model-agnostic check (as a cheap unit assertion) -- operator
  priority, worth the guard.
- CONVERGENCE: reached. Coding plan is build-ready.
