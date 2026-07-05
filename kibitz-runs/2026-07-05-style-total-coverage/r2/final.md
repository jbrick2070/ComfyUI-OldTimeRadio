# r2 JUDGMENT (Cowork Claude, anchor + judge) -- style total-coverage, coding round

## Accepted
- CODEX M1: chunk-B field count corrected (9 str + 2 dict) with the exact key list.
- CODEX M2 + AG M2: `announcer_subject_ltx_mouth` is a TEMPLATE (the current
  _RADIO_CONSOLE_MOUTH carries %s form -- :203/:335); v2 packs use `{form}` exactly once
  and the call site switches to .format(form=...) (one formatting convention across all
  template fields).
- CODEX M3: `non_character_emblem_fallback` is a TEMPLATE requiring `{base}` exactly once
  (dynamic intent/setting context, :1226).
- CODEX M4: chunk C concretized -- `still_word_typography: dict` + `still_word_backdrop:
  dict` (exact keys {noir, sci-fi, western, pulp, default}) + `still_word_title_mood_style:
  str` (:631/:642/:662).
- CODEX M5 + AG S1: provenance = ADDITIVE keys `visual_style` + `prompt_field_source` on
  the request observability dict, ADDED to the trace-copy allowlist (:2033-2035);
  acceptance reworded: prompt text + existing sha/chars byte-identical, new keys additive.
- CODEX M6 + AG S2: the "photographic and period-consistent" text lives ONLY in
  `_build_char_prompt_request` (:1068); `_build_char_scene_request` gets the style-look
  insertion specified separately; both builders take the resolved style.
- CODEX S1 + AG M3: final signature `get_open_subject(role, synthetic, meta=None,
  style=None)`; compose_still_prompt passes its already-resolved _style (:510->:522) --
  helpers never re-resolve.
- CODEX S2: forbidden-terms load lint EXTENDED over all new string leaves + dict values.
- CODEX S3: with exact-key packs, `motion_registers[_motion_key]` indexes directly and
  raises on a missing console key; the silent `or ...["announcer"]` fallback (:1656) is
  RETIRED (non-console roles keep the `""` no-op arm).
- CODEX OPT: nested dicts stored as immutable mappings in the frozen dataclass.
- CODEX CUT: seam-level string-equality gates first; full-episode byte-identity is the
  operator acceptance, not the build gate.
- AG M1: `_style_anchor_for_aspect(aspect, talking=False, style=None)` + all four callers
  thread the resolved style.
- AG OPT2: test_visual_styles_3b.py schema-version + exact-key pins re-pointed SAME
  COMMIT (v1 -> v2).
- ANCHOR M1: role->key selector + OTR_LTX_OPEN_MOTION_KEY env stay Python; packs own the
  four VALUES; BUG-LOCAL-112 240-char budget enforced at LOAD on pack motion values.
- ANCHOR M2: the two ia2v talking PROMPTS are probe-proven verbatim constants (P8: a
  paraphrase halves articulation) -- never pack fields; the plan carries the citation;
  `announcer_subject_ltx_mouth` load lint requires mouth-prominence vocabulary.
- ANCHOR M3: the :1656 fallback path routes through pack values (superseded by CODEX S3's
  retire-the-fallback -- adopted).
- ANCHOR S4: chunk A split into A1 (image lane) + A2 (video/mesh lane), each
  byte-identical + green. ANCHOR S5 folded into CODEX M5.

## Rejected
- AG OPT1 (v1 custom-pack back-compat defaults): violates the fail-loud/no-fallback law --
  a v1 pack fails load with a clear upgrade message; user packs upgrade or are removed
  (AG's own assumption concurs).

## Verify-at-build
- The exact insertion semantics preserving byte-identity for _build_char_scene_request.
- The trace allowlist location (:2033-2035) + node-92 /history propagation.
- Chunk-C map key "sci-fi" contains a hyphen -- keep the on-disk key EXACTLY.
