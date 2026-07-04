# R3-chunk-3 no-fallback rip -- grounded review prompt (agy + Sonnet)

Paste this to **Antigravity (`agy -p`)** and **Sonnet** as a READ-ONLY, code-grounded
review. Repo root: `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`,
branch `v2.0-alpha`. Do NOT edit; report findings only. Ground EVERY claim against the
real files (cite file:line). This is the LAST code chunk of the stack-wide NO-FALLBACK
rip; the operator directive is: **every model failure fails LOUD with a named raise --
never a silent swap or canned template.**

## What changed in chunk-3 (the diff to review)
Three model->template fallbacks were ripped to loud raises. Files touched:

1. `nodes/otr_meta_brief_image_prompt.py`
   - `_compose_char_scene_prompt` (~:797-865): when a writer LLM was ATTEMPTED
     (`llm_fn` present AND the beat has dialogue) and it yields no usable PERSON
     prompt (empty / non-person / all-gear), it now `raise RuntimeError(... no-fallback
     rip ...)`. When `llm_fn is None` OR the beat has no dialogue, it STILL returns the
     deterministic `compose_still_prompt(scene_character)` -- that is the legit local
     template lane, NOT a fallback.
   - `derive_image_prompts` portrait loop (~:1108-1230): three tiers now raise, but
     ONLY when the prompt came from the LLM (`source == "llm"`):
       (a) tier-2 empty (`if not prompt`) -> raise iff `llm_fn is not None`; `llm_fn
           is None` keeps the deterministic template (primary lane).
       (b) consistency gate -> raise iff `source == "llm"` and NOT
           `consistency_gate_warn_only`; warn_only OR template-path keeps + logs.
       (c) person guard -> raise iff `source == "llm"`; template-path keeps + logs.
       (d) gear-scrub emptying an LLM prompt -> raise.
   - Contract docstring updated (was "Never raises; never emits an empty prompt").

2. `nodes/_otr_casting.py`
   - `_apply_llm_slot_fill` (~:1340-1365): the OPT-IN (`name_mode == "llm_slot_fill"`)
     naming overlay now raises `CastValidationLLMError` on BOTH failure paths
     (generate_fn raises; validation `not result.ok`) instead of silently keeping the
     deterministic RNG-pool names. Constructor is `(attempts:list[(str,str)], name)`;
     the "no-fallback rip" tag is in the attempt error string.

3. Tests inverted in the SAME change (pinning the raise, never deleted):
   `tests/test_image_platform_c1.py` (reseed_fallback + consistency_gate),
   `tests/test_brief_prompt_finishing.py` (person_guard),
   `tests/test_still_spine_helpers.py` (added char-scene fail-loud test; kept the
   `llm_fn=None` keep test), `tests/test_cast_llm_naming.py` (both R8 paths).

## Questions to answer (ground each against code)
1. **Swallowed raise (the E4 lesson).** Trace every caller of `derive_image_prompts`,
   `_compose_char_scene_prompt`, and `lock_cast`/`_apply_llm_slot_fill`. Does ANY
   upstream `except Exception` swallow the new raises and silently continue (making the
   rip inert)? Cite the catch site if so.
2. **llm_fn=None carve-out.** Is "no writer LLM configured -> keep the deterministic
   template, do not raise" the CORRECT reading of the operator directive, or should a
   configured-but-absent writer LLM also fail loud? Check `_resolve_writer_llm`
   (otr_meta_brief_image_prompt.py ~:1454) -- it returns None on ANY resolution
   exception (silently degrades a CONFIGURED-but-broken writer LLM to the template
   lane). Is that a hidden fallback that should be in scope, or legit local-lane
   selection? (I left it OUT of chunk-3.)
3. **person-guard judgment.** For a portrait, the person guard now RAISES on a
   non-person LLM prompt (preserving its protection loudly) instead of swapping to the
   appearance template. The consistency gate raises too (default). Is raising right for
   BOTH, or should either "keep the AI output" like the announcer-outro F3 hedge
   precedent (`_otr_line_composer.py` ~:3567-3579)? The R3 test-inversion checklist
   said all three tiers RAISE -- confirm that matches the operator's intent.
4. **OUT-OF-SCOPE sibling (please rule on this).** `nodes/otr_shot_lock.py
   derive_creative_directives` has the SAME class of model->template fallback
   (`template_consistency`, person-anchor template, empty-reseed template) and is
   pinned by `tests/test_look_qa_round5.py` + `tests/test_video_platform_aseam.py`. It
   is NOT in the operator's explicit 9-site R3 set, so I did NOT touch it. Should it be
   ripped as a 10th site (closing the arc), or is it deliberately deferred (e.g. to the
   S3-full ShotLock audit)? This is the main open question.
5. **Happy-path byte-safety.** On a real episode (valid cast, writer LLM produces a
   valid person+consistent prompt), do any of the new raises fire? Confirm the rip is
   dormant on the happy path.
6. **Named-raise discipline.** Every new raise is a named RuntimeError /
   CastValidationLLMError carrying "no-fallback rip 2026-07-03" -- no bare `raise`,
   no invented exception types. Confirm.

Report: CONFIRMED / DISPUTED per item with file:line evidence, plus any surviving
silent fallback in these three files that I missed.
