# Question -- 2026-05-10

TECHNICAL CODE REVIEW (NOT architectural). Just shipped commit fd32a6a
to v2.0-alpha branch of https://github.com/jbrick2070/ComfyUI-OldTimeRadio
adding the cast contract foundation. 30/30 unit tests pass and the
Bug Bible regression still holds at 23/1/2/0. Architecture is settled;
I want a Python-craft code review focused on the NEW module's
implementation quality, edge cases, and idiom.

Files added in this commit (visible at
https://github.com/jbrick2070/ComfyUI-OldTimeRadio/tree/v2.0-alpha):

  config/__init__.py                  (package marker, 5 lines)
  config/cast_pools.py                (~190 lines: name pools,
                                       voice profiles, helpers)
  nodes/_otr_casting.py               (~430 lines: schema, validator,
                                       per-character LLM caller,
                                       cast assembler, top-level
                                       lock_cast orchestrator)
  tests/test_otr_casting.py           (~440 lines: 30 unit tests)

WHAT I WANT YOU TO REVIEW (technical / line-level only, not architecture):

1. nodes/_otr_casting.py:cast_one_character()
   The 3-attempt reroll loop with attempt-3 as repair-prompt. Is the
   loop control correct? The attempt index check
   (attempt_idx == max_attempts - 1 and last_raw) -- does this fire
   the repair branch only on the LAST attempt, only when there's a
   prior raw response to repair against? What happens if max_attempts
   is set to 1 by a caller? Should I reject max_attempts < 2?

2. nodes/_otr_casting.py:lock_cast()
   The voice-pool-exhaustion check happens BEFORE each per-slot LLM
   call. Is there an off-by-one hiding here? The pool starts at 9 (or
   8 if LEMMY hits) and num_characters caps at 6, so on paper we can
   never exhaust. But if num_characters=6 + LEMMY hits, we need 5
   open slots from 8 voices -- should still fit. Is my safety check
   in the right place, or should it also gate at the entry point of
   lock_cast() to fail-fast?

3. nodes/_otr_casting.py:_extract_json_block()
   Copied verbatim from _otr_outline.py. Is there a known failure
   mode where this returns garbage that json.loads() then chokes on
   with a confusing error? Would adding a "reject if first/last brace
   balance is wrong" check be worth the complexity, or is the
   try/except in the caller good enough?

4. nodes/_otr_casting.py:_format_prior_entry()
   Trims the description to 60 chars when building "Cast so far:".
   Two questions:
   (a) Does ".rstrip(',.; ')" before appending "..." cover all
       reasonable trailing punctuation? Should I also strip "!?"?
   (b) The whole prior_cast list grows linearly per call -- on a
       6-character episode with LEMMY, the 5th open-slot call passes
       4 prior entries (LEMMY + 3 already-cast). At 60 chars each
       that's roughly 240 tokens of prior context. Plus the rest of
       the prompt (~150 tokens). Total ~390 tokens for the LAST call,
       under the 400-token hard ceiling but close. Worth pre-truncating
       the prior_cast to last-N entries if cast size grows further?
       (Today num_characters caps at 6 so this is theoretical.)

5. config/cast_pools.py:pick_first_last()
   The 50-retry loop for collision avoidance. With ~110 first names
   and ~50 last names = ~5500 unique full names, collisions are
   astronomically rare. Is 50 retries overkill? Or fine? Any reason
   to log when the retry budget is exceeded (since the fallback
   accepts the collision silently)?

6. config/cast_pools.py:open_voice_pool()
   The "vocal" tag filter selects only adjectival quality tags and
   drops role-shaped tags like "officer", "pilot", "android". Reasoning
   in code: role tags would bias selection without helping. Sound? Or
   am I throwing away signal the LLM could use to differentiate
   characters?

7. nodes/_otr_casting.py: imports
   The try/except for relative-vs-absolute import of config/cast_pools
   handles both ComfyUI runtime (relative works) and pytest test
   harness (absolute works after sys.path.insert). Is this the cleanest
   way? I've seen suggestions to use importlib.util.spec_from_file_location
   but that adds complexity. The current approach silently catches both
   ImportError and ValueError -- ValueError is for "attempted relative
   import beyond top-level package". Anything more specific I should
   catch?

8. tests/test_otr_casting.py:test_assemble_pre_locked_rows_announcer_5050_balance
   200 trials, 80-120 male tolerance. Is this band correct? With 4
   announcer presets (2 male + 2 female) picked via random.choice, the
   theoretical mean is exactly 100/100, std dev sqrt(200 * 0.5 * 0.5) =
   ~7.1. So 80-120 is roughly +/- 2.8 sigma. Want me to tighten or loosen?

9. CastingResponse pydantic schema
   character_description bounded to 10-200 chars. Is that range
   reasonable for a 1-line character brief that flows into HuMo/FLUX
   prompts? Anything I should add (e.g. forbid leading "A " or
   "The " articles, since they bloat downstream prompts)?

10. CastingFailedError
    Mirrors OutlineFailedError shape. Stores attempts as a list of
    (raw_response, error_message). Should I also stash the
    available_voices list and prior_cast at the time of failure so
    debug logs can reproduce? Or is the raw_response sufficient?

11. ANYTHING ELSE technical you'd flag in this implementation. The
    most important question.

Architecture is settled per a prior round-robin synthesis (control-plane
vs prose-plane routing, lean-prompts strategy, no ModelAdapter
pre-build). Please don't revisit those. Focus only on:
- Python idioms that could be cleaner
- Edge cases I've missed
- Off-by-one or loop-control bugs
- Test gaps that would catch real failures
- Anything that would bite us when this gets wired into the writer
  in the next commit.

Brevity preferred over completeness. If you spot 3 real issues that
matter and 5 nits, give me the 3. Skip the nits unless you think they
add up to something.
