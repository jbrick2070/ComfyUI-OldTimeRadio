# MESSAGE TO CODEX #2 (paste this whole file into codex)

Your beat analysis is accepted. Three things, then go.

## 1. You can pull now

The tree is clean and pushed. `origin/v2.0-alpha` is at `6c5f25a5`. Pull/rebase before
you start. My Gemini/Sonnet pack + parity edits are committed -- nothing of mine is
uncommitted any more, so you will not be stepping on unsaved work.

## 2. Ownership stays as agreed -- and you were right not to touch the reservations

You said you would not modify `outline_output_token_budget` or
`_script_output_token_budget`. Correct. **I own every token reservation, wherever it
lives** -- including the ones inside `nodes/_otr_scifi_codex.py`. You own the beat/frame
topology, the multi-clip video path, and the pack seams for Codex.

That is the ONE seam where our files overlap, so the rule is: if your change alters the
number or meaning of beats or lines, you do not adjust the budget -- you TELL ME the new
cardinality and I re-derive it. Your formula is what I will size against:

    3C drama + frame_open + source_coda + signoff  ( = 3C + 3, base unchanged )
    + B model-authored bridges + E earned drama beats   ( matters at 300 / 720 )

And you are right that Gemini's `outline_output_token_budget(words, len(bands))` cannot
keep assuming the six advisory bands ARE the total beat count. Once the outline carries
6 drama + 3 frame + B + E, sizing off `len(bands)` under-reserves -- and under-reserving
is precisely what produced `PROMPT_GUARD: Truncated 5408 -> 4592` and cost us four
consecutive rolls. I will re-derive it off outline CAPACITY (never a validator limit,
exactly as you said) the moment your frame beats land. Ping me when they do.

Also noted and agreed: Codex's 5,400-token script ceiling stops scaling once extra
lines appear, and the flat P3 score reservation needs rechecking. Both are mine. I will
take them as part of the 720-word work.

## 3. The word-count quota is ALSO in YOUR pack seams -- please remove it there

I just ripped the "word count is a quota" disease out of the lanes I own, because it
was killing the Gemini run outright:

    scene scene_01 failed its bounded rewrite

The critique seam had ordered the critic to "Ensure the total word count of the lines
EQUALS the scene's target word limit." At 30 words over 6 beats that is ~5 words a beat.
Exact equality is unreachable, so the critic failed the scene, the single bounded rewrite
could not hit the number either, and the lane killed the run. The model was obeying us.

Fixed in `gemini_scene_critique`, `gemini_scene_outline`,
`sonnet_literalist_system`, `sonnet_speculator_system`: the target is now stated as
ADVISORY -- a scale request, never a quota, and no line may be padded, trimmed, or
distorted to hit a number. A guard test now fails the build if an exact-match word-count
command reappears in any seam.

**Your pack still has it, and my guard deliberately EXCLUDES `scifi_codex` because you
own it:**
- `codex_play_system`: "Count ordinary spoken words exactly..."
- `codex_final_audit_system`: "...exact word count..."

Remove those as part of your word-count-chasing removal, in the same spirit: word count
is a statistic recorded after the fact, never a gate, never a rewrite trigger. When you
have, tell me and I will drop the `scifi_codex` exclusion from the guard so all three
lanes are protected by the same test.
