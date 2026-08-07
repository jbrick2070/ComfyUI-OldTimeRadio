# PLAN -- banana-route QA fixes (the six steps)

**Date:** 2026-08-06. **HEAD:** `ec9da848` on `v2.0-alpha`.
**Baseline, measured at this HEAD:** suite 9033 passed / 111 skipped / 1 xfailed;
Bug Bible 17.
**Contract this serves:** `docs/2026-08-06-BUILD-SPEC-banana-route.md`.

**Provenance.** The banana route itself already ran a full four-round
`kibitz-plugin:kibitz` arc before it landed. THIS plan is a different document:
it is the six fixes produced by a read-only QA pass over the UNCOMMITTED build
(one blocking defect, two live-proven bugs, two coverage holes, one cosmetic).
Every claim below was reproduced by running the real code in the ComfyUI venv
against the real Windows tree -- no claim here is static-read speculation.

**The uncommitted diff under repair:**
* NEW `nodes/_otr_banana_route.py`
* NEW `tests/test_banana_route.py`
* EDIT `nodes/otr_image_gen_dispatcher.py`
* EDIT `nodes/_otr_video_engines/render_driver.py`

---

## DECIDED BY THE OPERATOR -- do not relitigate

1. **VISUALS ONLY.** The spoken script is never touched.
2. **Quoted `still_word` card text is SHIELDED** -- text rendered inside the
   picture is script, not picture.
3. **Two env switches default ON**, with `shakespeare` / `public_domain`
   excluded via the copied `_LEMMY` idiom; `OTR_BANANA_INCLUDE_FIDELITY_BANKS`
   is the operator's force-on override.
4. **NO node widget and NO `workflows/otr_canonical.json` change.**
5. **Option A at the two existing funnels** -- the emptied
   `append_visual_safety_clause` seam's route, repaired, not rebuilt.
6. **Default-ON is intended**, and still seeds / cache keys are EXPECTED to move
   on the original lane.

---

## FIX 1 (BLOCKING) -- the cap measures the wrong budget

### The defect

`nodes/_otr_video_engines/render_driver.py:2909`:

```python
_btext = _banana_cap(_bres.text, max(188, len(_banana_prompt)))
```

`_banana_prompt` is captured at `:2885`, BEFORE the transform. So for any branch
whose composed prompt already exceeds 188 characters, the budget IS the
incidental pre-transform length: the prompt is forbidden to grow by even one
character, and `cap_phrase_safe` (`_otr_banana_route.py:383-396`) trims the
growth off the END.

The comment above the line claims the opposite -- *"the budget is the LARGER of
188 and the pre-transform length -- honors the promise where it was made, never
slashes a branch that made no such promise."* It does slash such a branch.

### Live evidence (run in the ComfyUI venv against the real constants)

| branch | pre | post-transform | budget | capped | destroyed |
|---|---|---|---|---|---|
| ia2v compact talking | 185 | 200 | 188 | 185 | `Static camera.` -- gone entirely |
| brief+beat scene | 204 | 219 | 204 | 196 | `slow cinematic camera drift` -> `slow,` |

The ia2v case is the sharp one. That branch is explicitly engineered to protect
its own tail -- `render_driver.py:2633-2638`:

```python
_frag_budget = (_LTX_MOTION_PROMPT_MAX          # 240, :1261
                - len(_cue_prefix)
                - len(_IA2V_TALKING_CLAUSE_CHARACTER)   # 131 chars
                - 1)
```

with the comment *"the fragment shrinks so the proven IA2V talking clause still
stays intact."* The banana cap then trims from exactly that end. The branch's
real budget is 240; the cap pinned it at 185.

### The production failure

On the default-ON original lane, every character face beat whose identity
fragment carries `gunman` (+15 / +19 chars) or `rifle` (+6) ships a lip-sync
prompt with the camera lock stripped, and scene beats ship a mangled dangling
clause. Silent: no warning, no receipt field records the truncation, and
`banana_sha256_after` is recomputed over the amputated string at `:2911-2914`,
so the receipt looks clean.

### The proposed fix -- thread the DECLARED budget

Seven sites assign `req["text_prompt"]` in `build_request_from_shot`. Only three
declare a budget:

| site | declared budget |
|---|---|
| `:2662` (M4 / ia2v talking / motion register) | `_LTX_MOTION_PROMPT_MAX` = 240 (`:1261`, used `:2634`, `:2783`) |
| `:2831` (google provider) | `620` (`:2819`) + the `_prefix_video_style_cue` prefix |
| `:2872` (brief+beat) | `188` (`:2866`) + the `_prefix_video_style_cue` prefix (`:2867`) |
| `:2689`, `:2695`, `:2736` | **none** -- no cap is owed |

(`:1505` is inside `_apply_visual_safety_prompt`, which is identity since the
2026-08-05 rip.)

Edits:

1. Initialize `_prompt_char_budget = None` once, before the branch cascade.
2. In each of the three budgeted branches, set it to the number that branch
   actually promised. On `:2831` / `:2872` set it AFTER `_prefix_video_style_cue`
   runs, as `max_chars + (len(after_cue) - len(before_cue))`, so the prefix the
   branch legitimately added is not charged against the body.
3. Replace `:2909` with a conditional: cap to `_prompt_char_budget` only when it
   is set AND the post-transform text exceeds it; otherwise pass `_bres.text`
   through uncapped.
4. Delete the "LARGER of 188" comment -- it will no longer describe the code.

**Acceptance for this step:** with `gunman` in the identity fragment, an ia2v
talking prompt keeps `Static camera.`, and a brief+beat prompt keeps
`slow cinematic camera drift` intact.

---

## FIX 2 -- `_bool_env` maps a present-but-empty value to TRUE

`nodes/_otr_banana_route.py:404`:

```python
_TRUE_TOKENS = frozenset({"", "1", "true", "yes", "on"})
```

`raw is None` at `:412-413` already covers "unset", so `""` in the true set is
both redundant and wrong. `_bool_env` never consults its `default` parameter for
a present value, so an empty or whitespace-only value returns True regardless.

Live evidence:

```
OTR_BANANA_INCLUDE_FIDELITY_BANKS=''     -> include_fidelity=True   shakespeare gate=True
OTR_BANANA_INCLUDE_FIDELITY_BANKS=' '    -> include_fidelity=True   shakespeare gate=True
OTR_BANANA_INCLUDE_FIDELITY_BANKS='0'    -> include_fidelity=False  shakespeare gate=False
OTR_BANANA_INCLUDE_FIDELITY_BANKS unset  -> include_fidelity=False  shakespeare gate=False
```

A launcher line with a trailing space, or a programmatically built subprocess env
dict carrying `""`, silently turns the route ON for `shakespeare` and
`public_domain` -- the exact exclusion operator ruling 3 protects. Invisible on
the two default-ON keys; load-bearing only here. It also contradicts the
contract at `BUILD-SPEC:45`, which lists only `unset/1/true/yes/on` as ON and
sends everything else to *default + one warning*.

**Fix:** delete `""` from the frozenset.

---

## FIX 3 -- an odd trailing backslash drops the quote shield for the whole card

`nodes/otr_meta_brief_image_prompt.py:958-970` (`_still_word_clean_line`) scrubs
brackets `:965`, parentheticals `:966`, a leading `SPEAKER:` label `:967`, and
every double-quote variant via `_fold_inner_dquotes` `:968` -- but never a
BACKSLASH. `_still_word_fit_card` (`:928-947`) only strips whitespace.

A cleaned line ending in an ODD backslash run makes the card's CLOSING `"` read
as escaped (`_otr_banana_route.py:233-240`). No other unescaped `"` exists in the
composed card -- the legibility guard `:836-838`, text guard `:842-843`,
lettering, backdrop, era and grade pieces are all quote-free (verified across all
nine style packs) -- so `_shielded_spans` returns `[]` and the ENTIRE prompt
transforms.

Live evidence:

```
spoken='He drew his revolver!'    spans=1 subs=0   <- shielded, correct
spoken='He drew his revolver \'   spans=0 subs=1   <- LEAK: card reads "He drew his banana \"
spoken='He drew his revolver\\'   spans=1 subs=0   <- even run, correct
```

This is precisely the visuals-only violation the shield exists to prevent, on the
one audience-readable surface. Reachability is low (a writer LLM emitting a
trailing backslash in dialogue) but the consequence is the worst one in the
build.

**Fix at the source, not at the shield:** in `_still_word_clean_line`, add
`s = s.replace("\\", " ")` before the `_fold_inner_dquotes` call at `:968`. A
backslash is never legible card lettering. Leave `_is_escaped` alone -- QA
ruling 9 (same-style pairing, odd/even backslash parity) stands.

**NOTE:** this widens the commit pathspec to `nodes/otr_meta_brief_image_prompt.py`,
a file outside the original diff. Call that out explicitly; do not let it ride.

---

## FIX 4 -- three tests that close what fixes 1-2 open

**(a) The ia2v cap shape.** A >188-char prompt containing `gunman`, branch budget
240: assert the transformed prompt is LONGER than the original and the trailing
clause survives byte-identical. This is the regression guard for fix 1 -- without
it, fix 1 can silently regress to the pre-length budget.

**(b) The cap RETREAT branch** (`_otr_banana_route.py:388-395`) has ZERO coverage
today. Hand-traced: in `tests/test_banana_route.py:220-225` the phrase occupies
indices 152-176, `cut` lands at 151 (the space BEFORE the phrase), and
`lowered.rfind(phrase, 0, cut + len(phrase))` returns -1 because the phrase ends
at 177. The branch never executes, and the assertion passes on its left disjunct
with the phrase absent from the output entirely. `BUILD-SPEC:252-253` explicitly
demands *"unit tests with prompts built to land the boundary inside `man wielding
a red banana`."* A working input: `body = "x"*100 + " man wielding a red banana"`,
`max_chars=110` -> `cut` 104 -> retreat to 101 -> strip to 100.

**(c) The env guard.** `OTR_BANANA_INCLUDE_FIDELITY_BANKS=""` and `" "` must both
leave the shakespeare gate OFF.

---

## FIX 5 -- one `dispatch_images` integration test

The six receipt keys on the cache-HIT row (`otr_image_gen_dispatcher.py:1216`),
the fresh row (`:1377`) and the `stills_manifest.json` projection (`:1540-1547`)
have no assertion anywhere in the suite. About 30 existing `dispatch_images`
calls across six test files DO execute the still funnel (they carry no
`meta.source_bank`, so the gate defaults ON, and two of them parse the produced
manifest) -- so the code runs green in CI, but delete the whole dispatcher hunk
and the suite stays green.

`BUILD-SPEC:398` lists all of this as required integration coverage.
`test_still_seam_hash_covers_the_transform` (`tests/test_banana_route.py:452`)
proves the ORDERING property on the composed functions, not the wiring.

**Fix:** call `dispatch_images` with an original-lane ledger; assert the six keys
on a fresh row, on a cache-HIT row (second call, same ledger), and in the parsed
`stills_manifest.json`. Model it on the harness at
`tests/test_image_platform_c1.py:1058`, which already produces and parses the
manifest.

---

## FIX 6 (cosmetic) -- the dead `.replace` in the article test

`tests/test_banana_route.py:118` carries
`"an long banana".replace("an long", "an long")` -- a no-op that reads like
someone patched around a wart rather than recording it. Replace with the plain
literal plus a one-line comment recording that indefinite-article agreement is
deliberately NOT repaired: `an assault rifle -> an long banana` and
`an ice pick -> an banana` are real output (both sources are vowel-initial,
`_ROWS:108` and `:136`), cosmetic only, with no hash or receipt consequence.

---

## GATES

Focused run of `tests/test_banana_route.py` + `tests/test_image_platform_c1.py`
-> full suite (expect 9033 + the new tests) -> Bug Bible 17 -> AST / BOM /
zero-byte on every touched file -> Sonnet QA on the diff -> Fable gate -> ONE
pathspec commit -> push -> `HEAD == origin`.

**No `workflows/` change.** If the diff touches `workflows/`, stop: a decision
was made against that.

Commit pathspec (five files):

```
nodes/_otr_banana_route.py
nodes/_otr_video_engines/render_driver.py
nodes/otr_meta_brief_image_prompt.py
tests/test_banana_route.py
tests/test_image_platform_c1.py   (or wherever fix 5 lands)
```

plus `nodes/otr_image_gen_dispatcher.py` and the two spec/plan docs from the
underlying build.

---

## WHAT THE PANEL IS ASKED

**r1 (arc).** Is fix 1's "declared budget" model the right SHAPE at all, or is
there a cleaner invariant -- cap at the branch budget only when the transform
GREW the text; let the prompt grow and let the adapter clamp; move the cap inside
`finish_visual_prompt`? Is fix 3 better solved at the card composer or at the
shield? Is anything in the six-step plan actually a symptom of a wrong SEAM
choice made when the route landed?

**r2 (coding).** Exact edits; the `_prompt_char_budget` threading; the style-cue
accounting on `:2831` / `:2872`; the budget for branches that declare none; and
whether `cap_phrase_safe` itself has a latent defect -- the `budget <= 0`
hard-cut at `:381` that returns `text[:max_chars]` mid-word and can drop the
trailing clause entirely; the single-`break` phrase loop at `:389-395`; a
`_TRAILING_CLAUSE` `rfind` at `:371` that can match INSIDE a quoted span.

**r3 (wiring).** Confirm nothing here needs a `workflows/` change. The commit
pathspec now spans five files plus the docs. The six receipt keys must still
reach the node-92 `/history` report via the copy list at
`render_driver.py:3788-3792`. Confirm the fixes do not disturb the pinned
ordering (safety hook -> banana -> cap -> assign -> restamp WITHOUT logging ->
seed) or the ON/OFF seed equality (`:2924-2947`, seed derives only from the
shot's stamped request hash).

**r4 (convergence).** Confirm no new must-fix, and give the gate order and the
acceptance the first live leg should prove.
