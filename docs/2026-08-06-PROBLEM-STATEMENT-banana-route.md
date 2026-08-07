# PROBLEM STATEMENT -- the banana route for stills and video

**Date:** 2026-08-06. **HEAD:** `fa53a8b0` on `v2.0-alpha`.
**Written because:** the operator asked whether the still + video engine needs a
master re-route that turns every weapon and gun into a banana (the ComfyUI-Goofer
behaviour), or whether the filter has to be bolted onto every still and video
route individually.

**STATIC.** No code was changed to write this. Nothing here enters
`docs/PROD_BUG_LOG.md` without a live artifact. Every claim cites a real file at
HEAD, read on the Windows tree.

---

## 0. THE SHORT ANSWER

**Neither. You do not have to redo the engine, and you do not have to touch every
route. Both master routes already exist, and one of them already has this exact
seam -- you emptied it yesterday.**

| Lane | The single funnel that already exists | Line |
|---|---|---|
| Stills | `nodes/otr_image_gen_dispatcher.py` -- the ONLY place a prompt becomes an `ImageRequest` | `:1000` |
| Stills | `nodes/otr_image_gen_dispatcher.py` -- the ONLY `render_image()` call site in the repo | `:1569` |
| Video | `nodes/_otr_video_engines/render_driver.py` -- after every prompt branch settles `req["text_prompt"]` | `:2874` |
| Video | `nodes/_otr_video_engines/render_driver.py` -- the only two `render_clip()` call sites, both downstream | `:3048`, `:3057` |

Two lines of live code carry every still and every clip this pipeline makes.
The work is not building a master route. The work is **repairing the master
route that is already there** -- it is currently a no-op, and on the video side
it is wired to only a third of the engines.

---

## 1. WHAT GOOFER ACTUALLY DOES (the reference implementation)

`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-Goofer\goofer_sanitizer.py`

It is not an engine feature and not a model call. It is **one ordered list of
string pairs applied to the prompt text before the prompt reaches the model**:

* `_BANANA_REPLACEMENTS` -- 90 pairs, `:127-222` (counted, not estimated).
  Ordered longest-first so `machine gun` matches before `gun`.
* Applied at `_apply_banana_filter`, `:498-510`: `re.sub` per pair, word-boundary
  anchored (`\b...\b`), `re.IGNORECASE`.
* Called as **step 8 of 9** in the sanitize chain, `:354-356` -- after names,
  studios, brands and PII, before the custom blocklist.
* Operator control: one `BOOLEAN` widget, `banana_filter`, default `True`,
  `:248-255`.

Coverage in that list is three groups: weapons (`shotgun`, `assault rifle`,
`grenade`, `knife`, `bazooka`, `tank`), violence verbs (`shoot`, `stab`,
`strangle`, `murder`, `kill`), and gore (`blood`, `gore`, `explosion`, `bomb`).

**Nothing about it is Goofer-specific.** It is a pure text transform with zero
imports beyond `re`. It ports to OTR as a module, not as a rewrite.

---

## 2. THE STILL ROUTE ALREADY FUNNELS

`nodes/otr_image_gen_dispatcher.py:1000`, inside the per-object dispatch loop:

```python
prompt = append_visual_safety_clause(str(obj.get("prompt") or ""))
prompt_hash = _prompt_content_hash(prompt)
```

Every still in the pipeline passes through that line. Proof: `render_image()` is
called from exactly one place in the entire repo -- `:1569`,
`return eng.render_image(request, prepared)`. Behind it sit **eight adapter
modules** under `nodes/_otr_image_engines/` exposing **thirteen registered
engine ids** (local `flux_gen1`, `flux2_klein`, `lumina_image`, `z_image_turbo`,
`sd35_large`, `hidream_i1`; cloud `cloud_flux_pro`, `cloud_nano_banana_2`,
`cloud_seedream_2`, `cloud_krea_2_turbo`, `cloud_luma_photon_flash`, `ideo`,
`google_image`), all reached from that one call. *(Corrected on r1: the earlier
draft said "eleven adapters" while listing thirteen ids.)*

Note the ordering, because it is already correct: the content hash on `:1001` is
taken **after** the transform. Flip a banana switch and every cached still
re-mints instead of serving a stale gun.

---

## 3. THE VIDEO ROUTE ALREADY FUNNELS

`nodes/_otr_video_engines/render_driver.py`. The composed prompt lands on
`req["text_prompt"]` from six different branches -- motion register, cheap-family
fallback, default, env override, motion-role, brief+beat (`:2661`, `:2688`,
`:2694`, `:2735`, `:2830`, `:2871`). All six converge on one line:

```python
:2874    _apply_visual_safety_prompt(req, shot)
```

and `render_clip()` is not called until `:3048` / `:3057`. Every video engine --
local `wan_ti2v`, `wan_i2v`, `ltx_video`, `ltx_8gb`, `ltx_av`, `fastwan_8gb`,
`humo`, the `cheap_families` still-pan set; cloud `eng_cloud_video`; the Google
`veo` / `omni` / `vid_sfx` lane -- reads `text_prompt` off that request.

**One line, every clip.** That is the master route you were asking whether you
would have to build.

---

## 4. THE SEAM IS BUILT AND IT IS EMPTY

`nodes/_otr_story_brief_helpers.py:25-52`, written **yesterday**:

> `RETIRED 2026-08-05 (operator directive: no content guardrails on generated
> episodes)` ... `Emptied rather than deleted: they are imported by name across
> the render path, and an empty clause appends nothing.`

`VISUAL_SAFETY_POSITIVE_CLAUSE = ""`, `VISUAL_SAFETY_NEGATIVE_PROMPT = ""`, and
`append_visual_safety_clause()` now returns the prompt unchanged. The docstring
is explicit that the function survives **"because it is a named seam on the
render path, not because it still does anything."**

So the plumbing for a repo-wide prompt transform is installed, imported by name
in **seven production files** outside its own module
(`otr_image_gen_dispatcher.py:44`, `eng_google_image.py:27`,
`eng_cloud_image.py:39`, `eng_cloud_video.py:47`, `eng_google_omni_video.py:32`,
`eng_google_veo_video.py:33`, `render_driver.py:1498`), and covered by tests --
it just has nothing in it. The banana route is that pipe with something in it.

**One honest note on intent.** What was ripped on 08-05 was a *content
guardrail*, and the standing directive says do not put those back. A banana
filter is a different animal: it is a **house-style comedic transform**, an
authorial choice, not a safety gate. It should be named, switched and documented
as style -- `banana_route`, not `visual_safety` -- so nobody in six months reads
it as the guardrail creeping back in. Section 7 is the part of that call I will
not make for you.

---

## 5. THE FOUR DEFECTS BETWEEN "SEAM EXISTS" AND "BANANA ROUTE WORKS"

### D0 -- THE VOCABULARY IS UNSHIPPABLE AS-IS. This is the real problem.

*Added after r1. Codex and Fable found this independently, from different files.
It outranks everything below: the wiring was never the hard part.*

**D0a -- `("shot", "hit with a banana")` (`goofer_sanitizer.py:191`) collides
with OTR's camera vocabulary.** Confirmed at five composition sites:

| Site | Text it composes |
|---|---|
| `render_driver.py:2855` | `cinematic establishing shot` -- the fallback core for every scene clip with no brief |
| `otr_meta_brief_image_prompt.py:146` | `in-character cinematic medium shot, head and shoulders` |
| `otr_meta_brief_image_prompt.py:264` | `centered and fully visible, wide shot` |
| `otr_meta_brief_image_prompt.py:1347` | `for a 16:9 LANDSCAPE shot of this character` |
| `otr_meta_brief_image_prompt.py:1349` | `a medium/wide shot` |

Ported verbatim, a large fraction of prompts become *"cinematic establishing hit
with a banana"* -- corrupted framing on renders with no weapon anywhere near
them. `tank -> armored banana cart` has the same disease against period water
and gas tanks.

**D0b -- the table is not even single-pass safe.** `_apply_banana_filter`
(`:498-510`) walks the pairs **in order**, mutating as it goes.
`("hand grenade", "banana bomb")` at `:137` is later re-matched by
`("bomb", "banana bomb")` at `:221`: **`hand grenade` -> `banana banana bomb` in
ONE pass**, before D2's double-application is even in play. Nine pairs have this
shape.

**D0c -- the table changes the STAKES, not just the props.**
`murder -> banana party`, `strangle -> tickle`, `killed -> slipped on a banana`
contain no prop at all -- they rewrite what happened. That is the retired
guardrail in a funnier hat, and it is exactly what the 08-05 rip removed.

**The rule this settles: bananas replace the instruments, never the stakes.**
Ship OTR's own table -- unambiguous weapon nouns, camera-safe, single-pass
closed (no replacement may contain any row's source term), idempotence tested as
a property. Drop the violence and gore groups for v1. Goofer's list is a seed to
harvest, not a file to copy.

### D1 -- the video seam only fires for cloud engines

`render_driver.py:1488-1493`:

```python
def _apply_visual_safety_prompt(req, shot) -> None:
    engine_id = str((shot or {}).get("engine_id") or "").strip()
    if engine_id in _GOOGLE_VIDEO_SFX_ENGINES:
        return
    if not _is_cloud_video_engine(engine_id):
        return
```

That gate was correct for its old job -- appeasing a cloud provider's content
policy is pointless on a local checkpoint. It is **wrong** for a banana route:
every local Wan / LTX / HuMo render, which is most of what this box makes, would
sail straight past with its guns intact. The banana transform must be
unconditional over engines.

### D2 -- double application, and Goofer's table is not idempotent

The adapters call the clause a **second** time on their own composed prompt:
`eng_cloud_image.py:283`, `eng_google_image.py:141`, `eng_cloud_video.py:251`,
`:541`, `:993`, `eng_google_omni_video.py:78`, `eng_google_veo_video.py:114`.

Harmless today, because the clause is a no-op. **Not harmless with Goofer's
table**, which compounds on re-entry:

| input | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| `bomb` | `banana bomb` | `banana banana bomb` | `banana banana banana bomb` |
| `grenade` | `banana bomb` | `banana banana bomb` | ... |
| `explosion` | `banana explosion` | `banana banana explosion` | ... |

Three replacement pairs re-match their own output. The fix is a one-shot with a
receipt: stamp the request when the transform runs and skip if already stamped
(the existing `observability["visual_safety_prompt"] = "applied"` at `:1510` is
the precedent), or make the adapters stop re-calling it. Either way this must be
decided before code, not discovered in a render.

### D3 -- WITHDRAWN 2026-08-06. I was wrong; there is no cache to invalidate.

*Original claim: the video request hash does not cover the transform, so
flipping the switch would serve a cached pre-banana clip. Codex called this a
false premise on r1 and it is. Kept visible rather than deleted so nobody
re-derives it.*

`req_hash` occurs exactly three times in `render_driver.py`: the assignment at
`:2875`, and two seed derivations at `:2894` / `:2896` feeding
`req["seed_bundle"]` at `:2898`. It is the deterministic **sampler seed**, not a
clip-cache lookup. Nothing re-renders stale, and holding the seed constant is
positively desirable -- it makes switch-ON vs switch-OFF the same noise, so the
only difference in the output is the banana.

The still lane never had this problem either: `:1000` transforms, `:1001` hashes
the transformed text, so still identity covers the route for free.

**Verify-at-build:** no consumer outside the reviewed render path reads
`render_request_hash` for reuse.

### D4 -- no ledger receipt

Nothing records that a prompt was transformed. Per the standing ledger rule, a
new owner of prompt text owes downstream consumers a field: which prompt was
sent, that the banana route touched it, and how many substitutions fired. The
observability dict already carries `prompt_source` / `prompt_sha8` /
`prompt_chars` (`:1470-1485`) and is the natural home.

---

## 6. THE TWO DESIGNS, COMPARED HONESTLY

**Option A -- master re-route (recommended).** One new pure module,
`nodes/_otr_banana_route.py`: the pair table, an idempotent `apply()`, and a
receipt. Wired at exactly two live sites -- `otr_image_gen_dispatcher.py:1000`
and `render_driver.py:2874` (with the D1 gate removed). Adapters keep calling
their existing helper, which becomes a documented no-op or a receipt check.

* Two call sites to review. One vocabulary. One switch. One receipt.
* A new engine added tomorrow inherits it for free -- it cannot route around a
  funnel it does not know exists.
* Costs: fixing D1-D4 properly, and a re-baseline of any cached stills/clips.

**Option B -- filter on every route.** Nine-plus call sites across the image
adapters, the video adapters and the Google lane.

* This repo **already ran this experiment**. `append_visual_safety_clause` was
  wired into every route, and the scatter is exactly why D2 exists.
* Every future engine is a fresh chance to forget one, and a missed one is a
  silent gun in a published episode, not a crash.
* The only thing B buys is per-engine variation of the vocabulary, which nobody
  has asked for.

**Verdict: Option A.** B is not a different design, it is the same design with
the failure mode already demonstrated in this tree.

**A side benefit I claimed and cannot support -- struck 2026-08-06.** I wrote
that banana prompts "sail through provider moderation" that weapon prompts do
not. No live artifact or provider contract in this repo supports it, and an
unchanged source image can still be seen by image-side moderation. If it turns
out true it is a bonus; it is not a reason to build.

---

## 7. SWITCH SCOPE

**OPERATOR RULING 2026-08-06 (a): it is an OPTION, not a policy.** A switch,
exposed, so anyone running this repo can turn it off if they want it off. That
settles the shape -- the banana route is never hardcoded, never mandatory, and
never un-disableable. It also settles the naming argument in section 4: a thing
the user is invited to switch off is a style option, not a guardrail.

**OPERATOR RULING 2026-08-06 (b): TWO master switches -- one at the still funnel,
one at the video funnel -- and the SMALLEST diff that delivers them.** The switch
is the requirement; everything else is subordinate to it. Two switches map 1:1
onto the two chokepoints in section 0, so the wiring is one boolean read at
`otr_image_gen_dispatcher.py:1000` and one at `render_driver.py:2874`. It also
buys a real capability for free: stills bananafied while video is left alone, or
the reverse, without a code change.

**What "smallest diff" rules OUT, and this matters:** it does NOT license
skipping D2 (the double-application compounding bug) or D3 (the video cache key).
Those are not polish -- a switch you cannot turn off cleanly, or that serves a
cached pre-banana clip when you flip it, is a switch that does not work. The
minimal-diff target is *new surface area*: one new pure module, two boolean
reads, one config key, one canonical-workflow widget. Not: a new node, not an
engine refactor, not a per-adapter edit.

What that ruling does NOT yet settle is the DEFAULT and the REACH. Goofer ships
one global boolean, default on. OTR has lanes with opposite intent, and the
standing rule for two of them is that they invent nothing and stay true to
source.

* **`original`** -- pure invented radio drama. Bananas here are free comedy and
  cost nothing.
* **`shakespeare` / `public_domain`** -- fidelity lanes. `Is this a dagger which
  I see before me` becomes `Is this a banana which I see before me`. Macbeth's
  dagger, Lear's sword, every duel in the corpus.

That is either the funniest thing this repo has ever done or a fidelity defect,
and it is a taste call, not an engineering one. Given the ruling above, the
switch exists either way; the open question is only what it reads by default:

1. **One switch, default on** -- Goofer parity. Simplest. Bananas Macbeth until
   somebody turns it off.
2. **One switch, default on, with the fidelity banks defaulting it off** --
   respects the invents-nothing rule; costs one config key per bank; the operator
   can still force it on anywhere.
3. **Per-episode operator widget** -- most control, one more widget to carry, and
   widget order in the canonical workflow is positional (append only).

### 7a. RULED: VISUALS ONLY (operator, 2026-08-06)

This half of the call is CLOSED. Asked whether the filter should reach spoken
lines as well as image prompts, the operator: *"No. Just visuals. I do not want
people discussing the Cavendish versus the other variety."*

**So the substitution happens on the STILL/VIDEO PROMPT and nowhere else.** The
announcer says "he drew his revolver" over a shot of a man holding a banana.
This document called that "either exactly the joke, or the thing that breaks
it". **It is the joke.**

What that rules OUT, so no round re-opens it:

* the dialogue ledger, `_otr_line_composer`, the writer and every adaptation
  lane are **OUT OF SCOPE** -- the spoken script is not touched, at all;
* no prompt-surface change to the writer, which keeps this clear of the closed
  story-quality directive (2026-08-04);
* the fidelity lanes' invents-nothing rule is untouched **at the TEXT level** --
  Macbeth still says "dagger". Only the picture changes.

It also collapses the blast radius to exactly the two funnels section 0 names:
`otr_image_gen_dispatcher.py:1000` and `render_driver.py:2874`. Nothing in the
audio or script half of the pipeline is in this build.

### 7b. "VISUALS ONLY" IS A SCOPE, NOT A SITE -- enumerate every visual PROMPT surface

The ruling narrows WHICH HALF of the pipeline is edited. It does not by itself
say the two funnels are the only places a weapon word reaches an image model,
and filtering one surface while a sibling leaks is the defect this repo keeps
finding. Grounded at HEAD:

* **`req["text_prompt"]` -- the video funnel.** `render_driver.py:1488-1498`
  already reads it, transforms it and writes it back
  (`req["text_prompt"] = safe`). This IS the empty seam of section 4; the banana
  pass goes here, beside `append_visual_safety_clause`.
* **The still funnel.** `otr_image_gen_dispatcher.py:1000`, same shape.
* **THE MOTION CLAUSE IS A SECOND VIDEO PROMPT SURFACE and it is easy to miss.**
  `_motion_clause_override` (`render_driver.py:1393-1401`) composes text from
  `_otr_motion_clause` INDEPENDENTLY of `text_prompt`. It is **default OFF**
  (`_otr_motion_clause.py:13-14`, and `PBUG-20260805-04` relied on exactly that
  fact), so it leaks nothing today -- but a banana route that filters only
  `text_prompt` is one env flag away from shipping an unfiltered action clause.
  Decide it explicitly: filter it too, or state in the code that it is out of
  scope BECAUSE it is off, so the next person who enables it finds the note.
* **VERIFY AT BUILD, not from this document:** every adapter's
  `negative_prompt` (e.g. `eng_ltx_video._build_render_request`), any style-cue
  prefix (`_prefix_video_style_cue`, `:2654`), and the still-plan
  `framing_geometry` / `style_tail` strings. These are unlikely to carry weapon
  nouns -- they are quality and framing vocabulary -- but "unlikely" is the word
  that preceded every leak this repo has logged. Grep them for the 90-term list
  once and record the result.

The point of 7b is not to widen the build. It is that "visuals only" must be
implemented as **every visual prompt surface**, not as "the first visual prompt
surface we found".

### 7c. DECIDED: option 2 -- global default ON, the FIDELITY LANES default OFF

Decided by the driver under the operator's standing rule (*"take the design that
has the least amount of coding for most elegant"*), with Fable consulted on the
taste half. **The widget is cut entirely.**

**THE COST OBJECTION IN THIS DOCUMENT WAS WRONG.** Section 7 said option 2
"costs one config key per bank". It does not. The pattern already exists,
already has a name, and already states this exact principle:

```python
# nodes/_otr_casting.py:1238
_LEMMY_EXCLUDED_SOURCE_BANK_IDS = frozenset({"public_domain", "shakespeare"})
```

with `_source_bank_excludes_lemmy()` at `:1249` and, at `:1246`, the rule in as
many words: **"fidelity is a family behaviour, not a per-row opt-in."** The
codebase already decides a CREATIVE question (may Lemmy be cast?) by exactly
this test. Option 2 is a three-line copy of a blessed idiom, not new machinery --
so it is behind option 1 by a whisker on lines and ahead of it on elegance,
because the alternative is two different answers in one tree to "what is a
fidelity lane".

Option 3 fails the operator's rule outright: the most code, and it adds a
POSITIONAL widget to the canonical workflow (the BUG-LOCAL-097 drift class) to
buy control option 2 gives away. A one-off taste call can flip the global for
that render.

**THE TASTE ARGUMENT, and it is not that bananas are unfunny.** Incongruity
comedy needs a straight man, and the operator's visuals-only ruling supplies a
perfect one -- the announcer says "revolver" dead straight over a banana. That
works when the show OWNS BOTH CHANNELS. On `original`, Signal Lost is the
author, and the banana is the show's own voice. On `shakespeare`, the author is
Shakespeare, and the same dissonance makes the ADAPTATION the butt of the joke:
it turns "we adapted Lear faithfully" into "we did a bit on top of Lear", which
is a different show.

**And fidelity DOES reach the frame.** The invents-nothing rule was written
about script because script was the product surface then; the principle is the
audience contract, and in a video product the frame is half the storytelling.
Rendering the dagger in 1940s-radio style is COSTUME -- translation. Replacing
it with a banana is COMMENTARY. The precedent is already ruled, in `CLAUDE.md`:
the packs that forbade "blood, guns, knives, and graphic violence" while
adapting Macbeth and King Lear were **"a fidelity defect rather than a safety
win."** A default-ON banana filter on that lane is that defect wearing a clown
nose -- and the dagger in Macbeth is not a prop, it is the most famous speech in
the play. **Option 2 is the anti-guardrail: it lets the dagger be a dagger.**

### 7d. THE RISK NOBODY HAD NAMED: partial coverage makes the joke illegible

Ninety word-pairs cannot cover violence. *"Macbeth stands over Duncan's body"*
contains no listed term and renders a corpse. So even on the lane where the gag
is WANTED, episodes will MIX real weapons with bananas -- and an audience cannot
tell a deliberate bit from a broken renderer. **A joke has to be legible as
deliberate; inconsistent substitution reads as failure**, and a half-banana
episode looks worse than either extreme.

That is an argument about the ORIGINAL lane, where the filter is on by default,
and it is not a reason to cut the feature. It is a reason to look at a rendered
episode before publishing one, and to expect the term list to grow from real
output rather than from the ported 90.

Smaller, same family: visual models draw bananas BRIGHT YELLOW and MODERN -- in
a 1940s monochrome-ish frame the substitution may be the only object that is not
period. Worth one look on the first leg.

---

## 8. WHAT I DID NOT TOUCH

No code changed. No workflow JSON changed. `workflows/otr_canonical.json` is
untouched. `_otr_content_safety.py` (which still carries a usable
`EXPLICIT_WEAPON_TERMS` vocabulary at `:48-71`) has been at **zero production
references** since the 08-05 rip and stays that way -- it is a term list to
harvest, not a module to re-arm.

---

## 9. NEXT

Per the standing directive this is a coding item, so it gets the full four-round
`kibitz-plugin:kibitz` arc (Codex + Antigravity, Claude drives and judges) with
Fable consulted on r1, before any code is written. This document is the driver
anchor for that campaign.
