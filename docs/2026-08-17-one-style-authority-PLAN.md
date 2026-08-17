# ONE STYLE AUTHORITY -- build plan v2 (2026-08-17)

Operator-directed, stated twice. Queue item 1. Repo `ComfyUI-OldTimeRadio`,
branch `v2.0-alpha`, HEAD `772d5432`.

**v2 supersedes v1 and the arc is being RESTARTED on this document at the
operator's instruction** ("for good measure restarting r1... or maybe r2").
That is the right call: v1 had three parts, and this has five, including a
whole new component (the visual ledger) and a new instrument (the traceroute)
that no r1 lane ever saw. Everything below is grounded at the real files.

## The defect (a live published episode, not a review finding)

`signal_lost_kinetic_motion_clause_live_test_20260817_050130`, `visual_style` =
`cartoon`. Delivered mp4: 00:04 bookend CARTOON, 00:16 announcer CARTOON,
00:34 character PHOTOREAL, 00:43 announcer painterly. One 74-second episode
cutting between an animated short, a live-action film and a painting.

**Root cause:** `nodes/_otr_image_engines/z_image_turbo.py:216-219` ships a
style-blind default negative containing `"cartoon, illustration"`, sampled at
cfg 2.0 (`:222`, "keeps the negative live"). Every still on a cartoon episode
was minted with positive "bright cartoon illustration" AND negative "cartoon,
illustration". The engine vetoes the style the episode selected.

**Amplifier:** `z_image_turbo` is the only engine declaring
`accepts_reference_image = True` (`:162`); the poisoned photoreal portrait
became the `reference_latent` applied to BOTH conditionings (`:318-331`) and
propagated into every character beat. i2v then carried each still faithfully.

**Why it looked like a gradient:** where the positive was pack-authored and
cartoon-front-loaded (bookends, announcer) cartoon won; where it was LLM prose
leading with an anatomy-dense face description, photoreal won.

## Settled facts -- do NOT re-derive these

* **Execution order is `ShotLock (90) -> MetaBrief (89) -> ImageGenDispatcher
  (91) -> VideoRenderBatch (92)`**, proven from the link table in
  `workflows/otr_canonical.json`. Node IDs are NOT execution order.
* **Still prompts ride `image_prompts_json`** (89 -> 91 slot 2). **Shot prompts
  ride `ledger["video"]["shots"]`** (90 -> 91 slot 0 -> 92). Two families, two
  wires. Node 91 is the ONLY node that sees both.
* The default negative literally mis-serves **FOUR** of nine packs --
  `anime`, `cartoon`, `sci_fi_radio`, `storybook_engraving` -- measured by
  the phrase match the traceroute uses. `recur_frac` ("clean digital" vs
  "pristine raster geometry") and `video_art` ("oversaturated, glossy" vs
  "phosphor bloom") read as mis-served on JUDGMENT but are not literal
  collisions, so they are not counted. An earlier draft said "six" by mixing
  the judgment into the measurement; four is the measured number.
* A negative is only live where cfg > 1: `z_image_turbo` 2.0, `lumina_image`
  4.0. `flux_gen1` runs **cfg 1.0** (`:87`) so its negative is inert.
  `flux2_klein`, `sd35_large`, `hidream_i1` and the two cloud engines have no
  negative conditioning at all.
* The safety negative is **empty** (`VISUAL_SAFETY_NEGATIVE_PROMPT = ""`,
  `_otr_story_brief_helpers.py:34`) and `append_visual_safety_clause` is a
  documented pass-through retired 2026-08-05. Do NOT reintroduce one.
* `visual_safety_negative` (`:49-63`) survives as a comma-split dedupe/merge
  utility, which is the seam any negative composition should reuse.

## Part 2 -- the pack-aware negative (THE FIX; build first)

### 2a. A new pack field `negative_tail`, KNOWN but OPTIONAL

**The trap that dictates this shape:** `get_visual_style`
(`_otr_visual_styles.py:541-567`) pulls `embedded_visual_style_pack` out of a
FROZEN ledger, runs `validate_pack` on it, then sha256s the embedded dict
against `visual_style_receipt["sha256"]`. A new REQUIRED key therefore bricks
resume/re-render of every existing `visual_storybased` episode -- and injecting
a default before validation changes the canonical bytes and trips the hash
instead. The receipt structurally forbids back-compat defaults.

**Resolution: split KNOWN from REQUIRED.** Today one dict does both jobs --
`validate_pack` computes `unknown` against `_REQUIRED_FIELDS` (`:338`) and
`missing` against the same dict (`:343`). Introduce `_KNOWN_FIELDS` (superset)
for the unknown check; leave `missing` on `_REQUIRED_FIELDS`. Then:
* the nine shipped packs may carry `negative_tail` (not rejected as unknown),
* old embedded packs without it still validate, bytes untouched, sha intact,
* `VisualStyle` carries `negative_tail: str = ""`, and empty means "this pack
  expresses no style negative" -- today's behaviour for every legacy ledger.

This also removes any need to touch `_V2_EMPTY_LEGAL_STR_FIELDS`, and it keeps
the unknown-key guard intact for genuine pack typos.

**Sites that land together:**
1. `VisualStyle` frozen dataclass (`:139-166`) -- new field with default LAST.
2. `_KNOWN_FIELDS` (new) + `_REQUIRED_FIELDS` unchanged (`:104-108`).
3. `validate_pack`'s explicit `VisualStyle(...)` constructor (`:422-450`) --
   pass `negative_tail=raw.get("negative_tail", "")`, or it raises `TypeError`
   on every pack load.
4. The `unknown` check at `:338` -- point it at `_KNOWN_FIELDS`.
5. All **nine** JSONs in `nodes/visual_styles/`.
6. `compose_pack_from_card` (`:259-300`) -- may omit the optional key, but
   emits a derived value for clarity.

**Values.** These three keep TODAY'S STRING VERBATIM, byte-identical:
`sci_fi_radio`, `shakespeare_stage_realism`, `archival_documentary`.
The six mis-served packs get negatives that never veto their own
`positive_tail`; every pack keeps "text, watermark" (anti-artifact, not
anti-style).

### 2b. Close the dropped request channel

`z_image_turbo._zimage_params` (`:216-219`) reads env-or-hardcoded and ignores
`request["negative_prompt"]` entirely -- so the announcer's
`radio_host_negative` is computed, safety-merged, carried, and **discarded**
today. Same in `lumina_image.py:105`.

**Composition happens in the DISPATCHER, into the EXISTING field.** Node 91
resolves `vstyle = get_visual_style(ledger["meta"])` and builds

    negative_prompt = visual_safety_negative(
        ", ".join(filter(None, [vstyle.negative_tail, obj_negative])))

then the engines simply HONOUR `request["negative_prompt"]`, with
`OTR_ZIMAGE_NEGATIVE` / `OTR_LUMINA_NEGATIVE` surviving as explicit overrides
only. One field, no new request keys (`ImageRequest` is a `_Forbid` model), no
schema change, and the existing dedupe collapses duplicates.

COMPOSE, never precedence: the pack negative is about STYLE and the request
negative is about FACELESSNESS on announcer stills (`radio_host_negative`); any
ordering silently drops one. Do not clobber `RADIO_CONSOLE_NEG` or
`MESH_FODDER_NEG_SCAFFOLD`.

**THE TRAP THAT WOULD SHIP THIS GREEN AND COSMETIC:** if the pack negative were
MERGED ON TOP of the engine default, `"cartoon, illustration"` would survive in
the default and still fire at cfg 2.0. The pack negative REPLACES the
hardcoded constant; the style half moves INTO the packs. This gets an explicit
regression test, not a comment.

**Out of scope, recorded not built:** `flux_gen1` (inert at cfg 1.0),
`sd35_large`, `hidream_i1`, `flux2_klein`, and the cloud engines (no negative
conditioning at all). The report must never imply nine engines got styled
negatives.

## Part 1 -- the style pass (hygiene; one choke point)

In `OTR_ImageGenDispatcher` (node 91), the only node that sees both families.

**The predicate is POSITIONAL, not membership.** `finish_visual_prompt` already
appends `positive_tail` LAST (`_otr_story_brief_helpers.py:634-637`), so the
token is present but in the weakest position on models that weight early
tokens. "Prepend where absent" would find it present and do nothing on exactly
the prompts that fractured. Copy `_prefix_video_style_cue`
(`render_driver.py:1389-1411`) which tests `startswith` -- front-anchored by
construction -- including its "cue minus trailing ' style'" case. Prepending
while the tail copy stays put is purely ADDITIVE, so
`otr_shot_lock.py:915-919` holds (its very next line is itself an additive
prepend).

**Two families, two loops:**
* **Stills:** the `objects` from `image_prompts_json`, styled BEFORE
  `_prompt_content_hash` (`otr_image_gen_dispatcher.py:1043`). Banana runs at
  `:1034-1038` and hashing at `:1043`, so a prepend before that is hashed with
  the text actually rendered -- **no restamp needed on this family.**
* **Shots: NOT TOUCHED, deliberately.** An earlier cut of this plan prescribed
  a pre-loop over `ledger["video"]["shots"]` plus a `prompt_hash` restamp.
  **Do not build it.** Two reasons, both verified:
  1. `render_driver` already front-anchors the IDENTICAL token at five call
     sites (`:2790`, `:2825`, `:2830`, `:2970`, `:3054`), so video coverage is
     already 100%. Adding a node-91 prepend would DOUBLE it: the ia2v
     talking branch (`render_driver.py:2744-2764`) slices a raw
     `text_prompt[:N]` fragment with no idempotence check, yielding
     "Cartoon style Cartoon style. ..." -- and the later idempotent prefix call
     would then see a string that already starts with the cue and preserve the
     double.
  2. The restamp was never needed anyway. Nothing downstream reads a shot's
     `prompt_hash` for identity: the seed and cache key use
     `cache_keys.request_hash`, derived at `otr_shot_lock.py:941-943` from
     `[brief_hash, cast_hash, beat_id, char_id]` and never from prompt text.

**`sci_fi_radio` is EXEMPT from part 1** -- byte-identity goldens pin the
default lane (`tests/test_visual_styles_3a.py`), and defaulting-out has two
precedents already: `_compact_style_talking_cue` (`render_driver.py:1372`) and
`_mesh_style_spice` (`otr_meta_brief_image_prompt.py:1612`).

**Reconcile with the existing prepender.** `_prefix_video_style_cue` already
runs at five sites (`:2790`, `:2825`, `:2830`, `:2970`, `:3054`). ONE
derivation of "the style token" and one prepender -- three derivations already
exist and a fourth would disagree with all of them.

**Carry the punctuation fix.** `render_driver.py:1406-1410` has a real latent
bug: a prompt opening `"Cartoon, a man..."` yields `"cartoon style. , a
man..."`. Use `rest.lstrip(".,:; ")` and fix the shipped helper too.

**Known cost, stated not discovered:** `prompt_hash` feeds the SEED
(`resolve_seed_and_mode`, `:1181-1184`) and the cache key (`:1191-1193`), so
styling re-mints cached stills once. That is the correct outcome -- a stale
still minted under the cartoon-vetoing negative must not be served.

## Part 3 -- the style-spread measurement (TELEMETRY, never a gate)

One scalar: Laplacian variance. **Reuse the proven implementation at
`scripts/run_ltx_av_q_bakeoff.py:487-500` but note its shape:** it expects 4D
`[N,H,W,C]` and indexes `frames.shape[0]`; a single still is 3D `[H,W,C]` and
would slice HEIGHT as frame count. Wrap as `[1,H,W,C]` or use a single-frame
helper.

* Compute per still right after pixels decode; store on the record.
* Aggregate ONCE after the loop -- never inside it.
* Compare RELATIVE to the episode's own median, never an absolute constant
  (the q-bakeoff uses ">=15% over baseline", `:46-49`).
* **Stratify by `kind`** and exclude `source == "still_word"` cards: word
  cards, abstract music stills, plates, mesh fodder and character scenes are
  structurally different populations.
* Stamp scalars + spread into `stills_manifest.json` (`:1567`, fail-soft) and
  into the visual ledger.
* Portrait object rows do not currently carry `visual_style` (`:1902-1911`) --
  stamp it so a still can be attributed to a pack.
* **On exceed: a LOUD warning + a receipt at mint time, before video GPU.
  Never a refusal, never a re-roll, never a re-mint.** THE LAW forbids failing
  OR rerolling a story for style or visual vocabulary, and the fail-closed
  carve-out is structural only. The repo already implements telemetry-only
  (`OTR_LedgerScriptWriter.py:3234-3236`).
* It cannot see procedural `viz_*` bookends, so it audits a SUBSET of the
  fracture surface -- say so in the receipt.
* Calibrate on two anchors: the broken episode FLAGS, a known-good one is
  quiet.

Palette-reconstruction error is **CUT** -- no in-repo precedent, and one
proven scalar beats two unproven ones.

## Part 4 -- THE VISUAL LEDGER (operator-authorized 2026-08-17)

> *"if we need to store the prompts at some point in a mini ledger or something
> or in an expanded ledger schema I am fine if that's the best approach -- a
> prompt ledger or prompt schema -- visual ledger."*

It is the best approach. Today the final visual prompts are transient: still
prompts exist only on a wire and are never persisted post-transform, so nothing
on disk records what was actually rendered, and parts 1 and 3 would otherwise
emit two unrelated receipts.

**Mechanism -- additive, no migration, no new failure site.** The production
ledger is a plain dict; `stamp_durable(sections=...)` replaces top-level keys
wholesale (`production_ledger.py:527-552`). Node 91 ALREADY calls it at
`otr_image_gen_dispatcher.py:1533-1538`, so `visual` rides that SAME call:

    stamp_durable(
        sections={"images": ledger["images"], "visual": ledger["visual"]},
        meta_updates={"image_engines": ...},
        source="image_dispatcher",
    )

**Shape** (`ledger["visual"]`): `style_id`, `style_token`, `pack_negative`,
`negative_source` (`pack` | `pack+request` | `env_override`),
`authority_version`; `prompts[]` with one row per visual prompt across BOTH
families (`kind`, `object_id`|`beat_id`, `role`, `source`, `styled`,
`already_styled`, `prompt_sha8`, `laplacian`); and `spread`
(`metric`, `median`, `max_pairwise`, `threshold`, `exceeded`, `excluded`).

No separate sidecar file -- the ledger section IS the durable audit trail.
This ADDS a section and removes no pass, so no existing field changes owner.

## Part 5 -- THE TRACEROUTE (operator-required 2026-08-17)

> *"yes I realize it touches all still packs and all video packs so will need a
> good traceroute."*

`scripts/otr_style_traceroute.py` -- read-only, renders nothing, loads no
model, spends no GPU, wired into no node, and NOT in the canonical workflow.
For each of the ten style identities it prints every surface carrying that
pack's style and the effective negative each live engine would use.

**The style surface inventory it must cover** (from `_otr_visual_styles.py:75-102`):
* STILL: `positive_tail`, `image_grade_tail`, `broadcast_tail`, `era_tail`,
  `portrait_look`, `portrait_look_talking`, `portrait_instruction_look`,
  `scene_instruction_look`, `plate_look`, `announcer_subject_face`,
  `announcer_subject_ltx_mouth`, `announcer_subject_object`,
  `radio_object_look`, `still_word_title_mood_style`,
  `still_word_typography` (5), `still_word_backdrop` (5),
  `non_character_emblem_fallback`, `open_subjects` (3).
* VIDEO: `motion_registers` (4, 240-char budget) and `_prefix_video_style_cue`
  at five call sites.
* NEGATIVE: per engine, cfg-gated.

**The three questions it answers:**
1. Coverage -- every surface accounted for per identity, and which are
   legally empty by design.
2. **The fight** -- does a pack's own negative contradict its own
   `positive_tail`? Today it must report the FOUR literally mis-served packs; after the
   build it must report ZERO. That before/after flip is the build's proof.
3. Byte-identity -- the three photoreal packs resolve to the HISTORICAL
   z_image string, character for character.

It is not a classifier and not a gate: it prints a report and exits 0.

## Acceptance (Fable's test, adopted verbatim)

> Mint the same cartoon episode's character still on z_image before and after:
> the request's effective negative must contain no illustration-family terms,
> and the effective positive must lead with the pack token.

Plus: the traceroute reports **EFFECTIVE FIGHTS: 0** (authored conflicts may be
non-zero and are resolved at mint -- see part 6), and the suite is green from a
settled tree.

**Part 3's threshold is DEFERRED, not delivered.** `threshold` and `exceeded`
ship as `null`: the scalars are recorded and the spread is computed, but no
calibration run has yet proved Laplacian variance separates the broken episode
from a known-good one. Until it does there is no threshold and no warning, and
the receipt says so. Do not report part 3 as "done" -- it is "measuring".

## Part 6 -- NO NEGATIVE MAY CONFLICT WITH A STYLE (operator ruling 2026-08-17)

> *"yes that's a bug, we can't have negative prompts conflicting with any
> visual style."*

Moving the negative into the packs fixed the CROSS-pack case. It left the
SELF-veto case: `sci_fi_radio` declares "cartoon, illustration" while its own
`announcer_subject_ltx_mouth` asks for "a living cartoon appliance face" and
its `still_word_title_mood_style` asks for "atmospheric period illustration".
The traceroute found it; it predates this build and nobody had measured it.

`_otr_visual_styles.effective_negative(style)` drops any negative phrase the
pack's own positive surfaces ask for, comparing comma-separated PHRASES so
"plastic skin" survives a portrait that wants realistic skin. Anti-artifact
terms ("text", "watermark") are never dropped.

Fixing it here rather than by hand-editing one string is the ROOT fix: it holds
for all ten identities, for the dynamic pack composed at runtime, and for any
pack authored later. A pack cannot veto itself by construction. It reads our
own CONFIG against itself and edits our own NEGATIVE -- it never inspects or
alters an authored prompt, so it is not the forbidden prompt-scanning injector.

The authored string is preserved, and the ledger records both
(`pack_negative_authored`, `pack_negative`, `self_veto_resolved`).

## Invariants (violate one and the change is wrong)

The three photoreal packs byte-identical. No classifier at node 89 and its
ruling comment stays. THE LAW -- telemetry may never fail or reroll a story for
style. No content guardrails, and specifically do not reintroduce a safety
negative. No 188/620 rebalance, no `style_tail=True` on character video
prompts, no touching the kinetic clause or the probe-locked talking clause, no
prompt-scanning conditional injector. Add NO widget and no `INPUT_TYPES`
change (`widgets_values` is positional -- BUG-LOCAL-097). Root cause only, no
shims. Every ledger field keeps exactly one owner. UTF-8 no BOM, ASCII where
practical, clean names, never "dummy".

## The ia2v talking-portrait lane -- what is and is NOT built

`otr_meta_brief_image_prompt.py:1850-1852` skips `finish_visual_prompt` and the
grade tail for the lip-sync portrait, giving it only "warm dramatic lighting".
It is a real latent style hole on ia2v lanes. The measured episode ran
`still_flat` with talking=False, so it explains none of THIS artifact.

**NOT built:** we do not edit `:1850-1852`, and we do not reopen its S4b
darkness rationale. Widening a root-cause fix into an unmeasured lane upstream
is how a green build hides a regression.

**BUILT, deliberately:** part 1 runs at node 91 and styles every still prompt
that reaches it, portraits included. The skip happens upstream, so this covers
that lane's OUTPUT for free without touching its code. Portraits are
explicitly IN scope -- guarding them out would exempt exactly the beats that
fractured (the 00:34 character still is portrait-derived, and
`reference_latent` propagation from the portrait is the documented amplifier).
