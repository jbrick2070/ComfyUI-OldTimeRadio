# Visual styles -- how the episode looks

`visual_style` picks the look. It rewrites the prompts that mint the still
images **and** the prompts that drive the video, so one dropdown changes the
character portraits, the scene plates, the announcer's radio console, the title
cards and the way things move. Every style does both; none of them is
stills-only.

It never touches the story. Same script, same cast, same lines -- drawn a
different way.

One thing before the table, because it is the most common surprise:

**On the canonical workflow as shipped, this control changes nothing you can
see.** The three video lanes the canonical selects -- `viz_mxc_cpu`,
`viz_green` and `viz_camera` -- draw their own frames from the audio. They read
no image and write no prompt, so there is nothing for a style to dress. Switch
a video lane to one that draws a picture and this becomes one of the strongest
controls on the graph. [Where you actually see it](#where-you-actually-see-it)
says which shipped graph is which.

---

## The styles

The dropdown lists style *ids*, alphabetically, with the roll sitting on top.

| Style | The look |
|---|---|
| `anime` | Hand-drawn cel animation. Clean expressive linework, cel-shaded colour, dramatic rim light, painted anime backgrounds. |
| `archival_documentary` | A restored archive photograph. Period-accurate costume and set, grounded natural light, tactile paper and film texture under a subtle patina. |
| `cartoon` | Rubber-hose cartoon. Bold clean outlines, flat cheerful colour, simple friendly shapes on flat colourful sets. |
| `paper_origami` | A folded-paper diorama. Characters are paper puppets with crisp crease lines and layered papercraft texture, under soft studio light. |
| `recur_frac` | Recursive fractal light field. Nested self-similar geometry, emissive shader surfaces, volumetric bloom on luminous black -- laser cyan, spectral magenta, acid green, ultraviolet blue. |
| `sci_fi_radio` | The house look. Cinematic 35mm: film grain, volumetric light, anamorphic lens, heavy vignette, muted grade, period-accurate sets. |
| `shakespeare_stage_realism` | Photoreal theatre. A candlelit period stage, real costume detail, rich stage contrast, warm candle colour, composed like a live production. |
| `storybook_engraving` | A hand-tinted engraved plate. Fine ink crosshatching, hand-coloured costume, warm parchment ground. |
| `video_art` | Analogue video feedback. Phosphor bloom, prismatic halos, saturated additive rainbow over velvety charcoal darks, layered temporal planes. |
| `visual_storybased` | Not a fixed look -- see [the story-based one](#the-story-based-one-is-not-a-look) below. |

`sci_fi_radio` is also the fallback: an episode whose ledger carries no style at
all is drawn this way, so it is the one look you can get without ever choosing
it.

---

## roll (any style)

`roll (any style)` sits at the top of the dropdown, and every graph that ships
with this pack -- the canonical and every variant -- is saved on it. It
is a command, not a style: at run time it draws one of the ids above at
equal odds, and the episode uses that.

Drop a fresh **OTR_LedgerScriptWriter** on the canvas yourself and it starts on
`sci_fi_radio` instead. Only the saved graphs ship on the roll.

Two consequences worth knowing:

- **A roll is not a fair comparison.** If you are judging any other change,
  pin the same style on both runs or you are looking at two different shows.
- **The roll can land on `visual_storybased`**, which behaves differently from
  a pack loaded from disk.

---

## Where you actually see it

A style is carried by a *prompt*. Lanes that write a prompt wear it; lanes that
draw their own frames from the audio do not.

| Shipped graph | Its video lanes | Does the style show? |
|---|---|---|
| `otr_canonical`, `otr_16gb_low`, `otr_8gb_low`, `otr_mac16_low` | `viz_mxc_cpu`, `viz_mxc_mandala`, `viz_green`, `viz_camera` | **No.** Audio-reactive; no image, no prompt. |
| `otr_16gb_still`, `otr_amd_still` | two audio-reactive plus `still_motion` | **On the character beats only.** |
| `otr_8gb_still` | `still_flat`, `viz_green`, `still_motion` | **On the announcer and character beats.** |
| `otr_mac16_still` | `still_motion` throughout | **Yes, everywhere.** |
| `otr_16gb_video`, `otr_8gb_video`, `otr_mac16_video`, `otr_16gb_foley`, `otr_16gb_mime` | the LTX lanes | **Yes, everywhere.** |
| `otr_16gb_animatediff`, `otr_8gb_animatediff`, `otr_mac16_animatediff` | the AnimateDiff lanes | **Yes, everywhere.** These mint no still but write their own styled prompt and their own negative. |

[MACHINES.md](MACHINES.md#which-graph-do-i-open) names the variant file for your machine.
If you are on the canonical and want to see a style, the change you want is on
**OTR_VideoDirector**: switch `announcer_video_model`, `music_video_model` and
`character_video_model` off the `viz_` lanes. Check the lane against your
hardware in [MACHINES.md](MACHINES.md#will-this-engine-run-on-my-machine) first.

---

## The story-based one is not a look

`visual_storybased` has no pack on disk. Instead the writer reads the finished
script and invents a look for that one episode -- a nine-field card covering the
art medium, what the radio console is made of, how characters are drawn,
texture, linework, lighting, lettering and how surfaces move -- then builds a
full style pack out of it and freezes it into the episode's ledger.

- **It does not add a pass.** The same reflection step runs on every episode;
  picking this one just asks it for more (a 1024-token budget instead of 512).
- **If that pass fails**, the episode does not fall back to a generic look and
  does not stop. It draws another shipped pack, copies that pack, and records
  what happened -- `status: "floor"` with the style it borrowed. The dynamic
  lane can never be re-selected as its own fallback.
- **It is the one style that can differ run to run** with the same dropdown
  setting, because it is written fresh each time.

---

## The pool is not scoped to the bank

Every one of them is available to every source bank. The roll draws from the
same pool whichever bank writes the episode, so yes, a Shakespeare adaptation can
come out as a cartoon. If you do not want that, pin the style.

What *is* scoped per bank is a different thing with a confusingly similar name:
each bank draws its **radio grammar** -- the sound world, the shape of the
conflict, how the episode ends -- from a pool chosen by that bank. The
adaptation banks (Shakespeare, Public Domain) draw only from source-deferential
grammars so Macbeth stays on the heath, and the archive bank has its own
curated set. That is not a dropdown, you do not pick it, and it has nothing to
do with `visual_style`. [BANKS.md](BANKS.md) does not use the word "grammar"
-- it covers the same adapt-versus-invent split these pools follow, described
in its own terms.

---

## One style asks for a download

`anime` names an SD-1.5 checkpoint it would rather be drawn with:
`Counterfeit-V3.0_fp16.safetensors`, about 4.2 GB. **Nothing fetches this for
you.** Put it in your `models/checkpoints/` folder and it is used
automatically; leave it out and nothing breaks.

That is the whole contract, and it is deliberate: the file is a *preference*,
never a requirement. The still engine takes it only when ComfyUI can actually
see the file, and otherwise loads its normal checkpoint
(`v1-5-pruned-emaonly-fp16.safetensors`, which does download itself). A pack
naming weights you do not have costs you nothing -- no grey-out, no refusal, no
failed render.

Two more notes on it:

- It only applies when the still is being minted by the `sd15` image engine.
  The canonical's image engine is `z_image_turbo`, which ignores the field.
- The environment variable `OTR_SD15_CKPT` overrides the engine's default for
  every style, but a pack's own checkpoint wins over it.

No other shipped style names a checkpoint, and no style requires any download of
its own.

---

## Which style did I get?

The published filename says so. The first code after the title and timestamp is
the style:

```
the_clanking_chains_20260912_153041__anim__stfl__sd15__..._final.mp4
```

`anim` is `anime`. The codes are four characters: `anim`, `arch`, `cart`,
`pori`, `rfrc`, `scif`, `shst`, `sbke`, `vart`, `vstb`. The name always records
the style that actually ran, never the word "roll".

The closing credits also list which dropdowns rolled rather than being picked,
under **Rolled:** -- so a style you did not choose announces itself on screen.

---

## When it comes out wrong

Nearly always one of these, in this order:

- **You picked a style and the picture did not change.** Your video lanes are
  audio-reactive. See [Where you actually see it](#where-you-actually-see-it).
  This is the answer far more often than anything else on this page.
- **You were on the roll and did not notice.** Every shipped graph is. Read
  the code in the filename.
- **The look changes between two runs on the same setting.** You are on
  `visual_storybased`, which is written fresh each episode by design.

A style you typed wrong does not silently fall back to something else -- it
stops the run and names the ids that exist.

---

## Adding your own style pack

This one is easier than it looks, and easier than adding a bank or an engine,
because a style pack is **data, not code**. There is no registration step, no
Python, and no consent flag: a valid file in the right folder is a style.

Packs live inside the installed pack, one flat file each:

```
custom_nodes/ComfyUI-OldTimeRadio/nodes/visual_styles/<your_style_id>.json
```

### How to add one

- **Copy the shipped pack closest to what you want.** Do not start from a
  blank file -- every key is required and the validator is strict.
  `cartoon.json` is the plainest; `sci_fi_radio.json` is the house look.
- **Rename the file.** The filename *is* the id: `pulp_woodcut.json` must
  declare `"style_id": "pulp_woodcut"`, and a mismatch is refused by name.
  Lowercase letters, digits and underscores only.
- **Rewrite the values, keep the keys.** Leave `"schema_version": "v2"` alone.
- **Restart ComfyUI.** The folder is read once, at startup.

Your id is then in the `visual_style` dropdown, in alphabetical order, and in
the roll at the same odds as everything else.

### What the validator insists on

It reads the whole folder at startup and refuses loudly, naming the file, if
anything is off. The ones that catch people:

- **Every key must be present, and no extra keys are allowed.** Presence and
  non-emptiness are different rules, though. The newer look/subject fields --
  the look/subject fields you rewrite -- must be non-empty,
  with one exception: `scene_instruction_look` may be an empty string. The
  older tail fields (`positive_tail`, `image_grade_tail`,
  `broadcast_tail`, `era_tail`) and `label` must be present but are never
  checked for emptiness -- `cartoon.json`,
  `anime.json` and `paper_origami.json` ship today with `image_grade_tail`,
  `broadcast_tail` and `era_tail` all set to `""`, and that validates fine.
  `negative_tail` (what the style should *avoid* drawing) and `checkpoint` are
  optional outright -- leave them out of the file and nothing complains.
- **The placeholders are exact.** `{form}` appears exactly once in
  `announcer_subject_ltx_mouth` and once in each of the three `open_subjects`
  values; `{base}` appears exactly once in `non_character_emblem_fallback`. No
  other braces anywhere.
- **`announcer_subject_ltx_mouth` must mention a mouth or lips.** The lip-sync
  lane animates whatever reads as a mouth, so a console with no mouth in its
  description has nothing to move.
- **The dictionary keys are fixed.** `open_subjects` takes exactly `synthetic`,
  `announcer`, `default`. `motion_registers` takes exactly `announcer`,
  `music_open`, `music_close`, `music_inter`, each 240 characters or fewer.
  `still_word_typography` and `still_word_backdrop` each take exactly `noir`,
  `sci-fi`, `western`, `pulp`, `default`.
- **The folder holds flat `.json` files and nothing else.** A subfolder, a
  `.bak`, or an editor's leftover will stop the whole dropdown from building.
  If the writer node breaks right after you added a pack, look there first.

### The one real catch

That folder belongs to the installed pack, so **updating OTR through the Node
Manager replaces it and your pack goes with it.** Keep the file somewhere of
your own and copy it back after an update. There is no user-packs directory for
styles the way there is for source banks.

Adding a **source bank** or an **engine** is a different and larger job with its
own steps -- [EXTENDING.md](EXTENDING.md) covers both. Nothing there applies to
a style pack, and none of it is needed for one.
