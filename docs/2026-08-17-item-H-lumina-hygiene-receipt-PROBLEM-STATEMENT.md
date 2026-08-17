# Item H -- the lumina hygiene floor and the `engine_hygiene` receipt

**One question, scoped.** Not a four-round arc. HEAD `9804f7a2`, branch
`v2.0-alpha`. Everything below was verified at the real files on 2026-08-17;
where a claim is inherited from `docs/GO_FORWARD_PLAN.md` rather than re-checked,
it says so.

## The defect as originally recorded

`GO_FORWARD_PLAN.md` item H says: `z_image_turbo._resolve_negative` ends
`.strip() or _HYGIENE_NEGATIVE`, while `lumina_image` has neither the strip nor a
floor -- so an empty request negative reaches the encoder as `""` and a
whitespace-only one is passed verbatim. The path is reachable
(`VISUAL_SAFETY_NEGATIVE_PROMPT` is `""`, and a pack may ship an empty
`negative_tail`). The sharp end is a receipt that lies: the dispatcher stamps
`_neg_source="engine_hygiene"` for exactly that case, so the ledger claims a
hygiene floor this engine does not have.

The recorded instruction is: *either give lumina a floor, or make the ledger stop
claiming one it does not have -- but do not leave the receipt wrong.*

## Two corrections found while storing the prompt-style overlays

**1. There is no `lumina_image._resolve_negative`.** That name exists only in
`z_image_turbo`. Lumina resolves its negative INLINE inside `_lumina_params`, as

```python
"negative": (_env_neg if (_env_neg := os.environ.get(
    "OTR_LUMINA_NEGATIVE")) is not None
    else str(get("negative_prompt") or "")),
```

No `.strip()`, no hygiene constant, and no such constant exists anywhere in the
file. The BEHAVIOUR item H describes is real; the function it names is not. A
plan that says "add the floor to `_resolve_negative`" targets a ghost.

**2. The receipt is not wrong for lumina. It is UNVERIFIED FOR EVERY ENGINE.**
This is the load-bearing new finding. In `otr_image_gen_dispatcher.py` the label
is computed as:

```python
_neg_source = ("pack+request" if _pack_negative and _obj_negative
               else "pack" if _pack_negative
               else "request" if _obj_negative
               else "engine_hygiene")
```

and `engine_id` for the row is not bound until `resolve_engine_for_role(...)`
roughly **56 lines later in the same per-object iteration** (the whole block sits
at one indent level inside a single `for` loop). So the `engine_hygiene` arm
asserts a property of an engine the code has not chosen yet. It is true of
`z_image_turbo` by COINCIDENCE -- that engine does have a floor -- and false of
`lumina_image`. In neither case was anything consulted.

The other three arms are honest: they describe contributions the dispatcher
actually observed (`_pack_negative`, `_obj_negative`). Only the fourth makes a
claim about an engine.

## Blast radius, measured rather than assumed

* `engine_hygiene` appears in exactly **two code sites**: the dispatcher comment
  explaining the label, and the value itself. Everything else is docs.
* The value is written to the ledger under key `"negative_source"` at
  `otr_image_gen_dispatcher.py:1413` and `:1608`.
* **No on-disk ledger carries it.** A scan of **4,795** JSON files under the real
  ComfyUI output base found ZERO occurrences of `negative_source`, and also zero
  of its siblings from the same 2026-08-17 change (`self_veto_resolved`,
  `_style_spread`). The scan is sensitive: it finds `visual_style` in 770 files
  and `prompt_hash` in 1,022. So the field has never been written to disk and no
  episode has been rendered since it landed.
* Corollary worth its own line: the whole `visual` ledger section from the
  one-style-authority work is **unproven on a live render**. That sharpens D-BIS
  finding 5 from "zero tests cover the ledger recording" to "zero tests AND zero
  live observations".

## The two options, now asymmetric

**Option A -- fix the RECEIPT (driver-sized, zero pixels).** Stop the fourth arm
asserting engine behaviour the dispatcher cannot see; name it for what is
actually known (no composed negative contributed). No recipe touched, no render
owed, and with no historical ledgers carrying the old value there is nothing to
become inconsistent with. A genuinely engine-aware label is a DIFFERENT and
larger job: it requires moving the stamp after engine resolution, which is a
production reordering.

**Option B -- give `lumina_image` a hygiene floor.** This changes conditioning on
a live engine at cfg 4.0 (z_image runs 2.0, and it is a different model with its
own artifact profile). The standing operator directive is that the recipes are
not on the table, and the standing trap is that green gates are not a working fix
-- budget a render whenever a negative changes. The existing inline comment in
`lumina_image` already defers this explicitly: *"Whether this engine should grow
its own floor is a RENDER decision on a different model, not a comment fix; it is
logged, not folded in here."*

## What the panel is asked to break

1. Is Option A actually safe, or is renaming a ledger value a contract change
   that downstream consumers (TTS, per-beat slicing, video/shot direction,
   captions, credits, `obs_publish`) can feel even with no historical rows? The
   project rule is that consumers read FIELDS, not intentions.
2. Is "no composed negative contributed" the right name, or does the arm need to
   distinguish *no negative was composed* from *the engine will supply a floor*?
   If the latter, does that force the stamp to move after engine resolution --
   and is that reordering safe in a loop that also computes cache keys, seeds and
   the banana transform?
3. Is there a third option neither the plan nor the driver has seen -- for
   example, dropping the arm entirely, or recording the resolved negative TEXT
   plus the cfg (D-BIS finding 4 asks for the cfg, since at cfg 1.0 a logged
   negative conditioned nothing)?
4. Does Option B belong to the operator at all, or is a hygiene floor so clearly
   an anti-artifact default that it is not a "recipe" in the protected sense?
5. What is the cheapest LIVE observation that would prove any of this, given the
   ledger section has never been written to disk?

## Invariants a fix may not break

* A negative may never conflict with a visual style (operator ruling). The pack
  owns the style negative; the engine owns only hygiene.
* The recipes are not on the table; measurement runs the shipped recipe unchanged.
* A ripped or repurposed pass must still fill the ledger completely -- every
  field gets exactly one owner.
* Part 3 style telemetry has `threshold: null` and may never gate or reroll.
* No content guardrails on generated episodes.
