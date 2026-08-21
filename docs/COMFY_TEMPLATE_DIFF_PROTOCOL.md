# Comfy Template Diff Protocol

Use this protocol when adapting a ComfyUI reference workflow into an OTR media
engine. It records the LTX 2.5 two-stage lesson as one worked precedent, not as
a universal recipe for other models.

## Decision rule

`NO APPROVED ARTIFACT -> NO AUTHORITY TO ENTER THE SHIPPING GRAPH`

Prefer failure-driven scope over speculative hardening: ship the smallest path
supported by accepted evidence, observe the real OTR result, and fix concrete
failures. Do not add mechanisms or review lanes merely because they might help.

This rule is a universal **admission boundary**, not a universal classifier. A
SHA-256 proves which exact source bytes were reviewed; it does not prove that the
LTX 2.5 reasoning transfers to Wan, Hunyuan, Humo, MiniMax, or even to another
LTX transport. Every engine/model/transport combination needs its own completed
diff and decision receipt.

While that complete diff is still being enumerated, a material difference is
**UNCLASSIFIED**. Do not prune it because another model rejected a similarly
named node, port, or setting. Once enumeration is complete, every difference
belongs to exactly one engine-local bucket:

1. **IN** -- the node, literal, port, or wire is on the causal path that produced
   an operator-approved artifact for this engine and transport. Preserve its
   semantics exactly.
2. **ADAPT** -- OTR changes representation without changing recipe semantics.
   Examples: UI constant nodes become literals; reference save nodes become
   OTR's encoder; OTR retains role stills, role prompts, terminal-frame chaining,
   silence/master-audio ownership, receipts, and the final delivery canvas.
3. **OUT** -- the element lacks quality authority for this engine and transport.
   It stays visible in the receipt and may become a separate candidate, but it
   does not enter the shipping graph until it produces its own accepted artifact.

“Official,” “newer,” and “looks helpful” are weaker evidence than “this exact
path caused the output the operator approved.” A template can confirm installed
classes and port contracts; it cannot silently redefine the selected recipe.

## Byte-level review sequence

1. Freeze the reference workflow bytes, accepted recipe if one exists, OTR
   engine source/topology, model filenames, artifacts, and receipt hashes before
   editing. Record package/version and a SHA-256 for every source.
2. Normalize only serialization noise such as node IDs and UI positions.
3. Compare every executed class, input literal, output slot, and wire.
4. Finish the enumeration before narrowing the candidate list. Mark every
   difference IN, ADAPT, or OUT for this engine with its evidence source and
   reasoning; retain rejected differences in the receipt.
5. Query the running server's `/object_info` for the actual installed signatures;
   a downloaded graph does not override the classes registered on this machine.
6. Preserve OTR's request and delivery contracts: unique role assets/prompts,
   audio ownership, multi-clip continuity, frame count, color/container rules,
   and downstream canvas behavior.
7. Add executor-owned positive evidence for expensive or easily dormant stages.
   Adapter plan/pass messages are summaries, not proof of execution.
8. Run one independent finished-diff review. If it is clean and grounded, stop
   reviewing. Fan out only for a blocker, disagreement, unverifiable material
   claim, or the repository's two-strikes rule.
9. Prove the full OTR path with a fresh `otr/obs` publish; isolated lab output is
   evidence for a recipe, never evidence that OTR integration shipped.

The resulting engine-local decision receipt is the re-blend guide. It must bind
the exact reference and OTR hashes, installed model/transport identities, the
normalizer/differ version, the complete difference set, each IN/ADAPT/OUT ruling,
and any accepted artifact hashes. Never inherit a ruling solely from the LTX 2.5
precedent or from a matching class name.

## Visual-payoff gate

Only after the complete diff may GPU time be spent on candidate differences
selected from the receipt; that may include retesting an OUT element as a
separate candidate. Hold the still, prompt, seed, canvas, frame count, delivery
path, and every unrelated variable fixed; change one difference at a time, or
one irreducible topology when the graph cannot be split honestly.

Give an AI vision judge matched native-pixel stills or crops in both presentation
orders. Ask countable questions -- resolved facial features, readable
letterforms, seams, preserved small objects, identity at named frames -- rather
than "which looks better." Record the visible payoff alongside render time,
VRAM, integration complexity, and judging cost. No or marginal visible payoff
ends that candidate; material visible payoff makes it eligible for operator
acceptance and the live OTR proof, not automatically IN.

## LTX 2.5 precedent

This classification is binding only for the accepted LTX 2.5 two-stage use case
and its recorded transport. It illustrates the method; it is not the decision
logic for another engine.

- **IN:** latent x2 upscaler, same-still second anchor, refine sigmas, second
  sampler, and the only video decode at 1664x960.
- **ADAPT:** OTR role stills/prompts, silent clip encoder, master-audio mux,
  executor audit, and prior-segment terminal-frame chaining.
- **OUT:** FLF endpoint anchors, prompt enhancer, generated-audio decode, and the
  unpromoted A2V front-end. Each requires a separate recipe and acceptance.

For an ultra-smoke, count actual rendered segments rather than story beats. The
target design is two deliberately short story beats plus short opening/closing
music, covering music, announcer, and character in four 97-frame HQ renders.

## Z-Image Turbo counterexample

The same method produced the opposite recipe ruling for the installed Z-Image
Turbo transport:

- **IN:** the clean shared base skeleton -- model/text/VAE loaders, AuraFlow
  sampling, text conditioning, empty SD3 latent, eight-step sampler, and plain
  VAE decode -- as proven on the installed transport.
- **ADAPT:** the installed NVFP4/Qwen-FP8 transport, OTR role prompt/negative
  ownership and live CFG/sampler, exact requested still canvas, deterministic
  portrait-derived identity seed, receipts, and delivery path.
- **OUT:** the extra generic `LoadImage -> ImageScale -> VAEEncode -> dual
  ReferenceLatent` branch and both sampler rewires. The official base graph did
  not use that semantic path, and a matched fresh-boot OFF/ON A/B proved that ON
  recreated the square grid even though both arms executed successfully.

Both cases froze inputs, grounded the real graph, used native pixels rather than
success labels or a broken scalar, invalidated old caches, and closed with a live
OTR artifact. Yet the model-native LTX same-still re-anchor was IN while the
generic Z-Image reference injection was OUT. The protocol transfers; the ruling
does not.
