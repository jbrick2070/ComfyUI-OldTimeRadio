# Comfy Template Diff Protocol

Use this protocol when adapting a ComfyUI reference workflow into an OTR media
engine. It records the LTX 2.5 two-stage lesson so later model sweeps preserve
what caused the approved output without importing unrelated template behavior.

## Decision rule

`NO APPROVED ARTIFACT -> NO AUTHORITY TO ENTER THE SHIPPING GRAPH`

Prefer failure-driven scope over speculative hardening: ship the smallest path
supported by accepted evidence, observe the real OTR result, and fix concrete
failures. Do not add mechanisms or review lanes merely because they might help.

Every reference-graph difference belongs to exactly one bucket:

1. **IN** -- the node, literal, port, or wire is on the causal path that produced
   the operator-approved artifact. Preserve its semantics exactly.
2. **ADAPT** -- OTR changes representation without changing recipe semantics.
   Examples: UI constant nodes become literals; reference save nodes become
   OTR's encoder; OTR retains role stills, role prompts, terminal-frame chaining,
   silence/master-audio ownership, receipts, and the final delivery canvas.
3. **OUT** -- the element comes from another workflow or an experiment without
   quality approval. It may become a separate candidate, but it does not enter
   the shipping graph until it produces its own accepted artifact.

“Official,” “newer,” and “looks helpful” are weaker evidence than “this exact
path caused the output the operator approved.” A template can confirm installed
classes and port contracts; it cannot silently redefine the selected recipe.

## Byte-level review sequence

1. Freeze the accepted recipe, reference workflows, model filenames, artifact,
   and receipt hashes before editing.
2. Normalize only serialization noise such as node IDs and UI positions.
3. Compare every executed class, input literal, output slot, and wire.
4. Mark every difference IN, ADAPT, or OUT with its evidence source.
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

## LTX 2.5 precedent

- **IN:** latent x2 upscaler, same-still second anchor, refine sigmas, second
  sampler, and the only video decode at 1664x960.
- **ADAPT:** OTR role stills/prompts, silent clip encoder, master-audio mux,
  executor audit, and prior-segment terminal-frame chaining.
- **OUT:** FLF endpoint anchors, prompt enhancer, generated-audio decode, and the
  unpromoted A2V front-end. Each requires a separate recipe and acceptance.

For an ultra-smoke, count actual rendered segments rather than story beats. The
target design is two deliberately short story beats plus short opening/closing
music, covering music, announcer, and character in four 97-frame HQ renders.
