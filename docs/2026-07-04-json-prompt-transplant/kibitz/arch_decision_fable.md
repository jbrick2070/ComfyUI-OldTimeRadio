# Fable verdict -- Transplant (T) vs In-repo (R)

## VERDICT
**R.** The production_mirror is the tell: infrastructure whose only job is to simulate being in the same repo means the split is accidental, not architectural.

## Why
The operator's end-goal is selectable lanes as DATA inside the production pipeline. Every gram of that goal lives in ONE place at runtime: OTR nodes loading JSON packs, failing loud on unknown ids. The sibling contributes nothing to that goal that a `story_packs/` directory plus a loader module in OTR does not -- the brief itself concedes the JSON-owns-content law is achievable in either shape.

What the sibling DOES contribute is pure overhead, and Phase A's own plan is the evidence. Chunk 0 exists solely to hand-copy prod files into `production_mirror/` and re-pin SHAs. The byte-identity harness AST-parses *copies* of files that, in shape R, would be sitting next to the test -- pin the constants directly with in-repo snapshot tests and the mirror, the manifest, the refresh procedure, and the drift test all evaporate. The "bridge artifact" is a serialization seam between two things that want to be one thing. This is a lab that grew load-bearing by accretion; nobody would design it this way from a blank page.

Operator rules amplify the tax: one coder window, commit+push+verify per chunk, workflow-JSON-and-code in the SAME change. Two repos means every Phase B change that touches a seam spans two commit cycles and can never be atomic -- the exact 4D-miss failure mode (shipped but unwired) the rules exist to prevent. ComfyUI packaging adds a papercut: a second folder under `custom_nodes` is either a phantom node pack or needs careful de-node-ing forever.

## The complexity test (does the sibling split pay for itself?)
No. The honest case for T is a sandbox where broken experimental packs cannot touch production. But R already delivers that via fail-loud loading, the test suite, and git branches -- packs are inert data until a lane id selects them. The only asset the lab genuinely owns is scratch experimentation (the `EXPERIMENTAL_PIPELINE_SEAMS` material), which can stay there as a true scratch space that production never imports. Sandbox: keep. Bridge, mirror, cross-repo drift tests: dead weight.

## Migration cost if we switch now
Near-minimum it will ever be. Phase A is uncommitted and sibling-only. The extractor, seam list, profile fields, and coverage tests port nearly verbatim into an OTR module; Chunk 0 is deleted, not ported; byte-identity becomes plain snapshot tests against OTR's own constants. Roughly a day. Waiting until Phase B wires the bridge makes this a rip instead of a redirect.

## Confidence
**HIGH.** Every argument for T dissolves into something R already provides, and the plan's largest single chunk exists only to compensate for the split.
