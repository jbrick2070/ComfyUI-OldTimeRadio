# Operational decision -- Transplant (sibling lab repo) vs In-repo JSON

Single-round kibitz gut-check. NOT plan-hardening. ONE architecture fork.

## The fork
OTR's story-LLM prompts currently live as Python constants in the PRODUCTION repo
(`ComfyUI-OldTimeRadio`). The chosen "transplant" architecture moves prompt CONTENT
into JSON, but hosts that JSON + the loader in a SEPARATE sibling ComfyUI custom-node
repo (`ComfyUI-OTR-UpstreamStoryLab`) that feeds production via a "bridge artifact",
with a `production_mirror/` copy of prod files pinned by SHA for drift tests. Phase A
(in flight, green, uncommitted) extracts the prompt strings into the sibling as JSON,
byte-identical, touching NO production code. Phase B later wires the bridge into prod.

The operator is now questioning whether the two-repo transplant is the best shape, or
whether it would be simpler and more robust to do it ALL IN THIS REPO:

- **T (TRANSPLANT):** keep the sibling lab repo. Prompts-as-JSON + registry + bridge
  live in `ComfyUI-OTR-UpstreamStoryLab`; production consumes the bridge artifact.
- **R (IN-REPO):** no sibling repo. Put the prompt JSON packs directly in
  `ComfyUI-OldTimeRadio` (e.g. `nodes/story_packs/*.json`) and have the existing OTR
  nodes load them. No `production_mirror/`, no bridge artifact, no cross-repo mirror.

## The operator's REAL end-goal (weigh against this, not the mechanism)
Multiple SELECTABLE lanes as DATA not code: source_bank (science RSS / media archive /
public-domain), story_model (tone lanes), story_pipeline (LLM pass structure),
visual_style (render language) -- each a JSON pack, "adding one touches zero routing
code", no hidden fallbacks, unknown ids fail loud. Core law: "JSON owns content +
configuration; Python owns validation, routing, execution, fail-loud errors."

## Decide + weigh
Pick **T** or **R**. Weigh: the real end-goal above; maintenance cost of two repos +
`production_mirror/` + a bridge vs one repo; the JSON-owns-content law (achievable in
EITHER shape); ComfyUI custom-node packaging/versioning realities; blast radius +
reversibility; how each affects Phase B; whether the sibling split earns its complexity
or is accidental (it began as a scratch "lab" workspace).

## Read
- `docs/2026-07-04-json-prompt-transplant/PHASE_A_JSON_EXTRACTION_PLAN_FINAL.md`
- `CLAUDE.md` (operator rules -- workflow JSON is source of truth; one coder window;
  no shims; fix at root)
