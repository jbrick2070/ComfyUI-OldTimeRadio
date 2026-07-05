# Judgment -- Transplant (sibling lab) vs In-repo JSON

**Panel:** Fable + grounded analytical seat (Sonnet). Claude = grounded anchor + judge.
**Question:** keep the separate sibling lab repo + bridge + production_mirror (T), or
do the JSON prompt extraction ALL in the production repo (R)?

## VERDICT: R (in-repo). UNANIMOUS, HIGH confidence.

Fable: R, HIGH. Analytical: R, HIGH. Anchor: R. No dissent.

## The three load-bearing reasons
1. **`production_mirror/` is a permanent drift-tax.** It is a hand-copied, SHA-pinned
   second copy of production files, refreshed on every baseline bump. Phase A's
   single biggest chunk (Chunk 0) exists ONLY to maintain that cross-repo copy. In
   the in-repo shape it does not exist at all.
2. **The bridge artifact is pure indirection.** A JSON pack loaded by the existing
   OTR node via a direct `json.load()` gets the SAME "JSON owns content, Python owns
   behavior" property with none of the frozen-intermediate-file machinery.
3. **The split fights CLAUDE.md.** The operator's hard rules -- workflow JSON is the
   source of truth, every node change lands in it in the SAME change, one coder
   window -- cannot be honored by a Phase B change that spans TWO git repos. A
   same-commit, atomic prod change is impossible across a repo boundary. In-repo
   restores atomicity.

The operator's REAL goal ("add a lane = drop a JSON file, zero routing code",
JSON-owns-content, fail-loud on unknown ids) is fully achievable IN THIS REPO. The
sibling repo is mechanism, not goal -- and it began life as a scratch "lab", so the
split is accidental, not designed-for.

## Cost to switch NOW (cheap -- the window is open)
Phase A is sibling-only and UNCOMMITTED (touches zero production code). Redirecting:
- Chunk 0 (production_mirror refresh) -> DELETED outright.
- Chunk 1/3/4/5 (contracts seams, profiles fields, `extractor.py`
  `get_pack_prompt_or_none`, byte-identity harness, coverage tests) -> PORT roughly
  verbatim into `ComfyUI-OldTimeRadio` (e.g. `nodes/story_packs/*.json` +
  a small loader on the existing nodes).
- The bridge artifact -> DROPPED.
Estimate ~a day. After Phase B entangles the bridge with the workflow JSON, the same
switch is far more expensive. So decide now.

## Recommendation
Pivot to R. Keep `ComfyUI-OTR-UpstreamStoryLab` ONLY as a throwaway scratch sandbox
for experimental seams production never imports -- it is NOT a production dependency.
Re-scope Phase A as an IN-REPO JSON extraction (byte-identical prompt strings, still
no behavior change), then run the same regression + Bug Bible discipline.
