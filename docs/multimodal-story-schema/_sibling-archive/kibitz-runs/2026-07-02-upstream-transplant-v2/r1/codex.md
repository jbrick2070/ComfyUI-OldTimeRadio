VERDICT: no. The plan’s core story is “no hidden fallbacks, all LLM passes declared,” but it omits a live production LLM phase, conflicts with the repository’s transplant sequencing, and leaves source fetching/binding semantics underspecified.

MUST-FIX BEFORE BUILD:
1. [§4, §7 C1/C3] Missing LLM phase in the canonical seam/pipeline vocabulary. The plan says every LLM-touching pass is declared in `pipelines.json` and no pass may call a model without declaration (`kibitz-runs/2026-07-02-upstream-transplant-v2/r1/input.md:68-71`), but the seam list omits the live `story_brief_reflection` pass. Production calls `run_story_brief_reflection` through the technical slot at `production_mirror/nodes/OTR_LedgerScriptWriter.py:5419-5441`. Concrete fix: add a `story_brief_reflection` seam and pipeline pass, or explicitly disable/defer that production phase and add a test proving it is not reachable in the bridge path.

2. [§1, §7 C5] The bridge/transplant arc contradicts the repo’s own staging rules. The plan sends the bridge artifact to a “production adapter (tomorrow’s transplant chunk)” (`input.md:39-44`) and includes production-ready modules plus patch specs in tonight’s C5 (`input.md:190-205`). `TRANSPLANT_MANIFEST.md` says the bridge is not the first build; the first build is a translator head, and bridge/wiring comes only after that (`TRANSPLANT_MANIFEST.md:49-76`). Concrete fix: split the arc into “lab translator/bridge artifact now” and “production adapter/patch specs later,” or update the manifest and gates before C5 is in scope.

3. [§2, §3, §7 C1] Source-bank extensibility is overstated. The plan claims “Adding a bank touches zero routing code” (`input.md:51-52`), but also requires every bank behavior to be a named Python binding resolved through an allowlist (`input.md:72-76`), while C1 only lists fixture interpreters and no fetcher protocol (`input.md:169-173`). Concrete fix: define `SourceFetcher` and `SourceInterpreter` protocols, state that new behavior requires a Python binding plus allowlist entry, and narrow “zero routing code” to banks that reuse an existing binding.

4. [§3, §7 C5] “No fallbacks” is not reconciled with existing production fallback behavior. Production currently pads style candidates and falls back to the first chooser candidate (`production_mirror/nodes/_otr_style_picker.py:109-115`, `production_mirror/nodes/_otr_style_picker.py:681-775`), title regen falls back to `outline.title` (`production_mirror/nodes/OTR_LedgerScriptWriter.py:5201-5255`), and announcer outro falls back deterministically (`production_mirror/nodes/_otr_line_composer.py:3595-3601`, `production_mirror/nodes/_otr_line_composer.py:3652-3661`). C5 only names loud non-science failure for `_otr_style_picker` (`input.md:199-201`). Concrete fix: add a fallback inventory and lane policy matrix: preserve only explicitly grandfathered science baseline fallbacks, fail loud for new lanes, and test each named site.

SHOULD-FIX:
1. [§6, §7 C5] Visual transplant scope is too broad for this round. C5 includes `_otr_story_brief_helpers`, `otr_meta_brief_image_prompt`, `otr_shot_lock`, and `render_driver`-adjacent policy reads (`input.md:201-203`), while the manifest says deep visual prompts are risky and must be staged one by one (`TRANSPLANT_MANIFEST.md:32-47`). Concrete fix: keep only shared-tail policy seams in this build; defer deep render-driver prompt extraction to a separate gated stage.

2. [§7 C2] “Diverged v1 Python/JSON lists reconciled by union” is a weak creative-resolution rule (`input.md:178-180`). A union can preserve contradictory guardrails instead of choosing the correct authorial source. Concrete fix: require a per-seam diff artifact with one chosen winner and a reason; use union only for non-ordered forbidden-term metadata.

3. [§5, §8] [ASSUMPTION] If any mirror/drift/patch tests import production modules, the mirror is incomplete for that. `OTR_LedgerScriptWriter.py` imports `._otr_model_catalog` and `._otr_story_brief` (`production_mirror/nodes/OTR_LedgerScriptWriter.py:121-131`), but those files are not in the copied-file manifest (`PRODUCTION_MIRROR_MANIFEST.md:41-66`). Concrete fix: state drift tests are AST-only, or mirror import dependencies before any import smoke.

OPTIONAL / NICE-TO-HAVE:
- [§7b] Keep provenance hashes, but define `lab_state_digest` deterministically before it becomes a gate (`input.md:225-230`).
- [§7b] Single-level pack `extends` can stay future-only; do not add it before the basic pack resolution is green (`input.md:244-248`).

CUT THESE (scope / over-engineering):
1. [§7 C5] Cut production patch specs from tonight. It is safe because C1-C4 already prove the registry, fixtures, bridge artifact, nodes, and validation loop; C5 conflicts with `TRANSPLANT_MANIFEST.md:63-76`.

2. [§3, §9] Cut the adaptive-cleanup experiment from `pipelines.json` entirely for this build. A documented-disabled experiment still expands schema/test surface while the non-goal says no implementation tonight (`input.md:77-79`, `input.md:262-263`).

3. [§7b] Cut exhaustive full Cartesian testing over invalid bank/model/pipeline/style products. Keep the declared compatibility matrix plus one negative test per unknown axis. The plan’s “ALL combos” wording (`input.md:231-237`) risks turning source-scoped story models (`input.md:53-54`) into noisy test bloat.