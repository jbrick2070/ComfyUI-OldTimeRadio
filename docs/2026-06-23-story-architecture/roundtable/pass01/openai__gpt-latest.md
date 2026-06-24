<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The direction is mostly coherent, but it is not build-ready because the primary “pitch room + taste selector + critic-to-divergent-replan” arc is underspecified at the integration/decision level, while the document simultaneously expands into B3/B4, external repo mining, character interviews, frontier models, refine-loop policy, and scene composition without a hard MVP boundary.

MUST-FIX BEFORE BUILD:

1. [S2, S3, S6] The primary lever is named but not specified as an executable architecture.
   Defect: “Pitch room -- DIVERGE” and “Greenlight -- SELECT” are described as prompts/intent, not as a concrete subsystem contract. The plan does not define the input/output schema for candidate seeds, episode takes, greenlight rationale, rejected candidates, selected premise, or how the selected take modifies `script_brief` / outline inputs. Without that, implementers can bolt on another prose prompt that does not actually affect the planner.
   Concrete fix: Add an MVP contract:
   - `news_interpreter` output or current `script_brief` in.
   - `PitchCandidate[]` out with fixed fields: premise/logline, protagonist, antagonist/pressure, genre mode, emotional core, ending promise, why-different.
   - `GreenlightDecision` out with selected candidate id, taste rationale, risk flags, and rewritten `script_brief` / planning brief.
   - Explicit handoff point into `_otr_outline.py` / `score_outline`.
   - Persist enough metadata for critic escalation to request a different candidate, not just regenerate text.

2. [S2.2, S3.8, S6.1] “Route structural verdict to pitch room” is conceptually correct but operationally ambiguous.
   Defect: The document says Wave-1C structural failures should go to pitch room, not same-seed regen, but does not define which structural verdicts trigger divergent re-plan, whether the same news seed is reused, whether rejected pitch candidates are excluded, whether the greenlight selector can select the previous premise again, or how many cycles are allowed. This risks creating a larger non-monotonic loop around the whole episode.
   Concrete fix: Define an escalation policy:
   - Structural categories eligible: e.g. `premise_clarity`, `continuity`, `resolution`, `emotional_arc` as already named in REV 2.
   - On structural failure, call pitch room with critic report plus “do not repeat failed premise shape.”
   - Maintain `failed_premise_fingerprints` / rejected candidate ids for the run.
   - Cap divergent replans separately from line rerolls.
   - If exhausted, ship keep-best or fail according to existing no-fallbacks policy.
   - Define whether “different plan” means new premise, new ending, new protagonist, or new beat topology.

3. [S2.1, S5, S6] The local-model ceiling decision is treated as operator policy but the proposed primary lever depends on it.
   Defect: The plan says pitch-room selection raises quality only if the pool contains a good candidate, and also says local models top out around B. But S6 still lists pitch-room divergence as Sprint 1 without requiring the frontier-vs-accept-B decision first. That is a narrative mismatch: the A+ mission depends on the model ceiling, but the build sequence defers the decision.
   Concrete fix: Put an explicit gate before Sprint 1:
   - Mode A: local-only goal = reduce sameness / improve median, not promise A+.
   - Mode B: frontier-draft lane enabled = pursue A/A+ candidates.
   - The SPEC must require operator selection of one mode before benchmark claims or campaign planning.
   - If local-only, rename success criteria away from A+.

4. [S2A, S3.7, S6.3] B3/B4 whole-scene/whole-episode prose is too large and underdesigned for the same build arc as pitch-room MVP.
   Defect: The document correctly says the prose-to-ledger parser is “the whole ballgame,” but still includes “Climb the B-ladder to whole-story prose” in the same deliverable sequence. That creates a second architecture project with failure modes across speaker attribution, cast mapping, beat segmentation, voice tags, freeze cascade, video roles, and captions. It competes with the stated primary lever.
   Concrete fix: Move B3/B4 out of the initial SPEC’s build sprints into a separate research spike with parser acceptance tests only. Keep Sprint 1 focused on premise divergence + greenlight + critic replan. Keep B2 `use_exchange` as the only composition-granularity change in this plan.

5. [S3.1 vs S1] “Assignment desk -> THREE candidate seeds” conflicts with the stated primary gap unless scoped carefully.
   Defect: S2 says the missing lever is “one news/script_brief -> 3 radically different episode takes.” S3.1 instead starts by changing `news_interpreter` to surface “3 headlines” / candidate seeds. That expands from premise divergence to source-material selection and risks changing the control-plane behavior before proving the story architecture hypothesis.
   Concrete fix: For MVP, do not modify headline/seed selection. Start with the existing single `script_brief` and generate 3 divergent takes from it. Leave multi-seed assignment desk as a later feature if pitch-room selection proves useful.

6. [S2.1, S3.3, S4] “Taste selector” is underspecified and risks becoming another deterministic metric selector.
   Defect: The plan distinguishes `score_outline` as deterministic metric-level selection from the desired “showrunner taste,” but does not define what taste means, what rubric it uses, or how it differs from the existing critic/grade machinery. “Pick the ONE that makes a listener sit in their parked car” is evocative but not buildable or testable.
   Concrete fix: Define a compact greenlight rubric separate from defect grading:
   - surprise / freshness;
   - human want;
   - audio-stageability;
   - ending promise;
   - OTR fit;
   - risk of console-standoff collapse;
   - best image/line heard in the pitch.
   Require the selector to quote evidence from each candidate and reject at least one candidate for sameness/staleness.

7. [S0, S2, S5] The plan claims the root cause is premise-level sameness, but some named symptoms are beat/staging failures.
   Defect: “climax off-stage,” “announcer narrates the outcome,” and “every beat must turn” are not purely premise-level problems; they are outline/beat-role/staging constraints. The document risks overclaiming that premise divergence solves symptoms that may persist after a better premise.
   Concrete fix: Split success hypotheses:
   - Pitch room addresses premise sameness / console-standoff collapse.
   - Beat-turn and on-mic climax constraints address staging/outline execution.
   Add minimal outline-level acceptance rules to Sprint 1 or explicitly defer them.

SHOULD-FIX:

1. [S3.4, S2A Other levers] “Theme & ending first” is introduced as new but not integrated with pitch selection or outline generation.
   Defect: It could duplicate or conflict with the selected pitch’s “ending promise.”
   Concrete fix: Fold it into `PitchCandidate` / selected planning brief as `theme_sentence` and `final_20_seconds`, rather than a separate pipeline step.

2. [S3.5, S2A Other levers] Character interviews are scope creep for the first build.
   Defect: They address voice differentiation, not the declared primary root cause. They also touch casting/persona behavior, increasing blast radius.
   Concrete fix: Defer to a later sprint. If retained, make it optional metadata generated only after greenlight selection.

3. [S3.6] “Require every beat to change the temperature” is a strong high-level rule but lacks enforcement location.
   Defect: It is unclear whether this belongs in `_otr_outline.py`, `_otr_story_quality_l12.py`, `score_outline`, or critic rubric.
   Concrete fix: Define it as an outline scoring/validation addition: each beat must declare `turn`: power/status/knowledge/emotion change. Use it as a selector penalty before composition.

4. [S5] “Default-OFF / byte-identical until proven” conflicts with “flip `use_exchange` ON in canonical workflow JSON” unless the validation and rollout path are explicit.
   Defect: The plan says story changes ship dark behind flags, but S2.3 says flip ON after N=3. This is acceptable only if the canonical JSON change is treated as post-validation, not part of the initial dark build.
   Concrete fix: Specify: build no code for `use_exchange`; run GPU N=3; if pass, submit a config-only PR changing canonical workflow default/link. Otherwise leave default false.

5. [S5, S6] Success metrics are too thin for the central premise.
   Defect: The only concrete live metric for `use_exchange` is VRAM/slot drift. The pitch-room lever lacks measurable acceptance criteria beyond grades.
   Concrete fix: Add acceptance metrics:
   - no selected pitch shares the same protagonist/location/conflict shape as another candidate;
   - selected premise is reflected in outline;
   - structural-failure replan produces a different pitch fingerprint;
   - median/keep-best grade improves over a fixed seed set;
   - reduced recurrence of “console standoff” / off-stage climax tags.

6. [S4] External repo mining is not bounded.
   Defect: Mining Open-Theatre, how-to-make-script, Dramatron, radio-drama-generator, and podcast repos can sprawl into prompt archaeology instead of shipping the MVP.
   Concrete fix: Limit research extraction to one deliverable per repo, or cut all but Open-Theatre for Sprint 1. No framework imports.

7. [S5] “No-fallbacks pipeline” is invoked, but the proposed divergent replan loop needs a failure mode.
   Defect: If pitch room generation, taste selection, or structural replan fails validation, the document does not say whether to fail loud, use original brief, or ship best prior artifact. “No fallback” suggests fail loud, but existing keep-best/refine behavior suggests shipping best.
   Concrete fix: Define failure behavior explicitly for research mode and production mode.

8. [S1 Grounding] Several file/function claims are grounded by the document, but the requested review mentions real files in S1 without excerpts.
   Defect: I cannot verify line numbers, defaults, or wiring beyond the document’s assertions.
   Concrete fix: In the SPEC, include a grounding appendix with exact observed signatures/defaults for touched integration points. [ASSUMPTION]

OPTIONAL / NICE-TO-HAVE:

- [S2.4] Listener-taste augmentation can be valuable, but it should trail the pitch-room MVP. It is not needed to prove premise divergence.
- [S2A] Frontier lane can be A/B tested as a drafting-only lane before wiring it into all passes.
- [S3] Store rejected pitch candidates for later analytics; useful for diagnosing whether the taste selector is actually selecting freshness.

CUT THESE (scope / over-engineering):

1. [S3.1] Cut “THREE candidate seeds/headlines” from the first sprint.
   Why safe: The stated missing lever is divergent takes from one `script_brief`; changing upstream news selection is a separate product behavior and increases variance before the premise experiment is measurable.

2. [S3.5] Cut character interviews from MVP.
   Why safe: They address character voice sameness, not premise-level divergence. Add after selected premise/outline path works.

3. [S6.3, S2A B3/B4] Cut whole-episode prose/transcribe from the initial build sprints.
   Why safe: The document itself identifies the parser as high-risk and make-or-break. It should be a separate spike with fixtures and failure tests, not coupled to pitch-room delivery.

4. [S4] Cut podcast/TTS plumbing repo mining from this SPEC.
   Why safe: The plan states OTR already exceeds those systems and story work is upstream of the ledger/render. They do not serve the current story-architecture goal.

5. [S2.4] Cut listener-taste critic augmentation from Sprint 1.
   Why safe: The existing critic already detects structural defects per the REV 2 correction. The missing cure is divergent planning, not another review signal.

6. [S2A Other levers] Cut frontier-lane implementation work from this SPEC unless the operator chooses frontier mode.
   Why safe: The lane is described as already built/opt-in/cost-guarded. The required action is a policy decision and test matrix, not new architecture, unless verification shows otherwise. [ASSUMPTION]

7. [S5] Cut refine-loop hardening from the first pitch-room sprint.
   Why safe: Keep-best already masks some non-monotonicity. Hardening revision is a separate optimization after the candidate pool and structural replan path exist.