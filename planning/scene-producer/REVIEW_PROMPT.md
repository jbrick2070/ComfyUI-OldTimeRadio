# Fable — focused contrarian design review before coding

Review the repository's **OTR Scene Producer implementation plan** as one scoped, overall-design pass. Challenge whether we are taking the right path and whether a smaller, clearer design would serve this ComfyUI workflow better. Do not implement code or start a multi-round review campaign.

Plan: [OTR Scene Producer plan](PLAN.md), at `planning/scene-producer/PLAN.md` from the repository root. The approved page design is `planning/scene-producer/mockup.html`.

Use the actual repository checkout available in this review session, including a cloud checkout. Read `AGENTS.md` and `CLAUDE.md` before source tools. Verify claims against this tree and label assumptions; local Windows runtime instructions in the plan are not prerequisites for cloud design review. Do not claim tests, filesystem access outside this checkout or hardware verification you did not perform. This plan is the sole design baseline.

## User requirements that must survive

- **This is optional. Existing automatic ComfyUI/Studio flows and story banks must keep working.** Users must not be forced into stages, checkpoints, scene editing or a new interface to make a story.
- Even in Producer, generate the scene/card structure and defaults automatically. Someone can edit one shot or leave everything unchanged; no mandatory full scene breakdown or per-card editing. Keep a simple generate-all/batch-approval path.
- Preserve frozen audio, cast/voice identity and timing across visual edits, selective regeneration and restart. Partial outputs cannot masquerade as a complete published episode.
- Keep ComfyUI as the execution engine, retain canonical wiring/widget compatibility, and make the user flow understandable.
- One shared backend and canonical JSON with derived profile/stage workflows is the current recommendation. **You may recommend a separate optional workflow JSON, module or repository if it is demonstrably simpler or safer.** Structure is open to challenge; compatibility and the optional user experience are hard requirements.

## Questions to pressure-test

1. **Overall architecture:** Are three stages ended by `ExecutionBlocker`, extended replay bundles, the existing still cache, per-clip reuse and a one-page Studio editor the smallest adequate design? Identify necessary complexity versus machinery we can remove or reuse. Trace what fails if your simplification removes a mechanism.
2. **Workflow/repo boundary:** Compare the current recommendation with at most one credible alternative. Would an optional Producer workflow generated from the same basic canonical JSON be clearer than submitting the whole shipped graph and stopping each stage with a blocker? Does a separate module/repo materially improve maintenance or isolation, or create duplicated engines/contracts? Pick one recommended path and explain the concrete migration/compatibility cost.
3. **ComfyUI execution:** Check terminal roots, dependency closure, cache invalidation, node ordering, model lifecycle and durable state ownership. Existing full runs must not be paused or redirected by optional Producer semantics. Base your best-practice claims on actual code or official ComfyUI documentation where available; distinguish principles from unverified implementation assumptions.
4. **Canvas contract:** Be specific about each modified node: existing/new inputs, widget name/type/default/order, sockets/output indices, conversion-to-input effects, terminal status and previews. Is the trailing ShotLock `direction_json` widget plus writer's existing `replay_from` pointer appropriate? Check that a blank widget leaves node order and outputs unchanged, and that a stage's blocker stops exactly the nodes downstream of it. Explain exactly what normal Queue Prompt does and what changes only when Producer is chosen.
5. **Studio experience:** Walk through automatic story generation, zero-edit Producer, one-shot adjustment, selected rerender, approval, publish, error/cancel and reconnect. Start with no supplied images: the system must generate portraits and scene stills; user photos remain optional for My Story. Check button states, inherited/default fields, dirty/stale dependencies, frozen audio, submitted-direction identity and progress. One atomic project file/pending submission must avoid duplicate work without a job framework. Check that Studio can read each stage's `cards.json` and media from the bundle through ComfyUI's `/view`. Do not add a second inference authority.
6. **Optional idea to noodle on:** Could all existing story routes populate the same scene structure/view incrementally from already-produced ledger/progress data? This is a nice-to-have, **not a requirement**. It must not add LLM/GPU calls, mandatory scene work, blocking writes or a new failure mode to existing flows. If no safe seam exists, recommend deferral; it must not hold up or destabilize the working routes.

## How to judge and harden

Try to refute the design rather than endorse it. For each real gap, give the concrete failure/user journey, evidence or clearly labeled assumption, smallest correction and acceptance check. Separate genuine omissions from requirements already covered elsewhere. Challenge a mechanism when justified, while preserving its necessary guarantees; do not silently discard good parts or weaken validation to make a design look simpler.

No implementation, renders, dependency installation or source/workflow changes. Only the named plan may receive justified targeted amendments. If you can edit the plan file, record its starting hash and apply only justified targeted changes to that same file after checking it has not changed underneath you. Otherwise return exact replacement passages. Keep one self-contained plan without changelogs, review history or references to other drafts.

Return:

1. **KEEP, SIMPLIFY or REPLACE**, with the principal reason and at most one concrete alternative.
2. Up to five material findings, each with evidence/assumption, correction and proof.
3. Exact node/socket/widget and user-interface consequences of your recommendation.
4. Minimal amendments to the plan, or explicitly no justified amendments; identify unresolved decisions requiring the user.

Keep this a focused design judgment. Do not replay a full implementation audit or claim passing tests/hardware proof. Your output and any amended plan will return to Codex for final grounding before coding.
