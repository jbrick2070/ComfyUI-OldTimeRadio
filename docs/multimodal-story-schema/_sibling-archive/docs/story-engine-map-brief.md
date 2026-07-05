# Story-Engine Map + Assertion Inventory -- Task Brief

**Status:** prep only. Run this in a FRESH window AFTER the current coding project closes.
Read-only mapping -- no code changes in this pass.

## Why this exists
The story engine is a grown-organically spaghetti of LLM calls. Before combining it with
the new model, we want a plain-language map of what it actually does and a full inventory of
its assertions -- classified structural vs model-specific -- so we know what survives the
model swap and what was only ever babysitting the current model. See CLAUDE.md sections on
cloud-first direction and the "fix at source" rule; this map is Phase 0 of the model-combine
project.

## Where to run
Whichever repo holds the live story engine when you start. The nodes are mirrored in this
staging repo under `production_mirror/nodes/_otr_story_*.py` (+ casting, style, pitch, spine,
select, quality). One coder window at a time (CLAUDE.md section 1).

## Model routing (this is the point of the brief)
- **Tasks 1 and 2 (the map + the inventory) = general-purpose / Explore.** Pure read-and-catalog.
  Do NOT spend Fable here -- it is mechanical work and a general agent does it better and cheaper.
- **Task 3 (the judgment pass) = ONE Fable call.** Only after the map + inventory exist.
  This is the narrative-judgment slice Fable is actually for.
- **Task 4 (final accuracy QA) = Sonnet fan-out.** The LAST step before any coding. Several
  parallel Sonnet subagents re-check the finished plan against the REAL code and report
  discrepancies. Mechanical grounding, not creative -- Sonnet, not Fable.

## Task 1 -- Plain-language engine map (general-purpose)
Read every story node and produce a map a non-expert can follow:
- Each node: its role in one sentence; its inputs and outputs; where it hands off.
- Every LLM call site: which node, what it is asked to produce, what model/prompt it uses.
- The end-to-end flow: brief -> spine -> ... -> selected story, in order.
Output: `docs/story-engine-map.md` (prose + a simple flow list, plain language).

## Task 2 -- Assertion inventory (general-purpose)
Find every assertion / validation / constraint / guard in the story path (asserts, schema
checks, retries-on-bad-output, hard caps). For each row:
- file + line
- what it checks
- what happens when it fails (raise / retry / silent fix)
- **classification:** STRUCTURAL (true under any model -- frame counts, required fields,
  chunk ordering, referential integrity) vs MODEL-SPECIFIC (exists only to babysit the
  current model's tics).
Output: an assertion table appended to `docs/story-engine-map.md`.

## Task 3 -- Judgment pass (ONE Fable call, after 1 and 2)
Give Fable the finished map + inventory and ask:
- Which MODEL-SPECIFIC assertions can be dropped or generalized once the new model is in?
- Which STRUCTURAL assertions should move into the workflow JSON as declarative rules,
  enforced by one generic validator node (not scattered Python)?
- Does the engine's creative intent (voice, arc, pacing) survive the model swap, or does any
  guardrail encode taste that a new model would break?
Output: Fable's judgment appended as a "Model-swap readiness" section.

## Task 4 -- Final accuracy QA before coding (Sonnet fan-out)
The gate between "we have a plan" and "we start coding." Do NOT begin coding until this is clean.
- Split the finished map + assertion table + Fable readiness section into slices (by node
  cluster: spine, casting, style, pitch, select, quality, line-composer, etc.).
- Spawn a Sonnet subagent PER SLICE, in parallel. Each one re-reads the REAL node code and
  audits the plan's claims about it:
  - Does every node role / input / output / LLM call site match the actual code?
  - Is every assertion row real (file + line exists) and correctly classified
    STRUCTURAL vs MODEL-SPECIFIC?
  - Anything the map MISSED -- a node, an LLM call, an assertion not in the table?
- Each agent reports discrepancies only (found X, plan says Y). Reconcile every discrepancy
  back into `docs/story-engine-map.md` before coding. A hallucinated or stale plan is worse
  than no plan -- this step is what makes the map trustworthy to build from.
Output: a "QA sign-off" section listing what was corrected; zero open discrepancies = green to code.

## Guardrails
- Read-only. No edits to nodes or the workflow JSON in this pass.
- Fix-at-source is for the LATER coding phase: any real fix lands in prompt/node/workflow,
  never a downstream Python band-aid (CLAUDE.md).
- UTF-8 no BOM. Meaningful names. SFW.

## Definition of done
`docs/story-engine-map.md` exists with: (1) the plain-language map, (2) the classified
assertion table, (3) the Fable model-swap-readiness section, (4) the Sonnet QA sign-off with
zero open discrepancies. No code changed. Only then is it green to start coding.
