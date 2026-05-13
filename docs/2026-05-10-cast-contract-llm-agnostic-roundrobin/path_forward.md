---
name: OTR LLM-Agnostic — Path Forward
description: Synthesis of three round-robin critiques into a priority-ordered plan for making OTR call paths LLM-agnostic without freezing the design around a model that does not yet exist.
type: project-plan
status: proposed
---

# Path Forward

The three reviews are not three takes on the same problem. They are three layers stacked on top of each other:

- **Review C** found a structural contradiction in the constraint itself.
- **Review B** found the document is too philosophical and needs measurable acceptance criteria.
- **Review A** found five concrete code paths already violating it.

Fix them in that order. Start with the structural one, because the other two depend on which way you resolve it.

---

## P0 — The One Decision That Unblocks Everything

**The constraint as written cannot hold.** A model fine-tuned hard on 1903 period prose will be *worse* at structural work (JSON outlines, cast contracts, validator-style logic) — narrowing is what fine-tuning does. You cannot simultaneously have:

1. One canonical prompt per call path
2. A narrow period specialist in the pool
3. That specialist handling every call type

Pick one resolution. Recommended:

> **Role-based routing. The period model is a prose-plane specialist only.**

Split call paths into two planes:

| Plane | Calls | Model Pool |
|---|---|---|
| **Control-plane** | outline, cast contract, critic, validator, repair, planner | general-purpose (Mistral / Gemma / Qwen) — must be LLM-agnostic |
| **Prose-plane** | line composer, in-character dialogue, period-flavor passes, title generation | general pool OR period specialist — agnostic over the general pool, period model is an *opt-in* enhancement |

This makes the constraint coherent: control-plane is strictly agnostic; prose-plane tolerates a specialist whose contract is "produce period-flavored text from a structured brief that the control plane already validated."

If you reject this split, the only other coherent resolutions are: (a) drop the period model from the pool, or (b) accept that the period model dilutes its period training in exchange for structural competence. Both are worse outcomes than the split.

**Action:** Add a "Plane Routing" section to the canonical constraint doc. Tag every existing LLM caller with its plane. Open question: which plane does `OTR_LLMScriptCritic` sit in? (Probably control — critique reasoning is structural even when judging prose.)

---

## P0 — Pick the Proxy Model Today

You cannot test against a model that does not exist. "Test against the weakest expected model" is currently untestable, which means it is unenforceable, which means it is being silently violated.

**Recommendation:** Designate **Gemma-2-2B-it** as the gating proxy for control-plane calls, effective immediately.

Why this one over alternatives:

- **vs. Phi-3-mini:** Gemma's chat-template strictness (no system role, role-alternation enforcement) exposes the exact class of bugs the future period model is most likely to have. Review A item #1 is literally a Gemma bug today.
- **vs. Qwen2.5-3B:** Qwen is more forgiving; it will hide bugs Gemma surfaces.
- **vs. waiting:** speculative gating against a non-existent model is how this whole constraint becomes vibes.

For prose-plane, designate a current proxy too (suggest **Gemma-2-2B-it** again, or any small instruction-tuned model in the 2-4B range). The point is to have *something* to fail against today.

**Action:** Add `proxy_models: { control: gemma-2-2b-it, prose: gemma-2-2b-it }` to the constraint doc frontmatter. When the period model lands, swap it into the prose slot and re-run the matrix.

---

## P0 — Lock the ModelAdapter Interface

Without a concrete shape, every caller invents its own wrapper, and "the prompt is one canonical version" becomes folklore. Lock something minimal *now*, even if you iterate later:

```python
@dataclass
class PromptObject:
    system: str | None          # may be None for models that reject system
    user: str                   # the canonical task content
    examples: list[Turn] = []   # few-shot, structured not concatenated
    output_contract: Schema     # for validator + repair-prompt construction

class ModelAdapter(Protocol):
    role_layout: Literal["user_only", "system_user", "system_user_assistant"]
    json_mode: Literal["native", "tag_wrapped", "regex_extract"]
    sampling_defaults: SamplingConfig

    def render(self, p: PromptObject) -> list[ChatMessage]: ...
    def normalize_sampling(self, intent: CreativityIntent) -> SamplingConfig: ...
```

Two non-obvious points from the reviews worth honoring:

1. **PromptObject is structured, not a string.** The moment a prompt is a flat formatted string, chat-template assumptions are already baked in.
2. **Sampling is expressed as intent, translated by the adapter.** Mistral's `temperature=0.9` is not Gemma's `temperature=0.9`. Callers say `creativity=low|med|high`; the adapter maps.

**Action:** Stub `model_adapter.py` with the interface above. Implement adapters for Mistral-Nemo (current) and Gemma-2-2B (proxy). Refactor exactly one caller through it as a pilot — recommend `_otr_outline.py` since it already has the cleanest retry pattern.

---

## P1 — Code Fixes From Review A (Immediate, Mechanical)

All five are real bugs. Order by impact:

1. **Strip the "1940s" literal from `OTR_LedgerScriptWriter.py`.** Five-minute fix. The doc contradicts itself; the code contradicts the doc. Replace with `style` / `style_custom` variable interpolation.
2. **Move chat-template / system-role folding into `_build_truncating_generate_fn` (or the adapter).** This is Review A #1 — it is also the gating bug for Gemma-2-2B and for the future period model. Required before the proxy can be added to the matrix.
3. **Fix left-truncation to be middle-out / keep-system.** Anchor system + most recent user turn; truncate the middle payload. The current behavior silently drops the instruction when context fills, which manifests as "model went off the rails" with no signal.
4. **True repair loop in `compose_line` (mirror outline's pattern).** Review A #3 is correct that bumping temperature on a blind-failing prompt is just hallucinating harder. Reuse the outline repair structure.
5. **Tighten `_MD_BOLD_ITALIC_RE`.** Only strip at word/phrase boundaries, or swap for a real markdown parser. Low priority — it bites only when sci-fi vocabulary like "Sector_7" appears, which is rare but real.

**Action:** Items 1 and 2 are blocking — do them before adding Gemma to the smoke matrix. Items 3–5 can be follow-ups.

---

## P1 — The Silent Failure Mode: Parser / Normalizer Layer

Review B #9 is the most important point neither A nor C raised: the real cross-model failure is rarely the *prompt* — it is the *parsing*. Gemma wraps JSON in prose. Qwen renames fields. Period models will hallucinate "helpful" preambles.

Add a shared normalization layer that runs *before* validation on every structured call:

```
raw_output
  → strip_markdown_fences
  → extract_first_balanced_json
  → reject_if_multiple_conflicting
  → schema_validate
  → repair_loop_if_failed
```

The raw output stays in debug logs. The validator never sees the prose wrapper.

**Action:** Add `llm_output_normalizer.py`. Route all structured-output callers through it. Non-structured callers (line composer prose) get a separate, lighter normalizer.

---

## P1 — Define "Reliable" With a Number

The constraint says "must run reliably" without a number, which means every caller picks a different bar in their head. Lock:

- **Schema-valid on attempt 1:** ≥ 80%
- **Schema-valid within 3 attempts:** ≥ 98%
- **Hard fail behavior:** structured error with raw output preserved. Never silently substitute partial content into the episode.

**Retry ladder, universal:**

1. Fresh attempt, default sampling.
2. Fresh attempt with *adjusted decoding* (not always hotter — for JSON callers, often *cooler*). Per-caller spec.
3. Repair prompt: original output + validation errors + explicit correction directive.
4. Hard fail. Operator queue or skip-segment fallback.

**Action:** Add reliability targets and retry ladder to the constraint doc as **MUST** rules (Review B #10 — mark every bullet as MUST vs SHOULD).

---

## P2 — Capability Tiers (Worth Doing, Don't Over-Engineer)

Review B #6 is right that not every model should be expected to do every job. But with three to five call paths, a five-tier system is bureaucracy. Collapse to:

- **Structural:** outline, cast contract, critic, validator. Requires JSON discipline + cast-contract preservation.
- **Generative:** line composer, prose passes. Requires voice, not structure.
- **Reasoning:** critique + repair. Requires long-context coherence.

Each call path declares one. Each adapter declares which tiers it is qualified for. A model is supported for a call path only after passing that tier's fixtures.

**Action:** Tag each `OTR_*` node with its tier in code comments. Defer formal capability matrix until there are >5 callers.

---

## P2 — Fixtures > Live Smoke Tests

Review B #11 is correct that live generation tests are not enough. Build a fixture set per caller with known inputs and expected output shapes:

- minimal cast / maximum cast
- Lemmy cameo hit / miss
- weird names (apostrophes, non-Latin, numerals)
- long scene brief / short scene brief
- known-invalid model output (for normalizer + repair tests)
- known-invented speaker tag (for cast-contract enforcement test)

Fixtures run offline against the adapter layer with recorded outputs. Smoke tests run against real models.

**Action:** `tests/fixtures/otr_outline/`, `tests/fixtures/otr_line_composer/`, etc. Start with five fixtures per caller; expand as bugs surface.

---

## P3 — Process Hygiene

These are doc-level, not code-level. Bundle into a single constraint-doc rewrite:

- **MUST vs SHOULD tags** on every rule (Review B #10).
- **Review triggers:** revisit constraint when period model spec lands, when a new family enters the pool, or when single-call success rate on any pool member drops below 70%.
- **Round-robin question template:** "Will an instruction-tuned 2B–14B local model with [schema/tag conventions] reliably produce [output] when prompted with [structure]?" — keep model names out of the question.
- **Prompt-change discipline:** changes to a canonical prompt require re-running that caller's smoke matrix.

---

## What NOT to Do

- **Do not pre-build adapters for every hypothetical model.** Adapter per pool member, no more.
- **Do not freeze the period model's spec around speculation.** Write a falsifiable readiness checklist (small, instruction-tuned, narrow corpus, schema-fragile) and treat it as a profile rather than a model.
- **Do not let the agnosticism rule scare you out of real prompt engineering.** Few-shot examples, explicit schemas, output-only constraints, failure rules — these are universal, not model-specific. The line is: no *model-name* hacks, no *chat-template* assumptions. Everything else is fair game.
- **Do not silently lower the ceiling on Mistral to match Gemma.** Mistral can use better sampling defaults via the adapter without changing the canonical prompt. That is the whole point of the adapter layer.

---

## Acceptance Criteria (consolidated, MUST-level)

A call path is LLM-agnostic only if:

1. One canonical `PromptObject` shared across supported models.
2. Model-specific differences live in `ModelAdapter`, not the prompt.
3. Output runs through the shared normalizer before validation.
4. Validation is Python contract, not model trust.
5. Retry uses the shared ladder (fresh / adjusted / repair / hard-fail).
6. Raw output is logged on every failure.
7. Cast / ledger state is preserved across retries.
8. Fixtures pass offline.
9. Smoke matrix passes against current pool + proxy.
10. No model name appears in the canonical prompt.
11. Fails closed, never silently.

---

## Sequenced Execution

If picking one week of work:

- **Day 1:** Constraint doc rewrite with plane routing, proxy model designation, reliability numbers, MUST/SHOULD tags.
- **Day 2:** Strip "1940s" literals. Stub `ModelAdapter` + `PromptObject`. Implement Mistral adapter as null-op baseline.
- **Day 3:** Implement Gemma-2-2B adapter (role folding, sampling translation). Refactor `_otr_outline.py` through the adapter.
- **Day 4:** Implement normalizer layer. Route outline + ledger through it.
- **Day 5:** Build first fixture set for outline. Run smoke matrix (Mistral + Gemma) against it. Document failures.

Everything else (capability tiers, full fixture coverage, repair-loop unification in line composer) can follow once the foundation holds.

---

## What Would Sharpen This Plan

Sending these would let me get specific where this plan is currently structural:

1. **`_otr_outline.py`** — to verify the 3-attempt pattern is actually fresh / fresh-hotter / repair, and to extract the first concrete `PromptObject` shape.
2. **`OTR_LedgerScriptWriter.py`** — to write the exact diff for the "1940s" strip and the middle-out truncation rewrite.
3. **`_otr_line_composer.py`** — to design the repair-loop refactor against the actual current code.
4. **Current cast-contract and critic prompts** (snippets are fine) — to confirm they belong in control-plane and to spec their fixtures.

Without those, the plan above is correct in direction but generic in detail. With them, the Day 2–4 work becomes line-level diffs.
