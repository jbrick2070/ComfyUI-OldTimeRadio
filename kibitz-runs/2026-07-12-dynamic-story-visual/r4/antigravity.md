VERDICT: yes-with-fixes. Unresolved Google API constrained generation mapping and ambiguity in evidence-target resolution block immediate build lock.

MUST-FIX BEFORE BUILD:
1. [section 4.3 / 9.2] **Google API Constrained Generation Ambiguity**: The plan has an unresolved "State which" directive regarding whether the Google API creative lane adds a constraint branch or runs unconstrained. [ASSUMPTION] Under the current implementation, `make_constrained_generate_fn` in `nodes/_otr_constrained_generate.py` expects a tokenizer and will fail for a `google_api` provider. *Concrete fix*: State explicitly that the Google API lane runs unconstrained (with the typed-repair ladder as its sole schema defense) and define the fallback in `make_constrained_generate_fn` to return the unconstrained Google generate closure when `provider == "google_api"`.
2. [section 2.2 / 2.6 / 7.1] **Evidence Target Text Ambiguity**: The plan checks that a factual quote is a verbatim substring of "its cited text" but does not define what the target text is for non-line IDs (`cast:<char_id>`, `brief:<key>`, `meta:<key>`, `title`). *Concrete fix*: Define the target text mapping exactly:
   - `line:<line_id>` -> `lines[line_id].text`
   - `cast:<char_id>` -> `cast[char_id].character_description`
   - `brief:<key>` -> `brief[key]` (or list element string if list-valued)
   - `meta:<key>` -> `meta[key]`
   - `title` -> `meta.episode_title`
3. [section 2.2 / 3.3] **Clue Visual Treatment for Clueless Episodes**: The schema requires `clue_visual` with inline evidence, but some story banks or episodes may lack clues/lost-objects. *Concrete fix*: State that if the story DTO lacks clue-related elements, `clue_visual.evidence` is allowed to be empty (`[]`) and the `clue_visual.treatment` can be a standard string (e.g., `"No clue mechanism present in this episode"`).

SHOULD-FIX:
1. [section 8.3] **Workflow Node Execution Order Placeholder**: The `order` field for node 96 is a placeholder (`<topological -- after 62, before 89/90>`). *Concrete fix*: Specify that Node 96 is assigned topological order `4`, and all subsequent nodes in `workflows/otr_canonical.json` (such as Node 88, which starts at order 4) have their topological orders incremented by 1.
2. [section 2.6 / 7.1] **Shots Sorting Normalization**: The plan requires `shots[]` to be in source-line order, but does not specify how Python enforces this order when assembling the final artifact from batched P-B runs. *Concrete fix*: Specify that the final assembly helper will sort the batched shots list by the DTO's line-order sequence before hashing and stamping.

OPTIONAL / NICE-TO-HAVE:
1. [section 8.3] **Output slot_index explicit definition**: Output slots `direction_report` and `done` on Node 96 lack explicit `slot_index` fields in the JSON definition, which is inconsistent with LiteGraph production conventions. Add `slot_index: 1` and `slot_index: 2` respectively.

CUT THESE:
None.

VERIFY-AT-BUILD checklist:
None — no items were flagged UNVERIFIABLE in rounds 1–3 review judgments.
