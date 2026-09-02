VERDICT: build-ready as-is? no. Critical runtime crashes on temperature=0.0 with hardcoded do_sample=True, broken acceptance tests from superseding Ariel/Puck, manifest schema crashes when running the stamper on Shakespeare scenes, and broken data flow in RosterGenderVerdict and Ledger.set_cast make this unbuildable as specified.

MUST-FIX BEFORE BUILD:

1. [D1, D6 vs nodes/_otr_constrained_generate.py:296-297] Fatal crash on `temperature=0.0` in `make_constrained_generate_fn`.
   Defect: D1 and D6 specify executing Tier 3 LLM calls at `temperature 0.0` using `make_constrained_generate_fn` for greedy, deterministic decoding. However, `nodes/_otr_constrained_generate.py:296-297` hardcodes `do_sample=True`:
   `out = model.generate(**inputs, do_sample=True, temperature=temperature, ...)`
   In HuggingFace Transformers, invoking `model.generate()` with `do_sample=True` and `temperature=0.0` raises a fatal `ValueError: do_sample is set to True, but temperature is set to 0.0. Please set do_sample=False or temperature > 0.` (or causes a ZeroDivisionError during logits scaling). The prototype in `scripts/otr_gender_secondopinion_lab.py:75` succeeded only because it passed `temperature=0.2`. Passing `temperature=0.0` as specced in D1/D6 will crash the stamper on the very first LLM call.
   Fix: In `nodes/_otr_constrained_generate.py:296`, dynamically evaluate sampling mode: `do_sample = bool(temperature > 0.0)`. When `do_sample` is False, omit `temperature` and `top_p` from `model.generate()`. Alternatively, if modifying the shared generator closure is deferred, D1/D6 must explicitly specify a small positive temperature (e.g. `temperature=0.01` or `0.1`) or pass an explicit greedy decoding parameter.

2. [D3 vs tests/test_roster_gender.py:21-23, 134-151 & config/source_banks/shakespeare/roster_gender_supplement.json:24-27] Resolving Ariel and Puck breaks shipping corpus acceptance test and violates standing editorial ruling.
   Defect: D3 assumes that leaving Ariel and Puck on the roll was merely "a limitation of the old mechanism" and specifies allowing Tier 3 to pin them in the sidecar. This directly breaks the test suite:
   `tests/test_roster_gender.py:23` explicitly defines `UNRESOLVABLE_BY_DESIGN = {"ARIEL", "PUCK"}`.
   `tests/test_roster_gender.py:148-150` (`test_every_shipped_cast_hint_resolves_except_the_two_left_to_the_operator`) asserts:
   `assert len(resolved) + len(unresolved) == 42`
   `assert set(unresolved) == UNRESOLVABLE_BY_DESIGN, unresolved`
   `assert len(resolved) == 40`
   Furthermore, `config/source_banks/shakespeare/roster_gender_supplement.json:24-27` documents: *"ARIEL and PUCK are deliberately NOT here. Folger's stage directions use 'he' for both, but neither has a defensible roster fact and both are genuinely editorial calls that belong to the operator. They stay on the existing roll, which is why the corpus test asserts 40 of 42 rather than 42 of 42."*
   Resolving Ariel and Puck in the sidecars immediately turns `test_roster_gender.py` red (`len(resolved) == 42 != 40`).
   Fix: Treat "leave them to the roll" as a binding editorial ruling. In `config/source_banks/shakespeare/character_gender_index.json`, lock entries for `("The Tempest", "ARIEL")` and `("A Midsummer Night's Dream", "PUCK")` with `locked: true` and `gender: ""` (or `"unsure"`), or explicitly exclude them in the stamper candidate loop so they remain omitted/unresolved and continue rolling per `UNRESOLVABLE_BY_DESIGN`.

3. [D3, D5 vs config/source_banks/shakespeare/sources/*.provenance.json & nodes/_otr_roster_gender.py:290-311] Byte-identical known Shakespeare rows contradict `gender_confidence` field injection.
   Defect: D3 mandates: *"KNOWN rows are byte-identical before and after (test: hash the 53 known rows)."* However, the 53 existing known Shakespeare rows in `config/source_banks/shakespeare/sources/*.provenance.json` (e.g. `comedy_errors__act3_scene1.provenance.json:61-67`) do not have a `gender_confidence` field; they contain only `name`, `roster_name`, `description`, `gender`, `gender_source`. Meanwhile, D5 mandates: *"Sidecar rows carry gender_source (already) and new gender_confidence in {stated, recalled, inferred}: roster/pronouns/relation/title -> stated... _verdict_from :290 fills them from the matched row."*
   If the stamper injects `"gender_confidence": "stated"` into the 53 known rows, the byte-identity hash test fails. If the stamper preserves them byte-identically without `gender_confidence`, then `row.get("gender_confidence")` in `_verdict_from` returns `None`/`""`. Consequently, all 53 known Shakespeare rows will propagate `gender_confidence: ""` downstream to `gender_map_for_names` and `cast_source_contract`, violating D5's stated contract.
   Fix: In `nodes/_otr_roster_gender.py:_verdict_from`, add read-through fallback logic when the sidecar row lacks `gender_confidence`:
   `confidence = str(r.get("gender_confidence") or "").strip() or ("stated" if r.get("gender_source") in ("roster", "relation", "title", "pronouns") else "")`.
   This preserves byte-identity on the legacy Shakespeare sidecars while satisfying D5's contract in memory at render time.

4. [D3 vs scripts/otr_stamp_character_genders.py:151-233, 244-251 & config/source_banks/shakespeare/curated_scenes.sample.json] Stamper control flow and schema crash on Shakespeare manifest.
   Defect: D3 extends the stamper to the Shakespeare bank. However, `scripts/otr_stamp_character_genders.py:246-250` iterates strictly over `manifest.get("sources")` and `source.get("units")`. Shakespeare's manifest (`config/source_banks/shakespeare/curated_scenes.sample.json`) has no `sources` key; its schema is `{"schema_version": "v1", "scenes": [...]}` where each scene is a flat dictionary without a `units` array. Passing Shakespeare's manifest to `stamp_unit` fails immediately with `AttributeError`/`KeyError`.
   Furthermore:
   - `_base_identity` (:132-148) expects `source["source_id"]`, `unit["unit_id"]`, `source["title"]`, which do not exist in Shakespeare scene objects (`play_code`, `play_title`, `act`, `scene`).
   - `_candidates(source)` (:73-89) extracts `source.get("cast_hints")`. In Shakespeare, `cast_hints` contains only primary leads (e.g. `["Antipholus", "Dromio", "Adriana"]`), missing the exact `unknown` rows D3 is tasked with resolving (`LUCE`, `ANGELO`, `BALTHASAR`).
   - If `stamp_unit` overwrites rows with the prose dictionary shape (`name`, `gender`, `gender_source`, `gender_confidence`, `evidence`), it strips Shakespeare's required `roster_name` and `description` fields, breaking `_candidate_names` in `nodes/_otr_roster_gender.py:280-287`.
   Fix: In `scripts/otr_stamp_character_genders.py`, introduce a dedicated scene stamper (e.g. `stamp_shakespeare_scene`) or bank adapter. For Shakespeare: loop over scenes in `curated_scenes.sample.json`; candidate names must be sourced from the sidecar's existing `characters` rows where `gender == "unknown"`; preserve `roster_name` and `description` when updating filled rows; and preserve the root Shakespeare provenance fields rather than running `_base_identity`.

5. [D5 vs nodes/_otr_roster_gender.py:290-311, 468-470, 494-500] `RosterGenderVerdict` and `gender_map_for_names` data flow missing wiring.
   Defect: D5 asserts: *"RosterGenderVerdict gains gender_source: str = "" and gender_confidence: str = "" with defaults (six sites untouched; _verdict_from :290 fills them from the matched row)... gender_map_for_names passes both through to lock_cast -> cast_source_contract"*.
   This wiring does not exist:
   - Line 310 (`return RosterGenderVerdict(gender, reason, tier, matched)`) IS construction site #2. If it is "untouched", `gender_source` and `gender_confidence` take their default empty strings `""` on every successful roster match.
   - Line 468 (`resolve_with_supplement`) is site #6 (`return RosterGenderVerdict(entry["gender"], entry["evidence"], "supplement", verdict.matched)`). If left untouched, supplement resolutions emit `gender_source=""` and `gender_confidence=""`.
   - `gender_map_for_names` (:494-499) explicitly constructs a 4-key dictionary: `{"gender": verdict.gender, "evidence": verdict.evidence, "tier": verdict.tier, "roster_name": ...}`. It does not pass `gender_source` or `gender_confidence`.
   Fix:
   - In `nodes/_otr_roster_gender.py:_verdict_from` (:310), extract `gender_source` and `gender_confidence` from the matched row and pass them to `RosterGenderVerdict(..., gender_source=src, gender_confidence=conf)`.
   - In `resolve_with_supplement` (:468), pass `gender_source="supplement"` and `gender_confidence="stated"`.
   - In `gender_map_for_names` (:494-500), add `"gender_source": verdict.gender_source` and `"gender_confidence": verdict.gender_confidence` to the returned dictionary.

6. [D5, D7(c) vs nodes/production_ledger.py:1070-1108] Ledger cast row vs cast source contract receipt location mismatch.
   Defect: D7(c) specifies acceptance criteria: *"ledger cast rows ELIZABETH BENNET female / DARCY male with gender_source llm_recall, gender_confidence recalled."*
   However, `nodes/production_ledger.py:1070-1108` (`Ledger.set_cast()`) rebuilds each cast row dictionary using a strict, fixed whitelist of keys (`char_id`, `name`, `character_description`, `gender`, `tts_model`, `voice_preset`, `voice_params`, `line_count`, `word_count`, `presentation_gender`, `accent`, `dialogue_orthography`, `speech_signature`). Lines 1084-1085 warn: *"this method rebuilds a FIXED row and silently drops every key it does not name."* Any `gender_source` or `gender_confidence` passed on cast rows is dropped before saving.
   Meanwhile, D5 claims: *"Ledger.set_cast drops nothing because the contract is a nested dict the ledger already stores verbatim -- verify: _otr_casting.py:2339 writes gender_by_name into the contract."*
   D5 confuses the cast rows (`ledger.data["cast"]`) with the provenance contract (`ledger.data["meta"]["cast_source_contract"]`).
   Fix: Clarify the contract in D7(c): assert `gender_source` and `gender_confidence` in `ledger.data["meta"]["cast_source_contract"]["evidence"]["ELIZABETH BENNET"]` (and Darcy). If the operator requires these fields directly on `ledger.data["cast"]` rows, update `nodes/production_ledger.py:set_cast` to whitelist and copy `gender_source` and `gender_confidence`.

7. [D4 vs input.md:96-98 & tests/test_character_gender_sidecars.py:208-212] "Equal or lower never overwrites" permanently freezes stale/defective rows and ignores text changes.
   Defect: D4 defines monotonic merge as: *"A re-run REPLACES a row only when the new rung is HIGHER than the row's gender_source or the row is absent; equal or lower never overwrites"*.
   This creates critical failure modes:
   - If an existing row was stamped at rung 2 (`pronouns`), but a code fix or text edit causes the pronoun scan to decline, Tier 3 (`llm_recall`, rung 3) or Tier 4 (`name_frequency`, rung 4) runs. Because Rung 3 and 4 are lower than Rung 2, D4 blocks them from replacing the obsolete `pronouns` row. The flawed pronoun pin is frozen permanently.
   - Because "equal never overwrites", re-running the stamper after updating pronoun scan logic or after updating the Tier 3 model/prompt in the index will never update an existing rung 2 or rung 3 row.
   - D4 ignores `body_sha256`. Prior review r2 Must-Fix 6 specifically required: *"Preserve an old known row only when the old sidecar body_sha256 equals the current body digest"*.
   - If a re-run keeps stale rows while recomputing `tier_counts` from the current pass, `tier_counts` in `gender_ladder` will disagree with the actual counts in `characters[]`.
   Fix: Re-anchor monotonicity to `body_sha256`:
   1. If `body_sha256` changed, the source text changed—re-run the ladder fresh for the unit without retaining obsolete pronoun rows.
   2. If `body_sha256` is identical, allow equal-rung updates (to pick up prompt/model revisions), and prevent only strictly lower rungs from demoting higher rungs (e.g. `llm_recall` cannot overwrite a confirmed `pronouns` row).
   3. Always derive `gender_ladder["tier_counts"]` directly from the final merged `characters` list.

---

SHOULD-FIX:

1. [D1 vs tests/test_character_gender_sidecars.py:167, 243-244] Vocabulary drift: `llm_web` vs `llm_recall` breaks `test_no_row_is_ever_UNKNOWN`.
   Defect: D1 renames Tier 3 to `llm_recall`. In `tests/test_character_gender_sidecars.py:243-244`, `test_no_row_is_ever_UNKNOWN` asserts:
   `assert row["gender_source"] in ("roster", "pronouns", "llm_web", "name_frequency")`
   Stamping `llm_recall` will fail this test. Furthermore, existing sidecars have `"llm_web": 0` in `tier_counts` (`scripts/otr_stamp_character_genders.py:167`). Renaming the key to `"llm_recall"` will flag all 65 sidecars as `substantive_change=True` on the next run even if no character rows change.
   Fix: Add `"llm_recall"` to the allowed tuple in `test_no_row_is_ever_UNKNOWN`, and explicitly migrate the `tier_counts` dictionary key across the 65 sidecars in the same commit.

2. [D1 vs Appendix A:220-224] Spec contradiction: ladder is not "total" when Tier 4 declines.
   Defect: Spec v2 Section "The ladder -- FOUR tiers, still TOTAL" claims: *"Tier 4 is the floor and it never abstains. That is what makes the ladder total without any tier having to guess beyond its evidence."* But D1 states: *"then name_frequency (config/cast_pools.gender_of_first_name on the honorific-stripped first token, accepting only male|female; unisex|unknown -> DECLINE, row omitted, the roll stays)."*
   `_FIRST_NAME_GENDER_INDEX` in `config/cast_pools.py:210` contains only ~500 curated first names. Surnames ("Scrooge"), role designations ("the Hatter", "the Creature"), non-English names, or unlisted names return `unknown`, which D1 specifies will decline and omit the row. The ladder is therefore partial/conservative, not total.
   Fix: Update Spec v2 documentation to remove the claim that Tier 4 is total and never abstains. State explicitly that Tier 4 is a conservative dictionary lookup that declines to the 40/40/20 roll on unlisted/unisex tokens.

3. [D1 vs nodes/_otr_roster_gender.py:44-48, 274-278] Incomplete honorific stripping for Tier 4 first-name lookup.
   Defect: D1 specifies running Tier 4 on the "honorific-stripped first token". `_strip_honorifics` in `nodes/_otr_roster_gender.py` is private, and its `_HONORIFICS` set (:44-48) omits common titles such as `"miss"`, `"uncle"`, `"aunt"`, `"father"`, `"sister"`, `"brother"`. For example, `"Miss Mix"` retains `"Miss"`, causing `gender_of_first_name("Miss") -> "unknown"`.
   Fix: Expose a public `strip_honorifics` function in `_otr_roster_gender.py`, add `"miss"` and familial titles to `_HONORIFICS`, and define behavior when the remaining token is a surname (e.g. `"Mrs. Sappleton"` -> `"Sappleton"` -> `unknown`).

4. [D2, D5 vs Appendix A:298-314] Ghost path: `asked_as: "bare"` has no production caller and contradicts Spec v2.
   Defect: D2 introduces `"asked_as": "title|bare"` into `character_gender_index.json`, and D5 defines confidence rules for `llm_recall bare-name -> inferred`. However, in both `public_domain_story` and `shakespeare`, every source unit has an authoritative title (`work_title` / `play_title`). The prompt builder has no code path or caller for bare-name queries, and Spec v2 line 298 explicitly forbids asking without a title ("ASK WITH THE WORK'S TITLE, NOT THE BARE NAME").
   Fix: Remove `asked_as: "bare"` and bare-name handling. All Tier 3 asks in this architecture are title-grounded (`asked_as="title"`, `gender_confidence="recalled"`).

5. [D7(c) vs .kibitz/comfyui.local.md & scripts/otr_api.py:859-875] Acceptance run omits headless canonical workflow override mechanism.
   Defect: D7(c) specifies running one public-domain episode on `pride_prejudice_proposal` published to `otr/obs/`. As documented in `.kibitz/comfyui.local.md` and r2 #11 / r3 #7, `workflows/otr_canonical.json` defaults to `source_bank: "scifi_news"`. The acceptance run must not mutate `banks.json` or create an ad-hoc graph.
   Fix: State explicitly that the headless acceptance run loads `workflows/otr_canonical.json` and overrides node 1 widgets in memory using `patch_creative(workflow, writer_id, "source_bank", "public_domain", schemas)` and `patch_creative(workflow, writer_id, "source_ref", "pride_prejudice_proposal:main", schemas)` via `scripts/otr_api.py:859`.

6. [D6 vs nodes/_otr_model_loader.py:18-20] Uncaught exception during stamping leaves 8 GB model allocated in VRAM across tests.
   Defect: D6 specifies that the model is unloaded at exit via `unload_llm`. If `load_llm` is called in unit or integration test processes without a `try...finally: LOADER.unload_llm()` block, an assertion failure leaves ~8 GB allocated in GPU memory, violating the 14.5 GiB project gate for subsequent test runs.
   Fix: Encase the model execution block in `scripts/otr_stamp_character_genders.py` in a strict `try...finally: LOADER.unload_llm()` construct, and ensure unit tests mock the generator or execute cleanup fixtures.

---

OPTIONAL / NICE-TO-HAVE:

1. [D1, D6] Structured census in stamper stdout: include a count of index hits vs LLM calls vs Tier 4 name lookups in `[gender-stamper]` output.
2. [D6] Consensus flag: add an opt-in CLI flag `--verify-consensus` in the index builder to run two passes when temperature > 0 is selected.

---

CUT THESE (over-engineering):

1. [D2, D5] Cut `asked_as: "bare"` and bare-name LLM inference.
   Why safe to cut: Every public-domain and Shakespeare unit has a known work title. A bare-name LLM prompt duplicates Tier 4 name-frequency inference with higher cost and lower reliability. Cutting it eliminates dead schema and code paths.

2. [D6] Cut two-ask consensus under greedy decoding.
   Why safe to cut: Under deterministic greedy decoding (`do_sample=False`), forward passes are mathematically identical. Running a second ask with identical inputs burns compute without providing any new signal.

---

ASSUMPTIONS:
- [ASSUMPTION] In Must-Fix 6, inferring that the operator's intended receipt verification for D7(c) is `ledger.data["meta"]["cast_source_contract"]`, matching PBUG-20260815-04 and `_otr_casting.py:2336-2344`.
- [ASSUMPTION] In Section 5.1, inferring that the operator prefers declining unlisted/ambiguous names to the 40/40/20 random roll rather than forcing ungrounded pins at render time.
- [ASSUMPTION] In Must-Fix 5, inferring that `resolve_with_supplement` is intended to emit `gender_source="supplement"` and `gender_confidence="stated"`.
