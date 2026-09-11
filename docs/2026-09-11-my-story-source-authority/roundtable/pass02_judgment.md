# R3 grounded judgment

Opus5/Gemini3.1 Pro completed real API reviews, about USD0.3247. Gemini found
no defects. Opus raised the following; root grounds and decides each. No code yet.

| Claim | Decision and evidence |
|---|---|
| Corrector cannot see author target / duplication | Corrector DOES receive exact author_context in quoted data. R3 wording overstated the lack of context. Retain static system scope as an instruction governing that data; author scope is also visible as historical context. This duplication is intentional role/context separation, not two raw-source blocks. No additional calls/data representation. |
| Dynamic ending injected into system / caps proposed | Accept removing dynamic story text from system instruction. Final scope is now static and refers to the explicit target in existing quoted context. Keep actual ending only in author user target (and original treatment JSON). No arbitrary length cap, normalization, extra fit policy or gate. Existing capacity contract remains. |
| Local endpoint lost | Misread: full accepted treatment still includes local ending_state. Final target governs conflicts only. Preserve treatment snapshot and compatible detail; no second local labelled slot necessary. Semantic outcome remains unproved. |
| Stripping changes content / multiline parsing | Accept literal preservation: select original treatment.ending only if is_last and ending.strip() nonempty; do not bind the stripped version. Tests use complete delimited prompt suffix, allowing newline, percent signs and marker-like strings. No flattening/truncation of ending. |
| Private kwargs / future silent override | _pass_act's **source_kwargs is a fresh Python dict, distinct from orchestrator's source_kwargs(model_id) function. Its known contents are journal/scheduler/model. Act owner derives authoritative scope. No actual collision/loss and no extra assertion rejecting hypothetical callers. Scope leakage to later phases is tested through real runner. |
| Author/corrector need different role wording | Same static artifact obligation is coherent with correction: return corrected ActScript under existing schema, preserve original source, unchanged if already correct. No verdict request or style command. Do not weaken omission restoration into only retaining an already-present ending. |
| Final act may cram earlier events | Add short static permission that earlier events need not repeat to both final branches. Common partial-artifact outside-scope clause also applies. No claim this guarantees model decisions. |
| Whole/frame scope undefined | Existing author_context and artifact schema define each role; no new frame-specific behavior is claimed. The common prompt edit intentionally affects existing My Story correction consumers. No new per-phase enum, scope object, schema or model pass. |
| P0 strength default incompatible | Default required is conservative and unchanged; optionality definition guides explicit model labels. No deterministic enforcement claimed. Existing optional metadata/omission tests stay. |
| P0/P1 text missing / redundant | Quote exact text in R4. P1 coherence corrects the observed upstream plan mismatch; final target consumer is still required when model fails to align it. They are two existing owners, no new subsystem. |
| CPU tests cannot prove behavior | Already disclosed. Tests prove changed instructions/routing/retention; subsequent full canonical measures actual fidelity. No fixed-seed harness substituted for real canonical. One result cannot prove reliability/causality, and failure triggers diagnosis rather than lucky repeat. |
| Negative coverage | Add unchanged other-phase correction scope and multiline/source-marker fixtures to actual runner route checks. Journal-free author-only behavior remains intentional; do not add fake failure receipts for a correction not invoked. |
| Baseline51 failures hide regressions | Compare exact failure IDs AND normalized failure payloads, not just count. Focused source/call tests must pass. Any changed failure is investigated. Existing failures remain explicitly disclosed, not declared green. |
| No new production bug IDs penalizes tests / required checks ceremony | Operator admission rule permits PBUG/Bible only with live evidence. Existing live05/06 qualifies extension of11.39; fixtures do not create new production incidents. Full tests/Bible/canonical/encoding/Sonnet/push are explicit operator requirements; preserve them. |

Final convergence scope: static existing-owner instructions, preserved exact data,
unconditional final target selection, conserved other phases and fixed budgets.
No new architectural branch remains. R4 reviews the final exact wording below.
