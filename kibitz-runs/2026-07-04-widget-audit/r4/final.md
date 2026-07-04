# r4 convergence judgment (re-verify vs HEAD 8c3e4911)
Arc complete: r1-r3 (2026-07-03 folder) + this r4. Panel: codex (gpt-5.5, 4 calls) + antigravity manual (r1-r3 scope; reverify packet issued, pending). Claude = anchor + judge throughout.

## r4 judgment log
- MF1 (tail order with node 95): ACCEPTED. Canonical 86-owner chain = 84 -> 93 -> 86 -> 95 -> 85 (captions after final composition, BEFORE credits; credits frames caption-free; node 95 declared-tail -> 85 slot 6 untouched). Design judgment corroborated by the code reads; agy reverify asked the same question independently -- fold its answer in when it lands.
- MF2 (output contract :183-192) CONFIRMED (judge re-read at HEAD) -- carried.
- MF3 (ledger resolution + sibling audio/ fallback :98-115) CONFIRMED -- carried, now includes the fallback port.
- MF4/CUT1 (enablement): ACCEPTED with codex's cut -- canonical workflow/profile wiring is the single enablement path; the env-only alternative is dropped.
- MF5 (strict-types CLI gap) CONFIRMED by judge read of tools\validate_workflow_links.py:61-81 vs the dynamic merge (__init__.py + _otr_class_registry.py). Nodes 80-83 ARE registered at runtime; the CLI misses new_node_modules_table(). New standalone fix item.
- SF1 (docstring cite :176-182 not :183-188) CONFIRMED by judge read -- corrected.
- SF2 (CreditsRoll registration cite) accepted as stated.
- SF3 (stale positional comment :823-826) accepted.
- No new must-fix beyond the above; no r1-r3 judgment overturned. CONVERGED.

## Line-cite refresh at HEAD
All previously cited lines re-verified stable except _otr_voice_node_common docstring (now :176-182). The no-fallback rip did not touch the audited caption/voice files.
