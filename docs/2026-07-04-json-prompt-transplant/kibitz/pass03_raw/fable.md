# r3 Review -- Fable (wiring / integration / sequencing)

## VERDICT
**NOT WIRE-READY -- GO-WITH-FIXES.** Audio byte-identity is trivially safe (chunks 0-6 touch zero OTR production .py). The real break is **lab-suite green at the Chunk 4 boundary**, plus a hand-wavy Chunk 5.

## MUST-FIX

**MF-W1 -- MF-C6's "populated OR OMITTED" rule is a misread; Chunk 4 as written goes RED at registry construction.** `registry.py:169-170` does `pack.prompt_stages.get(seam, "").strip()` -- omission and empty-string are *identical*, both raise RegistryError for non-experimental packs. Chunk 4's "passthrough: omit all keys" strips science's 6 required seams -> `Registry(ROOT)` fails -> every registry-fixture test (`conftest.py:16-20`) fails, including `test_science_profile_leaves_style_picker_constants`. Fix: the `banks.json:24-31` science `required_seams` change must land **in the same commit as Chunk 4** (or fold into Chunk 2).

**MF-W2 -- Chunk 2 scope too narrow and carries an unresolved fork.** Relaxing only `profiles.py:60-65` (line_grounding) does not unblock Chunk 4; `registry.py:167-176` is the earlier gate for the other 5 required seams. Chunk 2's "alternatively..." is an operator decision blocking Chunk 4 -- resolve it before Chunk 3, not mid-build. (Note: `StoryPromptProfile.line_grounding_instruction` at `contracts.py:266` is a required `str`; after relaxation resolve_profile must pass `""`, which validates -- spec the hunk.)

**MF-W3 -- Chunk 5 capture mechanics are the hand-wavy chunk.** "Assembled stage system produced by the current writer" is not importable -- `_make_system` is a closure at `_otr_outline.py:1854-1857`; capturing it live means running the writer with a stubbed `structured_call` (no fixture spec given). The repo's established mechanism is AST/text extraction from `production_mirror` (`test_compat_drift.py:27-52`, `mirror_nodes` fixture) -- snapshot the **constants** (`:1102/:1115/:1130/:532`; overlay is None at default per `:1832/:1847`), against the Chunk-0-refreshed mirror. Separately, the identity pytest `resolve_creative_system_prompt(...) is module._SYSTEM_PROMPT` requires live import of OTR's `nodes` package from the sibling suite -- cross-repo, heavy imports, no `OTR_TEST_MODE` in sibling conftest. Put that one test in OTR's own suite.

## SHOULD-FIX
- **Rollback story absent** (r3 asked; plan silent). Each chunk = one commit -> prescribe `git revert <sha>` + push + rerun suite; never force-push `main`/`v2.0-alpha`.
- Chunk 7 says "push per chunk to v2.0-alpha (OTR)" but chunks 0-6 change no OTR code -- state that OTR pushes are kibitz-docs-only to avoid empty commits.
- Doc contradiction: MF-C6 corollary says "adds 3 new seams, not 4"; Chunk 1 and the revised total say 4 new keys -> 14. Reconcile.
- State explicitly that IS_CHANGED/import-order is **moot in Phase A**: extractor is lab-side; lab has no ComfyUI node module (src/ = 9 files, none node-facing).

## CLEAN
Chunk 1 is safe: `TEMPLATE_SEAMS` referenced only at `contracts.py:185/:232/:351` (keep union); no fixture uses the runtime variables (grep: zero hits), so the SEAM_RUNTIME_VARIABLES fix breaks nothing; new keys are add-only.

## Grounding table
| claim | evidence | status |
|---|---|---|
| Omitted required seam fails like empty | `registry.py:167-176` `.get(seam,"")` | CONFIRMED (plan's MF-C6 "OR OMITTED" = MISREAD) |
| science required_seams = 6 | `banks.json:24-31` | CONFIRMED |
| science pack 7 keys, omits style_pick_* | `science_news_default.json:7-15` | CONFIRMED |
| profiles.py hard-errors empty line_grounding | `profiles.py:60-65` | CONFIRMED |
| Only identity check at outline | `_otr_outline.py:1847` (sole grep hit); `_otr_line_composer.py:2060-2066` direct assign | CONFIRMED |
| Constants :532/:1102/:1115/:1130 | grep `_otr_outline.py` | CONFIRMED |
| Router singleton imports | `_otr_creative_prompt_router.py:43-64` | CONFIRMED |
| Mirror extraction is established pattern | `test_compat_drift.py`, `conftest.py:23-25` | CONFIRMED |
| OTR on v2.0-alpha @ 7655ead0; lab on main @ 7df7c80 | `.git/HEAD` + refs, both repos | CONFIRMED |
| Chunks 0-6 preserve OTR audio byte-identity | no OTR production .py in any chunk | CONFIRMED |
| Chunk 5 "live writer" fixture setup | not specified anywhere in plan | UNVERIFIABLE (unspecified) |
