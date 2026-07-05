# r2 Review -- Sonnet panelist (coding plan / implementability)

## VERDICT: NO-GO -- send back, plan is not code-ready

The r1 synthesis that r2 must elaborate contains at least one factual error baked into its single most load-bearing item (MF-C1), an invented seam vocabulary in MF-C3, and MF-C6's "empty-override" pattern is not proven safe for "every seam" -- it is safe only for seams a bank's `required_seams` list excludes, which is an implicit property of `banks.json`, not a stated rule. r2 cannot produce a correct chunk plan on top of a mis-specified foundation without first correcting the foundation.

### 1. Chunk ordering for MF-C1

MF-C1's premise -- that `_otr_line_composer.py` has an `is _SYSTEM_PROMPT` identity check symmetric to outline's -- is **wrong**. Grep across all of `nodes/` for `is _SYSTEM_PROMPT|is _MODERN` returns exactly **one** hit: `_otr_outline.py:1847`. `_otr_line_composer.py:2061` does `system = _SYSTEM_PROMPT` (direct assignment) gated on `creative_repo_id is None` (line 2060) -- a different, simpler contract with no identity comparison to break. This means MF-C1's fix is only load-bearing for **outline**, and the router (`_otr_creative_prompt_router.py:43-46,61-64`) imports `_SYSTEM_PROMPT` from both files as singletons at module-import time -- so any chunk that runs those two files' module bodies once and doesn't reassign the names is safe, without needing a merged "atomic" chunk. r2's ordering question is easier than r1 posed it, but r2 must first correct MF-C1's premise, not silently build a chunk plan atop the wrong claim.

### 2. Extractor helper signature (MF-C4)

`get_pack_prompt_or_none(bank_id, seam_key) -> str | None` is workable for the literal-passthrough case (production keeps calling its own constant when `None`) but the plan never specifies the **empty-string vs None** distinction at the call site. Lab's `runner.py:65` and `registry.py:170,189` both use `pack.prompt_stages.get(seam, "").strip()` truthiness -- meaning the lab's own contract treats **absent-key and empty-string identically** (both falsy). If Phase A's extractor returns `""` for the empty-override case rather than `None`, and production code does `override or LITERAL`, both work identically for falsy values -- but the plan doesn't pin which one the extractor returns, and `MF-C6`'s worked example returns literal `""` strings (`test_transplant_modules.py:70-77`), not `None`. r2 must pin one, not both, or downstream `if resolved is _SYSTEM_PROMPT` (MF-C1) breaks: `"" or _SYSTEM_PROMPT` returns `_SYSTEM_PROMPT` by identity (safe), but `None or _SYSTEM_PROMPT` also returns `_SYSTEM_PROMPT` by identity (also safe) -- so this particular seam happens to survive either choice, but the signature contract must still be pinned in writing.

### 3. Empty-science-overrides (MF-C6) -- does not hold for every seam as written

Grounded at `fixtures/banks.json:24-31`: `science_news`'s `required_seams` is **6 items** (`outline_system, pitch_room_system, dramatic_state_system, line_grounding, coda_system, title_system`), deliberately **excluding** `style_pick_inventor`, `style_pick_chooser`, `style_pick_chooser_user_template`, `story_select_system`. `registry.py:168-176` raises `RegistryError` if any `required_seams` member resolves falsy (`.strip()` on `""` is falsy) -- so an **explicit empty-string override on a `required_seams` member fails loud at load time**, it does not "stay byte-identical." MF-C6's claim that the pattern "extends to ALL 14 template seams" is false as stated: it only works because the excluded seams are simply **absent from the science pack's JSON** (confirmed: `science_news_default.json` has 7 keys, no `style_pick_*` keys at all -- absence, not empty-string). r2 must restate MF-C6 as "empty-string override for non-required seams; omission (not override) for anything in a bank's `required_seams`" -- a materially different, more fragile rule that depends on `banks.json` staying in sync with any Phase A extraction.

### 4. 14-seam vocabulary in contracts.py

MF-C3's 4 seams-to-add are half-fabricated. `outline_macro_system`/`outline_phase_system`/`outline_beat_system` do not exist as identifiers anywhere in production; the real constants are `_otr_outline.py:1102 _MACRO_SYSTEM_PROMPT`, `:1115 _PHASE_SYSTEM_PROMPT`, `:1130 _BEAT_SYSTEM_PROMPT`. The lab-name mapping table MF-C3 asks r2 to "publish" doesn't exist yet even in concept -- r2 must derive real seam names from these three constants (e.g. `outline_macro_system` -> `_MACRO_SYSTEM_PROMPT`), not assume the lab's proposed names are already grounded. `line_composer_system` (real, `_otr_line_composer.py:1174`) is fine.

### 5. Other invariant at risk

The plan's docstring at `_otr_creative_prompt_router.py:15-19` claims "four phases" (adds `polish_character`, `polish_announcer`) neither implemented in `Phase` Literal nor `_MODERN_BY_PHASE` -- stale/aspirational text that could mislead a chunk author into thinking those phases are already wired. r2 should flag this as a doc-hygiene fix, not in scope for Phase A code.

## Grounding table

| claim | file:line | status |
|---|---|---|
| `resolved is _SYSTEM_PROMPT` check | `_otr_outline.py:1847` | CONFIRMED |
| `_SYSTEM_PROMPT` module-level literal (outline) | `_otr_outline.py:532` | CONFIRMED |
| Router imports both `_SYSTEM_PROMPT`s as singletons | `_otr_creative_prompt_router.py:43-46` | CONFIRMED |
| `_MODERN_BY_PHASE` built at import time | `_otr_creative_prompt_router.py:61-64` | CONFIRMED |
| line_composer has NO identity check; direct assign instead | `_otr_line_composer.py:2060-2061` | CONFIRMED (r1's "16-site/symmetric" framing is MISREAD) |
| Only 1 `is _SYSTEM_PROMPT`-class site in nodes/ | repo-wide grep | CONFIRMED |
| `TEMPLATE_SEAMS` 14 entries incl. 4 experimental | `contracts.py:25-42` | CONFIRMED |
| `outline_macro_system` etc. not literal production names | grep `_otr_outline.py` | MISREAD in r1 (real names: `_MACRO_SYSTEM_PROMPT:1102`, `_PHASE_SYSTEM_PROMPT:1115`, `_BEAT_SYSTEM_PROMPT:1130`) |
| science_news `required_seams` excludes style_pick_* | `fixtures/banks.json:24-31` | CONFIRMED |
| Empty-string on `required_seams` member raises `RegistryError` | `registry.py:168-176` | CONFIRMED -- MF-C6 as stated is FALSE for required seams |
| science_news pack has 7 keys, no style_pick_* (absent not empty) | `science_news_default.json` | CONFIRMED |
| `_INVENTOR_SYSTEM`/`_CHOOSER_SYSTEM` no placeholders; `_CHOOSER_USER_TEMPLATE`/`_INVENTOR_USER_TEMPLATE` have `.format()` placeholders | `_otr_style_picker.py:296,301,329,334` | CONFIRMED |
| Production OTR HEAD is `c98a67ab...`, not `a7bdc42d` | `.git/refs/heads/v2.0-alpha` | CONFIRMED mismatch -- MF-C5 baseline pin is currently WRONG in the working tree |
| Lab HEAD `7df7c805...` | `.git/refs/heads/main` | CONFIRMED matches `7df7c80` |

## MUST-FIX for r2

- **MF-R2-1:** Correct MF-C1's premise before chunking: only outline has an identity check; line_composer's contract is different (direct assign, no comparison). Chunk plan must not invent a "merge both files" requirement that isn't needed for line_composer's safety, though `line_composer_system` extraction (MF-C2) is still separately valid as a seam to add.
- **MF-R2-2:** Restate MF-C6: empty-string override is safe ONLY for seams outside the consuming bank's `required_seams`; anything in `required_seams` must be omitted from the pack (or extraction deferred), never empty-overridden, or `RegistryError` fires at load.
- **MF-R2-3:** Replace the 3 invented outline sub-seam names with the real production constants (`_MACRO_SYSTEM_PROMPT`, `_PHASE_SYSTEM_PROMPT`, `_BEAT_SYSTEM_PROMPT`) before publishing the seam table.
- **MF-R2-4:** Re-verify/re-pin MF-C5: production OTR working tree HEAD does not match `a7bdc42d` as claimed; r2 must re-pin against actual current HEAD or explicitly justify the stated SHA.

## SHOULD-FIX

- Pin extractor return contract to `None` (not `""`) for "use production literal," to avoid ambiguity with the seams that ARE meant to be empty-string per-pack content.
- Flag `_otr_creative_prompt_router.py:15-19` docstring as stale (claims 4 phases, only 2 wired) -- Phase A doc hygiene, not blocking.
