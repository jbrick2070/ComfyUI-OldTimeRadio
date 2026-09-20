# Close-out audit -- HEAD a7697898, 2026-09-20

Reviewer: Gemini 3.1 Pro. Read-only audit per operator brief; findings grounded at file:line.

## MUST-FIX before ship

| Location | What is wrong | Exact fix |
|---|---|---|
| 
odes/cast_lock.py:1944 | The normalizer drops the oice_preset leftover when engine != "bark", but the local Bark adapter is "chatterbox", causing 	est_lemmy_provisional_tier.py to fail | if str(getattr(ref, "engine", "") or "") not in ("bark", "chatterbox"): |
| .comfyignore:EOF | Excluded directories scratch/, kibitz-runs/, and secrets/ are missing from the list, risking leaking them in the shipped registry zip | Add scratch/, kibitz-runs/, and secrets/ to .comfyignore |
| 
odes/OTR_LedgerScriptWriter.py:6376 | A cp1252 console dies on the non-ASCII section sign (§) inside the log string | Remove the § character from the log.info string |

## SHOULD-FIX

| Location | What is wrong | Exact fix |
|---|---|---|
| 
odes/_otr_passage_selector.py:204 | _line_tokens was added with no caller outside tests and its own module (orphan helper) | Remove the function or wire it correctly if intended for outside use |

## DOCS-ONLY

| Location | What is wrong | Exact fix |
|---|---|---|
| 
odes/OTR_LedgerScriptWriter.py:2467 | lemmy_cameo tooltip states "'always' / 'never' consume one of the num_characters slots", which contradicts ssemble_pre_locked_rows where 
ever does not consume a slot | Replace with "'always' consumes one of the num_characters slots, exactly as a natural roll does." |

## NOTHING FOUND

| Checklist item | Result |
|---|---|
| **TODO/FIXME/XXX added today** | nothing found |
| **Debug print in changed .py** | nothing found (added print lines in scripts/otr_vendor_scan.py are CLI diagnostics) |
| **Hard-coded absolute path in changed .py** | nothing found |
| **README.md and pyproject.toml numbers vs tree** | nothing found: twenty-one saved graphs, eight admitted languages, 43 Shakespeare scenes, and 25 nodes all match the current file tree perfectly. |

---

**ONE THING before Reddit:** 
The single most important finding to fix is .comfyignore missing the secrets/ directory. If the repository is published and installed via the ComfyUI Registry, any local secrets/ directory containing API keys or private data would be zipped and distributed to the world. Leaking credentials to the public internet is an irreversible security breach that eclipses any functional bug, and preventing it requires only a one-line addition to the ignore list.

COUNTS: must=3 should=1 docs=1
