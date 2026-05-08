# Round A -- ChatGPT (gpt-5.5) elapsed=80.2s

I’m reviewing the pasted snippets, not a live checkout, so I won’t invent repo line numbers. References below are to the specific functions/expressions shown, e.g. `assignments[name]` inside `build_contract_from_director_plan`.

| # | Element | Fix-needed probability | Badge | One-line reasoning | Failure mode to watch in full soak |
|---:|---|---:|:---:|---|---|
| 1 | `CastContract.stamp_version` | **18%** | **AMBER** | The hash is deterministic and content-addressed, but version freshness depends on callers remembering to re-stamp after any alias/character mutation. | Alias is applied after initial stamp, ledger lines carry the old `sha:...`, and merge rejects otherwise valid episode output as a contract-version mismatch. |
| 2 | `lock_to_episode` | **40%** | **RED** | The “existing lock is always fatal” rule is clean architecturally but brittle under ComfyUI reruns, retries, resumes, or partial episode regeneration. | Bark health check passes, `cast_contract.locked.json` is written, then the graph/node is re-executed for the same `episode_dir` and the run hard-fails before production. |
| 3 | `parse_voice_spec` | **24%** | **AMBER** | Forward-compatible unknown-engine pass-through is intentional, but it also lets engine typos become late backend-dispatch failures. | Director/helper emits `brak:v2/en_speaker_5`; contract accepts it, version locks it, and the actual voice backend later errors or falls back incorrectly. |
| 4 | `build_contract_from_director_plan` | **55%** | **RED** | The helper normalizes assignment keys into `sorted_names` but then indexes the original dict with the normalized key via `assignments[name]`, which is a real integration trap. | A plan contains `{" Monty ": "v2/en_speaker_3"}` or a non-string key; `name` becomes `"Monty"`, then `assignments[name]` raises `KeyError`. |
| 5 | `detect_aliases` | **46%** | **RED** | The 4-character prefix heuristic is useful as a fast path, but first-match-wins can silently assign the wrong `character_id` before §4 exists to arbitrate. | Contract has `MARTIN` and `MARTHA`; script tag `MART:` or `MARTINDALE:` matches whichever character appears first, potentially locking a wrong alias/voice. |
| 6 | `_extract_dialogue_tags` | **42%** | **RED** | The extractor is deliberately conservative, but common OTR/script forms — especially `NARRATOR:` and mixed-case tags — can be invisible to the whole alias/orphan pipeline. | A real script contains `NARRATOR:` or `Dr. Morgan:`; the tag is not returned, so neither `detect_aliases` nor future orphan classification sees the drift. |

## Where I’d push back

1. **The §2 lock behavior is too absolute for a ComfyUI workflow.**  
   “Never overwrite a differing contract” is strong; “any existing lock is fatal” is likely to collide with normal graph reruns, crash recovery, or acceptance-soak repetition.

2. **First-match-wins aliasing is dangerous if its output can be applied automatically.**  
   As a diagnostic fast path it is fine; as a source of truth it can recreate the exact class of silent wrong-voice pooling the Cast Contract is meant to prevent.

## Load-bearing weak spot

**Element 6, `_extract_dialogue_tags`.**

If a tag is not extracted, everything downstream is blind: no heuristic alias, no orphan classification, no repair loop later. The explicit `NARRATOR` exclusion is especially risky for an OTR system, because narrator is often a real spoken role, not just a structural header.
