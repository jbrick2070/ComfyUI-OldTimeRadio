# Manual Antigravity (agy) prompt — R3 straggler sweep (run in parallel)

Run from the repo root. Paste the output back to me; I'll ground it against the
real code and rip any straggler I missed before the final gate.

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
agy -p "You are a fast, thorough code auditor. Read the REAL repo you are sitting in. Do NOT edit anything. We are ripping SILENT LLM->template / fail-soft fallbacks out of the writer lane so a failed writer LLM FAILS THE EPISODE LOUD instead of quietly shipping canned filler. I have ALREADY mapped these 4 sites (do NOT re-report them): (1) episode title fallback to outline.title in OTR_LedgerScriptWriter.py ~5251 + the swallow in _generate_title_from_script ~978-984; (2) announcer-outro template in _otr_line_composer.py ~3519-3524/3600-3602/3657-3662; (3) news-coda template floor in _otr_line_composer.py ~3446-3457; (4) character portrait 3-tier template in otr_meta_brief_image_prompt.py derive_image_prompts ~1113-1156.

YOUR JOB: find EVERY OTHER place in the writer / story lane where an LLM call failing or returning junk causes a SILENT substitution of a template / canned / deterministic-pool / 'floor' / 'default' output that then ships as if it were AI-written -- i.e. stragglers I have NOT listed. Sweep these files: nodes/OTR_LedgerScriptWriter.py, nodes/_otr_line_composer.py, nodes/story_orchestrator.py, nodes/otr_meta_brief_image_prompt.py, nodes/_otr_outline*.py, nodes/news_interpreter.py, nodes/_otr_story_brief*.py, and anything else in the writer path.

For each straggler report: file:line, the exact mechanism (what LLM call fails -> what canned/template/pool output silently ships), and whether it is GATED by a visible operator widget/toggle (like news_briefs_required) -- because operator-GATED degrades are INTENTIONAL and must NOT be flagged. I only want the HIDDEN ones. Also flag any bare 'except Exception ... # never break the writer' that swallows an LLM failure and substitutes canned content (vs. one that merely skips an optional computation -- those are fine to keep).

Output: a tight file:line list of HIDDEN LLM->template stragglers only, each one line. If you find none beyond my 4, say so plainly. Cite everything. Do NOT edit."
```

If `agy` complains about read permissions, add `--dangerously-skip-permissions`.
