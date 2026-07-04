# Manual Antigravity (agy) review pass — character image-role rename

Run from the repo root (`C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio`):

```
agy --model gemini-3.5-pro -p "<paste the PROMPT below>"
```

(interactive; read the review on stdout — no --dangerously-skip-permissions needed for a read-only review)

---

## PROMPT

You are an independent reviewer of an UNCOMMITTED rename in this ComfyUI custom-node repo. Review
only — do NOT edit files. Verify every claim by reading the real files. IMPORTANT: many soak scripts
under scripts/ (the `_otr_*.py` files) are git-ignored, so use ignore-blind reading (open the files
directly by path; a git-aware grep will MISS ~13 of them).

Context: the operator renamed the third IMAGE role token from `other_beats` to `character` (NOT
`character`). The image MODEL widget `character_image_model` is a SEPARATE already-done migration and
is correct as-is — leave it. The VIDEO-side `other_beats_visual` / `other_beats_video_model` mentions
are a different, deliberately-out-of-scope migration — ignore them, and ignore recorded result-data
JSON (scripts/*summary*.json, scripts/_otr_soak_capstone_results/).

Check and report:
1. COMPLETENESS — any LIVE producer of `other_beats_image`, `other_beats_granularity`, or
   `other_beats_image_model` left in nodes/**, scripts/** (INCLUDING git-ignored `_otr_*.py`),
   tests/**, config/**, or workflows/otr_scifi_16gb_full.json? Give file:line for any survivor.
2. COUPLING — is `character_granularity` fully consistent across otr_image_director.py INPUT_TYPES
   key + direct() signature param + the consumer line, AND workflows/otr_scifi_16gb_full.json node 88
   input[4] (localized_name + name + widget.name all character_granularity, widgets_values[2] ==
   "per_object")? A half-applied rename raises TypeError at runtime.
3. COLLISION — the scripts were bulk-edited (other_beats_image_model -> character_image_model, then
   other_beats_image -> character_image). Grep for `character_image_model` (must be ZERO) and confirm
   `character_image_model` wherever a widget name is expected.
4. MAPPING — does `role_overrides.character_image` in config/profiles/widget_mapping.json resolve to
   the `character_image_model` widget so apply_profile won't fail loud?

Return a one-line VERDICT (GO / GO-WITH-FIXES), then a terse numbered list: each item file:line,
finding, CONFIRMED/UNVERIFIED, concrete fix. If nothing is wrong, say so explicitly.
