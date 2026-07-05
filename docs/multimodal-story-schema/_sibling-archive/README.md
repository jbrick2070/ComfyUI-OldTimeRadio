# OTR Transplant Workspace (formerly Upstream Story Lab)

Status: transplant workspace. Cleared and rebuilt 2026-07-02 from the
SFX-free production baseline. The v1 standalone lab (contracts, catalogs,
fixtures, preview nodes, tests) lives in git history at commit `41c6512`;
recover any piece of it with `git show 41c6512:<path>`.

## What this folder is now

```text
production_mirror/   pristine copies of the production files the transplant
                     will touch, pinned to ComfyUI-OldTimeRadio commit
                     d48a9d76 (post rip-sfx-broll). Read-only reference.
workflows/           the EDITABLE working copy of the SFX-free canonical
                     workflow (otr_scifi_16gb_full.json). Transplant edits
                     are staged and validated here first.
docs/                planning and review artifacts, including the grounded
                     final review (FABLE_FINAL_REVIEW_2026-07-02.md) whose
                     gates control when production may be edited.
kibitz-runs/         review-run history from the v1 lab phase.
```

See `PRODUCTION_MIRROR_MANIFEST.md` for the exact file list, hashes, and the
drift-check rule.

## Hard rules (unchanged)

- Production `ComfyUI-OldTimeRadio` is not edited until the transplant chunk.
- No hidden fallbacks; unknown source/story/style ids fail loudly.
- JSON owns content and configuration; Python owns validation, routing,
  execution, and fail-loud errors.
- Workflow JSON edits are append-only widgets + forceInput sockets, applied
  only after every gate in the final review passes.
- The old SFX surface is deleted (rip-sfx-broll 6bad6e5b); nothing here may
  reintroduce the `sfx` speaker role, `scene_broll`/`background_abstract`
  video roles, or `[SFX:]` tokens.

## Where the plan lives

1. `docs/FABLE_FINAL_REVIEW_2026-07-02.md` - verdict, must-fixes, revised
   transplant plan, validation gates.
2. `PROMPT_SURGERY_CHECKLIST.md` - per-file prompt surgery map.
3. `TRANSPLANT_MANIFEST.md` - transplant gates and staging rules.
4. `production_mirror/docs/...` - the production audit docs (line-level
   prompt site maps) at the mirrored baseline.
