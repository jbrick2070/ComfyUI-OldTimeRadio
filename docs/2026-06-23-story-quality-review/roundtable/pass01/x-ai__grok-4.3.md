<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: yes-with-fixes. Plan is coherent on deterministic levers but L1 prompt edits contradict the "re-instructing adds nothing" premise and several code claims rest on unshown sources.

MUST-FIX BEFORE BUILD:
1. [Core finding, para 2] Claims that exact "ACTION UNDER PRESSURE ... reversal" text and verb menu already exist at _otr_outline._build_beat_user_prompt:1166 and _otr_line_composer:1065 cannot be asserted here; verify: actual prompt strings and line numbers in both files before any edit.
2. [L1 (a)] Denylist + post-outline cap that triggers beat-INTENT regeneration creates a new regeneration path; add explicit guard that this path is non-interactive and counted in meta.compose_flags so it does not become a hidden critic loop.
3. [L3] Composer {internal_action, spoken_dialogue} split is audio-affecting; the plan must name the exact flag name, default value, and re-baseline commit procedure before any implementation.
4. [L2] Required personal-stake + climax slots are added to EpisodeBudget.arc_phases; the monotonic validators must be updated in the same change or phase ordering can silently drop the new slots.

SHOULD-FIX:
1. [The lever set intro] L1 still edits prompts (denylist, palette, verb menu); either remove L1 or reconcile the sentence "re-instructing the model will add nothing".
2. [L5] "fix the critic too_many_edits abort" is a harness change; confirm it touches only grading/telemetry and never the frozen ledger schema.
3. [Build order] Re-soak is scheduled only after L1+L2; insert a minimal matrix run after L5 alone to catch any default-writer side effects before structural changes land.

OPTIONAL / NICE-TO-HAVE:
- Expose the crisis-noun cap count in meta so later analysis can measure L1 effect size without new fields.

CUT THESE:
1. L6 best-of-N — safe to drop for v0; it is explicitly secondary, adds N generations, and cannot fix the structural sameness the plan identifies as root cause.
2. Any change to workflows/otr_scifi_16gb_full.json — already forbidden by constraint 3 and not required by any lever.

[ASSUMPTION] Plan assumes gemma-12b is already installed and selectable as default writer with zero environment or node changes.