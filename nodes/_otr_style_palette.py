"""Single source of truth for OTR style slugs.

Consumed by:
  - musicgen_theme._STYLE_PALETTE (per-cue prompts)
  - OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL (writer pool)
  - _otr_ledger_freeze._check_meta_invariants
    (freeze-time slug validation; rejects writer drift before consumers see it)

Drift between these three is pinned by tests/test_style_palette_drift.py.

S25 / MG-6 (BUG-LOCAL-216). Hoisted from musicgen_theme.py:77-159 +
OTR_LedgerScriptWriter.py:222-233 after the two parallel lists drifted in
soak. Going forward the two consumer surfaces re-export from here so a
single-file edit keeps all three in lockstep.
"""
from __future__ import annotations

# Era-neutral cue prompts keyed by snake_case style slug. Unknown slug is
# a hard fail; no default palette survives. Mirrors the writer's seed pool
# 1:1. The drift test pins set-equality across the three surfaces.
STYLE_PALETTE: dict[str, dict[str, str]] = {
    "closed_room_suspense": {
        "opening":      "slow muted strings, low piano cluster, sparse texture, "
                        "rising tension on a single chord",
        "closing":      "diminished chord fade, slow exhale of low woodwind, "
                        "resolves to silence",
        "interstitial": "single sustained bass note, room tone bed, "
                        "minimal percussion, brief and unresolved",
    },
    "detective_case_file": {
        "opening":      "methodical minor piano motif, walking double bass, "
                        "soft brushed snare, procedural and unhurried",
        "closing":      "piano cadence resolving down a minor third, "
                        "single cymbal bell tap, settles cleanly",
        "interstitial": "short walking bass figure, two-bar minor piano "
                        "phrase, professional and brief",
    },
    "pulp_serial_cliffhanger": {
        "opening":      "bold orchestral brass theme, full strings, "
                        "rolling timpani, theatrical and dramatic, "
                        "ends on a held suspended chord",
        "closing":      "orchestral hit, dramatic crescendo into snare "
                        "roll, resolves on a held major chord",
        "interstitial": "brass stab, quick timpani roll, single "
                        "high cymbal accent",
    },
    "mission_control_procedural": {
        "opening":      "clean synth pulse, instrument-panel beep accents, "
                        "tight arpeggiated bass, calm institutional energy",
        "closing":      "descending synth bleeps, soft pad release, "
                        "console-shutdown texture",
        "interstitial": "single synth bleep sequence, quiet hum bed, "
                        "very short",
    },
    "deep_space_distress_call": {
        "opening":      "modal synthesizer pad, distant pulsing beacon, "
                        "sub-bass swell, isolated and vast, slow build",
        "closing":      "isolated low pad, signal dropout, granular "
                        "static decay into silence",
        "interstitial": "single sustained tone, faint radio interference, "
                        "brief and dimensional",
    },
    "noir_interrogation": {
        "opening":      "muted solo trumpet, low double bass walk, "
                        "smoky tenor saxophone, dim and atmospheric",
        "closing":      "muted trumpet fade, low bass note hold, "
                        "single piano chord trailing off",
        "interstitial": "single muted trumpet phrase, sparse bass, "
                        "smoky and short",
    },
    "small_town_uncanny": {
        "opening":      "diffuse string pad, muted bell tone, slow harmonic "
                        "drift, quiet wrongness in the mid register",
        "closing":      "soft string fade, single high bell strike, "
                        "decay into ambient hush",
        "interstitial": "single muted bell, faint string pad, brief and "
                        "off-balance",
    },
    "radio_newsroom_emergency": {
        "opening":      "urgent ticker percussion, telegraph rhythm pattern, "
                        "tight low brass pulse, motion and momentum",
        "closing":      "decisive low brass figure, snare hit, fast cadence "
                        "to a resolving chord",
        "interstitial": "short telegraph rhythm, single brass accent, "
                        "newsroom urgency",
    },
    "haunted_broadcast_signal": {
        "opening":      "degraded signal artefacts, ghostly choir-like synth pad, "
                        "granular static texture, distant and unstable",
        "closing":      "pad decay through layered static, fading into "
                        "noise floor and silence",
        "interstitial": "short granular swell, signal flutter, brief "
                        "ghosted texture",
    },
    "laboratory_containment": {
        "opening":      "sterile electronic tone, precise arpeggio, "
                        "high pure sine layer, clean and isolated",
        "closing":      "tone descends to a held low frequency, single "
                        "click, controlled fade",
        "interstitial": "short electronic pulse sequence, sterile and "
                        "precise",
    },
}


# Public set used by validators (freeze cascade, drift tests). Frozenset
# so callers cannot mutate the contract.
KNOWN_STYLE_SLUGS: frozenset[str] = frozenset(STYLE_PALETTE.keys())


__all__ = ["STYLE_PALETTE", "KNOWN_STYLE_SLUGS"]
