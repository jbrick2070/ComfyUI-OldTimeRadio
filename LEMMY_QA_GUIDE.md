# 🔧 Lemmy Roll QA Guide

Quick procedure for validating the **Lemmy easter egg** + the **voice collision fix** in any ComfyUI-OldTimeRadio workflow.

---

## What Lemmy Is

A 100% optional easter-egg character: a grizzled, wrench-wielding engineer who appears in **11% of episodes** by random roll, named after Lemmy Kilmister. When he shows up, his first line is preceded by a signature SFX cue:
`[SFX: heavy wrench strike on metal pipe, single resonant clank]`

He has a **fixed iconic voice profile** — always `v2/en_speaker_8` (English, gravelly, mid-40s) — and should never be confused with another character.

---

## The Toggle: `summon_lemmy`

Located on **Node 1 — "1. Gemma Writes the Story"** in any workflow.

| Setting | Behavior |
|---------|----------|
| `summon_lemmy = OFF` (default) | Standard 11% RNG roll. Lemmy is rare. Use for production. |
| `summon_lemmy = ON`            | Lemmy guaranteed in every run. Use for testing only. |

---

## How to Read the Logs

After every run, search the ComfyUI console for the line starting with `[Gemma4ScriptWriter]`. You'll see one of three messages:

| Log message | Meaning |
|-------------|---------|
| `🔧 Lemmy was summoned by the boss (force toggle ON)` | You flipped the toggle. Lemmy guaranteed. |
| `🎲 Lemmy rolled in on his own (lucky 11%)` | Natural RNG hit. Lemmy is in the script. |
| `💤 Lemmy stayed in the garage tonight` | RNG missed and toggle was off. No Lemmy this run. |

---

## QA Procedure (3 minutes)

### Step 1 — Toggle ON, run test workflow

1. Load `old_time_radio_test.json` (1-min smoke test)
2. On Node 1, flip **`summon_lemmy`** to **ON**
3. Queue
4. Watch the log for: `🔧 Lemmy was summoned by the boss`

### Step 2 — Open the treatment file

After the run finishes, open the generated `_treatment.txt` from your output folder.

### Step 3 — Verify the cast block

Find the `CAST & VOICES` section. It should look like:

```
ANNOUNCER  →  v2/en_speaker_5    male · casual · warm
DRAKE      →  v2/en_speaker_3    male · gruff · weathered
JAX        →  v2/en_speaker_1    male · measured · calm
LEMMY      →  v2/en_speaker_8    male · clipped · precise
SERRA      →  v2/en_speaker_4    female · bright · energetic
```

### Step 4 — Check the three pass criteria

✅ **Pass 1: Lemmy is in the cast list**
He must appear with the name `LEMMY` (all caps).

✅ **Pass 2: Lemmy's voice is `v2/en_speaker_8`**
This is his locked iconic preset. Anything else is a regression.

✅ **Pass 3: NO other character has `v2/en_speaker_8`**
This is the **voice collision fix**. Drake must not share Lemmy's voice. Jax must not share it. Nobody but Lemmy.

### Step 5 — Verify Lemmy's signature SFX

In the `FULL SCRIPT` section, find Lemmy's first line. Immediately before it, you should see:

```
[SFX: heavy wrench strike on metal pipe, single resonant clank]
```

This SFX should appear **exactly once** — never before subsequent Lemmy lines.

### Step 6 — Verify Lemmy's character voice

Lemmy speaks in blunt, colorful, mechanic-style metaphors. Sample known-good lines:
- *"Protocols are made for labs, not voids."*
- *"Clearance expired when the math broke, Doc."*
- *"Hold steady — it'll THUD before it settles."*

If Lemmy sounds formal, scientific, or like a regular character, the personality directive isn't landing. Flag for re-prompt review.

---

## Failure Modes & Fixes

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Toggle ON but Lemmy missing from script | Gemma ignored the directive | Re-queue; increase prompt weight in `lemmy_directive` |
| Lemmy present but voice ≠ `v2/en_speaker_8` | `_LEMMY_PROFILE` corrupted or VoiceHealth disabled the preset | Check boot log for `VoiceHealth: LEMMY preset disabled` |
| Lemmy + another char both `v2/en_speaker_8` | Voice collision fix regressed | Verify two-pass cast iteration in `gemma4_orchestrator.py` ~line 3005 (LEMMY/ANNOUNCER processed first) |
| Wrench SFX appears before EVERY Lemmy line | Prompt rule didn't enforce "first line only" | Re-tighten the SFX rule in `lemmy_directive` |
| Lemmy fires every run with toggle OFF | Browser/session widget state cached | Hard-refresh tab (Ctrl+Shift+R), reload workflow JSON from disk |

---

## Reset Procedure (when in doubt)

1. Set `summon_lemmy = OFF` on Node 1
2. Save workflow to disk
3. Hard-refresh the ComfyUI tab (Ctrl+Shift+R)
4. File → Open → reload the workflow JSON
5. Confirm toggle reads OFF
6. Queue 5 runs in a row
7. Statistically expect Lemmy in 0–2 of them (11% × 5 ≈ 0.55)

If Lemmy fires in 4+/5 runs with the toggle confirmed OFF, the bug is real and needs investigation.

---

## Production Defaults

| Workflow | `summon_lemmy` default |
|----------|------------------------|
| `old_time_radio_test.json` | OFF |
| `old_time_radio_scifi_lite.json` | OFF |
| `old_time_radio_scifi_full.json` | OFF |

Lemmy stays a rare easter egg in production. The toggle is a QA tool only.
