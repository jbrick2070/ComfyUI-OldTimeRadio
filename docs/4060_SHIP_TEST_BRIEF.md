# BRIEF: the 4060 zero-friction ship test (2026-09-05)

**For the Claude window running ON the 4060.** The 5080 window wrote this and owns
the shipping surface; you own the portability surface. Read the whole thing before
you delete anything.

---

## 1. WHY THIS RUN EXISTS

`2.0.0-alpha.23` went **Active** on the Comfy Registry tonight -- the first
installable version since the alpha.13/.14 ban. Every earlier attempt at this test
was blocked on exactly that.

You are answering the one question the dev box **structurally cannot answer about
itself**: does this pack work on hardware it was not written on, installed the way
a stranger installs it?

**THE FRICTION IS THE MEASUREMENT.** You are not here to make it work by hand. Every
hand step you need is a DEFECT to be recorded and fixed at the root. A run that
"passes" because you fixed six things in the terminal has measured nothing.

---

## 2. WHAT COUNTS AS A PASS

A pass is all four, together:

1. `RESULT SUCCESS` in the leg log.
2. `obs_publish OK`.
3. **The episode file actually on disk** in `otr/obs/` (check it, do not infer it).
4. **ZERO hand steps** between a clean portable and that file.

Anything less is a fail with findings, which is still a useful result -- report it
as a fail, never as a pass with caveats.

---

## 3. WHAT TO NUKE, AND WHAT NOT TO

**NUKE (they are stale stand-ins and they will mask the thing you are testing):**
* The whole clean-room ComfyUI at `C:\OTR-CleanRoom` -- the portable, its
  `custom_nodes`, its venv.
* The OTR clone inside it (it sits at `da2b7a36`, far behind).
* The **untracked** clean-room profiles on that box (e.g.
  `otr_cleanroom_8gb_klein_ltx25`). They are local stand-ins that were never
  shipped, so testing with them tests nothing a user can get.

**DO NOT NUKE:**
* **The model weights / HF cache.** They are the expensive thing to recreate and
  they are NOT what is under test -- a stranger downloads them too, and re-pulling
  tens of GB proves nothing about the pack. Point the fresh install at the existing
  cache.
* Anything under `docs/` you have not pushed, and `docs/ship-audit-2026-09-01/4060_CLEANROOM.md`
  -- that is the friction log and it is evidence.
* Your git identity / SSH keys.

---

## 4. THE RUN

1. **Fresh portable ComfyUI, Python 3.13.** Nothing carried over from the old
   clean room except the weights cache.
2. **Install the pack the way a stranger does: ComfyUI Manager -> the registry ->
   `comfyui-old-time-radio`, version `2.0.0-alpha.23`.**
   Do NOT `git clone`. Do NOT copy from the 5080. The published zip is the artifact
   under test. (Verified from the 5080 tonight: the zip contains 710 files and
   registers all 25 nodes, so a zero-node result on your box is a real finding.)
   * Expect the registry page's **NODES panel to say "No nodes found"** -- that is
     a registry-wide extraction bug (KJNodes, rgthree, Impact Pack and
     VideoHelperSuite all 404 the same endpoint). It is NOT a signal about the
     pack. Judge only by the ComfyUI console.
3. **Restart ComfyUI and read the console.** You want
   `[OldTimeRadio] OK - All 25 nodes loaded successfully`.
   A `[OldTimeRadio] Skipped '<name>': <reason>` line is a per-node dependency
   miss, not a dead pack -- record each one.
4. **Load `workflows/variants/otr_nvidia_8gb_haunted.json`** -- the daily low-VRAM
   variant. It runs `animatediff15_v3_haunted_video` on all three video roles with
   `flux2_klein` stills, and it is the graph an 8 GB user actually gets.
5. **Run one episode.** Start with `--act-count 1`. Only go longer once one act
   publishes.

---

## 5. OWNERSHIP -- WHAT YOU MAY AND MAY NOT EDIT

This is the part that protects the 5080, so it is not negotiable.

**YOU OWN (edit and push freely):**
* `docs/4060_DRILL_LOG.md`, `docs/ship-audit-2026-09-01/4060_CLEANROOM.md`
* Any profile you have PROVEN on 8 GB hardware
* `docs/PROD_BUG_LOG.md` -- shared and **append-only**; add rows, never rewrite

**YOU DO NOT OWN (the 5080 owns these -- do NOT edit them):**
* `nodes/` -- any Python in the pack
* `workflows/otr_canonical.json` and everything in `workflows/variants/`
* `pyproject.toml` and anything registry-facing
* Profile `status` promotions

**If the fix is in one of those, do not make it.** Message the 5080 window instead:
run `ListAgents`, find the OTR peer (`comfyui-oldtimeradio-NN`), and `SendMessage`
with the file, the line, the symptom and the leg log excerpt. Do not route it
through the operator, and do not "just fix it quickly" -- two windows editing the
same file is how the workflow JSON got corrupted before.

---

## 6. DO NOT BREAK THE 5080 -- AND PROVE IT, DO NOT ASSERT IT

CLAUDE.md section 0B is the rule; here is what it means for you tonight.

* **CONFINE IT IF YOU CAN.** A per-machine choice belongs in a PROFILE or a
  VARIANT, never in `otr_canonical.json` and never in shared code. A change that
  lives entirely inside a 4060 profile **cannot reach the 5080**, and that is the
  preferred shape of every portability fix you will find.
* **IF SHARED CODE GENUINELY MUST CHANGE, that is a 5080 change** -- hand it over
  (section 5). When the 5080 makes it, it must run the 5080's own path and print
  the number **before and after**. The worked example is PBUG-20260829-07: the
  fix touched `_plan_max_memory`, which every machine calls, and the proof was one
  printed line showing `gemma-4-12b-it @ 15.99 GB -> {0: '13.5GiB'}` byte-identical
  on both sides.
* **"The suite passes" is NOT the same claim as "the 5080's numbers did not move."**
* **Unsure whether it reaches the other box? It reaches the other box.** Treat it
  as shared and hand it over.
* **START BY PULLING.** `git fetch origin v2.0-alpha`, then
  `git log --oneline HEAD..origin/v2.0-alpha`, then `git pull --rebase`. Say what
  came down. Tonight the 5080 landed a gender-ladder fix (`c4f369d9`) and a ghost
  prompt-uniqueness fix (`a8e4d5a6`) -- the second one is in YOUR variant's path,
  so make sure you have it before you judge any ghost behaviour.

---

## 7. WHAT TO RECORD

For every friction point, in `docs/PROD_BUG_LOG.md` (append) and the drill log:
* the exact command or click,
* the exact error text,
* what you had to do by hand to get past it,
* and whether the root fix is a 4060 profile (yours) or shared code (the 5080's).

Record ONLY what actually published into `config/machine_classes.json`. Do not
advertise a lane the clean room did not finish.

**Known-open items that may bite you, so you recognise them rather than debug them
fresh:**
* Eight profiles pair an 8 GB ceiling with the **12B writer** and die in the writer
  preflight (`Needed=8.13 GB` under a 6.8 GB ceiling). `otr_g4_ltx_8gb` and
  `otr_w45_ltx_8gb` are in the shipping pair. The 5080 is fixing these tonight.
  **`otr_nvidia_8gb_haunted` is NOT in that set**, so your run is clear.
* bark renders a long announcer line as a tone (PBUG-20260902-03) -- kokoro on both
  voice slots.
* The ghost/haunted lane's bookend beats collapse to 2 distinct motifs on every
  seed, so some beats will still take `deterministic_fallback`. That is a KNOWN,
  un-fixed floor -- report the count, do not chase it.

---

## 8. THE ONE-LINE VERSION

Nuke the stale clean room, install alpha.23 from the registry like a stranger, run
`otr_nvidia_8gb_haunted` for one act, and publish. Fix nothing by hand; record
everything. If the fix lives in `nodes/` or a workflow JSON, it is the 5080's --
message that window and keep going.
