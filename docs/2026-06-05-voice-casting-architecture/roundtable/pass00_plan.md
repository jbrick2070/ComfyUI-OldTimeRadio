# Voice-casting -> per-engine I/O contract: verify + future-proof

Panel: review the REAL code (appended as grounding) and pressure-test whether the
character **gender -> voice** casting logic correctly meets the inputs/outputs of
every voice engine (IndexTTS2, Chatterbox, Kokoro, Bark) and is scalable to
future models. You are critiquing an existing, mostly-implemented design -- find
the gaps, the unsafe assumptions, and the scaling cliffs. Cite file:line. A clean
"this is already handled here" is as valuable as a new gap.

## What already exists (the 2026-06-01 plan, implemented)

A model-agnostic per-role audio-engine registry + a voice-casting subsystem
(docs/2026-06-01-audio-overhaul__FULL-FINAL-plan.md, section 5). Two casters live
side by side: a FROZEN legacy Bark-preset caster (default path, byte-identical)
and a NEW registry caster (`assign_voice_for_slot`).

**The per-engine input contract** is the adapter attribute `voice_ref_field`,
which tells the dispatch which cast-row field feeds that engine's reference slot:

| engine     | voice_ref_field   | generate_voice 2nd arg | sample_rate | nature              |
|------------|-------------------|------------------------|-------------|---------------------|
| indextts2  | `voice_ref_path`  | ref_clip_path          | 22050       | clone from WAV clip |
| chatterbox | `voice_ref_path`  | ref_clip_path          | 24000       | clone from WAV clip |
| kokoro     | `voice_ref_id`    | voice_ref (bank id)    | KOKORO_SR   | pinned preset voice |
| bark       | `voice_preset`    | voice_preset string    | 24000       | discrete preset     |

The dispatch (`_otr_voice_node_common._render_per_line`) reads
`ref_field = adapter.voice_ref_field`, pulls that field off the cast row, and
passes it to `adapter.generate_voice(text, <that>, delivery_vector, seed)`.

**The caster** (`assign_voice_for_slot`, `_otr_voice_bank.py`) is
engine-parameterized: candidates = bank entries whose `engine` matches, scored
gender(100)/timbre(40)/role(20)/age(10), a match ladder down to gender-only with
a hard **gender floor**, one seeded deterministic `random.Random` choice, reuse
only if `allow_voice_reuse`. CastLock `auto_registry` stamps the chosen
`voice_ref_id` onto the frozen cast; `preserve_ledger` (default) stamps nothing
and the voice node best-effort-resolves at render time
(`_resolve_clone_ref_path`, which re-runs the caster by gender with reuse=True
and returns a path only if the WAV exists on disk, else -> bark).

## Grounded current state (this machine, 2026-06-05)

- **Reference WAVs on disk:** indextts2 **4/4** (3 male, 1 female), chatterbox
  **0/5** (none installed), kokoro **1/1** (announcer). The bank also defines 4
  chatterbox char refs + 1 chatterbox announcer that do not exist on disk.
- **Default policy is `preserve_ledger`** -> the registry caster never runs ->
  characters reach the voice stage with no stamped ref -> the best-effort
  resolver decides index-vs-bark per line. Observed: a 6-char cast (3M/3F)
  rendered partly on index (22050) and partly on **bark** (24000) even though all
  4 index refs are on disk -- i.e. the resolver returned None for some rows
  despite reuse=True. Headless proof: `auto_registry` + `allow_voice_reuse=True`
  assigns indextts2 to **6/6** characters (the 3 females share `vz_caro_davy`);
  with reuse off it is 4/6 (2 females -> bark).
- **Plan-vs-reality routing drift:** the plan routed Chatterbox = commercial
  character workhorse, IndexTTS2 = research-only behind a non-commercial flag.
  Reality inverted: IndexTTS2 is the promoted default char voice, its CC0
  LibriVox refs are stamped `commercial_clean=true`, and Chatterbox is
  uninstalled + hard-pins old torch/numpy that brick the torch 2.10/cu130 venv.
- A just-shipped fix added `resample_audio` so a bark fallback clip (24000) is
  downsampled to the primary engine rate before packing (the per-line batch is
  single-rate); SceneSequencer already standardizes all batches to 48000.

## Questions for the panel (verify + future-proof)

1. **Is `voice_ref_field` the right seam for new models?** Adding an engine today
   means: a bank `engine` tag, a `voice_ref_field` ("voice_ref_path" |
   "voice_ref_id" | "voice_preset"), an adapter `generate_voice(text, ref, vec,
   seed)` + `sample_rate`, and (for clip engines) on-disk WAVs. Is that the
   complete, minimal contract? What is undocumented or implicit that a new-model
   author would trip on? Propose the canonical "add a voice engine" checklist.

2. **Canonical stamp vs per-engine field.** Casting stamps `voice_ref_id`, but
   clip engines actually consume `voice_ref_path` (resolved from the id at render
   by `_resolve_clone_ref_path`/`_resolve_ref_to_disk`), kokoro consumes the id
   directly, bark consumes a `voice_preset`. Should casting ALWAYS stamp a single
   canonical `voice_ref_id` and let each adapter resolve its own concrete input
   (uniform), or is the current mixed contract better? Weigh determinism, the
   preserve_ledger best-effort path, and scalability.

3. **The gender model.** Casting is binary male/female with a hard gender floor
   (no gender match -> fail closed / bark). How should this scale for (a) a larger
   bank, (b) non-binary / unspecified gender, (c) richer attributes (accent, age,
   language) -- WITHOUT breaking C7 byte-reproducibility (the seed is derived from
   gender/timbre/role/age)? Is "gender" the right primary key, or should it be a
   weighted attribute with a guaranteed non-empty candidate pool?

4. **Default policy now that a CLIP engine is the default voice.** The plan kept
   `preserve_ledger` (legacy bark presets) as the byte-safe default. But with
   IndexTTS2 as the default character voice, preserve_ledger forces the
   best-effort resolver and yields bark whenever it can't resolve. Should the
   default flip to `auto_registry`? What exactly breaks byte-identical / C7 if it
   does, and can that be bounded (e.g. auto_registry only when the selected char
   engine is a clip engine)?

5. **The preserve_ledger best-effort resolver is fragile.** `_resolve_clone_ref_path`
   re-derives a ref from `cast.get("gender")` at RENDER time; we saw it return
   None (-> bark) for rows that have a gender at casting time, suggesting gender
   may not survive onto the cast row the voice node reads. Is re-deriving at
   render the right design, or must casting always pre-stamp so the voice node
   never recomputes? Where is the gender actually populated on the cast row, and
   is it guaranteed present at `_render_per_line`?

6. **commercial_clean routing.** The plan's release gate fails a build if any
   voice is non-clean (IndexTTS2 was non-commercial there). Reality stamps the
   CC0 index refs `commercial_clean=true`. Is the gate still coherent, and how
   should a genuinely non-clean future model be fenced so it can't ship by
   accident while still being usable for drafts?

7. **Heterogeneous sample rates as engines multiply.** Rates today: 22050 / 24000
   / kokoro / 44100 (music). The new `resample_audio` fixes the per-line bark
   mix; SceneSequencer normalizes batches to 48000. Are there remaining hard-coded
   rate assumptions that a new engine at, say, 16000 or 48000 would break (the
   per-line packer, the announcer/music mux, HuMo lip-sync timing)?

8. **Reference-bank scarcity + operator visibility.** 1 female index ref forces
   reuse, so all female characters share one voice (distinctness silently
   degrades). Should the caster emit a loud, structured WARNING when it reuses or
   falls back (so the operator knows to install more refs), and what is the
   right scaling path (more CC0 refs via the dl script, a per-character clip
   option, gender-balanced minimums) before this is "future-proof"?

## Invariants the design must not break

- **C7 byte-reproducibility / determinism:** casting is seeded; any change keeps
  same-seed -> same-assignment. No `hash()`; OS entropy only via the documented
  cast/style RNG envs.
- **PD1 "audio is king":** an episode always renders; a missing ref degrades
  (bark) but never hard-fails (unless an explicit release gate is on).
- **Frozen legacy Bark path stays byte-identical** when selected.
- **Model-agnostic registry:** no per-engine `if engine == "..."` ladders in the
  generic dispatch; engines self-describe (interface, voice_ref_field, rate).
- **Import-time side-effect-free (C-5); 16 GB VRAM (Blackwell sm_120).**

Grounding appended: `config/voice_reference_bank.json`, `_otr_voice_bank.py`
(the caster), `cast_lock.py` (auto_registry / preserve_ledger), the voice-node
dispatch + resolvers, and the four engine adapters
(`eng_indextts2/eng_chatterbox/eng_kokoro/eng_bark`).
