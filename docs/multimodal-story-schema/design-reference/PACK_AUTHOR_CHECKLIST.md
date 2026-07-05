# Pack Author Checklist (roundtable pass01 fold)

Start a pack with `python tools/new_pack.py --bank <bank> --model <id>`.
Never hand-copy a pack file (a stale forbidden_leakage_terms list scans for
the wrong lane's leakage and nothing can catch that semantically).

Per seam, before setting status to `ready_fixture`:

1. outline_system - put the strongest CONSTRAINT first (the first sentence
   survives context pressure best). Name what resolves the story (human
   choice, recognition, disclosure). Do NOT specify scene/beat counts -
   production's outline machinery owns structure.
2. pitch_room_system - name what must VARY between pitches (protagonist,
   object, institution, stakes texture) and what must NOT (bank fidelity
   rules). For faithful-adaptation models: variants are compression plans,
   premise identical.
3. dramatic_state_system / line_grounding / coda_system / title_system -
   ground each in the bank's source vocabulary (archive material / source
   text / news facts); no other lane's vocabulary.
4. tone_guardrails - POSITIVE constraints only; these ARE injected into
   executable-pipeline prompts. forbidden_plot_patterns /
   forbidden_leakage_terms are METADATA - scanned post-generation, never
   rendered into prompts (models copy negated terms). Include the cross-lane
   base leakage set for every non-science pack: "science-fiction audio
   drama", "real science", "news facts".
5. Run the gates: pytest + scripts/validate_lab.py (leakage, template
   variables, seam coverage, duplicates all fail loudly).

Re-pinning compat.py after a production refactor: refresh production_mirror
+ regenerate PRODUCTION_MIRROR_MANIFEST.md hashes, run the drift tests, and
for each CompatDriftError manually diff the mirror file, update BOTH the
pinned tuple and the extractor if the syntactic form changed
(class-with-annotations, dict literal, constant strings). The mirror-hash
test refuses a partially refreshed mirror.
