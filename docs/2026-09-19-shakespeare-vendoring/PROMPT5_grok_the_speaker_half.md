Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 76f88d8c.

THE BOUNDARY HALF IS DONE. `--pages START-END` is on main, and all eight
Spanish windows return their scene with the COMPLETE English roster
speaking. Run any of them yourself:

    python scripts/otr_vendor_scan.py --iso es --play tempest --scene 3.1 \
        --stem tempest__act3_scene1 --pages 58-62 --scene-label "ESCENA PRIMERA ."

and it prints the honest state of the SPEAKER half:

    [scan] scene span: 4822 chars
    [scan] all-caps runs that resolve to nobody (ignored): FER x6, MIR x5
    [scan] 0 speeches / 0 distinct labels

Zero speeches from a perfect span. That half is yours, because your cue
inventories across all eight cells are the best measurements anyone has.

YOUR JOB IS TO SPECIFY, THEN TO BREAK YOUR OWN SPEC BY MEASURING IT.
Read-only on the repo; scratch in %TEMP%; no git writes; no GPU.
Python: C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe, PYTHONUTF8=1.

THE RULING THAT GOVERNS IT (docs/OTR_STANDING_RULINGS.md, 2026-09-19): a
mangled native text beats a clean AI translation; an unresolvable cue is
NOT an abort -- it becomes its own speaker under the name as printed, left
unbound, costing a voice and never the dialogue; what is still refused is a
speech in another character's mouth. Today `speeches_from_span` DISCARDS an
unresolved label and merges its text into the previous speaker. The ruling
calls that indefensible. So the change is: an unrecognised cue is EMITTED,
not dropped.

SPECIFY THE CUE DETECTOR FOR THE SPANISH SHAPES, EACH GROUNDED IN A CELL
  * abbreviation with period, either case:  `FER .`  `Fer.`  `MIR`  `Prós.`
  * the Clark honorific split, one printed row:  `D . PED.`  `D . TOB.`
    `D . AND.` -- 25 in much_ado 2.3, 89 of 93 in the Twelfth Night act-2
    window, ZERO in Macpherson. You found this. What regex takes it as ONE
    cue, and what does that regex wrongly take elsewhere?
  * the joint cue:  `ALB . Y CORN.`  -- one row, two speakers. The runtime
    already knows a collective (`_COLLECTIVE_SPEAKERS` in
    nodes/_otr_passage_selector.py); say whether it should map there or be
    left unbound.
  * WHAT MUST NOT BE A CUE: `Mañana .`, `creo.`, `bres.`, `señora.` -- a
    wrapped word ending a line. You showed edit distance cannot separate
    these. Say what can: position on the row? a following dash? the word
    being in a small per-edition list? the registry saying which forms are
    cues and everything else being dialogue? Pick one and measure its
    false-cue rate on all eight cells.

THEN MEASURE THE SPEC. For each of the eight cells, under your detector
with unbound emission: speeches emitted, labels resolved to the roster,
labels left UNBOUND (name them), and -- the only number that matters --
speeches that land in the WRONG mouth. Zero is the bar. Say where it is
not zero and why.

DO NOT propose changes to extract() or find_label() -- scoped out.

FINISH WITH:
    DETECTOR:     the rule, precise enough to code from, with its
                  false-cue rate per cell
    UNBOUND:      per cell, the labels that would ship unbound
    WRONG MOUTH:  per cell, any speech still misattributed, with evidence
    REFUTED:      every claim above you could not confirm
