You are reviewing an ARCHITECTURE DECISION, not yet written, in a real
repository at
C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
on branch main, near commit 0c437238.

YOUR ONLY JOB IS TO REFUTE. Ground every claim in a file and a line number.
If you cannot ground a claim, say "refuted -- could not confirm in <file>"
rather than agreeing with it. Read-only: do not edit any file, do not run the
render pipeline, do not run any git command that writes, do not touch the GPU.

You reviewed the boundary layer of this lane once already and were right on
both of your refutations. This is the same layer, one level deeper.

READ
  scripts/otr_vendor_shakespeare.py -- find_label() and extract(), about
      lines 768 to 1016
  scripts/otr_vendor_scan.py -- main(), _SPLIT_HEADING, _normalise_heading

THE DEFECT, MEASURED

`find_label` runs an EXACT-match loop over the WHOLE line list, and only if
that finds nothing does it run a prefix loop that tolerates trailing chrome.
Consequence, measured on the Jaime Clark volume (204 pages, two plays: LA
TEMPESTAD and LA NOCHE DE REYES):

  * The Tempest's own next heading is printed `ESCENA II .` -- with a space
    before the period -- at flat line 1882. It matches only in the PREFIX
    loop.
  * `ESCENA II.` -- no space -- appears at flat line 4165, inside LA NOCHE DE
    REYES. It matches in the EXACT loop.
  * So `extract(..., end_label="ESCENA II")` returned line 4165, and Tempest
    act 3 scene 1 came back as a 78,766-character span running across the
    play boundary into the other play. No error, no warning.

A `play_label` was passed and did not help, because it only sets where the
search STARTS.

THE PROPOSED FIX, WHICH YOU SHOULD ATTACK

Do NOT change `find_label` or `extract` -- they are shared with the HTML
vendoring path, which is out of scope. Instead, in the SCANNER only, slice the
volume's lines to one play's range BEFORE calling extract:

    a = find_label(lines, this_play_anchor)
    b = find_label(lines[a+1:], next_play_anchor)
    play_lines = lines[a : a+1+b]      # or lines[a:] for the last play

so the exact-match loop physically cannot reach the following play. Sliced
this way, Tempest 3.1 returns 4,822 characters, which is a plausible scene.

ANSWER THESE, EACH GROUNDED IN A FILE AND A LINE

1. Does the slice actually close the hole, or move it? The same two-pass
   asymmetry still exists INSIDE the slice. Construct the case where it still
   bites: a play whose own text contains two spellings of the end label, the
   later one exact and the earlier one chromed. Does such a case exist in
   either Spanish volume? Look, do not speculate.

2. The slice depends on `find_label` locating the PLAY anchor correctly -- and
   the anchor is also the RUNNING HEAD on every page of that play. In the
   Clark volume `LA TEMPESTAD` first appears at flat line 22. Is that the
   play's start, a table of contents entry, or a running head on a front
   matter page? Check it. If the anchor can land in front matter, say what the
   slice then contains and whether that is harmful or merely wasteful.

3. THE LAST PLAY IN A VOLUME has no next anchor, so the slice runs to the end
   of the file. Does that reintroduce the original defect for every scene of
   the last play? Name the affected cells among: es/twelfth_night 1.5,
   es/twelfth_night 2.5, es/midsummer 3.1, es/midsummer 3.2.

4. A SEPARATE BLOCKER IN THE SAME LAYER. `es/twelfth_night 1.5` cannot be
   located at all: extract() returns "scene heading 'ESCENA V' not found under
   ACTO PRIMERO". The volume prints, on two lines, `ESCENA` then `V .`.
   `_SPLIT_HEADING` requires the numeral to be the last thing on its line, so
   the trailing period defeats the rejoin. Confirm that against the regex.
   Then say what the minimal correct widening is, and name every OTHER string
   that widening would newly rejoin -- in particular, whether it could weld a
   heading onto a line of dialogue that happens to start with a roman numeral
   or a lone capital.

5. ORDER OF OPERATIONS. `_SPLIT_HEADING` is applied to the flat text AFTER
   `strip_running_titles` has already voted on and blanked furniture. Is that
   the right order for the fix in (4)? If a heading is rejoined only after the
   furniture vote, did the vote see a different document than the boundary
   finder does? Answer from the code, and say whether it matters.

6. Is there any reason the scanner should NOT own the slice -- that is, an
   argument that `extract` is the right place after all, even though changing
   it touches the HTML path? Make that argument as strongly as you can, then
   say whether you believe it.

FINISH WITH TWO LISTS, headed exactly:

    MUST-FIX:
      places where a scene could still be stored from the wrong play, or
      silently truncated

    REFUTED:
      every claim in this brief you could not confirm against the real files
