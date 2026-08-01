# The story writer must produce a ledger downstream consumers can take

**Operator directive, 2026-08-01:** "the story writer needs to generate a good
ledger that downstream consumers can take and not fail stories."

**Status:** live production failures, root-caused in code, NOT yet fixed.
Branch `v2.0-alpha`, HEAD `265c4beb`. Every claim below is grounded with file:line
and was verified against a real run on this box.

---

## 1. THE FAILURE, AS IT ACTUALLY HAPPENS

A 6-leg canonical campaign (45-word episodes, local LLM, both operator
randomizers rolled) lost **3 of 6 legs** in the writer. The pass dies as:

    [scifi_fable2] pass 'script' failed after 4 attempt(s):
    markup ladder exhausted; last defects:
    - UNKNOWN_SPEAKER: **TITLE (line 1)
    - SKELETON_BREAK: character line (**TITLE) before SCENE 1 (line 1)
    - UNKNOWN_SPEAKER: **MUSIC (line 3)
    - SKELETON_BREAK: character line (**MUSIC) before SCENE 1 (line 3)
    - UNKNOWN_SPEAKER: **ANNOUNCER (line 5)
    - UNKNOWN_SPEAKER: **SCENE 1 (line 7)
    - UNKNOWN_SPEAKER: **MARKUS BUEHLER (line 9)

Aggregate defect counts across the failed legs: **UNKNOWN_SPEAKER 106,
SKELETON_BREAK 106**, BAD_LINE_SHAPE 22, CAST_MEMBER_SILENT 8, MISSING_TITLE 4,
MISSING_END 4.

**The model emitted Markdown.** Every structural line came back as `**TITLE:`,
`**MUSIC:`, `**SCENE 1:`, `**ANNOUNCER:` instead of the bare delimiter.

**Cost of the failure is not small.** One leg burned **82 minutes** before dying;
another ran 69 minutes to the video stage on a previous run. A writer failure
discards the whole episode.

## 2. ROOT CAUSE, GROUNDED

`nodes/_otr_fable2_markup.py`:

    _RE_TITLE   = ^TITLE:\s*(.+)$              # :37
    _RE_MUSIC   = ^MUSIC:\s*(.+)$              # :38
    _RE_SCENE   = ^SCENE\s+(\d{1,2}):\s*(.+)$  # :39
    _RE_CODA    = ^CODA:\s*(.+)$               # :40
    _RE_END     = ^END\.\s*$                   # :41
    _RE_SPEAKER = ^([^:\r\n]+):\s*(\S(?:.*\S)?)$   # :42

    def _normalize_line(line):                 # :45-47
        """Normalize transport whitespace only; authored content is untouched."""
        return str(line).strip(), ()

Every classifier is `^`-anchored and the normalizer is whitespace-only. So for
`**TITLE: The Persistent Strawberry`:

1. `_RE_TITLE` does NOT match (the line starts with `**`).
2. `_RE_SPEAKER` DOES match -- group(1) = `**TITLE`.
3. `on_speaker("**TITLE", ...)` -> the name is not in the cast roster ->
   `UNKNOWN_SPEAKER` (`:269`).
4. A character line before SCENE 1 also raises `SKELETON_BREAK` (`:175`).

That is 2 defects per line, on every line, for 4 attempts -- exactly the 106/106.

**The `_RE_SPEAKER` catch-all is what turns a formatting artifact into a cast
error.** Any decorated delimiter is silently reinterpreted as a character name
rather than being recognized as a decorated delimiter.

## 3. THE DESIGN TENSION THIS SITS ON

The strictness is DELIBERATE and must not be casually loosened:

* `_otr_scifi_fable2.py:1591` -- defects surface "instead of being silently
  massaged into something else."
* `parse_fable2_markup` docstring -- "Pure: no I/O, no mutation of arguments,
  **never rewrites a spoken word**."

Both are right. The question is whether `**` on a STRUCTURAL DELIMITER is
"authored content" at all, or transport decoration that the normalizer's own
docstring already claims responsibility for. The caller at `:322` even has
`if not line: continue  # the whole line was decoration`, so the design
anticipates a normalizer that removes decoration -- the current implementation
just never removes any.

## 4. WHAT IS NOT THE CAUSE (ruled out by measurement, do not re-litigate)

* **NOT the word count.** Suspected 35 words was too short; the same defect
  class appears and one 35-word leg (`ltx_8gb`) SUCCEEDED end-to-end, publishing
  a 118.9 s 1080p episode.
* **NOT the randomizers.** The successful leg rolled both randomizers too.
* **NOT a VRAM/ctx problem in the writer itself.** Separate issue, section 6.

## 5. THE BAR: A LEDGER DOWNSTREAM CONSUMERS CAN TAKE

CLAUDE.md is explicit that the ledger must be filled COMPLETELY for TTS, per-beat
audio slicing, video/shot direction, captions, credits and `obs_publish` -- "they
read FIELDS, not intentions." So the fix is not merely "make the parser accept
more"; it is "guarantee the artifact handed downstream is complete and correct,
or fail loudly before an hour of media work is spent."

Note the ordering problem: the writer is node 1. A defect that only surfaces
after stills+audio (as in the 82-minute leg) has already cost the episode.

## 6. A SECOND, INDEPENDENT DEFECT FOUND WHILE RUNNING THIS DOWN

The VRAM fit gate blocks the CANONICAL model on the tier that configures it.

    check_vram_fit(model_id, ctx, ceiling_gb=...)   # _otr_model_loader.py:899
    _estimate_resident_gb(...)                      # _otr_model_catalog.py:1476

For a `gguf_native` row it returns `weights_gb + kv_gb` where `weights_gb` is the
row's single pinned `approx_safetensors_gb` and `kv_gb` uses **`_row.context_window`,
not the `ctx` argument the caller passed**. Consequences, both measured:

* The `ctx` argument is INERT -- identical 14.60 GB estimate at ctx 2048, 4096
  and 8192.
* There is NO quant parameter, so a profile pinning `Q4_K_M` is priced at the
  repo's largest quant.

Measured on disk: `gemma-4-12b-it-Q8_0.gguf` = **11.80 GB** (exactly the pinned
`approx_safetensors_gb`), `gemma-4-12b-it-Q4_K_M.gguf` = **6.63 GB**.

    gate estimate  Q8_0   @4096 = 11.80 + 2.80 = 14.60 GB   -> FAIL vs 6.8 ceiling
    tier reality   Q4_K_M @2048 =  6.63 + 1.40 =  8.03 GB

A **1.8x over-estimate** that FAILs the 8 GB tier on a model that tier explicitly
pins. (Note 8.03 > 6.8, so that tier's ceiling is ALSO too low for a 12B -- two
separate problems stacked.)

## 7. WHAT THIS ROUND MUST RULE ON

### Q1 -- Where does emphasis get stripped, and how far does that go?
Is stripping `**`/`__`/`*`/`_` from the STRUCTURAL PREFIX the right fix, or does
it start a slide toward the "silent massaging" the design forbids? Name the exact
boundary: which characters, on which line classes, and what MUST remain
byte-identical. Does the spoken payload stay untouched in every case?

### Q2 -- Should `_RE_SPEAKER` stop being a catch-all?
It currently swallows any `X: y` line, which is what converts a decorated
delimiter into an UNKNOWN_SPEAKER. Would recognizing a decorated delimiter as a
DECORATED DELIMITER (a distinct, repairable defect) be better than reinterpreting
it as a cast error? What breaks if the catch-all narrows?

### Q3 -- Prompt, parser, or repair ladder -- which owns this?
Three candidate owners: forbid Markdown in the prompt; strip it in the parser;
repair it in the retry ladder (which already re-prompts 4x and has the defect
list in hand). Rule on the owner and say why the other two are wrong or
insufficient. Note the ladder currently retries with the SAME instruction and
therefore repeats the same failure 4 times.

### Q4 -- Fail EARLY, not after an hour.
One leg spent 82 minutes before the writer's failure ended the episode. Is there
a cheap pre-flight that proves the model emits parseable markup BEFORE the
pipeline commits to stills/audio? What would it cost, and what would it miss?

### Q5 -- Ledger completeness for downstream consumers.
Given the operator's bar, what does the writer owe beyond "the script parsed"?
Enumerate the fields downstream reads and state which are currently only
guaranteed by the happy path.

## 8. CONSTRAINTS

- Never rewrite a spoken word. The parser's purity promise holds.
- No silent massaging: anything repaired must be REPORTED, not hidden.
- Fail loud and EARLY beats fail loud and late.
- Do not touch `wan_ti2v`'s frozen recipe or any video adapter -- unrelated lane.
- The randomizers stay on; a fix that only works on unrolled defaults is not a fix.
