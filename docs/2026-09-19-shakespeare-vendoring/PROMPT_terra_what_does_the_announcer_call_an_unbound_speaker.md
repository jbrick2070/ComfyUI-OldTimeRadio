Repository: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
Branch main, at or after commit 0b35b419.

PRIORITY THREE, AND IT IS THE ONE NOBODY HAS ASKED. Your last trace proved
the runtime PERFORMS an unbound speaker: `_bind` returns the speech
unchanged, `lock_cast` seats it, the gender roll fills in, Kokoro voices
it. Accepted. But a speaker in this pipeline is not only a voice. The
name reaches places that READ IT ALOUD or PRINT IT: the announcer's cast
introduction, the credit roll, captions, the title card, the episode
ledger, `obs_publish`. For a roster-bound speaker those all get a real
name. For an unbound one they get the printed form -- and the printed
form is about to be things like `D . PED`, `MIR`, `BUF`, `ALB . Y CORN`,
`R.DEF`, upper-cased through `clean_label`.

THE QUESTION: what does the listener HEAR and the viewer SEE when the
speaker's only name is `D . PED`?

This is the repo's oldest law wearing a new face: "downstream consumers
read FIELDS, not intentions, and every field must have exactly one
owner." An unbound speaker's spoken name is a field, and this trace finds
out who owns it and what they do with a string that is not a name.

YOUR JOB IS TO REFUTE THE CLAIM THAT THE PIPELINE COPES. Ground every
step in a file and a line. Read-only; no git writes; no render pipeline;
no GPU.

TRACE THE NAME, EACH SINK IN ORDER
  1. THE CAST INTRODUCTION. Find where the announcer's opening names the
     cast -- grep nodes/ for the announcer/host lines that enumerate
     characters, and for the credit template the verbatim lane uses
     (tests/test_verbatim_translation.py has
     `test_every_row_formats_every_credit_template_without_error`; read
     what it formats). What string lands in the TTS prompt for a speaker
     named `D . PED`? Would a voice engine say "D period PED", "Dee Ped",
     or refuse? Read the engine's text normalisation if there is one.
  2. THE CREDIT ROLL AND CAPTIONS. Where are speaker names rendered to
     the caption track and the closing credits? Quote the formatter. Does
     it upper-case, title-case, strip punctuation, or pass through? What
     does `ALB . Y CORN` become on screen?
  3. THE SPOKEN NAME vs THE LABEL. The manifest `speaker_map` has a
     `spoken` field separate from `roster` -- read `speaker_bindings` in
     nodes/_otr_verbatim_corpus.py and where `spoken` is consumed. An
     UNBOUND label has NO map entry at all (an empty roster entry is
     malformed at line ~669). So where does its spoken name come from?
     The raw label? Trace it to the exact line.
  4. THE GENDER ROLL AND THE VOICE. You said the ordinary gender roll
     fills in. Read `cast_lock.py` around line 1282, where `google_tts`
     raises `VoiceCastingError ... NO FALLBACK` on a missing gender. Does
     the Kokoro path draw a gender for `D . PED` from the name, from the
     seed, or from nothing? And on the Google lane (stills-only, but a
     real lane) does an unbound speaker with no gender CRASH the episode?
     That would be the crash-class defect the bar is about.
  5. THE LEDGER. What does the episode ledger record as this character's
     identity -- and does anything downstream (per-beat slicing, shot
     direction, the video prompt that draws "the character") try to
     RESOLVE the name to a face, a description, a gender? A prompt that
     says "D . PED enters" is a different render from one that says "the
     Prince enters."

THEN ANSWER
  * For each sink, one of: PASSES THROUGH (the string is read/printed as
    is), NORMALISES (say to what), CRASHES (say where), or REFUSES (say
    where). A table.
  * The ONE place the vendor should set a human-readable spoken name for
    an unbound label -- `PRINCE` for `D . PED` when the fold table knows
    it, the printed form when it does not -- so every sink gets a name
    and none gets punctuation. Say which field, and whether it is the
    manifest `spoken`, a new field, or the label itself.
  * The Google-lane gender question is the one that can crash: yes or no,
    with the line.

DO NOT propose changes to extract() or find_label().

FINISH WITH:
    SINKS:        the table -- each sink, what it does with `D . PED`
    CRASHES:      any sink that raises, with file and line
    ONE OWNER:    the field that should carry the spoken name
    REFUTED:      every claim above you could not confirm
