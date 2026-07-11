# OTR LOOK-AHEAD #4 -- paste into agy AND into codex

REVIEWER ONLY. Read anything; do NOT edit source, do NOT git add/commit/push.
Write to `qa4_<yourname>.md` and stop. Pull first. Label every claim CONFIRMED or
[ASSUMPTION]. Five things you are sure of beat twenty guesses.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha

## Where we are

- **Codex**: publishes 30w. **Gemini**: publishes 30w (`signal_lost_echoes_of_mars`,
  verified in `otr\obs\`). **Sonnet**: has never completed; a roll is in flight now.
- Sonnet's four fixes are in: the rewrite is written back by `line_ref`, ceremonial lines
  cite nothing honestly (`non_fact`) instead of the impossible `fact_0`, the attestation
  is capped at 3 cites, and the Warden speaks his own closing line
  (`rewrite.vesh_resolution`) instead of a string Python hardcoded.
- Codex's catch landed too: the first P3 audit and the P5 rewrite were numbering lines by
  two different contracts. One contract now.
- Latest live kill: the auditor blocked the episode on *"line 2 repeats line 0's claim
  without adding new information."* That is a CRAFT note, not a grounding defect -- and
  in a 30-word session ORUM gets two lines against a two-fact dossier, so "say something
  new" can be literally unsatisfiable. The audit seam now names its defect taxonomy
  exhaustively (unsupported assertion / contradiction / cites that do not support the
  line / not-SFW) and states that repetition, pacing, thin drama and word choice are
  craft notes that may never block an episode.

**The pattern, five kills running: a gate that blocks production may only block on things
that are (a) objectively checkable and (b) actually fixable. Everything else is a note.**

## JOB 1 -- what kills Sonnet AFTER the audit? (highest value)

Sonnet is about to clear its audit for the first time. Walk everything downstream and
predict the kills, ranked:
`_assemble`, `_SonnetTailFinalizer`, `validate_spoken_text_and_lock`, the shared writer
tail, CastLock, the freeze cascade, credits, media, `obs_publish`.

Pay particular attention to:
- Sonnet's cast is `announcer, c02, c03, c04` -- **it has no c01.** Every other lane uses
  c01..c03. Does anything (voice maps, cast invariants, `_assert_unique_bark_voices`,
  the ledger, the render plan, credits) assume `c01` exists or that char_ids are
  contiguous? This is my prime suspect.
- Ceremonial lines now carry `cites=[]`. Does ANY downstream consumer -- coverage math,
  the fact ledger, the freeze cascade's per-line invariants, captions -- assume every
  line has at least one cite? Grep every consumer of `cites`.
- `CastLockV4` declares `tts_model` / `voice_preset` and the lane hardcodes
  kokoro/bm_george + v2/en_speaker_6/_3/_0. Does that satisfy CastLock's Gate 1
  invariants (announcer excluded by the exact name "ANNOUNCER"; every other row a unique
  `v2/` preset)?

## JOB 2 -- Python that AUTHORS (the worst class we have)

agy found four spoken-dialogue fallbacks in `nodes/_otr_line_composer.py`
(`fallback_announcer_intro`, `fallback_safe_open`, `fallback_announcer_outro`,
`_resolved_outro_fallback`). Python writing dialogue is a direct violation of the law.

For EACH, and for anything else you find across all four lanes and the shared tail:
1. file:line, and the exact string/template it would speak.
2. **Can a content-owned sci-fi lane actually REACH it?** Trace the call path. A dead
   fallback is a rip candidate; a reachable one is a bug that will one day put words in a
   character's mouth that no model wrote.
3. What model field SHOULD supply that line instead? (The `vesh_resolution` case is the
   template: the model had already written the line and Python was throwing it away.)
4. If it is genuinely unreachable, say so plainly -- I would rather rip it than guard it.

Rank by "would a listener hear it".

## JOB 3 -- the 720w gate: converge, then hand me the patch

You now agree: `resolve_context_cap` is live, `compute_effective_context_limit` is dead,
and 16k is a GO on the 5080 (agy: +1.25 GiB KV over 8k, head_dim 128, NF4). Codex --
confirm or refute that VRAM number INDEPENDENTLY from the loader's actual quantization
config, and state the KV formula you used.

Then, jointly, give me:
1. The exact minimal edit set to make the effective writer cap 16384, with every
   file:line, and every test that pins 8192 (and whether each asserts runtime behavior or
   just a constant).
2. Which passes must set `prompt_must_fit=True`, and the exact reservation formula each
   whole-script pass should use at 720w so prompt + output fit the new cap.
3. The proof that the default (env unset) stays byte-identical at 8192 -- name the test.

## JOB 4 -- the frame plane (Codex's own proposal, now due)

Codex proposed: the announcer leaves the conflict cast, the score gains an explicit
listener-facing frame plane (`frame_open` / `source_coda` / `signoff`), and a validator
fails closed on invalid frame TOPOLOGY only -- never on the words. Base beat cardinality
is unchanged (`3C + 3`).

Both of you: what breaks when that lands?
- Gemini's `outline_output_token_budget(words, len(bands))` is sized off the advisory band
  count; Codex's script budget off the accepted line count. Both under-reserve the moment
  the outline grows by 3 frame beats + bridges -- and under-reserving is what caused
  `PROMPT_GUARD: Truncated 5408 -> 4592`. Give me the capacity formula each should use.
- Sonnet already HAS a frame (registrar cold open, warden rulings, sign-off) but it is
  hardcoded structure, not a frame plane. Does it need converting, or is it already
  compliant in substance?
- What in the ledger, render plan, captions, or credits assumes the announcer IS cast?

## Output (`qa4_<yourname>.md`)

JOB 1 SONNET TAIL KILL LIST (ranked; c01 gap and empty-cites consumers first)
JOB 2 PYTHON-THAT-AUTHORS (reachable vs dead, ranked by audibility)
JOB 3 720W PATCH (edit set, VRAM settled, prompt_must_fit, baseline proof)
JOB 4 FRAME-PLANE BLAST RADIUS (capacity formulas)
CONFIDENCE on every line.
