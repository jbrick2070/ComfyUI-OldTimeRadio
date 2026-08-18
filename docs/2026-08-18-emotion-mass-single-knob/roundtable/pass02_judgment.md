# R2 judgment -- Claude as sole judge

Panel: `openai/gpt-5.6-sol`, `google/gemini-3.1-pro-preview`, `x-ai/grok-4.3`
(Grok swapped in for DeepSeek, which returned empty content on both R1
attempts). All three answered. **Spend this pass: ~$0.1843.**

## The round's two best findings

**1. The R1 fix I accepted would have INTRODUCED a contamination bug.** All
three models caught it independently. R1's GPT told me to make the shipped arm
*clear* its env overrides instead of restating the constants, which I accepted.
But `_render_arm` applies environment incrementally --
`for key, value in env.items(): os.environ[key] = value` -- and never removes a
key the arm omits. Arms run in a **shuffled** order. So an arm that "clears"
alpha by omission would silently inherit whatever the previously-rendered arm
left in `os.environ`, and the blinding shuffle means it would not even fail the
same way twice. CONFIRMED at `scripts/otr_lemmy_production_audition.py:118-120`.
**Fix: every arm pops all three managed variables first, then applies only its
own overrides.** Each arm's environment becomes total, not incremental.

**2. The profile key I was about to add is inert.** Grok, then GPT
independently. R1's GPT argued the profile should declare the knob that
governs, so I planned `emo_mass_cap: 0.56` in `default_params`. Grounding says
`current_emo_mass_cap` and `emotion_payload` read **only** the env and the
module constant -- there is no profile lookup -- and `_begin_line_runtime`
overwrites the profile value with `render_time_params()` anyway. So the key
would be decoration that reads as authority: the exact "evidence-shaped field"
anti-pattern `cast_pools.py` argues against. **REVERSED: do not add it.**
`emo_alpha` moves 0.4 -> 1.0 (it is already overwritten the same way, so leaving
it at 0.4 would simply be false in a file people read) and the YAML says plainly
that the adapter constants are the authority.

## Also ACCEPTED

| Claim | Source | Grounding |
|---|---|---|
| No listening gate before the record is written | GPT 1 | CONFIRMED, and it binds hard: I may not write an operator PASS for clips he has not heard. Resolved below |
| Run the suite BEFORE freezing and rendering | GPT 2 | CONFIRMED. Test files are not in `RUNTIME_FINGERPRINT_SOURCES`, but a fix landing in `_otr_voice_node_common.py` or `_otr_resolved_request.py` would move the fingerprint and invalidate a completed render |
| `zip(lines, captured)` silently truncates a partial arm | GPT 5 | CONFIRMED at line 196. If two of three lines render, the manifest describes a complete arm that never happened. Assert one capture per line before writing anything |
| Refusal must cover WAVs, `KEY.json`, a path that is a file, and a non-empty dir | GPT SF2, Grok 3, Gemini(R1) 2 | CONFIRMED |
| Retention is `1 - 0.5590 = 0.4410`, not 0.440 | GPT 8 | CONFIRMED. Record per-line retention, and state the invariant as "at least 0.440" |
| Every two-arm assumption moves together: labels, shuffle, KEY text, module docstring, `"prefix"` | GPT SF6 | CONFIRMED |
| "Below-cap passes through unchanged" means unchanged vs `sanitize_delivery_vector`, not vs raw ledger JSON | GPT SF3 | CONFIRMED -- sanitize legitimately clamps, rounds and fills |
| Bump `char_indextts2_v1.engine_impl_version` 2 -> 3 | GPT 3, Grok | ACCEPTED on reflection. Redundant for cache correctness (both knobs already key via `render_time_params`, proven by `test_lemmy_emo_alpha_cache_key.py`) but the marker is the profile's declared "this engine's behaviour changed" statement and it should stay truthful. One test updates |

## REJECTED

* **Grok 4 -- "`live_engine_impl_version` is not present in `eng_indextts2.py`,
  sequencing cannot be executed."** MISREAD caused by the grounding set: it
  lives in `nodes/_otr_voice_route.py:179`, which was not among the four files
  sent. Verified directly; it sha256s whole normalized file bytes.
* **GPT 3, in part -- "record the profile source SHA."** No such hash exists.
  `audio_engine_profiles.yaml` is loaded by `_otr_engine_profiles.py` with no
  digest receipt anywhere, so recording one would mean inventing a mechanism to
  cite. Record `profile_id` and the adapter fingerprint, which are real.
* **Grok OPTIONAL -- fast path when `cap == EMOTION_MASS_CAP_DISABLED`.**
  Skipped. It would bypass the JSON round-trip that `emotion_payload` measures
  on purpose, and the only beneficiary is a diagnostic control arm.

## The listening gate -- how this resolves

GPT is right that §6 wrote a qualification immediately after rendering with no
human in the loop, and I will not fabricate an operator verdict.

What is TRUE: he approved **0.560 by ear**, verbatim, on the log-odds ladder
rendered from this same reference voice -- that is a real listening verdict on
the setting. What is NOT true: that he has heard the re-rendered production
audition, and in particular the **neutral** lines, which are 56 of 57 real
production lines and the shape the original defect lived on.

So the record cites the ladder as the listening evidence for the setting and the
production audition as the technical evidence that the shipped path delivers it,
quotes only words he actually said, and names the neutral-line listen as
outstanding **inside the evidence** rather than papering over it. The listen
page is built and handed to him.

## Convergence call

Converged on design; the remaining risk is execution order, which R4 gates on
the finished diff rather than on prose.

**Running total across R1 + R2: ~$0.4241.**
