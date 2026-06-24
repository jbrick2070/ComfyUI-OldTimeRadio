# Story-Grammar LIVE A/B -- does the local writer OBEY the ending in DIALOGUE?

**Date:** 2026-06-24  **Branch/HEAD:** `v2.0-alpha` @ `3437a8fc` (code) -- workflow
JSON **UNCHANGED** (flags are env-only). Box reset before + after (S4).

**Setup:** full canonical `workflows/otr_scifi_16gb_full.json`, run AS SAVED
(no role overrides; the OFF baseline used the saved LTX-AV bookends, then a fast
visualizer-render variant for throughput -- the WRITER/critic/freeze path being
measured is identical either way). 320 target words, writers rotating
mistral-nemo (in-process) / gemma-4-12b (Ollama). Leg env: **OFF** = clean;
**ON** = `OTR_ENABLE_STYLE_GRAMMAR=1` + `OTR_STORY_QUALITY_L12=1`. Measurement is
read from each FROZEN ledger (final character beat spoken text + ungrounded
crisis-noun count + announcer close + `meta.story_quality`).

> **Sample note (honest):** each full episode runs ~16-20 min on this box (writer
> + critic/reroll + per-line indextts2 + per-beat flux portraits dominate -- the
> render engine is a minor share). I captured **N=2 per leg** in-session, not the
> requested 6; the box cannot do 12 full episodes in one sitting. The signal below
> is consistent and decisive, but a wider overnight N would harden it.

## Wiring -- PROVEN live, end-to-end, on BOTH writers
Server log, ON leg:
```
story-grammar ON: style=time_loop_command_log ending_tag=reversal climax_beat=b017   (mistral)
story-grammar ON: style=first_contact_diplomatic_standoff ending_tag=reconciliation climax_beat=b017 (gemma)
story-quality L1/L2 ON: shaped 16 voiced beat(s) (domain=space, 5 distinct conflict objects)
```
`select_style -> ending_tag as climax_role -> climax-beat ending-template
injection -> meta.story_quality {style_slug, ending_tag, final_beat_crisis_nouns}`
all fire; the shipped ledger carries the telemetry. OFF ledgers carry only the
base `story_quality` keys (grammar absent) -- confirms default-OFF is inert.

## The A/B (final character beat = the climax; announcer close = the sign-off)

### OFF (lever off) -- the problem
| premise | FINAL character line | announcer CLOSE |
|---|---|---|
| orbit intruder (mistral) | "Tower, this is Vance. Scramble all interceptors. We're playing catch-up, people. Let's not drop the ball." | "With the Puma safely in orbit, tonight, the skies over New Zealand bear witness to a **new era of openness**." |
| pod in peril (mistral) | "Negative, MEREDITH TERWILLIGER. Not in these conditions." | "And so, Starfall **splashes down safely**, marking a **new era for SpaceX**." |

OFF pattern: final line is a flat imperative / terse refusal; the **announcer
narrates the news OUTCOME** (who/what succeeded + a "new era" tag).

### ON (lever on) -- the fix
| style -> ending | FINAL character line | announcer CLOSE |
|---|---|---|
| time_loop_command_log -> **reversal** (mistral) | "**I've already taken steps, Rick. The truth has its own jammer.**" | "In the end, the satellite stood tall, and the truth took flight. This is SIGNAL LOST, signing off." |
| first_contact_diplomatic_standoff -> **reconciliation** (gemma) | "The weather in the capital is supposed to be quite clear this evening; it's much easier for everyone to see what's coming." | "...as the lights dim over the galactic bulge, Nancy Grace Roman stands ready for the truth." |

## Honest read
1. **Announcer close: clear, consistent win.** OFF states the outcome ("Puma
   safely in orbit / new era", "splashes down safely / new era for SpaceX"); ON
   lands a non-outcome image + sign-off ("satellite stood tall, truth took
   flight"; "lights dim over the galactic bulge"). The C5 close-gate works.
2. **Final-beat SHAPE: the writer obeys to the degree of the model.** mistral
   obeyed the **reversal** crisply -- "I've already taken steps... the truth has
   its own jammer" is a genuine turn in dialogue, not a command or a machine
   action. gemma went **oblique/soft** on reconciliation -- it AVOIDED the
   console-standoff machinery (the core win) but the reconciliation gesture lands
   loosely rather than crisply. Net: machinery-avoidance is strong on both; the
   specific archetype is crisper on the stronger writer.
3. **Crisis-noun density did NOT discriminate** -- it was **0 in every episode,
   OFF and ON**. On these space-news premises the writers never reached for the
   generic console/lever/countdown vocabulary even OFF, so that metric is flat
   here. The discriminating signals are the **announcer close** and the
   **final-beat shape**, not crisis-noun count. (The deterministic structural
   A/B -- irreversible_choice 100% -> 2.1% -- remains the headline; this live leg
   confirms the writer ACTS on the injected shape/close.)
4. **No render bugs.** The full workflow ran clean OFF (through the LTX-AV
   bookends) and ON (full visualizer render to an OBS final,
   `signal_lost_ink_stir_20260624_115957`, 16 min) -- lever on and off, zero
   tracebacks.
5. **Flags stayed default-OFF in the JSON** (env-only, zero workflow change). The
   default flip is the operator's call after reading these endings.

## Recommendation
The bet leans YES: with the lever on, the announcer stops narrating the outcome
and the climax stops being a console standoff; the writer lands the injected
ending in dialogue (crisply on mistral, loosely-but-machinery-free on gemma). If
you want it crisper on the weak writer, the lever is the right seam to iterate on
(strengthen the final-beat ending instruction, or prefer the stronger writer for
the climax beat). Suggest a wider overnight N before flipping defaults.
