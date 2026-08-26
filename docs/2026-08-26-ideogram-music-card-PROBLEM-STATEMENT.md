# PROBLEM STATEMENT -- Ideogram 4 (local) refuses the wordless music card

**Date:** 2026-08-26. **Status:** open, needs an input-layer design.
**Ask:** design the caption/input layer that makes Ideogram 4 RENDER this card.

---

## 1. The engine and how we talk to it

Ideogram 4, LOCAL weights, run through ComfyUI. It was trained on **structured
JSON captions**, not prose. Handing it prose made it refuse 6 of 6 real lines;
rebuilding the same lines in its caption schema took that to 0 of 6. So we
always send this exact shape, minified to a single line, as the `prompt`:

```json
{
  "aspect_ratio": "16:9",
  "high_level_description": "<one sentence: the subject>",
  "compositional_deconstruction": {
    "background": "<the setting / atmosphere>",
    "elements": [
      {"type": "text",
       "bbox": [200, 60, 700, 940],
       "text": "<the exact words to render>",
       "desc": "<how the lettering should look>"}
    ]
  }
}
```

* Exactly **three** top-level keys, in that order. A foreign key is not
  ignored -- it gets rendered onto the card as visible junk.
* `bbox` is normalized 0-1000 as `[y1, x1, y2, x2]`.
* The only `element.type` we have ever observed is `"text"`.
* There is **no negative channel**: the guider's negative is a zeroed copy of
  the positive. Every token is positive conditioning, so a prohibition can only
  ADD its own nouns -- "no logos" once came back painted on a card as
  "NO MISCOS". We therefore strip all prohibition clauses before sending.

**Recipe knobs we control:** steps 20, std 1.75, cfg 7.0 (override 3.0 from
0.7-1.0), sampler euler, mu 0.5, canvas /16-legal with a 256 floor. ~95 s per
card, 11 GB of weights.

---

## 2. What works, measured tonight

One live episode, all 8 stills routed to Ideogram:

| card type | has words? | count | result |
|---|---|---|---|
| character beat | yes | 4 | **rendered** |
| announcer beat | yes | 2 | **rendered** |
| **music bookend** | **no** | **2** | **REFUSED** |

**6 of 8 rendered. 100% of the worded cards succeeded. 100% of the wordless
cards were refused.** This is the cleanest signal we have.

A refusal is not an error and not a black frame: the graph completes with
status SUCCESS and returns a flat pale placeholder at the exact requested
dimensions. We detect it statistically -- a refusal measures **min 78-87,
std 10.2-10.5**; a real card measures **min ~0, std 27-41**.

---

## 3. The exact caption that was REFUSED

Verified clean: no control characters, no face/people language, no prohibition
clauses, and the two description fields differ. Nothing we previously
identified as a defect is present. It still refused (min=78.0, std=10.5).

```json
{
  "aspect_ratio": "16:9",
  "high_level_description": "An abstract picture evoking 'The Holographic Infant's Mimicry', a purely pictorial composition of shape, colour and texture, anime style. abstract evocative anime key-art, painterly cel-shaded mood illustration, symbolic non-literal composition, The air is thick with unspoken questions and the cold reality of the technology. sepia tones, deep shadows, warm",
  "compositional_deconstruction": {
    "background": "anime style. abstract evocative anime key-art, painterly cel-shaded mood illustration, symbolic non-literal composition, The air is thick with unspoken questions and the cold reality of the technology. sepia tones, deep shadows, warm",
    "elements": []
  }
}
```

The second music card refused identically (min=80.0, std=10.5) on a different
seed, so it is not a bad draw.

**Note `"elements": []`.** That is the one structural difference between this
card and the six that rendered.

---

## 4. Why this card is wordless

The music bookend is the show's opening/closing theme. By a standing product
decision it carries **no words** -- it is a mood image under the theme music,
not a title card. Every other card has the script's own line to set in type.

So the wordless music card asks a display-typography engine for the one thing
it is not for: an abstract with no type to anchor.

---

## 5. The constraints on any answer

1. **Ideogram MUST remain selectable for the music role.** Routing music to a
   different engine is not an acceptable answer.
2. **It must PRODUCE AN OUTPUT.** A poor card is fine. A refusal is not.
   *"It could be a horrible card but it needs to produce an output."*
3. **ONE render per card. No re-rolls.** Reseeding until it works is out --
   ~95 s each, and a standing rule forbids spending extra GPU on retries.
4. **Ideogram-only.** Six other local image engines take our prose verbatim and
   must keep doing so. Whatever is designed here lives in Ideogram's own
   translation layer.
5. No prohibition phrasing (see the no-negative-channel note above). State what
   you WANT, never what to avoid.

---

## 6. The question

**What should we put in the three caption keys so Ideogram 4 renders a
theme-music card instead of refusing it?**

Concretely:

* Is `"elements": []` itself the refusal trigger, or is it the abstract
  `high_level_description`? (We can test either independently.)
* If it needs an element to anchor on, what is the most on-genre thing to give
  a 1940s-radio theme card that is NOT the episode's spoken words -- the show
  title as display type? A station call-sign? A dial face with numerals? An
  album-sleeve treatment? A pure graphic mark?
* Is there a non-`"text"` element type this checkpoint accepts?
* Does `bbox` need to be present/absent/different for a non-lettering card?
* Would a concrete pictorial subject ("a vintage tube radio on a table")
  outperform an abstract one, given the engine renders concrete scenes fine on
  the other six cards?

The ideal deliverable is a filled-in version of the JSON in section 1 that we
can send for a music card, plus a one-line rule for when to use it.
