# DECISION: the cloud video negative fights five packs' own motion_registers

**Driver anchor (Claude, Cowork), 2026-08-17. Every claim below was checked
against the real Windows files before it was written.** Panel: Fable, Sonnet,
Antigravity. Codex is quota-held until 2026-08-19 20:31 and is NOT in this
round. Claude is a grounded panelist AND the sole judge (CLAUDE.md section 8).

## THE QUESTION, and it is one question

The cloud Wan video negative bans motions that five visual-style packs
explicitly ask for. **Do we resolve that, and if so how -- given the video
negatives are FROZEN RECIPE and may never be edited?**

This is on the table because it is a real design fork: "leave it" and "guard it
at compose time" are both defensible, and the repo's own precedent cuts both
ways. Per `7f6a6eca` that earns a panel before any code.

## THE MEASURED FACTS (file:line, all verified)

**The negative.** `nodes/_otr_video_engines/eng_cloud_video.py:221-224`:

```python
_WAN_NEGATIVE_DEFAULT = visual_safety_negative(
    "jump cuts, whip pans, rapid zooms, handheld shake, jitter, flicker, "
    "melting geometry, warped face, distorted hands, drifting text, unreadable "
    "text, black frame, pillarbox bars")
```

Reached at `eng_cloud_video.py:815-817`, through an env channel:
`visual_safety_negative(os.environ.get("OTR_CLOUD_WAN_NEGATIVE_PROMPT",
"").strip() or _WAN_NEGATIVE_DEFAULT)`.

**The packs that contradict it.** Five, all in `nodes/visual_styles/`:

| pack | field | text | fights |
|---|---|---|---|
| `anime` | `motion_registers.music_open` | "Dial whip-pans across frequencies." | whip pans |
| `cartoon` | `motion_registers.music_open` | "Dial whip-pans wildly." | whip pans |
| `paper_origami` | `motion_registers.music_open` | "Paper dial whip-pans across frequencies." | whip pans |
| **`sci_fi_radio`** | `motion_registers.music_open` | "Dial whip-pans across frequencies." | whip pans |
| `shakespeare_stage_realism` | `motion_registers.announcer` | "Candlelight flickers across polished wood." | flicker |

`sci_fi_radio` is the **DEFAULT pack**. That is what makes this wide rather than
an exotic-pack curiosity.

**Every LOCAL video engine is clean** -- `eng_ltx_av`, `eng_humo`,
`eng_ltx_8gb`, `eng_ltx_video` carry only quality/artifact terms ("low quality,
blurry, distorted, watermark, text, static"). The exposure is the opt-in cloud
lane ONLY. Pinned by 22 tests in `tests/test_style_traceroute_video.py`.

**Two corrections to the record this measurement forced.** The D-BIS entry said
the negative bans "flicker" while FOUR packs ask for it. Wrong on both halves:
flicker is now ONE pack (the `anime` rewording to "alternate" already fixed the
rest), and the unrecorded, wider conflict is "whip pans" on four. It hid partly
because the engine writes "whip pans" and every pack writes "whip-pans", so a
strict literal match finds none of the four.

## THE CONSTRAINTS THAT BIND (do not propose around these)

1. **"The recipes are not on the table."** No VRAM, speed or quality finding
   justifies a recipe change. `_WAN_NEGATIVE_DEFAULT` is a recipe string. Any
   fix that edits it is out of scope, full stop.
2. **The admission rule.** `PROD_BUG_LOG.md` is reserved for defects verified by
   a live artifact. This is a STATIC-AUDIT finding from a repo census. It is not
   a PBUG and may not become one without one live observation.
3. **The cloud lane is UNVERIFIABLE from this repo.** Opt-in engine, credentials
   required, provider-side behaviour cannot be observed here. We cannot get the
   live observation constraint 2 demands without spending on a paid provider --
   and the standing scope rule is 100% local, offline-first, no paid services.
4. **The B6 reproducibility lesson.** `eng_ltx_8gb.py:818-829` deliberately
   demoted a boot-time env channel because it "made two boxes render visibly
   different clips from the same episode while both stamped the same recipe
   receipt". Any new channel must not repeat that.
5. **THE LAW.** An audit may never fail a story for style or visual vocabulary.
   A guard here conditions a render; it may never reject an episode.

## THE OPTIONS

**(a) LEAVE IT, report only.** The traceroute now surfaces the five conflicts on
every run. Cost: zero. Risk: a known conflict stays live on the default pack's
`music_open` register whenever anyone runs the cloud lane.

**(b) COMPOSE-TIME GUARD, mirroring the still side.** `_otr_visual_styles.py:714
effective_negative` already does exactly this shape for STILLS: it drops any
negative phrase the pack's own positive asks for, at compose time, WITHOUT
editing the authored string. The video analogue would drop a negative phrase the
pack's own `motion_registers` ask for. The precedent is blessed and shipped.

**(c) GUARD THE CLOUD LANE ONLY.** Narrower than (b); the local engines have no
conflicts to resolve, so guarding them is dead code today.

**(d) REWORD THE FOUR PACKS** instead of touching any engine -- the `anime`
"flicker" -> "alternate" edit is the existing precedent for this, and it was
already done once.

## MY ANCHOR POSITION (which I want the panel to break)

**I lean (a) now, with (b) designed but NOT built.** Reasoning:

* The still-side resolver was built in response to a **measured live defect** --
  PBUG-20260817-01, an announcer still minting as a photograph on a cartoon
  episode, which I re-proved on pixels today. The video side has **no such
  measurement and cannot get one** from this box (constraint 3). Building a
  conditioning guard on an unverified finding inverts the admission rule.
* A negative phrase is not obviously symmetric with a positive one at the
  provider. "whip pans" in a Wan negative and "whip-pans" in a motion register
  may be tokenized and weighted very differently; I have no evidence they
  actually cancel, only that they contradict in English. The still-side case had
  pixels proving the fight was real. Here I have text.
* **Option (d) is cheaper than it looks and I may be underrating it.** Four
  `music_open` strings, one word each, the precedent already exists. It has no
  engine surface and no reproducibility exposure at all. Its cost is that it
  edits authored creative content to accommodate an engine -- which is precisely
  the inversion PBUG-20260817-01 was about, so it may be exactly wrong.

**The strongest argument against me,** which I want tested: the asymmetry itself
is the defect. The STILL side has a compose-time authority that guarantees a
pack can never veto itself; the VIDEO side has none. That is an architectural
gap, and "we have not measured it yet" is a weak reason to leave a known gap
open when the resolver pattern is already shipped, already blessed, and
already tested on the other half.

## WHAT I AM ASKING THE PANEL

1. Is (a) or (b) correct, or is (d) the one I am underrating? Argue the losing
   side of your own answer.
2. If (b): where exactly does it belong so it never touches a recipe string, and
   how does it avoid the B6 receipt problem -- i.e. how does the ledger record
   that a term was dropped, so two boxes cannot render differently while
   stamping the same receipt?
3. Is my "text is not pixels" objection sound, or am I hiding behind it to avoid
   work that the shipped still-side precedent already justifies?
4. What breaks that none of us have named?

Ground every claim in the real files. A claim I cannot verify gets discarded.
