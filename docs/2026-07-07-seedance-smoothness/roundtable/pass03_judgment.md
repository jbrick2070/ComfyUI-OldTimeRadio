# Pass 03 Judgment - Wiring And JSON

## Verdict

Proceed with an adapter-local Seedance fix. Workflow JSON is allowed if needed,
but this pass does not need it.

The panel found one real implementation trap: a naive softener would mutate the
smooth-motion clause on the second call because the clause itself contains risky
phrases such as "whip pans", "handheld shake", and "rapid zooms". The helper
must detect the stable marker before applying any softeners and return unchanged
when already conditioned.

## Workflow JSON Decision

No edit to `workflows/otr_scifi_16gb_full.json` in this pass.

Reason: no node, widget, input, output, link, or workflow-visible parameter is
being added. The existing workflow already routes a prompt into
`cloud_seedance_2`; this pass changes only how that adapter conditions the
existing prompt before sending it to the installed Partner Node.

If a later pass adds a real runtime control, such as a selectable Seedance
motion mode, provider model widget, extra reference-video input, or guide-clip
input, then the workflow JSON must be changed in the same patch.

## Style JSON Decision

Do not globally soften `nodes/visual_styles/sci_fi_radio.json` yet.

That file does contain rough motion language such as `whip-pans`,
`vibrates aggressively`, and `Dynamic dolly push`. However, repo history and
comments show that this aggressive `music_open` register was deliberately
restored for the LTX path after a later A/B. A global style-pack edit would
change non-Seedance behavior too.

Seedance needs a provider-specific stabilizer. If later A/B shows every video
engine benefits from calmer source text, then edit the style pack and the pinned
Python extraction fixture together.

## Accepted Pass 03 Fixes

- Detect the stable marker before softening:
  `Gentle parallax only; all motion gradual and physically continuous.`
- If the marker exists, return the prompt unchanged with `changed=False`.
- Apply regex replacements in longest/specific-first order.
- Sanitize log excerpts by collapsing whitespace before truncating.
- Define helper metadata exactly:
  - `changed`
  - `original_sha8`
  - `conditioned_sha8`
  - `original_excerpt`
  - `conditioned_excerpt`
  - `softeners_applied`
- SHA8 is `hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]`.
- Preserve the current `_duration_seconds()` call arguments exactly and assign
  the duration once.
- Assert the current Seedance Partner Node request shape exactly in tests.

## R3 Spend

- Pass 03 GPT/Gemini spend: about `$0.1491`
- Pass 03 Hy3 retry spend: `$0.0000`
- Running total: about `$0.3919`
