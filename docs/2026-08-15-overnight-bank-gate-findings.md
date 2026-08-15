# Overnight six-bank gate, 2026-08-15 -- 4/6 pass, both news lanes fail

Run: `scripts/otr_writer_bank_gate.py --acts 3 --profile otr_w45_still_flat`
(stills on `z_image_turbo`), against HEAD `f5fee5b9`. Pass 1 of 3.

| bank | verdict | minutes |
|---|---|---:|
| media_archive | PASS | 21.5 |
| original | PASS | 17.8 |
| public_domain | PASS | 16.9 |
| shakespeare | PASS | 16.0 |
| **scifi_news** | **FAIL (writer)** | 13.6 |
| **scifi_news_pro** | **FAIL (writer)** | 3.3 |

Both failures are WRITER-stage, and both are on the two news lanes. **These are
the two lanes changed on 2026-08-15**, so the null hypothesis is that this
session caused them. Neither failure has been diagnosed yet -- what follows is
evidence, not a conclusion.

## scifi_news (codex lane)

```
CodexPreTailAuditError: line receipt mismatch for <line_id>
```

Raised in the PRE-TAIL AUDIT, after the script is assembled. The audit compares
each ledger row against the per-line receipt the lane recorded for it.

**Prime suspect: the act-topology change, not the no-shims fixes.** A 3-act
episode went from 8 voiced beats to 12 in `9c2d721d`, and the schema caps that
previously TRUNCATED the spine at 12 beats total were raised at the same time.
So this leg is the first to assemble a materially longer codex script, and the
first where the beat count is not silently clipped. A receipt/row bookkeeping
path that only ever saw <= 12 rows is the thing to read first.

Note the leg ran 13.6 minutes and died AFTER the writing -- this is not a
capacity refusal, it is an integrity check failing on assembled output.

**Do NOT assume it is the schema caps and revert them.** The caps are an
operator ruling ("if I ask for 7 acts it needs to generate a spine of 7 acts").
If the audit is what is wrong, the audit is what gets fixed.

## scifi_news_pro (fable2 lane)

```
Fable2ScriptError: pass 'script' failed after 4 attempt(s):
  markup ladder exhausted; last defects:
  BAD_LINE_SHAPE: END (line 23) | MISSING_END (no fallback to legacy_many_pass)
```

The script pass could not produce parseable markup in four attempts. Died at
3.3 minutes -- fast, so it never got near the later passes.

**This is a KNOWN defect class, not obviously new:** GO_FORWARD row 2 already
carries *"resolve the fable2 BAD_LINE interaction"* as a live open item. What
is new is that a longer script (12 beats rather than 8) gives the markup ladder
more rows in which to emit one bad `END`, so the change may have raised an
existing failure rate rather than introduced a failure.

**The `_pass_news_read` post_validator added in `3661bc42` is NOT implicated by
this trace** -- that validator guards the news READ pass, and this died in the
SCRIPT pass, earlier. Rule it out by reading, do not assume it.

## What to do first, in order

1. **Reproduce both on the SAME HEAD with the act change reverted locally**
   (do not commit the revert). That single experiment separates "the act
   change did this" from "tonight's writer fixes did this" and is worth more
   than any amount of reading.
2. If the act change is implicated, fix the CONSUMER (the pre-tail audit, the
   markup ladder), never the topology -- the topology is an operator ruling.
3. Passes 2 and 3 of the gate were queued behind this one and will re-run all
   six banks. If both news lanes fail identically all three times, that is a
   deterministic defect and not a sampling accident -- valuable, and free.

## What DID work

Four banks passed a full path end to end -- writer, audio, stills, publish --
at the new 12-beat act shape, including both fidelity lanes. So the act change
is not broadly fatal; it is specifically the two news lanes that break, which
is also where the only two lane-specific code changes landed.
