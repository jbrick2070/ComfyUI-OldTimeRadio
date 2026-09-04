"""THE CLEANUP LAB -- a fictional bad ledger per bank, and a real local model.

    python scripts/otr_clean_stage_lab.py                 # every bank
    python scripts/otr_clean_stage_lab.py --bank original
    python scripts/otr_clean_stage_lab.py --repeat 3      # variance
    python scripts/otr_clean_stage_lab.py --dry           # no model, patterns only
    python scripts/otr_clean_stage_lab.py --show          # print every verdict

WHY THIS EXISTS
---------------
Operator, 2026-08-14: *"create a bad ledger realistic scenario for each source
bank and simulate having the local LLM clean it up -- what works, what doesn't.
Make a fictional sandbox where it's fixing a bad ledger, looking at the act,
looking at the real before and after lines. Keep experimenting until you get
the right cleanup recipe."*

That is the right instrument, and the reason is measurement cost. Proving the
clean stage on a rendered episode takes ten minutes, produces ONE noisy sample,
and has no ground truth -- you are left reading six lines and guessing whether
the model was right. Here the defects are PLANTED, so every run scores itself:

    RECALL     -- of the defects we planted, how many did it catch?
    PRECISION  -- of the clean lines we planted as traps, how many did it
                  wrongly condemn? This is the half that a live episode
                  cannot measure at all, and it is where the pass has
                  actually been failing.
    REPAIRED   -- of what it caught, how many came back clean?
    COST       -- model calls spent, which decides whether a recipe is
                  affordable on a 2B at 10 tok/s.

THE TRAPS ARE THE POINT. Any prompt can be made to find stage directions by
telling the model to hunt; the measured live failure was the opposite -- a 2B
condemning "This shipment was listed as A.P.O. 86-574, reel two, right?"
because the prompt had put it in a hunting mood. A recipe is only better if
recall goes up WITHOUT precision going down, and nothing but a labelled
fixture can tell you that.

WHAT IS FICTIONAL AND WHAT IS REAL
----------------------------------
The LEDGERS are invented -- cast, beats, acts, dialogue, all written here.
Everything else is production: the real `run_ledger_clean`, the real prompts,
the real `_otr_spoken_text_policy`, and a real local model loaded through the
same `_otr_model_loader` seam the writer uses. So a recipe that wins here is
a recipe that ships, not a recipe that wins in a mock.

The fixtures carry `lab` metadata on each row -- `planted` (a defect, with
what kind) or `trap` (clean speech that must survive). Production ignores
unknown row keys, so the ground truth rides along without changing behaviour.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_ledger_clean as CLEAN  # noqa: E402
from nodes import _otr_spoken_text_policy as POLICY  # noqa: E402

DEFAULT_MODEL = "google/gemma-2-2b-it"


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------


#: WHICH LANE PUTS THE ACT FIELDS WHERE -- and this is not a detail.
#: Measured off live ledgers 2026-08-14: the WRITER lane stamps `arc_phase`
#: and `beat_intent` on the LINE row and leaves the beat rows carrying pure
#: transport; the CODEX lane populates both. The clean stage read the BEAT
#: first and therefore ran BLIND to the act on every writer-lane episode --
#: and the unit tests missed it because their fixtures put the fields on the
#: beat, which no writer-lane ledger does.
#:
#: So each bank's fixture is shaped like ITS OWN lane. Operator, 2026-08-14:
#: *"we need to apply the lesson and create a lab test for all source banks
#: -- they work a bit differently."* A lab whose fixtures all share one
#: convenient shape cannot catch a per-lane wiring fault, which is the exact
#: class of bug that got through.
LANE_SHAPE = {
    "original": "writer",
    "shakespeare": "writer",
    "public_domain": "writer",
    "media_archive": "writer",
    "scifi_news_pro": "codex",
}


def _ledger(bank: str, cast: "list[tuple[str, str]]", rows: "list[dict]") -> dict:
    """Assemble a ledger the way THIS BANK'S LANE actually shapes one.

    Each row in ``rows`` is {speaker, text, [planted|trap], [why]}. Where the
    act fields land depends on the lane -- see ``LANE_SHAPE``.
    """
    shape = LANE_SHAPE.get(bank, "writer")
    names = {name: char_id for name, char_id in cast}
    lines: "list[dict[str, Any]]" = []
    beats: "list[dict[str, Any]]" = []
    arcs = ("setup", "rising", "turn", "fall", "close")
    for i, row in enumerate(rows, start=1):
        speaker = row["speaker"]
        char_id = names.get(speaker, "announcer")
        beat_id = f"b{i:03d}"
        arc = arcs[min(i * len(arcs) // max(1, len(rows)), 4)]
        intent = row.get("intent", "carry the scene forward")

        beat: "dict[str, Any]" = {
            "beat_id": beat_id,
            "speaker": speaker,
            "char_id": char_id,
            "line_ids": [f"L{i:03d}"],
            "scene_id": None,
            "shot_id": None,
        }
        line: "dict[str, Any]" = {
            "line_id": f"L{i:03d}",
            "beat_id": beat_id,
            "char_id": char_id,
            "speaker": speaker,
            "speaker_role": (
                "announcer" if char_id == "announcer" else "character"),
            "text": row["text"],
            "lab": {
                "planted": row.get("planted", ""),
                "trap": row.get("trap", ""),
                "why": row.get("why", ""),
            },
        }
        # The writer lane's beats are transport ONLY -- the act lives on the
        # line. The codex lane populates both. Shaping the fixture per lane
        # is what would have caught the blindness bug.
        line["arc_phase"] = arc
        line["beat_intent"] = intent
        if shape == "codex":
            beat["arc_phase"] = arc
            beat["beat_intent"] = intent

        beats.append(beat)
        lines.append(line)
    return {
        "episode_id": f"lab_{bank}",
        "cast": [{"char_id": cid, "name": nm} for nm, cid in cast],
        "beats": beats,
        "lines": lines,
        "meta": {"source_bank": bank},
    }


# The defect taxonomy, planted deliberately across the banks:
#   bracket_start / bracket_end  -- production markup, one at each end
#   two_segments                 -- BOTH ends in one line (the multi-segment case)
#   scene_report                 -- unpunctuated narration; no pattern catches it
#   delivery_note                -- how a line is said
#   speaker_label                -- a NAME: prefix
#   third_person                 -- a character described from outside
#
# And the traps, which must come back CLEAN:
#   question / argument / metaphor / quoting / imperative / literary


def fixtures() -> "dict[str, dict]":
    banks: "dict[str, dict]" = {}

    banks["original"] = _ledger(
        "original",
        [("ANNOUNCER", "announcer"), ("Nan Reyes", "c01"), ("Web Doyle", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, from the lighthouse at Cutter's Reach.",
             "trap": "a clean announcer open", "why": "plain address"},
            {"speaker": "Nan Reyes",
             "text": "(She turns from the window.) The lamp has not turned "
                     "since Tuesday.",
             "planted": "bracket_start",
             "intent": "admit the lamp has been dead for days"},
            {"speaker": "Web Doyle",
             "text": "The log was signed Tuesday, right? Someone signed it.",
             "trap": "question", "why": "a question to another person"},
            {"speaker": "Nan Reyes",
             "text": "The door closes behind him. I told you he would not stay.",
             "planted": "scene_report",
             "intent": "note that the keeper has walked out"},
            {"speaker": "Web Doyle",
             "text": "It's more than a logbook, Nan. We have to see what's "
                     "inside.",
             "trap": "argument", "why": "argues and names the addressee"},
            {"speaker": "Nan Reyes",
             "text": "I can assure you the light will burn. (Nan throws the "
                     "keys at Web, who catches them.)",
             "planted": "bracket_end",
             "intent": "hand over the keys in anger"},
            {"speaker": "Web Doyle",
             "text": "That lamp stands as our only shield.",
             "trap": "metaphor", "why": "an image, not stage business"},
            {"speaker": "Nan Reyes",
             "text": "Her voice tight, she said the tower was never ours.",
             "planted": "delivery_note",
             "intent": "recall what the keeper's widow said"},
            {"speaker": "Web Doyle",
             "text": "She said she would come back before the tide.",
             "trap": "quoting", "why": "quoting another person is speech"},
            {"speaker": "ANNOUNCER",
             "text": "And so the light at Cutter's Reach went dark.",
             "trap": "clean announcer close", "why": "plain narration to air"},
        ],
    )

    banks["media_archive"] = _ledger(
        "media_archive",
        [("ANNOUNCER", "announcer"),
         ("Montgomery Bernard", "c01"), ("Dale Halloway", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, from the vault beneath the archive.",
             "trap": "clean announcer open", "why": "plain address"},
            {"speaker": "Montgomery Bernard",
             "text": "(Montgomery sighs) I've already given you the reel. "
                     "(Montgomery sighs again)",
             "planted": "two_segments",
             "intent": "refuse to release the fragment"},
            {"speaker": "Dale Halloway",
             "text": "This shipment was listed as A.P.O. 86-574, reel two, "
                     "right?",
             "trap": "question", "why": "THE MEASURED FALSE POSITIVE, live"},
            {"speaker": "Montgomery Bernard",
             "text": "The air in the studio crackles as Dale Bernard stares "
                     "at Montgomery.",
             "planted": "scene_report",
             "intent": "hold the silence between them"},
            {"speaker": "Dale Halloway",
             "text": "It's more than just a reel number, Montgomery. We have "
                     "to see what's inside.",
             "trap": "argument", "why": "THE MEASURED FALSE POSITIVE, live"},
            {"speaker": "Montgomery Bernard",
             "text": "MONTGOMERY: The special screening is next week.",
             "planted": "speaker_label",
             "intent": "reveal the screening"},
            {"speaker": "Dale Halloway",
             "text": "Then step back, Montgomery, and let the projector run.",
             "trap": "imperative", "why": "an order is speech"},
            {"speaker": "ANNOUNCER",
             "text": "The reel returned to its can, and the vault closed.",
             "trap": "clean announcer close", "why": "plain narration to air"},
        ],
    )

    banks["shakespeare"] = _ledger(
        "shakespeare",
        [("ANNOUNCER", "announcer"), ("MALVOLIO", "c01"), ("OLIVIA", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, a scene from Twelfth Night.",
             "trap": "clean announcer open", "why": "plain address"},
            {"speaker": "MALVOLIO",
             "text": "She sighed, and the crown grew heavy on her brow.",
             "trap": "literary",
             "why": "FIDELITY: the author's own third person is not a defect"},
            {"speaker": "OLIVIA",
             "text": "MALVOLIO: Is this a letter which I see before me?",
             "planted": "speaker_label",
             "intent": "read the letter aloud"},
            {"speaker": "MALVOLIO",
             "text": "Some are born great, some achieve greatness.",
             "trap": "literary", "why": "FIDELITY: source language"},
            {"speaker": "OLIVIA",
             "text": "(He crosses the garden.) Go to, thou art made if thou "
                     "desir'st to be so.",
             "planted": "bracket_start",
             "intent": "send Malvolio away"},
            {"speaker": "MALVOLIO",
             "text": "Then step back, madam, and let the letter speak.",
             "trap": "imperative", "why": "an order is speech"},
            {"speaker": "ANNOUNCER",
             "text": "So ends the scene, as Shakespeare set it down.",
             "trap": "clean announcer close", "why": "plain narration"},
        ],
    )

    banks["public_domain"] = _ledger(
        "public_domain",
        [("ANNOUNCER", "announcer"), ("Jonathan", "c01"), ("Mina", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, a chapter from a traveller's diary.",
             "trap": "clean announcer open", "why": "plain address"},
            {"speaker": "Jonathan",
             "text": "He turned his face to the wall, and the candle guttered.",
             "trap": "literary",
             "why": "FIDELITY: the author's own third person is not a defect"},
            {"speaker": "Mina",
             "text": "The latch lifts by itself. I have written it all down.",
             "planted": "scene_report",
             "intent": "notice the door opening"},
            {"speaker": "Jonathan",
             "text": "You asked me for the whole of it, and here it is.",
             "trap": "argument", "why": "speaks directly to another"},
            {"speaker": "Mina",
             "text": "[door creaks] The count keeps no mirrors in this house.",
             "planted": "bracket_start",
             "intent": "report the absent mirrors"},
            {"speaker": "ANNOUNCER",
             "text": "And the diary closes upon that night.",
             "trap": "clean announcer close", "why": "plain narration"},
        ],
    )

    banks["scifi_news_pro"] = _ledger(
        "scifi_news_pro",
        [("ANNOUNCER", "announcer"), ("Ruiz", "c01"), ("Halloway", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, from a hospital that keeps no night shift.",
             "trap": "clean announcer open", "why": "plain address"},
            {"speaker": "Ruiz",
             "text": "UCLA Health listed eleven sites. Her hands shake as she "
                     "reads.",
             "planted": "delivery_note",
             "intent": "read the site count aloud"},
            {"speaker": "Halloway",
             "text": "Eleven sites, fourteen institutions -- is that the whole "
                     "list?",
             "trap": "question", "why": "a question full of figures"},
            {"speaker": "Ruiz",
             "text": "The monitor flatlines. Someone should call the desk.",
             "planted": "scene_report",
             "intent": "react to the alarm"},
            {"speaker": "Halloway",
             "text": "She said the ward was closed before midnight.",
             "trap": "quoting", "why": "quoting another person is speech"},
            {"speaker": "ANNOUNCER",
             "text": "Drawn from reporting on the UCLA Health network.",
             "trap": "clean coda", "why": "names the source, pure speech"},
        ],
    )

    return banks


# ---------------------------------------------------------------------------
# F2 -- the wrong character's speech
# ---------------------------------------------------------------------------
# The operator's OTHER failure class, and the one with no detector anywhere in
# the pipeline: *"I'm more concerned about not finding and fixing non-dialogue,
# or the WRONG CHARACTER'S SPEECH."*
#
# It is harder to plant honestly than F1, because the defect is not IN the
# words -- it is in the relationship between the words and the roster. So each
# planted row is wrong for a REASON a reader could name, and each trap is a
# thing that LOOKS like the defect and is not.
#
# The traps matter more here than they did for F1. Characters name each other
# constantly, quote each other, and finish each other's thoughts; a judge that
# treats any of that as misattribution would rewrite half an episode.


def f2_fixtures() -> "dict[str, dict]":
    banks: "dict[str, dict]" = {}

    banks["original"] = _ledger(
        "original",
        [("ANNOUNCER", "announcer"), ("Nan Reyes", "c01"), ("Web Doyle", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, the keeper and the inspector.",
             "trap": "announcer open", "why": "plain address"},
            {"speaker": "Nan Reyes",
             "text": "Then step back, Nan, and let me look at the log myself.",
             "planted": "self_address",
             "why": "she orders HERSELF by name -- nobody addresses themselves",
             "intent": "order the other one away from the log"},
            {"speaker": "Web Doyle",
             "text": "Step back yourself, Nan. I have read that log twice.",
             "trap": "names the other",
             "why": "naming ANOTHER character is ordinary dialogue"},
            {"speaker": "Nan Reyes",
             "text": "As the inspector, I am required to file this by Friday.",
             "planted": "role_claim",
             "why": "Nan is the KEEPER; Web is the inspector",
             "intent": "assert authority over the log"},
            {"speaker": "Web Doyle",
             "text": "She said she would file it by Friday, and she has not.",
             "trap": "quoting",
             "why": "reporting another's words is ordinary dialogue"},
            {"speaker": "Nan Reyes",
             "text": "I keep this light. You only visit it.",
             "trap": "role stated correctly",
             "why": "the keeper saying she keeps the light"},
            {"speaker": "ANNOUNCER",
             "text": "The lamp stayed dark until morning.",
             "trap": "announcer close", "why": "plain narration"},
        ],
    )

    banks["shakespeare"] = _ledger(
        "shakespeare",
        [("ANNOUNCER", "announcer"), ("MALVOLIO", "c01"), ("OLIVIA", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, a scene from Twelfth Night.",
             "trap": "announcer open", "why": "plain address"},
            {"speaker": "MALVOLIO",
             "text": "Go to, Malvolio, thou art made if thou desir'st to be so.",
             "planted": "self_address",
             "why": "the letter's line to Malvolio, put in HIS mouth",
             "intent": "read the letter's promise"},
            {"speaker": "OLIVIA",
             "text": "Go to, Malvolio, thou art made if thou desir'st to be so.",
             "trap": "same words, right mouth",
             "why": "IDENTICAL text -- correct when SHE says it. The defect "
                    "is never in the words alone"},
            {"speaker": "MALVOLIO",
             "text": "Some are born great, some achieve greatness.",
             "trap": "source language", "why": "FIDELITY: the author's line"},
            {"speaker": "OLIVIA",
             "text": "My lady bade me wear these cross-garters.",
             "planted": "role_claim",
             "why": "OLIVIA IS the lady; she has no lady to obey",
             "intent": "explain the garters"},
            {"speaker": "ANNOUNCER",
             "text": "So ends the scene, as Shakespeare set it down.",
             "trap": "announcer close", "why": "plain narration"},
        ],
    )

    banks["scifi_news_pro"] = _ledger(
        "scifi_news_pro",
        [("ANNOUNCER", "announcer"), ("Dr. Ruiz", "c01"), ("Halloway", "c02")],
        [
            {"speaker": "ANNOUNCER",
             "text": "Tonight, from a hospital that keeps no night shift.",
             "trap": "announcer open", "why": "plain address"},
            {"speaker": "Dr. Ruiz",
             "text": "You are the physician here, Ruiz -- you tell me what "
                     "the count means.",
             "planted": "self_address",
             "why": "Ruiz defers to Ruiz",
             "intent": "demand an explanation of the figures"},
            {"speaker": "Halloway",
             "text": "You are the physician here, Doctor. You tell me what "
                     "the count means.",
             "trap": "same demand, right mouth",
             "why": "correct when the LAYMAN says it"},
            {"speaker": "Dr. Ruiz",
             "text": "I have never read a chart in my life.",
             "planted": "knowledge_mismatch",
             "why": "the physician cannot claim never to have read a chart",
             "intent": "admit being out of depth"},
            {"speaker": "Halloway",
             "text": "I have never read a chart in my life. That is why you "
                     "are here.",
             "trap": "same claim, right mouth",
             "why": "true and ordinary from the layman"},
            {"speaker": "ANNOUNCER",
             "text": "Drawn from reporting on the UCLA Health network.",
             "trap": "clean coda", "why": "names the source"},
        ],
    )

    return banks


# ---------------------------------------------------------------------------
# the real corpus -- REAL bad ledgers, with the act artifacts derived
# ---------------------------------------------------------------------------
# Operator, 2026-08-14: *"we have lots of bad ledgers to pull, but not sure
# about the act / story artifacts ... you can derive your own."*
#
# He is right on both counts, and both halves matter. The invented fixtures
# above give LABELLED ground truth, which is the only way to measure
# precision. The real corpus gives the true DEFECT DISTRIBUTION -- what the
# writer actually produces, in the proportions it actually produces it, with
# real neighbours and real speakers. A recipe should be judged on both: the
# fixtures say whether it is correct, the corpus says whether it is correct
# on the material that actually exists.
#
# The act artifacts really are thin on older episodes -- most rows on disk
# predate `beat_intent` entirely. So where the ledger carries them we use
# them, and where it does not we DERIVE what is honestly derivable (the arc
# position, from where the row sits in the episode) and leave the rest empty
# rather than inventing a dramatic intent no one ever wrote. An invented
# intent would be a fabricated input, and the pass would be measured against
# a story that never existed.


def _derive_beats(data: dict) -> "list[dict]":
    """Beats for a ledger that may have none, or none with intent.

    Only the arc position is invented, and only because it IS derivable: a
    row two thirds of the way through an episode is in its falling action
    whether or not anybody stamped that. `beat_intent` is left alone -- if
    the episode never recorded what a moment was for, this lab will not
    pretend to know.
    """
    existing = {
        str(b.get("beat_id")): dict(b)
        for b in (data.get("beats") or []) if isinstance(b, dict)
    }
    rows = [r for r in (data.get("lines") or []) if isinstance(r, dict)]
    voiced = [r for r in rows if r.get("speaker_role") in POLICY.VOICED_ROLES]
    total = max(1, len(voiced))
    arcs = ("setup", "rising", "turn", "fall", "close")

    out: "list[dict]" = []
    seen = 0
    for row in rows:
        beat_id = str(row.get("beat_id") or "")
        if not beat_id:
            continue
        beat = existing.get(beat_id) or {
            "beat_id": beat_id,
            "speaker": row.get("speaker"),
            "char_id": row.get("char_id"),
        }
        if row.get("speaker_role") in POLICY.VOICED_ROLES:
            seen += 1
        if not str(beat.get("arc_phase") or "").strip():
            beat["arc_phase"] = arcs[min(seen * len(arcs) // total, 4)]
        out.append(beat)
    return out


def score(ledger: dict, receipt: dict) -> dict:
    """Grade the run against the ground truth planted in the fixture."""
    lab = {
        str(r.get("line_id")): (r.get("lab") or {})
        for r in ledger.get("lines", [])
    }
    flagged = {str(r.get("line_id")) for r in receipt.get("rows", [])}
    # An F2 run scores on the ATTRIBUTION verdict, not on F1 findings.
    if receipt.get("f2_content_rows") is not None and receipt.get("f2"):
        f2_hits = {
            str(e.get("line_id")) for e in receipt.get("f2", [])
            if e.get("outcome") in ("reattributed", "misattributed_unfixed")
        }
        if f2_hits or receipt.get("f2_content_rows"):
            flagged = f2_hits
    outcome = {
        str(r.get("line_id")): str(r.get("outcome") or "")
        for r in receipt.get("rows", [])
    }

    planted = {lid for lid, m in lab.items() if m.get("planted")}
    traps = {lid for lid, m in lab.items() if m.get("trap")}

    caught = planted & flagged
    missed = planted - flagged
    false_alarms = traps & flagged
    for entry in receipt.get("f2", []):
        if entry.get("outcome"):
            outcome[str(entry.get("line_id"))] = str(entry["outcome"])
    repaired = {
        lid for lid in caught
        if outcome.get(lid) in ("repaired", "reattributed",
                               "reattributed_unverified")
    }
    improved = {lid for lid in caught if outcome.get(lid) == "improved"}

    return {
        "planted": len(planted),
        "caught": len(caught),
        "missed": sorted(missed),
        "traps": len(traps),
        "false_alarms": sorted(false_alarms),
        "repaired": len(repaired),
        "improved": len(improved),
        "recall": (len(caught) / len(planted)) if planted else 1.0,
        "precision_on_traps": (
            1.0 - (len(false_alarms) / len(traps))) if traps else 1.0,
        "model_calls": receipt.get("model_calls", 0),
        # THE SIGHT PROOF, per bank. A recipe that scores well while blind is
        # scoring by luck, and the banks differ in where they put the act --
        # so this has to be read per bank, not once.
        "saw": dict(receipt.get("context_seen") or {}),
        "act_briefs": dict(receipt.get("act_briefs") or {}),
        "missed_kinds": sorted(
            lab[lid].get("planted", "") for lid in missed),
        "false_alarm_reasons": sorted(
            lab[lid].get("trap", "") for lid in false_alarms),
    }


def _verdict_table(name: str, s: dict) -> str:
    return (
        f"{name:16s} recall {s['caught']}/{s['planted']}"
        f"  traps kept {s['traps'] - len(s['false_alarms'])}/{s['traps']}"
        f"  repaired {s['repaired']}+{s['improved']}"
        f"  calls {s['model_calls']}"
    )


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


def build_slot_fn(model_id: str):
    """A real local model on the real loader seam -- not a mock.

    A recipe that wins against a stub proves nothing: the whole question is
    what a 2B actually does with the prompt.
    """
    from nodes import _otr_model_loader as LOADER

    print(f"[lab] loading {model_id} ...", flush=True)
    started = time.time()
    entry = LOADER.load_llm(model_id, optimization_profile="Standard")
    fn = LOADER.make_generate_fn(entry)
    print(f"[lab] loaded in {time.time() - started:.1f}s", flush=True)
    return fn


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", default=None, help="one bank, or all")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each bank N times -- a 2B is not deterministic")
    ap.add_argument("--dry", action="store_true",
                    help="no model: what the pattern floor alone would do")
    ap.add_argument("--show", action="store_true",
                    help="print every flagged line and what happened to it")
    ap.add_argument("--json", default=None, help="write the scores here")
    # A/B THE RECIPE. Operator, 2026-08-14: *"the repair needs to be model
    # agnostic -- do a bunch of A/Bs until you get something that works."*
    ap.add_argument("--votes", type=int, default=None,
                    help="judge reads per row; 2 keeps only what both name")
    ap.add_argument("--judge-temp", type=float, default=None,
                    help="judge temperature")
    ap.add_argument("--f2", action="store_true",
                    help="run the F2 fixtures: the WRONG CHARACTER'S speech")
    ap.add_argument("--per-sentence", action="store_true",
                    help="shrink the job: one judge call per SENTENCE")
    ap.add_argument("--brief-only", action="store_true",
                    help="split the load: the repair reads the act brief and "
                         "its two neighbours instead of the whole window")
    args = ap.parse_args(argv)

    if args.votes is not None:
        CLEAN.JUDGE_VOTES = args.votes
    if args.judge_temp is not None:
        CLEAN.JUDGE_TEMPERATURE = args.judge_temp
    if args.brief_only:
        CLEAN.REPAIR_READS_BRIEF_ONLY = True
    if args.per_sentence:
        CLEAN.JUDGE_PER_SENTENCE = True

    # F2 is its own fixture bank and its own judge -- the defect is not in
    # the words, so the F1 fixtures cannot test it.
    if args.f2:
        CLEAN.JUDGE_ATTRIBUTION = True
        banks = f2_fixtures()
    else:
        banks = fixtures()
    if args.bank:
        if args.bank not in banks:
            print(f"unknown bank {args.bank!r}; have: {', '.join(banks)}")
            return 2
        banks = {args.bank: banks[args.bank]}

    slot_fn = None if args.dry else build_slot_fn(args.model)

    print()
    print(f"POLICY {POLICY.SPOKEN_TEXT_POLICY_ID}   "
          f"STAGE {CLEAN.LEDGER_CLEAN_VERSION}   "
          f"MODEL {'(none -- patterns only)' if args.dry else args.model}")
    print(f"RECIPE votes={CLEAN.JUDGE_VOTES} "
          f"judge_temp={CLEAN.JUDGE_TEMPERATURE} "
          f"attempts={CLEAN._MAX_ATTEMPTS} "
          f"brief_only={CLEAN.REPAIR_READS_BRIEF_ONLY} "
          f"per_sentence={CLEAN.JUDGE_PER_SENTENCE} "
          f"before={CLEAN._CONTEXT_ROWS} after={CLEAN._AFTER_ROWS}")
    print("=" * 78)

    all_scores: "list[dict[str, Any]]" = []
    for bank, template in banks.items():
        for run in range(1, args.repeat + 1):
            ledger = json.loads(json.dumps(template))  # a fresh copy each run
            started = time.time()
            receipt = CLEAN.run_ledger_clean(
                ledger, slot_fn=slot_fn, bank_id=bank)
            elapsed = time.time() - started
            s = score(ledger, receipt)
            s.update({"bank": bank, "run": run, "seconds": round(elapsed, 1)})
            all_scores.append(s)

            label = bank if args.repeat == 1 else f"{bank} #{run}"
            print(_verdict_table(label, s) + f"  {elapsed:.0f}s")
            if s["missed"]:
                print(f"                 MISSED: {', '.join(s['missed'])} "
                      f"({', '.join(s['missed_kinds'])})")
            if s["false_alarms"]:
                print(f"                 FALSE ALARM: "
                      f"{', '.join(s['false_alarms'])} "
                      f"({', '.join(s['false_alarm_reasons'])})")
            saw = s.get("saw") or {}
            rows_n = saw.get("rows_with_arc_phase", 0)
            if not args.dry:
                blind = "  *** BLIND ***" if not rows_n else ""
                print(f"                 saw: act on {rows_n} row(s), "
                      f"intent on {saw.get('rows_with_beat_intent', 0)}, "
                      f"brief on {saw.get('rows_with_act_brief', 0)}, "
                      f"before {saw.get('rows_with_lines_before', 0)}, "
                      f"after {saw.get('rows_with_lines_after', 0)}"
                      f"{blind}")
                for phase, brief in (s.get("act_briefs") or {}).items():
                    print(f"                   {phase}: {brief}")
            if args.show:
                for row in receipt.get("rows", []):
                    lid = str(row.get("line_id"))
                    truth = (
                        "PLANTED" if lid in
                        {r for r in s["missed"]} | set() else "")
                    meta = next(
                        (r.get("lab") or {}) for r in ledger["lines"]
                        if str(r.get("line_id")) == lid
                    )
                    truth = "PLANTED" if meta.get("planted") else "TRAP"
                    print(f"    {lid} [{truth}] {row.get('outcome')} "
                          f"found_by={row.get('found_by')}")
                    for c in (row.get("complaint") or []):
                        print(f"        -> {c.get('quote')!r} ({c.get('why')})")

    print("=" * 78)
    if all_scores:
        planted = sum(s["planted"] for s in all_scores)
        caught = sum(s["caught"] for s in all_scores)
        traps = sum(s["traps"] for s in all_scores)
        alarms = sum(len(s["false_alarms"]) for s in all_scores)
        repaired = sum(s["repaired"] for s in all_scores)
        calls = sum(s["model_calls"] for s in all_scores)
        print(f"TOTAL   recall {caught}/{planted} "
              f"({caught / planted:.0%})   "
              f"traps kept {traps - alarms}/{traps} "
              f"({(traps - alarms) / traps:.0%})   "
              f"repaired {repaired}   calls {calls}")
        print()
        print("A recipe is BETTER only if recall rises and traps kept does "
              "not fall. Chasing recall alone is how the last cut condemned "
              "clean dialogue.")

    if args.json:
        Path(args.json).write_text(
            json.dumps(all_scores, indent=2), encoding="utf-8")
        print(f"\nscores -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
