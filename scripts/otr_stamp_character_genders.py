"""Write the gender a source states -- or, failing that, recalls -- about its own
characters into its provenance sidecar. Four rungs, run offline, results committed.

WHY. `cast_source_contract.gender_by_name` shipped `{}` on every public_domain
episode, because the field is fed from a provenance sidecar and 64 of 65 units
had none. Empty field -> blind 40/40/20 roll -> three measured inversions on
three different sources: GERTRUDE male, LORD RONALD female (the script calls the
male-voiced character *"Miss McFiggins"*), and then AHAB, of Moby-Dick, in a
woman's voice. Tiers 1-2 closed 132 names; five units (ten names, Alice and
Elizabeth Bennet among them) still declined every hint and rolled, and 32 of 85
Shakespeare roster rows sat at `unknown`. Tiers 3-4 close those (spec v2,
2026-08-28; operator rulings of that day baked in).

THE LADDER, highest rung first. The first rung that answers wins:
  1. roster          the source's own cast block (prose rarely has one).
  2. pronouns        the author's own pronouns around every mention (mechanical
                     scan, in-process; the text never leaves this machine).
  3. llm_recall      ONE question to a local model, naming the character AND ITS
                     WORK ("In 'Pride and Prejudice' by Jane Austen, is the
                     character 'Elizabeth Bennet' male or female?") -- recall of a
                     published work, never a guess from the shape of a name, and
                     never any source text. Cached in a committed per-bank INDEX
                     so each (work, name) is asked once, ever, and the operator can
                     read and correct the answers by hand (`locked: true`).
  4. name_frequency  the curated first-name pool (config/cast_pools.py). A
                     conservative dictionary: unlisted, unisex or descriptive names
                     ("the Creature") DECLINE. It is NOT total by design.

DECLINING IS AN ANSWER. A candidate no rung can decide is OMITTED from the prose
`characters` list rather than written as `unknown`. Downstream that is a
render-time JOIN MISS, which preserves today's roll -- exactly the behaviour that
name has now, so an omitted name is a no-change rather than a regression. On the
Shakespeare bank a declined row simply keeps its fetcher-written `unknown`.

OWNERSHIP, because a sidecar is shared property (one owner per field):
  * prose: `characters[]` and `gender_ladder` -- THIS tool, sole writer. Every base
    provenance field -- the FETCHER (`otr_fetch_public_domain.py`). This tool never
    overwrites one; it FILLS an absent base field from local truth only (the
    manifest row plus the text on disk) and never invents `fetched_utc`.
  * shakespeare: the fetcher writes every row from the parsed dramatis personae;
    THIS tool fills ONLY rows whose `gender` is `unknown` (operator ruling 1,
    2026-08-28: known rows are untouchable, byte for byte) and adds
    `gender_ladder`. The curated supplement beside the manifest is consulted
    before any model, so sidecar and supplement can never disagree.
  * the index (`character_gender_index.json` beside each bank's manifest) -- THIS
    tool writes new entries; the operator may edit any entry and lock it.

MONOTONIC MERGE (re-runs). Anchored on `body_sha256`: if the text changed, the
ladder runs fresh for that unit. If the text is unchanged, an existing row is
replaced only by an EQUAL or HIGHER rung (equal = refresh after a prompt/model
revision; lower never demotes -- a model's recall can never displace the author's
pronouns), and a declined name keeps its old row. `tier_counts` is always derived
from the final rows. `ran_utc` moves only when content moves.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nodes._otr_character_roster import (  # noqa: E402
    infer_gender, parse_character_roster,
)
from nodes._otr_gender_pronoun_scan import scan_gender  # noqa: E402
from nodes._otr_roster_gender import (  # noqa: E402
    load_gender_supplement, sidecar_path_for_text, strip_honorifics,
)

BANK_DIR = REPO_ROOT / "config" / "source_banks" / "public_domain_story"
MANIFEST = BANK_DIR / "manifest.sample.json"
SHAKESPEARE_DIR = REPO_ROOT / "config" / "source_banks" / "shakespeare"
SHAKESPEARE_MANIFEST = SHAKESPEARE_DIR / "curated_scenes.sample.json"
SHAKESPEARE_AUTHOR = "William Shakespeare"

LADDER_VERSION = "gender_ladder_v2"
SIDECAR_SCHEMA = "otr_source_provenance_v1"
INDEX_FILENAME = "character_gender_index.json"
INDEX_SCHEMA = 1
PROMPT_VERSION = "character_gender_recall_v1"
DEFAULT_MODEL = "google/gemma-4-E4B-it"

#: Rung order, highest first. A row's `gender_source` is one of these (or the
#: fetcher's roster vocabulary on the Shakespeare bank, which ranks as rung 1).
TIER_ORDER = ("roster", "pronouns", "llm_recall", "name_frequency")
_ROSTER_SOURCES = frozenset({
    "roster", "relation", "title", "group", "back_reference", "supplement",
})
CONFIDENCE = {
    "roster": "known", "pronouns": "known", "supplement": "known",
    "llm_recall": "recalled", "name_frequency": "inferred",
}
#: Ruling 3: the announcer's gender is random BY DESIGN, so pinning it would
#: remove variety the show wants. `cradle_protocol` really does carry "the
#: Announcer" in its cast_hints, so this exclusion is load-bearing, not defensive.
#: Group speakers (ALL / BOTH / CHORUS) are not people and never get a gender.
_EXCLUDED = frozenset({
    "ANNOUNCER", "THE ANNOUNCER", "ALL", "BOTH", "CHORUS", "OMNES", "SERVANTS",
})


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).upper()


def _rung(source: str) -> int:
    """Rung number of a `gender_source` value; 1 is the highest, 5 is unknown."""
    src = str(source or "")
    if src in _ROSTER_SOURCES:
        return 1
    try:
        return TIER_ORDER.index(src) + 1
    except ValueError:
        return 5


# ---------------------------------------------------------------------------
# tiers 1-2 (unchanged shape: tests call `_decide(name, text, others)`)
# ---------------------------------------------------------------------------

def _candidates(source: Dict[str, Any]) -> List[str]:
    """The names to decide, from the manifest's own cast hints.

    `cast_hints` is schema-required non-empty and authored in the vendor table,
    and it already names every character the three live failures inverted --
    "Captain Ahab", "Gertrude", "Lord Ronald". The brief's own extracted names
    join these rows at render through the four join tiers in
    `_otr_roster_gender.resolve_roster_gender`, so a wider net is not needed.
    """
    out: List[str] = []
    for hint in source.get("cast_hints") or []:
        name = str(hint or "").strip()
        if not name or name.upper() in _EXCLUDED:
            continue
        if name not in out:
            out.append(name)
    return out


def _aliases_for(name: str) -> List[str]:
    """Extra join keys, because the render extracts "SCROOGE", not "EBENEZER
    SCROOGE". The existing join tiers cannot bridge that on their own."""
    from nodes._otr_gender_pronoun_scan import mention_forms

    forms = [f for f in mention_forms(name) if f != name.strip().lower()]
    return [f for f in forms if f]


def _decide(name: str, text: str, others: List[str]) -> Tuple[str, str, str]:
    """(gender, gender_source, evidence) -- or ("", "", reason) when declined.

    Tier 1 ROSTER first: free, deterministic, and it makes this ladder uniform
    with the Shakespeare lane that already works. It fires on roughly zero prose
    units (prose has no cast block) and is kept anyway, exactly as specified.
    Tier 2 PRONOUNS second: the author's own words.
    """
    for record in parse_character_roster(text):
        if record.matches(name):
            # TWO arguments and a TUPLE back -- `infer_gender(name, description)`
            # returns `(gender, source)`, and the roster module's own call sites
            # (`_otr_character_roster.py:290,333,335`) all unpack it. This line
            # once passed ONE argument and assigned the result to a bare name, so
            # it raised TypeError the instant a cast block parsed, and nothing
            # caught it -- it would have killed the whole 65-unit run rather
            # than one unit. It never fired only because prose has no cast
            # block, so every shipped sidecar reads `"roster": 0`.
            gender, basis = infer_gender(
                record.name, record.description or record.name)
            if gender in ("male", "female"):
                return gender, "roster", (
                    "source cast list (%s): %s"
                    % (basis, record.description or record.name)
                )
    verdict = scan_gender(name, text, other_names=others)
    if verdict.decided:
        return verdict.gender, "pronouns", verdict.evidence
    return "", "", verdict.evidence


# ---------------------------------------------------------------------------
# tier 3: recall of the character in its work, through a committed index
# ---------------------------------------------------------------------------

def recall_messages(name: str, title: str, author: str = "") -> List[Dict[str, str]]:
    """The ONE question tier 3 asks. It carries the work's title (and author
    when known) and the character's name -- never a passage, never the text.
    Blocker B1 of spec v2 stays answered structurally: there is no parameter
    through which source text could reach the model."""
    where = 'In "%s"%s' % (title, (" by %s" % author) if author else "")
    return [
        {"role": "system", "content":
            "You answer questions about characters in published literature. "
            "If you do not know the work or the character, answer 'unsure'. "
            "Never guess to sound confident."},
        {"role": "user", "content": (
            '%s, is the character "%s" male or female?\n'
            "Answer 'male', 'female', or 'unsure' if you do not know this "
            "character. Give a one-phrase reason." % (where, name))},
    ]


def _parse_opinion(raw: str) -> Tuple[str, str]:
    """(gender, reason) from the model's JSON, or ("unparseable", raw head)."""
    try:
        data = json.loads(str(raw or ""))
        gender = str(data.get("gender") or "").strip().lower()
        reason = " ".join(str(data.get("reason") or "").split())[:200]
    except (ValueError, AttributeError):
        return "unparseable", str(raw or "")[:120]
    if gender not in ("male", "female", "unsure"):
        return "unparseable", str(raw or "")[:120]
    return gender, reason


class GenderIndex:
    """The committed name index: (work, name) -> the verdict the model gave, once.

    Human-correctable: edit `gender` and set `locked: true` and the stamper takes
    the entry as an operator ruling and never asks again. A locked entry with an
    EMPTY gender means "leave this one to the roll" (ARIEL and PUCK, 2026-08-28).
    """

    def __init__(self, path: pathlib.Path):
        self.path = pathlib.Path(path)
        self.data: Dict[str, Any] = {
            "schema_version": INDEX_SCHEMA,
            "notes": [],
            "entries": {},
        }
        self._loaded_bytes = b""
        if self.path.is_file():
            self._loaded_bytes = self.path.read_bytes()
            self.data = json.loads(self._loaded_bytes.decode("utf-8"))
            self.data.setdefault("entries", {})

    @staticmethod
    def key(title: str, name: str) -> str:
        return "%s|%s" % (_norm(title), _norm(name))

    def get(self, title: str, name: str) -> Optional[Dict[str, Any]]:
        entry = self.data["entries"].get(self.key(title, name))
        return entry if isinstance(entry, dict) else None

    def put(self, title: str, name: str, entry: Dict[str, Any]) -> None:
        self.data["entries"][self.key(title, name)] = entry

    def render(self) -> bytes:
        body = dict(self.data)
        body["entries"] = {k: body["entries"][k] for k in sorted(body["entries"])}
        return (json.dumps(body, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    def save(self) -> bool:
        """Write only when the bytes moved. Returns True when written."""
        rendered = self.render()
        if rendered == self._loaded_bytes:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(rendered)
        self._loaded_bytes = rendered
        return True


class Recall:
    """Tier 3. `generate_fn` is the constrained closure from
    `_otr_constrained_generate.make_constrained_generate_fn(entry, CharacterGenderOpinion)`
    or None (tier 3 switched off: every ask declines with a reason)."""

    def __init__(self, generate_fn: Optional[Callable[..., str]], index: GenderIndex,
                 *, model_id: str = DEFAULT_MODEL, clock: Optional[Callable[[], str]] = None):
        self.generate_fn = generate_fn
        self.index = index
        self.model_id = model_id
        self.clock = clock or (lambda: _dt.datetime.now(_dt.timezone.utc)
                               .replace(microsecond=0).isoformat())
        self.census = {"cached": 0, "asked": 0, "unsure": 0, "unparseable": 0,
                       "locked": 0, "off": 0}

    def ask(self, name: str, title: str, author: str = "") -> Tuple[str, str, str]:
        """(gender, evidence, gender_source) -- gender "" when tier 3 declines.
        A LOCKED entry is the operator's word and is stamped as `supplement`
        (known), never as the model's recall."""
        entry = self.index.get(title, name)
        if entry is not None and entry.get("locked"):
            self.census["locked"] += 1
            gender = str(entry.get("gender") or "").strip().lower()
            if gender in ("male", "female"):
                return gender, "operator-locked index entry: %s" % (
                    entry.get("reason") or "no reason recorded"), "supplement"
            return "", "operator-locked index entry says: stays on the roll", ""
        if entry is not None and entry.get("prompt_version") == PROMPT_VERSION \
                and entry.get("model") == self.model_id:
            self.census["cached"] += 1
            gender = str(entry.get("gender") or "").strip().lower()
            if gender in ("male", "female"):
                return gender, self._evidence(title, author, entry), "llm_recall"
            return "", "recall (cached): model was unsure about %r in %r" % (name, title), ""
        if self.generate_fn is None:
            self.census["off"] += 1
            return "", "tier 3 off (no model loaded); not asked", ""
        self.census["asked"] += 1
        raw = self.generate_fn(recall_messages(name, title, author),
                               temperature=0.0, max_new_tokens=120)
        gender, reason = _parse_opinion(raw)
        if gender == "unparseable":
            self.census["unparseable"] += 1
            return "", "recall: unparseable answer for %r in %r: %s" % (name, title, reason), ""
        new_entry = {
            "gender": gender if gender in ("male", "female") else "",
            "answer": gender,
            "reason": reason,
            "asked_as": "title",
            "model": self.model_id,
            "prompt_version": PROMPT_VERSION,
            "asked_utc": self.clock(),
            "locked": False,
        }
        self.index.put(title, name, new_entry)
        if gender == "unsure":
            self.census["unsure"] += 1
            return "", "recall: model was unsure about %r in %r" % (name, title), ""
        return gender, self._evidence(title, author, new_entry), "llm_recall"

    @staticmethod
    def _evidence(title: str, author: str, entry: Dict[str, Any]) -> str:
        return 'recall of "%s"%s: %s (%s, %s)' % (
            title, (" by %s" % author) if author else "",
            entry.get("reason") or "no reason recorded",
            entry.get("model") or "?", str(entry.get("asked_utc") or "")[:10])


# ---------------------------------------------------------------------------
# tier 4: the curated first-name pool -- conservative, declines when unlisted
# ---------------------------------------------------------------------------

def name_frequency(name: str) -> Tuple[str, str]:
    """(gender, evidence) from `config/cast_pools.gender_of_first_name` on the
    honorific-stripped first token. "the Creature", "Sancho", "Fitzwilliam" and
    every unisex name DECLINE: the floor is a dictionary, not a guess."""
    from config.cast_pools import gender_of_first_name

    stripped = strip_honorifics(name).strip()
    if not stripped or stripped.lower().startswith(("the ", "a ", "an ")):
        return "", "name_frequency: %r is a description, not a name" % (name,)
    if len(stripped.split()) == 1 and len(str(name).split()) > 1:
        # "Dr. Kelly", "Mrs. Sappleton": a title plus ONE token is a SURNAME,
        # and a surname that happens to be in the first-name pool ("Kelly")
        # would come back with a confident wrong gender. Measured 2026-09-02
        # on man_size_in_marble (Dr. Kelly -> female).
        return "", "name_frequency: %r is a title plus a surname; declined" % (name,)
    head = stripped.split()[0]
    verdict = gender_of_first_name(head)
    if verdict in ("male", "female"):
        return verdict, "name_frequency: first name %r is %s in the curated pool" % (
            head, verdict)
    return "", "name_frequency: first name %r is %s in the curated pool; declined" % (
        head, verdict)


def decide_all(name: str, text: str, others: List[str], *, work_title: str = "",
               author: str = "", recall: Optional[Recall] = None) -> Tuple[str, str, str]:
    """All four rungs. (gender, gender_source, evidence); gender "" = declined,
    and then the evidence names every rung's reason."""
    gender, tier, evidence = _decide(name, text, others)
    if gender:
        return gender, tier, evidence
    reasons = [evidence]
    if recall is not None and work_title:
        gender, evidence, source = recall.ask(name, work_title, author)
        if gender:
            return gender, source, evidence
        reasons.append(evidence)
    gender, evidence = name_frequency(name)
    if gender:
        return gender, "name_frequency", evidence
    reasons.append(evidence)
    return "", "", "; ".join(reasons)


# ---------------------------------------------------------------------------
# the prose bank
# ---------------------------------------------------------------------------

def _base_identity(source: Dict[str, Any], unit: Dict[str, Any],
                   text: str, text_path: pathlib.Path) -> Dict[str, Any]:
    """Base fields derivable from LOCAL truth only -- no network, no clock."""
    rel = text_path.relative_to(REPO_ROOT).as_posix() if _under(text_path, REPO_ROOT) \
        else text_path.as_posix()
    return {
        "schema_version": SIDECAR_SCHEMA,
        "slug": str(source.get("source_id") or ""),
        "unit": str(unit.get("unit_id") or ""),
        "work_title": str(source.get("title") or ""),
        "author": str(source.get("author") or ""),
        "license_label": str(source.get("license_status") or ""),
        "source_url": str(source.get("source_url") or ""),
        "body_sha256": _sha256_text(text),
        "body_bytes": len(text.encode("utf-8")),
        "body_words": len(text.split()),
        "text_path": rel,
    }


def _under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _row_for(name: str, gender: str, tier: str, evidence: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": name,
        "gender": gender,
        "gender_source": tier,
        # Redundant with gender_source by construction, stamped anyway so a
        # ledger reader never has to carry the mapping table in their head.
        "gender_confidence": CONFIDENCE.get(tier, ""),
        "evidence": evidence,
    }
    aliases = _aliases_for(name)
    if aliases:
        row["aliases"] = aliases
    return row


def _merge_rows(fresh: Dict[str, Optional[Dict[str, Any]]], existing_rows: List[Dict[str, Any]],
                *, body_changed: bool) -> Tuple[List[Dict[str, Any]], List[str]]:
    """The monotonic merge. `fresh` maps candidate name -> new row or None
    (declined), in candidate order. Returns (rows, notes)."""
    old_by_name = {}
    for r in existing_rows or []:
        if isinstance(r, dict) and r.get("name"):
            old_by_name.setdefault(_norm(r["name"]), dict(r))
    rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    visited = set()
    for name, new_row in fresh.items():
        visited.add(_norm(name))
        old = None if body_changed else old_by_name.get(_norm(name))
        if new_row is None:
            if old is not None:
                rows.append(old)
                notes.append("%s: declined this run; kept the %s row" % (name, old.get("gender_source")))
            continue
        if old is None:
            rows.append(new_row)
            continue
        if _rung(new_row["gender_source"]) <= _rung(old.get("gender_source", "")):
            rows.append(new_row)          # equal = refresh, higher = promote
        else:
            rows.append(old)              # lower never demotes
            notes.append("%s: kept the %s row over a %s answer" % (
                name, old.get("gender_source"), new_row["gender_source"]))
    # A committed row whose name is no longer a candidate (the manifest's
    # cast_hints shrank) is carried forward, never silently deleted: the join
    # can only benefit from it, and a deletion should be a visible edit.
    if not body_changed:
        for key, old in old_by_name.items():
            if key not in visited:
                rows.append(old)
                notes.append("%s: not a candidate this run; carried forward" % (old["name"],))
    return rows, notes


def _tier_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {t: 0 for t in TIER_ORDER}
    for r in rows:
        src = str(r.get("gender_source") or "")
        key = "roster" if src in _ROSTER_SOURCES else src
        if key in counts:
            counts[key] += 1
    return counts


def stamp_unit(source: Dict[str, Any], unit: Dict[str, Any], *, write: bool,
               bank_dir: Optional[pathlib.Path] = None,
               recall: Optional[Recall] = None, fresh: bool = False) -> Dict[str, Any]:
    """Decide every candidate for one prose unit and merge the result into its sidecar."""
    bank = pathlib.Path(bank_dir) if bank_dir is not None else BANK_DIR
    text_path = bank / str(unit.get("text_path") or "")
    result: Dict[str, Any] = {
        "source_id": str(source.get("source_id") or ""),
        "unit_id": str(unit.get("unit_id") or ""),
        "decided": [], "declined": [], "kept": [], "changed": False,
    }
    if not text_path.is_file():
        result["error"] = "missing text: %s" % (text_path,)
        return result
    text = text_path.read_text(encoding="utf-8", errors="replace")

    path = sidecar_path_for_text(text_path)
    existing: Dict[str, Any] = {}
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            result["error"] = "existing sidecar is not valid JSON: %s" % (path,)
            return result
        if not isinstance(existing, dict):
            result["error"] = "existing sidecar is not an object: %s" % (path,)
            return result

    body_sha = _sha256_text(text)
    # `fresh` = ignore every committed row and rebuild from the rungs (the index
    # still answers tier 3, so it is cheap). The way to purge a row a since-fixed
    # rule wrote, because the merge otherwise keeps a declined name's old row.
    body_changed = fresh or (bool(existing) and str(existing.get("body_sha256") or "") != body_sha)
    names = _candidates(source)
    fresh: Dict[str, Optional[Dict[str, Any]]] = {}
    for name in names:
        gender, tier, evidence = decide_all(
            name, text, names, work_title=str(source.get("title") or ""),
            author=str(source.get("author") or ""), recall=recall)
        if not gender:
            result["declined"].append({"name": name, "why": evidence})
            fresh[name] = None
            continue
        fresh[name] = _row_for(name, gender, tier, evidence)
        result["decided"].append({"name": name, "gender": gender, "tier": tier})
    rows, notes = _merge_rows(fresh, existing.get("characters") or [], body_changed=body_changed)
    result["kept"] = notes
    tier_counts = _tier_counts(rows)

    merged = dict(existing)
    # Fill base identity ONLY where the fetcher has not spoken -- except the
    # body hash, which this tool owns as its staleness anchor and refreshes.
    for key, value in _base_identity(source, unit, text, text_path).items():
        merged.setdefault(key, value)
    merged["body_sha256"] = body_sha

    previous_ladder = existing.get("gender_ladder")
    previous_rows = existing.get("characters")
    substantive_change = (previous_rows != rows) or (
        not isinstance(previous_ladder, dict)
        or previous_ladder.get("version") != LADDER_VERSION
        or previous_ladder.get("tier_counts") != tier_counts
        or str(existing.get("body_sha256") or "") != body_sha
    )
    merged["characters"] = rows
    ladder = {
        "version": LADDER_VERSION,
        "tier_counts": tier_counts,
        "candidates": len(names),
        "declined": [n for n, r in fresh.items() if r is None
                     and not any(_norm(x.get("name")) == _norm(n) for x in rows)],
    }
    # ran_utc is audit metadata, NOT part of the staleness identity. Preserved
    # verbatim on a no-op so re-running is genuinely idempotent.
    if isinstance(previous_ladder, dict) and previous_ladder.get("ran_utc"):
        ladder["ran_utc"] = previous_ladder["ran_utc"]
    if substantive_change or "ran_utc" not in ladder:
        ladder["ran_utc"] = (recall.clock() if recall is not None else
                             _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat())
    merged["gender_ladder"] = ladder

    result["changed"] = bool(substantive_change)
    result["sidecar"] = str(path)
    if write and substantive_change:
        _write_json(path, merged)
    return result


def _write_json(path: pathlib.Path, data: Dict[str, Any]) -> None:
    """LF line endings on every platform, the fetcher's exact formatting, so a
    no-op re-dump is byte-identical (Windows `write_text` would emit CRLF)."""
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------
# the Shakespeare bank: fill ONLY the unknown rows the fetcher left
# ---------------------------------------------------------------------------

def _supplement_entry(bucket: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """The curated supplement is keyed by the CAST HINT ("ANTIPHOLUS", "DROMIO");
    the sidecar row carries the speech prefix ("ANTIPHOLUS OF EPHESUS"). Match the
    way the render join does: exact, or the hint as the row's leading word(s)."""
    key = _norm(name)
    if key in bucket:
        return bucket[key]
    # longest hint first, so a more specific entry always beats a shorter one
    for hint, entry in sorted(bucket.items(), key=lambda kv: -len(kv[0])):
        if hint and key.startswith(hint + " "):
            return entry
    return None


def stamp_scene(scene: Dict[str, Any], *, write: bool,
                bank_dir: Optional[pathlib.Path] = None,
                recall: Optional[Recall] = None,
                supplement: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fill the `unknown` rows of one curated scene's sidecar. Known rows are
    never touched (ruling 1). Order of authority for an unknown row: the
    operator's curated supplement, the model's recall of the play, the
    first-name pool. (No pronoun scan: see the note in the loop.)"""
    bank = pathlib.Path(bank_dir) if bank_dir is not None else SHAKESPEARE_DIR
    text_path = bank / str(scene.get("text_path") or "")
    play_code = str(scene.get("play_code") or "")
    title = str(scene.get("play_title") or "")
    result: Dict[str, Any] = {
        "source_id": str(scene.get("source_ref") or text_path.stem),
        "unit_id": "act%s-scene%s" % (scene.get("act"), scene.get("scene")),
        "decided": [], "declined": [], "kept": [], "changed": False,
    }
    path = sidecar_path_for_text(text_path)
    if not text_path.is_file() or not path.is_file():
        result["error"] = "missing text or sidecar for %s" % (text_path,)
        return result
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError:
        result["error"] = "existing sidecar is not valid JSON: %s" % (path,)
        return result
    rows = data.get("characters")
    if not isinstance(rows, list):
        result["error"] = "sidecar has no characters list: %s" % (path,)
        return result
    text = text_path.read_text(encoding="utf-8", errors="replace")
    bucket = (supplement or {}).get(play_code) or {}
    names = [str(r.get("name") or "") for r in rows if isinstance(r, dict)]

    new_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            new_rows.append(row)
            continue
        name = str(row.get("name") or "")
        known = str(row.get("gender") or "").lower() in ("male", "female")
        fetcher_owned = known and str(row.get("gender_source") or "") not in (
            "pronouns", "llm_recall", "name_frequency", "supplement")
        if fetcher_owned or not name or _norm(name) in _EXCLUDED:
            new_rows.append(row)            # untouchable, byte for byte
            continue
        gender = tier = evidence = ""
        entry = _supplement_entry(bucket, name)
        if entry:
            gender, tier, evidence = entry["gender"], "supplement", entry["evidence"]
        else:
            # NO pronoun scan on a scene. Measured 2026-09-02 on the real Folger
            # text: the scan called LUCE (the kitchen maid) MALE, because a play's
            # mentions are speech prefixes and the window after "LUCE" is her OWN
            # line, whose pronouns point at the men she is shouting at. The scan
            # was built for prose narration and is not evidence here.
            reasons = ["no pronoun scan on scene text (speech prefixes)"]
            if recall is not None:
                gender, evidence, tier = recall.ask(name, title, SHAKESPEARE_AUTHOR)
                if not gender:
                    reasons.append(evidence)
            if not gender:
                gender, evidence = name_frequency(name)
                if gender:
                    tier = "name_frequency"
                else:
                    reasons.append(evidence)
                    evidence = "; ".join(reasons)
        if not gender:
            if known:
                new_rows.append(row)        # a stamper-filled row, kept
                result["kept"].append("%s: declined this run; kept the %s row" % (
                    name, row.get("gender_source")))
            else:
                new_rows.append(row)
                result["declined"].append({"name": name, "why": evidence})
            continue
        if known and _rung(tier) > _rung(str(row.get("gender_source") or "")):
            new_rows.append(row)            # lower never demotes
            result["kept"].append("%s: kept the %s row over a %s answer" % (
                name, row.get("gender_source"), tier))
            continue
        filled = dict(row)
        filled["gender"] = gender
        filled["gender_source"] = tier
        filled["gender_confidence"] = CONFIDENCE.get(tier, "")
        filled["evidence"] = evidence
        new_rows.append(filled)
        result["decided"].append({"name": name, "gender": gender, "tier": tier})

    tier_counts = _tier_counts(new_rows)
    previous_ladder = data.get("gender_ladder")
    ladder = {
        "version": LADDER_VERSION,
        "tier_counts": tier_counts,
        # The rows this tool is responsible for: everything the fetcher did not
        # gender itself. Stable across re-runs (a filled row still counts), so a
        # no-op second run is a no-op in the ladder block too.
        "candidates": sum(1 for r in rows if isinstance(r, dict) and (
            str(r.get("gender") or "").lower() not in ("male", "female")
            or str(r.get("gender_source") or "") in (
                "pronouns", "llm_recall", "name_frequency", "supplement"))),
        "declined": [d["name"] for d in result["declined"]],
    }
    if isinstance(previous_ladder, dict) and previous_ladder.get("ran_utc"):
        ladder["ran_utc"] = previous_ladder["ran_utc"]
    substantive_change = (new_rows != rows) or (
        not isinstance(previous_ladder, dict)
        or {k: v for k, v in previous_ladder.items() if k != "ran_utc"}
        != {k: v for k, v in ladder.items() if k != "ran_utc"})
    if substantive_change or "ran_utc" not in ladder:
        ladder["ran_utc"] = (recall.clock() if recall is not None else
                             _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat())
    merged = dict(data)
    merged["characters"] = new_rows
    merged["gender_ladder"] = ladder
    result["changed"] = bool(substantive_change)
    result["sidecar"] = str(path)
    if write and substantive_change:
        _write_json(path, merged)
    return result


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------

def _load_recall(model_id: str, index: GenderIndex):
    """Load the local model behind a constrained closure. Returns (Recall, teardown)."""
    from pydantic import BaseModel, Field

    from nodes import _otr_model_loader as LOADER
    from nodes._otr_constrained_generate import make_constrained_generate_fn

    class CharacterGenderOpinion(BaseModel):
        gender: str = Field(..., pattern="^(male|female|unsure)$")
        reason: str = Field(..., max_length=200)

    entry = LOADER.load_llm(model_id, optimization_profile="Standard")
    gen = make_constrained_generate_fn(entry, CharacterGenderOpinion)
    return Recall(gen, index, model_id=model_id), LOADER.unload_llm


def _run_bank(bank: str, args, recall_factory) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if bank == "public_domain":
        index = GenderIndex(BANK_DIR / INDEX_FILENAME)
        recall = recall_factory(index)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for source in manifest.get("sources") or []:
            if args.source and source.get("source_id") != args.source:
                continue
            for unit in source.get("units") or []:
                results.append(stamp_unit(source, unit, write=args.write, recall=recall,
                                          fresh=args.fresh))
    else:
        index = GenderIndex(SHAKESPEARE_DIR / INDEX_FILENAME)
        recall = recall_factory(index)
        manifest = json.loads(SHAKESPEARE_MANIFEST.read_text(encoding="utf-8"))
        supplement = load_gender_supplement(SHAKESPEARE_DIR)
        for scene in manifest.get("scenes") or []:
            if args.source and args.source not in (scene.get("source_ref"), scene.get("play_code")):
                continue
            results.append(stamp_scene(scene, write=args.write, recall=recall,
                                       supplement=supplement))
    if args.write and index.save():
        print("[gender-stamper] index written: %s" % (index.path,))
    if recall is not None:
        print("[gender-stamper] %s recall census: %s" % (bank, recall.census))
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="write sidecars and the index; without it this only reports")
    ap.add_argument("--bank", choices=("public_domain", "shakespeare", "all"), default="all")
    ap.add_argument("--source", default="",
                    help="limit to one source_id / source_ref / play_code")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="local model for tier 3 (loaded once; needs the GPU free)")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip tier 3 entirely (offline census; cached index entries still count)")
    ap.add_argument("--fresh", action="store_true",
                    help="prose bank: rebuild every row from the rungs, ignoring committed rows "
                         "(purges rows an older rule wrote; the index still answers tier 3)")
    args = ap.parse_args(argv)

    banks = ("public_domain", "shakespeare") if args.bank == "all" else (args.bank,)
    teardown = None
    loaded: Dict[str, Any] = {}

    def recall_factory(index: GenderIndex) -> Optional[Recall]:
        nonlocal teardown
        if args.no_llm:
            return Recall(None, index, model_id=args.model)
        if "gen" not in loaded:
            print("[gender-stamper] loading %s for tier 3 ..." % args.model, flush=True)
            recall, teardown = _load_recall(args.model, index)
            loaded["gen"] = recall.generate_fn
            return recall
        return Recall(loaded["gen"], index, model_id=args.model)

    results: List[Dict[str, Any]] = []
    try:
        for bank in banks:
            results.extend(_run_bank(bank, args, recall_factory))
    finally:
        if teardown is not None:
            teardown()

    errors = [r for r in results if r.get("error")]
    decided = sum(len(r["decided"]) for r in results)
    declined = sum(len(r["declined"]) for r in results)
    changed = [r for r in results if r.get("changed")]
    total = decided + declined

    print("[gender-stamper] units=%d candidates=%d decided=%d declined=%d "
          "coverage=%.1f%%" % (
              len(results), total, decided, declined,
              100.0 * decided / max(1, total)))
    print("[gender-stamper] sidecars %s: %d" % (
        "written" if args.write else "that WOULD change", len(changed)))
    for r in results:
        for d in r["decided"]:
            print("  DECIDED  %s:%s -> %s (%s)" % (r["source_id"], d["name"], d["gender"], d["tier"]))
        for d in r["declined"]:
            print("  DECLINED %s:%s -- %s" % (r["source_id"], d["name"], d["why"][:160]))
        for k in r.get("kept") or []:
            print("  KEPT     %s:%s" % (r["source_id"], k))
    for r in errors:
        print("  ERROR %s: %s" % (r["source_id"], r["error"]))
    # Fail loud in an authoring-time tool: a unit whose text is missing means
    # the corpus is incomplete, and a silent skip there is how 64 empty
    # sidecars went unnoticed for a month in the first place.
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
