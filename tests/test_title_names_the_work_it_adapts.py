"""PBUG-20260815-05 -- the episode title named a DIFFERENT play.

A Macbeth scene shipped as `signal_lost_tempests_midnight_revelations_...`.
Scene selection was CORRECT; `meta.title_source == "llm_post_composition"`.
`_generate_title_from_script` received dialogue excerpts, `outline.premise`,
an empty `arc_verdict` and a GENERIC bank label -- nothing in its context ever
named the work. The model free-associated "Tempest" off the scene's genuine
storm sound-world, and "The Tempest" is a SIBLING PLAY in the same curated
manifest (`config/source_banks/shakespeare/curated_scenes.sample.json`).

THE FIX IS AN ANCHOR, NOT A GUARD. The operator ruled on 2026-08-19: *"dont
waste too much time overengineering for hard to replicate bugs im accepting
some level of story quirks since a new story is gen every time"*. So the
code-side "reject a title containing a sibling work title" check specified in
`PROD_BUG_LOG.md` was deliberately NOT built -- substring containment is a
false-positive generator and no sound matching rule was available. What is
built is the root fix: the pass is no longer blind.

TWO PROPERTIES THIS FILE EXISTS TO PIN, because each is a way the fix silently
becomes worthless:

1. **The anchor is purely ADDITIVE.** With no work title -- the `original`
   lane, every legacy caller, every self-test -- the rendered prompt must be
   BYTE-IDENTICAL to the pre-fix prompt. A fix that perturbs the prompt on
   lanes it was never about is a story-quality change nobody asked for.
2. **The anchor must not become title MATERIAL.** Told only "this is Macbeth",
   a model answers "The Macbeth Prophecy" on every adaptation episode --
   trading a rare fidelity defect for a constant blandness one. The prompt
   carries the name AND the rule that keeps it out of the title.

And the invariant that governs the CALL SITE, which is where this class of fix
has already died once in this very method: the identity read must be
method-local, inside a `try`, and LANE-GATED. See
`test_call_site_*` below.
"""

from __future__ import annotations

import ast
from pathlib import Path

from nodes.OTR_LedgerScriptWriter import _generate_title_from_script
from nodes import _otr_source_identity as SID


REPO_ROOT = Path(__file__).resolve().parent.parent
WRITER_PATH = REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"

_SCRIPT = "\n".join(
    f"[VOICE: MACBETH, uneasy] Line {i} spoken on the blasted heath."
    for i in range(24)
)

_REPLY = (
    "DETAILS:\nthe heath\nthe drum\nthe torn banner\n"
    "CANDIDATES:\nThe Torn Banner\nDrums on the Heath\nWhat the Watchers Said\n"
    "TITLE: Drums on the Heath"
)


def _capture():
    """Return (generate_fn, calls) where calls collects the message lists."""
    calls: list = []

    def fn(messages, **kwargs):
        calls.append(messages)
        return _REPLY

    return fn, calls


def _user_msg(calls) -> str:
    return calls[-1][1]["content"]


def _sys_msg(calls) -> str:
    return calls[-1][0]["content"]


# --------------------------------------------------------------------------
# 1. The anchor is purely additive
# --------------------------------------------------------------------------

def test_no_work_title_renders_the_pre_fix_prompt_byte_for_byte():
    """The `original` lane adapts nothing and must be untouched by this fix.

    Built by removing the two anchor fragments from the ANCHORED prompt and
    asserting the result equals the DEFAULT prompt exactly -- which proves the
    anchor is additive rather than a rewrite that happens to look similar.
    """
    fn, calls = _capture()
    _generate_title_from_script(fn, _SCRIPT, premise="A soldier is greeted.")
    default_user = _user_msg(calls)
    default_sys = _sys_msg(calls)

    _generate_title_from_script(
        fn, _SCRIPT, premise="A soldier is greeted.", work_title="Macbeth",
    )
    anchored_user = _user_msg(calls)

    stripped = anchored_user.replace(
        "THIS EPISODE ADAPTS: Macbeth\n\n", "", 1,
    ).replace(
        " - this episode adapts Macbeth; keep that name OUT of the title, "
        "and never name a different work\n", "", 1,
    )
    assert stripped == default_user
    # The system message frames the FORM, never the work -- unchanged.
    assert _sys_msg(calls) == default_sys


def test_blank_and_whitespace_work_titles_are_the_same_as_none():
    fn, calls = _capture()
    _generate_title_from_script(fn, _SCRIPT)
    baseline = _user_msg(calls)

    for blank in ("", "   ", "\n\t "):
        _generate_title_from_script(fn, _SCRIPT, work_title=blank)
        assert _user_msg(calls) == baseline, f"blank {blank!r} perturbed prompt"


# --------------------------------------------------------------------------
# 2. The anchor names the work, and keeps it out of the title
# --------------------------------------------------------------------------

def test_anchor_names_the_work_being_adapted():
    fn, calls = _capture()
    _generate_title_from_script(fn, _SCRIPT, work_title="Macbeth")
    user = _user_msg(calls)
    assert "THIS EPISODE ADAPTS: Macbeth" in user


def test_anchor_forbids_putting_the_work_name_in_the_title():
    """Without this rule the fix trades a rare defect for a constant one."""
    fn, calls = _capture()
    _generate_title_from_script(fn, _SCRIPT, work_title="Macbeth")
    user = _user_msg(calls)
    assert "keep that name OUT of the title" in user
    assert "never name a different work" in user


def test_no_sibling_work_title_ever_enters_the_prompt():
    """The craft rule from this PBUG's own log entry: the feared failure may
    never appear in the model's context. Only the work being adapted is named.
    """
    fn, calls = _capture()
    _generate_title_from_script(fn, _SCRIPT, work_title="Macbeth")
    user = _user_msg(calls)
    for sibling in ("Tempest", "King Lear", "Twelfth Night", "Hamlet"):
        assert sibling not in user, f"sibling {sibling!r} leaked into prompt"


def test_anchor_does_not_disturb_title_parsing():
    fn, _calls = _capture()
    assert _generate_title_from_script(
        fn, _SCRIPT, work_title="Macbeth",
    ) == "Drums on the Heath"


# --------------------------------------------------------------------------
# 3. The lane gate -- `work_title` means two different things
# --------------------------------------------------------------------------

def test_shakespeare_meta_yields_the_play_title_as_the_anchor():
    meta = {
        "source_bank": "shakespeare",
        "source_meta": {
            "play_title": "Macbeth",
            "play_code": "Mac",
            "act": 1,
            "scene": 3,
        },
    }
    identity = SID.identity_from_meta(meta)
    assert identity.source_kind in SID.ADAPTATION_SOURCE_KINDS
    assert identity.work_title == "Macbeth"


def test_media_archive_publication_is_gated_out_of_the_anchor():
    """`work_title` holds the PUBLICATION on media_archive -- 56 of 98 live
    ledgers carry a `source_label`. Anchoring a feed post's title pass to
    "Now See Hear!" would invent a work, which is worse than the wrong-play
    title this fix removes. The gate is on the LANE, never on truthiness.
    """
    meta = {
        "source_bank": "media_archive",
        "source_meta": {
            "source_label": "Now See Hear!",
            "post_headline": "A reel returns to the archive",
        },
    }
    identity = SID.identity_from_meta(meta)
    assert identity.work_title == "Now See Hear!"       # truthy...
    assert identity.source_kind not in SID.ADAPTATION_SOURCE_KINDS  # ...gated


def test_unknown_lane_yields_no_anchor():
    identity = SID.identity_from_meta({"source_bank": "original",
                                       "source_meta": {}})
    assert identity.source_kind not in SID.ADAPTATION_SOURCE_KINDS
    assert identity.work_title == ""


# --------------------------------------------------------------------------
# 4. The call site -- where this class of fix already died once
# --------------------------------------------------------------------------

def _tail_method() -> ast.FunctionDef:
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_writer_tail":
            return node
    raise AssertionError("_run_writer_tail not found")


def _title_regen_call() -> ast.Call:
    for node in ast.walk(_tail_method()):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_generate_title_from_script"
        ):
            return node
    raise AssertionError("title regen call not found in _run_writer_tail")


def _title_anchor_try() -> ast.Try:
    """The `try` that resolves the TITLE anchor, found by the name it binds.

    SCOPING IS THE WHOLE POINT OF THIS HELPER, and the first cut of this file
    did not have it. `_run_writer_tail` contains a SECOND, PRE-EXISTING
    identity read -- the announcer work-frame splice -- which is itself
    imported method-locally, guarded by a `try`, and gated on
    ADAPTATION_SOURCE_KINDS. Measured on the real file: 4 matching imports and
    2 qualifying `try` blocks inside this one method.

    So an AST walk over the whole method passes on the OLD block alone, even
    if this fix were deleted outright -- a test that pins nothing while
    reading as if it pins everything. Caught by the codex QA lane on the
    finished diff, verified against the file, and fixed here.
    """
    for node in ast.walk(_tail_method()):
        if isinstance(node, ast.Try) and "_title_identity" in ast.dump(node):
            return node
    raise AssertionError(
        "no try block binding `_title_identity` in _run_writer_tail -- the "
        "title work-anchor read is missing or was renamed"
    )


def test_call_site_actually_passes_the_anchor():
    """The fix is worthless unwired. This is the whole point of the change."""
    call = _title_regen_call()
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert "work_title" in kwargs
    # ...and it must carry the RESOLVED anchor, not a literal or a stray name.
    passed = kwargs["work_title"]
    assert isinstance(passed, ast.Name) and passed.id == "_title_work", (
        "work_title= must be passed the resolved `_title_work` local"
    )


def test_call_site_identity_import_is_method_local():
    """`_run_writer_tail` is a separate METHOD, not a closure over `run()`.

    A module-level name bound in `run()` raises NameError here on EVERY
    episode, and the enclosing `except Exception` swallows it -- which is
    exactly how a previous fix in this same method shipped as dead code with a
    green suite behind it.

    Scoped to the anchor block: see `_title_anchor_try`.
    """
    imports = [
        n for n in ast.walk(_title_anchor_try())
        if isinstance(n, (ast.Import, ast.ImportFrom))
        and "_otr_source_identity" in ast.dump(n)
    ]
    assert imports, (
        "the title work-anchor block must import _otr_source_identity itself, "
        "method-locally -- it cannot borrow a name bound in run()"
    )


def test_call_site_identity_read_is_guarded_and_lane_gated():
    """The anchor read must be inside its `try` AND gated on
    ADAPTATION_SOURCE_KINDS. Nothing about naming the work may fail an
    episode, and an ungated read re-opens the media_archive collision.

    Scoped to the anchor block: see `_title_anchor_try`.
    """
    body = ast.dump(_title_anchor_try())
    assert "identity_from_meta" in body, (
        "the anchor block must resolve identity inside its own try"
    )
    assert "ADAPTATION_SOURCE_KINDS" in body, (
        "the anchor block must apply the lane gate, never truthiness"
    )


def test_anchor_block_scoping_helper_is_not_matching_the_old_work_frame_block():
    """Guard the guard.

    If `_title_anchor_try` ever starts resolving to the pre-existing
    work-frame block, the three tests above silently stop testing this fix.
    The anchor block is identified by binding `_title_work`; the work-frame
    block binds `_tail_work_title` and must NOT be the one found.
    """
    body = ast.dump(_title_anchor_try())
    assert "_title_work" in body
    assert "_tail_work_title" not in body, (
        "_title_anchor_try matched the announcer work-frame block, not the "
        "title anchor block -- the call-site tests are no longer scoped"
    )


def test_successful_anchor_read_stamps_a_ledger_receipt():
    """A published episode must be able to prove which way this went.

    Stamped ONLY on a successful read (an ABSENT key means the read raised),
    which is the same convention `meta["bank_roll"]` uses in this file.
    """
    body = ast.dump(_title_anchor_try())
    assert "title_work_anchor" in body, (
        "the anchor block must stamp meta['title_work_anchor'] so a frozen "
        "ledger records the anchor without a re-run"
    )
