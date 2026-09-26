# -*- coding: utf-8 -*-
r"""HF_HOME must never be pinned somewhere the weights cannot be written.

THE BUG THIS LOCKS DOWN, measured across two machines 2026-09-23.

``prestartup_script.py`` pinned ``HF_HOME`` to ``<install>/models/huggingface``
so the cache would sit beside ComfyUI's models. On a ComfyUI Desktop install
that root is ``89 + len(username)`` characters. The longest cache tail this pack
asks for is 162::

    hub\models--Comfy-Org--Lumina_Image_2.0_Repackaged\snapshots\<40-char sha>
        \split_files\diffusion_models\lumina_2_model_bf16.safetensors

Windows' usable path budget is 259, so the root may occupy at most 96 -- and
``89 + 7 == 96`` means **the longest safe Windows username was seven
characters**. An eighth broke Lumina's auto-download on a stock install.

IT FAILED UNRECOGNISABLY, which is why it needed two boxes to find.
``huggingface_hub`` 1.30.0 adds the ``\\?\`` extended-length prefix to
``lock_path`` and ``blob_path`` but not to ``pointer_path``. So the full 5.22 GB
blob landed at 220 characters, and then materialising the 261-character pointer
went ``_create_symlink`` -> ``shutil.move`` -> ``os.rename`` -> WinError 3 ->
``FileNotFoundError``. Windows never says "too long" for this. Empirically on the
affected box: 259 succeeded, 260 failed, and the same 265-character path
succeeded WITH the prefix. Its cache held 86 materialised files, the longest at
238, and the single missing pointer was the 261-character one -- length was the
only variable separating them.

WHY THE FIX IS OURS AND NOT THE MODEL'S. Swapping Lumina for something
shorter-named would have treated the symptom and left the seven-character cliff
in place for the next long-named repo. We chose the deep root. We also already
had a short default -- ``nodes/_otr_hf_env.py``'s ``C:\ComfyUI-Models\huggingface``
-- but ``resolve()`` reads ``os.environ`` first, and prestartup had already
populated it, so our own safe default was unreachable on every live boot.

WHAT THESE TESTS ARE FOR. The 162 is a CONSTANT in a file that must not import
the catalogue (a prestartup that raises takes the boot with it), so nothing stops
a new model from quietly outgrowing it. The first test re-derives it from the
real ``_SOURCES`` tuples and fails when it drifts -- which is the only way that
number stays true.
"""
import importlib.util
import pathlib
import re
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent

#: Windows MAX_PATH is 260 counting the terminating NUL, so 259 is usable.
_BUDGET = 259
#: A hub pointer path is ``hub\models--<org>--<repo>\snapshots\<sha>\<file>``.
_SHA_LEN = 40


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _visual_assets():
    return _load("_otr_visual_assets_maxpath", "nodes/_otr_visual_assets.py")


def _tail_for(repo_id, filename):
    """The characters the HF cache adds after the root, for one spec."""
    org, name = repo_id.split("/", 1)
    return len("hub\\models--%s--%s\\snapshots\\%s\\%s"
               % (org, name, "0" * _SHA_LEN, filename.replace("/", "\\")))


def _prestartup_constant(name):
    """Read a constant out of prestartup_script.py WITHOUT importing it.

    Importing it would install its transformers mock and pin env vars in the
    test process. The values are plain integer literals, so the source is the
    honest place to read them from.
    """
    text = (_ROOT / "prestartup_script.py").read_text(encoding="utf-8")
    match = re.search(r"^%s\s*=\s*(\d+)\s*$" % re.escape(name), text, re.M)
    assert match, "%s is not a plain int literal in prestartup_script.py" % name
    return int(match.group(1))


def test_the_declared_longest_tail_still_covers_every_real_spec():
    """The constant in prestartup must cover the actual catalogue.

    THIS IS THE TEST THAT MATTERS. prestartup cannot import the catalogue, so
    the 162 is copied by hand -- and a new model with a longer org, repo or
    subfolder would silently shrink the real headroom while the guard kept
    quoting the old number and waving installs through.
    """
    sources = _visual_assets()._SOURCES
    measured = [(_tail_for(repo, filename), repo, filename)
                for _category, repo, filename in sources]
    assert measured, "_SOURCES is empty; this test would prove nothing"
    worst, repo, filename = max(measured)
    declared = _prestartup_constant("_OTR_LONGEST_HF_TAIL")
    assert declared >= worst, (
        "prestartup_script.py declares _OTR_LONGEST_HF_TAIL = %d but the longest "
        "real tail is %d, from %s/%s. Raise the constant to %d (and re-read the "
        "comment above it -- the root budget it implies is %d characters, and a "
        "ComfyUI Desktop root is 89 + len(username))."
        % (declared, worst, repo, filename, worst, _BUDGET - worst - 1))


def _source_int(relative, name):
    text = (_ROOT / relative).read_text(encoding="utf-8")
    match = re.search(r"^%s\s*=\s*(\d+)\s*$" % re.escape(name), text, re.M)
    assert match, "%s is not a plain int literal in %s" % (name, relative)
    return int(match.group(1))


def test_the_env_resolver_uses_the_same_room_literals():
    """ensure_hf_home refuses with the same 259 and 162 prestartup uses."""
    assert _source_int("nodes/_otr_hf_env.py", "_HF_WINDOWS_PATH_BUDGET") == (
        _prestartup_constant("_OTR_WINDOWS_PATH_BUDGET"))
    assert _source_int("nodes/_otr_hf_env.py", "_HF_LONGEST_TAIL") == (
        _prestartup_constant("_OTR_LONGEST_HF_TAIL"))


def test_the_budget_constant_is_259_in_both_places_that_use_it():
    """259, not 260. The off-by-one is the difference between pass and fail.

    An earlier reading of this called 260 the edge and concluded a particular
    username was safe; it was not. MAX_PATH counts the terminating NUL, so a
    path may occupy 259.
    """
    assert _prestartup_constant("_OTR_WINDOWS_PATH_BUDGET") == _BUDGET
    assert _visual_assets()._WINDOWS_PATH_BUDGET == _BUDGET


def test_a_comfy_desktop_install_is_what_made_this_bite():
    """The regression case, stated as arithmetic rather than as a story.

    Seven characters passed and eight failed, on a stock Desktop install, for
    the pack's longest model. If this stops holding the guard's comment is
    wrong and should be rewritten from whatever the new numbers are.
    """
    worst = _prestartup_constant("_OTR_LONGEST_HF_TAIL")
    room = _BUDGET - worst - 1

    def desktop_root(username):
        return (r"C:\Users\%s\AppData\Local\Comfy-Desktop\ComfyUI-Installs"
                r"\ComfyUI\ComfyUI\models\huggingface" % username)

    assert len(desktop_root("a" * 7)) <= room, (
        "a 7-character username should fit: root=%d, room=%d"
        % (len(desktop_root("a" * 7)), room))
    assert len(desktop_root("a" * 8)) > room, (
        "an 8-character username should NOT fit: root=%d, room=%d"
        % (len(desktop_root("a" * 8)), room))


def test_the_short_models_root_is_29_characters():
    """29, not 27. The off-by-two is a miscount of ``huggingface``."""
    assert len(r"C:\ComfyUI-Models\huggingface") == 29


def test_the_short_roots_all_fit_so_the_fallback_is_worth_taking():
    """A fallback root is only a fix if it fits in the same room."""
    worst = _prestartup_constant("_OTR_LONGEST_HF_TAIL")
    room = _BUDGET - worst - 1
    for root in (r"C:\Users\christopher\.cache\huggingface",   # HF's own default
                 r"C:\ComfyUI-Models\huggingface",            # _otr_hf_env's
                 r"C:\ComfyUI_windows_portable\ComfyUI\models\huggingface"):
        assert len(root) <= room, "%s is %d chars, room is %d" % (
            root, len(root), room)


class TestTheErrorNowNamesItsCause:
    """``_hf_fetch`` reported only the exception TYPE, which hid this bug.

    A bare "visual weight transfer failed (FileNotFoundError)" carries no path
    and no length, so the one fact that identifies the failure was absent from
    the only message a user or a reviewer ever saw.
    """

    def test_it_reports_the_real_length_not_the_escaped_one(self):
        """An OSError's str() doubles backslashes; the count must not.

        Measured while writing this: the first cut reported 282 characters for a
        264-character path, inflated by one per separator. A reader deciding how
        much to shorten by would have been eighteen out.
        """
        scrub = _visual_assets()._scrub_transfer_error
        path = (r"C:\Users\christopher\AppData\Local\Comfy-Desktop"
                r"\ComfyUI-Installs\ComfyUI\ComfyUI\models\huggingface\hub"
                r"\models--Comfy-Org--Lumina_Image_2.0_Repackaged\snapshots"
                "\\" + "0" * _SHA_LEN +
                r"\split_files\diffusion_models\lumina_2_model_bf16.safetensors")
        out = scrub(FileNotFoundError(
            2, "The system cannot find the path specified", path))
        assert "%d chars" % len(path) in out, out
        assert "OVER the %d-char" % _BUDGET in out, out

    def test_a_path_within_budget_does_not_cry_wolf(self):
        scrub = _visual_assets()._scrub_transfer_error
        out = scrub(FileNotFoundError(
            2, "no such file", r"C:\ComfyUI-Models\huggingface\hub\f.safetensors"))
        assert "OVER the" not in out, out

    #: Every shape an adversarial review got a credential through in. The first
    #: redaction matched only ``scheme://<non-whitespace>`` and FIVE OF THESE SIX
    #: leaked -- each one a shape a real Hugging Face error can produce. They are
    #: parametrised rather than written as prose because the lesson is that the
    #: SHAPE cannot be anticipated; the KEY can, which is why the redaction is
    #: key-based now.
    LEAK_CASES = (
        ("no scheme, bare host and query",
         "cdn-lfs.hf.co/repos/ab/m.safetensors?X-Amz-Signature=%s failed"),
        ("an Authorization header echoed into the message",
         "request failed: Authorization: Bearer %s"),
        ("a percent-encoded url",
         "GET https%%3A%%2F%%2Fcdn-lfs.hf.co%%2Ff%%3FX-Amz-Signature%%3D%s"),
        ("a url split across a newline",
         "https://cdn-lfs.hf.co/file?\nX-Amz-Signature=%s"),
        ("a token in no url at all",
         "hf_token=%s was rejected"),
        ("a plain scheme url, which the first version did catch",
         "GET https://cdn-lfs.hf.co/f?X-Amz-Signature=%s"),
    )

    @pytest.mark.parametrize("label,template", LEAK_CASES,
                             ids=[c[0] for c in LEAK_CASES])
    def test_a_credential_never_survives(self, label, template):
        """The property the caller's ``from None`` was protecting.

        Keeping the message is only acceptable because credentials are removed.
        If one of these fails the fix is a better redaction -- never going back to
        discarding the diagnosis, which is what hid the MAX_PATH bug for a day.
        """
        secret = "TEST_SECRET_DO_NOT_LEAK"
        out = _visual_assets()._scrub_transfer_error(OSError(template % secret))
        assert secret not in out, "%s leaked: %r" % (label, out)
        assert "redacted" in out, (
            "%s: the redaction must be VISIBLE so a reader knows something was "
            "removed rather than never present: %r" % (label, out))

    def test_an_exception_that_cannot_be_stringified_still_returns(self):
        """The formatter runs inside an exception handler and may never raise.

        An earlier fallback called ``exc.__class__.__name__`` unguarded, which an
        exception overriding attribute access defeats -- turning a useful error
        into a confusing one at the worst moment.
        """
        scrub = _visual_assets()._scrub_transfer_error

        class Hostile(Exception):
            def __str__(self):
                raise RuntimeError("no str for you")

        assert scrub(Hostile()) == "Hostile"

    def test_a_formatter_that_raises_degrades_to_the_type_name(self):
        """It runs inside an exception handler; it may never add a failure."""
        scrub = _visual_assets()._scrub_transfer_error

        class Unprintable(Exception):
            def __str__(self):
                raise RuntimeError("cannot render")

        assert scrub(Unprintable()) == "Unprintable"


def test_the_pin_is_one_assignment_after_the_choice():
    """One call to ``_choose_hf_home``, then one assignment of its result.

    THIS IS A SOURCE-INSPECTION TEST ON PURPOSE. The length comparison lives
    inside the chooser, which the behavioural tests call directly. What source
    inspection is for is proving prestartup has a single assignment site and
    does not import ``nodes/`` to make the choice.
    """
    text = (_ROOT / "prestartup_script.py").read_text(encoding="utf-8")
    assignments = re.findall(r'^\s*environ\["HF_HOME"\]\s*=', text, re.M)
    assert len(assignments) == 1, (
        "expected exactly one HF_HOME assignment in prestartup, found %d -- a "
        "second one may bypass the chooser" % len(assignments))

    room = text.index(
        "_otr_room = _OTR_WINDOWS_PATH_BUDGET - _OTR_LONGEST_HF_TAIL - 1")
    call = text.index("_otr_pin._choose_hf_home(_otr_adjacent, _otr_room)")
    assignment = text.index('environ["HF_HOME"] = _otr_hf_home')
    assert room < call < assignment, (
        "room, then the chooser, then the one assignment; got room=%d call=%d "
        "assignment=%d" % (room, call, assignment))
    between = text[call:assignment]
    assert "if _otr_hf_home:" in between, (
        "an empty choice must not be assigned; text was:\n%s" % between)

    assert "NOT pinning" not in text
    assert "set HF_HOME yourself" not in text
    assert "enable Windows long paths" not in text

    pin = (_ROOT / "otr_hf_home_pin.py").read_text(encoding="utf-8")
    assert "import nodes" not in pin
    assert "_otr_hf_env" not in pin
    assert "_choose_hf_home" in pin


def _room():
    return (_prestartup_constant("_OTR_WINDOWS_PATH_BUDGET")
            - _prestartup_constant("_OTR_LONGEST_HF_TAIL") - 1)


def _desktop_root(username):
    return (r"C:\Users\%s\AppData\Local\Comfy-Desktop\ComfyUI-Installs"
            r"\ComfyUI\ComfyUI\models\huggingface" % username)


def _decide(log, **overrides):
    """Call the chooser with every live dependency replaced."""
    pin = _load("otr_hf_home_pin_choose", "otr_hf_home_pin.py")
    args = {
        "platform": "win32",
        "registry_value": None,
        "hub_cache": None,
        "legacy_hub_cache": None,
        "long_paths_enabled": False,
        "models_root": r"C:\ComfyUI-Models\huggingface",
        "models_root_exists": False,
        "user_cache": r"C:\Users\chris\.cache\huggingface",
        "write_probe": lambda _path: True,
        "log": log.append,
    }
    args.update(overrides)
    adjacent = args.pop("models_adjacent", r"C:\short\models\huggingface")
    return pin._choose_hf_home(adjacent, _room(), **args)


def test_a_short_registry_value_wins():
    log = []
    chosen = _decide(
        log,
        registry_value=r"C:\ComfyUI-Models\huggingface",
        hub_cache=r"C:\somewhere\else\huggingface",
        models_adjacent=_desktop_root("a" * 8),
        models_root_exists=True,
        long_paths_enabled=True,
    )
    assert chosen == r"C:\ComfyUI-Models\huggingface"
    assert log == []


def test_a_too_long_registry_value_is_not_kept():
    log = []
    registry = "R:\\" + ("r" * 120)
    chosen = _decide(
        log,
        registry_value=registry,
        models_adjacent=r"C:\short\models\huggingface",
    )
    assert chosen == r"C:\short\models\huggingface"
    assert registry not in (chosen or "")
    assert any(str(len(registry)) in line and registry in line for line in log)


def test_a_short_hub_cache_wins_and_a_long_one_does_not():
    log = []
    short = r"C:\ComfyUI-Models\huggingface"
    chosen = _decide(log, hub_cache=short,
                     models_adjacent=_desktop_root("a" * 8))
    assert chosen == short

    log.clear()
    long_cache = "H:\\" + ("h" * 120)
    legacy = r"C:\legacy\huggingface"
    chosen = _decide(
        log,
        hub_cache=long_cache,
        legacy_hub_cache=legacy,
        models_adjacent=_desktop_root("a" * 8),
        models_root_exists=True,
    )
    assert chosen == legacy
    assert any(str(len(long_cache)) in line for line in log)


def test_too_long_adjacent_uses_comfyui_models_when_present():
    log = []
    probed = []
    adjacent = _desktop_root("a" * 8)
    assert len(adjacent) > _room()

    def probe(path):
        probed.append(path)
        return True

    chosen = _decide(
        log,
        models_adjacent=adjacent,
        models_root_exists=True,
        long_paths_enabled=False,
        write_probe=probe,
    )
    assert chosen == r"C:\ComfyUI-Models\huggingface"
    assert probed == [r"C:\ComfyUI-Models\huggingface"]
    assert any(
        adjacent in line and str(len(adjacent)) in line for line in log
    ), log


def test_neither_adjacent_nor_models_root_uses_the_user_cache():
    log = []
    user = r"C:\Users\christopher\.cache\huggingface"
    adjacent = _desktop_root("a" * 8)
    chosen = _decide(
        log,
        models_adjacent=adjacent,
        models_root_exists=False,
        long_paths_enabled=False,
        user_cache=user,
    )
    assert chosen == user
    assert any("does not exist" in line and "(29 characters)" in line
               for line in log), log


def test_nothing_that_fits_leaves_hf_home_unset():
    log = []
    adjacent = _desktop_root("a" * 8)
    user = "C:\\Users\\" + ("n" * 90) + "\\.cache\\huggingface"
    assert len(user) > _room()
    chosen = _decide(
        log,
        models_adjacent=adjacent,
        models_root_exists=False,
        long_paths_enabled=False,
        user_cache=user,
        write_probe=lambda _path: (_ for _ in ()).throw(AssertionError("probe")),
    )
    assert chosen is None
    assert any(str(len(adjacent)) in line for line in log), log
    assert any(str(len(user)) in line for line in log), log
    assert any("leaving HF_HOME unset" in line for line in log), log


def test_long_paths_keep_a_too_long_adjacent_root():
    """LongPathsEnabled=1 still pins the models-adjacent root."""
    adjacent = _desktop_root("a" * 8)
    chosen = _decide(
        [],
        models_adjacent=adjacent,
        long_paths_enabled=True,
        models_root_exists=True,
    )
    assert chosen == adjacent


def test_off_windows_the_adjacent_root_wins_even_when_long():
    adjacent = "/home/" + ("u" * 80) + "/ComfyUI/models/huggingface"
    chosen = _decide(
        [],
        platform="linux",
        registry_value=r"C:\ComfyUI-Models\huggingface",
        models_adjacent=adjacent,
        long_paths_enabled=False,
    )
    assert chosen == adjacent


def test_prestartup_stays_ascii_only():
    """Its own docstring says this is load-bearing, and it is.

    A non-ASCII character here once raised UnicodeEncodeError on a cp1252
    console, so every boot logged PRESTARTUP FAILED while the pack actually
    worked -- and anything below the failing print would silently never run.
    The MAX_PATH guard added today sits below it.
    """
    text = (_ROOT / "prestartup_script.py").read_text(encoding="utf-8")
    offenders = [(i + 1, line) for i, line in enumerate(text.splitlines())
                 if any(ord(ch) > 127 for ch in line)]
    assert not offenders, "non-ASCII in prestartup_script.py: %r" % offenders[:3]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
