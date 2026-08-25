"""Bark must generate on the device its model is actually loaded on.

2026-08-25. ``_generate_single_line`` hardcoded the literal ``"cuda"`` in three
places -- the ``_move_to_device`` call, the pre-generate assert, and the
``torch.tensor`` / ``torch.arange`` monkeypatches -- while ``_load_bark`` has
always been device-aware (``cuda if torch.cuda.is_available() else cpu``). So on
a Mac, an Intel box, or any CUDA-less install the model loaded happily on CPU
and the FIRST spoken line blew up.

Nothing gated it, either. The CAPABILITIES row declares bark
``device_backends: ["cuda"]``, but that table feeds ``capability_profiles`` to
derive per-profile enable-sets and is NOT consulted at voice dispatch --
``registry.assert_usable`` only checks "registered + role-compatible". bark is
the zero-setup voice engine a fresh install reaches for (it auto-downloads
suno/bark and its voices are presets baked into the weights), so this was
reachable by exactly the user least equipped to diagnose it, and it fired only
after the story, the casting and every still had already been paid for.

These tests run on CPU. The suite's conftest sets ``CUDA_VISIBLE_DEVICES=''``,
so ``.to("cuda")`` raises -- which means the old code cannot pass them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from nodes import _otr_bark_lib as bl  # noqa: E402


class _FakeGenerationConfig:
    sample_rate = 24000


class _FakeBarkModel:
    """A bark-shaped model whose parameters live on CPU.

    ``parameters()`` is the whole point: the fix asks the MODEL where it is
    rather than assuming, and this is what it asks.
    """

    generation_config = _FakeGenerationConfig()

    def __init__(self, device="cpu", raise_on_generate=False):
        self._p = torch.zeros(1, device=device)
        self.seen_input_devices = []
        self.seen_history_prompt_devices = []
        self.generate_calls = 0
        self._raise_on_generate = raise_on_generate

    def parameters(self):
        yield self._p

    def generate(self, **kwargs):
        self.generate_calls += 1
        ids = kwargs.get("input_ids")
        if ids is not None:
            self.seen_input_devices.append(ids.device)
        hp = kwargs.get("history_prompt")
        if isinstance(hp, dict):
            self.seen_history_prompt_devices.extend(
                v.device for v in hp.values() if torch.is_tensor(v)
            )
        if self._raise_on_generate:
            raise RuntimeError("simulated generate() failure")
        # bark returns a waveform tensor; its shape and values are irrelevant.
        return torch.zeros(1, 512, device=self._p.device)


class _FakeBatchEncoding:
    """Shaped like transformers' real ``BatchEncoding``, not like a plain
    ``dict`` -- and that distinction is the point of this fake.

    ``BatchEncoding`` is a ``UserDict`` subclass, so
    ``isinstance(obj, dict)`` is FALSE for it in production. ``_move_to_device``
    therefore does NOT take its recursive dict-walk branch for a real
    processor output; it takes the ``hasattr(obj, "to")`` branch instead and
    calls ``obj.to(device)`` as a single whole-object move -- exactly what
    transformers' own ``BatchEncoding.to()`` does (move every tensor value,
    including nested containers like ``history_prompt``, then return self).
    A fake built from a plain ``dict`` exercises a DIFFERENT branch than
    production ever takes, so it doesn't prove what it looks like it proves.
    This fake takes the same branch the real processor output does.
    """

    def __init__(self, input_ids, history_prompt=None):
        self.data = {"input_ids": input_ids}
        if history_prompt is not None:
            self.data["history_prompt"] = history_prompt

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v

    def __contains__(self, k):
        return k in self.data

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def to(self, device):
        """Mirrors real BatchEncoding.to(): moves every tensor value
        in-place (including the nested history_prompt dict) and returns
        self."""
        for k, v in list(self.data.items()):
            if k == "history_prompt" and isinstance(v, dict):
                self.data[k] = {hk: hv.to(device) for hk, hv in v.items()}
            elif torch.is_tensor(v):
                self.data[k] = v.to(device)
        return self


class _FakeProcessor:
    """Returns a BatchEncoding-shaped output with a nested history_prompt,
    exactly as the real BarkProcessor does before the explicit move to the
    model's device -- this is what ``_move_to_device`` exists to walk."""

    def __call__(self, text, voice_preset=None):
        return _FakeBatchEncoding(
            input_ids=torch.ones(1, 4, dtype=torch.long),
            history_prompt={
                "semantic_prompt": torch.zeros(3, dtype=torch.long),
                "coarse_prompt": torch.zeros(2, 3, dtype=torch.long),
            },
        )


def test_bark_generates_on_a_cpu_model_without_demanding_cuda():
    """The regression itself: a CPU-loaded bark must render a line.

    Before the fix this raised -- either a RuntimeError from moving inputs to
    an unavailable "cuda", or the AssertionError that demanded
    ``device.type == "cuda"``.
    """
    model = _FakeBarkModel(device="cpu")
    audio, sr = bl._generate_single_line(
        "The relay is quiet tonight.",
        "v2/en_speaker_3",
        model,
        _FakeProcessor(),
    )

    assert sr == 24000
    assert audio is not None
    assert model.seen_input_devices, "generate() was never reached"
    for dev in model.seen_input_devices:
        assert dev.type == "cpu", (
            f"inputs arrived on {dev}, but the model is on CPU -- the generate "
            "path is still assuming a device instead of asking the model"
        )
    # The NESTED history_prompt dict is exactly what _move_to_device's
    # docstring says it exists to walk -- prove it actually reached CPU too,
    # not just the top-level input_ids.
    assert model.seen_history_prompt_devices, "history_prompt never reached generate()"
    for dev in model.seen_history_prompt_devices:
        assert dev.type == "cpu"


def test_bark_moves_a_multi_chunk_line_consistently_and_proves_the_patch_ran():
    """A line long enough to split into multiple chunks re-applies the
    monkeypatch on every iteration. Prove it actually installed (not just
    that it was absent afterward, which a no-op implementation would also
    satisfy) by having generate() itself observe a patched torch.tensor."""
    long_text = (
        "The signal drifts through the static, searching for a voice. "
        "Somewhere beyond the ridge, a tower hums to itself. "
        "Nobody has answered in three winters, and still it calls. "
        "Tonight, perhaps, tonight the wire finally carries something back."
    )
    model = _FakeBarkModel(device="cpu")
    original_tensor = torch.tensor

    seen_patched_during_generate = []
    real_generate = model.generate

    def _spying_generate(**kwargs):
        seen_patched_during_generate.append(torch.tensor is not original_tensor)
        return real_generate(**kwargs)

    model.generate = _spying_generate

    bl._generate_single_line(
        long_text, "v2/en_speaker_2", model, _FakeProcessor(), speech_only=True
    )

    assert model.generate_calls >= 2, "text did not actually split into multiple chunks"
    assert seen_patched_during_generate, "generate() was never called"
    assert all(seen_patched_during_generate), (
        "torch.tensor was not patched during at least one chunk's generate() "
        "call -- the patch/restore is not being reapplied per chunk"
    )
    assert torch.tensor is original_tensor, "patch leaked past the final chunk"


def test_bark_restores_torch_tensor_and_arange_even_on_cpu():
    """The monkeypatch must never leak into the wider ComfyUI process.

    It is restored in a ``finally``, and that must hold on the CPU path too --
    a leaked ``torch.tensor`` defaulting to some device would corrupt every
    other node in the graph.
    """
    original_tensor = torch.tensor
    original_arange = torch.arange

    bl._generate_single_line(
        "Testing one two.", "v2/en_speaker_1", _FakeBarkModel(), _FakeProcessor()
    )

    assert torch.tensor is original_tensor
    assert torch.arange is original_arange


def test_bark_restores_torch_tensor_and_arange_when_generate_raises():
    """The case that actually matters for a leak: generate() itself fails.

    A restore test that only exercises the happy path would pass against a
    ``finally``-free implementation too, as long as nothing errors. This
    proves the restore survives the exception the ``finally`` exists for.
    """
    original_tensor = torch.tensor
    original_arange = torch.arange

    model = _FakeBarkModel(device="cpu", raise_on_generate=True)
    with pytest.raises(RuntimeError, match="simulated generate"):
        bl._generate_single_line(
            "This line will fail mid-generate.",
            "v2/en_speaker_1",
            model,
            _FakeProcessor(),
        )

    assert torch.tensor is original_tensor, "patch leaked after generate() raised"
    assert torch.arange is original_arange, "patch leaked after generate() raised"


def test_bark_generate_path_has_no_hardcoded_cuda_literal():
    """Source-level backstop, in the style of this repo's other bark tests.

    The behavioural tests above only prove the CPU path. This one prevents a
    future edit from reintroducing the literal on any branch they do not cover.
    """
    import inspect

    src = inspect.getsource(bl._generate_single_line)
    offending = []
    for line in src.splitlines():
        stripped = line.strip()
        if '"cuda"' not in stripped or stripped.startswith("#"):
            continue
        # A GUARDED fallback is fine and is not what broke Mac/Intel -- e.g.
        # `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
        # What must never come back is an UNGUARDED literal that assumes the
        # device without asking either the model or torch.
        if "is_available()" in stripped:
            continue
        offending.append(stripped)
    assert not offending, (
        "unguarded hardcoded 'cuda' is back in the bark generate path: "
        + repr(offending)
    )
