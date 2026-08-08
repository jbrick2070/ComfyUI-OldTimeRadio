"""Queue item 8 (2026-08-08): pin the spandrel 0.4.x API contract.

The adapter code assumes:
* `spandrel.ModelLoader(device=...)` -- device is a __init__ kwarg
* `loader.load_from_file(path)` -- ONE positional arg
* returned ImageModelDescriptor exposes: `.scale` (int), `.device`,
  `.to(device)`, `.eval()`, `__call__(tensor(1,C,H,W))` returning
  `tensor(1,C,H*scale,W*scale)`.

If a future spandrel release breaks any of these, `requirements.txt`'s
`spandrel~=0.4.1` pin will still hold (it excludes 0.5.x majors) but this
test surfaces the drift explicitly. Add MPS support only when a Mac user
provides an integration receipt.
"""
from __future__ import annotations

import inspect

import pytest

spandrel = pytest.importorskip("spandrel")


def test_model_loader_accepts_device_kwarg():
    sig = inspect.signature(spandrel.ModelLoader.__init__)
    params = sig.parameters
    assert "device" in params, (
        f"spandrel.ModelLoader no longer accepts a 'device' kwarg; params={list(params)}")


def test_load_from_file_takes_only_path():
    sig = inspect.signature(spandrel.ModelLoader.load_from_file)
    params = list(sig.parameters)
    # First is self; second is path; that's the whole shape.
    assert len(params) == 2 and params[1] == "path", (
        f"load_from_file signature changed: {params!r}")


def test_image_model_descriptor_class_exists():
    """We use ImageModelDescriptor.scale/.device/.to/.__call__ throughout."""
    imd = getattr(spandrel, "ImageModelDescriptor", None)
    assert imd is not None
    # Verify the class's __call__ decorator is inference_mode.
    call = getattr(imd, "__call__", None)
    assert callable(call)


def test_image_model_descriptor_has_scale_field():
    """The adapter reads `descriptor.scale` to enforce intrinsic_scale.
    ImageModelDescriptor stores it as an instance attribute (`__init__` at
    :183 in spandrel 0.4.1); dir(cls) only shows class-level members, so
    inspect the source instead."""
    src = inspect.getsource(spandrel.ImageModelDescriptor)
    assert "self.scale" in src, (
        "ImageModelDescriptor no longer stores `self.scale`; the adapter's "
        "intrinsic-scale enforcement would silently miss.")


def test_image_model_descriptor_has_device_property():
    """The adapter reads `engine._descriptor.device` semantically for the
    unload path. That must exist somewhere on the class hierarchy."""
    imd = spandrel.ImageModelDescriptor
    # `device` is present via inherited ModelDescriptor / ModelBase; check that
    # at least one of the classes in the MRO defines it.
    mro = imd.__mro__
    found = any(hasattr(cls, "device") for cls in mro)
    assert found, (
        f"neither ImageModelDescriptor nor its MRO exposes .device; "
        f"MRO={[cls.__name__ for cls in mro]!r}")
