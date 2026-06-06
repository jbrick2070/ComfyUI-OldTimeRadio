"""OTR shared platform primitives -- dependency-free, cold-import-clean.

This package holds the cross-subproject building blocks the model-agnostic
video/image platform is built on (A-Seam deltas AS-1..AS-5):

* :mod:`engine_registry_base` -- the reusable fail-closed engine-registry
  pattern + usability taxonomy, anchored on the SHAPE of the SHIPPED audio
  ``AudioEngine(Protocol)`` (AS-4). One pattern, three parallel namespaces
  (audio is frozen/independent; video + image reference this base).
* :mod:`role_compat` -- the ONE shared role<->required-inputs engine filter
  (AS-1), imported by the video director, the image director and the 3D
  availability check alike.
* :mod:`resolver` -- execution-group DAG validation + the conditional-skip
  RESOLVER-PRUNE of orphaned background groups on fallback-restamp (AS-2).

Every module here imports NOTHING heavy at module scope (no torch /
transformers / diffusers / pydantic). Importing this package -- and any
submodule -- must never pull a model framework into memory (invariant V-12,
the cold-import test). Keep this ``__init__`` free of eager submodule imports
so a consumer pulls in only what it names.
"""
