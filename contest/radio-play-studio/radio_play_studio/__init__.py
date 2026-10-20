"""Radio Play Studio -- a small app, built on the Comfy SDK, that writes, casts,
performs and films an old-time radio play on your own ComfyUI.

The app owns no models and no pipeline. It reads one of the Old-Time Radio pack's
shipped workflows (by default `otr_16gb_foley.json`), turns it into an API graph against YOUR server's live node
definitions, sets the few things a person chooses, and drives the run through
`comfy-sdk`. Everything that renders is open weights on your own GPU.
"""

__version__ = "0.1.0"
