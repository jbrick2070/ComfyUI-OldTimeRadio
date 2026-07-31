# Pass 03 judgment: wiring and lifetime

The local review strengthens the rejection of a formula-first fix. The current
wrapper does not prove that a Python value's last consumer is a GPU lifetime
boundary, and its direct graph execution delays important AIMDO cleanup. A
pairwise whole-stage maximum would be as invented as the clean stage maximum.

Accepted implementation contract:

- split conditioning, sampling, and decode into explicit executions;
- split heavy VAE encode/decode lifetimes where they do not overlap;
- add a real quiescence/cleanup barrier;
- expose placement through one OTR-owned loader, not parallel graph branches;
- make persistent conditioning a provenance-complete model-free source; and
- carry every new control and receipt through the canonical workflow/ledger.

Admission will be recipe-versioned and empirical and will reject every
extrapolation with a named error.
