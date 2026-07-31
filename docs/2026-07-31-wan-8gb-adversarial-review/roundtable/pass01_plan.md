# Pass 01 converged plan

1. Freeze product semantics: exact canvas, frames, steps, scheduler, seed,
   conditioning, tiling, and no-fallback behavior.
2. Make the request authoritative through the canonical workflow and add
   continuous phase/lifetime telemetry. Do not calibrate against the current
   absolute NVML seed.
3. Add qualification-only controls that can select actual patcher/format and
   encoder placement without global launch-policy confounds. The controls and
   receipts must traverse the real canonical JSON.
4. Run a staged experiment rather than the proposed four-cell Cartesian:
   mechanism viability, encoder/cache behavior, decode/frame/pixel envelope,
   then repeat/soak.
5. Model the observed lifetime envelope. Use an empirical monotone table or
   upper piecewise envelope per exact recipe; do not decompose weight/latent/
   activation terms without phase instrumentation.
6. Label an 8 GiB reserve clamp as dev-card prequalification only. Physical
   8 GB remains unverified until a real-card artifact exists.
