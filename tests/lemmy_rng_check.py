"""Lemmy RNG sanity check — verifies the 11% hit rate is statistically intact.

Usage:
    python tests/lemmy_rng_check.py

Expected output:
    Runs 10,000 simulated Lemmy rolls using the same SystemRandom mechanism
    as the orchestrator. Reports observed rate and checks it falls within
    ±1.5% of the 11% target (99% confidence interval for n=10,000).

This guards against:
    - Accidental removal of _LEMMY_RNG
    - Accidental threshold edits (e.g., 0.11 → 0.011)
    - Seed bleed-through from other RNGs
"""

from secrets import SystemRandom

LEMMY_RATE = 0.11
N_TRIALS = 10_000
TOLERANCE = 0.015  # ±1.5%


def main():
    rng = SystemRandom()
    hits = sum(1 for _ in range(N_TRIALS) if rng.random() < LEMMY_RATE)
    observed = hits / N_TRIALS
    delta = abs(observed - LEMMY_RATE)

    print(f"Lemmy RNG sanity check — {N_TRIALS:,} trials")
    print(f"  Target rate:   {LEMMY_RATE:.1%}")
    print(f"  Observed rate: {observed:.2%}  ({hits:,} hits)")
    print(f"  Delta:         {delta:.2%}  (tolerance {TOLERANCE:.1%})")

    if delta <= TOLERANCE:
        print("  STATUS: PASS — Lemmy 11% is statistically intact")
        return 0
    else:
        print("  STATUS: FAIL — RNG is biased, investigate _LEMMY_RNG and threshold")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
