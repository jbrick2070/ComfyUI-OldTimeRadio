VERDICT: build-ready as-is? yes-with-fixes. The active-output design matches the confirmed defect, but the proposed iterable contract must close one category hole and the live acceptance receipt must distinguish deterministic prompt proof from subjective dialogue quality.

MUST-FIX BEFORE BUILD:

1. [P2.1, P4.2] CONFIRMED — `Iterable[str]` alone accepts a scalar string, which would iterate character-by-character and silently miss `LEMMY`. Make the accepted container contract explicit (`Sequence[str]` or tuple input), reject `str`/`bytes` as the container, validate every element is `str`, and have both real callers pass tuples. Add a scalar-string failure test as well as the dict/object category tests.
2. [P5.3] CONFIRMED — the plan mixes two different acceptance claims: captured prompts can prove policy scope, while a live ledger/listen can only demonstrate resulting dialogue quality. State that deterministic prompt tests are the hard regression gate and the forced-Lemmy canonical episode is the production reachability/listening gate; do not treat a small lexical sample as proof that bleed is impossible.

SHOULD-FIX:

1. [P1] CONFIRMED — add the three named published ledgers that demonstrate the failure and one `scifi_news_pro` control ledger. This makes the causal chain auditable without asking the builder to rediscover the artifacts.
2. [P3.3] CONFIRMED — the direct mixed-group prompt test is the exchange test that kills the current implementation. Label the later non-Lemmy/prepass test as a scope-invariant regression rather than implying it alone reproduces the current grouped defect; production `_normalize_cast` already converts full-cast dictionaries to objects the old detector ignores.
3. [P6] CONFIRMED — split the project-repo commit/push from the separate Bug Bible repository update. The repositories cannot share one commit or one HEAD/origin receipt.

OPTIONAL / NICE-TO-HAVE:

- [P5.3] Preserve a compact before/after excerpt in the PBUG receipt, but do not turn Cockney vocabulary into a production blacklist.

CUT THESE (scope / over-engineering):

1. [P3.4] Do not add a new `scifi_news_pro` test unless a touched dependency reaches it; its existing cameo/register tests plus the full suite are sufficient for unchanged code.
2. [P2.4] Keep `_normalize_cast` voice-card widening out of this change. It is adjacent but not required to close prompt scope.
