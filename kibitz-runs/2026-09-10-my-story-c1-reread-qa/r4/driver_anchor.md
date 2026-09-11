# Codex grounded reread correction

VERDICT: original must-fix confirmed and fixed. _repair_row uses reread_ok before
both acceptance paths and before best_count/best_text. No validated reread means
no cleanliness/progress claim. Previous verified progress may still ship flagged.
Candidate neighbor text remains current; original repair text/scope remain fixed.

Six new cases pass: malformed reread cannot win either path; initially malformed
judge can recover on a real candidate reread; provider, memory, cancellation and
terminal-capacity errors propagate specifically from candidate reread. Full
183-test focused set passes; final full and Bible evidence pending. One targeted
Sonnet follow-up is warranted by the first review's blocker, not reviewer count.
