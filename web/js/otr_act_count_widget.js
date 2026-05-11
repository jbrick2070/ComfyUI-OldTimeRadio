/* web/js/otr_act_count_widget.js
 *
 * Phase 2A (2026-05-11) -- live act_count widget management.
 *
 * Watches the target_words integer widget on every OTR_LedgerScriptWriter
 * node and live-updates the act_count widget's selectable range +
 * current value clamping so the user can never select something
 * Queue Prompt will reject.
 *
 * Mirror of _DEFAULT_ACT_BREAKPOINTS + max_act_count in
 * nodes/_otr_episode_budget.py. Python is authoritative; this duplicate
 * exists for live UI feedback only. Update BOTH when changing the
 * thresholds.
 */

import { app } from "../../scripts/app.js";

const DEFAULT_BREAKS = [
  { floor: 300, acts: 3 },
  { floor: 150, acts: 2 },
  { floor: 30,  acts: 1 },
];

function defaultActCount(words) {
  if (typeof words !== "number" || words < 30) return null;
  for (const { floor, acts } of DEFAULT_BREAKS) {
    if (words >= floor) return acts;
  }
  return 1;
}

function maxActCount(words) {
  if (typeof words !== "number" || words < 30) return 1;
  return Math.min(7, Math.max(1, Math.floor(words / 50)));
}

function range(lo, hi) {
  const out = [];
  for (let i = lo; i <= hi; i++) out.push(i);
  return out;
}

app.registerExtension({
  name: "OTR.ActCountWidget",
  nodeCreated(node) {
    if (node.comfyClass !== "OTR_LedgerScriptWriter") return;

    const wordsWidget = node.widgets.find(w => w.name === "target_words");
    const actsWidget  = node.widgets.find(w => w.name === "act_count");
    if (!wordsWidget || !actsWidget) return;

    // Behavior contract (post-Phase-3 review 2026-05-11, per
    // Jeffrey's "350 -> 3 and dynamic if i change" feedback):
    //
    //   - On node creation OR target_words change, the widget's
    //     displayed value is updated to reflect the DERIVED default
    //     act_count for the current target_words. The 0 sentinel is
    //     replaced with the real default so the user sees the
    //     correct number, not a placeholder.
    //   - The widget's allowed range becomes [default..max] for the
    //     current target_words. The user can pick UP within that
    //     band but never below the default.
    //   - User picks within the band are preserved across
    //     target_words changes, UNLESS the new target_words shifts
    //     the band such that the user's pick is out of range -- in
    //     which case the value is clamped to the nearest legal pick.
    //   - When target_words is below 30 (sub-minimum), the widget
    //     shows 1; Python's run-time validator surfaces the
    //     InvalidEpisodeBudgetError when Queue Prompt fires.
    const refresh = () => {
      const w = wordsWidget.value;
      const dflt = defaultActCount(w);
      const max = maxActCount(w);

      if (dflt === null) {
        // Sub-minimum target_words. Pin the widget to 1 -- the
        // Python validator fails-loud on Queue Prompt anyway.
        actsWidget.options = actsWidget.options || {};
        actsWidget.options.values = [1];
        actsWidget.value = 1;
        node.setDirtyCanvas(true, true);
        return;
      }

      // Allowed band: [default..max].
      const allowed = range(dflt, max);
      actsWidget.options = actsWidget.options || {};
      actsWidget.options.values = allowed;

      const current = actsWidget.value;
      // Snap to default when value is the 0 sentinel OR below the
      // current default. Clamp down when above the current max.
      // Preserve any user pick that's already inside [dflt..max].
      let next;
      if (current === 0 || current < dflt) {
        next = dflt;
      } else if (current > max) {
        next = max;
      } else {
        next = current;
      }
      if (actsWidget.value !== next) {
        actsWidget.value = next;
      }
      node.setDirtyCanvas(true, true);
    };

    const prev = wordsWidget.callback;
    wordsWidget.callback = (v) => {
      if (typeof prev === "function") prev(v);
      refresh();
    };
    // Also run refresh once on creation so an existing graph that
    // loads with act_count=0 immediately snaps to the right value.
    refresh();
  },
});
