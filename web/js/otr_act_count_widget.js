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

    const refresh = () => {
      const w = wordsWidget.value;
      const dflt = defaultActCount(w);
      const max = maxActCount(w);

      // act_count value 0 means "Python auto-derives". Keep it
      // selectable even when target_words is below 30; the Python
      // validator surfaces the right error in run().
      if (dflt === null) {
        actsWidget.options = actsWidget.options || {};
        actsWidget.options.values = [0, 1];
        if (actsWidget.value > 1) actsWidget.value = 0;
        node.setDirtyCanvas(true, true);
        return;
      }

      // Allowed values: 0 (auto) + [dflt .. max].
      const allowed = [0, ...range(dflt, max)];
      actsWidget.options = actsWidget.options || {};
      actsWidget.options.values = allowed;
      // The act_count widget is declared as INT in INPUT_TYPES, so
      // the options.values list is purely informational for any UI
      // dropdown that surfaces it; we ALSO clamp the raw value so
      // an out-of-band saved graph snaps to the nearest legal pick.
      if (actsWidget.value !== 0) {
        if (actsWidget.value < dflt) actsWidget.value = dflt;
        if (actsWidget.value > max) actsWidget.value = max;
      }
      node.setDirtyCanvas(true, true);
    };

    const prev = wordsWidget.callback;
    wordsWidget.callback = (v) => {
      if (typeof prev === "function") prev(v);
      refresh();
    };
    refresh();
  },
});
