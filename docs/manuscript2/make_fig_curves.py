"""One-off: personal-test MAE curves for the manuscript. Run from manuscript2/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Locked means: 7 T1DM users; 60 d is n=6 (User 1082 has no 60-day cell).
# SugarOne / JEPA-288: jepa_paper table / jepa_mae_by_days.csv MEAN rows.
# NBEATSx / TFT: PERSONALIZATION_NF_REPORT.md per-user tables, averaged here.
labels = ["ZS", "3", "7", "14", "30", "60", "full"]
sugarone = [19.48, 19.61, 19.62, 19.56, 19.64, 19.09, 18.67]
jepa288 = [18.13, 18.08, 17.99, 17.92, 17.82, 18.09, 17.51]
nbeatsx = [23.05, 29.73, 28.60, 26.86, 25.66, 23.91, 21.58]
tft = [24.41, 32.78, 29.56, 27.04, 22.65, 21.53, 19.87]
x = list(range(len(labels)))

fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
ax.plot(x, sugarone, "o-", color="#4C78A8", label="SugarOne", linewidth=1.6, markersize=5)
ax.plot(x, jepa288, "s-", color="#F58518", label="SugarJEPA-288", linewidth=1.6, markersize=5)
ax.plot(x, nbeatsx, "^-", color="#E45756", label="NBEATSx", linewidth=1.6, markersize=5)
ax.plot(x, tft, "D-", color="#54A24B", label="TFT", linewidth=1.6, markersize=5)
ax.axvline(4, color="#9E9E9E", linestyle=":", linewidth=1.0)
ax.scatter([4], [sugarone[4]], s=90, facecolors="none", edgecolors="#4C78A8", linewidths=1.6, zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Personal train budget (days; ZS = frozen global model)")
ax.set_ylabel("Personal-test MAE (mg/dL)")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(16.5, 34.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out = Path(__file__).with_name("fig_personalization_curves.png")
fig.savefig(out, bbox_inches="tight")
print(out)
