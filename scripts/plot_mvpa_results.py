from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


results_dir = Path("results/mvpa")

summary = pd.read_csv(
    results_dir / "mvpa_subject_results.csv"
)

group_null = np.load(
    results_dir / "mvpa_group_null.npy"
)

observed_mean = summary["accuracy"].mean()


# Individual accuracies
fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(summary))

ax.scatter(
    x,
    summary["accuracy"],
    s=55,
)

ax.axhline(
    0.5,
    linestyle="--",
    label="Chance level",
)

ax.axhline(
    observed_mean,
    linestyle="-",
    label=f"Group mean = {observed_mean:.3f}",
)

ax.set_xticks(x)
ax.set_xticklabels(
    summary["subject"].str.replace("sub-", ""),
    rotation=45,
    ha="right",
)

ax.set_ylabel("Classification accuracy")
ax.set_xlabel("Subject")
ax.set_ylim(0.25, 0.85)
ax.legend()

fig.tight_layout()
fig.savefig(
    results_dir / "mvpa_subject_accuracies.png",
    dpi=200,
)
plt.close(fig)


# Group null distribution
fig, ax = plt.subplots(figsize=(7, 5))

ax.hist(
    group_null,
    bins=35,
)

ax.axvline(
    observed_mean,
    linestyle="--",
    linewidth=2,
    label=f"Observed mean = {observed_mean:.3f}",
)

ax.set_xlabel("Mean classification accuracy")
ax.set_ylabel("Count")
ax.legend()

fig.tight_layout()
fig.savefig(
    results_dir / "mvpa_group_permutation.png",
    dpi=200,
)
plt.close(fig)

print("Figures saved in", results_dir)