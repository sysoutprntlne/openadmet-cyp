from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Find the saved holdout results.
folder = Path(__file__).resolve().parents[1] / "results/holdout_random_seed42"
files = sorted(folder.glob("decision_tree_CYP*_pIC50_direct_inhibition.csv"))
if len(files) != 4:
    raise ValueError(f"Expected 4 CSV files, found {len(files)} in {folder}")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for file, ax in zip(files, axes.flat):
    data = pl.read_csv(file)
    actual = data["experimental_pIC50"].to_numpy()
    predicted = data["predicted_pIC50"].to_numpy()
    name = file.stem.removeprefix("decision_tree_").split("_")[0]

    low = min(actual.min(), predicted.min()) - 0.2
    high = max(actual.max(), predicted.max()) + 0.2

    ax.scatter(actual, predicted, s=18, alpha=0.5)
    ax.plot([low, high], [low, high], "k--")
    ax.set(xlim=(low, high), ylim=(low, high), title=name,
           xlabel="Experimental pIC50", ylabel="Predicted pIC50")
    ax.set_aspect("equal", adjustable="box")

fig.suptitle("Decision tree: held-out predictions vs. experimental labels")
fig.tight_layout()
fig.savefig(folder / "heldout_scatter.png", dpi=200)
plt.close(fig)
print("Saved:", folder / "heldout_scatter.png")