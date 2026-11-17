"""Create the report figures from the retained per-run experiment results.

Run from any directory: python scripts/plot_report.py [--output-dir PATH]
Search seeds are averaged within each outer split before split means are plotted.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (("California Housing", "California", "#156082"),
            ("Diabetes Progression", "Diabetes", "#C05B32"))
ARMS = {
    "phase_1": ("adaptive", [("selection", "Selection"),
                               ("construction", "Construction"),
                               ("combined", "Selection + construction")]),
    "phase_2": ("individual", [("quadratic", "Quadratic"), ("rbf", "RBF"),
                                 ("interactions", "Interactions"), ("hinge", "Hinge")]),
    "phase_3": ("balanced", [("quadratic", "Quadratic"), ("rbf", "RBF"),
                               ("hinge", "Hinge"), ("quadratic_rbf", "Quadratic + RBF"),
                               ("quadratic_hinge", "Quadratic + hinge"),
                               ("quadratic_rbf_hinge", "Quadratic + RBF + hinge")]),
    "phase_4": ("asymmetric", [("quadratic", "Quadratic"), ("rbf", "RBF"),
                                 ("quadratic_rbf_balanced", "Quadratic + RBF: balanced"),
                                 ("quadratic_rbf_rbf_heavy", "Quadratic + RBF: RBF-heavy"),
                                 ("quadratic_rbf_hinge_balanced", "Three-family: balanced"),
                                 ("quadratic_rbf_hinge_rbf_heavy", "Three-family: RBF-heavy"),
                                 ("quadratic_rbf_hinge_quadratic_heavy", "Three-family: quadratic-heavy")]),
}


def load(phase, run):
    path = ROOT / "results" / phase / run / "runs.csv"
    data = pd.read_csv(path)
    required = {"dataset", "split", "search_seed", "arm", "learner", "mae", "mae_reduction_pct"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing required columns in {path}: {required - set(data.columns)}")
    return data


def mean_by_split(data, keys, metric):
    """Prevent search seeds from being treated as independent data splits."""
    return data.groupby([*keys, "split"], as_index=False, dropna=False)[metric].mean()


def reduction_chart(phase, output):
    run, arms = ARMS[phase]
    data = load(phase, run)
    data = data[data.arm.isin([arm for arm, _ in arms])]
    split = mean_by_split(data, ["dataset", "arm"], "mae_reduction_pct")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, max(3.2, 0.46 * len(arms) + 1.5)),
                             sharey=True, constrained_layout=True)
    y = np.arange(len(arms))
    for ax, (dataset, label, color) in zip(axes, DATASETS):
        for index, (arm, _) in enumerate(arms):
            values = split.loc[(split.dataset == dataset) & (split.arm == arm), "mae_reduction_pct"]
            if len(values) != 5:
                raise ValueError(f"Expected five splits for {dataset}, {arm}; got {len(values)}")
            ax.scatter(values, np.full(len(values), index), s=23, color=color, alpha=0.28, zorder=2)
            ax.scatter(values.mean(), index, s=72, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.axvline(0, color="#707070", linewidth=0.8)
        ax.set_title(label, fontweight="bold", color=color)
        ax.grid(axis="x", color="#dedede", linewidth=0.7)
        ax.set_xlabel("Outer-test MAE reduction (%)")
        ax.set_yticks(y, [name for _, name in arms])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("")
    fig.suptitle(f"{phase.replace('_', ' ').title()}: performance by dataset", fontweight="bold")
    fig.savefig(output / f"{phase}_reduction.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def optimizer_chart(output):
    data = load("phase_5", "equal_time_30s")
    data = data[data.arm != "no_fe"].copy()
    split = mean_by_split(data, ["dataset", "family", "optimizer"], "mae_reduction_pct")
    methods = ["GA", "PSO", "ES", "DE", "CMA-ES"]
    tasks = [("rbf", "RBF"), ("quadratic_rbf_hinge_rbf_heavy", "RBF-heavy mixture")]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=True, constrained_layout=True)
    for ax, (family, title) in zip(axes, tasks):
        for offset, (dataset, label, color) in zip((-0.14, 0.14), DATASETS):
            values = []
            for method in methods:
                item = split[(split.dataset == dataset) & (split.family == family) & (split.optimizer == method)]
                if len(item) != 5:
                    raise ValueError(f"Expected five splits for {family}, {method}, {dataset}")
                values.append(item.mae_reduction_pct.mean())
            ax.scatter(values, np.arange(len(methods)) + offset, label=label, s=60, color=color, zorder=3)
        ax.axvline(0, color="#707070", linewidth=0.8)
        ax.set_title(title, fontweight="bold")
        ax.set_yticks(np.arange(len(methods)), methods)
        ax.set_xlabel("Mean outer-test MAE reduction (%)")
        ax.grid(axis="x", color="#dedede", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Phase 5: optimizer performance with a 30-second search", fontweight="bold")
    fig.savefig(output / "phase_5_optimizers.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def learner_chart(output):
    data = load("phase_6", "linear_learners")
    split = mean_by_split(data, ["dataset", "learner", "arm"], "mae")
    learners = [("linear_panel_ols", "OLS"), ("linear_panel_ridge", "Ridge"),
                ("linear_panel_lasso", "Lasso"), ("linear_panel_elastic", "Elastic Net"),
                ("linear_panel_huber", "Huber")]
    arms = [("no_fe", "No FE", "#777777"), ("rbf", "RBF", "#156082"),
            ("rbf_heavy_mixture", "RBF-heavy mixture", "#C05B32")]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True, constrained_layout=True)
    for ax, (dataset, label, _) in zip(axes, DATASETS):
        for offset, (arm, arm_label, color) in zip((-0.2, 0, 0.2), arms):
            values = []
            for learner, _ in learners:
                item = split[(split.dataset == dataset) & (split.learner == learner) & (split.arm == arm)]
                if len(item) != 5:
                    raise ValueError(f"Expected five splits for {dataset}, {learner}, {arm}")
                values.append(item.mae.mean())
            ax.scatter(values, np.arange(len(learners)) + offset, label=arm_label, color=color, s=55, zorder=3)
        ax.set_title(label, fontweight="bold")
        ax.set_yticks(np.arange(len(learners)), [label for _, label in learners])
        ax.set_xlabel("Mean outer-test MAE (dataset units)")
        ax.grid(axis="x", color="#dedede", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    fig.suptitle("Phase 6: absolute error by learner and representation", fontweight="bold")
    fig.savefig(output / "phase_6_learners.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def capacity_chart(output):
    data = load("phase_2", "individual")
    arms = ARMS["phase_2"][1]
    counts = [0, 1, 2, 4, 8]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), sharey=True, constrained_layout=True)
    for ax, (dataset, title, _) in zip(axes, DATASETS):
        matrix = []
        for arm, _ in arms:
            rows = data[(data.dataset == dataset) & (data.arm == arm)]
            if len(rows) != 25 or set(rows.selected_k) - set(counts):
                raise ValueError(f"Unexpected capacity choices for {dataset}, {arm}")
            matrix.append([int((rows.selected_k == count).sum()) for count in counts])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=25, aspect="auto")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center",
                        color="white" if value >= 13 else "#202020", fontsize=10)
        ax.set_xticks(range(len(counts)), [str(k) for k in counts])
        ax.set_yticks(range(len(arms)), [name for _, name in arms])
        ax.set_xlabel("Selected number of constructed features (k)")
        ax.set_title(title, fontweight="bold")
        ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, label="Selections out of 25 searches", shrink=0.8)
    fig.suptitle("Phase 2: how often each feature count was selected", fontweight="bold")
    fig.savefig(output / "phase_2_capacity.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def validation_chart(output):
    data = load("phase_5", "equal_time_30s")
    data = data[(data.dataset == "Diabetes Progression") & (data.arm != "no_fe")].copy()
    data["inner_reduction"] = 100 * (data.baseline_inner_mae - data.selected_inner_mae) / data.baseline_inner_mae
    methods = ["GA", "PSO", "ES", "DE", "CMA-ES"]
    tasks = [("rbf", "RBF"), ("quadratic_rbf_hinge_rbf_heavy", "RBF-heavy mixture")]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True, constrained_layout=True)
    for ax, (family, title) in zip(axes, tasks):
        subset = data[data.family == family]
        for offset, (metric, label, color) in zip((-0.2, 0.2),
                [("inner_reduction", "Inner validation", "#56738c"),
                 ("mae_reduction_pct", "Outer test", "#C05B32")]):
            split = mean_by_split(subset, ["optimizer"], metric)
            values = []
            for method in methods:
                scores = split.loc[split.optimizer == method, metric]
                if len(scores) != 5:
                    raise ValueError(f"Expected five splits for {family}, {method}")
                values.append(scores.mean())
            ax.barh(np.arange(len(methods)) + offset, values, height=0.35, color=color, label=label)
        ax.axvline(0, color="#707070", linewidth=0.8)
        ax.set_yticks(np.arange(len(methods)), methods)
        ax.set_xlabel("Mean MAE reduction (%)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color="#dedede", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Phase 5: validation and test results on Diabetes", fontweight="bold")
    fig.savefig(output / "phase_5_validation_gap.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def ridge_error_chart(output):
    data = load("phase_6", "linear_learners")
    data = data[data.learner == "linear_panel_ridge"]
    metrics = [("mae", "MAE"), ("rmse", "RMSE"), ("p95_abs_residual", "95th-percentile error")]
    colors = ["#156082", "#C05B32"]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    x = np.arange(len(metrics))
    for offset, (dataset, label, _), color in zip((-0.18, 0.18), DATASETS, colors):
        subset = data[data.dataset == dataset]
        baseline = subset[subset.arm == "no_fe"].set_index("split")
        featured = subset[subset.arm == "rbf_heavy_mixture"].copy()
        if len(baseline) != 5 or len(featured) != 25:
            raise ValueError(f"Unexpected Ridge run count for {dataset}")
        values = []
        for metric, _ in metrics:
            paired = featured.join(baseline[[metric]], on="split", rsuffix="_baseline")
            paired["reduction"] = 100 * (paired[metric + "_baseline"] - paired[metric]) / paired[metric + "_baseline"]
            values.append(paired.groupby("split").reduction.mean().mean())
        bars = ax.bar(x + offset, values, width=0.35, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.axhline(0, color="#707070", linewidth=0.8)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Mean error reduction vs Ridge without FE (%)")
    ax.grid(axis="y", color="#dedede", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, ncols=2)
    fig.suptitle("Phase 6: Ridge with RBF-heavy mixture across error metrics", fontweight="bold")
    fig.savefig(output / "phase_6_ridge_error_metrics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def learner_compute_chart(output):
    data = load("phase_6", "linear_learners")
    data = data[data.arm == "rbf_heavy_mixture"]
    learners = [("linear_panel_ols", "OLS", "#156082"),
                ("linear_panel_ridge", "Ridge", "#5B7D3A"),
                ("linear_panel_lasso", "Lasso", "#BA7830"),
                ("linear_panel_elastic", "Elastic Net", "#805B9A"),
                ("linear_panel_huber", "Huber", "#C05B32")]
    offsets = {"OLS": (0, -17), "Ridge": (-2, 8), "Lasso": (-8, -17),
               "Elastic Net": (0, 8), "Huber": (0, 8)}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for ax, (dataset, label, _) in zip(axes, DATASETS):
        for name, display, color in learners:
            rows = data[(data.dataset == dataset) & (data.learner == name)]
            split = rows.groupby("split", as_index=False)[["mae", "runtime_seconds"]].mean()
            if len(rows) != 25 or len(split) != 5:
                raise ValueError(f"Expected 25 runs on five splits for {dataset}, {name}")
            runtime = split.runtime_seconds.mean()
            mae = split.mae.mean()
            ax.scatter(runtime, mae, color=color, s=75, edgecolor="white", linewidth=0.8, zorder=3)
            label_offset = offsets[display]
            if dataset == "Diabetes Progression" and display in ("OLS", "Ridge"):
                label_offset = (-26, 9) if display == "OLS" else (28, -12)
            ax.annotate(display, (runtime, mae), xytext=label_offset, textcoords="offset points",
                        ha="center", va="center", fontsize=9)
        ax.set_xscale("log")
        ticks = [15, 30, 60, 120] if dataset == "California Housing" else [5, 10, 20]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Mean FE-arm runtime (seconds, log scale)")
        ax.set_ylabel("Mean outer-test MAE (dataset units)")
        ax.set_title(label, fontweight="bold")
        ax.grid(color="#dedede", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(x=0.18, y=0.22)
    fig.suptitle("Phase 6: error and computation for the RBF-heavy mixture", fontweight="bold")
    fig.savefig(output / "phase_6_compute_tradeoff.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})
    for phase in ARMS:
        reduction_chart(phase, args.output_dir)
    optimizer_chart(args.output_dir)
    learner_chart(args.output_dir)
    capacity_chart(args.output_dir)
    validation_chart(args.output_dir)
    ridge_error_chart(args.output_dir)
    learner_compute_chart(args.output_dir)
    print(f"Saved ten figures to {args.output_dir}")


if __name__ == "__main__":
    main()
