"""Explore and visualize the synthetic accident classification dataset.

The source data is synthetic educational data, not actual accident records.
"""

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FEATURE_COLUMNS = [
    "Machine_Status",
    "Floor_Condition",
    "Worker_Behavior",
    "PPE_Usage",
    "Lighting",
    "Training_Status",
    "Maintenance_Status",
    "Machine_Guarding",
]
TARGET_COLUMNS = ["Accident_Cause", "Safety_Category"]


def save_count_plot(
    dataset: pd.DataFrame,
    column: str,
    output_path: Path,
    title: str,
    x_label: str,
) -> None:
    """Create and save a readable count plot for one categorical column."""
    figure, axis = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=dataset,
        x=column,
        hue=column,
        legend=False,
        palette="Set2",
        ax=axis,
    )
    axis.set_title(title, fontsize=14, fontweight="bold")
    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel("Number of records", fontsize=11)
    axis.tick_params(axis="x", rotation=10)
    for container in axis.containers:
        axis.bar_label(container, padding=3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_feature_distributions(
    dataset: pd.DataFrame, output_path: Path
) -> None:
    """Create and save a grid of count plots for all input features."""
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, feature in zip(axes.flat, FEATURE_COLUMNS):
        sns.countplot(
            data=dataset,
            x=feature,
            hue=feature,
            legend=False,
            palette="Set3",
            ax=axis,
        )
        axis.set_title(feature.replace("_", " "), fontsize=11, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("Records")
        axis.tick_params(axis="x", rotation=25)
        for container in axis.containers:
            axis.bar_label(container, padding=2, fontsize=8)
    figure.suptitle("Input feature distributions", fontsize=16, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def print_observations(dataset: pd.DataFrame) -> None:
    """Print simple observations that help beginners interpret the analysis."""
    cause_counts = dataset["Accident_Cause"].value_counts()
    category_counts = dataset["Safety_Category"].value_counts()
    print("\nSimple observations:")
    print(
        f"- {cause_counts.idxmax()} is the most common accident cause "
        f"({cause_counts.max()} records)."
    )
    print(
        f"- {category_counts.idxmax()} is the most common safety category "
        f"({category_counts.max()} records)."
    )
    print(
        "- The feature charts show how often each possible condition appears "
        "in the synthetic dataset."
    )
    print(
        "- These patterns describe generated educational data and should not "
        "be interpreted as real workplace risk measurements."
    )


def main() -> None:
    """Load, summarize, and visualize the synthetic accident dataset."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "dataset" / "accident_data.csv"
    results_path = project_root / "results"

    try:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        dataset = pd.read_csv(dataset_path)
        missing_columns = [
            column
            for column in FEATURE_COLUMNS + TARGET_COLUMNS
            if column not in dataset.columns
        ]
        if missing_columns:
            raise ValueError(f"Dataset is missing columns: {missing_columns}")

        results_path.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")

        print("Synthetic educational data only; not actual accident records.")
        print(f"\nDataset shape: {dataset.shape}")
        print("\nDataset information:")
        info_output = StringIO()
        dataset.info(buf=info_output)
        print(info_output.getvalue())
        print(f"Missing values by column:\n{dataset.isna().sum()}")
        print(f"\nDuplicate records: {dataset.duplicated().sum()}")
        print(f"\nDescriptive statistics:\n{dataset.describe(include='all').T}")

        save_count_plot(
            dataset,
            "Accident_Cause",
            results_path / "accident_cause_distribution.png",
            "Accident cause distribution",
            "Accident cause",
        )
        save_count_plot(
            dataset,
            "Safety_Category",
            results_path / "safety_category_distribution.png",
            "Safety category distribution",
            "Safety category",
        )
        save_feature_distributions(
            dataset, results_path / "feature_distributions.png"
        )
        print_observations(dataset)

        saved_charts = sorted(results_path.glob("*.png"))
        print("\nSaved charts:")
        for chart in saved_charts:
            print(f"- {chart}")
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Data analysis failed: {error}")
        raise


if __name__ == "__main__":
    main()
