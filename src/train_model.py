"""Train and evaluate classification pipelines for synthetic accident data.

The dataset is synthetic educational data, not actual accident records.
Saved pipelines include preprocessing, so they accept the original categorical
feature columns directly at prediction time.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.2
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
TARGETS = {
    "Accident_Cause": ["Mechanical", "Environmental", "Personnel"],
    "Safety_Category": ["Unsafe Act", "Unsafe Condition"],
}


def build_pipeline(classifier: object) -> Pipeline:
    """Build a fresh encoder/classifier pipeline for one experiment."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS)
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_confusion_matrix(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list[str],
    target_name: str,
    algorithm_name: str,
    output_path: Path,
) -> None:
    """Save a labeled confusion matrix plot for one trained model."""
    matrix = confusion_matrix(actual, predicted, labels=labels)
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=axis,
    )
    axis.set_title(f"{target_name}: {algorithm_name}", fontweight="bold")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_name: str,
    algorithm_name: str,
    labels: list[str],
    results_path: Path,
) -> tuple[dict[str, object], Pipeline]:
    """Fit one independent pipeline, evaluate it, and save its matrix plot."""
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    metrics = {
        "Target": target_name,
        "Algorithm": algorithm_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, labels=labels, average="weighted", zero_division=0
        ),
    }
    plot_name = (
        f"{target_name.lower()}_{algorithm_name.lower().replace(' ', '_')}"
        "_confusion_matrix.png"
    )
    save_confusion_matrix(
        y_test,
        predictions,
        labels,
        target_name,
        algorithm_name,
        results_path / plot_name,
    )
    return metrics, pipeline


def main() -> None:
    """Train both targets with two algorithms and save the best pipelines."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "dataset" / "accident_data.csv"
    results_path = project_root / "results"
    models_path = project_root / "models"

    try:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        dataset = pd.read_csv(dataset_path)
        required_columns = FEATURE_COLUMNS + list(TARGETS)
        missing_columns = [
            column for column in required_columns if column not in dataset.columns
        ]
        if missing_columns:
            raise ValueError(f"Dataset is missing columns: {missing_columns}")
        if dataset[required_columns].isna().any().any():
            raise ValueError("Training data contains missing values.")

        results_path.mkdir(parents=True, exist_ok=True)
        models_path.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        all_metrics: list[dict[str, object]] = []
        best_models: dict[str, tuple[float, Pipeline, str]] = {}
        classifiers = {
            "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
            ),
        }

        print("Synthetic educational data only; not actual accident records.")
        print(f"Loaded dataset with shape: {dataset.shape}")

        for target_name, labels in TARGETS.items():
            x = dataset[FEATURE_COLUMNS]
            y = dataset[target_name]
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )

            for algorithm_name, classifier in classifiers.items():
                # Each call receives a new pipeline and estimator instance.
                pipeline = build_pipeline(classifier.__class__(**classifier.get_params()))
                metrics, fitted_pipeline = evaluate_model(
                    pipeline,
                    x_train,
                    x_test,
                    y_train,
                    y_test,
                    target_name,
                    algorithm_name,
                    labels,
                    results_path,
                )
                all_metrics.append(metrics)
                print(
                    f"{target_name} - {algorithm_name}: "
                    f"accuracy={metrics['Accuracy']:.4f}, "
                    f"precision={metrics['Precision']:.4f}, "
                    f"recall={metrics['Recall']:.4f}, "
                    f"f1={metrics['F1-score']:.4f}"
                )

                if (
                    target_name not in best_models
                    or metrics["F1-score"] > best_models[target_name][0]
                ):
                    best_models[target_name] = (
                        float(metrics["F1-score"]),
                        fitted_pipeline,
                        algorithm_name,
                    )

            best_f1, best_pipeline, best_algorithm = best_models[target_name]
            model_filename = (
                f"{target_name.lower()}_best_model.joblib"
            )
            joblib.dump(best_pipeline, models_path / model_filename)
            print(
                f"Best {target_name} model: {best_algorithm} "
                f"(weighted F1={best_f1:.4f})"
            )

        metrics_frame = pd.DataFrame(all_metrics)
        metrics_frame.to_csv(results_path / "model_comparison.csv", index=False)
        print("\nModel comparison:")
        print(metrics_frame.to_string(index=False))
        print(f"\nMetrics saved to: {results_path / 'model_comparison.csv'}")
        print(f"Models saved to: {models_path}")
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Model training failed: {error}")
        raise


if __name__ == "__main__":
    main()
