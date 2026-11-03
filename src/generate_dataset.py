"""Generate synthetic workplace accident data for educational use.

The records created by this module are synthetic examples only. They are not
actual accident records and must not be used as real-world safety evidence.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
RECORD_COUNT = 1_000


def generate_dataset(
    record_count: int = RECORD_COUNT, random_seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Create reproducible synthetic accident records with consistent labels."""
    if record_count <= 0:
        raise ValueError("record_count must be greater than zero.")

    rng = np.random.default_rng(random_seed)
    binary_features = {
        "Machine_Status": ("Normal", "Faulty"),
        "Floor_Condition": ("Dry", "Wet"),
        "Worker_Behavior": ("Safe", "Unsafe"),
        "PPE_Usage": ("Yes", "No"),
        "Lighting": ("Good", "Poor"),
        "Training_Status": ("Trained", "Untrained"),
        "Maintenance_Status": ("Good", "Poor"),
        "Machine_Guarding": ("Proper", "Missing"),
    }

    data = {
        feature: rng.choice(values, size=record_count, p=(0.7, 0.3))
        for feature, values in binary_features.items()
    }
    frame = pd.DataFrame(data)
    frame.insert(0, "Incident_ID", np.arange(1, record_count + 1))

    # Labeling rules:
    # 1. An unsafe act is present when behavior is unsafe, PPE is not used, or
    #    training is incomplete.
    # 2. An unsafe condition is present when equipment, floor, lighting,
    #    maintenance, or guarding is unsafe.
    # 3. If a generated row has neither, make its floor wet so every incident
    #    contains at least one unsafe act or unsafe condition.
    unsafe_act = (
        (frame["Worker_Behavior"] == "Unsafe")
        | (frame["PPE_Usage"] == "No")
        | (frame["Training_Status"] == "Untrained")
    )
    unsafe_condition = (
        (frame["Machine_Status"] == "Faulty")
        | (frame["Floor_Condition"] == "Wet")
        | (frame["Lighting"] == "Poor")
        | (frame["Maintenance_Status"] == "Poor")
        | (frame["Machine_Guarding"] == "Missing")
    )

    safe_rows = ~(unsafe_act | unsafe_condition)
    frame.loc[safe_rows, "Floor_Condition"] = "Wet"
    unsafe_condition = unsafe_condition | safe_rows

    # Accident causes are mutually exclusive. Mechanical takes precedence
    # when mechanical and another hazard occur together; environmental is
    # next; remaining incidents are personnel-related.
    mechanical = (
        (frame["Machine_Status"] == "Faulty")
        | (frame["Maintenance_Status"] == "Poor")
        | (frame["Machine_Guarding"] == "Missing")
    )
    environmental = (
        (frame["Floor_Condition"] == "Wet") | (frame["Lighting"] == "Poor")
    )
    frame["Accident_Cause"] = np.select(
        [mechanical, environmental],
        ["Mechanical", "Environmental"],
        default="Personnel",
    )

    # Safety_Category is also mutually exclusive. Unsafe Condition is the
    # primary category whenever any unsafe condition exists; otherwise the
    # incident is an Unsafe Act. This avoids assigning contradictory labels.
    frame["Safety_Category"] = np.where(
        unsafe_condition, "Unsafe Condition", "Unsafe Act"
    )

    return frame


def main() -> None:
    """Generate, save, and summarize the synthetic dataset."""
    output_path = Path(__file__).resolve().parents[1] / "dataset" / "accident_data.csv"

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = generate_dataset()
        dataset.to_csv(output_path, index=False)

        print("Synthetic educational data only; not actual accident records.")
        print(f"Dataset saved to: {output_path}")
        print(f"Dataset shape: {dataset.shape}")
        print("\nAccident_Cause distribution:")
        print(dataset["Accident_Cause"].value_counts())
        print("\nSafety_Category distribution:")
        print(dataset["Safety_Category"].value_counts())
        print("\nFirst five records:")
        print(dataset.head().to_string(index=False))
    except (OSError, ValueError, ImportError) as error:
        print(f"Failed to generate dataset: {error}")
        raise


if __name__ == "__main__":
    main()
