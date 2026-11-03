"""Streamlit dashboard for the Accident Cause Classification project."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURES = [
    "Machine_Status",
    "Floor_Condition",
    "Worker_Behavior",
    "PPE_Usage",
    "Lighting",
    "Training_Status",
    "Maintenance_Status",
    "Machine_Guarding",
]

INPUT_OPTIONS = {
    "Machine Status": ("Machine_Status", ["Normal", "Faulty"]),
    "Floor Condition": ("Floor_Condition", ["Dry", "Wet"]),
    "Worker Behavior": ("Worker_Behavior", ["Safe", "Unsafe"]),
    "PPE Usage": ("PPE_Usage", ["Yes", "No"]),
    "Lighting": ("Lighting", ["Good", "Poor"]),
    "Training Status": ("Training_Status", ["Trained", "Untrained"]),
    "Maintenance Status": ("Maintenance_Status", ["Good", "Poor"]),
    "Machine Guarding": ("Machine_Guarding", ["Proper", "Missing"]),
}

st.set_page_config(
    page_title="Accident Cause Classification",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_models() -> dict[str, object]:
    """Load the trained preprocessing-and-model pipelines once per session."""
    model_paths = {
        "cause": MODEL_DIR / "accident_cause_best_model.joblib",
        "safety": MODEL_DIR / "safety_category_best_model.joblib",
    }
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required trained model files are missing: " + ", ".join(missing)
        )
    return {name: joblib.load(path) for name, path in model_paths.items()}


def display_header() -> None:
    """Show the shared dashboard header."""
    st.title("Accident Cause Classification System")
    st.caption("Synthetic educational data dashboard for workplace safety analysis")


def display_probability_chart(
    probabilities: object, classes: object, title: str
) -> None:
    """Render a compact probability chart for one prediction."""
    probability_frame = pd.DataFrame(
        {"Class": list(classes), "Probability": list(probabilities)}
    ).sort_values("Probability", ascending=True)
    st.subheader(title)
    st.bar_chart(
        probability_frame.set_index("Class"),
        y="Probability",
        x_label="Class",
        y_label="Probability",
        horizontal=True,
        height=240,
    )


def recommendations(inputs: dict[str, str]) -> list[str]:
    """Return practical recommendations based on the selected conditions."""
    items = []
    if inputs["Worker_Behavior"] == "Unsafe":
        items.append("Review safe work procedures and provide supervisor coaching.")
    if inputs["PPE_Usage"] == "No":
        items.append("Require the appropriate PPE before work begins.")
    if inputs["Training_Status"] == "Untrained":
        items.append("Complete task-specific safety training before assignment.")
    if inputs["Machine_Status"] == "Faulty":
        items.append("Stop and isolate the machine until it is inspected.")
    if inputs["Maintenance_Status"] == "Poor":
        items.append("Schedule corrective maintenance and document completion.")
    if inputs["Machine_Guarding"] == "Missing":
        items.append("Install and verify the required machine guarding.")
    if inputs["Floor_Condition"] == "Wet":
        items.append("Remove the slip hazard and place warning signage.")
    if inputs["Lighting"] == "Poor":
        items.append("Restore adequate lighting in the work area.")
    return items or ["Continue routine inspections and follow established controls."]


def render_home() -> None:
    """Render the project overview page."""
    st.header("Safer decisions through data")
    st.write(
        "This beginner-friendly application uses trained classification pipelines "
        "to estimate an accident cause and safety category from workplace conditions."
    )
    st.info(
        "The dataset and predictions are synthetic educational examples. "
        "They are not actual accident records or a substitute for a safety assessment."
    )
    st.subheader("Project objectives")
    objective_columns = st.columns(2)
    objective_columns[0].markdown(
        "- Classify causes as **Mechanical**, **Environmental**, or **Personnel**.\n"
        "- Classify safety events as **Unsafe Act** or **Unsafe Condition**."
    )
    objective_columns[1].markdown(
        "- Demonstrate categorical preprocessing with OneHotEncoder.\n"
        "- Present model results in an accessible safety dashboard."
    )
    st.subheader("Classification categories")
    cards = st.columns(5)
    for card, label in zip(
        cards,
        ["Mechanical", "Environmental", "Personnel", "Unsafe Act", "Unsafe Condition"],
    ):
        card.metric(label, "Category")


def render_prediction() -> None:
    """Render the interactive prediction page."""
    st.header("Accident prediction")
    st.write("Select the workplace conditions, then run both independent predictions.")
    inputs: dict[str, str] = {}
    columns = st.columns(2)
    for index, (label, (feature, options)) in enumerate(INPUT_OPTIONS.items()):
        with columns[index % 2]:
            inputs[feature] = st.selectbox(label, options, key=f"input_{feature}")

    if st.button("Predict accident", type="primary", width="stretch"):
        try:
            models = load_models()
            input_frame = pd.DataFrame([inputs], columns=FEATURES)
            cause_model = models["cause"]
            safety_model = models["safety"]
            cause_prediction = cause_model.predict(input_frame)[0]
            safety_prediction = safety_model.predict(input_frame)[0]

            result_columns = st.columns(2)
            with result_columns[0]:
                st.success(f"Accident cause: {cause_prediction}")
            with result_columns[1]:
                st.warning(f"Safety category: {safety_prediction}")

            probability_columns = st.columns(2)
            with probability_columns[0]:
                display_probability_chart(
                    cause_model.predict_proba(input_frame)[0],
                    cause_model.classes_,
                    "Accident cause probabilities",
                )
            with probability_columns[1]:
                display_probability_chart(
                    safety_model.predict_proba(input_frame)[0],
                    safety_model.classes_,
                    "Safety category probabilities",
                )

            st.subheader("Recommended safety actions")
            for recommendation in recommendations(inputs):
                st.markdown(f"- {recommendation}")
        except (FileNotFoundError, OSError, ValueError, AttributeError) as error:
            st.error(f"Prediction is unavailable: {error}")


def render_analytics() -> None:
    """Render saved distributions, metrics, and confusion matrices."""
    st.header("Analytics")
    st.write("These charts and metrics come from the executed training workflow.")
    chart_columns = st.columns(2)
    distribution_charts = [
        ("Accident cause distribution", "accident_cause_distribution.png"),
        ("Safety category distribution", "safety_category_distribution.png"),
    ]
    for column, (title, filename) in zip(chart_columns, distribution_charts):
        chart_path = RESULTS_DIR / filename
        with column:
            st.subheader(title)
            if chart_path.is_file():
                st.image(str(chart_path), caption=title, width="stretch")
            else:
                st.warning(f"Chart not found: {filename}")

    metrics_path = RESULTS_DIR / "model_comparison.csv"
    st.subheader("Model evaluation metrics")
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, width="stretch", hide_index=True)
        chart_data = metrics.copy()
        chart_data["Model"] = (
            chart_data["Target"] + " - " + chart_data["Algorithm"]
        )
        chart_data = chart_data.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1-score"]
        ]
        st.subheader("Model comparison")
        st.bar_chart(
            chart_data,
            y=["Accuracy", "Precision", "Recall", "F1-score"],
            x_label="Target and algorithm",
            y_label="Score",
        )
    else:
        st.warning("Model evaluation metrics are not available yet.")

    st.subheader("Confusion matrices")
    confusion_paths = sorted(RESULTS_DIR.glob("*_confusion_matrix.png"))
    if confusion_paths:
        columns = st.columns(2)
        for index, path in enumerate(confusion_paths):
            with columns[index % 2]:
                st.image(str(path), caption=path.stem.replace("_", " "), width="stretch")
    else:
        st.warning("No confusion matrix plots are available yet.")


def render_about() -> None:
    """Render project documentation and limitations."""
    st.header("About the project")
    st.subheader("Technologies used")
    st.write("Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn, and Joblib.")
    st.subheader("Machine Learning algorithms")
    st.write(
        "Decision Tree and Random Forest classifiers are trained independently for "
        "each target. Each saved pipeline includes OneHotEncoder preprocessing."
    )
    st.subheader("Dataset")
    st.write(
        "The project uses 1,000 reproducible synthetic records with eight categorical "
        "workplace-condition features and two target labels."
    )
    st.subheader("Limitations")
    st.write(
        "Synthetic labels reflect educational rules rather than observed incidents. "
        "The models should not be used for compliance decisions, incident investigation, "
        "or real-world risk assessment."
    )


def main() -> None:
    """Run the Streamlit application."""
    display_header()
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page",
            ["Home", "Accident Prediction", "Analytics", "About Project"],
        )
        st.divider()
        st.caption("Industrial safety education dashboard")

    if page == "Home":
        render_home()
    elif page == "Accident Prediction":
        render_prediction()
    elif page == "Analytics":
        render_analytics()
    else:
        render_about()


if __name__ == "__main__":
    main()