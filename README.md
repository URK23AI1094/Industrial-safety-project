# Accident Cause Classification System

## Project introduction

This beginner-friendly Python project demonstrates how machine learning can
classify synthetic workplace accident scenarios. It provides two independent
predictions:

1. **Accident cause:** Mechanical, Environmental, or Personnel
2. **Safety category:** Unsafe Act or Unsafe Condition

The dataset is synthetic educational data. It is not a collection of actual
accident records and must not be used as evidence for real-world safety,
compliance, or incident-investigation decisions.

## Problem statement

Workplace safety teams may need to organize incident information into
consistent categories. This project uses categorical workplace conditions such
as machine status, floor condition, worker behavior, and PPE usage to
demonstrate automated classification.

## Objectives

- Generate 1,000 reproducible synthetic workplace accident records.
- Explore the data and save distribution charts.
- Train separate Decision Tree and Random Forest pipelines for both targets.
- Compare accuracy, precision, recall, and F1-score.
- Provide a Streamlit dashboard for predictions, probabilities, recommendations,
  and analytics.

## Technology stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Folder structure

```text
Accident_Cause_Classification/
├── dataset/
│   └── accident_data.csv
├── models/
│   ├── accident_cause_best_model.joblib
│   └── safety_category_best_model.joblib
├── results/
│   ├── model_comparison.csv
│   ├── *_distribution.png
│   ├── feature_distributions.png
│   └── *_confusion_matrix.png
├── src/
│   ├── generate_dataset.py
│   ├── data_analysis.py
│   └── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Dataset description

The generated CSV contains 1,000 records and these input features:

- `Machine_Status`: Normal or Faulty
- `Floor_Condition`: Dry or Wet
- `Worker_Behavior`: Safe or Unsafe
- `PPE_Usage`: Yes or No
- `Lighting`: Good or Poor
- `Training_Status`: Trained or Untrained
- `Maintenance_Status`: Good or Poor
- `Machine_Guarding`: Proper or Missing

The target columns are `Accident_Cause` and `Safety_Category`. The generator
uses documented deterministic rules and a fixed random seed. Every generated
record contains at least one unsafe act or unsafe condition.

## Machine learning workflow

1. Generate the synthetic CSV using NumPy and Pandas.
2. Inspect missing values, duplicates, statistics, and feature distributions.
3. Split each target independently into stratified 80% training and 20% test
   sets.
4. One-hot encode categorical features inside each Scikit-learn pipeline.
5. Train independent Decision Tree and Random Forest classifiers.
6. Calculate accuracy, precision, recall, and weighted F1-score.
7. Save comparison metrics and confusion-matrix charts.
8. Save the best pipeline for each target with Joblib.
9. Load the pipelines in Streamlit; they accept the original categorical
   feature columns directly.

## Step-by-step setup and execution

These commands are for Windows 11 PowerShell. Run them in the order shown from
the project root. Using `.venv\Scripts\python.exe` explicitly avoids the
common error where `streamlit` is not recognized in PowerShell.

### Step 1: Open the project directory

```powershell
cd "D:\Industrial safety project\Accident_Cause_Classification"
```

Confirm that the project files are present:

```powershell
Test-Path .\app.py
Test-Path .\requirements.txt
```

Both commands should return `True`.

### Step 2: Create the virtual environment

Run this only if `.venv` does not already exist:

```powershell
python -m venv .venv
```

If `python` is not recognized, install Python 3.10 or newer and enable
**Add Python to PATH** during installation.

### Step 3: Activate the virtual environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The PowerShell prompt should show `(.venv)`. Activation is optional when using
the explicit `.venv\Scripts\python.exe` commands below.

### Step 4: Install or update dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The final command should print `No broken requirements found`.

### Step 5: Generate the dataset

```powershell
.\.venv\Scripts\python.exe .\src\generate_dataset.py
```

This creates `dataset\accident_data.csv`.

### Step 6: Run data analysis

```powershell
.\.venv\Scripts\python.exe .\src\data_analysis.py
```

This checks the data and saves charts in `results\`.

### Step 7: Train and save the models

```powershell
.\.venv\Scripts\python.exe .\src\train_model.py
```

This saves evaluation metrics to `results\model_comparison.csv` and the
reusable Joblib pipelines to `models\`.

### Step 8: Start the Streamlit dashboard

Use the module form rather than `streamlit run app.py`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

Open the displayed URL, normally `http://localhost:8501`.

If port 8501 is already in use, start the dashboard on another port:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py --server.port 8502
```

### Step 9: Stop the dashboard

Return to the PowerShell window running Streamlit and press:

```text
Ctrl+C
```

### Quick rerun after setup

After the environment and dependencies are already configured, use:

```powershell
cd "D:\Industrial safety project\Accident_Cause_Classification"
.\.venv\Scripts\python.exe .\src\generate_dataset.py
.\.venv\Scripts\python.exe .\src\data_analysis.py
.\.venv\Scripts\python.exe .\src\train_model.py
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

### Troubleshooting

- **`streamlit` is not recognized:** use
  `.\.venv\Scripts\python.exe -m streamlit run .\app.py`.
- **Activation is blocked:** run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.
- **Model files are missing:** run the dataset generation and model training
  commands again.
- **Port 8501 is unavailable:** use the port 8502 command above or close the
  existing Streamlit process.

## Limitations

- The records and labels are synthetic and do not represent measured workplace
  risk.
- Perfect or high evaluation scores on synthetic data do not establish
  real-world model performance.
- The application does not replace professional safety inspections, training,
  hazard analysis, reporting, or regulatory compliance processes.
- The recommendations are educational examples based only on the selected
  categorical inputs.
