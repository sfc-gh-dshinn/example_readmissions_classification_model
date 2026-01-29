# Example Readmissions Model

## Summary
This project provides an example of building an XGBoost classifier for predicting hospital readmissions using the [diabetes hospital readmissions dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

### Features
- **Out-of-time validation** with custom time-series cross-validation using randomly generated dates
- **Hyperparameter tuning** with FLAML using CFO (Cost-Frugal Optimization) algorithm
- **Clean feature transformations** using sklearn's [FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) and [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- **Model interpretability** with SHAP analysis and partial dependence plots
- **Performance metrics** including lift tables, classification reports, confusion matrices, and feature importance

## Project Structure
```
├── readmission_model.py      # Main model training script
├── readmission_analysis.ipynb # Jupyter notebook with SHAP analysis
├── model_utils.py            # Shared utility functions
├── requirements.txt          # Python dependencies
└── shap_results/             # Output directory for visualizations
```

## Setup
```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Usage

### Run the model training script
```bash
venv/bin/python readmission_model.py
```

### Run the analysis notebook
```bash
venv/bin/jupyter notebook readmission_analysis.ipynb
```

## Snowflake Container Runtime
This project is compatible with Snowflake Container Runtime. The dependencies in `requirements.txt` are aligned with pre-installed packages, so no `pip install` is required when running in that environment.
