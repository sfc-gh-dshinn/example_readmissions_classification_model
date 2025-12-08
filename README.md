# Example Readmissions Model

## Summary
The `readmission_model.py` provides an example of building a simple xgboost classifier
on the [diabetes hospital readmissions dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).
Specific features include the following:
- Using out of time validation with `sktime` with randomly generated dates
- Using sklearn's [FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html) and [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to ensure that feature transformations are cleanly performed within each partitions training data
- Performance metrics such as lift, classification stats, confusion matrices, and feature importance

## Setup
```
$ python3 -m venv venv
$ venv/bin/pip install -r requirements.txt
```

## Run model
```
$ python readmission_model.py
```