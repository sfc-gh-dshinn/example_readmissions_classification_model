import pandas as pd
import numpy as np

print("Loading CSV...")
df = pd.read_csv('diabetic_data_with_dates.csv')

print(f"Original shape: {df.shape}")
print(f"Original missing values: {df.isnull().sum().sum()}")

# Set random seed for reproducibility
np.random.seed(42)

# Add missing values to categorical features (about 2% of rows)
categorical_columns = ['race', 'gender', 'age', 'medical_specialty', 'payer_code', 
                       'A1Cresult', 'max_glu_serum', 'insulin', 'diabetesMed']
n_rows = len(df)
missing_pct = 0.02

for col in categorical_columns:
    if col in df.columns:
        n_missing = int(n_rows * missing_pct)
        missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan
        print(f"Added {n_missing} missing values to {col}")

# Add missing values to numerical features (about 1.5% of rows)
numerical_columns = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                     'num_medications', 'number_diagnoses']
missing_pct_num = 0.015

for col in numerical_columns:
    if col in df.columns:
        n_missing = int(n_rows * missing_pct_num)
        missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
        df.loc[missing_indices, col] = np.nan
        print(f"Added {n_missing} missing values to {col}")

print(f"\nFinal shape: {df.shape}")
print(f"Total missing values added: {df.isnull().sum().sum()}")
print(f"\nMissing values by column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Save the modified CSV
print("\nSaving modified CSV...")
df.to_csv('diabetic_data_with_dates.csv', index=False)
print("Done!")
