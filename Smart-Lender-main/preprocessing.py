"""
Smart Lender — Data Preprocessing Module
=========================================
Contains the full preprocessing pipeline:
  1. Missing value handling
  2. Label encoding
  3. Feature engineering
  4. Scaling
  5. Train/test split

Can be imported by train_model.py or used standalone.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"[Load]    Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Categorical → mode
      - Numerical   → median
    """
    df = df.copy()

    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
    num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']

    print("\n[Missing] Before:")
    print(df[cat_cols + num_cols].isnull().sum().to_string())

    for col in cat_cols:
        mode = df[col].mode()[0]
        df[col].fillna(mode, inplace=True)
        print(f"  {col:20s} → filled with mode: '{mode}'")

    for col in num_cols:
        median = df[col].median()
        df[col].fillna(median, inplace=True)
        print(f"  {col:20s} → filled with median: {median}")

    total_null = df.isnull().sum().sum()
    print(f"\n[Missing] After: {total_null} nulls remain")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    print(f"\n[Dedup]   Removed {before - len(df)} duplicates. Rows: {len(df)}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label encode all categorical columns.
    Returns (encoded_df, label_encoder_dict)
    """
    df = df.copy()
    le_dict = {}
    encode_cols = ['Gender', 'Married', 'Dependents', 'Education',
                   'Self_Employed', 'Property_Area', 'Loan_Status']

    print("\n[Encode]  Label encoding:")
    for col in encode_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        print(f"  {col:20s} → classes: {list(le.classes_)}")

    return df, le_dict


# ──────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
      - TotalIncome        = ApplicantIncome + CoapplicantIncome
      - TotalIncome_log    = log1p(TotalIncome)
      - LoanAmount_log     = log1p(LoanAmount)
      - EMI                = LoanAmount / Loan_Amount_Term
      - Balance_Income     = TotalIncome - EMI * 1000
    """
    df = df.copy()

    df['TotalIncome']     = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
    df['LoanAmount_log']  = np.log1p(df['LoanAmount'])
    df['EMI']             = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-5)
    df['Balance_Income']  = df['TotalIncome'] - (df['EMI'] * 1000)

    print("\n[Feature] Engineered 5 new features:")
    print("  TotalIncome, TotalIncome_log, LoanAmount_log, EMI, Balance_Income")
    print(f"  Final shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area',
    'TotalIncome_log', 'LoanAmount_log', 'EMI', 'Balance_Income'
]


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Apply StandardScaler to feature matrix."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"\n[Scale]   StandardScaler applied. Shape: {X_scaled.shape}")
    return X_scaled, scaler


# ──────────────────────────────────────────────────────────────────────────────
def split_data(X, y, test_size=0.20, random_state=42):
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    print(f"\n[Split]   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"          Approval rate — Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────────────────
def full_pipeline(path: str):
    """Run the complete preprocessing pipeline end-to-end."""
    print("=" * 55)
    print("  Smart Lender — Preprocessing Pipeline")
    print("=" * 55)

    df = load_data(path)
    df.drop(columns=['Loan_ID'], inplace=True, errors='ignore')
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df, le_dict = encode_features(df)
    df = engineer_features(df)

    X = df[FEATURE_COLS]
    y = df['Loan_Status']

    X_scaled, scaler = scale_features(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLS)

    X_train, X_test, y_train, y_test = split_data(X_scaled_df, y)

    print("\n✓ Preprocessing complete!")
    print("=" * 55)
    return X_train, X_test, y_train, y_test, scaler, le_dict


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os
    data_path = os.path.join('data', 'loan_dataset.csv')
    X_train, X_test, y_train, y_test, scaler, le_dict = full_pipeline(data_path)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")
