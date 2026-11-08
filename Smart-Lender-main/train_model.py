"""
Smart Lender - Model Training Script
=====================================
Trains Decision Tree, Random Forest, KNN, and XGBoost models.
Compares all models and saves the best one as model.pkl.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Visualisation ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')           # Non-interactive backend (safe on servers)
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Scikit-learn ─────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
from xgboost import XGBClassifier

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join('data', 'loan_dataset.csv')
MODEL_DIR   = 'models'
STATIC_DIR  = os.path.join('static', 'images')
os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

print("=" * 65)
print("         SMART LENDER — AI Powered Loan Approval")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7]  Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"       Shape: {df.shape}")
print(f"       Columns: {list(df.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CLEANING & MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7]  Cleaning data …")

# Drop Loan_ID – not a feature
df.drop(columns=['Loan_ID'], inplace=True, errors='ignore')

print(f"       Nulls before:\n{df.isnull().sum().to_string()}")

# Fill categorical columns with mode
cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Fill numerical columns with median
num_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

print(f"       Nulls after:  {df.isnull().sum().sum()} (all resolved)")

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"       Duplicates removed: {before - len(df)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  ENCODING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7]  Encoding categorical features …")

le_dict = {}
encode_cols = ['Gender', 'Married', 'Dependents', 'Education',
               'Self_Employed', 'Property_Area', 'Loan_Status']

for col in encode_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

print("       Encoded:", encode_cols)

# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7]  Feature engineering …")

df['TotalIncome']       = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log']   = np.log1p(df['TotalIncome'])
df['LoanAmount_log']    = np.log1p(df['LoanAmount'])
df['EMI']               = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1e-5)
df['Balance_Income']    = df['TotalIncome'] - (df['EMI'] * 1000)

print("       New features: TotalIncome, TotalIncome_log, LoanAmount_log, EMI, Balance_Income")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  FEATURE SELECTION & SCALING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7]  Scaling features …")

feature_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area', 'TotalIncome_log',
                'LoanAmount_log', 'EMI', 'Balance_Income']

X = df[feature_cols]
y = df['Loan_Status']

# Ensure no NaN values remain before scaling
X = X.fillna(X.median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)

print(f"       Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7]  Training models …\n")

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                            random_state=42, n_jobs=-1),
    'KNN':           KNeighborsClassifier(n_neighbors=7, metric='euclidean'),
    'XGBoost':       XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                   eval_metric='logloss',
                                   random_state=42, verbosity=0)
}

results = {}
roc_data = {}

DIVIDER = "-" * 55

for name, model in models.items():
    print(f"{'─'*55}")
    print(f"  Model: {name}")
    print(DIVIDER)

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cv   = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy').mean()

    results[name] = {'Accuracy': acc, 'Precision': prec,
                     'Recall': rec, 'F1': f1, 'AUC': auc, 'CV': cv,
                     'model': model}
    roc_data[name] = roc_curve(y_test, y_proba)

    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  CV (5-fold): {cv:.4f}")
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")

    # Confusion Matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    ax.set_title(f'{name} — Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, f'cm_{name.replace(" ", "_")}.png'), dpi=150)
    plt.close()

# ─── ROC Curves (combined) ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
for (name, (fpr, tpr, _)), color in zip(roc_data.items(), colors):
    ax.plot(fpr, tpr, lw=2, color=color,
            label=f"{name} (AUC={results[name]['AUC']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'roc_curves.png'), dpi=150)
plt.close()

# ─── Model Comparison ────────────────────────────────────────────────────────
comparison = pd.DataFrame({k: {m: v for m, v in v.items() if m != 'model'}
                           for k, v in results.items()}).T
print("\n" + "=" * 65)
print("  MODEL COMPARISON")
print("=" * 65)
print(comparison[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'CV']].to_string())

fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
comparison[metrics].plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_xticklabels(comparison.index, rotation=25, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(STATIC_DIR, 'model_comparison.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# 7.  SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7]  Saving best model …")

best_name  = comparison['F1'].idxmax()
best_model = results[best_name]['model']

print(f"       Best model: {best_name}  (F1={results[best_name]['F1']:.4f})")

# Feature importance (if supported)
if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    imp.plot(kind='bar', ax=ax, color='#1a73e8', edgecolor='black')
    ax.set_title(f'Feature Importance — {best_name}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(STATIC_DIR, 'feature_importance.png'), dpi=150)
    plt.close()

# Pickle artefacts
payload = {
    'model':        best_model,
    'scaler':       scaler,
    'le_dict':      le_dict,
    'feature_cols': feature_cols,
    'model_name':   best_name
}

with open('model.pkl', 'wb') as f:
    pickle.dump(payload, f)

with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
    pickle.dump(payload, f)

print("       model.pkl saved ✓")
print("\n" + "=" * 65)
print("  Training complete! Run app.py to start the Flask server.")
print("=" * 65)
