"""
Smart Lender - Realistic Loan Dataset Generator
Generates 1000 records with realistic distributions
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

genders = np.random.choice(['Male', 'Female'], n, p=[0.80, 0.20])
married = np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35])
dependents = np.random.choice(['0', '1', '2', '3+'], n, p=[0.57, 0.17, 0.16, 0.10])
education = np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.78, 0.22])
self_employed = np.random.choice(['Yes', 'No'], n, p=[0.14, 0.86])

applicant_income = np.random.lognormal(mean=8.3, sigma=0.6, size=n).astype(int)
applicant_income = np.clip(applicant_income, 1500, 81000)

coapplicant_income = np.where(
    married == 'Yes',
    np.random.lognormal(mean=7.5, sigma=0.8, size=n).astype(int),
    np.random.choice([0], n)
)
coapplicant_income = np.clip(coapplicant_income, 0, 41667)

loan_amount = ((applicant_income + coapplicant_income) * 
               np.random.uniform(0.5, 3.5, n) / 1000).astype(int)
loan_amount = np.clip(loan_amount, 9, 700)

loan_amount_term = np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                                     n, p=[0.01, 0.01, 0.02, 0.02, 0.04, 0.04, 0.04, 0.04, 0.70, 0.08])

credit_history = np.random.choice([1.0, 0.0], n, p=[0.84, 0.16])

property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.38, 0.37, 0.25])

# Realistic loan approval logic
approval_prob = np.zeros(n)
approval_prob += np.where(credit_history == 1.0, 0.45, -0.30)
approval_prob += np.where(education == 'Graduate', 0.10, -0.05)
approval_prob += np.where(married == 'Yes', 0.05, 0.0)
approval_prob += np.where(property_area == 'Semiurban', 0.08, 0.0)
approval_prob += np.where(property_area == 'Urban', 0.04, 0.0)
approval_prob += np.where(self_employed == 'No', 0.05, -0.02)
income_ratio = (applicant_income + coapplicant_income) / (loan_amount * 1000 + 1)
approval_prob += np.clip(income_ratio * 0.05, -0.10, 0.15)
approval_prob = np.clip(approval_prob + 0.55, 0.10, 0.95)
loan_status = np.where(np.random.uniform(0, 1, n) < approval_prob, 'Y', 'N')

# Introduce realistic missing values (~5-10%)
def add_missing(arr, pct=0.05):
    mask = np.random.random(n) < pct
    arr = arr.astype(object)
    arr[mask] = np.nan
    return arr

gender_col = add_missing(genders, 0.02)
married_col = add_missing(married, 0.01)
dependents_col = add_missing(dependents, 0.02)
self_employed_col = add_missing(self_employed, 0.03)
loan_amount_col = add_missing(loan_amount.astype(float), 0.04)
loan_term_col = add_missing(loan_amount_term.astype(float), 0.02)
credit_col = add_missing(credit_history, 0.08)

loan_ids = ['LP' + str(100001 + i) for i in range(n)]

df = pd.DataFrame({
    'Loan_ID': loan_ids,
    'Gender': gender_col,
    'Married': married_col,
    'Dependents': dependents_col,
    'Education': education,
    'Self_Employed': self_employed_col,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount_col,
    'Loan_Amount_Term': loan_term_col,
    'Credit_History': credit_col,
    'Property_Area': property_area,
    'Loan_Status': loan_status
})

df.to_csv('loan_dataset.csv', index=False)
print(f"Dataset generated: {df.shape}")
print(df['Loan_Status'].value_counts())
print(f"\nMissing values:\n{df.isnull().sum()}")
print("\nFirst 5 rows:")
print(df.head())
