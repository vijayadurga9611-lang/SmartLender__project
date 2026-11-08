"""
Smart Lender - Realistic Loan Dataset Generator (Pure Python)
Generates 1000 records with realistic distributions
"""
import random
import csv

random.seed(42)

def generate_dataset(n=1000):
    records = []
    
    for i in range(n):
        # Generate realistic data
        gender = random.choices(['Male', 'Female'], weights=[80, 20])[0]
        married = random.choices(['Yes', 'No'], weights=[65, 35])[0]
        dependents = random.choices(['0', '1', '2', '3+'], weights=[57, 17, 16, 10])[0]
        education = random.choices(['Graduate', 'Not Graduate'], weights=[78, 22])[0]
        self_employed = random.choices(['Yes', 'No'], weights=[14, 86])[0]
        
        # Income generation
        applicant_income = int(random.lognormvariate(8.3, 0.6))
        applicant_income = max(1500, min(81000, applicant_income))
        
        if married == 'Yes':
            coapplicant_income = int(random.lognormvariate(7.5, 0.8))
            coapplicant_income = max(0, min(41667, coapplicant_income))
        else:
            coapplicant_income = 0
        
        # Loan details
        total_income = applicant_income + coapplicant_income
        loan_amount = int((total_income * random.uniform(0.5, 3.5)) / 1000)
        loan_amount = max(9, min(700, loan_amount))
        
        loan_term = random.choices([12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                                   weights=[1, 1, 2, 2, 4, 4, 4, 4, 70, 8])[0]
        
        credit_history = random.choices([1.0, 0.0], weights=[84, 16])[0]
        property_area = random.choices(['Urban', 'Semiurban', 'Rural'], weights=[38, 37, 25])[0]
        
        # Calculate approval probability
        approval_score = 0.50
        if credit_history == 1.0:
            approval_score += 0.40
        else:
            approval_score -= 0.30
        if education == 'Graduate':
            approval_score += 0.10
        if married == 'Yes':
            approval_score += 0.05
        if property_area == 'Semiurban':
            approval_score += 0.08
        elif property_area == 'Urban':
            approval_score += 0.04
        if self_employed == 'No':
            approval_score += 0.05
        
        income_ratio = total_income / (loan_amount * 1000 + 1)
        approval_score += min(0.15, income_ratio * 0.05)
        approval_score = max(0.10, min(0.95, approval_score))
        
        loan_status = 'Y' if random.random() < approval_score else 'N'
        
        # Add missing values randomly
        if random.random() < 0.02:
            gender = ''
        if random.random() < 0.01:
            married = ''
        if random.random() < 0.02:
            dependents = ''
        if random.random() < 0.03:
            self_employed = ''
        if random.random() < 0.04:
            loan_amount = ''
        if random.random() < 0.02:
            loan_term = ''
        if random.random() < 0.08:
            credit_history = ''
        
        record = {
            'Loan_ID': f'LP{100001 + i}',
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area,
            'Loan_Status': loan_status
        }
        records.append(record)
    
    return records

# Generate dataset
print("Generating dataset...")
data = generate_dataset(1000)

# Write to CSV
with open('loan_dataset.csv', 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 
                  'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 
                  'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                  'Property_Area', 'Loan_Status']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"✓ Dataset generated: 1000 records")
print(f"✓ File saved: loan_dataset.csv")

# Statistics
approved = sum(1 for d in data if d['Loan_Status'] == 'Y')
print(f"\nLoan Status Distribution:")
print(f"  Approved (Y): {approved} ({approved/10}%)")
print(f"  Rejected (N): {1000-approved} ({(1000-approved)/10}%)")
