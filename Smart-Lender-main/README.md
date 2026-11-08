# 🏦 Smart Lender — AI Powered Loan Approval Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-f89820?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A complete end-to-end machine learning web application that predicts loan approval eligibility using applicant financial and personal data — built with Python, Flask, and XGBoost.

---

## 📋 Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [ML Models](#ml-models)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Run](#run)
- [Usage](#usage)
- [Future Scope](#future-scope)
- [Author](#author)

---

## 📌 Project Description

Smart Lender is an AI-powered loan approval prediction system. It takes applicant information (personal, financial, property) and predicts whether a loan application is likely to be **approved or rejected** using machine learning models trained on 1,000 realistic loan records.

The project covers the complete ML pipeline:
1. Data collection & generation
2. Exploratory Data Analysis (EDA)
3. Data preprocessing & feature engineering
4. Training 4 ML models & comparing them
5. Deploying the best model via a Flask web app

---

## ✨ Features

- 🤖 **AI Prediction** — Instant loan eligibility decision
- 📊 **4 ML Models** — Decision Tree, Random Forest, KNN, XGBoost
- 🏆 **Best Model Auto-Selected** — Based on F1 Score
- 💡 **15 Engineered Features** — Including EMI, Balance Income, Log-transformed income
- 🎨 **Professional UI** — Glassmorphism banking theme with Bootstrap 5
- 📱 **Responsive** — Works on desktop, tablet, and mobile
- 📈 **Visual Analytics** — Confusion matrices, ROC curves, feature importance charts
- 🔒 **Secure** — Input validation and error handling

---

## 🛠 Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Web Framework | Flask 3.0 |
| ML Library | Scikit-learn 1.5, XGBoost 2.0 |
| Data Processing | Pandas 2.2, NumPy 1.26 |
| Visualisation | Matplotlib 3.9, Seaborn 0.13 |
| Frontend | Bootstrap 5, Font Awesome, Google Fonts |
| Model Saving | Pickle |

---

## 📂 Dataset

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0, 1, 2, 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income (₹) |
| CoapplicantIncome | Numerical | Co-applicant income (₹) |
| LoanAmount | Numerical | Loan amount (₹ thousands) |
| Loan_Amount_Term | Numerical | Term in months |
| Credit_History | Binary | 1 = Good, 0 = Poor |
| Property_Area | Categorical | Urban / Semiurban / Rural |
| Loan_Status | Target | Y = Approved, N = Rejected |

- **Records**: 1,000
- **Missing values**: Introduced realistically (2–8%)
- **Approval rate**: ~86%

---

## 🤖 ML Models

| Model | Description |
|-------|-------------|
| Decision Tree | Max depth 6, interpretable |
| Random Forest | 200 estimators, ensemble |
| KNN | K=7, Euclidean distance |
| XGBoost | 200 estimators, learning rate 0.05 |

Evaluated on: **Accuracy, Precision, Recall, F1 Score, ROC-AUC, 5-fold CV**

---

## 📁 Folder Structure

```
Smart-Lender/
├── app.py                    # Flask web application
├── train_model.py            # Model training script
├── preprocessing.py          # Data preprocessing module
├── model.pkl                 # Saved best model (generated)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/
│   ├── loan_dataset.csv      # Dataset (1000 records)
│   └── generate_dataset.py   # Dataset generator
│
├── models/
│   └── model.pkl             # Model backup
│
├── notebooks/
│   └── EDA_SmartLender.ipynb # Jupyter EDA notebook
│
├── templates/
│   ├── base.html             # Base template
│   ├── index.html            # Home page
│   ├── predict.html          # Prediction form
│   ├── result.html           # Prediction result
│   ├── about.html            # About page
│   ├── 404.html              # 404 error page
│   └── 500.html              # 500 error page
│
└── static/
    ├── css/
    │   └── style.css         # Premium glassmorphism CSS
    ├── js/
    │   └── main.js           # JavaScript
    └── images/               # Generated model charts
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Smart-Lender.git
cd Smart-Lender
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Run

### Step 1 — Train the Model
```bash
python train_model.py
```
This will:
- Preprocess the dataset
- Train all 4 ML models
- Print metrics comparison
- Save `model.pkl`
- Generate charts in `static/images/`

### Step 2 — Launch the Flask App
```bash
python app.py
```

Open your browser and go to: **http://localhost:5000**

---

## 💻 Usage

1. Navigate to the **Predict** page
2. Fill in your personal and financial details
3. Click **Predict Loan Eligibility**
4. View the AI decision with approval probability

---

## 🔮 Future Scope

- [ ] Add SHAP values for model explainability
- [ ] Integrate real bank API for live data
- [ ] Add user authentication and history tracking
- [ ] Deploy on Heroku / AWS / Render
- [ ] Add PDF report generation
- [ ] Support for multi-language UI
- [ ] Add LIME local explanations
- [ ] Retrain model on larger dataset

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: youremail@example.com

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> ⚠️ **Disclaimer**: This project is for educational purposes only. The predictions do not constitute actual financial advice.
