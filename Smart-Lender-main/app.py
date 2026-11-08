"""
Smart Lender — Flask Web Application
=====================================
Routes:
    GET  /           → Home page
    GET  /predict    → Prediction form
    POST /predict    → Process form & return result
    GET  /about      → About page
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'smartlender_secret_2024'

# ─── Load model artefacts ─────────────────────────────────────────────────────
MODEL_PATH = 'model.pkl'

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

payload = load_model()

# ─── Helpers ──────────────────────────────────────────────────────────────────
PROPERTY_AREA_MAP = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
EDUCATION_MAP     = {'Graduate': 0, 'Not Graduate': 1}
DEPENDENTS_MAP    = {'0': 0, '1': 1, '2': 2, '3+': 3}

def encode_input(form):
    """Convert raw form values to model-ready feature vector."""
    gender         = 1 if form['gender'] == 'Male' else 0
    married        = 1 if form['married'] == 'Yes' else 0
    dependents     = DEPENDENTS_MAP.get(form['dependents'], 0)
    education      = EDUCATION_MAP.get(form['education'], 0)
    self_employed  = 1 if form['self_employed'] == 'Yes' else 0
    applicant_inc  = float(form['applicant_income'])
    coapplicant_inc= float(form['coapplicant_income'])
    loan_amount    = float(form['loan_amount'])
    loan_term      = float(form['loan_term'])
    credit_history = float(form['credit_history'])
    property_area  = PROPERTY_AREA_MAP.get(form['property_area'], 1)

    total_income     = applicant_inc + coapplicant_inc
    total_income_log = np.log1p(total_income)
    loan_amount_log  = np.log1p(loan_amount)
    emi              = loan_amount / (loan_term + 1e-5)
    balance_income   = total_income - (emi * 1000)

    features = [gender, married, dependents, education, self_employed,
                applicant_inc, coapplicant_inc, loan_amount, loan_term,
                credit_history, property_area, total_income_log,
                loan_amount_log, emi, balance_income]
    return features


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    # ── POST: process the form ────────────────────────────────────────────────
    if payload is None:
        flash("Model not found. Please run train_model.py first.", "danger")
        return redirect(url_for('predict'))

    try:
        features = encode_input(request.form)
        scaler   = payload['scaler']
        model    = payload['model']

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]

        approved       = bool(prediction == 1)
        approval_pct   = round(probability[1] * 100, 1)
        rejection_pct  = round(probability[0] * 100, 1)
        model_name     = payload.get('model_name', 'ML Model')

        # Build a friendly summary dict for the template
        form_data = {
            'gender':            request.form.get('gender'),
            'married':           request.form.get('married'),
            'dependents':        request.form.get('dependents'),
            'education':         request.form.get('education'),
            'self_employed':     request.form.get('self_employed'),
            'applicant_income':  request.form.get('applicant_income'),
            'coapplicant_income':request.form.get('coapplicant_income'),
            'loan_amount':       request.form.get('loan_amount'),
            'loan_term':         request.form.get('loan_term'),
            'credit_history':    'Good' if request.form.get('credit_history') == '1' else 'Poor',
            'property_area':     request.form.get('property_area'),
        }

        return render_template(
            'result.html',
            approved=approved,
            approval_pct=approval_pct,
            rejection_pct=rejection_pct,
            model_name=model_name,
            form_data=form_data
        )

    except Exception as e:
        flash(f"Prediction error: {str(e)}", "danger")
        return redirect(url_for('predict'))


@app.route('/about')
def about():
    return render_template('about.html')


# ─── Error handlers ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
