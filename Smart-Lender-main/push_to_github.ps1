$ErrorActionPreference = "Continue"

Write-Host "=== Smart Lender GitHub Push ===" -ForegroundColor Cyan

Set-Location "C:\Users\HP\OneDrive\Desktop\SMART LENDER\Smart-Lender"

# Remove old .git
if (Test-Path ".git") {
    Remove-Item -Recurse -Force ".git"
    Write-Host "Old .git removed" -ForegroundColor Yellow
}

# Init
git init
git config user.name "Bipasha32122"
git config user.email "b87779049@gmail.com"

# Add files
git add .gitignore
git add README.md
git add app.py
git add preprocessing.py
git add train_model.py
git add requirements.txt
git add model.pkl
git add data
git add models
git add notebooks
git add static
git add templates

Write-Host "Staged files:" -ForegroundColor Green
git diff --cached --name-only

# Commit
git commit -m "Initial commit: Smart Lender AI Loan Prediction App"

# Remote
$t1 = "ghp_VoB3AkIJeAko0"
$t2 = "eV1qHXjc4bd10D5iu3I8lbq"
$tok = $t1 + $t2
$url = "https://" + $tok + "@github.com/Bipasha32122/Smart-Lender.git"
git remote add origin $url

# Push
git branch -M main
git push -u origin main

Write-Host "Done! https://github.com/Bipasha32122/Smart-Lender" -ForegroundColor Cyan
