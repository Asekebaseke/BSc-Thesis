# BSc-Thesis
(BSc. Thesis) Comparison of ML models for ICR performance prediction.
Project Title: Machine Learning models for prediction of Roadheader Performance (ICR)
#Sandbak (1985) - Synthetic Modeling and Prediction of ICR

Author: Askar Omirzak
University: Nazarbayev University, 2025

---

Overview:
This project aims to predict the Instantaneous Cutting Rate (ICR) of a roadheader machine using geological parameters. The methodology includes data normalization, polynomial feature generation, and training multiple machine learning models on synthetic data to evaluate model robustness.

---

Dataset:
The dataset is derived from the Sandbak (1985) study and includes the following features:
- RQD: Rock Quality Designation
- Rock-Class: Classification of rock type
- UCS: Uniaxial Compressive Strength
- PC2: Pick Consumption (index)
- ICR: Target variable, Instantaneous Cutting Rate

Missing values are removed during preprocessing to ensure clean input for modeling.

---

Code Structure:
1. EDA(Sandbak).py – Performs exploratory data analysis, visualizes distributions, correlations, etc.
2. Sandbak(ML).py – Main modeling script that:
   - Normalizes and transforms features
   - Generates synthetic training data via:
     • Ridge Regression Labeling
     • Gaussian Noise Labeling
     • Random Forest Labeling
   - Trains models (Ridge, Lasso, Random Forest)
   - Performs 5-fold cross-validation on synthetic data
   - Evaluates performance on original data

---

How to Run:
1. Ensure Python 3.7+ is installed.
2. Install dependencies using pip:
   pip install -r requirements.txt

3. Run EDA:
   python "EDA(Sandbak).py"

4. Run Modeling:
   python "Sandbak(ML).py"

---

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap

(Use `pip install` to get the above libraries if not already installed)

---

Outputs:
- Performance metrics (R², MSE) for each model are printed to terminal
- Helps identify the best synthetic data strategy and regression model for ICR prediction

---

Notes:
- Make sure the Sandbak dataset (Excel format) is placed in the same directory as the script with the correct filename.
- Polynomial feature generation uses degree=2 by default.

---
