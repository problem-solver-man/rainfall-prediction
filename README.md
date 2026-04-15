# 🌧️ Rainfall Prediction using Machine Learning

![Python](https://img.shields.io/badge/Python 3.9.13-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview
This project predicts whether it will rain based on weather conditions using Machine Learning models.  
It includes data preprocessing, exploratory data analysis (EDA), model training, and a Streamlit-based web application for real-time predictions.

---

## 🚀 Features
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Handling missing values
- Handling imbalanced dataset using oversampling
- Feature scaling using StandardScaler
- Multiple ML models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - XGBoost Classifier
- Model evaluation using ROC-AUC, confusion matrix, and classification report
- Interactive GUI using Streamlit

---

## 📂 Project Structure

rainfall-prediction/
│── Rainfall.csv    # Dataset
│── main.py         # Data processing and model training
│── app.py          # Streamlit application

---

## Prerequisites

- Python 3.9.13
- Git

Ensure both are installed and available in your system PATH.

---

## ⚙️ Installation & Run

```bash
git clone https://github.com/problem-solver-man/rainfall-prediction.git
cd rainfall-prediction
py -3.9 -m venv rainenv
rainenv\Scripts\activate
pip install -r requirements.txt
python main.py 
streamlit run app.py
```

## 👨‍💻 Author
Vishesh Karn

---

⭐ If you like this project, consider giving it a star!
