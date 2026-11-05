  
# Uplift Predictor: Who Responds BECAUSE of the Email? 🎯

A machine learning web application that implements **Uplift Modeling (Incremental Response Modeling)**. Unlike traditional propensity models that predict *who will convert overall*, this model predicts **who will convert specifically because of a marketing intervention (receiving an email)**, allowing organizations to target only the persuadable customers and avoid wasting marketing budget on "sure things" or "lost causes".

---

## 🚀 Key Features
* **Individual Customer Scoring:** Interactive sliders and dropdowns to score a single customer and visualize their expected visit probability *With Email*, *Without Email*, and the resulting **Uplift**.
* **Batch Processing via CSV:** Upload a customer dataset CSV file, automatically compute uplift scores, rank customers from highest to lowest uplift, and download the full scored results[cite: 1].
* **T-Learner Architecture:** Uses a Two-Model (T-Learner) framework powered by Gradient Boosting (`scikit-learn`) where one model estimates treatment outcomes ($m_1$) and the other estimates control outcomes ($m_0$)[cite: 1].

---

## 🛠️ Project Structure
```text
├── app.py              # Main Streamlit web application
├── uplift_model.pkl    # Pre-trained T-Learner model bundle (joblib)
├── hillstrom.csv       # Benchmark dataset (MineThatData Email Experiment)
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
