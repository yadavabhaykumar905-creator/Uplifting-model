# Uplift Predictor: Who Responds BECAUSE of the Email?

An end-to-end Machine Learning web application built with **Streamlit** and **scikit-learn** that implements **Uplift Modeling (Incremental Response Modeling)** to optimize marketing campaigns by predicting causal treatment effects.

---

## 🚀 Overview

Traditional predictive models answer: *"Which customers are most likely to buy or visit?"* 
The problem? They often target **"Sure Things"** (customers who would buy anyway) and **"Lost Causes"** (customers who won't buy regardless of outreach), wasting marketing budget.

**Uplift Modeling** answers a better question: *"Which customers will take action **specifically because** they received the email?"* (Persuadables). 

This project uses the benchmark **Hillstrom MineThatData Email Experiment dataset** (64,000 randomized customer records) to train a causal machine learning model and deploy it via an interactive web interface.

---

## 🧠 Methodology: T-Learner

This project implements a **T-Learner (Two-Model Approach)** powered by **Gradient Boosting**:
1. **Treatment Model ($m_1$)**: Trained exclusively on customers who received the email to estimate probability of visiting ($P_1$).
2. **Control Model ($m_0$)**: Trained exclusively on customers who did *not* receive the email to estimate baseline probability of visiting ($P_0$).
3. **Uplift Score**: Calculated per customer as:
   $$\text{Uplift} = P_1 - P_0$$

---

## 🛠️ Project Structure

```text
├── app.py              # Interactive Streamlit application
├── hillstrom.csv       # MineThatData email marketing dataset
├── uplift_model.pkl    # Serialized scikit-learn models and metadata bundle
└── requirements.txt    # Python package dependencies
```

---

## 📦 Installation & Setup

1. **Clone or download** this repository into your local directory.
2. **Install dependencies** (ensure you match the exact scikit-learn version used during training):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

---

## 💡 How to Use the App

The Streamlit web interface provides two main workflows:

1. **Score One Customer (Tab A):**
   * Use the sidebar/form inputs to adjust customer features such as recency of purchase, past spending amount, history segment, channel, area type, and prior category purchases.
   * Click **Predict uplift** to view the estimated visit chance *with* the email, *without* the email, and the resulting net uplift score with actionable targeting recommendations.

2. **Score Many Customers from CSV (Tab B):**
   * Download the CSV template provided in the app.
   * Upload your custom batch CSV file containing matching feature columns.
   * The app automatically computes individual uplift scores, ranks customers from highest to lowest incremental impact, and lets you download the scored results (`uplift_scores.csv`).

---

## ⚠️ Limitations & Notes

* **Noise Sensitivity:** Individual uplift estimates can be noisy due to low baseline website visit rates. Use them as a relative ranking signal rather than absolute guarantees.
* **Feature Scope:** The model relies strictly on the tabular features present in the Hillstrom training set (e.g., history, recency, channel, zip code type).
* **Target Metric:** The model predicts **website visits**, not final purchases.