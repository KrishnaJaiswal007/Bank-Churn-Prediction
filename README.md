# 🏦 Bank Customer Churn Prediction + Sentiment Analysis

An end-to-end data science project that predicts whether a bank customer will churn using Machine Learning, explains predictions using SHAP, and analyzes customer complaints using NLP sentiment analysis.

🚀 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/Krishna-Jaiswal/bank-churn-prediction)  
📓 **Notebook:** [Google Colab](https://colab.research.google.com/drive/1PZDsbJzN2OPnfsGxwG9Vx7Rp2AXp5Q-6?usp=sharing)

---

## 📌 Problem Statement

Banks lose millions every year to customer churn. This project builds a system that:
- Predicts which customers are likely to close their accounts
- Explains *why* the model makes each prediction
- Analyzes real customer reviews to understand qualitative churn signals

---

## 📁 Project Structure

```
Bank-Churn-Prediction/
├── notebook/
│   └── bank_churn_analysis.ipynb   ← full analysis (EDA → ML → SHAP → NLP)
├── app.py                          ← Gradio app (deployed on Hugging Face)
├── requirements.txt                ← dependencies
├── rf_model.pkl                    ← trained Random Forest model
├── scaler.pkl                      ← fitted StandardScaler
├── reviews_sentiment.csv           ← processed reviews with sentiment labels
└── README.md
```

---

## 📊 Dataset

| Dataset | Source | Size |
|---|---|---|
| Bank Customer Churn | Kaggle | 10,000 rows, 12 features |
| Bank Customer Reviews | Kaggle | 1,000 reviews |

**Features:** `credit_score`, `country`, `gender`, `age`, `tenure`, `balance`, `products_number`, `credit_card`, `active_member`, `estimated_salary`

---

## 🔬 Project Phases

### Phase 1 — EDA & Preprocessing
- Explored churn distribution (20% churn rate)
- Visualized numerical and categorical features by churn
- Correlation heatmap revealed `age` as strongest predictor
- Label encoding, StandardScaler, SMOTE for class imbalance

### Phase 2 — ML Models
- Logistic Regression (baseline) vs Random Forest (main model)
- Evaluated using classification report, confusion matrix, ROC-AUC curve

| Model | AUC Score |
|---|---|
| Logistic Regression | 0.76 |
| **Random Forest** | **0.86** ✅ |

### Phase 3 — SHAP Explainability
- Used `shap.TreeExplainer` to interpret Random Forest predictions
- Feature importance bar plot — `age` and `products_number` are top drivers
- Beeswarm plot showing direction of impact per customer

**Key findings:**
- Older customers (high age) are most likely to churn
- Customers with only 1 product churn significantly more
- Inactive members have higher churn risk

### Phase 4 — NLP Sentiment Analysis
- Applied TextBlob sentiment analysis on 1,000 bank reviews
- Classified reviews as positive / negative / neutral
- Wordcloud of negative reviews revealed top complaints

**Top customer complaints:**
- Minimum balance requirements
- Hidden charges and fees
- Net banking issues
- Poor customer service

### Phase 5 — Deployment
- Built interactive Gradio app with real-time churn prediction
- SHAP bar chart explains every prediction
- Deployed on Hugging Face Spaces

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Data processing | pandas, numpy |
| Visualization | matplotlib, seaborn, wordcloud |
| ML Models | scikit-learn (Logistic Regression, Random Forest) |
| Explainability | SHAP |
| NLP | TextBlob |
| Class imbalance | imbalanced-learn (SMOTE) |
| Deployment | Gradio, Hugging Face Spaces |

---

## 📈 Results

- Random Forest AUC: **0.86**
- Correctly identified **64% of churning customers**
- Top churn predictor: **age** (SHAP value: 0.155)
- Most common complaint in negative reviews: **minimum balance & hidden charges**

---

## 💡 Business Insights

1. **Target retention campaigns at customers aged 40+** — age is the strongest churn signal
2. **Promote cross-selling** — customers with only 1 product churn at much higher rates
3. **Launch re-engagement programs** — inactive members are significantly more likely to leave
4. **Address hidden fees** — NLP analysis shows charges are the top customer complaint

---

## 🚀 Run Locally

```bash
git clone https://github.com/KrishnaJaiswal007/Bank-Churn-Prediction
cd Bank-Churn-Prediction
pip install -r requirements.txt
python app.py
```

---

## 👤 Author

**Krishna Jaiswal**  
B.Tech(AI & ML) — VIPS-TC, GGSIPU  
[GitHub](https://github.com/KrishnaJaiswal007) · [Hugging Face](https://huggingface.co/Krishna-Jaiswal)
