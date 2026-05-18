import gradio as gr
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# load model and scaler
rf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

country_map = {"France": 0, "Germany": 1, "Spain": 2}
gender_map = {"Male": 1, "Female": 0}

# create explainer once at startup
explainer = shap.TreeExplainer(rf)

def predict_churn(credit_score, country, gender, age, tenure,
                  balance, products_number, credit_card, active_member, estimated_salary):

    # prepare input
    input_data = pd.DataFrame({
        'credit_score': [float(credit_score)],
        'country': [country_map[country]],
        'gender': [gender_map[gender]],
        'age': [float(age)],
        'tenure': [float(tenure)],
        'balance': [float(balance)],
        'products_number': [int(products_number)],
        'credit_card': [1 if credit_card == "Yes" else 0],
        'active_member': [1 if active_member == "Yes" else 0],
        'estimated_salary': [float(estimated_salary)]
    })

    # scale numerical columns
    num_cols = ['credit_score', 'age', 'tenure', 'balance', 'estimated_salary']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # predict
    prob = rf.predict_proba(input_data)[0][1]
    label = "🔴 Likely to Churn" if prob > 0.5 else "🟢 Likely to Stay"
    result = f"{label}\n\nChurn Probability: {prob*100:.1f}%"

    # shap values
    shap_values = explainer.shap_values(input_data)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0, :, 1]

    feature_names = input_data.columns.tolist()

    # shap bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red' if v > 0 else 'blue' for v in shap_vals]
    ax.barh(feature_names, shap_vals, color=colors)
    ax.set_xlabel("SHAP value\n(red = pushes toward churn, blue = pushes toward stay)")
    ax.set_title("Why this prediction?")
    ax.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()

    return result, fig

# gradio interface
demo = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Slider(300, 850, value=600, step=1, label="Credit Score"),
        gr.Dropdown(["France", "Germany", "Spain"], value="France", label="Country"),
        gr.Dropdown(["Male", "Female"], value="Male", label="Gender"),
        gr.Slider(18, 92, value=35, step=1, label="Age"),
        gr.Slider(0, 10, value=5, step=1, label="Tenure (years)"),
        gr.Number(value=50000, label="Account Balance"),
        gr.Dropdown(["1", "2", "3", "4"], value="1", label="Number of Products"),
        gr.Dropdown(["Yes", "No"], value="Yes", label="Has Credit Card"),
        gr.Dropdown(["Yes", "No"], value="Yes", label="Is Active Member"),
        gr.Number(value=50000, label="Estimated Salary"),
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Plot(label="SHAP Explanation")
    ],
    title="🏦 Bank Customer Churn Predictor",
    description="Enter customer details to predict whether they will churn. SHAP chart explains why.",
)

demo.launch(share=True)