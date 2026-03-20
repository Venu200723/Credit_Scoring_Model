import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

def load_artifacts():
    paths = {
        "model":   os.path.join(MODEL_DIR, "best_model.pkl"),
        "scaler":  os.path.join(MODEL_DIR, "scaler.pkl"),
        "encoders":os.path.join(MODEL_DIR, "label_encoders.pkl"),
        "meta":    os.path.join(MODEL_DIR, "metadata.pkl"),
    }
    missing = [k for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files: {missing}. "
            "Please run credit_scoring_model.py first."
        )
    model          = joblib.load(paths["model"])
    scaler         = joblib.load(paths["scaler"])
    label_encoders = joblib.load(paths["encoders"])
    meta           = joblib.load(paths["meta"])
    return model, scaler, label_encoders, meta

try:
    model, scaler, label_encoders, meta = load_artifacts()
    NUMERIC_COLS     = meta["numeric_cols"]
    CATEGORICAL_COLS = meta["categorical_cols"]
    FEATURE_COLS     = meta["feature_cols"]
    TARGET_COL       = meta["target_col"]
    N_CLASSES        = meta["n_classes"]
    MODEL_NAME       = meta["best_model_name"]
    MODEL_ACCURACY   = meta.get("accuracy", None)
    ARTIFACTS_LOADED = True
    print(f"[OK] Model loaded: {MODEL_NAME}  (accuracy={MODEL_ACCURACY:.4f})")
except FileNotFoundError as e:
    ARTIFACTS_LOADED = False
    LOAD_ERROR = str(e)
    print(f"[ERROR] {e}")

BAND_INFO = {
    "Poor":     ("🔴", "#FF4444", "High risk. Loan likely to be declined."),
    "Standard": ("🟡", "#FF9800", "Moderate risk. May qualify with conditions."),
    "Good":     ("🟢", "#4CAF50", "Low risk. Likely to qualify for most loans."),
}

def get_band_info(label: str):
    label_clean = str(label).strip()
    for key in BAND_INFO:
        if key.lower() in label_clean.lower():
            return BAND_INFO[key]
    return ("⚪", "#888888", "Score predicted.")

def make_importance_chart(top_n=10):
    """Build feature importance chart using permutation importances stored in metadata."""
    if not ARTIFACTS_LOADED:
        return None

    # Prefer permutation importances saved during neural network training
    perm_imp = meta.get("perm_importances", None)

    if perm_imp is not None:
        importance = np.array(perm_imp)
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = (np.abs(model.coef_[0])
                      if model.coef_.ndim > 1 else np.abs(model.coef_))
    else:
        return None

    feat_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": importance
    }).sort_values("Importance", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(feat_df)))
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors, edgecolor="none")
    ax.set_xlabel("Mean Accuracy Decrease (Permutation Importance)", fontsize=9)
    ax.set_title(f"Top {top_n} Feature Importances — Neural Network", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

IMPORTANCE_CHART = make_importance_chart() if ARTIFACTS_LOADED else None

FIELD_DEFAULTS = {
    "Num_Bank_Accounts":       2,
    "Num_Credit_Card":         3,
    "Num_of_Loan":             2,
    "Changed_Credit_Limit":    5,
    "Total_EMI_per_month":     300,
    "Amount_invested_monthly": 200,
    "Monthly_Balance":         2000,
    "Payment_Behaviour":       "Low_spent_Medium_value_payments",
}

def predict_credit_score(
    customer_name, age, monthly_salary, occupation,
    outstanding_debt, credit_utilization, interest_rate,
    delay_from_due, delayed_payments, credit_history_age,
    num_inquiries, credit_mix, min_amount_paid
):
    if not ARTIFACTS_LOADED:
        return (f"❌ Model not loaded.\n{LOAD_ERROR}", None, None)

    annual_income = monthly_salary * 12

    user_inputs = {
        "Age":                     age,
        "Annual_Income":           annual_income,
        "Monthly_Inhand_Salary":   monthly_salary,
        "Outstanding_Debt":        outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Interest_Rate":           interest_rate,
        "Delay_from_due_date":     delay_from_due,
        "Num_of_Delayed_Payment":  delayed_payments,
        "Credit_History_Age":      credit_history_age,
        "Num_Credit_Inquiries":    num_inquiries,
        "Occupation":              occupation,
        "Credit_Mix":              credit_mix,
        "Payment_of_Min_Amount":   min_amount_paid,
    }

    row = {}
    for col in FEATURE_COLS:
        if col in user_inputs:
            row[col] = [user_inputs[col]]
        else:
            row[col] = [FIELD_DEFAULTS.get(col, 0)]

    input_df = pd.DataFrame(row)

    for col in CATEGORICAL_COLS:
        if col in input_df.columns and col in label_encoders:
            le = label_encoders[col]
            known = set(le.classes_)
            input_df[col] = input_df[col].astype(str).apply(
                lambda x: x if x in known else str(le.classes_[0])
            )
            input_df[col] = le.transform(input_df[col])

    input_df[NUMERIC_COLS] = scaler.transform(input_df[NUMERIC_COLS])

    pred_class = model.predict(input_df)[0]
    if hasattr(model, "predict_proba"):
        proba      = model.predict_proba(input_df)[0]
        confidence = float(np.max(proba)) * 100
    else:
        confidence = None

    if TARGET_COL in label_encoders:
        pred_label = label_encoders[TARGET_COL].inverse_transform([pred_class])[0]
    else:
        pred_label = str(pred_class)

    icon, color, description = get_band_info(pred_label)
    conf_str = f"{confidence:.1f}%" if confidence is not None else "N/A"

    name_line = (
        f'<div style="font-size:14px;color:#ccc;margin-bottom:8px;">'
        f'Customer: <strong style="color:#fff;">{customer_name.strip()}</strong></div>'
        if customer_name.strip() else ""
    )

    result_html = f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.08);
    ">
        {name_line}
        <div style="font-size:64px;margin-bottom:8px;">{icon}</div>
        <div style="
            font-size: 38px;
            font-weight: 800;
            color: {color};
            margin-bottom: 6px;
            text-shadow: 0 0 20px {color}55;
        ">{pred_label}</div>
        <div style="font-size:16px;color:#aaa;margin-bottom:20px;">{description}</div>
        <div style="
            display: inline-block;
            background: rgba(255,255,255,0.08);
            border-radius: 30px;
            padding: 8px 28px;
            font-size: 15px;
            color: #fff;
        ">
            Confidence: <strong style="color:{color};">{conf_str}</strong>
        </div>
    </div>
    """

    confidence_fig = None
    if hasattr(model, "predict_proba"):
        proba_all    = model.predict_proba(input_df)[0]
        class_labels = (label_encoders[TARGET_COL].classes_
                        if TARGET_COL in label_encoders
                        else [str(i) for i in range(len(proba_all))])

        fig2, ax2 = plt.subplots(figsize=(5, 3))
        bar_colors = ["#2ecc71" if c == pred_label else "#444" for c in class_labels]
        ax2.bar(class_labels, proba_all * 100, color=bar_colors, edgecolor="none", width=0.5)
        ax2.set_ylabel("Probability (%)")
        ax2.set_title("Confidence by Class", fontsize=11)
        ax2.set_ylim(0, 115)
        for i, v in enumerate(proba_all * 100):
            ax2.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10, color="white")
        ax2.set_facecolor("#1a1a2e")
        fig2.patch.set_facecolor("#1a1a2e")
        ax2.tick_params(colors="white")
        ax2.yaxis.label.set_color("white")
        ax2.title.set_color("white")
        ax2.spines[:].set_color("#444")
        fig2.tight_layout()
        confidence_fig = fig2

    return result_html, confidence_fig, IMPORTANCE_CHART


CSS = """
body, .gradio-container {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3, label, .label-wrap span { color: #e0e0e0 !important; }
.gr-button-primary {
    background: linear-gradient(90deg, #7b2ff7, #f107a3) !important;
    border: none !important;
    border-radius: 30px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 12px 40px !important;
    color: white !important;
    transition: all 0.3s ease !important;
}
.gr-button-primary:hover { opacity: 0.85; transform: scale(1.02); }
footer { display: none !important; }
"""

def build_ui():
    if not ARTIFACTS_LOADED:
        with gr.Blocks(css=CSS, title="Credit Scoring Model") as demo:
            gr.HTML("""
            <div style='text-align:center;padding:60px;color:#ff6b6b;'>
                <h1>⚠️ Model Not Found</h1>
                <p style='font-size:18px'>Run <code>python credit_scoring_model.py</code> first.</p>
            </div>
            """)
        return demo

    with gr.Blocks(css=CSS, title="Credit Scoring Model") as demo:

        gr.HTML(f"""
        <div style="text-align:center;padding:28px 20px 10px;">
            <h1 style="
                font-size:40px;font-weight:900;margin:0;
                background:linear-gradient(90deg,#7b2ff7,#f107a3);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
            ">💳 Credit Score Predictor</h1>
            <p style="font-size:14px;color:#aaa;margin-top:8px;">
                Powered by <strong style="color:#ccc;">{MODEL_NAME}</strong>
                &nbsp;•&nbsp; Accuracy: <strong style="color:#ccc;">{MODEL_ACCURACY:.1%}</strong>
            </p>
        </div>
        """)

        with gr.Row():

            with gr.Column(scale=1):
                gr.Markdown("### 👤 Personal Info")
                customer_name = gr.Textbox(
                    label="Full Name",
                    placeholder="e.g. Ravi Kumar",
                    max_lines=1
                )
                age        = gr.Slider(18, 100, value=30, step=1,   label="Age")
                monthly_salary = gr.Slider(0, 100000, value=4000, step=500,
                                           label="Monthly Income ($)")
                occupation = gr.Dropdown(
                    choices=["Scientist", "Teacher", "Engineer", "Entrepreneur",
                             "Developer", "Lawyer", "Media_Manager", "Doctor",
                             "Journalist", "Manager", "Accountant", "Musician",
                             "Mechanic", "Writer", "Architect"],
                    value="Engineer", label="Occupation"
                )

                gr.Markdown("### 💳 Credit Behaviour")
                credit_mix = gr.Radio(
                    choices=["Bad", "Standard", "Good"],
                    value="Standard", label="Credit Mix"
                )
                min_amount_paid = gr.Radio(
                    choices=["Yes", "No", "NM"],
                    value="No", label="Pays Minimum Amount on Time?"
                )
                delayed_payments = gr.Slider(0, 30, value=1, step=1,
                                             label="No. of Delayed Payments")
                delay_from_due   = gr.Slider(0, 100, value=5, step=1,
                                             label="Avg Days Past Due Date")

            with gr.Column(scale=1):
                gr.Markdown("### 📊 Financial Details")
                outstanding_debt    = gr.Slider(0, 100000, value=5000, step=500,
                                                label="Outstanding Debt ($)")
                credit_utilization  = gr.Slider(0, 100, value=35, step=1,
                                                label="Credit Utilization (%)")
                interest_rate       = gr.Slider(1, 50, value=15, step=1,
                                                label="Interest Rate (%)")
                credit_history_age  = gr.Slider(0, 600, value=120, step=6,
                                                label="Credit History Age (months)")
                num_inquiries       = gr.Slider(0, 30, value=2, step=1,
                                                label="No. of Credit Inquiries")

                gr.Markdown("### 🎯 Result")
                result_html      = gr.HTML()
                conf_chart       = gr.Plot(label="Confidence by Class")
                importance_chart = gr.Plot(label="Feature Importance")
                importance_chart.value = IMPORTANCE_CHART

        predict_btn = gr.Button("🔍 Predict Credit Score", variant="primary", size="lg")

        predict_btn.click(
            fn=predict_credit_score,
            inputs=[
                customer_name, age, monthly_salary, occupation,
                outstanding_debt, credit_utilization, interest_rate,
                delay_from_due, delayed_payments, credit_history_age,
                num_inquiries, credit_mix, min_amount_paid
            ],
            outputs=[result_html, conf_chart, importance_chart],
        )

        gr.HTML("""
        <div style="text-align:center;margin-top:16px;color:#555;font-size:12px;">
            CodeAlpha Credit Scoring Model &nbsp;|&nbsp; Deep Learning Neural Network &amp; Gradio
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
