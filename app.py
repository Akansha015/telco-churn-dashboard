import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────
# Custom dark theme + styling
# ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: white; }
    .stSidebar { background-color: #1e293b; padding: 1rem; }
    .stButton>button { background-color: #6366f1; color: white; border: none; border-radius: 6px; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #4f46e5; }
    .stMetric { background-color: #1e293b; border-radius: 10px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); text-align: center; }
    .stSelectbox, .stSlider { background-color: #1e293b; border-radius: 6px; padding: 8px; }
    hr { border-color: #334155; }
    .footer { text-align: center; color: #94a3b8; margin-top: 3rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# Page config
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco ChurnGuard Dashboard",
    page_icon="📉",
    layout="wide"
)

# Title & subtitle
st.title("📉 ChurnGuard – Telco Customer Retention Dashboard")
st.markdown("**XGBoost + SHAP + Interactive What-If Analysis**")

# ────────────────────────────────────────────────
# Load assets (silent unless error)
# ────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('telco_churn_model.pkl')
        explainer = joblib.load('telco_shap_explainer.pkl')
        X_test = pd.read_pickle('X_test_sample.pkl')
        y_test = joblib.load('y_test.pkl')
        return model, explainer, X_test, y_test
    except Exception as e:
        st.error(f"Failed to load files: {str(e)}")
        st.error("Required files: telco_churn_model.pkl, telco_shap_explainer.pkl, X_test_sample.pkl, y_test.pkl")
        st.stop()

model, explainer, X_test, y_test = load_assets()

# ────────────────────────────────────────────────
# Sidebar Navigation
# ────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio("", [
    "Customer Lookup + SHAP",
    "What-If Simulator",
    "Top At-Risk Customers",
    "Key Insights & Recommendations",
    "About"
])

st.sidebar.markdown("---")
# st.sidebar.caption("Built by Akansha Pruthi • Feb 2026")

# ────────────────────────────────────────────────
# Tab 1: Customer Lookup + SHAP (KPIs in one row + cleaned delta)
# ────────────────────────────────────────────────
if page == "Customer Lookup + SHAP":
    st.header("Customer Lookup & SHAP Explanation")

    customer_idx = st.selectbox(
        "Select customer index from test set",
        options=range(len(X_test)),
        format_func=lambda x: f"Index {x} (original row {X_test.index[x]})"
    )

    with st.spinner("Loading customer data & computing SHAP..."):
        customer_data = X_test.iloc[customer_idx:customer_idx+1]
        actual_churn = "Yes" if y_test.iloc[customer_idx] == 1 else "No"
        prob = float(model.predict_proba(customer_data)[0][1])  # native float

        # Original readable profile
        @st.cache_data
        def load_original_data():
            return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

        original_df = load_original_data()
        orig_idx = X_test.index[customer_idx]
        orig_row = original_df.iloc[orig_idx]

    # ── All KPIs in ONE ROW with EQUAL HEIGHT ──
    st.markdown("""
    <style>
        .kpi-card {
            height: 140px !important;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            background-color: #1e293b;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-align: center;
        }
        .kpi-title {
            font-size: 1rem;
            color: #94a3b8;
            margin-bottom: 8px;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .kpi-delta {
            font-size: 1.1rem;
            margin-top: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Churn Probability
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Churn Probability</div>
            <div class="kpi-value" style="color: {'#ef4444' if prob > 0.5 else '#10b981'}">
                {prob:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Actual Churn
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Actual Churn</div>
            <div class="kpi-value" style="color: {'#ef4444' if actual_churn == 'Yes' else '#10b981'}">
                {actual_churn}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Compared to Average (only delta, no duplicate %)
    with col3:
        avg_prob = float(model.predict_proba(X_test)[:, 1].mean())
        delta = prob - avg_prob
        delta_color = "#10b981" if delta < 0 else "#ef4444" if delta > 0 else "#6b7280"
        arrow = "↓" if delta < 0 else "↑" if delta > 0 else "≈"
        label = "lower risk (good)" if delta < 0 else "higher risk (alert)" if delta > 0 else "average"

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Compared to Average</div>
            <div class="kpi-value" style="color:{delta_color}">
                {arrow}{abs(delta):.1%}
            </div>
            <div class="kpi-delta" style="color:{delta_color}">
                {label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Original features table
    st.subheader("Customer Profile (Readable Features)")
    st.dataframe(
        orig_row.to_frame().T.style.set_properties(**{'text-align': 'left'}),
        use_container_width=True
    )

    st.subheader("Why does the model predict this probability?")
    shap_values_single = explainer.shap_values(customer_data)[0]

    # Waterfall plot
    fig = plt.figure(figsize=(12, 7))
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values_single,
            base_values=explainer.expected_value,
            data=customer_data.values[0],
            feature_names=customer_data.columns.tolist()
        ),
        max_display=12,
        show=False
    )
    st.pyplot(fig, clear_figure=True)

    # Quick summary of biggest driver
    st.subheader("Key Driver Summary")
    abs_shap = np.abs(shap_values_single)
    top_idx = np.argmax(abs_shap)
    top_feature = customer_data.columns[top_idx]
    top_shap = shap_values_single[top_idx]
    direction = "increases churn risk" if top_shap > 0 else "decreases churn risk"
    magnitude = f"{abs(top_shap):.3f}"
    st.info(f"""
    The **strongest driver** for this customer is **{top_feature}**  
    → It **{direction}** by **{magnitude}** (SHAP value).
    """)

    # Top 5 SHAP contributors
    st.subheader("Top 5 Contributors to this Prediction")
    shap_abs = np.abs(shap_values_single)
    top5_idx = np.argsort(shap_abs)[-5:][::-1]

    for i in top5_idx:
        feat = customer_data.columns[i]
        val = shap_values_single[i]
        sign = "↑" if val > 0 else "↓"
        impact = "increases churn" if val > 0 else "decreases churn"
        st.markdown(f"- **{sign} {feat}** ({customer_data.iloc[0,i]:.2f}) → {impact} ({val:+.3f})")

    # Risk level progress bar
    st.subheader("Risk Level")
    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
    st.progress(prob)
    st.caption(f"**{risk_level} Risk** ({prob:.1%} probability)")

    # Export button
    if st.button("Export this customer's prediction", use_container_width=True):
        export_data = {
            "customer_index": int(customer_idx),
            "original_row": int(orig_idx),
            "churn_probability": float(prob),
            "actual_churn": actual_churn,
            "top_shap_feature": top_feature,
            "top_shap_value": float(top_shap)
        }
        st.download_button(
            "Download JSON",
            data=str(export_data),
            file_name=f"customer_{customer_idx}_prediction.json",
            mime="application/json"
        )

# ────────────────────────────────────────────────
# Tab 2: What-If Simulator (expanded)
# ────────────────────────────────────────────────
elif page == "What-If Simulator":
    st.header("What-If Retention Simulator")
    st.markdown("Adjust features to simulate churn probability changes")

    with st.form("what_if_form"):
        col1, col2 = st.columns(2)
        with col1:
            contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            monthly_charges = st.slider("Monthly Charges ($)", 10.0, 130.0, 70.0, step=1.0)
            tenure = st.slider("Tenure (months)", 0, 80, 12, step=1)

        with col2:
            fiber_optic = st.checkbox("Has Fiber Optic Internet", False)
            electronic_check = st.checkbox("Pays with Electronic Check", False)
            paperless = st.checkbox("Paperless Billing", False)
            online_security = st.checkbox("Has Online Security", False)
            tech_support = st.checkbox("Has Tech Support", False)

        submitted = st.form_submit_button("Calculate New Prediction", use_container_width=True)

    if submitted:
        with st.spinner("Running simulation..."):
            input_dict = {col: 0.0 for col in X_test.columns}

            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['tenure'] = tenure

            if contract_type == "One year":
                input_dict['Contract_One year'] = 1
            elif contract_type == "Two year":
                input_dict['Contract_Two year'] = 1

            if fiber_optic:
                input_dict['InternetService_Fiber optic'] = 1
            if electronic_check:
                input_dict['PaymentMethod_Electronic check'] = 1
            if paperless:
                input_dict['PaperlessBilling_Yes'] = 1
            if online_security:
                input_dict['OnlineSecurity_Yes'] = 1
            if tech_support:
                input_dict['TechSupport_Yes'] = 1

            input_df = pd.DataFrame([input_dict])
            new_prob = model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Churn Probability", f"{new_prob:.1%}",
                      delta=f"{new_prob - prob:+.1%}" if 'prob' in locals() else None,
                      delta_color="inverse")
        with col2:
            st.success("Simulation complete!")

        st.info("**Best retention combo**: Two-year contract + charges < $70 + tenure > 40 months")

# ────────────────────────────────────────────────
# Tab 3: Top At-Risk Customers
# ────────────────────────────────────────────────
elif page == "Top At-Risk Customers":
    st.header("Top 10 At-Risk Customers")

    with st.spinner("Calculating risk scores..."):
        proba = model.predict_proba(X_test)[:, 1]
        risk_df = pd.DataFrame({
            'Original Index': X_test.index,
            'Churn Probability': proba,
            'Actual Churn': ['Yes' if y == 1 else 'No' for y in y_test]
        }).sort_values('Churn Probability', ascending=False).head(10).reset_index(drop=True)

    # Keep numeric for gradient, format only for display
    styled_df = risk_df.style \
        .format({'Churn Probability': '{:.1%}'}) \
        .background_gradient(subset=['Churn Probability'], cmap='Reds') \
        .set_properties(**{'text-align': 'center'})

    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True
    )

    # ── Color Legend for Heatmap ──
    st.markdown("### Color Legend (Churn Risk Heatmap)")
    col_legend1, col_legend2, col_legend3 = st.columns(3)

    with col_legend1:
        st.markdown("""
        <div style="background-color:#8b0000; color:white; padding:10px; border-radius:6px; text-align:center; font-weight:bold;">
        High Risk (95%+)
        </div>
        """, unsafe_allow_html=True)

    with col_legend2:
        st.markdown("""
        <div style="background-color:#ff6347; color:white; padding:10px; border-radius:6px; text-align:center; font-weight:bold;">
        Medium-High Risk (~90–95%)
        </div>
        """, unsafe_allow_html=True)

    with col_legend3:
        st.markdown("""
        <div style="background-color:#ffb6c1; color:black; padding:10px; border-radius:6px; text-align:center; font-weight:bold;">
        Moderate Risk (among top 10)
        </div>
        """, unsafe_allow_html=True)

    st.caption("🔴 Darker red = higher predicted churn probability → higher priority for retention actions")

    # ── Download button ──
    download_df = risk_df.copy()
    download_df['Churn Probability'] = download_df['Churn Probability'].map('{:.1%}'.format)
    csv = download_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Top 10 At-Risk List (CSV)",
        data=csv,
        file_name="top_10_at_risk_customers.csv",
        mime="text/csv",
        use_container_width=True
    )

# ────────────────────────────────────────────────
# Tab 4: Key Insights & Recommendations
# ────────────────────────────────────────────────
elif page == "Key Insights & Recommendations":
    st.header("Key Insights from SHAP Analysis")

    st.subheader("Top Drivers of Churn")
    st.markdown("""
    1. **Contract_Two year** — strongest retention factor  
    2. **tenure** — longer stay = much lower risk  
    3. **MonthlyCharges** — high bills strongly increase churn  
    4. **Contract_One year** — protective but weaker  
    5. **InternetService_Fiber optic** — tends to increase churn  
    6. **PaymentMethod_Electronic check** — clear churn signal
    """)

    st.subheader("Recommended Actions (Prioritized)")
    st.markdown("""
    **Highest impact:**
    - Aggressively promote **2-year contracts**  
    - Focus early retention on first **12–18 months**  
    - Offer discounts for bills **> $85–90**  
    - Move electronic check users to auto-pay  

    **Supportive:**
    - Bundle **Online Security** & **Tech Support**  
    - Re-evaluate pricing & satisfaction for **fiber optic** customers
    """)

# ────────────────────────────────────────────────
# Tab 5: About (with Google Sheets feedback logging)
# ────────────────────────────────────────────────
elif page == "About":
    st.header("About This Dashboard")
    st.markdown("""
    **Project**: Telco Customer Churn Prediction & Retention Simulator  
    **Model**: XGBoost classifier (AUC ≈ 0.85)  
    **Explainability**: SHAP (global & local)  
    **Dataset**: IBM Telco Customer Churn (Kaggle)  
    **Tools**: Python, XGBoost, SHAP, Streamlit  
    """)

    st.markdown("---")

    # ── Feedback Section ──
    st.subheader("📬 Send Feedback / Suggestions")
    st.caption("Love it? Hate it? Got ideas for new features? Let me know — every comment helps improve this portfolio project!")

    with st.form(key="feedback_form"):
        feedback_text = st.text_area(
            "Your feedback / ideas / questions",
            height=150,
            placeholder="Example: \"Great dashboard! Would love to see a retention ROI calculator...\" or \"The colors could be brighter in dark mode.\""
        )

        email = st.text_input(
            "Your email (optional – only used if I need to reply)",
            placeholder="you@example.com"
        )

        submitted = st.form_submit_button("Submit Feedback", use_container_width=True)

    if submitted:
        if feedback_text.strip():
            try:
                import gspread
                from oauth2client.service_account import ServiceAccountCredentials
                from datetime import datetime

                # Google Sheets setup
                scope = [
                    "https://spreadsheets.google.com/feeds",
                    "https://www.googleapis.com/auth/drive"
                ]
                # Path to your credentials JSON file (update if needed)
                import json
                creds_dict = json.loads(st.secrets["credentials_json"])
                creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
                client = gspread.authorize(creds)

                # Open your sheet (update name if different)
                sheet = client.open("ChurnGuard_Feedback").sheet1

                # Append row
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row = [timestamp, feedback_text, email or "Anonymous"]
                sheet.append_row(row)

                # Success feedback
                st.success("Thank you very much! Your feedback has been recorded. 🎉")
                st.balloons()

            except FileNotFoundError:
                st.error("credentials.json file not found. Please upload it to Colab.")
            except Exception as e:
                st.error(f"Failed to save feedback: {str(e)}")
                st.info("Tip: Make sure the Google Sheet is shared with the service account email.")
        else:
            st.warning("Please write something before submitting.")

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div class='footer'>Built by Akansha Pruthi • For portfolio & educational use</div>", unsafe_allow_html=True)
