## ChurnGuard – Telco Customer Churn Prediction & Retention Dashboard
Overview
End-to-end interactive dashboard that helps telecom companies understand, predict, and reduce customer churn.
Built to showcase full ML workflow: data cleaning → modeling → explainability → deployment → user interaction.
Key Features

Customer Lookup + SHAP: Select any test customer → view churn probability, actual outcome, original features, and interactive SHAP waterfall plot showing exactly why the model predicts that risk.
What-If Retention Simulator: Adjust contract type, monthly charges, tenure, add-ons (security, tech support, etc.) → instantly see how changes impact predicted churn probability.
Top 10 At-Risk Customers: Ranked table with red heatmap (darker = higher risk), actual churn labels, and CSV download.
Key Insights & Recommendations: Summarized SHAP-based business drivers and prioritized retention actions (e.g., push 2-year contracts, target high-bill customers).
Feedback Collection: Live text box in About section — user suggestions are automatically logged to Google Sheets for future improvements.

Tech Stack

Model: XGBoost classifier (AUC ~0.85)
Explainability: SHAP (TreeExplainer)
Web app: Streamlit
Backend data: Pandas, NumPy, Joblib
Feedback logging: gspread + Google Service Account
Tunnel (during dev): Bore / ngrok / localtunnel
Deployment (planned): Streamlit Community Cloud

Dataset
IBM Telco Customer Churn (Kaggle) – 7,043 customers, 21 features (tenure, contract, monthly charges, services, demographics, etc.)
Business Value

Identifies strongest churn drivers (e.g., month-to-month contracts, high bills, electronic checks)
Allows non-technical stakeholders to simulate “what if” interventions
Prioritizes outreach to highest-risk customers

Live Demo - https://telco-churn-dashboard-a7buqje5caqqxz9p5antqc.streamlit.app/
