# frontend/app.py — Streamlit UI for the Mortgage Underwriting System
#
# Set BACKEND_URL in Streamlit Cloud secrets:
#   BACKEND_URL = "https://your-app.onrender.com"
#
# For local development, it defaults to http://localhost:8000

import streamlit as st
import requests
import os
import json

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Mortgage Underwriting System",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 AI Mortgage Underwriting System")
st.caption("Multi-agent underwriting powered by LangGraph · Groq · FastAPI")

# Backend health check in sidebar
with st.sidebar:
    st.header("System Status")
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.success("✅ Backend connected")
            st.write(f"**Model:** {data.get('model', 'unknown')}")
            st.write(f"**Policy store:** {'✅ Ready' if data.get('policy_store_ready') else '⏳ Loading'}")
        else:
            st.error(f"Backend returned {r.status_code}")
    except Exception as e:
        st.warning(f"⚠️ Backend unreachable\n\n{e}\n\nStart the FastAPI server or check BACKEND_URL.")

    st.divider()
    st.markdown("**Architecture**")
    st.markdown("""
- **Frontend:** Streamlit Cloud
- **Backend:** FastAPI on Render
- **LLM:** Groq (llama-3.3-70b)
- **Agents:** LangGraph (6 agents)
- **RAG:** ChromaDB + HuggingFace
""")

# ---------------------------------------------------------------------------
# Load sample applicants
# ---------------------------------------------------------------------------

@st.cache_data
def load_samples():
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "mortgage_test_cases.json")
    try:
        with open(sample_path) as f:
            return json.load(f)["test_cases"]
    except FileNotFoundError:
        return []

samples = load_samples()
sample_names = ["(enter manually)"] + [f"{c['case_id']} — {c['name']} (expected: {c['expected_decision']})" for c in samples]

st.subheader("Application Input")
selected = st.selectbox("Load a sample applicant or enter manually:", sample_names)

if selected != "(enter manually)" and samples:
    idx = sample_names.index(selected) - 1
    prefill = samples[idx]
else:
    prefill = {}

# ---------------------------------------------------------------------------
# Application form
# ---------------------------------------------------------------------------

with st.form("application_form"):
    st.markdown("### Applicant")
    col1, col2, col3 = st.columns(3)
    with col1:
        case_id     = st.text_input("Case ID",      value=prefill.get("case_id", "MTG-2025-NEW"))
        name        = st.text_input("Full Name",     value=prefill.get("name", ""))
        ssn         = st.text_input("SSN",           value=prefill.get("ssn", ""), help="Format: XXX-XX-XXXX")
    with col2:
        email       = st.text_input("Email",         value=prefill.get("email", ""))
        phone       = st.text_input("Phone",         value=prefill.get("phone", ""))
        address     = st.text_input("Address",       value=prefill.get("address", ""))
    with col3:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850,
                                       value=int(prefill.get("credit_score", 700)))

    st.divider()
    st.markdown("### Credit History")
    col1, col2, col3 = st.columns(3)
    ch = prefill.get("credit_history", {})
    with col1:
        bankruptcies       = st.number_input("Bankruptcies",          min_value=0, value=int(ch.get("bankruptcies", 0)))
        foreclosures       = st.number_input("Foreclosures",          min_value=0, value=int(ch.get("foreclosures", 0)))
    with col2:
        late_12mo          = st.number_input("Late Payments (12mo)",  min_value=0, value=int(ch.get("late_payments_12mo", 0)))
        late_24mo          = st.number_input("Late Payments (24mo)",  min_value=0, value=int(ch.get("late_payments_24mo", 0)))
    with col3:
        inquiries          = st.number_input("Inquiries (6mo)",       min_value=0, value=int(ch.get("inquiries_6mo", 0)))
        credit_notes       = st.text_area("Credit Notes",             value=ch.get("credit_notes", ""), height=80)

    st.divider()
    st.markdown("### Employment & Income")
    col1, col2 = st.columns(2)
    emp = prefill.get("employment", {})
    with col1:
        employer        = st.text_input("Employer",             value=emp.get("employer", ""))
        position        = st.text_input("Position",             value=emp.get("position", ""))
        emp_type        = st.selectbox("Employment Type",       ["W2", "Self-Employed", "1099"],
                                       index=["W2", "Self-Employed", "1099"].index(emp.get("type", "W2")) if emp.get("type") in ["W2", "Self-Employed", "1099"] else 0)
        years_employed  = st.number_input("Years at Employer", min_value=0.0, step=0.5, value=float(emp.get("years", 2.0)))
    with col2:
        monthly_income  = st.number_input("Monthly Gross Income ($)", min_value=1.0, step=100.0,
                                          value=float(emp.get("monthly_income", 5000)))
        employment_gap  = st.selectbox("Employment Gap?", ["None", "Yes"],
                                       index=0 if emp.get("employment_gap", "None") == "None" else 1)
        gap_explanation = st.text_input("Gap Explanation", value=emp.get("gap_explanation", "N/A"))

    st.divider()
    st.markdown("### Monthly Debts")
    col1, col2, col3 = st.columns(3)
    debts = prefill.get("debts", {})
    with col1:
        car_loan        = st.number_input("Car Loan ($)",        min_value=0.0, step=50.0, value=float(debts.get("car_loan", 0)))
        student_loan    = st.number_input("Student Loan ($)",    min_value=0.0, step=50.0, value=float(debts.get("student_loan", 0)))
    with col2:
        credit_cards    = st.number_input("Credit Cards ($)",    min_value=0.0, step=50.0, value=float(debts.get("credit_cards", 0)))
        personal_loan   = st.number_input("Personal Loan ($)",   min_value=0.0, step=50.0, value=float(debts.get("personal_loan", 0)))
    with col3:
        total_debt = st.number_input("Total Monthly Debt ($)", min_value=0.0, step=50.0,
                                     value=float(debts.get("total_monthly_debt", car_loan + student_loan + credit_cards + personal_loan)),
                                     help="Should match sum of debts above")

    st.divider()
    st.markdown("### Assets")
    col1, col2 = st.columns(2)
    assets = prefill.get("assets", {})
    with col1:
        checking        = st.number_input("Checking ($)",  min_value=0.0, step=1000.0, value=float(assets.get("checking", 0)))
        savings         = st.number_input("Savings ($)",   min_value=0.0, step=1000.0, value=float(assets.get("savings", 0)))
    with col2:
        retirement      = st.number_input("401k ($)",      min_value=0.0, step=1000.0, value=float(assets.get("401k", 0)))

    st.divider()
    st.markdown("### Loan Details")
    col1, col2, col3 = st.columns(3)
    loan = prefill.get("loan", {})
    with col1:
        loan_amount     = st.number_input("Loan Amount ($)",        min_value=1.0, step=5000.0, value=float(loan.get("amount", 300000)))
        down_payment    = st.number_input("Down Payment ($)",        min_value=0.0, step=5000.0, value=float(loan.get("down_payment", 60000)))
    with col2:
        est_payment     = st.number_input("Est. Monthly Payment ($)", min_value=1.0, step=100.0, value=float(loan.get("estimated_payment", 2000)))
        closing_costs   = st.number_input("Closing Costs ($)",       min_value=0.0, step=500.0,  value=float(loan.get("closing_costs", 0)))
    with col3:
        loan_use        = st.selectbox("Loan Use", ["Primary Residence", "Second Home", "Investment"],
                                       index=0 if loan.get("use", "Primary Residence") == "Primary Residence" else 1)

    st.divider()
    st.markdown("### Property")
    col1, col2, col3 = st.columns(3)
    prop = prefill.get("property", {})
    with col1:
        purchase_price  = st.number_input("Purchase Price ($)",    min_value=1.0, step=5000.0, value=float(prop.get("purchase_price", 360000)))
        appraised_value = st.number_input("Appraised Value ($)",   min_value=1.0, step=5000.0, value=float(prop.get("appraised_value", 360000)))
    with col2:
        prop_condition  = st.text_input("Condition",               value=prop.get("condition", "C3 - Average"))
        prop_type       = st.text_input("Property Type",           value=prop.get("type", "Single Family Home"))
    with col3:
        required_repairs = st.number_input("Required Repairs ($)", min_value=0.0, step=500.0, value=float(prop.get("required_repairs", 0)))
        repair_details   = st.text_input("Repair Details",         value=prop.get("repair_details", "None"))

    submitted = st.form_submit_button("🔍 Run Underwriting Analysis", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Submit + results
# ---------------------------------------------------------------------------

if submitted:
    payload = {
        "case_id": case_id,
        "name": name,
        "ssn": ssn,
        "email": email,
        "phone": phone,
        "address": address,
        "credit_score": int(credit_score),
        "credit_history": {
            "bankruptcies": int(bankruptcies),
            "foreclosures": int(foreclosures),
            "late_payments_12mo": int(late_12mo),
            "late_payments_24mo": int(late_24mo),
            "collections": [],
            "inquiries_6mo": int(inquiries),
            "oldest_tradeline_years": 5,
            "total_tradelines": 6,
            "credit_notes": credit_notes,
        },
        "employment": {
            "employer": employer,
            "position": position,
            "type": emp_type,
            "years": float(years_employed),
            "monthly_income": float(monthly_income),
            "employment_gap": employment_gap,
            "gap_explanation": gap_explanation,
            "employment_history": [],
            "income_details": {
                "base_salary": float(monthly_income) * 12,
                "bonus_2023": 0,
                "bonus_2024": 0,
                "bonus_stable": False,
                "employer_confirmation": "Not provided",
            },
        },
        "debts": {
            "car_loan": float(car_loan),
            "student_loan": float(student_loan),
            "credit_cards": float(credit_cards),
            "personal_loan": float(personal_loan),
            "total_monthly_debt": float(total_debt),
        },
        "assets": {
            "checking": float(checking),
            "savings": float(savings),
            "liquid_assets_total": float(checking + savings),
            "401k": float(retirement),
            "recent_deposits": [],
            "deposit_explanations": "",
            "reserves_months": 0,
        },
        "loan": {
            "amount": float(loan_amount),
            "down_payment": float(down_payment),
            "closing_costs": float(closing_costs),
            "estimated_payment": float(est_payment),
            "property_type": prop_type,
            "use": loan_use,
            "monthly_piti": float(est_payment),
        },
        "property": {
            "purchase_price": float(purchase_price),
            "appraised_value": float(appraised_value),
            "condition": prop_condition,
            "type": prop_type,
            "required_repairs": float(required_repairs),
            "repair_details": repair_details,
        },
    }

    with st.spinner("Running multi-agent underwriting analysis... (this takes ~30–60 seconds)"):
        try:
            response = requests.post(f"{BACKEND_URL}/analyze", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The backend may be waking up from sleep (Render free tier). Try again in 30 seconds.")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not reach the backend. Check BACKEND_URL in Streamlit secrets.")
            st.stop()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    # --- Decision banner ---
    st.divider()
    decision = result.get("final_decision", "UNKNOWN")
    risk_score = result.get("risk_score", 0)

    if decision == "APPROVED":
        st.success(f"## ✅ APPROVED — Risk Score: {risk_score}/100")
    elif decision == "CONDITIONAL_APPROVAL":
        st.warning(f"## ⚠️ CONDITIONAL APPROVAL — Risk Score: {risk_score}/100")
    else:
        st.error(f"## ❌ DENIED — Risk Score: {risk_score}/100")

    if result.get("human_review_required"):
        st.info("👤 **Human review flagged** — this application requires senior underwriter oversight.")

    if result.get("bias_flags"):
        st.warning(f"🚩 **Fair Lending flags detected:** {', '.join(result['bias_flags'])}")

    # --- Analysis tabs ---
    st.divider()
    tabs = st.tabs(["💳 Credit", "💵 Income", "💰 Assets", "🏠 Collateral", "🔎 Critic Review", "⚖️ Decision Memo", "📋 Audit Trail"])

    with tabs[0]:
        st.markdown(result.get("credit_analysis") or "_No analysis available._")
    with tabs[1]:
        st.markdown(result.get("income_analysis") or "_No analysis available._")
    with tabs[2]:
        st.markdown(result.get("asset_analysis") or "_No analysis available._")
    with tabs[3]:
        st.markdown(result.get("collateral_analysis") or "_No analysis available._")
    with tabs[4]:
        st.markdown(result.get("critic_review") or "_No review available._")
    with tabs[5]:
        st.markdown(result.get("decision_memo") or "_No memo available._")
    with tabs[6]:
        st.markdown("**Reasoning chain:**")
        for step in result.get("reasoning_chain", []):
            st.markdown(f"- {step}")
