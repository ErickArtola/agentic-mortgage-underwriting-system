# agents.py — All 6 LangGraph agent nodes + PII/bias helpers
#
# LLM swap from notebook:
#   Notebook used ChatOpenAI (GreatLearning endpoint, gpt-4o-mini).
#   Here we use ChatGroq (llama-3.3-70b-versatile) — free tier, fast inference.

import os
import re
from typing import Dict, Any, List
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from models import UnderwritingState
from tools import (
    calculate_dti_ratio,
    calculate_ltv_ratio,
    calculate_reserves,
    calculate_housing_expense_ratio,
    check_credit_score_policy,
    check_large_deposits,
    calculate_total_debt_obligations,
)
from rag import retrieve_relevant_policies


# ---------------------------------------------------------------------------
# LLM initialization
# ---------------------------------------------------------------------------

def get_llm() -> ChatGroq:
    """Return a ChatGroq instance. Called at startup from main.py."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,          # Deterministic for consistent underwriting decisions
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# PII sanitization + bias detection
# ---------------------------------------------------------------------------

def sanitize_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact personally identifiable information before passing data to the LLM."""
    sanitized = data.copy()

    if "ssn" in sanitized:
        ssn = sanitized["ssn"]
        sanitized["ssn"] = f"***-**-{ssn[-4:]}" if len(ssn) >= 4 else "***-**-XXXX"

    if "name" in sanitized:
        sanitized["name"] = "[APPLICANT_NAME]"

    if "address" in sanitized:
        sanitized["address"] = "[ADDRESS]"

    if "phone" in sanitized:
        phone = sanitized["phone"]
        sanitized["phone"] = f"***-***-{phone[-4:]}" if len(phone) >= 4 else "***-***-XXXX"

    return sanitized


def detect_bias_signals(analysis: str, applicant_data: Dict[str, Any]) -> List[str]:
    """Scan LLM output for potential Fair Lending Act violations."""
    flags = []
    analysis_lower = analysis.lower()

    # Standard protected terms — simple substring match is sufficient
    protected_terms = [
        "race", "color", "religion", "national origin",
        "sex", "marital status", "gender",
        "disability", "familial status",
    ]
    for term in protected_terms:
        if term in analysis_lower:
            flags.append(f"Analysis mentions protected characteristic: {term}")

    # "age" needs special handling — credit analysis legitimately references
    # "account age", "tradeline age", "average age of credit history", etc.
    # Strip those benign phrases first, then check for a remaining standalone "age".
    _CREDIT_AGE_PHRASES = [
        "account age", "tradeline age", "age of account", "age of credit",
        "average age", "age of the account", "age of the tradeline",
        "age of tradeline", "credit age", "loan age", "file age",
        "oldest account", "oldest tradeline", "length of credit",
    ]
    age_text = analysis_lower
    for phrase in _CREDIT_AGE_PHRASES:
        age_text = age_text.replace(phrase, "")
    if re.search(r"\bage\b", age_text):
        flags.append("Analysis mentions protected characteristic: age")

    if "zip" in applicant_data or "zipcode" in applicant_data:
        if "neighborhood" in analysis_lower or "area" in analysis_lower:
            flags.append("Potential geographic bias — review for Fair Lending compliance")

    return flags


# ---------------------------------------------------------------------------
# Agent node factory helpers
# ---------------------------------------------------------------------------

def _make_agent_call(llm: ChatGroq, system_prompt: str, user_prompt: str) -> str:
    """Invoke the LLM with a system + user message pair. Returns response text."""
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    return response.content


# ---------------------------------------------------------------------------
# Initialize node
# ---------------------------------------------------------------------------

def initialize_application(state: UnderwritingState) -> UnderwritingState:
    """Sanitize PII and set up initial state fields."""
    sanitized = sanitize_pii(state["applicant_data"])

    return {
        **state,
        "sanitized_data": sanitized,
        "analysis_complete": False,
        "human_review_required": False,
        "human_review_completed": False,
        "bias_flags": [],
        "policy_violations": [],
        "reasoning_chain": [f"Application {state.get('case_id')} initialized"],
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Specialist agents
# ---------------------------------------------------------------------------

def credit_analyst_node(state: UnderwritingState, llm: ChatGroq, policy_store) -> UnderwritingState:
    """Evaluate credit score, payment history, and derogatory items."""
    policies = retrieve_relevant_policies(
        "credit score requirements bankruptcies foreclosures late payments",
        policy_store,
    )

    app_data = state["sanitized_data"]
    credit_score = app_data.get("credit_score", 0)
    credit_score_analysis = check_credit_score_policy.invoke({"credit_score": credit_score})

    system_prompt = f"""
You are a Senior Credit Analyst with 15+ years of experience in mortgage underwriting.

RELEVANT POLICIES:
{policies}

ANALYSIS FRAMEWORK:
1. Credit Score Assessment — use provided assessment (DO NOT recalculate)
2. Payment History — review late payments and patterns
3. Derogatory Items — evaluate bankruptcies, foreclosures, collections
4. Policy Compliance — check against credit guidelines
5. Risk Rating — assign credit risk (Low/Medium/High)
6. Recommendations — provide conditions or concerns

Be thorough, objective, and policy-compliant. Support conclusions with data.
IMPORTANT: Use the EXACT credit score assessment provided. Do not recalculate.
"""

    user_prompt = f"""
Analyze the credit profile for case {app_data.get('case_id')}:

CALCULATED CREDIT SCORE ASSESSMENT (ACCURATE — DO NOT RECALCULATE):
{credit_score_analysis}

CREDIT HISTORY DATA:
- Bankruptcies: {app_data.get('credit_history', {}).get('bankruptcies', 0)}
- Foreclosures: {app_data.get('credit_history', {}).get('foreclosures', 0)}
- Late Payments (12mo): {app_data.get('credit_history', {}).get('late_payments_12mo', 0)}
- Collections: {app_data.get('credit_history', {}).get('collections', [])}

Provide your detailed credit analysis.
"""

    analysis = _make_agent_call(llm, system_prompt, user_prompt)
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "credit_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + [
            f"Credit Analyst: Completed credit analysis for {app_data.get('case_id')}"
        ],
    }


def income_analyst_node(state: UnderwritingState, llm: ChatGroq, policy_store) -> UnderwritingState:
    """Verify employment stability, calculate DTI, and assess repayment capacity."""
    policies = retrieve_relevant_policies(
        "employment income verification DTI ratio self-employed",
        policy_store,
    )

    app_data = state["sanitized_data"]
    debts = {k: v for k, v in app_data.get("debts", {}).items() if k != "total_monthly_debt"}
    proposed_payment = app_data.get("loan", {}).get("estimated_payment", 0)
    monthly_income = app_data.get("employment", {}).get("monthly_income", 0)
    total_debt = app_data.get("debts", {}).get("total_monthly_debt", 0)

    dti_result = calculate_dti_ratio.invoke({"monthly_debt": total_debt, "monthly_income": monthly_income})
    housing_ratio_result = calculate_housing_expense_ratio.invoke({"monthly_payment": proposed_payment, "monthly_income": monthly_income})
    debt_breakdown = calculate_total_debt_obligations.invoke({"debts": debts, "proposed_payment": proposed_payment})

    system_prompt = f"""
You are a Senior Income Analyst with 10+ years of experience in mortgage underwriting.

RELEVANT POLICIES:
{policies}

ANALYSIS FRAMEWORK:
1. Employment Stability — review job history and tenure
2. Income Verification — validate income sources
3. DTI Calculation — use provided calculation (DO NOT recalculate)
4. Capacity for Monthly Payment — assess affordability
5. Risk Assessment — identify income risks

IMPORTANT: Use the EXACT DTI calculation provided. Do not recalculate.
"""

    user_prompt = f"""
Analyze the income and debt history for case {app_data.get('case_id')}:

CURRENT EMPLOYMENT:
- Employer: {app_data['employment']['employer']}
- Position: {app_data['employment']['position']}
- Employment Type: {app_data['employment']['type']}
- Years at Current Employer: {app_data['employment']['years']}
- Monthly Income: ${app_data['employment']['monthly_income']:,.2f}
- Employment Gap: {app_data['employment'].get('employment_gap', 'None')}
- Gap Explanation: {app_data['employment'].get('gap_explanation', 'N/A')}

EMPLOYMENT HISTORY:
{chr(10).join([
    f"  - {job['employer']} | {job['position']} | {job['years']} years | ${job['income']:,.2f}/yr"
    for job in app_data['employment'].get('employment_history', [])
])}

INCOME DETAILS:
- Base Salary: ${app_data['employment'].get('income_details', {}).get('base_salary', 0):,.2f}
- Bonus 2023: ${app_data['employment'].get('income_details', {}).get('bonus_2023', 0):,.2f}
- Bonus 2024: ${app_data['employment'].get('income_details', {}).get('bonus_2024', 0):,.2f}
- Bonus Stable: {app_data['employment'].get('income_details', {}).get('bonus_stable')}

DEBT OBLIGATIONS:
{debt_breakdown}

CALCULATED DTI RATIO (ACCURATE — DO NOT RECALCULATE):
{dti_result}

CALCULATED HOUSING EXPENSE RATIO (ACCURATE — DO NOT RECALCULATE):
{housing_ratio_result}

Provide your detailed income analysis.
"""

    analysis = _make_agent_call(llm, system_prompt, user_prompt)
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "income_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + [
            "Income Analyst: Completed income analysis with DTI calculation"
        ],
    }


def asset_analyst_node(state: UnderwritingState, llm: ChatGroq, policy_store) -> UnderwritingState:
    """Check reserves, down payment source, and large deposit documentation."""
    policies = retrieve_relevant_policies(
        "down payment reserves assets large deposits gift funds",
        policy_store,
    )

    app_data = state["sanitized_data"]
    assets = app_data.get("assets", {})
    loan = app_data.get("loan", {})
    monthly_income = app_data.get("employment", {}).get("monthly_income", 0)

    liquid_assets = assets.get("checking", 0) + assets.get("savings", 0)
    monthly_payment = loan.get("estimated_payment", 0)

    reserves_result = calculate_reserves.invoke({
        "liquid_assets": liquid_assets,
        "monthly_payment": monthly_payment,
        "required_months": 2,
    })

    deposits_result = check_large_deposits.invoke({
        "deposits": assets.get("recent_deposits", []),
        "monthly_income": monthly_income,
    })

    system_prompt = f"""
You are a Senior Asset Analyst with 10+ years of experience in mortgage underwriting.

RELEVANT POLICIES:
{policies}

ANALYSIS FRAMEWORK:
1. Down Payment Adequacy — verify sufficient funds
2. Reserve Requirements — use provided calculation
3. Large Deposits — use provided analysis
4. Source of Funds — ensure proper sourcing
5. Risk Assessment — identify asset-related risks
6. Documentation Needs — list required documents

IMPORTANT: Use the EXACT reserve calculation provided. Do not recalculate.
"""

    user_prompt = f"""
Analyze the borrower's assets and reserves for case {app_data.get('case_id')}:

ASSETS:
- Checking: ${assets.get('checking', 0):,.2f}
- Savings: ${assets.get('savings', 0):,.2f}
- Loan Amount: ${loan.get('amount', 0):,.2f}

CALCULATED RESERVES (ACCURATE — DO NOT RECALCULATE):
{reserves_result}

LARGE DEPOSIT ANALYSIS (ACCURATE — DO NOT RECALCULATE):
{deposits_result}

Provide your detailed asset analysis covering all six framework areas.
"""

    analysis = _make_agent_call(llm, system_prompt, user_prompt)
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "asset_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + [
            "Asset Analyst: Completed asset analysis and deposit review"
        ],
    }


def collateral_analyst_node(state: UnderwritingState, llm: ChatGroq, policy_store) -> UnderwritingState:
    """Assess property value, LTV ratio, and condition."""
    policies = retrieve_relevant_policies(
        "appraisal property condition LTV collateral",
        policy_store,
    )

    app_data = state["sanitized_data"]
    property_data = app_data.get("property", {})
    loan = app_data.get("loan", {})

    loan_amount = loan.get("amount", 0)
    appraised_value = property_data.get("appraised_value", 0)

    ltv_result = calculate_ltv_ratio.invoke({
        "loan_amount": loan_amount,
        "property_value": appraised_value,
    })

    system_prompt = f"""
You are a Senior Collateral Analyst with expertise in property valuation.

RELEVANT POLICIES:
{policies}

ANALYSIS FRAMEWORK:
1. Appraisal Review — validate property value
2. LTV Calculation — use provided calculation (DO NOT recalculate)
3. Property Condition — evaluate habitability
4. Marketability — consider market factors
5. Risk Assessment — identify collateral risks
6. Recommendations — note any concerns

IMPORTANT: Use the EXACT LTV calculation provided. Do not recalculate.
"""

    user_prompt = f"""
Analyze property collateral for case {app_data.get('case_id')}:

PROPERTY:
- Type: {property_data.get('type')}
- Appraised Value: ${appraised_value:,.2f}
- Condition: {property_data.get('condition')}
- Use: {loan.get('use')}

LOAN:
- Loan Amount: ${loan_amount:,.2f}
- Down Payment: ${loan.get('down_payment', 0):,.2f}

CALCULATED LTV (ACCURATE — DO NOT RECALCULATE):
{ltv_result}

Provide your collateral analysis.
"""

    analysis = _make_agent_call(llm, system_prompt, user_prompt)
    bias_flags = detect_bias_signals(analysis, app_data)

    return {
        **state,
        "collateral_analysis": analysis,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + [
            "Collateral Analyst: Completed property analysis (LTV from tool)"
        ],
    }


def critic_agent_node(state: UnderwritingState, llm: ChatGroq) -> UnderwritingState:
    """Review all four specialist analyses for contradictions and completeness."""
    system_prompt = """
You are a Quality Assurance Critic reviewing underwriting analyses.

Your role is to:
1. Verify all analyses are complete and thorough
2. Identify any contradictions or inconsistencies
3. Ensure policy compliance
4. Flag any missing information
5. Provide a synthesis of key findings

Be critical but fair. Focus on ensuring decision quality.
"""

    user_prompt = f"""
Review all analyses for case {state.get('case_id')}:

CREDIT ANALYSIS:
{state.get('credit_analysis', 'Not completed')}

INCOME ANALYSIS:
{state.get('income_analysis', 'Not completed')}

ASSET ANALYSIS:
{state.get('asset_analysis', 'Not completed')}

COLLATERAL ANALYSIS:
{state.get('collateral_analysis', 'Not completed')}

BIAS FLAGS:
{state.get('bias_flags', [])}

Provide your critical review and synthesis.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {
        **state,
        "critic_review": response.content,
        "reasoning_chain": state.get("reasoning_chain", []) + [
            "Critic: Completed review of all specialist analyses"
        ],
    }


def decision_agent_node(state: UnderwritingState, llm: ChatGroq) -> UnderwritingState:
    """Synthesize all findings into a risk score and final decision."""
    system_prompt = """
You are a Senior Underwriter who synthesizes all specialist analyses.

Assign a risk score from 0 to 100 by ADDING points for each applicable risk factor below.
Apply every rule that fits — scores accumulate.

CREDIT SCORE:
- Below 620: +35 points
- 620–679: +20 points
- 680–719: +15 points
- 720–759: +5 points
- 760 or above: +0 points

DEBT-TO-INCOME RATIO (DTI):
- Above 50%: +35 points
- 43%–50% inclusive: +25 points
- 38%–42.9%: +15 points
- Below 38%: +0 points

LOAN-TO-VALUE RATIO (LTV):
- Above 97%: +20 points
- 90.1%–97%: +15 points
- 80.1%–90%: +5 points
- 80% or below: +0 points

DEROGATORY HISTORY:
- Bankruptcy in last 7 years: +15 points
- Foreclosure in last 7 years: +15 points
- 3 or more late payments in last 12 months: +10 points
- 1–2 late payments in last 12 months: +5 points
- Collections present: +5 points

EMPLOYMENT / INCOME STABILITY:
- Employment gap present OR less than 2 years at current employer: +5 points
- Self-employed with declining income: +10 points

PROPERTY / COLLATERAL:
- Required repairs above $5,000: +5 points
- Property condition C4 or worse: +5 points

OFFSETTING STRENGTHS (subtract points):
- Credit 760+, DTI below 36%, and no derogatory items: -10 points

DECISION MAPPING:
- 0–24: APPROVED
- 25–64: CONDITIONAL_APPROVAL
- 65–100: DENIED

OUTPUT FORMAT (required — must appear exactly):
RISK_SCORE: [number]
DECISION: [APPROVED/CONDITIONAL_APPROVAL/DENIED]
CREDIT_MEMO: [your detailed rationale showing which risk factors you applied and their point values]
"""

    user_prompt = f"""
Make final underwriting decision for case {state.get('case_id')}:

CREDIT ANALYSIS SUMMARY:
{str(state.get('credit_analysis', 'N/A'))[:500]}...

INCOME ANALYSIS SUMMARY:
{str(state.get('income_analysis', 'N/A'))[:500]}...

ASSET ANALYSIS SUMMARY:
{str(state.get('asset_analysis', 'N/A'))[:500]}...

COLLATERAL ANALYSIS SUMMARY:
{str(state.get('collateral_analysis', 'N/A'))[:500]}...

CRITIC REVIEW:
{str(state.get('critic_review', 'N/A'))[:500]}...

COMPLIANCE ALERTS:
- Bias Flags: {len(state.get('bias_flags', []))}
- Policy Violations: {len(state.get('policy_violations', []))}

Provide RISK_SCORE, DECISION, and CREDIT_MEMO.
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    content = response.content

    # Parse risk score — clamp to [0, 100]
    risk_score = 50  # safe default
    match = re.search(r"RISK_SCORE:\s*(-?\d+)", content)
    if match:
        risk_score = max(0, min(100, int(match.group(1))))

    # Python makes the final decision from the numeric risk score — more reliable
    # than parsing the LLM's text output, which can misread boundary cases.
    # Boundaries: 0-24 APPROVED | 25-64 CONDITIONAL_APPROVAL | 65-100 DENIED
    if risk_score <= 24:
        decision = "APPROVED"
    elif risk_score <= 64:
        decision = "CONDITIONAL_APPROVAL"
    else:
        decision = "DENIED"

    human_review_required = (
        risk_score >= 65
        or len(state.get("bias_flags", [])) > 0
        or decision == "DENIED"
    )

    return {
        **state,
        "decision_memo": content,
        "risk_score": risk_score,
        "final_decision": decision,
        "human_review_required": human_review_required,
        "reasoning_chain": state.get("reasoning_chain", []) + [
            f"Decision Agent: Final decision {decision} with risk score {risk_score}"
        ],
    }
