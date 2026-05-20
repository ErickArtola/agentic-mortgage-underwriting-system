# tools.py — Deterministic calculation tools for mortgage underwriting
# All arithmetic is done here by Python, not by the LLM, to prevent hallucinations.

from langchain_core.tools import tool


@tool
def calculate_dti_ratio(monthly_debt: float, monthly_income: float) -> str:
    """Calculate Debt-to-Income ratio.

    Args:
        monthly_debt: Total monthly debt obligations
        monthly_income: Total monthly gross income

    Returns:
        Formatted DTI ratio result
    """
    if monthly_income <= 0:
        return "Error: Monthly income must be greater than 0"

    dti = (monthly_debt / monthly_income) * 100

    result = {
        "dti_ratio": round(dti, 2),
        "monthly_debt": round(monthly_debt, 2),
        "monthly_income": round(monthly_income, 2),
        "status": "Acceptable" if dti <= 43 else "High" if dti <= 50 else "Excessive"
    }

    return (
        f"DTI Ratio: {result['dti_ratio']}% ({result['status']}) - "
        f"Debt: ${result['monthly_debt']:,.2f}, Income: ${result['monthly_income']:,.2f}"
    )


@tool
def calculate_ltv_ratio(loan_amount: float, property_value: float) -> str:
    """Calculate Loan-to-Value ratio.

    Args:
        loan_amount: The loan amount requested
        property_value: Appraised value of the property

    Returns:
        Formatted LTV ratio result
    """
    if property_value <= 0:
        return "Error: Property value must be greater than 0"

    ltv = (loan_amount / property_value) * 100

    result = {
        "ltv_ratio": round(ltv, 2),
        "loan_amount": round(loan_amount, 2),
        "property_value": round(property_value, 2),
        "status": "Excellent" if ltv <= 80 else "Good" if ltv <= 90 else "High" if ltv <= 97 else "Excessive"
    }

    return (
        f"LTV Ratio: {result['ltv_ratio']}% ({result['status']}) - "
        f"Loan: ${result['loan_amount']:,.2f}, Value: ${result['property_value']:,.2f}"
    )


@tool
def calculate_reserves(liquid_assets: float, monthly_payment: float, required_months: int = 2) -> str:
    """Calculate reserve coverage in months.

    Args:
        liquid_assets: Total liquid assets available
        monthly_payment: Monthly PITI payment
        required_months: Number of months reserves required (default 2)

    Returns:
        Formatted reserves analysis
    """
    if monthly_payment <= 0:
        return "Error: Monthly payment must be greater than 0"

    months_coverage = liquid_assets / monthly_payment
    required_amount = monthly_payment * required_months

    result = {
        "months_coverage": round(months_coverage, 1),
        "liquid_assets": round(liquid_assets, 2),
        "required_amount": round(required_amount, 2),
        "surplus_deficit": round(liquid_assets - required_amount, 2),
        "status": "Adequate" if months_coverage >= required_months else "Insufficient"
    }

    return (
        f"Reserves: {result['months_coverage']} months coverage ({result['status']}) - "
        f"Assets: ${result['liquid_assets']:,.2f}, Required: ${result['required_amount']:,.2f}"
    )


@tool
def calculate_housing_expense_ratio(monthly_payment: float, monthly_income: float) -> str:
    """Calculate housing expense ratio (front-end ratio).

    Args:
        monthly_payment: Monthly PITI payment
        monthly_income: Total monthly gross income

    Returns:
        Formatted housing expense ratio
    """
    if monthly_income <= 0:
        return "Error: Monthly income must be greater than 0"

    ratio = (monthly_payment / monthly_income) * 100

    result = {
        "housing_ratio": round(ratio, 2),
        "monthly_payment": round(monthly_payment, 2),
        "monthly_income": round(monthly_income, 2),
        "status": "Acceptable" if ratio <= 28 else "Elevated" if ratio <= 35 else "High"
    }

    return (
        f"Housing Ratio: {result['housing_ratio']}% ({result['status']}) - "
        f"Payment: ${result['monthly_payment']:,.2f}, Income: ${result['monthly_income']:,.2f}"
    )


@tool
def check_credit_score_policy(credit_score: int) -> str:
    """Check if credit score meets policy requirements.

    Args:
        credit_score: Borrower's credit score

    Returns:
        Policy compliance result
    """
    if credit_score >= 740:
        tier = "Excellent"
        rate_adjustment = "Best rates available"
    elif credit_score >= 700:
        tier = "Very Good"
        rate_adjustment = "Favorable rates"
    elif credit_score >= 660:
        tier = "Good"
        rate_adjustment = "Standard rates"
    elif credit_score >= 620:
        tier = "Fair"
        rate_adjustment = "Higher rates, may require compensating factors"
    else:
        tier = "Below Minimum"
        rate_adjustment = "Does not meet conventional loan requirements"

    return f"Credit Score: {credit_score} - Tier: {tier} - {rate_adjustment}"


@tool
def check_large_deposits(deposits: list, monthly_income: float) -> str:
    """Identify large deposits requiring sourcing documentation.

    Args:
        deposits: List of recent deposits [{'amount': float, 'date': str}, ...]
        monthly_income: Monthly income for threshold calculation

    Returns:
        Analysis of deposits requiring documentation
    """
    threshold = monthly_income * 0.25  # 25% of monthly income
    large_deposits = []

    for deposit in deposits:
        amount = deposit.get('amount', 0)
        if amount >= threshold:
            large_deposits.append({
                'amount': amount,
                'date': deposit.get('date', 'Unknown'),
                'sourcing_required': True
            })

    if not large_deposits:
        return f"No large deposits identified (threshold: ${threshold:,.2f}). All deposits are acceptable."

    result = f"Found {len(large_deposits)} large deposit(s) requiring documentation (threshold: ${threshold:,.2f}):\n"
    for i, dep in enumerate(large_deposits, 1):
        result += f"  {i}. ${dep['amount']:,.2f} on {dep['date']} - Sourcing documentation required\n"

    return result


@tool
def calculate_total_debt_obligations(debts: dict, proposed_payment: float) -> str:
    """Calculate total monthly debt obligations including proposed loan.

    Args:
        debts: Dictionary of current debts {'debt_name': amount, ...}
        proposed_payment: Proposed monthly loan payment

    Returns:
        Total debt calculation
    """
    current_debt = sum(debts.values())
    total_obligation = current_debt + proposed_payment

    result = {
        "current_debt": round(current_debt, 2),
        "proposed_payment": round(proposed_payment, 2),
        "total_obligation": round(total_obligation, 2),
        "debt_breakdown": {k: round(v, 2) for k, v in debts.items()}
    }

    breakdown = "\n".join([f"  - {k}: ${v:,.2f}" for k, v in result['debt_breakdown'].items()])

    return (
        f"Total Monthly Obligations: ${result['total_obligation']:,.2f}\n"
        f"Current Debt: ${result['current_debt']:,.2f}\n"
        f"{breakdown}\n"
        f"Proposed Payment: ${result['proposed_payment']:,.2f}"
    )
