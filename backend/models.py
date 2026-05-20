# models.py — Pydantic request/response schemas + LangGraph state definition

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# LangGraph State (TypedDict — used internally by the workflow)
# ---------------------------------------------------------------------------

class UnderwritingState(TypedDict):
    """Complete state of a loan application as it moves through the agent pipeline."""

    # Application inputs
    case_id: str
    applicant_data: Dict[str, Any]
    sanitized_data: Dict[str, Any]

    # Specialist agent outputs (None until each agent runs)
    credit_analysis: Optional[str]
    income_analysis: Optional[str]
    asset_analysis: Optional[str]
    collateral_analysis: Optional[str]

    # Coordination / decision
    critic_review: Optional[str]
    decision_memo: Optional[str]
    final_decision: Optional[str]
    risk_score: Optional[int]

    # Workflow control
    next_agent: Optional[str]
    analysis_complete: bool
    human_review_required: bool
    human_review_completed: bool
    human_notes: Optional[str]

    # Compliance / audit
    bias_flags: List[str]
    policy_violations: List[str]
    reasoning_chain: Annotated[List[str], "append"]
    timestamp: Optional[str]


# ---------------------------------------------------------------------------
# Pydantic models — FastAPI request validation
# ---------------------------------------------------------------------------

class CreditHistoryInput(BaseModel):
    bankruptcies: int = 0
    foreclosures: int = 0
    late_payments_12mo: int = 0
    late_payments_24mo: int = 0
    collections: List[Any] = Field(default_factory=list)
    inquiries_6mo: int = 0
    oldest_tradeline_years: float = 0
    total_tradelines: int = 0
    credit_notes: str = ""


class EmploymentHistoryItem(BaseModel):
    employer: str
    position: str
    years: float
    income: float


class IncomeDetails(BaseModel):
    base_salary: float = 0
    bonus_2023: float = 0
    bonus_2024: float = 0
    bonus_stable: bool = False
    employer_confirmation: str = ""


class EmploymentInput(BaseModel):
    employer: str
    position: str
    years: float
    monthly_income: float = Field(..., gt=0, description="Monthly gross income must be positive")
    type: str = Field(..., description="W2, Self-Employed, or 1099")
    employment_gap: str = "None"
    gap_explanation: str = "N/A"
    employment_history: List[EmploymentHistoryItem] = Field(default_factory=list)
    income_details: IncomeDetails = Field(default_factory=IncomeDetails)


class DebtsInput(BaseModel):
    """Dynamic debt fields — any key is a debt type, value is monthly payment."""
    model_config = {"extra": "allow"}

    total_monthly_debt: float = Field(..., ge=0)


class DepositItem(BaseModel):
    date: str
    amount: float
    description: str = ""


class AssetsInput(BaseModel):
    checking: float = Field(default=0, ge=0)
    savings: float = Field(default=0, ge=0)
    liquid_assets_total: float = Field(default=0, ge=0)
    retirement_401k: float = Field(default=0, ge=0, alias="401k")
    recent_deposits: List[DepositItem] = Field(default_factory=list)
    deposit_explanations: str = ""
    reserves_months: float = 0

    model_config = {"populate_by_name": True}


class LoanInput(BaseModel):
    amount: float = Field(..., gt=0)
    down_payment: float = Field(..., ge=0)
    closing_costs: float = Field(default=0, ge=0)
    estimated_payment: float = Field(..., gt=0, description="Estimated monthly PITI payment")
    property_type: str = ""
    use: str = Field(default="Primary Residence")
    monthly_piti: float = 0


class PropertyInput(BaseModel):
    purchase_price: float = Field(..., gt=0)
    appraised_value: float = Field(..., gt=0)
    condition: str = ""
    type: str = ""
    required_repairs: float = 0
    repair_details: str = ""


class ApplicationRequest(BaseModel):
    """Incoming loan application — validated by FastAPI before reaching the workflow."""

    case_id: str = Field(..., description="Unique case identifier, e.g. MTG-2025-001")
    name: str
    ssn: str = Field(..., min_length=9, description="Social Security Number")
    email: str = ""
    phone: str = ""
    address: str = ""
    credit_score: int = Field(..., ge=300, le=850)
    credit_history: CreditHistoryInput
    employment: EmploymentInput
    debts: Dict[str, Any] = Field(..., description="Debt obligations including total_monthly_debt")
    assets: AssetsInput
    loan: LoanInput
    property: PropertyInput

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Pydantic models — FastAPI response
# ---------------------------------------------------------------------------

class UnderwritingResponse(BaseModel):
    """Structured response returned from POST /analyze."""

    case_id: str
    final_decision: str = Field(..., description="APPROVED, CONDITIONAL_APPROVAL, or DENIED")
    risk_score: int = Field(..., ge=0, le=100)
    human_review_required: bool

    # Full agent outputs
    credit_analysis: Optional[str] = None
    income_analysis: Optional[str] = None
    asset_analysis: Optional[str] = None
    collateral_analysis: Optional[str] = None
    critic_review: Optional[str] = None
    decision_memo: Optional[str] = None

    # Compliance
    bias_flags: List[str] = Field(default_factory=list)
    policy_violations: List[str] = Field(default_factory=list)

    # Audit trail
    reasoning_chain: List[str] = Field(default_factory=list)
    timestamp: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    policy_store_ready: bool
