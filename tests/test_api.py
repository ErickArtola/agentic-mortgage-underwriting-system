# tests/test_api.py — pytest tests for the mortgage underwriting API
#
# Run from the repo root:
#   cd backend && pip install -r requirements.txt
#   cd ../tests && pytest test_api.py -v
#
# These tests use FastAPI's TestClient so no running server is needed.
# The workflow is NOT invoked — agents are mocked to keep tests fast and free.

import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add backend to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

VALID_APPLICATION = {
    "case_id": "TEST-001",
    "name": "Jane Doe",
    "ssn": "123-45-6789",
    "email": "jane@example.com",
    "phone": "555-123-4567",
    "address": "123 Main St, Springfield, CA 90001",
    "credit_score": 720,
    "credit_history": {
        "bankruptcies": 0,
        "foreclosures": 0,
        "late_payments_12mo": 0,
        "late_payments_24mo": 0,
        "collections": [],
        "inquiries_6mo": 1,
        "oldest_tradeline_years": 10,
        "total_tradelines": 8,
        "credit_notes": "Good payment history."
    },
    "employment": {
        "employer": "Acme Corp",
        "position": "Engineer",
        "years": 5.0,
        "monthly_income": 10000,
        "type": "W2",
        "employment_gap": "None",
        "gap_explanation": "N/A",
        "employment_history": [],
        "income_details": {
            "base_salary": 120000,
            "bonus_2023": 5000,
            "bonus_2024": 6000,
            "bonus_stable": True,
            "employer_confirmation": "Confirmed"
        }
    },
    "debts": {
        "car_loan": 400,
        "credit_cards": 600,
        "total_monthly_debt": 1000
    },
    "assets": {
        "checking": 50000,
        "savings": 40000,
        "liquid_assets_total": 90000,
        "401k": 100000,
        "recent_deposits": [],
        "deposit_explanations": "",
        "reserves_months": 6
    },
    "loan": {
        "amount": 300000,
        "down_payment": 60000,
        "closing_costs": 8000,
        "estimated_payment": 2000,
        "property_type": "Single Family",
        "use": "Primary Residence",
        "monthly_piti": 2000
    },
    "property": {
        "purchase_price": 360000,
        "appraised_value": 365000,
        "condition": "C2 - Good",
        "type": "Single Family Home",
        "required_repairs": 0,
        "repair_details": "None"
    }
}

MOCK_FINAL_STATE = {
    "case_id": "TEST-001",
    "final_decision": "APPROVED",
    "risk_score": 20,
    "human_review_required": False,
    "credit_analysis": "Credit looks good.",
    "income_analysis": "Income stable.",
    "asset_analysis": "Assets adequate.",
    "collateral_analysis": "LTV acceptable.",
    "critic_review": "All analyses consistent.",
    "decision_memo": "RISK_SCORE: 20\nDECISION: APPROVED\nCREDIT_MEMO: Strong application.",
    "bias_flags": [],
    "policy_violations": [],
    "reasoning_chain": ["Initialized", "Credit done", "Income done", "Asset done", "Collateral done", "Decision: APPROVED"],
    "timestamp": "2025-01-01T00:00:00",
}


@pytest.fixture(scope="module")
def client():
    """TestClient with LLM, policy store, and graph mocked out."""
    mock_llm = MagicMock()
    mock_store = MagicMock()
    mock_graph = MagicMock()
    mock_graph.ainvoke = MagicMock(return_value=MOCK_FINAL_STATE)

    with patch("main.get_llm", return_value=mock_llm), \
         patch("main.create_policy_store", return_value=mock_store), \
         patch("main.build_workflow", return_value=mock_graph):
        from main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_shape(client):
    data = response = client.get("/health").json()
    assert "status" in data
    assert "model" in data
    assert "policy_store_ready" in data


def test_health_status_is_ok(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /analyze — valid application
# ---------------------------------------------------------------------------

def test_analyze_returns_200(client):
    response = client.post("/analyze", json=VALID_APPLICATION)
    assert response.status_code == 200


def test_analyze_response_has_required_fields(client):
    data = client.post("/analyze", json=VALID_APPLICATION).json()
    for field in ["case_id", "final_decision", "risk_score", "human_review_required"]:
        assert field in data, f"Missing field: {field}"


def test_analyze_decision_is_valid_value(client):
    data = client.post("/analyze", json=VALID_APPLICATION).json()
    assert data["final_decision"] in {"APPROVED", "CONDITIONAL_APPROVAL", "DENIED"}


def test_analyze_risk_score_in_range(client):
    data = client.post("/analyze", json=VALID_APPLICATION).json()
    assert 0 <= data["risk_score"] <= 100


def test_analyze_returns_correct_case_id(client):
    data = client.post("/analyze", json=VALID_APPLICATION).json()
    assert data["case_id"] == "TEST-001"


def test_analyze_reasoning_chain_is_list(client):
    data = client.post("/analyze", json=VALID_APPLICATION).json()
    assert isinstance(data["reasoning_chain"], list)


# ---------------------------------------------------------------------------
# POST /analyze — Pydantic validation rejections
# ---------------------------------------------------------------------------

def test_analyze_rejects_missing_case_id(client):
    bad = {k: v for k, v in VALID_APPLICATION.items() if k != "case_id"}
    response = client.post("/analyze", json=bad)
    assert response.status_code == 422


def test_analyze_rejects_invalid_credit_score(client):
    bad = {**VALID_APPLICATION, "credit_score": 9999}
    response = client.post("/analyze", json=bad)
    assert response.status_code == 422


def test_analyze_rejects_zero_monthly_income(client):
    bad_employment = {**VALID_APPLICATION["employment"], "monthly_income": 0}
    bad = {**VALID_APPLICATION, "employment": bad_employment}
    response = client.post("/analyze", json=bad)
    assert response.status_code == 422


def test_analyze_rejects_negative_loan_amount(client):
    bad_loan = {**VALID_APPLICATION["loan"], "amount": -50000}
    bad = {**VALID_APPLICATION, "loan": bad_loan}
    response = client.post("/analyze", json=bad)
    assert response.status_code == 422
