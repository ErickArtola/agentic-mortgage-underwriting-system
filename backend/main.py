# main.py — FastAPI application entry point
#
# Endpoints:
#   GET  /health        → liveness check
#   POST /analyze       → run the full underwriting workflow

import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from agents import get_llm
from rag import create_policy_store
from workflow import build_workflow
from models import ApplicationRequest, UnderwritingResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App-level state (initialized once at startup)
# ---------------------------------------------------------------------------

app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LLM, policy store, and workflow graph at startup."""
    logger.info("Initializing LLM (Groq)...")
    app_state["llm"] = get_llm()

    logger.info("Loading policy store (ChromaDB + HuggingFace embeddings)...")
    app_state["policy_store"] = create_policy_store()

    logger.info("Compiling LangGraph workflow...")
    app_state["graph"] = build_workflow(app_state["llm"], app_state["policy_store"])

    logger.info("✅ Application ready.")
    yield

    # Cleanup (nothing to do for in-memory stores)
    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mortgage Underwriting API",
    description=(
        "Multi-agent mortgage underwriting system. "
        "Powered by LangGraph, FastAPI, and Groq (llama-3.3-70b-versatile)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit Community Cloud and localhost during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to your Streamlit URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    """Liveness check — confirms the API is up and the policy store is loaded."""
    return HealthResponse(
        status="ok",
        model="llama-3.3-70b-versatile",
        policy_store_ready="policy_store" in app_state,
    )


@app.post("/analyze", response_model=UnderwritingResponse, tags=["Underwriting"])
async def analyze(application: ApplicationRequest):
    """
    Run a mortgage application through the full multi-agent underwriting pipeline.

    Accepts a structured loan application and returns a risk score, decision
    (APPROVED / CONDITIONAL_APPROVAL / DENIED), and full audit trail.
    """
    if "graph" not in app_state:
        raise HTTPException(status_code=503, detail="Workflow not initialized. Try again shortly.")

    graph = app_state["graph"]

    # Convert Pydantic model → plain dict for LangGraph state
    applicant_data = application.model_dump(by_alias=True)

    # Unique thread ID so MemorySaver doesn't mix up concurrent requests
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "case_id": application.case_id,
        "applicant_data": applicant_data,
        "sanitized_data": {},
        "credit_analysis": None,
        "income_analysis": None,
        "asset_analysis": None,
        "collateral_analysis": None,
        "critic_review": None,
        "decision_memo": None,
        "final_decision": None,
        "risk_score": None,
        "next_agent": None,
        "analysis_complete": False,
        "human_review_required": False,
        "human_review_completed": False,
        "human_notes": None,
        "bias_flags": [],
        "policy_violations": [],
        "reasoning_chain": [],
        "timestamp": None,
    }

    try:
        logger.info(f"Starting underwriting workflow for case {application.case_id}")
        final_state = await graph.ainvoke(initial_state, config=config)
        logger.info(f"Workflow complete for {application.case_id}: {final_state.get('final_decision')}")
    except Exception as e:
        logger.error(f"Workflow error for {application.case_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")

    return UnderwritingResponse(
        case_id=final_state.get("case_id", application.case_id),
        final_decision=final_state.get("final_decision", "CONDITIONAL_APPROVAL"),
        risk_score=final_state.get("risk_score", 50),
        human_review_required=final_state.get("human_review_required", False),
        credit_analysis=final_state.get("credit_analysis"),
        income_analysis=final_state.get("income_analysis"),
        asset_analysis=final_state.get("asset_analysis"),
        collateral_analysis=final_state.get("collateral_analysis"),
        critic_review=final_state.get("critic_review"),
        decision_memo=final_state.get("decision_memo"),
        bias_flags=final_state.get("bias_flags", []),
        policy_violations=final_state.get("policy_violations", []),
        reasoning_chain=final_state.get("reasoning_chain", []),
        timestamp=final_state.get("timestamp"),
    )
