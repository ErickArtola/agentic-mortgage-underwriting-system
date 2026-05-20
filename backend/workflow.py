# workflow.py — LangGraph StateGraph definition
#
# Exports build_workflow(llm, policy_store) which returns a compiled graph.
# The llm and policy_store are injected at startup so agents don't
# need to reinitialize them on every request.

import functools

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

from models import UnderwritingState
from agents import (
    initialize_application,
    credit_analyst_node,
    income_analyst_node,
    asset_analyst_node,
    collateral_analyst_node,
    critic_agent_node,
    decision_agent_node,
)


# ---------------------------------------------------------------------------
# Supervisor routing helpers
# ---------------------------------------------------------------------------

def supervisor_node(state: UnderwritingState) -> UnderwritingState:
    """Decide which specialist agent runs next, or signal all analyses are done."""
    analyses_done = {
        "credit": state.get("credit_analysis") is not None,
        "income": state.get("income_analysis") is not None,
        "asset": state.get("asset_analysis") is not None,
        "collateral": state.get("collateral_analysis") is not None,
    }

    if not analyses_done["credit"]:
        next_agent = "credit"
    elif not analyses_done["income"]:
        next_agent = "income"
    elif not analyses_done["asset"]:
        next_agent = "asset"
    elif not analyses_done["collateral"]:
        next_agent = "collateral"
    else:
        next_agent = "critic"

    return {
        **state,
        "next_agent": next_agent,
        "analysis_complete": all(analyses_done.values()),
    }


def should_continue_to_agents(state: UnderwritingState) -> str:
    """Conditional edge: route to the next specialist, or advance to critic."""
    if state.get("analysis_complete", False):
        return "critic"
    return state.get("next_agent", "credit")


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_workflow(llm: ChatGroq, policy_store):
    """Build and compile the full underwriting StateGraph.

    Args:
        llm: ChatGroq instance (initialized once at app startup).
        policy_store: ChromaDB instance (loaded once at app startup).

    Returns:
        Compiled LangGraph CompiledGraph ready to invoke.
    """
    # Bind llm + policy_store into agent functions using functools.partial
    # so LangGraph node signatures remain (state) -> state
    credit_node   = functools.partial(credit_analyst_node,   llm=llm, policy_store=policy_store)
    income_node   = functools.partial(income_analyst_node,   llm=llm, policy_store=policy_store)
    asset_node    = functools.partial(asset_analyst_node,    llm=llm, policy_store=policy_store)
    collateral_node = functools.partial(collateral_analyst_node, llm=llm, policy_store=policy_store)
    critic_node   = functools.partial(critic_agent_node,     llm=llm)
    decision_node = functools.partial(decision_agent_node,   llm=llm)

    workflow = StateGraph(UnderwritingState)

    # Register nodes
    workflow.add_node("initialize",  initialize_application)
    workflow.add_node("supervisor",  supervisor_node)
    workflow.add_node("credit",      credit_node)
    workflow.add_node("income",      income_node)
    workflow.add_node("asset",       asset_node)
    workflow.add_node("collateral",  collateral_node)
    workflow.add_node("critic",      critic_node)
    workflow.add_node("decision",    decision_node)

    # Entry point and fixed edges
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "supervisor")

    # Supervisor conditional routing → specialist agents or critic
    workflow.add_conditional_edges(
        "supervisor",
        should_continue_to_agents,
        {
            "credit":     "credit",
            "income":     "income",
            "asset":      "asset",
            "collateral": "collateral",
            "critic":     "critic",
        },
    )

    # All specialists loop back to supervisor
    workflow.add_edge("credit",     "supervisor")
    workflow.add_edge("income",     "supervisor")
    workflow.add_edge("asset",      "supervisor")
    workflow.add_edge("collateral", "supervisor")

    # Critic → Decision → END
    workflow.add_edge("critic",   "decision")
    workflow.add_edge("decision", END)

    # Compile with in-memory checkpointing
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
