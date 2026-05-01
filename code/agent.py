"""Core orchestration for ticket triage."""

from __future__ import annotations

from typing import Any

import pandas as pd

from classifier import classify_ticket
from response_generator import generate_response
from retriever import retrieve_context
from safety import evaluate_safety


def _clean_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _normalized_ticket(ticket: dict[str, Any]) -> dict[str, str]:
    return {
        "issue": _clean_value(ticket.get("Issue", ticket.get("issue", ""))),
        "subject": _clean_value(ticket.get("Subject", ticket.get("subject", ""))),
        "company": _clean_value(ticket.get("Company", ticket.get("company", ""))),
    }


def _safe_canned_reply(safety: dict, classification: dict) -> dict:
    reason_code = safety.get("reason_code", "")
    if reason_code == "safe_smalltalk":
        response = "Happy to help"
        request_type = "invalid"
        product_area = None
    elif reason_code == "out_of_scope_safe_reply":
        response = "I am sorry, this is out of scope from my capabilities"
        request_type = "invalid"
        # Sample labels map this case to conversation_management.
        product_area = "conversation_management"
    else:
        response = "Escalate to a human"
        request_type = classification.get("request_type_hint", "product_issue")
        product_area = (
            None
            if classification.get("domain") in {"none", ""}
            else classification.get("product_area_hint", "general")
        )
    return {
        "status": "replied" if reason_code in {"safe_smalltalk", "out_of_scope_safe_reply"} else "escalated",
        "product_area": product_area,
        "response": response,
        "justification": safety.get("reason_text", "Safety policy decision."),
        "request_type": request_type,
    }


def triage_ticket(ticket: dict) -> dict:
    """Triage one ticket into required output fields."""
    t = _normalized_ticket(ticket)
    query = f"{t['subject']} {t['issue']}".strip()
    contexts = retrieve_context(
        query,
        company=t["company"],
        top_k=3,
    )
    classification = classify_ticket(
        t["issue"],
        t["subject"],
        t["company"],
        retrieved_contexts=contexts,
    )
    safety = evaluate_safety(t["issue"], t["subject"], t["company"])

    if safety["should_escalate"] or safety["reason_code"] in {"safe_smalltalk", "out_of_scope_safe_reply"}:
        return _safe_canned_reply(safety, classification)

    llm_out = generate_response(
        t,
        contexts,
        classification,
        allowed_product_areas=classification.get("product_area_candidates", [classification.get("product_area_hint", "general")]),
    )
    status = llm_out.status.lower().strip()
    if status not in {"replied", "escalated"}:
        status = "replied"
    if safety.get("reason_code") == "safe_to_answer" and status == "escalated" and contexts:
        status = "replied"

    req = llm_out.request_type.lower().strip()
    if req not in {"product_issue", "feature_request", "bug", "invalid"}:
        req = classification.get("request_type_hint", "product_issue")

    return {
        "status": status,
        # Keep product_area deterministic from calibrated classifier vote.
        "product_area": classification.get("product_area_hint", "general"),
        "response": llm_out.response,
        "justification": llm_out.justification,
        "request_type": req,
    }


def run_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run triage on each row and return output-shaped DataFrame."""
    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        ticket = _normalized_ticket(row.to_dict())
        result = triage_ticket(row.to_dict())
        rows.append(
            {
                "issue": ticket["issue"],
                "subject": ticket["subject"],
                "company": ticket["company"],
                "response": result["response"],
                "product_area": result["product_area"],
                "status": result["status"],
                "request_type": result["request_type"],
                "justification": result["justification"],
            }
        )
    return pd.DataFrame(rows)
