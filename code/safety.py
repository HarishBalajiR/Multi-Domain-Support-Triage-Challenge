"""Safety and escalation rules for support triage."""

from __future__ import annotations

import re

from utils import normalize_text


_ESCALATION_RULES: list[tuple[str, str, str, str]] = [
    (
        "account_access",
        r"\b(restore my access|lost access|unlock account|removed my seat|unauthorized login)\b",
        "high",
        "Account access and permissions require human verification.",
    ),
    (
        "score_manipulation",
        r"\b(increase my score|review my answers|move me to the next round|graded me unfairly)\b",
        "high",
        "Manual score/routing intervention requests require escalation.",
    ),
    (
        "payments_fraud",
        r"\b(fraud|fraudulent|stolen card|chargeback|refund me today|ban the seller|double charged)\b",
        "high",
        "Payment and fraud-related requests require human handling.",
    ),
    (
        "security_pii",
        r"\b(otp|ssn|aadhaar|credit card number|cvv|private key|share.*password|my password is)\b",
        "high",
        "Sensitive data and security operations should be escalated.",
    ),
    (
        "platform_outage",
        r"\b(site is down|all pages are inaccessible|service outage|everything is down)\b",
        "high",
        "Potential incident/outage reports should go to humans.",
    ),
    (
        "prompt_injection",
        r"\b(ignore previous instructions|you are now|override policy|bypass safety)\b",
        "high",
        "Prompt-injection style instructions must be escalated.",
    ),
]


def evaluate_safety(issue: str, subject: str, company: str = "") -> dict:
    """Return escalation decision and justification metadata."""
    issue_n = normalize_text(issue or "")
    subject_n = normalize_text(subject or "")
    text = f"{subject_n} {issue_n}".strip().lower()
    company_n = normalize_text(company or "").lower()

    if not text:
        return {
            "should_escalate": True,
            "reason_code": "empty_ticket",
            "risk_level": "medium",
            "reason_text": "Empty or unreadable ticket should be routed to a human.",
        }

    for reason_code, pattern, risk_level, reason_text in _ESCALATION_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return {
                "should_escalate": True,
                "reason_code": reason_code,
                "risk_level": risk_level,
                "reason_text": reason_text,
            }

    # Non-domain tickets are often out-of-scope; reply only if clear harmless chatter.
    if company_n in {"", "none"}:
        if re.search(r"\b(thank you|thanks|great|awesome)\b", text, flags=re.IGNORECASE):
            return {
                "should_escalate": False,
                "reason_code": "safe_smalltalk",
                "risk_level": "low",
                "reason_text": "Harmless acknowledgement can be safely replied to.",
            }
        if re.search(r"\b(actor in iron man|weather|stock price|movie)\b", text, flags=re.IGNORECASE):
            return {
                "should_escalate": False,
                "reason_code": "out_of_scope_safe_reply",
                "risk_level": "low",
                "reason_text": "Out-of-scope informational requests can receive a safe refusal.",
            }
        return {
            "should_escalate": True,
            "reason_code": "unknown_domain",
            "risk_level": "medium",
            "reason_text": "Unknown domain without clear safe response path.",
        }

    return {
        "should_escalate": False,
        "reason_code": "safe_to_answer",
        "risk_level": "low",
        "reason_text": "No high-risk trigger detected.",
    }
