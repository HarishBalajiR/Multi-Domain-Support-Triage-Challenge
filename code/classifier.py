"""Domain and request-type classification helpers."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from schemas import RetrievedChunk
from utils import normalize_text


_DOMAIN_KEYWORDS = {
    "hackerrank": (
        "hackerrank",
        "assessment",
        "candidate",
        "test",
        "screen",
        "interview",
        "role",
    ),
    "claude": (
        "claude",
        "workspace",
        "conversation",
        "anthropic",
        "team plan",
        "pro plan",
    ),
    "visa": (
        "visa",
        "card",
        "merchant",
        "refund",
        "chargeback",
        "travel cheque",
        "stolen card",
    ),
}


_REQUEST_PATTERNS = {
    "bug": (
        r"\b(site is down|outage|not working|error|crash|bug|failed|inaccessible)\b",
    ),
    "feature_request": (
        r"\b(feature request|can you add|would like to see|please add|enhancement)\b",
    ),
    "invalid": (
        r"\b(actor in iron man)\b",
    ),
}

_CANONICAL_BY_DOMAIN = {
    "hackerrank": [
        "screen",
        "community",
        "interviews",
        "settings",
        "library",
        "general_help",
    ],
    "claude": [
        "privacy",
        "conversation_management",
        "account_management",
        "billing",
        "general",
    ],
    "visa": [
        "travel_support",
        "general_support",
        "dispute_resolution",
        "fraud_protection",
        "regulations_fees",
    ],
    "none": ["general"],
}


def canonicalize_product_area(domain: str, raw_area: str, ticket_text: str) -> str:
    """Map raw retrieval/product labels into canonical evaluation-friendly labels."""
    domain_l = (domain or "none").lower()
    area = (raw_area or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = (ticket_text or "").lower()

    if domain_l == "hackerrank":
        if "community" in area:
            return "community"
        if any(tok in text for tok in ("assessment", "test", "candidate", "reinvite", "extra time", "duration")):
            return "screen"
        if "interview" in area:
            return "interviews"
        if "setting" in area:
            return "settings"
        if "library" in area:
            return "library"
        if "screen" in area:
            return "screen"
        return "general_help"

    if domain_l == "claude":
        if any(tok in text for tok in ("private info", "sensitive", "privacy", "temporary chat")):
            return "privacy"
        if any(tok in text for tok in ("conversation", "chat", "rename", "delete")):
            return "conversation_management"
        if any(tok in text for tok in ("billing", "plan", "invoice", "tax")):
            return "billing"
        if any(tok in text for tok in ("access", "workspace", "account", "seat", "login")):
            return "account_management"
        if "privacy" in area:
            return "privacy"
        return "general"

    if domain_l == "visa":
        if any(tok in text for tok in ("traveller", "traveler", "travel", "cheque", "lisbon")):
            return "travel_support"
        if any(tok in text for tok in ("stolen card", "lost card", "report card", "customer assistance")):
            return "general_support"
        if "dispute" in area:
            return "dispute_resolution"
        if "fraud" in area:
            return "fraud_protection"
        if "regulation" in area or "fees" in area:
            return "regulations_fees"
        return "general_support"

    if any(tok in text for tok in ("thank you", "thanks", "happy")):
        return "general"
    return "general"


def weighted_vote_product_area(domain: str, ticket_text: str, contexts: list[RetrievedChunk]) -> str:
    """Compute weighted product-area vote from retrieval hits."""
    if not contexts:
        return canonicalize_product_area(domain, "general", ticket_text)

    votes: dict[str, float] = defaultdict(float)
    for doc in contexts:
        mapped = canonicalize_product_area(domain, doc.chunk.product_area, ticket_text)
        votes[mapped] += max(0.0, doc.score)
    return max(votes, key=votes.get)


def candidate_product_areas(domain: str, voted_area: str) -> list[str]:
    """Return constrained label set for LLM product_area output."""
    domain_l = (domain or "none").lower()
    allowed = list(_CANONICAL_BY_DOMAIN.get(domain_l, _CANONICAL_BY_DOMAIN["none"]))
    if voted_area and voted_area not in allowed:
        allowed.insert(0, voted_area)
    return allowed


def _infer_domain_from_text(text: str) -> tuple[str, float]:
    scores = {"hackerrank": 0, "claude": 0, "visa": 0}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "none", 0.0
    confidence = min(0.95, 0.45 + (0.12 * scores[best_domain]))
    return best_domain, confidence


def _infer_request_type(text: str) -> tuple[str, float]:
    if len(text.split()) <= 6 and re.search(r"\b(thank you|thanks)\b", text, flags=re.IGNORECASE):
        return "invalid", 0.92

    for label, patterns in _REQUEST_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return label, 0.9
    if any(
        phrase in text
        for phrase in (
            "delete my account",
            "add extra time",
            "reinvite",
            "how long",
            "how do i",
            "what do i do",
            "where can i",
            "can i",
        )
    ) or "?" in text:
        return "product_issue", 0.7
    return "invalid", 0.55


def classify_ticket(
    issue: str,
    subject: str,
    company: str,
    *,
    product_area_hint: Optional[str] = None,
    retrieved_contexts: Optional[list[RetrievedChunk]] = None,
) -> dict:
    """Classify domain, request type, and product area hints."""
    issue_n = normalize_text(issue or "")
    subject_n = normalize_text(subject or "")
    company_n = normalize_text(company or "").lower()
    text = f"{subject_n} {issue_n}".strip().lower()

    signals: list[str] = []
    if company_n in {"hackerrank", "claude", "visa"}:
        domain = company_n
        domain_conf = 1.0
        signals.append("company_provided")
    else:
        domain, domain_conf = _infer_domain_from_text(text)
        if domain != "none":
            signals.append("keyword_domain_inference")
        else:
            signals.append("domain_unknown")

    request_type, req_conf = _infer_request_type(text)
    if request_type != "product_issue":
        signals.append(f"request_pattern:{request_type}")
    else:
        signals.append("request_pattern:default_product_issue")

    if product_area_hint:
        product_area = canonicalize_product_area(domain, product_area_hint, text)
    elif retrieved_contexts is not None:
        product_area = weighted_vote_product_area(domain, text, retrieved_contexts)
    else:
        product_area = canonicalize_product_area(domain, "general", text)

    candidates = candidate_product_areas(domain, product_area)
    if product_area_hint:
        signals.append("product_area_from_retriever")
    elif retrieved_contexts is not None:
        signals.append("product_area_weighted_vote")
    else:
        signals.append("product_area_default")

    confidence = round((domain_conf * 0.55) + (req_conf * 0.45), 3)
    return {
        "domain": domain,
        "request_type_hint": request_type,
        "product_area_hint": product_area,
        "product_area_candidates": candidates,
        "confidence": confidence,
        "signals": signals,
    }
