"""Grounded response generation using retrieved support snippets."""

from __future__ import annotations

from pydantic import ValidationError

from schemas import AgentLLMOutput, RetrievedChunk
from llm import run_llm_json
from classifier import canonicalize_product_area

MAX_SNIPPET_CHARS = 700


_SYSTEM_PROMPT = """You are a support triage assistant.
Rules:
1) Use only the provided support snippets. Do not use outside knowledge.
2) If snippets are insufficient, produce status=escalated.
3) Keep response concise and user-facing.
4) request_type must be one of: product_issue, feature_request, bug, invalid.
5) status must be one of: replied, escalated.
Return JSON with keys: status, product_area, response, justification, request_type.
"""


def _truncate_at_sentence(content: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    text = (content or "").strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = max(window.rfind("."), window.rfind("?"), window.rfind("!"))
    if cut > 120:
        return window[: cut + 1].strip()
    return window.strip()


def _build_context_block(context_docs: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    for i, doc in enumerate(context_docs, start=1):
        lines.append(
            "\n".join(
                [
                    f"[Snippet {i}] domain={doc.chunk.domain} area={doc.chunk.product_area} score={doc.score:.4f}",
                    f"title: {doc.chunk.title}",
                    f"source: {doc.chunk.source_path}",
                    f"content: {_truncate_at_sentence(doc.chunk.content)}",
                ]
            )
        )
    return "\n\n".join(lines)


def generate_response(
    ticket: dict,
    context_docs: list[RetrievedChunk],
    classification: dict,
    *,
    allowed_product_areas: list[str],
) -> AgentLLMOutput:
    """Generate a grounded structured response from retrieval context."""
    def _grounded_fallback(justification: str) -> AgentLLMOutput:
        if context_docs:
            source = context_docs[0].chunk.source_path
            title = context_docs[0].chunk.title
            snippet = _truncate_at_sentence(context_docs[0].chunk.content, max_chars=260)
            return AgentLLMOutput(
                status="replied",
                product_area=classification.get("product_area_hint", "general"),
                response=f"Based on our support guidance, {snippet}",
                justification=f"{justification} Returned grounded retrieval-only response from {title} ({source}).",
                request_type=classification.get("request_type_hint", "product_issue"),
            )
        return AgentLLMOutput(
            status="escalated",
            product_area=classification.get("product_area_hint", "general"),
            response="Escalate to a human",
            justification=f"{justification} No support snippets available.",
            request_type=classification.get("request_type_hint", "product_issue"),
        )

    prompt = f"""
Ticket:
- company: {ticket.get('company', '')}
- subject: {ticket.get('subject', '')}
- issue: {ticket.get('issue', '')}

Classifier hints:
- domain: {classification.get('domain', 'none')}
- product_area_hint: {classification.get('product_area_hint', 'general')}
- request_type_hint: {classification.get('request_type_hint', 'product_issue')}
- allowed_product_areas: {", ".join(allowed_product_areas)}

Support snippets:
{_build_context_block(context_docs)}
""".strip()

    payload = run_llm_json(prompt, system_prompt=_SYSTEM_PROMPT)
    if not payload:
        return _grounded_fallback("Model output was invalid or empty.")

    try:
        result = AgentLLMOutput.model_validate(payload)
    except ValidationError:
        return _grounded_fallback("Model output failed schema validation.")
    if not result.response.strip():
        return _grounded_fallback("Model produced an empty response.")
    if result.product_area not in allowed_product_areas:
        raw_ticket = f"{ticket.get('subject', '')} {ticket.get('issue', '')}".strip()
        result.product_area = canonicalize_product_area(
            classification.get("domain", "none"),
            result.product_area,
            raw_ticket,
        )
    if result.product_area not in allowed_product_areas:
        result.product_area = classification.get("product_area_hint", allowed_product_areas[0])
    return result
