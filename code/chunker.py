"""Markdown chunking utilities for corpus ingestion."""

import re
from typing import Optional


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    """Return parsed frontmatter key-values and body markdown."""
    match = _FRONTMATTER_RE.match(markdown)
    if not match:
        return {}, markdown

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    body = markdown[match.end() :]
    return metadata, body


def clean_markdown(text: str) -> str:
    """Strip noisy patterns that hurt retrieval quality."""
    cleaned = _HTML_COMMENT_RE.sub(" ", text)
    cleaned = _IMAGE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]\([^)]+\)", lambda m: m.group(0).split("](")[0].lstrip("["), cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _split_sections(text: str) -> list[tuple[Optional[str], str]]:
    """Split markdown into header sections while preserving content."""
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        return [(None, text.strip())] if text.strip() else []

    sections: list[tuple[Optional[str], str]] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _window_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]

    step = max(1, size - overlap)
    windows: list[str] = []
    for i in range(0, len(words), step):
        window = words[i : i + size]
        if not window:
            break
        windows.append(" ".join(window))
        if i + size >= len(words):
            break
    return windows


def chunk_markdown(
    markdown: str,
    *,
    title: str,
    breadcrumbs: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[str]:
    """Split markdown into retrieval chunks with section-aware boundaries."""
    _, body = parse_frontmatter(markdown)
    cleaned = clean_markdown(body)
    sections = _split_sections(cleaned)
    chunks: list[str] = []

    for section_title, section_body in sections:
        pref = f"{title}\n{breadcrumbs}\n"
        if section_title:
            pref += f"Section: {section_title}\n"
        for window in _window_words(section_body, size=chunk_size_words, overlap=chunk_overlap_words):
            chunks.append(f"{pref}\n{window}".strip())

    return chunks
