from __future__ import annotations

import re
from typing import Optional

from app.agents.ocr_agent import OCRResult

# Regular expression for matching Chemical Abstracts Service (CAS) numbers.
CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


def merge_ocr_results(primary: OCRResult, secondary: OCRResult) -> OCRResult:
    """
    Merges two OCRResult objects into a single, consolidated result.

    This function intelligently combines fields from a primary and a secondary scan,
    choosing the most likely correct data for each field based on heuristics.

    Args:
        primary: The primary OCRResult, often from the front of a product.
        secondary: The secondary OCRResult, often from the back.

    Returns:
        A new, merged OCRResult.
    """
    return OCRResult(
        full_text=_merge_full_text(primary.full_text, secondary.full_text),
        product_name=_choose_text(primary.product_name, secondary.product_name),
        product_code=_choose_product_code(primary.product_code, secondary.product_code),
        cas_number=_choose_cas_number(primary.cas_number, secondary.cas_number),
        manufacturer_name=_choose_text(
            primary.manufacturer_name, secondary.manufacturer_name
        ),
        description=_choose_text(primary.description, secondary.description),
    )


def _merge_full_text(text_a: str, text_b: str) -> str:
    """Merges two blocks of text, removing duplicate lines."""
    lines = []
    seen = set()
    for line in (text_a or "").splitlines() + (text_b or "").splitlines():
        normalized = re.sub(r"\s+", " ", line.strip()).lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line.strip())
    return "\n".join(lines).strip()


def _choose_text(primary: Optional[str], secondary: Optional[str]) -> Optional[str]:
    """Selects the better of two text strings based on length and containment."""
    p = (primary or "").strip()
    s = (secondary or "").strip()
    if not p:
        return s or None
    if not s:
        return p or None
    if p.lower() == s.lower():
        return p
    # Prefer the more specific/longer string if one contains the other.
    if p.lower() in s.lower():
        return s
    if s.lower() in p.lower():
        return p
    # Default to the longer string.
    return p if len(p) >= len(s) else s


def _choose_cas_number(primary: Optional[str], secondary: Optional[str]) -> Optional[str]:
    """
    Selects the most likely valid CAS number from two options.

    It prioritizes a value that strictly matches the CAS format.
    """
    p = _extract_cas(primary)
    s = _extract_cas(secondary)
    if p and s:
        if p == s:
            return p
        # Prefer the one that matches the official format if the other doesn't.
        if CAS_PATTERN.fullmatch(p) and not CAS_PATTERN.fullmatch(s):
            return p
        if CAS_PATTERN.fullmatch(s) and not CAS_PATTERN.fullmatch(p):
            return s
        return p
    return p or s


def _extract_cas(value: Optional[str]) -> Optional[str]:
    """Extracts a CAS number from a string if present."""
    if not value:
        return None
    match = CAS_PATTERN.search(value)
    return match.group(0) if match else value.strip() or None


def _choose_product_code(primary: Optional[str], secondary: Optional[str]) -> Optional[str]:
    """Selects the better of two product codes based on a scoring heuristic."""
    p = (primary or "").strip()
    s = (secondary or "").strip()
    if not p:
        return s or None
    if not s:
        return p or None
    if p.lower() == s.lower():
        return p
    if p.lower() in s.lower():
        return s
    if s.lower() in p.lower():
        return p

    # Fallback to a scoring mechanism.
    p_score = _product_code_score(p)
    s_score = _product_code_score(s)

    return s if s_score > p_score else p


def _product_code_score(value: str) -> int:
    """
    Calculates a simple score for a product code.

    The score is based on length and the presence of digits, which are common
    in product codes.
    """
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    score = len(normalized)
    if re.search(r"\d", value):
        score += 2  # Boost score for codes containing numbers.
    return score
