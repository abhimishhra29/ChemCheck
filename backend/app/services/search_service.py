from __future__ import annotations

from typing import Iterable, Optional
from urllib.parse import urlparse

from app.clients.tavily_client import TavilyClient
from app.core.constants import (
    AU_TLDS,
    BAD_DOMAINS,
    CORP_STOPWORDS,
    INTENT_QUERIES,
    SDS_KEYWORDS,
)
from app.utils.file_utils import extract_domain, is_au_domain, is_same_domain, normalized_tokens


class SearchService:
    """A service for finding manufacturer and SDS URLs."""

    def __init__(self, tavily_client: TavilyClient):
        self.tavily_client = tavily_client

    async def get_manufacturer_url(self, manufacturer_name: str) -> dict:
        """
        Resolves a manufacturer's official website URL using Tavily Search.
        This tool is optimized to find Australian websites.

        Returns:
            A dictionary containing the best URL found, the domain, the queries used,
            and a list of scored candidates for debugging.
        """
        name = (manufacturer_name or "").strip()
        if not name:
            raise ValueError("manufacturer_name is required")

        search_tokens = normalized_tokens(name, CORP_STOPWORDS)
        queries = [q.format(name=name) for q in INTENT_QUERIES]

        merged_results: list[dict] = []
        seen_urls: set[str] = set()

        for q in queries:
            for result in await self.tavily_client.search(q, max_results=6):
                url = result.get("url")
                if isinstance(url, str) and url and url not in seen_urls:
                    merged_results.append(result)
                    seen_urls.add(url)

        scored_candidates = []
        for result in merged_results:
            url = result.get("url")
            if not isinstance(url, str) or not url:
                continue

            domain = extract_domain(url)
            if not is_au_domain(domain, AU_TLDS):
                continue

            score = self._score_manufacturer_result(result, search_tokens)
            scored_candidates.append(
                {
                    "url": url,
                    "title": result.get("title"),
                    "score": round(score, 3),
                    "domain": domain,
                }
            )

        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        best_url = scored_candidates[0]["url"] if scored_candidates else None
        best_domain = scored_candidates[0]["domain"] if scored_candidates else None

        return {
            "manufacturer_url": best_url,
            "best_domain": best_domain,
            "queries": queries,
            "candidates": scored_candidates[:10],
        }

    async def get_sds_url(
        self,
        manufacturer_url: str,
        cas_number: Optional[str] = None,
        product_name: Optional[str] = None,
        product_code: Optional[str] = None,
        exclude_urls: Optional[list[str]] = None,
    ) -> dict:
        """
        Finds a Safety Data Sheet (SDS) URL for a product, restricted to the
        manufacturer's domain.
        """
        if not manufacturer_url or not manufacturer_url.strip():
            raise ValueError("manufacturer_url is required")

        cas_value = (cas_number or "").strip()
        product_value = (product_name or "").strip()
        product_code_value = (product_code or "").strip()

        if not cas_value and not product_value and not product_code_value:
            raise ValueError("CAS, product_name, or product_code is required")

        search_terms = [t for t in (cas_value, product_value, product_code_value) if t]
        manufacturer_domain = extract_domain(manufacturer_url)

        terms_query = " OR ".join(f'"{t}"' for t in search_terms)
        query = (
            f'site:{manufacturer_domain} ({terms_query}) '
            '("SDS" OR "Safety Data Sheet") filetype:pdf'
        )

        results = await self.tavily_client.search(query, max_results=10)

        search_tokens = normalized_tokens(" ".join(search_terms), stopwords=set())

        candidates = []
        for result in results:
            url = result.get("url", "")
            if not is_same_domain(url, manufacturer_domain):
                continue
            score = self._score_sds_result(result, search_tokens)
            candidates.append(
                {
                    "url": url,
                    "title": result.get("title"),
                    "score": score,
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)

        excluded = {((u or "").rstrip("/")) for u in (exclude_urls or []) if u}
        best_url = None
        for candidate in candidates:
            candidate_url = (candidate.get("url") or "").rstrip("/")
            if candidate_url and candidate_url not in excluded:
                best_url = candidate.get("url")
                break

        return {
            "sds_url": best_url,
            "query": query,
            "manufacturer_domain": manufacturer_domain,
            "candidates": candidates[:10],
        }

    def _score_manufacturer_result(self, result: dict, tokens: Iterable[str]) -> float:
        """Calculates a relevance score for a manufacturer search result."""
        url = (result.get("url") or "").lower()
        title = (result.get("title") or "").lower()
        content = (result.get("content") or "").lower()
        domain = extract_domain(url)
        score = 0.0

        if any(bad == domain or domain.endswith("." + bad) for bad in BAD_DOMAINS):
            score -= 10.0

        if url.endswith((".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx")):
            score -= 2.0

        for token in tokens:
            if token and token in domain:
                score += 4.0
            elif token and token in title:
                score += 2.0
            elif token and token in content:
                score += 1.0

        if any(keyword in title for keyword in ["contact", "about", "home"]):
            score += 1.0
        if any(keyword in content for keyword in ["contact", "email", "phone", "address", "sales@"]):
            score += 2.0

        path = urlparse(url).path or ""
        if path in ("", "/"):
            score += 1.5

        return score

    def _score_sds_result(self, result: dict, tokens: Iterable[str]) -> float:
        """Calculates a relevance score for an SDS search result."""
        url = (result.get("url", "") or "").lower()
        title = (result.get("title", "") or "").lower()
        score = 0.0

        if url.endswith(".pdf"):
            score += 3
        if any(keyword in title for keyword in SDS_KEYWORDS):
            score += 3

        for token in tokens:
            if token in title:
                score += 1
            elif token in url:
                score += 0.5

        return score
