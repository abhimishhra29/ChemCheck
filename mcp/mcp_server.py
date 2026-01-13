from __future__ import annotations

import os
import re
from typing import Iterable, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_S = 20

BAD_DOMAINS = {
    "wikipedia.org",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "bloomberg.com",
    "yahoo.com",
    "crunchbase.com",
    "zoominfo.com",
}

CORP_STOPWORDS = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "co",
    "company",
    "corp",
    "corporation",
    "gmbh",
    "ag",
    "plc",
    "srl",
    "sa",
    "bv",
    "nv",
    "kg",
    "oy",
    "ab",
    "sas",
    "pty",
    "pte",
    "sro",
    "kk",
}

# AU-only domain suffixes
AU_TLDS = (
    ".com.au",
    ".net.au",
    ".org.au",
    ".edu.au",
    ".gov.au",
    ".asn.au",
    ".id.au",
    ".au",
)

INTENT_QUERIES = [
    '"{name}" "contact us" Australia',
    '"{name}" contact Australia',
    '"{name}" "about us" Australia',
    '"{name}" website Australia',
    '"{name}" ".com.au"',
]

SDS_KEYWORDS = ("sds", "safety data sheet", "safety datasheet")

mcp = FastMCP("tavily-manufacturer")


@mcp.tool()
def get_manufacturer_url(
    manufacturer_name: str, api_key: Optional[str] = None
) -> dict:
    """
    Resolve a manufacturer's official website URL using Tavily.
    AU-only: only returns domains ending in .au / .com.au / etc.

    Returns:
      {
        "manufacturer_url": <best_url_or_None>,
        "best_domain": <domain_or_None>,
        "queries": [ ... ],
        "candidates": [ {url,title,score,domain}, ... ]
      }
    """
    name = (manufacturer_name or "").strip()
    if not name:
        raise ValueError("manufacturer_name is required")

    api_key = api_key or os.getenv(TAVILY_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            "Missing Tavily API key. Set it via:\n"
            f"  import os; os.environ['{TAVILY_API_KEY_ENV}'] = 'YOUR_KEY'\n"
            "or pass api_key=... to get_manufacturer_url()."
        )

    tokens = normalized_tokens(name)

    queries = [q.format(name=name) for q in INTENT_QUERIES]
    merged: list[dict] = []
    seen_urls: set[str] = set()

    for q in queries:
        for result in tavily_search(q, api_key, max_results=6):
            url = result.get("url")
            if isinstance(url, str) and url and url not in seen_urls:
                merged.append(result)
                seen_urls.add(url)

    scored = []
    for result in merged:
        url = result.get("url")
        if not isinstance(url, str) or not url:
            continue

        domain = extract_domain(url)
        if not is_au_domain(domain):
            continue

        score = score_result(result, tokens)
        scored.append(
            {
                "url": url,
                "title": result.get("title"),
                "score": round(score, 3),
                "domain": domain,
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    best_url = scored[0]["url"] if scored else None
    best_domain = scored[0]["domain"] if scored else None

    return {
        "manufacturer_url": best_url,
        "best_domain": best_domain,
        "queries": queries,
        "candidates": scored[:10],
    }


@mcp.tool()
def get_sds_url(
    manufacturer_url: str,
    cas_number: Optional[str] = None,
    product_name: Optional[str] = None,
    product_code: Optional[str] = None,
) -> dict:
    """
    Find an SDS URL for a product, restricted to the manufacturer's domain.

    Args:
        manufacturer_url: Manufacturer homepage or domain.
        cas_number: Optional CAS number for search (preferred if provided).
        product_name: Product name to use when CAS is not available.
        product_code: Optional product code (SKU/catalog/ID).

    Returns:
        A dict with sds_url, query, and manufacturer_domain.
    """
    if not manufacturer_url or not manufacturer_url.strip():
        raise ValueError("manufacturer_url is required")

    cas_value = (cas_number or "").strip()
    product_value = (product_name or "").strip()
    product_code_value = (product_code or "").strip()

    if not cas_value and not product_value and not product_code_value:
        raise ValueError("CAS, product_name, or product_code required")

    terms = [t for t in (cas_value, product_value, product_code_value) if t]
    manufacturer_domain = extract_domain(manufacturer_url)

    terms_query = " OR ".join(f'"{t}"' for t in terms)
    query = (
        f'site:{manufacturer_domain} ({terms_query}) '
        '("SDS" OR "Safety Data Sheet") filetype:pdf'
    )
    results = tavily_search(query)

    tokens = re.findall(r"[a-z0-9]+", " ".join(terms).lower())
    best, best_score = None, -1
    for result in results:
        url = result.get("url", "")
        if not is_same_domain(url, manufacturer_domain):
            continue
        score = score_sds_result(result, tokens)
        if score > best_score:
            best, best_score = url, score

    return {
        "sds_url": best,
        "query": query,
        "manufacturer_domain": manufacturer_domain,
    }


def tavily_search(
    query: str, api_key: Optional[str] = None, max_results: int = 6
) -> list[dict]:
    api_key = api_key or os.environ[TAVILY_API_KEY_ENV]
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    resp = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=DEFAULT_TIMEOUT_S)
    if not resp.ok:
        raise RuntimeError(f"Tavily search failed: {resp.status_code} {resp.text}")

    data = resp.json()
    results = data.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Unexpected Tavily response shape: results is not a list")
    return results


def normalized_tokens(name: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", name.lower())
    return [t for t in tokens if t not in CORP_STOPWORDS]


def extract_domain(url: str) -> str:
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def is_au_domain(domain: str) -> bool:
    value = (domain or "").lower()
    return any(value.endswith(suffix) for suffix in AU_TLDS)


def score_result(result: dict, tokens: Iterable[str]) -> float:
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


def is_same_domain(url: str, domain: str) -> bool:
    netloc = extract_domain(url)
    return netloc == domain or netloc.endswith("." + domain)


def score_sds_result(result: dict, tokens: Iterable[str]) -> float:
    url = (result.get("url", "") or "").lower()
    title = (result.get("title", "") or "").lower()
    content = (result.get("content", "") or "").lower()
    score = 0
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


if __name__ == "__main__":
    mcp.run()
