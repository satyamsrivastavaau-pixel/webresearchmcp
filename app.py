import os
import re
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
import trafilatura
from ddgs import DDGS
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse


APP_NAME = os.getenv("APP_NAME", "hf-research-mcp")
MCP_SHARED_SECRET = os.getenv("MCP_SHARED_SECRET", "")
MAX_PAGE_CHARS = int(os.getenv("MAX_PAGE_CHARS", "12000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

mcp = FastMCP(APP_NAME)


class RequireSharedSecretMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if MCP_SHARED_SECRET and request.headers.get("x-mcp-key") != MCP_SHARED_SECRET:
            return PlainTextResponse("Unauthorized", status_code=401)
        return await call_next(request)


app.add_middleware(RequireSharedSecretMiddleware)


def _truncate(text: str, limit: int = MAX_PAGE_CHARS) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        rows = DDGS(timeout=15).text(
            query,
            max_results=max_results,
            safesearch="moderate",
            backend="auto",
        )
    except Exception as exc:
        raise RuntimeError(f"Web search failed: {exc}") from exc

    results: List[Dict[str, Any]] = []
    for item in rows or []:
        url = item.get("href") or item.get("url") or ""
        title = item.get("title", "")
        snippet = item.get("body") or item.get("snippet") or ""

        if url and (title or snippet):
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "ddgs",
                }
            )

    return results


def _fetch_url_text(url: str) -> str:
    if not _is_valid_url(url):
        raise ValueError("url must be a valid http or https URL")

    response = requests.get(
        url,
        timeout=REQUEST_TIMEOUT,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; hf-research-mcp/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    response.raise_for_status()

    extracted = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=False,
    )
    return _truncate(extracted or response.text)


def _build_prompt(
    query: str,
    search_results: List[Dict[str, Any]],
    page_docs: List[Dict[str, Any]],
) -> str:
    parts = [
        "You are a research assistant. Answer only from the supplied web evidence.",
        "If evidence is weak or conflicting, state that clearly.",
        "Include source URLs for important claims.",
        "",
        f"QUERY: {query}",
        "",
        "SEARCH RESULTS:",
    ]

    for index, result in enumerate(search_results, start=1):
        parts.append(
            f"{index}. {result.get('title', '')}\n"
            f"URL: {result.get('url', '')}\n"
            f"SNIPPET: {result.get('snippet', '')}"
        )

    if page_docs:
        parts.extend(["", "PAGE CONTENTS:"])
        for index, document in enumerate(page_docs, start=1):
            parts.append(
                f"{index}. {document.get('title', '')}\n"
                f"URL: {document.get('url', '')}\n"
                f"TEXT: {document.get('text', '')}"
            )

    parts.extend(["", "Write the answer now."])
    return "\n".join(parts)


def _call_hf(prompt: str) -> str:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    response = requests.post(
        "https://router.huggingface.co/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": os.getenv("HF_MODEL", "Qwen/Qwen3.5-9B:cheapest"),
            "messages": [
                {
                    "role": "system",
                    "content": "Answer only from the supplied web sources and cite their URLs.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@mcp.tool()
def search_web(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search the web and return titles, URLs, and snippets."""
    return _search_duckduckgo(query, max_results)


@mcp.tool()
def fetch_url(url: str) -> Dict[str, Any]:
    """Fetch a web page and extract readable text."""
    return {"url": url, "text": _fetch_url_text(url)}


@mcp.tool()
def research(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web, fetch the best pages, and synthesize an answer."""
    search_results = _search_duckduckgo(query, max_results)
    if not search_results:
        return {
            "query": query,
            "answer": "No web search results were returned for this query.",
            "search_results": [],
            "sources": [],
        }

    page_docs: List[Dict[str, Any]] = []
    for result in search_results[:3]:
        url = result.get("url", "")
        if not url:
            continue

        try:
            text = _fetch_url_text(url)
            if text:
                page_docs.append(
                    {
                        "title": result.get("title", ""),
                        "url": url,
                        "text": text,
                    }
                )
        except Exception as exc:
            print(f"Skipping {url}: {exc}")

    prompt = _build_prompt(query, search_results, page_docs)
    answer = _call_hf(prompt)

    return {
        "query": query,
        "answer": answer,
        "search_results": search_results,
        "sources": [
            {"title": item.get("title", ""), "url": item.get("url", "")}
            for item in search_results
            if item.get("url")
        ],
    }

app = mcp.streamable_http_app()
app.add_middleware(RequireSharedSecretMiddleware)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
    