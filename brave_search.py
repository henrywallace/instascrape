#!/usr/bin/env python3
"""
Simple Brave Search CLI

Requirements:
  - requests
  - python-dotenv

Environment:
  - .env with BRAVE_API_KEY=... in the project root

Usage examples:
  python brave_search.py
  python brave_search.py --query "patient clinical trial experience reddit" --count 15
  python brave_search.py --links
  python brave_search.py --snippets
  python brave_search.py --raw --count 5
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv


DEFAULT_QUERY = (
    "patients' opinions on clinical trials experiences posts forum OR reddit OR blog"
)


def load_api_key() -> str:
    """Load the Brave API key from the environment, supporting .env files."""
    # Load .env if present; no-op if missing.
    load_dotenv()
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        print(
            "Error: BRAVE_API_KEY not found in environment. Set it in your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)
    return api_key


def brave_web_search(
    api_key: str,
    query: str,
    count: int = 10,
    search_lang: str = "en",
    safesearch: str = "moderate",
    timeout_seconds: int = 20,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Call Brave Web Search API and return (json, headers)."""
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
        # Optional but harmless; explicit user agent can help debugging.
        "User-Agent": "instascrape-brave-search/1.0 (+https://example.local)",
    }
    params = {
        "q": query,
        "count": max(1, min(count, 50)),  # Brave caps; stay reasonable
        "search_lang": search_lang,
        "safesearch": safesearch,
    }
    resp = requests.get(url, headers=headers, params=params, timeout=timeout_seconds)
    try:
        data = resp.json()
    except ValueError:
        data = {"error": "non-json-response", "text": resp.text}
    # Raise on 4xx/5xx after we captured the body for diagnostics
    if not resp.ok:
        # Include a compact diagnostic; caller may print raw data if desired
        diagnostic = {
            "status_code": resp.status_code,
            "headers": dict(resp.headers),
            "body": data,
        }
        raise RuntimeError(json.dumps(diagnostic, indent=2))
    return data, dict(resp.headers)


def extract_web_results(api_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract a normalized list of results from Brave web search JSON."""
    web = api_json.get("web", {})
    results = web.get("results", [])
    normalized = []
    for item in results:
        normalized.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("description"),
                "language": item.get("language"),
                "is_source_local": item.get("is_source_local"),
                "domain": item.get("meta_url", {}).get("hostname"),
            }
        )
    return normalized


def print_results(
    results: List[Dict[str, Any]],
    mode: str,
) -> None:
    """Print results in the requested mode: both, links, or snippets."""
    if not results:
        print("No results returned.")
        return

    for idx, r in enumerate(results, start=1):
        title = r.get("title") or "(no title)"
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        domain = r.get("domain") or ""

        if mode == "links":
            print(url)
            continue
        if mode == "snippets":
            if snippet:
                print(f"{idx}. {snippet}")
            else:
                print(f"{idx}. [no snippet] {url}")
            continue

        # default: both concise
        print(f"{idx}. {title}")
        if snippet:
            print(f"   Snippet: {snippet}")
        print(f"   Link: {url}")
        if domain:
            print(f"   Domain: {domain}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Query Brave Search for patient opinions on clinical trials and print snippets/links."
        )
    )
    parser.add_argument(
        "--query",
        "-q",
        default=DEFAULT_QUERY,
        help="Search query (default targets patient opinions/experiences on clinical trials)",
    )
    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=10,
        help="Number of results to fetch (1-50)",
    )
    parser.add_argument(
        "--search-lang",
        default="en",
        help="Search language (e.g., en, es)",
    )
    parser.add_argument(
        "--safesearch",
        choices=["off", "moderate", "strict"],
        default="moderate",
        help="Safe search setting",
    )
    parser.add_argument(
        "--links",
        action="store_true",
        help="Print only links (URLs)",
    )
    parser.add_argument(
        "--snippets",
        action="store_true",
        help="Print only raw snippets of opinions",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON from Brave API (first, then formatted results)",
    )
    parser.add_argument(
        "--show-headers",
        action="store_true",
        help="Print response headers (rate limits, etc.)",
    )

    args = parser.parse_args()

    api_key = load_api_key()

    try:
        data, headers = brave_web_search(
            api_key=api_key,
            query=args.query,
            count=args.count,
            search_lang=args.search_lang,
            safesearch=args.safesearch,
        )
    except Exception as e:
        # If API returns error, show the parsed diagnostic
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(2)

    if args.show_headers:
        # Minimize noise: show key rate-limit headers if present
        interesting = {
            k: v
            for k, v in headers.items()
            if k.lower()
            in {
                "x-ratelimit-limit",
                "x-ratelimit-remaining",
                "x-ratelimit-reset",
                "x-request-id",
            }
        }
        print("Response headers (subset):")
        print(json.dumps(interesting or headers, indent=2))

    if args.raw:
        # Show real returned data for inspection
        print("Raw JSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    results = extract_web_results(data)
    mode = "both"
    if args.links:
        mode = "links"
    elif args.snippets:
        mode = "snippets"

    if not args.raw:
        # If not showing raw JSON first, clarify what we print
        print("Formatted results:")
    print_results(results, mode)


if __name__ == "__main__":
    main()


