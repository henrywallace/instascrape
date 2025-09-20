#!/usr/bin/env python3
"""
LLM ETL over Brave Search -> post/comment (platform-agnostic)

Modes:
- Single: Search once (default) and extract from a Reddit post or comment
- Iterative: Define domain D and sample queries over N iterations; fetch results from any site and extract via LLM

Env vars (loaded via .env):
- BRAVE_API_KEY
- ANTHROPIC_API_KEY (preferred) or OPENAI_API_KEY (fallback)
- ANTHROPIC_MODEL / OPENAI_MODEL (optional)

Usage examples:
  python analyze_posts.py --provider anthropic --model claude-3-5-sonnet-latest --raw-output
  python analyze_posts.py --use-comment --provider anthropic --raw-output
  python analyze_posts.py --domain "patients' opinions on clinical trials" --niters 1 --results-per-iter 3 --provider anthropic --raw-output
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

try:
    # OpenAI Python SDK >=1.0
    from openai import OpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    # Anthropic (Claude) SDK
    from anthropic import Anthropic
except Exception as e:  # pragma: no cover
    Anthropic = None  # type: ignore

# Local import from the Brave search helper
from brave_search import brave_web_search, extract_web_results


class ParticipantProfile(BaseModel):
    role: Optional[str] = Field(
        None, description="patient | healthy-volunteer | caregiver | clinician | other"
    )
    age: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    disease: Optional[str] = None
    prior_trials_count: Optional[int] = None


class StudyInfo(BaseModel):
    phase: Optional[str] = None
    condition: Optional[str] = None
    intervention: Optional[str] = None
    sponsor: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    compensation: Optional[str] = None
    enrollment_status: Optional[str] = None
    trial_id_or_link: Optional[str] = None
    site_name: Optional[str] = None


class ThreadMetadata(BaseModel):
    thread_url: Optional[str] = None
    subreddit: Optional[str] = None
    thread_title: Optional[str] = None
    score: Optional[int] = None
    num_comments: Optional[int] = None
    created_iso: Optional[str] = None
    over_18: Optional[bool] = None
    link_flair_text: Optional[str] = None


class Engagement(BaseModel):
    upvotes: Optional[int] = None
    downvotes: Optional[int] = None
    score: Optional[int] = None
    upvote_ratio: Optional[float] = None
    comments_count: Optional[int] = None
    likes: Optional[int] = None
    shares: Optional[int] = None
    views: Optional[int] = None


class ETLResult(BaseModel):
    source_url: str
    source_domain: Optional[str] = None
    source_title: Optional[str] = None
    source_type: Optional[str] = Field(None, description="post or comment")

    author: Optional[str] = None
    author_is_op: Optional[bool] = None
    author_role: Optional[str] = None
    author_flair: Optional[str] = None
    date_iso: Optional[str] = Field(None, description="ISO8601 timestamp if available")
    raw_text: Optional[str] = None
    snippet: Optional[str] = None
    context: Optional[str] = None

    polarity: Optional[str] = Field(
        None,
        description="one of: positive, negative, mixed, neutral, uncertain",
    )
    opinion_tags: List[str] = Field(default_factory=list)
    opinion_summary: Optional[str] = None
    evidence_quotes: List[str] = Field(default_factory=list)
    stance_axes: Dict[str, Any] = Field(default_factory=dict)
    misc_facts: Dict[str, Any] = Field(default_factory=dict)

    participant_profile: Optional[ParticipantProfile] = None
    study_info: Optional[StudyInfo] = None
    thread_metadata: Optional[ThreadMetadata] = None
    engagement: Optional[Engagement] = None

    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainties: List[str] = Field(default_factory=list)
    is_human: Optional[bool] = None
    # Optional debug: not guaranteed to be present
    raw_html_excerpt: Optional[str] = None


def load_env() -> None:
    load_dotenv()


def get_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError(
            "OpenAI SDK not available. Install with: pip install openai"
        )
    # Accept multiple possible env var names
    candidate_names = [
        "OPENAI_API_KEY",
        "OPENAI_KEY",
        "OPENAI",
        "OPENAI_TOKEN",
    ]
    api_key = None
    for name in candidate_names:
        val = os.getenv(name)
        if val:
            api_key = val
            # Normalize into OPENAI_API_KEY for the SDK
            os.environ.setdefault("OPENAI_API_KEY", val)
            break
    # OpenRouter support
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    default_headers = {}
    if openrouter_key and not api_key:
        api_key = openrouter_key
        os.environ.setdefault("OPENAI_API_KEY", openrouter_key)
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        default_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "https://local.instascrape"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "instascrape-etl"),
        }
    if not api_key:
        present = [k for k in os.environ.keys() if "OPENAI" in k]
        print(
            "Error: No OpenAI API key found in environment.\n"
            "Expected one of: OPENAI_API_KEY, OPENAI_KEY, OPENAI, OPENAI_TOKEN.\n"
            f"Detected OPENAI-related env names: {present if present else 'none'}.\n"
            "Add your key to .env as OPENAI_API_KEY=...",
            file=sys.stderr,
        )
        sys.exit(2)
    if base_url or default_headers:
        return OpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> Anthropic:
    if Anthropic is None:
        raise RuntimeError(
            "Anthropic SDK not available. Install with: pip install anthropic"
        )
    # Accept multiple candidate names
    candidate_names = [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_KEY",
        "CLAUDE_API_KEY",
    ]
    api_key = None
    for name in candidate_names:
        val = os.getenv(name)
        if val:
            api_key = val
            break
    if not api_key:
        present = [k for k in os.environ.keys() if "ANTHROPIC" in k or "CLAUDE" in k]
        print(
            "Error: No Anthropic API key found in environment.\n"
            "Expected one of: ANTHROPIC_API_KEY, ANTHROPIC_KEY, CLAUDE_API_KEY.\n"
            f"Detected Anthropic-related env names: {present if present else 'none'}.\n"
            "Add your key to .env as ANTHROPIC_API_KEY=...",
            file=sys.stderr,
        )
        sys.exit(2)
    return Anthropic(api_key=api_key)


def pick_first_reddit_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in results:
        url = (r.get("url") or "").lower()
        if "reddit.com" in url and "/comments/" in url:
            return r
    return None


def pick_first_non_reddit(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for r in results:
        url = (r.get("url") or "").lower()
        if "reddit.com" not in url:
            return r
    return None
def extract_thread_metadata(reddit_json: List[Any], thread_url: str) -> Dict[str, Any]:
    try:
        post = reddit_json[0]["data"]["children"][0]["data"]
    except Exception:
        return {"thread_url": thread_url}
    created_iso = from_unix_ts_to_iso(post.get("created_utc"))
    return {
        "thread_url": thread_url,
        "subreddit": post.get("subreddit"),
        "thread_title": post.get("title"),
        "score": post.get("score"),
        "num_comments": post.get("num_comments"),
        "created_iso": created_iso,
        "over_18": post.get("over_18"),
        "link_flair_text": post.get("link_flair_text"),
    }



def fetch_reddit_json(thread_url: str, timeout_seconds: int = 20) -> Optional[List[Any]]:
    url = thread_url
    if not url.endswith(".json"):
        url = url.rstrip("/") + ".json"
    headers = {
        "Accept": "application/json",
        "User-Agent": "instascrape-etl/1.0 (+https://example.local)",
    }
    resp = requests.get(url, headers=headers, timeout=timeout_seconds)
    if not resp.ok:
        return None
    try:
        return resp.json()
    except ValueError:
        return None


def fetch_generic_page(url: str, timeout_seconds: int = 20) -> Dict[str, Any]:
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "User-Agent": "instascrape-etl/1.0 (+https://example.local)",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout_seconds)
        html = resp.text if resp.ok else ""
    except Exception:
        return {"url": url, "status": None, "title": None, "text": None, "html_excerpt": None}
    title = None
    text = None
    try:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None
        # Basic text extraction; we keep it minimal to avoid custom parsers
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join([p for p in paragraphs if p])[:8000]
        html_excerpt = html[:2000]
    except Exception:
        html_excerpt = None
    return {
        "url": url,
        "status": resp.status_code if 'resp' in locals() else None,
        "title": title,
        "text": text,
        "html_excerpt": html_excerpt,
    }

def from_unix_ts_to_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()
    except Exception:
        return None


def extract_reddit_post_payload(reddit_json: List[Any]) -> Optional[Dict[str, Any]]:
    """Return a dict with fields for the top-level post (not comments)."""
    try:
        post = reddit_json[0]["data"]["children"][0]["data"]
    except Exception:
        return None
    title = post.get("title")
    selftext = post.get("selftext") or ""
    author = post.get("author")
    author_flair = post.get("author_flair_text")
    created_utc = post.get("created_utc")
    date_iso = from_unix_ts_to_iso(created_utc)
    url = post.get("url")
    permalink = post.get("permalink")
    full_url = f"https://www.reddit.com{permalink}" if permalink else url
    engagement = {
        "upvotes": post.get("ups"),
        "downvotes": post.get("downs"),
        "score": post.get("score"),
        "upvote_ratio": post.get("upvote_ratio"),
        "comments_count": post.get("num_comments"),
    }

    return {
        "source_type": "post",
        "source_title": title,
        "author": author,
        "author_is_op": True,
        "author_role": "OP",
        "author_flair": author_flair,
        "date_iso": date_iso,
        "raw_text": f"{title}\n\n{selftext}".strip(),
        "context": "Top-level post",
        "source_url": full_url or url,
        "source_domain": "www.reddit.com",
        "engagement": engagement,
    }


def extract_first_top_comment_payload(reddit_json: List[Any]) -> Optional[Dict[str, Any]]:
    """Return a dict for the first top-level comment, if any."""
    try:
        post_title = reddit_json[0]["data"]["children"][0]["data"].get("title")
        comments = reddit_json[1]["data"]["children"]
        if not comments:
            return None
        # Find first real comment (skip kind == more)
        first = None
        for child in comments:
            if child.get("kind") == "t1":
                first = child["data"]
                break
        if not first:
            return None
    except Exception:
        return None

    body = first.get("body") or ""
    author = first.get("author")
    created_utc = first.get("created_utc")
    date_iso = from_unix_ts_to_iso(created_utc)
    permalink = first.get("permalink")
    full_url = f"https://www.reddit.com{permalink}" if permalink else None
    engagement = {
        "upvotes": first.get("ups"),
        "downvotes": first.get("downs"),
        "score": first.get("score"),
    }

    return {
        "source_type": "comment",
        "source_title": post_title,
        "author": author,
        "date_iso": date_iso,
        "raw_text": body,
        "context": f"In response to post: {post_title}",
        "source_url": full_url,
        "source_domain": "www.reddit.com",
        "engagement": engagement,
    }


def call_openai_extract(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Ask the model to extract the structured fields. Return parsed JSON dict."""
    schema_hint = {
        "source_url": "string",
        "source_domain": "string|null",
        "source_title": "string|null",
        "source_type": "post|comment|null",
        "author": "string|null",
        "date_iso": "ISO8601|null",
        "raw_text": "string|null",
        "snippet": "string|null",
        "context": "string|null",
        "polarity": "positive|negative|mixed|neutral|uncertain|null",
        "opinion_tags": ["strings"],
        "misc_facts": {"k": "v"},
        "is_human": "boolean|null",
        "engagement": {
            "upvotes": "number|null",
            "downvotes": "number|null",
            "score": "number|null",
            "upvote_ratio": "number|null",
            "comments_count": "number|null",
            "likes": "number|null",
            "shares": "number|null",
            "views": "number|null",
        },
        "confidence": "number 0..1|null",
        "uncertainties": ["strings"],
    }

    instructions = (
        "Extract structured fields about clinical trial opinions. "
        "Be conservative: do not infer beyond the text. If not explicitly stated, set null. "
        "Return STRICT JSON only, matching the schema keys exactly. "
        "Pick a short, non-redundant snippet (<=300 chars). Keep 'opinion_tags' concrete (e.g., 'safety concerns', 'compensation-motivated', 'side-effects-severity', 'access-difficulty', 'phase-1', 'positive-outcome', 'negative-outcome', 'logistics-burden'). "
        "'misc_facts' should contain non-opinion factual data (age, disease, phase, pay, location, sponsor) if present. Populate 'engagement' if metrics are present in provided metadata or text. "
        "Set 'is_human' = true if the content represents a real human opinion/experience; otherwise false and set polarity=null and tags=[]. "
        "Provide 'polarity' as one of: positive, negative, mixed, neutral, or uncertain."
    )

    content_doc = {
        "metadata": {
            "source_url": payload.get("source_url"),
            "source_domain": payload.get("source_domain"),
            "source_title": payload.get("source_title"),
            "source_type": payload.get("source_type"),
            "author": payload.get("author"),
            "date_iso": payload.get("date_iso"),
            "context": payload.get("context"),
        },
        "document": payload.get("raw_text"),
        "schema": schema_hint,
        "thread_metadata": payload.get("thread_metadata"),
    }

    messages = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": (
                "You will receive JSON with metadata and a document. "
                "Return only the final JSON extraction (no commentary).\n\n" + json.dumps(content_doc)
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Best effort: wrap raw text
        return {"_raw_model_text": text}


def call_anthropic_extract(
    client: Anthropic,
    model: str,
    payload: Dict[str, Any],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Ask Claude to extract structured fields. Return parsed JSON dict."""
    schema_hint = {
        "source_url": "string",
        "source_domain": "string|null",
        "source_title": "string|null",
        "source_type": "post|comment|null",
        "author": "string|null",
        "date_iso": "ISO8601|null",
        "raw_text": "string|null",
        "snippet": "string|null",
        "context": "string|null",
        "polarity": "positive|negative|mixed|neutral|uncertain|null",
        "opinion_tags": ["strings"],
        "misc_facts": {"k": "v"},
        "is_human": "boolean|null",
        "engagement": {
            "upvotes": "number|null",
            "downvotes": "number|null",
            "score": "number|null",
            "upvote_ratio": "number|null",
            "comments_count": "number|null",
            "likes": "number|null",
            "shares": "number|null",
            "views": "number|null",
        },
        "confidence": "number 0..1|null",
        "uncertainties": ["strings"],
    }

    instructions = (
        "Extract structured fields about clinical trial opinions. "
        "Be conservative: do not infer beyond the text. If not explicitly stated, set null. "
        "Return STRICT JSON only, matching the schema keys exactly. "
        "Pick a short, non-redundant snippet (<=300 chars). Keep 'opinion_tags' concrete (e.g., 'safety concerns', 'compensation-motivated', 'side-effects-severity', 'access-difficulty', 'phase-1', 'positive-outcome', 'negative-outcome', 'logistics-burden'). "
        "'misc_facts' should contain non-opinion factual data (age, disease, phase, pay, location, sponsor) if present. Populate 'engagement' if metrics are present in metadata or text. "
        "Set 'is_human' = true if the content represents a real human opinion/experience; otherwise false and set polarity=null and opinion_tags=[]. "
        "Provide 'polarity' as one of: positive, negative, mixed, neutral, or uncertain."
    )

    content_doc = {
        "metadata": {
            "source_url": payload.get("source_url"),
            "source_domain": payload.get("source_domain"),
            "source_title": payload.get("source_title"),
            "source_type": payload.get("source_type"),
            "author": payload.get("author"),
            "date_iso": payload.get("date_iso"),
            "context": payload.get("context"),
        },
        "document": payload.get("raw_text"),
        "schema": schema_hint,
        "thread_metadata": payload.get("thread_metadata"),
    }

    # Claude API
    response = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=1200,
        system=instructions,
        messages=[
            {
                "role": "user",
                "content": (
                    "You will receive JSON with metadata and a document. "
                    "Return only the final JSON extraction (no commentary).\n\n"
                    + json.dumps(content_doc)
                ),
            }
        ],
    )

    # Extract text content from Claude response
    try:
        # Claude returns a list of content blocks; pick text blocks
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        text = "\n".join(parts) if parts else "{}"
    except Exception:
        text = "{}"

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_raw_model_text": text}


def quality_review(extracted: Dict[str, Any]) -> List[str]:
    notes = []
    if "_raw_model_text" in extracted:
        notes.append("Model did not return strict JSON; parsing fallback used.")
    # Basic checks
    if not extracted.get("raw_text"):
        notes.append("Missing raw_text.")
    if extracted.get("polarity") not in {"positive", "negative", "mixed", "neutral", "uncertain", None}:
        notes.append("Polarity missing or invalid.")
    tags = extracted.get("opinion_tags") or []
    if isinstance(tags, list) and len(tags) == 0:
        notes.append("Opinion tags empty; may need better prompting.")
    if not extracted.get("snippet"):
        notes.append("Snippet missing; ensure extraction includes a concise quote.")
    if extracted.get("is_human") is False:
        notes.append("Classified as not a human opinion; skipping further analysis.")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search -> fetch Reddit JSON -> LLM extract structured fields (single test)."
    )
    parser.add_argument(
        "--query",
        "-q",
        default="clinical trial experience reddit",
        help="Brave search query (default focuses on Reddit experiences)",
    )
    parser.add_argument("--count", "-c", type=int, default=5, help="Search results to fetch")
    parser.add_argument("--max-items", type=int, default=1, help="Number of items to analyze")
    parser.add_argument(
        "--use-comment",
        action="store_true",
        help="Analyze the first top-level comment instead of the post",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-4-20250514",
        help="Model name (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider selection (default anthropic)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM extraction and emit baseline fields only",
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Print full extracted JSON and a brief quality review",
    )
    parser.add_argument(
        "--domain",
        help="High-level domain description D to guide iterative query sampling",
    )
    parser.add_argument(
        "--niters",
        type=int,
        default=0,
        help="Number of iterative sampling iterations (0 = single mode)",
    )
    parser.add_argument(
        "--results-per-iter",
        type=int,
        default=3,
        help="How many search results to attempt to extract per iteration",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.2,
        help="Delay in seconds between API calls (respect rate limits)",
    )

    args = parser.parse_args()

    load_env()

    brave_key = os.getenv("BRAVE_API_KEY")
    if not brave_key:
        print(
            "Error: BRAVE_API_KEY not found. Ensure .env is configured.",
            file=sys.stderr,
        )
        sys.exit(1)

    def llm_extract(payload: Dict[str, Any]) -> Dict[str, Any]:
        # Step 5: call LLM (unless no-llm)
        if args.no_llm:
            # Baseline extraction without LLM
            return {
                "source_url": payload.get("source_url"),
                "source_domain": payload.get("source_domain"),
                "source_title": payload.get("source_title"),
                "source_type": payload.get("source_type"),
                "author": payload.get("author"),
                "date_iso": payload.get("date_iso"),
                "raw_text": payload.get("raw_text"),
                "snippet": (payload.get("raw_text") or "")[:300],
                "context": payload.get("context"),
                "polarity": "uncertain",
                "opinion_tags": [],
                "misc_facts": {},
                "confidence": None,
                "uncertainties": [
                    "LLM disabled via --no-llm; fields are baseline only",
                ],
            }
        provider = args.provider
        if provider == "auto":
            provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else ("openai" if (os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")) else "openai")
        model = args.model
        try:
            if provider == "anthropic":
                client = get_anthropic_client()
                if "claude" not in model:
                    model = "claude-3-5-sonnet-latest"
                return call_anthropic_extract(client, model, payload)
            else:
                client = get_openai_client()
                return call_openai_extract(client, model, payload)
        except Exception as e:
            return {
                "source_url": payload.get("source_url"),
                "source_domain": payload.get("source_domain"),
                "source_title": payload.get("source_title"),
                "source_type": payload.get("source_type"),
                "author": payload.get("author"),
                "date_iso": payload.get("date_iso"),
                "raw_text": payload.get("raw_text"),
                "snippet": (payload.get("raw_text") or "")[:300],
                "context": payload.get("context"),
                "polarity": "uncertain",
                "opinion_tags": [],
                "misc_facts": {},
                "confidence": None,
                "uncertainties": [
                    f"LLM call failed: {str(e)}",
                ],
            }

    # Single or iterative mode
    all_outputs: List[Dict[str, Any]] = []
    if args.niters and args.domain:
        domain = args.domain
        prior_queries: List[str] = []
        for i in range(args.niters):
            # Sample a new query via LLM given the domain and history
            prompt = {
                "domain": domain,
                "prior": prior_queries,
                "instruction": "Generate ONE concise search query string to discover diverse, firsthand patient opinions on clinical trials. Avoid repeating prior queries; prefer new phrasing, different communities (forums, disease-specific boards), non-Reddit sites, and recent years in text."
            }
            q_payload = {
                "source_url": "about:blank",
                "source_domain": "local",
                "source_title": "query-sampler",
                "source_type": "meta",
                "author": None,
                "date_iso": datetime.now(timezone.utc).isoformat(),
                "raw_text": json.dumps(prompt),
                "context": "Return only the query text as snippet or opinion_summary.",
            }
            q_extracted = llm_extract(q_payload)
            # Heuristic: use opinion_summary or snippet as the query
            next_query = (
                (q_extracted.get("opinion_summary") or "").strip()
                or (q_extracted.get("snippet") or "").strip()
                or args.query
            )
            if not next_query:
                next_query = args.query
            prior_queries.append(next_query)

            # Search
            data, _headers = brave_web_search(
                api_key=brave_key,
                query=next_query,
                count=max(1, min(args.results_per_iter, 10)),
                search_lang="en",
                safesearch="moderate",
            )
            results = extract_web_results(data) or []

            # For each result, build a minimal payload and let LLM extract
            for r in results:
                url = r.get("url")
                if not url:
                    continue
                # Prefer Reddit JSON if available; otherwise generic fetch
                payload = None
                thread_meta = None
                if "reddit.com" in url:
                    rjson = fetch_reddit_json(url)
                    if rjson:
                        thread_meta = extract_thread_metadata(rjson, url)
                        payload = extract_first_top_comment_payload(rjson) or extract_reddit_post_payload(rjson)
                if not payload:
                    page = fetch_generic_page(url)
                    payload = {
                        "source_url": url,
                        "source_domain": r.get("domain") or r.get("meta_url", {}).get("hostname"),
                        "source_title": page.get("title") or r.get("title"),
                        "source_type": "post",
                        "author": None,
                        "date_iso": None,
                        "raw_text": page.get("text") or r.get("snippet") or r.get("description"),
                        "context": "Generic extraction from web page; classify if this is a real human opinion about clinical trials; if not, set polarity=null and tags empty.",
                        "raw_html_excerpt": page.get("html_excerpt"),
                    }
                payload["thread_metadata"] = thread_meta or {"thread_url": url}

                extracted = llm_extract(payload)
                try:
                    etl = ETLResult(**extracted)
                    all_outputs.append(etl.model_dump())
                except ValidationError:
                    all_outputs.append(extracted)
                time.sleep(args.delay)
            time.sleep(args.delay)
    else:
        # Single mode: prefer Reddit if present to show concrete demo
        data, _headers = brave_web_search(
            api_key=brave_key,
            query=args.query,
            count=max(1, min(args.count, 10)),
            search_lang="en",
            safesearch="moderate",
        )
        results = extract_web_results(data)
        if not results:
            print("No search results.")
            sys.exit(0)

        picked = pick_first_reddit_result(results) or pick_first_non_reddit(results)
        if not picked:
            print("No usable result found.")
            sys.exit(0)

        url = picked["url"]
        print(f"Selected result: {url}")
        payload: Optional[Dict[str, Any]] = None
        thread_meta = None
        if "reddit.com" in url:
            rjson = fetch_reddit_json(url)
            if rjson:
                thread_meta = extract_thread_metadata(rjson, url)
                payload = (
                    extract_first_top_comment_payload(rjson)
                    if args.use_comment
                    else extract_reddit_post_payload(rjson)
                )
        if not payload:
            page = fetch_generic_page(url)
            payload = {
                "source_url": url,
                "source_domain": picked.get("domain") or picked.get("meta_url", {}).get("hostname"),
                "source_title": page.get("title") or picked.get("title"),
                "source_type": "post",
                "author": None,
                "date_iso": None,
                "raw_text": page.get("text") or picked.get("snippet") or picked.get("description"),
                "context": "Generic extraction from web page; no custom parser",
            }
        payload["thread_metadata"] = thread_meta or {"thread_url": url}

        extracted = llm_extract(payload)
        try:
            etl = ETLResult(**extracted)
            all_outputs.append(etl.model_dump())
        except ValidationError:
            all_outputs.append(extracted)

    # Print and persist all outputs
    for out in all_outputs:
        if args.raw_output:
            print("\nExtracted JSON:")
            print(json.dumps(out, indent=2, ensure_ascii=False))
        notes = quality_review(out)
        if notes:
            print("\nQuality review notes:")
            for n in notes:
                print(f"- {n}")
        if out.get("uncertainties"):
            print("\nModel uncertainties:")
            for u in out["uncertainties"]:
                print(f"- {u}")
        try:
            with open("etl_results.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    main()


