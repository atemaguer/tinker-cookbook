
"""
Fetch xAI job roles from Greenhouse and write them to:
- roles.txt  (one-line summary per role)
- roles.json (full payload including job description HTML/text)

Uses the public Greenhouse board API:
https://boards-api.greenhouse.io/v1/boards/xai/jobs?content=true
"""

from __future__ import annotations

import html as html_lib
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None  # type: ignore

BOARD_TOKEN = "xai"
API_URL = f"https://boards-api.greenhouse.io/v1/boards/{BOARD_TOKEN}/jobs?content=true"
TXT_OUTFILE = "roles.txt"
JSON_OUTFILE = "roles.json"


def fetch_jobs() -> List[Dict[str, Any]]:
    """
    Fetch job data from Greenhouse.
    Returns a list of job dicts or an empty list on failure.
    """
    # Some environments may miss CA bundles; bypass verification to avoid failures.
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        API_URL, headers={"User-Agent": "Mozilla/5.0 (compatible; role-scraper/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            payload = resp.read()
    except urllib.error.URLError as exc:  # network / DNS / HTTP errors
        sys.stderr.write(f"Request failed: {exc}\n")
        return []

    try:
        data = json.loads(payload)
        return data.get("jobs", []) or []
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"Failed to parse JSON: {exc}\n")
        return []


def format_job(job: Dict[str, Any]) -> str:
    title = (job.get("title") or "").strip()
    location = (job.get("location") or {}).get("name", "").strip()
    url = (job.get("absolute_url") or "").strip()
    return f"{title} [{location}] - {url}"


def write_jobs(jobs: List[Dict[str, Any]], outfile: str = TXT_OUTFILE) -> int:
    lines = [format_job(job) for job in jobs if job.get("title") and job.get("absolute_url")]
    with open(outfile, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return len(lines)


def strip_tags(html_content: str) -> str:
    """
    Convert HTML to readable text:
    - unescape entities
    - remove scripts/styles
    - preserve list items as bullets
    - keep reasonable newlines
    """
    html_content = html_lib.unescape(html_content or "")

    # If BeautifulSoup is available, use it for robust parsing.
    if BeautifulSoup:
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Turn <br> into newlines.
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Convert list items to bullet lines.
        for li in soup.find_all("li"):
            bullet = f"- {li.get_text(' ', strip=True)}\n"
            li.replace_with(bullet)

        # Add a newline after common block elements to keep paragraphs separated.
        for block in soup.find_all(["p", "div", "section", "article", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6"]):
            if block and (not str(block).endswith("\n")):
                block.append("\n")

        text = soup.get_text()
    else:
        # Fallback regex-based stripper (less robust).
        html_content = re.sub(r"(?is)<(script|style)[^>]*>.*?</\\1>", " ", html_content)
        html_content = re.sub(r"(?i)<li[^>]*>", "- ", html_content)
        html_content = re.sub(r"(?i)</(p|div|section|article|ul|ol|li|br|h[1-6])>", "\n", html_content)
        text = re.sub(r"<[^>]+>", " ", html_content)
        text = html_lib.unescape(text)

    # Normalize whitespace.
    text = re.sub(r"[\r\t\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    content_html: Optional[str] = job.get("content")
    content_text = strip_tags(content_html) if content_html else ""
    return {
        "id": job.get("id"),
        "title": job.get("title"),
        "absolute_url": job.get("absolute_url"),
        "location": (job.get("location") or {}).get("name"),
        "departments": [d.get("name") for d in job.get("departments") or []],
        "offices": [o.get("name") for o in job.get("offices") or []],
        "metadata": job.get("metadata") or [],
        "updated_at": job.get("updated_at"),
        "content_html": content_html,
        "content_text": content_text,
    }


def write_jobs_json(jobs: List[Dict[str, Any]], outfile: str = JSON_OUTFILE) -> int:
    normalized = [normalize_job(job) for job in jobs]
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(normalized, fh, ensure_ascii=False, indent=2)
    return len(normalized)


def main(txt_out: str = TXT_OUTFILE, json_out: str = JSON_OUTFILE) -> None:
    jobs = fetch_jobs()
    txt_count = write_jobs(jobs, outfile=txt_out)
    json_count = write_jobs_json(jobs, outfile=json_out)
    print(f"Saved {txt_count} role summaries to {txt_out}")
    print(f"Saved {json_count} detailed roles to {json_out}")


if __name__ == "__main__":
    # Optional CLI: python scrape.py [roles.txt] [roles.json]
    txt_out = sys.argv[1] if len(sys.argv) > 1 else TXT_OUTFILE
    json_out = sys.argv[2] if len(sys.argv) > 2 else JSON_OUTFILE
    main(txt_out=txt_out, json_out=json_out)

