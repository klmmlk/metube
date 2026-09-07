"""Site-specific URL expanders for MeTube.

This module exposes expand_url(url) which returns a list of episode/item URLs
when the provided URL is a series/index page that links to multiple episode
pages (e.g. CCTV series listing pages). Returns an empty list or None if no
expansion is available or failed.

Keep this small and dependency-light: requests + bs4 are already common in the
project environment; if not present the caller will need to install them.
"""

from typing import List, Optional
from urllib.parse import urljoin, urlparse
import re

import requests
from bs4 import BeautifulSoup

_CCTV_EPISODE_RE = re.compile(r"https?://(?:tv\.)?cctv\.cn/\d{4}/\d{2}/\d{2}/VID[0-9A-Za-z]+\.shtml")


def _expand_cctv_series(url: str) -> List[str]:
    """Fetch a CCTV series/index page and return a list of episode page URLs.

    Conservative approach: parse <a href> links and also scan raw HTML for
    absolute episode links matching the common pattern.
    """
    session = requests.Session()
    resp = session.get(url, timeout=10)
    resp.raise_for_status()
    html = resp.text

    soup = BeautifulSoup(html, "html.parser")
    seen = set()
    ordered = []

    def add(u: str):
        if u not in seen:
            seen.add(u)
            ordered.append(u)

    # 1) look for anchors
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(url, href)
        if _CCTV_EPISODE_RE.match(full):
            add(full)

    # 2) fallback: regex search in HTML for absolute matches
    for m in _CCTV_EPISODE_RE.finditer(html):
        add(m.group(0))

    return ordered


def expand_url(url: str) -> Optional[List[str]]:
    """Return a list of expanded item URLs for known sites, or None/[] when
    no expansion applies.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    host = (parsed.hostname or "").lower()
    # CCTV series pages are on tv.cctv.cn (or cctv.com)
    if host.endswith("cctv.cn") or host.endswith("cctv.com"):
        try:
            eps = _expand_cctv_series(url)
            return eps
        except Exception:
            # Don't raise here; expansion failure should not block the default
            # yt-dlp extraction path. Caller will log a warning.
            return []

    return []
