"""Site-specific URL expanders for MeTube.

expand_url(url) fetches a series/index page for sites where one submitted URL
stands for many episode pages (today: CCTV series/合集 pages) and returns the
list of concrete episode URLs, so each episode enters the queue on its own --
and, for CCTV, goes on to the per-episode quality resolver in ``cctv.py``.

The contract is deliberately forgiving: any failure (unknown site, fetch
error, parse error, or the URL itself being an episode page) returns an empty
list, and the caller falls back to handing the original URL to yt-dlp.
Expansion must never make a URL that would have worked stop working.

Implemented with aiohttp + regex only. requests/bs4 are not declared in
pyproject.toml while aiohttp already is, and the call site is async, so the
fetch is async too. Redirects followed by aiohttp are not re-validated --
the same accepted stance as the classification extraction in
``DownloadQueue.__extract_info``; every URL this module *outputs* is a
VIDE page link that goes back through ``DownloadQueue.add`` and its
validate_url gate.
"""

import logging
import re
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp

log = logging.getLogger('site_expanders')

# One episode page, e.g. https://tv.cctv.com/2024/03/01/VIDAb2Xx123.shtml.
# Both cctv.com and cctv.cn serve them; older commits only recognized cctv.cn.
_CCTV_EPISODE_RE = re.compile(
    r"https?://(?:tv\.)?cctv\.(?:com|cn)/\d{4}/\d{2}/\d{2}/VID[0-9A-Za-z]+\.shtml")

# 'url': 'https://...VID....shtml' fields inside inline <script> JSON blobs
# (series pages embed their episode lists there as often as in anchors).
_JS_URL_FIELD_RE = re.compile(
    r"['\"]url['\"]\s*:\s*['\"]"
    r"(https?://(?:tv\.)?cctv\.(?:com|cn)/\d{4}/\d{2}/\d{2}/VID[0-9A-Za-z]+\.shtml)['\"]",
    re.IGNORECASE)

# <a href="..."> attribute extraction (regex stand-in for the old bs4 pass;
# good enough because the follow-up match against _CCTV_EPISODE_RE rejects
# everything that is not a clean episode URL anyway).
_HREF_RE = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)

_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=10, connect=5)
_MAX_HTML_BYTES = 2 * 1024 * 1024
_UA = 'Lavf/60.10.100'


def is_cctv_url(url: str) -> bool:
    """True for any *.cctv.com / *.cctv.cn (and bare-domain) URL."""
    try:
        host = (urlparse(url).hostname or '').lower()
    except ValueError:
        return False
    return host in ('cctv.com', 'cctv.cn') or host.endswith('.cctv.com') or host.endswith('.cctv.cn')


def _extract_episodes(html: str, base_url: str) -> List[str]:
    """Collect episode URLs from a series page, deduplicated in document order."""
    seen = set()
    ordered = []

    def add(u: str):
        if u not in seen:
            seen.add(u)
            ordered.append(u)

    # 1) anchors (also resolves relative hrefs)
    for m in _HREF_RE.finditer(html):
        full = urljoin(base_url, m.group(1).strip())
        if _CCTV_EPISODE_RE.match(full):
            add(full)

    # 2) absolute episode URLs anywhere in the raw HTML (script blocks included)
    for m in _CCTV_EPISODE_RE.finditer(html):
        add(m.group(0))

    # 3) 'url' fields inside inline script/JSON structures; pass 2 usually
    # already covers these, kept explicit for URLs that only appear quoted
    for m in _JS_URL_FIELD_RE.finditer(html):
        add(m.group(1))

    return ordered


async def expand_url(url: str) -> Optional[List[str]]:
    """Return the episode URLs a series/index page expands to, or [] when no
    expansion applies or anything failed. Never raises.
    """
    try:
        if not is_cctv_url(url):
            return []
        # An episode page must never expand into its own "related episodes"
        if _CCTV_EPISODE_RE.match(url):
            return []
        async with aiohttp.ClientSession(timeout=_FETCH_TIMEOUT, headers={'User-Agent': _UA}) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                body = await resp.content.read(_MAX_HTML_BYTES)
        html = body.decode('utf-8', errors='replace')
        episodes = _extract_episodes(html, url)
        if episodes:
            log.info('Expanded CCTV series page %s into %d episode(s)', url, len(episodes))
        return episodes
    except Exception:
        log.debug('CCTV series expansion failed for %s', url, exc_info=True)
        return []
