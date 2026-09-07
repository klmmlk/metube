"""CCTV whole-series episode listing.

Why: metube supports YouTube / Bilibili playlist downloads. For CCTV, a
"playlist" is a 剧集 (drama series): one user-submitted single-episode URL
(say https://tv.cctv.cn/2026/09/01/VIDAxOhtc2E2Nk3KrBRhYSbY260901.shtml,
episode 7 of a 13-episode animated series) belongs to a column / season
identified by a TOPC id embedded in that page's inline JS. Given that
column id, the ``api.cntv.cn/NewVideo/getVideoListByColumn`` endpoint
returns the entire season's episodes, paginated by month, so we can hand
the queue a list of all episode URLs in one round trip.

Trigger: opt-in only. Either the UI "download whole series" checkbox
(default unchecked) or a ``?cctv_all=true`` URL query marker. No API call
is made when the user does not opt in.

Three deliberate differences from cctv.py:

* The data source is api.cntv.cn (series metadata), not
  vdn.apps.cntv.cn (single-episode bitrates). The two APIs are
  independent: a miss here never affects the per-episode resolve_episode
  path.
* The endpoint is paginated (per month, per page) for long series. A
  probe call per month learns total; if total <= page_size we skip the
  page loop.
* Invoked only when the user opts in. When the user does not, this
  module is not called at all.

Never-worse contract: fetch_series_episodes returns SeriesResult with
``kind='unknown'`` or ``'single'`` on any failure. The caller proceeds
exactly as without this module -- the episode URL itself enters the
queue as a single download.
"""

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from typing import Optional
from urllib.parse import urlencode

import aiohttp

from cctv import is_cctv_episode_url
from url_guard import validate_url

log = logging.getLogger('cctv_series')

UA = 'Lavf/60.10.100'

# Endpoints (CCTVVideoDownloader/src/core/src/apiservice.cpp:
# buildVideoApiUrl, buildAlbumVideoListUrl, buildVideoAlbumInfoById).
COLUMN_API_URL = 'https://api.cntv.cn/NewVideo/getVideoListByColumn'
ALBUM_API_URL = 'https://api.cntv.cn/NewVideo/getVideoListByAlbumIdNew'
ALBUM_INFO_URL = 'https://api.cntv.cn/NewVideoset/getVideoAlbumInfoByVideoId'

# Per-call tuning.
_DEFAULT_PAGE_SIZE = 50
_MAX_EPISODES = 500            # hard ceiling on a single series result
_MAX_MONTHS = 60               # 5 years of month-paging
_API_ATTEMPTS = 3              # retry count for truncated JSON
_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=4, connect=3)
_MAX_BODY_BYTES = 256 * 1024

# Cache: episode lists change slowly (CCTV rarely reshuffles a finished
# season); a longer TTL than cctv.py's 600s keeps repeated adds / retries
# for the same series free. Keyed by (kind, *args) tuples.
_CACHE: 'OrderedDict[tuple, tuple[float, dict]]' = OrderedDict()
_CACHE_TTL = 3600.0
_CACHE_MAX = 256


class SeriesKind(str, Enum):
    SERIES = 'series'    # >= 1 other episode besides the current one
    SINGLE = 'single'    # API returned only the current episode (or nothing)
    UNKNOWN = 'unknown'  # could not determine -- caller treats as single


@dataclass
class SeriesResult:
    kind: SeriesKind
    episodes: list[str] = field(default_factory=list)
    column_id: Optional[str] = None
    source: str = 'none'   # 'column-api' | 'album-api' | 'html-fallback' | 'none'


# --- pure functions --------------------------------------------------------

# Column id (TOPC...) and item id (VIDE.../VIDA...) embedded in episode
# page inline JS. Kept loose on the id-prefix so old / new pages both work.
COLUMN_ID_RE = re.compile(r'var\s+column_id\s*=\s*["\'](TOPC[^"\']+)["\']', re.I)
ITEM_ID_RE = re.compile(r'var\s+item(?:id1?|id|_id)?\s*=\s*["\'](VID[0-9A-Za-z]+)["\']', re.I)
# Publication date (yyyy-MM-dd) anywhere on the page.
PUB_DATE_RE = re.compile(r'(\d{4})-(\d{2})-\d{2}')


def extract_column_id(html: str) -> Optional[str]:
    """Pull the column id (TOPC...) out of an episode page's inline JS."""
    m = COLUMN_ID_RE.search(html)
    return m.group(1) if m else None


def extract_item_id(html: str) -> Optional[str]:
    """Pull the item id (VIDE... or VIDA...) out of the same inline JS."""
    m = ITEM_ID_RE.search(html)
    return m.group(1) if m else None


def extract_pub_date_yyyymm(html: str) -> Optional[str]:
    """First yyyy-MM-dd in the page as yyyyMM (or None)."""
    m = PUB_DATE_RE.search(html)
    return f'{m.group(1)}{m.group(2)}' if m else None


def build_column_url(column_id: str, yyyymm: str, page: int = 1,
                     page_size: int = _DEFAULT_PAGE_SIZE) -> str:
    """getVideoListByColumn URL with all required parameters."""
    qs = urlencode({
        'id': column_id, 'd': yyyymm, 'p': page, 'n': page_size,
        'mode': 0, 'serviceId': 'tvcctv', 'sort': 'desc',
    })
    return f'{COLUMN_API_URL}?{qs}'


def build_album_url(album_id: str, page: int = 1,
                    page_size: int = _DEFAULT_PAGE_SIZE) -> str:
    """getVideoListByAlbumIdNew URL."""
    qs = urlencode({
        'id': album_id, 'mode': 0, 'pub': 1, 'serviceId': 'tvcctv',
        'sort': 'asc', 'p': page, 'n': page_size,
    })
    return f'{ALBUM_API_URL}?{qs}'


def build_album_info_url(video_id: str) -> str:
    """URL to look up the VIDA album id for a given VIDE video id."""
    return f'{ALBUM_INFO_URL}?{urlencode({"id": video_id, "serviceId": "tvcctv"})}'


def parse_list_response(payload: dict) -> list:
    """Extract the episode dict list from a getVideoListByColumn response.

    Returns ``[]`` on any structural miss (and never raises). The response
    shape: ``{"data": {"total": N, "list": [{guid, url, title, time, ...}]}}``.
    """
    if not isinstance(payload, dict):
        return []
    data = payload.get('data')
    if not isinstance(data, dict):
        return []
    lst = data.get('list')
    if not isinstance(lst, list):
        return []
    return [e for e in lst if isinstance(e, dict)]


def parse_album_info(payload: dict) -> Optional[str]:
    """Extract the VIDA album id from a getVideoAlbumInfoByVideoId response."""
    if not isinstance(payload, dict):
        return None
    data = payload.get('data')
    if not isinstance(data, dict):
        return None
    album_id = data.get('id')
    if isinstance(album_id, str) and album_id.startswith('VID'):
        return album_id
    return None


def _dedupe_key(ep: dict) -> Optional[str]:
    """Stable identity for an episode entry (guid preferred, else url)."""
    g = ep.get('guid')
    if isinstance(g, str) and g:
        return f'guid:{g}'
    u = ep.get('url')
    if isinstance(u, str) and u:
        return f'url:{u}'
    return None


def classify_series(episodes: list, current_item_id: Optional[str]) -> SeriesKind:
    """Decide whether *episodes* (from the series API) describes a real
    multi-episode series relative to the user's submitted episode page.

    Returns SERIES when at least one episode is *different* from the
    current one (by item id substring in url), SINGLE when only the
    current one appears (or none), UNKNOWN when the list is empty.
    """
    if not episodes:
        return SeriesKind.UNKNOWN
    for ep in episodes:
        u = ep.get('url')
        if not isinstance(u, str):
            continue
        if current_item_id and current_item_id in u:
            continue
        return SeriesKind.SERIES
    return SeriesKind.SINGLE


def month_range(start_yyyymm: str, end_yyyymm: str):
    """Yield every yyyyMM from start to end inclusive (both yyyyMM form)."""
    sy, sm = int(start_yyyymm[:4]), int(start_yyyymm[4:])
    ey, em = int(end_yyyymm[:4]), int(end_yyyymm[4:])
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f'{y:04d}{m:02d}'
        m += 1
        if m > 12:
            m, y = 1, y + 1


# --- cache -----------------------------------------------------------------

def _cache_get(key):
    hit = _CACHE.get(key)
    if hit is None:
        return None
    ts, payload = hit
    if time.monotonic() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return payload


def _cache_put(key, payload):
    _CACHE[key] = (time.monotonic(), payload)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


# --- network ---------------------------------------------------------------

async def _aiohttp_fetch_checked(session, url, allow_private):
    """GET *url* as text (capped) or None on any miss. Mirrors
    cctv._aiohttp_fetch_checked; duplicated here so cctv_series does not
    import cctv at module load time (cctv imports back in tests).

    Every outbound URL -- including the post-redirect final URL -- passes
    validate_url first: the main process runs no socket guard.
    """
    loop = asyncio.get_running_loop()
    err = await loop.run_in_executor(
        None, partial(validate_url, url, allow_private=allow_private))
    if err is not None:
        log.warning('CCTV series resolver skipping disallowed URL %s: %s', url, err)
        return None
    try:
        async with session.get(url) as resp:
            final = str(resp.url)
            final_err = await loop.run_in_executor(
                None, partial(validate_url, final, allow_private=allow_private))
            if final_err is not None:
                log.warning('CCTV series resolver skipping redirect to disallowed URL %s', final)
                return None
            if resp.status != 200:
                return None
            # content.read(n) is a size hint -- a single call may return
            # fewer bytes than n (TCP segment / chunked encoding). Loop
            # until the response is drained or we hit the cap.
            chunks = []
            received = 0
            while received < _MAX_BODY_BYTES:
                chunk = await resp.content.read(_MAX_BODY_BYTES - received)
                if not chunk:
                    break
                chunks.append(chunk)
                received += len(chunk)
                if len(chunk) < _MAX_BODY_BYTES - received:
                    # short read: server finished before we hit the cap
                    break
            body = b''.join(chunks)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return None
    return body.decode('utf-8', errors='replace')


async def _fetch_json(fetch, session, url):
    """GET, JSON-decode with exponential-backoff retries.

    *fetch* is partialed with ``allow_private`` by the caller (production
    default) or a test stub -- it accepts ``(session, url)`` and returns
    text or None.
    """
    for attempt in range(_API_ATTEMPTS):
        text = await fetch(session, url)
        if text is not None:
            try:
                payload = json.loads(text)
            except ValueError:
                log.debug('CCTV series API returned unparseable JSON (attempt %d/%d)',
                          attempt + 1, _API_ATTEMPTS)
            else:
                if isinstance(payload, dict):
                    return payload
        if attempt < _API_ATTEMPTS - 1:
            await asyncio.sleep(0.5 * (2 ** attempt))
    return None


# --- orchestration ---------------------------------------------------------

def _today_yyyymm() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m')


def _five_years_ago_yyyymm() -> str:
    """yyyyMM five years ago (used when no pub-date is found on the page)."""
    now = datetime.now(timezone.utc)
    y, m = now.year - 5, now.month
    return f'{y:04d}{m:02d}'


def _dedupe_and_trim(episodes: list) -> list:
    """Stable dedupe by guid/url, document order, capped at _MAX_EPISODES."""
    seen = set()
    ordered = []
    for ep in episodes:
        k = _dedupe_key(ep)
        if k is None or k in seen:
            continue
        seen.add(k)
        ordered.append(ep)
        if len(ordered) >= _MAX_EPISODES:
            log.warning('CCTV series truncated to %d episodes (cap)', _MAX_EPISODES)
            break
    return ordered


async def _fetch_column_page(fetch, session, column_id, yyyymm, page, page_size):
    """Cached fetch of one getVideoListByColumn page (returns dict or None)."""
    key = ('column', column_id, yyyymm, page, page_size)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = build_column_url(column_id, yyyymm, page, page_size)
    payload = await _fetch_json(fetch, session, url)
    if payload is not None:
        _cache_put(key, payload)
    return payload


async def _fetch_album_info(fetch, session, video_id):
    """Cached fetch of the VIDE -> VIDA mapping."""
    key = ('album_info', video_id)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = build_album_info_url(video_id)
    payload = await _fetch_json(fetch, session, url)
    if payload is not None:
        _cache_put(key, payload)
    return payload


async def _fetch_album_page(fetch, session, album_id, page, page_size):
    """Cached fetch of one getVideoListByAlbumIdNew page."""
    key = ('album', album_id, page, page_size)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    url = build_album_url(album_id, page, page_size)
    payload = await _fetch_json(fetch, session, url)
    if payload is not None:
        _cache_put(key, payload)
    return payload


async def _probe_column(fetch, session, column_id, start_yyyymm, end_yyyymm):
    """Walk months and pages on the column API; return deduped episode list
    (or [] on failure). One probe per month learns total; if total <=
    page_size we skip the page loop for that month. An ``errcode`` field
    (CCTV API uses errcode 1002 = "data empty") is treated as a definitive
    miss: there is nothing on this column id, so we stop iterating months
    instead of burning the rest of the 60-month budget.
    """
    collected = []
    months_searched = 0
    empty_months = 0
    for yyyymm in month_range(start_yyyymm, end_yyyymm):
        if months_searched >= _MAX_MONTHS:
            log.warning('CCTV series month loop cap reached (%d)', _MAX_MONTHS)
            break
        months_searched += 1
        probe = await _fetch_column_page(
            fetch, session, column_id, yyyymm, 1, _DEFAULT_PAGE_SIZE)
        if probe is None:
            empty_months += 1
            # A few network blips are normal; only give up if every probed
            # month has been empty, so a single 5xx doesn't kill detection.
            if empty_months >= 3 and not collected:
                break
            continue
        # CCTV uses errcode=1002 ("data empty") for months that genuinely
        # have no episodes on this column. Two consecutive empties in a
        # row means the column id is wrong / archived / out of season --
        # stop rather than burning the rest of the budget.
        if probe.get('errcode') == '1002':
            empty_months += 1
            if empty_months >= 2 and not collected:
                break
            continue
        empty_months = 0
        data = probe.get('data') if isinstance(probe.get('data'), dict) else {}
        list_part = parse_list_response(probe)
        if list_part:
            collected.extend(list_part)
        total = data.get('total') if isinstance(data.get('total'), int) else 0
        if total <= _DEFAULT_PAGE_SIZE:
            continue
        n_pages = (total + _DEFAULT_PAGE_SIZE - 1) // _DEFAULT_PAGE_SIZE
        for p in range(2, n_pages + 1):
            payload = await _fetch_column_page(
                fetch, session, column_id, yyyymm, p, _DEFAULT_PAGE_SIZE)
            if payload is None:
                continue
            extra = parse_list_response(payload)
            if extra:
                collected.extend(extra)
            if len(_dedupe_and_trim(collected)) >= _MAX_EPISODES:
                break
        if len(_dedupe_and_trim(collected)) >= _MAX_EPISODES:
            break
    return _dedupe_and_trim(collected)


async def _probe_album(fetch, session, album_id):
    """Walk pages on the album API; return deduped episode list (or [])."""
    collected = []
    page = 1
    while page * _DEFAULT_PAGE_SIZE <= _MAX_EPISODES:
        payload = await _fetch_album_page(
            fetch, session, album_id, page, _DEFAULT_PAGE_SIZE)
        if payload is None:
            break
        # Same definitive-miss guard as _probe_column.
        if isinstance(payload.get('errcode'), str) and payload['errcode'] != '1002':
            break
        list_part = parse_list_response(payload)
        if not list_part:
            break
        collected.extend(list_part)
        data = payload.get('data') if isinstance(payload.get('data'), dict) else {}
        total = data.get('total') if isinstance(data.get('total'), int) else 0
        if total <= page * _DEFAULT_PAGE_SIZE:
            break
        page += 1
    return _dedupe_and_trim(collected)


async def fetch_series_episodes(url, *, allow_private=False, _fetch=None):
    """Resolve a CCTV single-episode URL into the whole series it belongs to.

    Returns SeriesResult with ``kind=SERIES`` and the episode list when at
    least one OTHER episode was found (or ``kind=SINGLE`` when the API
    returned only the current one, ``kind=UNKNOWN`` when nothing could be
    determined). Never raises.
    """
    try:
        if not is_cctv_episode_url(url):
            return SeriesResult(kind=SeriesKind.UNKNOWN)

        if _fetch is not None:
            fetch = _fetch
        else:
            fetch = partial(_aiohttp_fetch_checked, allow_private=allow_private)

        async with aiohttp.ClientSession(
                timeout=_FETCH_TIMEOUT, headers={'User-Agent': UA}) as session:
            html = await fetch(session, url)
            if html is None:
                return SeriesResult(kind=SeriesKind.UNKNOWN)

            column_id = extract_column_id(html)
            item_id = extract_item_id(html)
            pub_yyyymm = extract_pub_date_yyyymm(html)
            end_yyyymm = _today_yyyymm()
            start_yyyymm = pub_yyyymm or _five_years_ago_yyyymm()

            # Level 1: column API via TOPC id (the common path).
            if column_id:
                episodes = await _probe_column(
                    fetch, session, column_id, start_yyyymm, end_yyyymm)
                if episodes:
                    kind = classify_series(episodes, item_id)
                    if kind is SeriesKind.SERIES:
                        urls = [ep['url'] for ep in episodes
                                if isinstance(ep.get('url'), str)]
                        return SeriesResult(
                            kind=SeriesKind.SERIES, episodes=urls,
                            column_id=column_id, source='column-api')
                    if kind is SeriesKind.SINGLE:
                        # API confirmed there is no other episode: stop. The
                        # caller treats this exactly like UNKNOWN (single
                        # download), but the source tells us why.
                        return SeriesResult(
                            kind=SeriesKind.SINGLE,
                            column_id=column_id, source='column-api')

            # Level 2: album API via VIDE -> VIDA conversion (rare path;
            # only used when the episode page has no column_id).
            if item_id:
                info = await _fetch_album_info(fetch, session, item_id)
                album_id = parse_album_info(info) if info else None
                if album_id:
                    episodes = await _probe_album(fetch, session, album_id)
                    if episodes:
                        kind = classify_series(episodes, item_id)
                        if kind is SeriesKind.SERIES:
                            urls = [ep['url'] for ep in episodes
                                    if isinstance(ep.get('url'), str)]
                            return SeriesResult(
                                kind=SeriesKind.SERIES, episodes=urls,
                                column_id=column_id, source='album-api')
                        if kind is SeriesKind.SINGLE:
                            return SeriesResult(
                                kind=SeriesKind.SINGLE,
                                column_id=column_id, source='album-api')

            # Level 3: HTML fallback (existing site_expanders._extract_episodes
            # run against the episode page itself; related-list blocks often
            # list other episodes of the same series).
            from site_expanders import _extract_episodes  # late: avoids cycles
            html_episodes = _extract_episodes(html, url)
            # Drop the current episode (and its #fragment variants) so the
            # caller only sees siblings; already-set in the queue catches it
            # too, but trimming here keeps the per-batch log clean.
            if item_id:
                html_episodes = [u for u in html_episodes
                                 if item_id not in u]
            if html_episodes:
                kind = classify_series(
                    [{'url': u} for u in html_episodes], item_id)
                if kind is SeriesKind.SERIES:
                    return SeriesResult(
                        kind=SeriesKind.SERIES, episodes=html_episodes,
                        column_id=column_id, source='html-fallback')
                # HTML anchors alone are not a reliable series indicator: a
                # single self-link is most likely a related/recommendation
                # block, not proof of a season. Downgrade to UNKNOWN.
                return SeriesResult(
                    kind=SeriesKind.UNKNOWN, column_id=column_id,
                    source='html-fallback')

            # Level 4: nothing worked -- caller treats the URL as a single.
            return SeriesResult(kind=SeriesKind.UNKNOWN, column_id=column_id)
    except Exception:
        log.warning('CCTV series detection failed for %s', url, exc_info=True)
        return SeriesResult(kind=SeriesKind.UNKNOWN)