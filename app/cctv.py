"""CCTV (央视网) highest-quality stream resolver.

Why this exists: yt-dlp's CCTVIE reads only ``video.chapters[0]``,
``video.lowChapters[0]`` and the top-level ``hls_url`` from
``vdn.apps.cntv.cn/api/getHttpVideoInfo.do``. CCTV's ``hls_url`` usually
points at a *single-bitrate media playlist* (one m3u8 per quality tier), so
the extractor ends up with one 'hls' format carrying no height/tbr and
MeTube's ``bestvideo+bestaudio/best`` selector has nothing to sort by --
whatever bitrate the CDN hands out by default is what gets downloaded.

The clear-CDN trick (from letr007/CCTVVideoDownloader, contentresolver.cpp):
``hls_url`` looks like ``.../asp/hls/main/<guid>/main.m3u8``; rewriting the
quality path segment to ``4000/3000/2000/1200/850/450`` (and ``main.m3u8``
to ``<q>.m3u8``) addresses each quality tier directly. Probing from the
highest tier down finds the best the clear CDN offers for that episode.
When the maxbr-stripped ``hls_url`` returns a master playlist, its variants
(whose ``BANDWIDTH`` is mandatory) are ranked directly instead.

Never-worse contract: resolve_episode returns None on *any* failure and the
caller leaves the download exactly as it would have been without this
module. Two subtleties keep a rewritten download working:

* The m3u8 URL is prefixed with ``generic:`` -- a bare ``*.cntv.cn`` /
  ``*.cctv.com`` URL would be claimed by yt-dlp's CCTVIE, which then fails
  to find a guid in the playlist text.
* Streams resolved to a single media playlist carry no height metadata, so
  the caller forces the format selector to ``bestvideo+bestaudio/best``; a
  tier filter like ``bestvideo[height<=720]`` would match nothing. The one
  exception is the master-playlist fallback, which hands yt-dlp the master
  itself: its variants carry real BANDWIDTH/RESOLUTION metadata, so metube's
  usual selectors sort correctly and no format is forced.

Two live-CDN behaviours learned from probing tv.cctv.com in 2026-09 and
encoded in ``_resolve_streams``:

* Episode masters are routinely *incomplete* -- a 新闻联播 master listed
  only the 450 (270p) tier while its 2000/1200/850 directories all served
  playlists. The quality-directory ladder therefore runs BEFORE the master
  is consulted; the master is only a last resort.
* The video-info API intermittently truncates its JSON body mid-string
  (server-side Content-Length overstatement, worse under request bursts;
  ``hls_url`` sits near the end of the payload so truncated responses are
  unusable). ``_fetch_api_info`` retries a couple of times.

Encrypted streams (``manifest.hls_h5e_url`` and friends) routinely hold the
1080p/4K tiers, but their TS segments use CCTV's proprietary H5E NAL-level
encryption; they are recorded here for reporting only (see CctvStream) --
downloading them needs a decryptor and is future work.

The right long-term home for this logic is yt-dlp's CCTVIE; when that lands
upstream, this module and its single call site can be deleted.
"""

import asyncio
import json
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from urllib.parse import urljoin, urlparse, urlencode

import aiohttp

from url_guard import validate_url

log = logging.getLogger('cctv')

API_URL = 'https://vdn.apps.cntv.cn/api/getHttpVideoInfo.do'
UA = 'Lavf/60.10.100'

# Clear-CDN quality tiers, highest first (CCTV CDN path conventions; the
# numbers are nominal bitrates in kbps). 2000~1080p, 1200~720p, 850~480p,
# 450~360p, 4000/3000 are the 4K/2K-plus tiers.
LADDER = ('4000', '3000', '2000', '1200', '850', '450')

# Which rung each MeTube quality starts probing from (then walks down).
_QUALITY_START_INDEX = {
    'best': 0,
    '2160': 0,
    '1440': 1,
    '1080': 2,
    '720': 3,
    '480': 4,
    '360': 5,
    '240': 5,
    'worst': 5,
}

# The format selector forced onto ladder-resolved streams (single media
# playlist -> no height metadata -> tier/codec filters would match nothing).
FORCED_FORMAT = 'bestvideo+bestaudio/best'

# Output-template fields pre-resolved from the classification entry before
# the rewritten (generic m3u8) URL loses them. ext/duration stay dynamic --
# the generic extractor computes those correctly from the stream itself.
OUTTMPL_PRE_RESOLVE_PREFIXES = ('title', 'id', 'uploader', 'upload')

# Overall budget for one resolution; the add() request pays this at most.
RESOLVE_TIMEOUT = 15.0
_PROBE_TIMEOUT = aiohttp.ClientTimeout(total=4, connect=3)
_MAX_BODY_BYTES = 256 * 1024
# The video-info API retries: see _fetch_api_info.
_API_ATTEMPTS = 3
_API_RETRY_DELAY = 0.5

# Episode pages we resolve: tv.cctv.com / tv.cctv.cn "VIDE....shtml" or
# "VIDA....shtml" URLs. Both prefixes occur: VIDE is the standard per-episode
# id, VIDA is the album id that CCTV also uses as a single-episode URL on
# some pages (e.g. 动画 series pages). The companion cctv_series module
# needs both to recognise剧集 source URLs.
_CCTV_HOSTS = ('tv.cctv.com', 'tv.cctv.cn', 'www.tv.cctv.com', 'www.tv.cctv.cn')
_EPISODE_PATH_RE = re.compile(r'/VID[AE][0-9A-Za-z]+\.s?html?$')
_GUID_RE = re.compile(r'^[0-9a-fA-F]{32}$')

# guid extraction from an episode page (the six JS shapes yt-dlp's CCTVIE
# searches for; kept in sync with its _search_regex list).
_GUID_PATTERNS = (
    r'var\s+guid\s*=\s*["\']([\da-fA-F]+)',
    r'videoCenterId(?:["\']\s*,|:)\s*["\']([\da-fA-F]+)',
    r'changePlayer\s*\(\s*["\']([\da-fA-F]+)',
    r'load[Vv]ideo\s*\(\s*["\']([\da-fA-F]+)',
    r'var\s+initMyAray\s*=\s*["\']([\da-fA-F]+)',
    r'var\s+ids\s*=\s*\[["\']([\da-fA-F]+)',
)

# Clear-CDN URL rewriting (contentresolver.cpp clearVariantUrl).
_QUALITY_DIR_RE = re.compile(r'/asp/hls/(?:main|4000|3000|2000|1200|850|450)(?=/)')
_TIER_M3U8_RE = re.compile(r'/(?:main|4000|3000|2000|1200|850|450)\.m3u8(?=[?#]|$)')
# Bare "main" path segment, for CCTV-4K channels whose hls_url does not use
# the /asp/hls/ layout (contentresolver.cpp replaces it with "4000").
_MAIN_SEGMENT_RE = re.compile(r'(?<=/)main(?=/)')

# Master playlist parsing (contentresolver.cpp parseVariants).
_BANDWIDTH_RE = re.compile(r'(?:^|[,:])\s*BANDWIDTH=(\d+)')
_RESOLUTION_RE = re.compile(r'(?:^|[,:])\s*RESOLUTION=(\d+x\d+)')

# Encrypted-stream host normalization onto the DRM CDN
# (contentresolver.cpp normalizeEncryptedPlaylistUrl).
_ENC_HOST_RE = re.compile(r'https://[^/]+/asp/enc2/')
_ENC_HOST = 'https://drm.cntv.vod.dnsv1.com/asp/enc2/'


@dataclass(frozen=True)
class Variant:
    bandwidth: int
    resolution: str
    url: str


@dataclass(frozen=True)
class CctvStream:
    url: str                    # already prefixed with 'generic:'
    forced_format: str | None   # media-playlist results force FORCED_FORMAT;
                                # the master fallback keeps yt-dlp's default
    source: str                 # 'clear-ladder' | '4k' | 'clear-main' | 'clear-master'
    probed_quality: str | None  # tier that hit, e.g. '2000'
    title: str | None           # from the API, when the entry had none
    encrypted_master: str | None = None
    # Future H5E work; populated only when the clear path fails and the
    # encrypted master is reachable (reported in logs, not downloaded).
    encrypted_heights: tuple = field(default=())


def is_cctv_episode_url(url: str) -> bool:
    """True for tv.cctv.com / tv.cctv.cn single-episode (VIDE) pages."""
    try:
        parts = urlparse(url)
    except ValueError:
        return False
    if (parts.hostname or '').lower() not in _CCTV_HOSTS:
        return False
    return bool(_EPISODE_PATH_RE.search(parts.path or '/'))


def ladder_for_quality(quality: str) -> tuple:
    """Ladder rungs to probe for a MeTube quality value, highest first.

    Unknown values yield () -- the caller then skips resolution entirely.
    """
    quality = (quality or '').strip().lower()
    start = _QUALITY_START_INDEX.get(quality)
    if start is None:
        return ()
    if quality == 'worst':
        return ('450',)
    return LADDER[start:]


def strip_maxbr(url: str) -> str:
    """Drop the CDN's maxbr bitrate-cap parameter (as yt-dlp's CCTVIE does)."""
    url = re.sub(r'maxbr=\d+&?', '', url)
    return re.sub(r'[?&]+$', '', url)


def clear_variant_url(hls_url: str, q: str):
    """Rewrite a clear-CDN hls_url to quality tier *q*, or None if the URL
    does not use the /asp/hls/<tier>/ layout (contentresolver.cpp
    clearVariantUrl: quality directory + tier-named playlist file).
    """
    result = _QUALITY_DIR_RE.sub(f'/asp/hls/{q}', hls_url, count=1)
    if result == hls_url:
        return None
    return _TIER_M3U8_RE.sub(f'/{q}.m3u8', result, count=1)


def _replace_main_segment(url: str, q: str):
    """Bare /main/ -> /<q>/ rewrite for URLs outside the /asp/hls/ layout
    (used for CCTV-4K channels; contentresolver.cpp 4K branch)."""
    if not _MAIN_SEGMENT_RE.search(url):
        return None
    return _MAIN_SEGMENT_RE.sub(q, url, count=1)


def parse_master_variants(text: str) -> list:
    """Parse #EXT-X-STREAM-INF variant entries (BANDWIDTH, RESOLUTION, URL)
    from a master playlist, in playlist order."""
    variants = []
    pending_bw = -1
    pending_res = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('#EXT-X-STREAM-INF'):
            bw = _BANDWIDTH_RE.search(line)
            res = _RESOLUTION_RE.search(line)
            pending_bw = int(bw.group(1)) if bw else -1
            pending_res = res.group(1) if res else None
        elif line and not line.startswith('#') and pending_bw >= 0:
            variants.append(Variant(pending_bw, pending_res or '', line))
            pending_bw = -1
            pending_res = None
    return variants


def select_variant_by_bandwidth(variants: list, max_bps):
    """Pick the highest-bandwidth variant at or below *max_bps*.

    max_bps None -> simply the largest. When every variant exceeds the
    ceiling, the smallest (least-exceeding) wins, mirroring
    contentresolver.cpp selectVariantIndex.
    """
    if not variants:
        return None
    if max_bps is None:
        return max(variants, key=lambda v: v.bandwidth)
    eligible = [v for v in variants if v.bandwidth <= max_bps]
    if eligible:
        return max(eligible, key=lambda v: v.bandwidth)
    return min(variants, key=lambda v: v.bandwidth)


def extract_guid(html: str):
    """Pull the 32-hex video guid out of an episode page's inline JS."""
    for pattern in _GUID_PATTERNS:
        m = re.search(pattern, html)
        if m:
            return m.group(1)
    return None


def encrypted_master_url(info: dict):
    """The encrypted-stream master URL (h5e preferred), host-normalized onto
    the DRM CDN. Reported only -- H5E TS segments need a private decryptor."""
    manifest = info.get('manifest') if isinstance(info.get('manifest'), dict) else {}
    for key in ('hls_h5e_url', 'hls_enc_url', 'hls_enc2_url'):
        u = manifest.get(key)
        if isinstance(u, str) and u:
            return _ENC_HOST_RE.sub(_ENC_HOST, u)
    return None


def _guid_from_entry(entry):
    if isinstance(entry, dict):
        entry_id = entry.get('id')
        if isinstance(entry_id, str) and _GUID_RE.match(entry_id):
            return entry_id
    return None


# --- resolution cache ------------------------------------------------------
# The API + tier probes are pure network reads of the same immutable episode,
# so a short TTL keeps repeated adds/retries free. Only successes are cached.
_CACHE = OrderedDict()  # key -> (monotonic_ts, CctvStream)
_CACHE_TTL = 600.0
_CACHE_MAX = 256


def _cache_get(key):
    hit = _CACHE.get(key)
    if hit is None:
        return None
    ts, stream = hit
    if time.monotonic() - ts > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return stream


def _cache_put(key, stream):
    _CACHE[key] = (time.monotonic(), stream)
    _CACHE.move_to_end(key)
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)


# --- network ---------------------------------------------------------------

async def _aiohttp_fetch_checked(session, url, allow_private):
    """GET *url* as text (capped), or None on any miss.

    Every outbound URL -- including the post-redirect final URL -- passes
    validate_url first: these hosts come from remote API JSON, and the main
    process runs no socket guard (see url_guard's module docstring).
    """
    loop = asyncio.get_running_loop()
    err = await loop.run_in_executor(None, partial(validate_url, url, allow_private=allow_private))
    if err is not None:
        log.warning('CCTV resolver skipping disallowed URL %s: %s', url, err)
        return None
    try:
        async with session.get(url) as resp:
            final = str(resp.url)
            final_err = await loop.run_in_executor(
                None, partial(validate_url, final, allow_private=allow_private))
            if final_err is not None:
                log.warning('CCTV resolver skipping redirect to disallowed URL %s', final)
                return None
            if resp.status != 200:
                return None
            body = await resp.content.read(_MAX_BODY_BYTES)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        return None
    return body.decode('utf-8', errors='replace')


async def _probe_encrypted_heights(session, fetch, enc_url):
    """Best-effort parse of the encrypted master's variant resolutions,
    for the 'clear path failed' log line. Never raises."""
    try:
        text = await fetch(session, enc_url)
        if not text:
            return ()
        return tuple(
            v.resolution.split('x')[-1] for v in parse_master_variants(text)
            if v.resolution)
    except Exception:
        return ()


async def _fetch_api_info(session, fetch, api_url):
    """GET the video-info API and parse it, retrying truncated responses.

    The API intermittently truncates its JSON mid-string (server-side
    Content-Length overstatement; observed to worsen under request bursts).
    ``hls_url`` sits near the end of the payload, so a truncated response
    is unusable -- try again rather than give up.
    """
    for attempt in range(_API_ATTEMPTS):
        text = await fetch(session, api_url)
        if text is not None:
            try:
                info = json.loads(text)
            except ValueError:
                log.debug('CCTV API returned unparseable JSON (attempt %d/%d)',
                          attempt + 1, _API_ATTEMPTS)
            else:
                if isinstance(info, dict):
                    return info
        if attempt < _API_ATTEMPTS - 1:
            await asyncio.sleep(_API_RETRY_DELAY)
    return None


async def _resolve_streams(session, fetch, info, ladder):
    hls_url = strip_maxbr((info.get('hls_url') or '').strip())
    if not hls_url:
        return None
    title = info.get('title')
    enc_url = encrypted_master_url(info)
    is_4k = 'cctv-4k' in str(info.get('play_channel') or '').lower()

    def stream(url, source, probed_quality, forced_format=FORCED_FORMAT):
        return CctvStream(
            url=f'generic:{url}', forced_format=forced_format, source=source,
            probed_quality=probed_quality, title=title, encrypted_master=enc_url)

    # 1) clear-CDN quality-directory ladder, highest requested tier down.
    #    Deliberately BEFORE consulting the master: episode masters are
    #    routinely incomplete (see module docstring), while unlisted tier
    #    directories keep serving playlists.
    for q in ladder:
        cand = clear_variant_url(hls_url, q)
        via_4k_rewrite = False
        if cand is None and q == '4000' and is_4k:
            cand = _replace_main_segment(hls_url, '4000')
            via_4k_rewrite = cand is not None
        if cand is None:
            continue
        source = '4k' if via_4k_rewrite else 'clear-ladder'
        text = await fetch(session, cand)
        if text is None or '#EXTM3U' not in text:
            continue
        variants = parse_master_variants(text)
        if variants:
            # a nested master: take its top variant (the tier is already
            # selected by the directory we asked for)
            chosen = select_variant_by_bandwidth(variants, None)
            return stream(urljoin(cand, chosen.url), source, q)
        if '#EXTINF' in text:
            return stream(cand, source, q)

    # 2) last resort: hand yt-dlp the master playlist itself. Its variants
    #    carry real BANDWIDTH/RESOLUTION metadata, so metube's usual format
    #    selectors sort correctly and no format is forced.
    text = await fetch(session, hls_url)
    if text is not None and '#EXT-X-STREAM-INF' in text:
        if parse_master_variants(text):
            return stream(hls_url, 'clear-master', None, forced_format=None)

    # 3) bare media playlist at main/ (the single tier the CDN admits to):
    #    no height metadata -> force the format selector
    if text is not None and '#EXTINF' in text:
        return stream(hls_url, 'clear-main', None)

    return None


async def resolve_episode(url, quality, *, entry=None, allow_private=False, _fetch=None):
    """Resolve a CCTV episode page to its best clear-CDN stream.

    Returns a CctvStream, or None when anything fails (the caller then
    proceeds exactly as without this module). Raises never.
    """
    try:
        if not is_cctv_episode_url(url):
            return None
        ladder = ladder_for_quality(quality)
        if not ladder:
            return None

        guid = _guid_from_entry(entry)
        cache_key = f'{guid or url}:{quality}'
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        if _fetch is not None:
            fetch = _fetch
        else:
            fetch = partial(_aiohttp_fetch_checked, allow_private=allow_private)

        async with aiohttp.ClientSession(timeout=_PROBE_TIMEOUT, headers={'User-Agent': UA}) as session:
            if guid is None:
                html = await fetch(session, url)
                if html is not None:
                    guid = extract_guid(html)
            if guid is None:
                return None

            api = f'{API_URL}?{urlencode({"pid": guid})}'
            info = await _fetch_api_info(session, fetch, api)
            if info is None:
                return None

            result = await _resolve_streams(session, fetch, info, ladder)

            if result is not None:
                _cache_put(cache_key, result)
            else:
                hint = ''
                enc_url = encrypted_master_url(info)
                if enc_url:
                    heights = await _probe_encrypted_heights(session, fetch, enc_url)
                    if heights:
                        hint = (f'; encrypted stream offers '
                                f'{", ".join(f"{h}p" for h in heights)}'
                                f' (H5E, not downloadable here)')
                    else:
                        hint = f'; encrypted stream at {enc_url} (H5E, not downloadable here)'
                log.info('CCTV: no clear stream found for %s%s', url, hint)
        return result
    except Exception:
        log.warning('CCTV resolve failed for %s', url, exc_info=True)
        return None

