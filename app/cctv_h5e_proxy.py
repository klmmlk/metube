"""Local decrypting proxy for CCTV hls_h5e playlists and segments.

Why: the encrypted CDN (``manifest.hls_h5e_url``) holds the tiers above what
the clear CDN serves (720p/1080p/4K), but its TS segments use CCTV's
proprietary H5E NAL-level encryption (see cctv_h5e.py). Rather than patching
yt-dlp, the resolver hands it a URL on this very server: the proxy fetches
the upstream playlist/segment, decrypts segments in a worker thread and
serves plain TS. After the first hop, yt-dlp's traffic never leaves
localhost.

Route (registered by ``register_routes``, wired in main.py *before* the
catch-all static routes):

    GET {prefix}cctv-h5e/u/<urlsafe-b64(upstream absolute URL)>

Every non-comment playlist line (and every ``URI="..."`` attribute) is
rewritten to the same scheme, so nested playlists and segments stay on the
proxy. The upstream of every request passes ``url_guard.validate_url`` with
``allow_private=False`` both before the fetch and after redirects: the b64
payload comes from our own rewrite of remote manifests and must not turn the
server into an open relay.

Per-stream state: the H5E "new mode" latch is carried by an
``cctv_h5e.H5eSession`` cached per segment *directory* (the key that is
stable across a variant's segments), together with the PMT-detected video
PID (fallback 0x100, the reference worker's default).

Never-worse contract: a segment whose decryption raises yields 502 instead
of silently shipping ciphertext, so a broken decryptor fails the download
loudly rather than corrupting it.
"""

import asyncio
import base64
import logging
import os
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from urllib.parse import urljoin

import aiohttp
from aiohttp import web

from cctv_h5e import H5eSession, decrypt_ts_inplace, detect_video_pid
from url_guard import validate_url

log = logging.getLogger('cctv.h5e')

# Path suffix appended to the deployment's URL_PREFIX (see register_routes).
ROUTE = 'cctv-h5e/u/'
UA = 'Lavf/60.10.100'

_SEGMENT_TIMEOUT = aiohttp.ClientTimeout(total=60, connect=10)
_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=5)
_MAX_SEGMENT_BYTES = 256 * 1024 * 1024
_READ_CHUNK = 1 << 20

_URI_ATTR_RE = re.compile(r'URI="([^"]+)"')

# CPU-bound NAL transform (~100-300 ms per 2 MB segment): a small dedicated
# pool keeps decryption off the event loop without starving it.
_decrypt_pool = ThreadPoolExecutor(
    max_workers=max(2, min(4, os.cpu_count() or 2)),
    thread_name_prefix='h5e-decrypt')

# Per segment-directory stream state (event-loop thread only; decrypt itself
# runs in the pool but H5eSession is touched only via on_nal under the
# segment-at-a-time access pattern of an HLS download).
_sessions: 'OrderedDict[str, H5eSession]' = OrderedDict()
_vpids: dict = {}
_SESSIONS_MAX = 128

_client: Optional[aiohttp.ClientSession] = None


def _get_client() -> aiohttp.ClientSession:
    """Shared upstream client (one per web-process loop, closed on shutdown);
    sharing keeps the TLS connection alive across a variant's segments."""
    global _client
    if _client is None or _client.closed:
        _client = aiohttp.ClientSession(headers={'User-Agent': UA})
    return _client


async def on_shutdown(app):
    global _client
    if _client is not None and not _client.closed:
        await _client.close()
    _client = None


# --- URL encoding -----------------------------------------------------------

def encode_upstream(url: str) -> str:
    """Opaque, URL-safe path token for an upstream URL (padding stripped)."""
    return base64.urlsafe_b64encode(url.encode('utf-8')).decode('ascii').rstrip('=')


def decode_upstream(token: str) -> Optional[str]:
    """Inverse of encode_upstream; None unless it decodes to an http(s) URL."""
    try:
        pad = '=' * (-len(token) % 4)
        url = base64.urlsafe_b64decode(token + pad).decode('utf-8')
    except (ValueError, UnicodeDecodeError):
        return None
    return url if url.startswith(('http://', 'https://')) else None


def proxy_url(prefix: str, upstream: str) -> str:
    """Proxy URL for *upstream* under *prefix* (an origin + URL_PREFIX +
    ROUTE string, as returned by proxy_prefix / proxy_base)."""
    return f'{prefix}{encode_upstream(upstream)}'


def proxy_base(host: str, port, url_prefix: str) -> str:
    """The prefix the *download side* uses to reach this server's proxy.

    Derived from the bound HOST/PORT rather than guessed at 127.0.0.1: when
    metube binds a single external address, that address (and not loopback)
    is the one the socket actually listens on. '*' binds are loopback-safe.
    """
    host = (host or '').strip() or '*'
    if host in ('*', '0.0.0.0', '::'):
        host = '127.0.0.1'
    elif ':' in host and not host.startswith('['):
        host = f'[{host}]'  # bare IPv6 literal needs brackets in a netloc
    return f'http://{host}:{port}{url_prefix}{ROUTE}'


def proxy_prefix(request: web.Request) -> str:
    """Prefix of the request's own proxy route, origin included, so rewritten
    lines are absolute URLs that work regardless of URL_PREFIX deployment."""
    marker = '/' + ROUTE
    path = request.path
    i = path.rfind(marker)
    prefix_path = path[:i + len(marker)] if i >= 0 else marker
    return f'{request.scheme}://{request.host}{prefix_path}'


# --- stream state -----------------------------------------------------------

def _get_session(key: str) -> H5eSession:
    session = _sessions.get(key)
    if session is None:
        session = H5eSession()
        _sessions[key] = session
    else:
        _sessions.move_to_end(key)
    while len(_sessions) > _SESSIONS_MAX:
        _sessions.popitem(last=False)
    return session


def _decrypt_segment(body: bytes, session: H5eSession, vpid: int):
    """Pool job: decrypt one TS segment (returns new bytes + NAL count)."""
    data = bytearray(body)
    count = decrypt_ts_inplace(data, session, vpid)
    return bytes(data), count


# --- upstream I/O -----------------------------------------------------------

async def _read_capped(resp: aiohttp.ClientResponse, cap: int) -> bytes:
    chunks = []
    received = 0
    while received < cap:
        chunk = await resp.content.read(min(_READ_CHUNK, cap - received))
        if not chunk:
            break
        chunks.append(chunk)
        received += len(chunk)
    return b''.join(chunks)


async def _fetch_upstream(upstream: str, cap: int, timeout):
    """GET *upstream* with SSRF re-validation of the post-redirect URL.
    Returns (final_url, body) or raises the usual network errors."""
    async with _get_client().get(upstream, timeout=timeout) as resp:
        final = str(resp.url)
        err = validate_url(final, allow_private=False)
        if err is not None:
            raise PermissionError(f'redirect to disallowed URL: {err}')
        if resp.status != 200:
            raise aiohttp.ClientResponseError(
                resp.request_info, resp.history, status=resp.status,
                message=f'upstream status {resp.status}')
        body = await _read_capped(resp, cap)
    return final, body


# --- responses ----------------------------------------------------------------

def _rewrite_playlist(request: web.Request, base: str, body: bytes) -> web.Response:
    prefix = proxy_prefix(request)
    lines = []
    for line in body.decode('utf-8', errors='replace').splitlines():
        stripped = line.strip()
        if stripped.startswith('#EXT-X-KEY'):
            # H5E is decrypted server-side; a leftover AES-128 key line would
            # make yt-dlp decrypt a second time and corrupt the output.
            continue
        if stripped.startswith('#'):
            if 'URI="' in line:
                line = _URI_ATTR_RE.sub(
                    lambda m: f'URI="{proxy_url(prefix, urljoin(base, m.group(1)))}"',
                    line)
        elif stripped:
            line = proxy_url(prefix, urljoin(base, stripped))
        lines.append(line)
    return web.Response(text='\n'.join(lines) + '\n',
                        content_type='application/vnd.apple.mpegurl')


async def _serve_segment(upstream: str, body: bytes) -> web.Response:
    if not body or body[0] != 0x47 or len(body) % 188:
        # Not an MPEG-TS segment (short read, key, init data): H5E only
        # transforms TS, so anything else passes through untouched.
        return web.Response(body=body, content_type='application/octet-stream')
    key = upstream.rsplit('/', 1)[0]
    session = _get_session(key)
    vpid = _vpids.get(key)
    if vpid is None:
        vpid = detect_video_pid(body) or 0x100
        _vpids[key] = vpid
    loop = asyncio.get_running_loop()
    try:
        data, count = await loop.run_in_executor(
            _decrypt_pool, _decrypt_segment, body, session, vpid)
    except Exception:
        log.warning('H5E decrypt failed for %s', upstream, exc_info=True)
        return web.Response(status=502, text='decrypt failed')
    if count == 0:
        log.debug('H5E: no video NAL on PID %#x in %s (served undecrypted)',
                  vpid, upstream)
    return web.Response(body=data, content_type='video/MP2T')


async def handle(request: web.Request) -> web.Response:
    upstream = decode_upstream(request.match_info['b64'])
    if upstream is None:
        return web.Response(status=400, text='bad upstream token')
    err = validate_url(upstream, allow_private=False)
    if err is not None:
        log.warning('H5E proxy refusing disallowed URL %s: %s', upstream, err)
        return web.Response(status=403, text='upstream not allowed')
    try:
        final, body = await _fetch_upstream(upstream, _MAX_SEGMENT_BYTES,
                                            _SEGMENT_TIMEOUT)
        if body.startswith(b'#EXTM3U'):
            return _rewrite_playlist(request, final, body)
        return await _serve_segment(final, body)
    except PermissionError as e:
        return web.Response(status=403, text=str(e))
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
        return web.Response(status=502, text=f'upstream fetch failed: {e}')


def register_routes(routes: web.RouteTableDef, url_prefix: str) -> None:
    """Register the proxy route on *routes*. Must run before any catch-all
    static route is added (aiohttp resolves in registration order)."""
    routes.get(url_prefix + ROUTE + '{b64}')(handle)


# --- resolver self-check ------------------------------------------------------

async def probe(base: str, media_url: str) -> bool:
    """End-to-end check that this server can decrypt-serve *media_url*.

    Fetches the proxied playlist (must be an m3u8 with at least one
    rewritten absolute line), then that first entry (must come back as
    MPEG-TS). The URLs are self-constructed -- loopback origin plus the b64
    of an upstream the caller already validated -- so no url_guard check
    applies here; the proxy handler validates the upstream itself.

    Returns False on any failure; never raises.
    """
    try:
        async with aiohttp.ClientSession(
                timeout=_UPSTREAM_TIMEOUT,
                headers={'User-Agent': UA}) as session:
            async with session.get(proxy_url(base, media_url)) as resp:
                if resp.status != 200:
                    return False
                text = await resp.text()
            if '#EXTM3U' not in text:
                return False
            entry = next((l.strip() for l in text.splitlines()
                          if l.strip() and not l.strip().startswith('#')), None)
            if not entry or not entry.startswith(('http://', 'https://')):
                return False
            async with session.get(entry) as resp:
                if resp.status != 200:
                    return False
                body = await resp.read()
        return (bool(body) and body[0] == 0x47 and len(body) % 188 == 0)
    except Exception:
        log.debug('H5E proxy probe failed', exc_info=True)
        return False
