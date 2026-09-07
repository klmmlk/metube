"""Tests for the local H5E decrypting proxy (app/cctv_h5e_proxy.py).

The proxy is exercised end-to-end against in-process aiohttp test servers:
a fake upstream serving an encrypted playlist/segments, and the proxy app
registered exactly like main.py registers it. The upstream is loopback, so
each test neutralises url_guard the same way test_cctv.py does (production
code must keep rejecting loopback upstreams -- one test re-arms the guard to
prove it).
"""

import pytest
from aiohttp import web

import cctv_h5e_proxy
from cctv_h5e_proxy import (
    decode_upstream,
    encode_upstream,
    probe,
    proxy_base,
    proxy_url,
    register_routes,
)
from test_cctv_h5e import (
    _encrypt_type5_nal,
    _parse_pes,
    _pat,
    _pes_packets,
    _pmt,
    _psi_packet,
    _type25_enable_nal,
)

PLAYLIST = (
    '#EXTM3U\n'
    '#EXT-X-TARGETDURATION:10\n'
    '#EXT-X-KEY:METHOD=AES-128,URI="https://key.example.com/k"\n'
    '#EXT-X-MAP:URI="init.ts"\n'
    '#EXTINF:10.0,\n'
    'seg1.ts\n'
    '#EXTINF:10.0,\n'
    'seg2.ts\n'
    '#EXT-X-ENDLIST\n'
)


@pytest.fixture(autouse=True)
async def _fresh_proxy_state(monkeypatch):
    """Loopback upstreams + a clean per-test loop for the shared client."""
    monkeypatch.setattr(cctv_h5e_proxy, 'validate_url',
                        lambda url, allow_private=False: None)
    cctv_h5e_proxy._sessions.clear()
    cctv_h5e_proxy._vpids.clear()
    cctv_h5e_proxy._client = None
    yield
    client = cctv_h5e_proxy._client
    cctv_h5e_proxy._client = None
    cctv_h5e_proxy._sessions.clear()
    cctv_h5e_proxy._vpids.clear()
    if client is not None and not client.closed:
        await client.close()


async def _make_proxy(aiohttp_client):
    app = web.Application()
    routes = web.RouteTableDef()
    register_routes(routes, '/')
    app.add_routes(routes)
    return await aiohttp_client(app)


async def _make_upstream(aiohttp_client, playlist=None, segments=None):
    app = web.Application()
    if playlist is not None:
        async def serve_playlist(request):
            return web.Response(text=playlist,
                                content_type='application/vnd.apple.mpegurl')
        app.router.add_get('/enc/variant.m3u8', serve_playlist)
    for name, body in (segments or {}).items():
        async def serve_segment(request, body=body):
            return web.Response(body=body, content_type='video/MP2T')
        app.router.add_get(f'/enc/{name}', serve_segment)
    return await aiohttp_client(app)


def _segment_ts(with_type25=True):
    """One encrypted TS segment: optional type25 'enable' NAL + a type5 NAL
    whose grid is TEA-encrypted. Returns (ts, expected_type5_rbsp)."""
    chunks = []
    if with_type25:
        chunks.append(bytes(_type25_enable_nal()))
    nal5, _stride, rbsp = _encrypt_type5_nal()
    chunks.append(bytes(nal5))
    return b''.join(_pes_packets(0x100, chunks)), rbsp


def _vpid_segment_ts():
    """Segment whose video rides a non-default PID announced by PAT/PMT."""
    nal5, _stride, rbsp = _encrypt_type5_nal()
    ts = (_psi_packet(0x0000, _pat(pmt_pid=0x0FFF))
          + _psi_packet(0x0FFF, _pmt(video_pid=0x233))
          + b''.join(_pes_packets(0x233,
                                  [bytes(_type25_enable_nal()), bytes(nal5)])))
    return ts, rbsp


def _proxy_prefix(proxy_client) -> str:
    return str(proxy_client.make_url('/cctv-h5e/u/'))


def _proxy_get(client, upstream_url: str):
    """GET the proxy path for *upstream_url*. aiohttp's test client refuses
    absolute URLs, so the request is issued by path (tests register the
    proxy under the '/' URL_PREFIX)."""
    return client.get('/cctv-h5e/u/' + encode_upstream(upstream_url))


def _assert_type5_decrypted(ts: bytes, rbsp: bytes, pusi_es=b'\x00\x00\x01\xe0'):
    pes = _parse_pes(ts, 0x100)
    assert pes[:3] == b'\x00\x00\x01'
    es = bytes(pes[9:])
    i = es.find(b'\x00\x00\x00\x01\x25')
    assert i >= 0, 'type5 NAL start code lost'
    assert es[i + 4:i + 4 + len(rbsp)] == rbsp, 'type5 NAL not decrypted'


# --- pure URL helpers ---------------------------------------------------------

def test_upstream_token_roundtrip():
    for url in ('https://drm.cntv.vod.dnsv1.com/asp/enc2/a/b.m3u8?x=1#f',
                'http://example.com/x',
                'https://example.com/%E4%B8%AD%E6%96%87/seg.ts'):
        assert decode_upstream(encode_upstream(url)) == url


def test_decode_upstream_rejects_junk():
    for token in ('', '!!!', 'AAAA', 'a', '//', 'AAAA=', encode_upstream('ftp://x/y')):
        assert decode_upstream(token) is None, token


def test_proxy_url_composition():
    assert proxy_url('http://h:1/cctv-h5e/u/', 'https://u/x.ts') == \
        'http://h:1/cctv-h5e/u/' + encode_upstream('https://u/x.ts')


def test_proxy_base_variants():
    assert proxy_base('*', '8081', '/') == 'http://127.0.0.1:8081/cctv-h5e/u/'
    assert proxy_base('', '8081', '/') == 'http://127.0.0.1:8081/cctv-h5e/u/'
    assert proxy_base('0.0.0.0', 8081, '/') == 'http://127.0.0.1:8081/cctv-h5e/u/'
    assert proxy_base('::', '8081', '/') == 'http://127.0.0.1:8081/cctv-h5e/u/'
    assert proxy_base('192.168.1.5', '8081', '/') == \
        'http://192.168.1.5:8081/cctv-h5e/u/'
    assert proxy_base('::1', '8081', '/metube/') == \
        'http://[::1]:8081/metube/cctv-h5e/u/'
    assert proxy_base('[::1]', '8081', '/') == 'http://[::1]:8081/cctv-h5e/u/'


# --- playlist rewriting --------------------------------------------------------

async def test_playlist_lines_rewritten_to_proxy_urls(aiohttp_client):
    upstream = await _make_upstream(aiohttp_client, playlist=PLAYLIST)
    proxy = await _make_proxy(aiohttp_client)
    resp = await _proxy_get(proxy, str(upstream.make_url('/enc/variant.m3u8')))
    assert resp.status == 200
    assert resp.content_type == 'application/vnd.apple.mpegurl'
    text = await resp.text()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # the AES key line is dropped: H5E is decrypted server-side
    assert not any(l.startswith('#EXT-X-KEY') for l in lines)
    segs = [l for l in lines if not l.startswith('#')]
    assert len(segs) == 2
    prefix = _proxy_prefix(proxy)
    for line, name in zip(segs, ('seg1.ts', 'seg2.ts')):
        assert line.startswith(prefix)
        assert decode_upstream(line[len(prefix):]) == \
            str(upstream.make_url(f'/enc/{name}'))
    # URI="..." attributes inside comment lines are rewritten too
    map_line = next(l for l in lines if l.startswith('#EXT-X-MAP'))
    token = map_line.split('URI="')[1].rstrip('"')[len(prefix):]
    assert decode_upstream(token) == str(upstream.make_url('/enc/init.ts'))


async def test_absolute_playlist_lines_resolve_against_upstream(aiohttp_client):
    playlist = '#EXTM3U\n#EXTINF:10.0,\nhttps://other.example.com/abs.ts\n'
    upstream = await _make_upstream(aiohttp_client, playlist=playlist)
    proxy = await _make_proxy(aiohttp_client)
    resp = await _proxy_get(proxy, str(upstream.make_url('/enc/variant.m3u8')))
    text = await resp.text()
    seg = next(l.strip() for l in text.splitlines()
               if l.strip() and not l.strip().startswith('#'))
    assert decode_upstream(seg[len(_proxy_prefix(proxy)):]) == \
        'https://other.example.com/abs.ts'


# --- segment decryption --------------------------------------------------------

async def test_segment_decrypted_and_packet_count_kept(aiohttp_client):
    ts, rbsp = _segment_ts()
    upstream = await _make_upstream(aiohttp_client, segments={'seg1.ts': ts})
    proxy = await _make_proxy(aiohttp_client)
    resp = await _proxy_get(proxy, str(upstream.make_url('/enc/seg1.ts')))
    assert resp.status == 200
    assert resp.content_type == 'video/mp2t'
    body = await resp.read()
    assert len(body) == len(ts)
    assert all(body[off] == 0x47 for off in range(0, len(body), 188))
    _assert_type5_decrypted(body, rbsp)


async def test_session_latches_new_mode_across_segments(aiohttp_client):
    # seg1 carries the type25 'enable' NAL; seg2 has none. Both share the
    # segment directory, so seg2 must decrypt through seg1's latched session.
    ts1, rbsp1 = _segment_ts(with_type25=True)
    ts2, rbsp2 = _segment_ts(with_type25=False)
    upstream = await _make_upstream(aiohttp_client,
                                    segments={'seg1.ts': ts1, 'seg2.ts': ts2})
    proxy = await _make_proxy(aiohttp_client)
    prefix = _proxy_prefix(proxy)
    for name, rbsp in (('seg1.ts', rbsp1), ('seg2.ts', rbsp2)):
        resp = await _proxy_get(proxy, str(upstream.make_url(f'/enc/{name}')))
        assert resp.status == 200
        _assert_type5_decrypted(await resp.read(), rbsp)


async def test_video_pid_detected_from_pmt(aiohttp_client):
    ts, rbsp = _vpid_segment_ts()
    upstream = await _make_upstream(aiohttp_client, segments={'seg1.ts': ts})
    proxy = await _make_proxy(aiohttp_client)
    resp = await _proxy_get(proxy, str(upstream.make_url('/enc/seg1.ts')))
    assert resp.status == 200
    body = await resp.read()
    # the video PES rides PID 0x233 (the default-guess 0x100 would decrypt
    # nothing); the plain type5 RBSP proves the PMT detection worked
    pes = _parse_pes(body, 0x233)
    es = bytes(pes[9:])
    i = es.find(b'\x00\x00\x00\x01\x25')
    assert i >= 0
    assert es[i + 4:i + 4 + len(rbsp)] == rbsp


async def test_non_ts_upstream_body_passes_through(aiohttp_client):
    blob = b'\x89PNG\r\n\x1a\n' + b'\x11' * 400      # not TS: no 0x47 sync
    upstream = await _make_upstream(aiohttp_client, segments={'seg1.ts': blob})
    proxy = await _make_proxy(aiohttp_client)
    resp = await _proxy_get(proxy, str(upstream.make_url('/enc/seg1.ts')))
    assert resp.status == 200
    assert await resp.read() == blob


# --- failure modes --------------------------------------------------------------

async def test_disallowed_upstream_is_403(aiohttp_client, monkeypatch):
    monkeypatch.setattr(cctv_h5e_proxy, 'validate_url',
                        lambda url, allow_private=False: 'blocked by test')
    proxy = await _make_proxy(aiohttp_client)
    resp = await _proxy_get(proxy, 'https://drm.cntv.vod.dnsv1.com/x.ts')
    assert resp.status == 403


async def test_bad_token_is_400(aiohttp_client):
    proxy = await _make_proxy(aiohttp_client)
    resp = await proxy.get('/cctv-h5e/u/%%%bad')
    assert resp.status == 400


async def test_upstream_miss_is_502(aiohttp_client):
    upstream = await _make_upstream(aiohttp_client)
    proxy = await _make_proxy(aiohttp_client)
    resp = await _proxy_get(proxy, str(upstream.make_url('/enc/missing.ts')))
    assert resp.status == 502


# --- resolver self-check --------------------------------------------------------

async def test_probe_happy_path(aiohttp_client):
    ts, _ = _segment_ts()
    upstream = await _make_upstream(aiohttp_client, playlist=PLAYLIST,
                                    segments={'seg1.ts': ts})
    proxy = await _make_proxy(aiohttp_client)
    assert await probe(_proxy_prefix(proxy),
                       str(upstream.make_url('/enc/variant.m3u8'))) is True


async def test_probe_failure_returns_false(aiohttp_client):
    upstream = await _make_upstream(aiohttp_client, playlist=PLAYLIST)
    proxy = await _make_proxy(aiohttp_client)
    base = _proxy_prefix(proxy)
    # playlist reachable but its first segment is missing upstream
    assert await probe(base, str(upstream.make_url('/enc/variant.m3u8'))) is False
    # playlist itself missing
    assert await probe(base, str(upstream.make_url('/enc/none.m3u8'))) is False
    # unreachable origin: never raises
    assert await probe('http://127.0.0.1:1/cctv-h5e/u/',
                       'https://drm.cntv.vod.dnsv1.com/x.m3u8') is False
