"""Unit tests for the CCTV quality resolver (app/cctv.py).

Pure-function cases need no network; orchestration cases inject a fake
``_fetch`` so resolve_episode's full decision tree runs offline.
"""

import asyncio
import json
from urllib.parse import urljoin

import aiohttp
import pytest

import cctv
from cctv import (
    FORCED_FORMAT,
    Variant,
    clear_variant_url,
    encrypted_master_url,
    extract_guid,
    is_cctv_episode_url,
    ladder_for_quality,
    maybe_rewrite_vida_landing,
    parse_master_variants,
    select_variant_by_bandwidth,
    strip_maxbr,
)

GUID = 'a' * 32
PAGE = f'https://tv.cctv.com/2024/02/21/VIDE{GUID[:16].upper()}.shtml'
VIDA_PAGE = 'https://tv.cctv.cn/2026/09/01/VIDAxOhtc2E2Nk3KrBRhYSbY260901.shtml'
VIDA_SIBLING = 'https://tv.cctv.cn/2026/08/27/VIDE2HwSnTrN1pK5ZHSgz1Z5260827.shtml'
HLS = 'https://dh5.cntv.myhwcdn.cn/asp/hls/main/MAIN123/abc/main.m3u8'

MEDIA = (
    '#EXTM3U\n'
    '#EXT-X-TARGETDURATION:10\n'
    '#EXTINF:10.0,\n'
    'seg001.ts\n'
    '#EXT-X-ENDLIST\n'
)

MASTER = (
    '#EXTM3U\n'
    '#EXT-X-STREAM-INF:BANDWIDTH=2048000,RESOLUTION=1920x1080\n'
    '2000/index.m3u8\n'
    '#EXT-X-STREAM-INF:BANDWIDTH=1228800,RESOLUTION=1280x720\n'
    '1200/index.m3u8\n'
)


@pytest.fixture(autouse=True)
def _fast_cache_and_retries(monkeypatch):
    cctv._CACHE.clear()
    monkeypatch.setattr(cctv, '_API_RETRY_DELAY', 0)
    yield
    cctv._CACHE.clear()


def api_body(hls_url=HLS, **extra):
    payload = {'hls_url': hls_url, 'title': '测试节目', 'play_channel': 'CCTV-1 综合'}
    payload.update(extra)
    return json.dumps(payload)


class FakeFetch:
    """Injectable _fetch stand-in: url -> text (None/missing = miss).

    A list value serves its items in order (then misses), modelling e.g.
    an API that truncates its JSON once and then succeeds.
    """

    def __init__(self, responses):
        self.responses = {k: (list(v) if isinstance(v, (list, tuple)) else v)
                          for k, v in responses.items()}
        self.requests = []

    async def __call__(self, session, url):
        self.requests.append(url)
        resp = self.responses.get(url)
        if isinstance(resp, list):
            return resp.pop(0) if resp else None
        return resp


def variant_url(q, name=None):
    return f'https://dh5.cntv.myhwcdn.cn/asp/hls/{q}/MAIN123/abc/{name or q + ".m3u8"}'


# --- pure functions ---------------------------------------------------------

@pytest.mark.parametrize('url,expected', [
    (f'{HLS}?maxbr=1200', HLS),
    (f'{HLS}?maxbr=1200&x=1', f'{HLS}?x=1'),
    (f'{HLS}?x=1&maxbr=850', f'{HLS}?x=1'),
    (f'{HLS}?x=1&maxbr=850&', f'{HLS}?x=1'),
    (HLS, HLS),
])
def test_strip_maxbr(url, expected):
    assert strip_maxbr(url) == expected


def test_clear_variant_url_rewrites_tier():
    assert clear_variant_url(HLS, '2000') == variant_url('2000')


def test_clear_variant_url_rewrites_already_tiered_url():
    assert clear_variant_url(variant_url('850'), '2000') == variant_url('2000')


def test_clear_variant_url_keeps_query():
    url = HLS + '?x=1'
    assert clear_variant_url(url, '1200') == variant_url('1200') + '?x=1'


def test_clear_variant_url_non_standard_layout_returns_none():
    assert clear_variant_url('https://x.example.com/video/main/abc/playlist.m3u8', '2000') is None


def test_ladder_for_quality_table():
    assert ladder_for_quality('best') == cctv.LADDER
    assert ladder_for_quality('2160') == cctv.LADDER
    assert ladder_for_quality('1440') == ('3000', '2000', '1200', '850', '450')
    assert ladder_for_quality('1080') == ('2000', '1200', '850', '450')
    assert ladder_for_quality('720') == ('1200', '850', '450')
    assert ladder_for_quality('480') == ('850', '450')
    assert ladder_for_quality('360') == ('450',)
    assert ladder_for_quality('240') == ('450',)
    assert ladder_for_quality('worst') == ('450',)
    assert ladder_for_quality('bogus') == ()


@pytest.mark.parametrize('url,expected', [
    (PAGE, True),
    ('https://tv.cctv.cn/2024/02/21/VIDEAbCdEf123.shtml', True),
    ('https://www.tv.cctv.com/2024/02/21/VIDEAbCdEf123.shtml', True),
    (PAGE + '?spm=noop', True),
    # VIDA (album) prefix is also a recognised episode URL -- some CCTV
    # single-episode pages (notably 动画 series) use VIDA in the path even
    # though cctv_series treats the ID as an album alias.
    ('https://tv.cctv.cn/2026/09/01/VIDAxOhtc2E2Nk3KrBRhYSbY260901.shtml', True),
    ('https://sports.cctv.com/2024/02/21/ARTIAbCdEf123.shtml', False),
    ('https://tv.cctv.com/lm/xwlb/videoset/', False),
    ('https://www.youtube.com/watch?v=abc', False),
    ('https://vdn.apps.cntv.cn/api/getHttpVideoInfo.do?pid=x', False),
    ('not a url', False),
])
def test_is_cctv_episode_url(url, expected):
    assert is_cctv_episode_url(url) is expected


def test_parse_master_variants():
    variants = parse_master_variants(MASTER.replace('\n', '\r\n'))
    assert variants == [
        Variant(2048000, '1920x1080', '2000/index.m3u8'),
        Variant(1228800, '1280x720', '1200/index.m3u8'),
    ]


def test_parse_master_variants_without_resolution():
    text = '#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=460800\n450/prog.m3u8\n'
    assert parse_master_variants(text) == [Variant(460800, '', '450/prog.m3u8')]


def test_parse_master_variants_rejects_non_master():
    assert parse_master_variants(MEDIA) == []
    assert parse_master_variants('') == []


def test_select_variant_unlimited_takes_max():
    variants = [Variant(1000, '', 'a'), Variant(3000, '', 'b')]
    assert select_variant_by_bandwidth(variants, None).url == 'b'


def test_select_variant_under_ceiling():
    variants = [Variant(4000000, '', '4k'), Variant(2048000, '', '1080'), Variant(1228800, '', '720')]
    assert select_variant_by_bandwidth(variants, 2048000).url == '1080'
    assert select_variant_by_bandwidth(variants, 870_400).url == '720'


def test_select_variant_all_exceeding_takes_smallest():
    variants = [Variant(4000000, '', '4k'), Variant(2048000, '', '1080')]
    assert select_variant_by_bandwidth(variants, 460800).url == '1080'


@pytest.mark.parametrize('html', [
    f'var guid = "{GUID}";',
    f'videoCenterId: "{GUID}"',
    f'videoCenterId", "{GUID}"',
    f"changePlayer('{GUID}')",
    f"loadVideo('{GUID}')",
    f"loadvideo('{GUID}')",
    f"var initMyAray = '{GUID}';",
    f'var ids = ["{GUID}"]',
])
def test_extract_guid_shapes(html):
    assert extract_guid(f'<html><script>{html}</script></html>') == GUID


def test_extract_guid_missing():
    assert extract_guid('<html>no guid here</html>') is None


def test_encrypted_master_url_priority_and_host_normalization():
    info = {'manifest': {
        'hls_enc2_url': 'https://x2.cntv.cn/asp/enc2/b.m3u8',
        'hls_enc_url': 'https://x1.cntv.cn/asp/enc2/a.m3u8',
        'hls_h5e_url': 'https://random.cdn.cntv.cn/asp/enc2/h5e.m3u8',
    }}
    assert encrypted_master_url(info) == 'https://drm.cntv.vod.dnsv1.com/asp/enc2/h5e.m3u8'


def test_encrypted_master_url_fallback_and_absence():
    info = {'manifest': {'hls_enc_url': 'https://x1.cntv.cn/asp/enc2/a.m3u8'}}
    assert encrypted_master_url(info) == 'https://drm.cntv.vod.dnsv1.com/asp/enc2/a.m3u8'
    assert encrypted_master_url({'manifest': {}}) is None
    assert encrypted_master_url({}) is None


# --- orchestration (injected fake fetch) ------------------------------------

async def test_resolve_uses_entry_guid_without_page_fetch():
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(hls_url=variant_url('2000')),
        variant_url('4000'): None,
        variant_url('3000'): None,
        variant_url('2000'): MEDIA,
    })
    stream = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch)
    assert PAGE not in fetch.requests
    assert stream.url == f'{variant_url("2000")}'


async def test_resolve_ladder_probes_from_top():
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        variant_url('4000'): None,
        variant_url('3000'): None,
        variant_url('2000'): MEDIA,      # highest available tier
    })
    stream = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch)
    assert stream.source == 'clear-ladder'
    assert stream.probed_quality == '2000'
    assert stream.url == f'{variant_url("2000")}'
    assert stream.forced_format == FORCED_FORMAT
    assert stream.title == '测试节目'
    # tiers below the hit are never probed, nor is the master consulted
    assert variant_url('1200') not in fetch.requests
    assert HLS not in fetch.requests


async def test_ladder_beats_incomplete_master():
    # live 2026-09 behaviour: a 新闻联播 master listed only the 450 tier
    # while its 2000 directory served a playlist -- the ladder must win or
    # 'best' is locked to 270p
    low_master = ('#EXTM3U\n'
                  '#EXT-X-STREAM-INF:BANDWIDTH=460800,RESOLUTION=480x270\n'
                  '450/450.m3u8\n')
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        HLS: low_master,
        variant_url('4000'): None,
        variant_url('3000'): None,
        variant_url('2000'): MEDIA,
    })
    stream = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch)
    assert stream.source == 'clear-ladder'
    assert stream.probed_quality == '2000'
    assert stream.url == f'{variant_url("2000")}'


async def test_resolve_master_fallback_when_ladder_misses():
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        **{variant_url(q): None for q in cctv.LADDER},
        HLS: MASTER,
    })
    stream = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch)
    assert stream.source == 'clear-master'
    # the master itself: variants carry real metadata, yt-dlp sorts them,
    # so no format is forced
    assert stream.url == f'{HLS}'
    assert stream.forced_format is None
    assert stream.probed_quality is None


async def test_resolve_bare_main_media_playlist_when_ladder_misses():
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        variant_url('450'): None,
        HLS: MEDIA,
    })
    stream = await cctv.resolve_episode(PAGE, 'worst', entry={'id': GUID}, _fetch=fetch)
    assert stream.source == 'clear-main'
    assert stream.url == f'{HLS}'
    assert stream.forced_format == FORCED_FORMAT


async def test_resolve_ladder_nested_master_takes_top_variant():
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        HLS: None,
        variant_url('2000'): MASTER,
    })
    stream = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch)
    assert stream.probed_quality == '2000'
    assert stream.url == f'{urljoin(variant_url("2000"), "2000/index.m3u8")}'


async def test_resolve_4k_channel_uses_main_segment_rewrite():
    odd_hls = 'https://dh5.cntv.myhwcdn.cn/video/main/ABC123/main.m3u8'
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(hls_url=odd_hls, play_channel='CCTV-4K'),
        odd_hls: None,
        'https://dh5.cntv.myhwcdn.cn/video/4000/ABC123/main.m3u8': MEDIA,
    })
    stream = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch)
    assert stream.source == '4k'
    assert stream.url == 'https://dh5.cntv.myhwcdn.cn/video/4000/ABC123/main.m3u8'


async def test_resolve_worst_only_probes_lowest_tier():
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        HLS: MEDIA,
        variant_url('450'): MEDIA,
    })
    stream = await cctv.resolve_episode(PAGE, 'worst', entry={'id': GUID}, _fetch=fetch)
    assert stream.probed_quality == '450'
    for q in ('4000', '3000', '2000', '1200', '850'):
        assert variant_url(q) not in fetch.requests


async def test_resolve_non_cctv_or_unknown_quality_short_circuits():
    fetch = FakeFetch({})
    assert await cctv.resolve_episode('https://youtu.be/x', 'best', _fetch=fetch) is None
    assert await cctv.resolve_episode(PAGE, 'bogus', _fetch=fetch) is None
    assert fetch.requests == []


async def test_resolve_retries_truncated_api_json():
    # the API intermittently truncates its JSON; one bad response then a
    # good one must still resolve
    api = f'{cctv.API_URL}?pid={GUID}'
    fetch = FakeFetch({
        api: ['{"hls_url": "https://x', api_body()],   # first truncated
        variant_url('4000'): None,
        variant_url('3000'): None,
        variant_url('2000'): MEDIA,
    })
    stream = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch)
    assert stream is not None
    assert fetch.requests.count(api) == 2


async def test_resolve_page_guid_fallback():
    fetch = FakeFetch({
        PAGE: f'<script>var guid = "{GUID}";</script>',
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        HLS: MEDIA,
        variant_url('2000'): MEDIA,
    })
    stream = await cctv.resolve_episode(PAGE, '1080', _fetch=fetch)
    assert stream is not None
    assert stream.probed_quality == '2000'


# --- VIDA landing-page rewrite ---------------------------------------------

VIDA_LANDING_HTML = (
    "<script>var column_id = \"TOPC1460958044779267\";</script>\n"
    "var jsonData=[];\n"
    "var jsonData2=[{\n"
    "    'title':'第23集',\n"
    "    'img':'//p5.img.cntv.cn/fmspic/2026/08/27/abc-1.jpg',\n"
    "    'brief':'简介',\n"
    f"    'url':'{VIDA_SIBLING}'\n"
    "},{\n"
    "    'title':'第24集',\n"
    "    'url':'https://tv.cctv.cn/2026/08/27/VIDEother123.shtml'\n"
    "}];\n"
)


async def test_vida_landing_rewrites_to_first_vide_sibling():
    fetch = FakeFetch({VIDA_PAGE: VIDA_LANDING_HTML})
    assert await maybe_rewrite_vida_landing(VIDA_PAGE, _fetch=fetch) == VIDA_SIBLING


async def test_vida_landing_ignores_query_string():
    fetch = FakeFetch({VIDA_PAGE + '?spm=from.x.y': VIDA_LANDING_HTML})
    assert await maybe_rewrite_vida_landing(
        VIDA_PAGE + '?spm=from.x.y', _fetch=fetch) == VIDA_SIBLING


@pytest.mark.parametrize('url', [
    PAGE,                                # VIDE page: nothing to rewrite
    'https://tv.cctv.com/lm/xwlb/videoset/',   # series page
    'https://www.youtube.com/watch?v=abc',
    'not a url',
])
async def test_vida_landing_non_vida_urls_return_none_without_fetch(url):
    fetch = FakeFetch({})
    assert await maybe_rewrite_vida_landing(url, _fetch=fetch) is None
    assert fetch.requests == []


@pytest.mark.parametrize('html', [
    '',                                  # empty body
    '<html>no jsonData here</html>',     # no sibling list
    # jsonData2 exists but holds no VIDE URL (only non-episode links)
    "var jsonData2=[{'title':'x','url':'https://tv.cctv.cn/lm/foo/'}];",
])
async def test_vida_landing_fetch_or_regex_miss_returns_none(html):
    fetch = FakeFetch({VIDA_PAGE: html or None})
    assert await maybe_rewrite_vida_landing(VIDA_PAGE, _fetch=fetch) is None


async def test_vida_landing_takes_first_sibling_not_later_entries():
    # The regex must anchor on jsonData2's opening [{ and not skip ahead to
    # a VIDE URL in a later entry or elsewhere on the page.
    html = (
        "var jsonData2=[{\n"
        "    'title':'第1集',\n"
        "    'url':'https://tv.cctv.cn/2026/08/27/VIDEfirstabc.shtml'\n"
        "},{\n"
        "    'title':'第2集',\n"
        "    'url':'https://tv.cctv.cn/2026/08/27/VIDEsecondabc.shtml'\n"
        "}];"
    )
    fetch = FakeFetch({VIDA_PAGE: html})
    assert await maybe_rewrite_vida_landing(
        VIDA_PAGE, _fetch=fetch) == 'https://tv.cctv.cn/2026/08/27/VIDEfirstabc.shtml'


@pytest.mark.parametrize('responses', [
    {},                                                   # API unreachable
    {f'{cctv.API_URL}?pid={GUID}': 'not json'},          # bad JSON
    {f'{cctv.API_URL}?pid={GUID}': '["list"]'},          # non-dict JSON
    {f'{cctv.API_URL}?pid={GUID}': json.dumps({'title': 'x'})},  # no hls_url
])
async def test_resolve_failures_return_none(responses):
    fetch = FakeFetch(responses)
    assert await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch) is None


async def test_resolve_all_tiers_missing_returns_none():
    fetch = FakeFetch({
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        HLS: None,
        # every tier probe misses; encrypted master reported in logs only
        **{variant_url(q): None for q in cctv.LADDER},
    })
    assert await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=fetch) is None


async def test_resolve_cached_second_call_makes_no_requests():
    responses = {
        f'{cctv.API_URL}?pid={GUID}': api_body(),
        HLS: MEDIA,
        variant_url('2000'): MEDIA,
    }
    first = FakeFetch(responses)
    stream1 = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=first)
    second = FakeFetch(responses)
    stream2 = await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=second)
    assert stream1 == stream2
    assert second.requests == []


async def test_fetch_rejects_disallowed_url(monkeypatch):
    monkeypatch.setattr(cctv, 'validate_url', lambda url, allow_private=False: 'blocked by test')
    async with aiohttp.ClientSession() as session:
        result = await cctv._aiohttp_fetch_checked(session, 'https://example.com/x', allow_private=False)
    assert result is None


async def test_resolve_never_raises_on_fetch_exception():
    async def exploding_fetch(session, url):
        raise RuntimeError('boom')
    assert await cctv.resolve_episode(PAGE, 'best', entry={'id': GUID}, _fetch=exploding_fetch) is None


def test_resolve_timeout_is_bounded():
    # RESOLVE_TIMEOUT backs the caller-side asyncio.wait_for; keep it sane.
    assert 5.0 <= cctv.RESOLVE_TIMEOUT <= 30.0
