"""Unit tests for the CCTV whole-series resolver (app/cctv_series.py).

Pure-function cases need no network; orchestration cases inject a fake
``_fetch`` so fetch_series_episodes's full 4-level decision tree runs
offline.
"""

import asyncio
import json

import pytest

import cctv_series
from cctv_series import (
    SeriesKind,
    SeriesResult,
    build_album_info_url,
    build_album_url,
    build_column_url,
    classify_series,
    extract_column_id,
    extract_item_id,
    extract_pub_date_yyyymm,
    fetch_series_episodes,
    month_range,
    parse_album_info,
    parse_list_response,
)


PAGE = 'https://tv.cctv.cn/2026/09/01/VIDAxOhtc2E2Nk3KrBRhYSbY260901.shtml'
NON_CCTV = 'https://www.youtube.com/watch?v=x'


@pytest.fixture(autouse=True)
def _fast_cache(monkeypatch):
    cctv_series._CACHE.clear()
    monkeypatch.setattr(cctv_series, '_API_ATTEMPTS', 3)
    # Pin the month range so URL mocks are deterministic regardless of
    # when the test runs. Tests that need a different range can override
    # these in their own setup; here we make sure today and the 5-year
    # anchor are different so any cross-month logic gets exercised.
    monkeypatch.setattr(cctv_series, '_today_yyyymm', lambda: '202403')
    monkeypatch.setattr(cctv_series, '_five_years_ago_yyyymm', lambda: '202301')
    # backoff 0/0/0: no sleeping in tests
    async def zero_sleep(*_a, **_k):
        pass
    monkeypatch.setattr(asyncio, 'sleep', zero_sleep)
    yield
    cctv_series._CACHE.clear()


# FakeFetch supports list values served in order to model a truncating API.
# dict values are JSON-encoded so the production path's json.loads can run
# against them (and the truncated-first-call case is still representable).
class FakeFetch:
    def __init__(self, responses):
        norm = {}
        for k, v in responses.items():
            if isinstance(v, (list, tuple)):
                norm[k] = [self._encode(x) for x in v]
            else:
                norm[k] = self._encode(v)
        self.responses = norm
        self.requests = []

    @staticmethod
    def _encode(v):
        if v is None or isinstance(v, str):
            return v
        return json.dumps(v)

    async def __call__(self, session, url):
        self.requests.append(url)
        resp = self.responses.get(url)
        if isinstance(resp, list):
            return resp.pop(0) if resp else None
        return resp


def episode(guid, url, title='t'):
    return {'guid': guid, 'url': url, 'title': title, 'time': '2026-09-01'}


def episode_html(column_id=None, item_id='VIDExOhtc2E2Nk3KrBRhYSbY260901',
                pub_date='2024-03-01'):
    parts = []
    if column_id is not None:
        parts.append(f'var column_id = "{column_id}";')
    parts.append(f'var itemid1 = "{item_id}";')
    if pub_date:
        parts.append(f'<meta name="pubdate" content="{pub_date}">')
    return '<html><head><script>' + ' '.join(parts) + '</script></head></html>'


# --- pure functions --------------------------------------------------------

@pytest.mark.parametrize('html,expected', [
    (f'var column_id = "TOPC12345";', 'TOPC12345'),
    (f"var column_id='TOPC99_x';", 'TOPC99_x'),
    (f"VAR Column_ID = \"TOPCmixed\";", 'TOPCmixed'),
    (f'var column_id="TOPC.with.dots";', 'TOPC.with.dots'),
    (f'var itemid1 = "VIDEabc";', None),  # item id, not column id
    ('', None),
])
def test_extract_column_id(html, expected):
    assert extract_column_id(html) == expected


@pytest.mark.parametrize('html,expected', [
    (f'var itemid1 = "VIDExOhtc2E2Nk3KrBRhYSbY260901";', 'VIDExOhtc2E2Nk3KrBRhYSbY260901'),
    (f"var item_id = 'VIDAx2';", 'VIDAx2'),
    (f'var itemid1 = "VIDAabc";', 'VIDAabc'),  # album id also accepted
    ('', None),
])
def test_extract_item_id(html, expected):
    assert extract_item_id(html) == expected


@pytest.mark.parametrize('html,expected', [
    ('<meta name="pubdate" content="2026-09-01">', '202609'),
    ('发布于2024-01-31 21:00', '202401'),
    ('no date here', None),
    ('', None),
])
def test_extract_pub_date(html, expected):
    assert extract_pub_date_yyyymm(html) == expected


def test_build_column_url_key_fields():
    url = build_column_url('TOPC1', '202401', page=2, page_size=50)
    assert 'id=TOPC1' in url
    assert 'd=202401' in url
    assert 'p=2' in url
    assert 'n=50' in url
    assert 'mode=0' in url
    assert 'serviceId=tvcctv' in url
    assert 'sort=desc' in url
    assert url.startswith('https://api.cntv.cn/NewVideo/getVideoListByColumn')


def test_build_album_url_key_fields():
    url = build_album_url('VIDAabc', page=1, page_size=50)
    assert 'id=VIDAabc' in url
    assert 'p=1' in url
    assert 'n=50' in url
    assert 'mode=0' in url
    assert 'pub=1' in url
    assert 'serviceId=tvcctv' in url
    assert 'sort=asc' in url
    assert url.startswith('https://api.cntv.cn/NewVideo/getVideoListByAlbumIdNew')


def test_build_album_info_url():
    url = build_album_info_url('VIDExOhtc2E2Nk3KrBRhYSbY260901')
    assert 'id=VIDExOhtc2E2Nk3KrBRhYSbY260901' in url
    assert 'serviceId=tvcctv' in url
    assert url.startswith('https://api.cntv.cn/NewVideoset/getVideoAlbumInfoByVideoId')


@pytest.mark.parametrize('payload,expected_count', [
    ({'data': {'total': 5, 'list': [{'guid': 'a'*32, 'url': 'u1'},
                                     {'guid': 'b'*32, 'url': 'u2'}]}}, 2),
    ({'data': {'total': 0, 'list': []}}, 0),
    ({'data': {}}, 0),
    ({'data': {'list': 'notalist'}}, 0),
    ({}, 0),
    ('not a dict', 0),
    ({'data': {'list': [{'guid': 'a'*32, 'url': 'u1'}, 'bogus', None]}}, 1),
])
def test_parse_list_response(payload, expected_count):
    assert len(parse_list_response(payload)) == expected_count


def test_parse_album_info():
    assert parse_album_info({'data': {'id': 'VIDAabc'}}) == 'VIDAabc'
    assert parse_album_info({'data': {'id': 'TOPC1'}}) is None  # wrong prefix
    assert parse_album_info({'data': {}}) is None
    assert parse_album_info({}) is None
    assert parse_album_info('not a dict') is None


def test_dedupe_key_prefers_guid():
    ep = {'guid': 'a'*32, 'url': 'https://x'}
    assert cctv_series._dedupe_key(ep).startswith('guid:')


def test_dedupe_key_falls_back_to_url():
    ep = {'url': 'https://x'}
    assert cctv_series._dedupe_key(ep).startswith('url:')


def test_dedupe_key_none_when_both_missing():
    assert cctv_series._dedupe_key({}) is None
    assert cctv_series._dedupe_key({'guid': '', 'url': ''}) is None


def test_classify_series_unknown_for_empty():
    assert classify_series([], 'VIDE1') is SeriesKind.UNKNOWN


def test_classify_series_single_when_only_current():
    eps = [episode('a'*32, 'https://tv.cctv.com/2024/03/01/VIDEx.shtml')]
    assert classify_series(eps, 'VIDEx') is SeriesKind.SINGLE


def test_classify_series_series_with_other():
    eps = [
        episode('a'*32, 'https://tv.cctv.com/2024/03/01/VIDEx.shtml'),
        episode('b'*32, 'https://tv.cctv.com/2024/03/02/VIDEy.shtml'),
    ]
    assert classify_series(eps, 'VIDEx') is SeriesKind.SERIES


def test_classify_series_handles_missing_current_id():
    eps = [episode('a'*32, 'https://tv.cctv.com/2024/03/01/VIDEx.shtml')]
    # current_item_id=None -> no comparison -> at least 1 valid URL -> SERIES
    assert classify_series(eps, None) is SeriesKind.SERIES


def test_classify_series_skips_eps_without_url():
    eps = [{'guid': 'a'*32}, episode('b'*32, 'https://tv.cctv.com/...VIDEy.shtml')]
    assert classify_series(eps, 'VIDEx') is SeriesKind.SERIES


def test_month_range_same_month():
    assert list(month_range('202401', '202401')) == ['202401']


def test_month_range_cross_year():
    assert list(month_range('202411', '202502')) == ['202411', '202412', '202501', '202502']


def test_month_range_full_year():
    months = list(month_range('202401', '202412'))
    assert len(months) == 12
    assert months[0] == '202401'
    assert months[-1] == '202412'


def test_dedupe_and_trim_respects_cap():
    eps = [episode(f'{i:032x}', f'https://x/{i}') for i in range(600)]
    trimmed = cctv_series._dedupe_and_trim(eps)
    assert len(trimmed) == cctv_series._MAX_EPISODES


def test_dedupe_and_trim_dedupes_by_guid():
    eps = [
        episode('a'*32, 'https://x/1'),
        episode('a'*32, 'https://x/2'),  # same guid -> duplicate
    ]
    assert len(cctv_series._dedupe_and_trim(eps)) == 1


# --- orchestration (fake fetch) --------------------------------------------

async def test_resolve_non_cctv_short_circuits():
    fetch = FakeFetch({})
    result = await fetch_series_episodes(NON_CCTV, _fetch=fetch)
    assert result.kind is SeriesKind.UNKNOWN
    assert result.episodes == []
    assert fetch.requests == []


async def test_resolve_column_api_success_single_month():
    html = episode_html(column_id='TOPC1', item_id='VIDEcur')
    eps = [
        episode('a'*32, 'https://tv.cctv.com/2024/03/01/VIDEcur.shtml'),
        episode('b'*32, 'https://tv.cctv.com/2024/03/02/VIDE2.shtml'),
        episode('c'*32, 'https://tv.cctv.com/2024/03/03/VIDE3.shtml'),
    ]
    page_url = build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE)
    fetch = FakeFetch({
        PAGE: html,
        page_url: {'data': {'total': len(eps), 'list': eps}},
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SERIES
    assert result.source == 'column-api'
    assert result.column_id == 'TOPC1'
    assert len(result.episodes) == 3
    assert result.episodes[1] == 'https://tv.cctv.com/2024/03/02/VIDE2.shtml'


async def test_resolve_column_api_paginates_within_month():
    """total > page_size triggers additional page fetches."""
    page_size = cctv_series._DEFAULT_PAGE_SIZE
    today_yyyymm = cctv_series._today_yyyymm()
    html = episode_html(column_id='TOPC1', item_id='VIDEcur')
    eps1 = [episode(f'{i:032x}', f'https://tv.cctv.com/u{i}') for i in range(page_size)]
    eps2 = [episode(f'{page_size+i:032x}', f'https://tv.cctv.com/u{page_size+i}')
            for i in range(20)]
    fetch = FakeFetch({
        PAGE: html,
        build_column_url('TOPC1', today_yyyymm, 1, page_size):
            {'data': {'total': page_size + 20, 'list': eps1}},
        build_column_url('TOPC1', today_yyyymm, 2, page_size):
            {'data': {'total': page_size + 20, 'list': eps2}},
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SERIES
    assert len(result.episodes) == page_size + 20


async def test_resolve_skips_other_episodes_when_only_current_returned():
    """The list contains only the user's episode -> SINGLE (not SERIES)."""
    item_id = 'VIDEcur'
    html = episode_html(column_id='TOPC1', item_id=item_id)
    eps = [episode('a'*32, f'https://tv.cctv.com/2024/03/01/{item_id}.shtml')]
    fetch = FakeFetch({
        PAGE: html,
        build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE):
            {'data': {'total': 1, 'list': eps}},
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SINGLE


async def test_resolve_falls_back_to_album_api_when_no_column_id():
    """Level 2 path: HTML has no column_id but has item_id; VIDE -> VIDA
    conversion succeeds; album API returns episodes."""
    html = episode_html(column_id=None, item_id='VIDEcur')
    eps = [
        episode('a'*32, 'https://tv.cctv.com/2024/03/01/VIDEcur.shtml'),
        episode('b'*32, 'https://tv.cctv.com/2024/03/02/VIDE2.shtml'),
    ]
    fetch = FakeFetch({
        PAGE: html,
        build_album_info_url('VIDEcur'):
            {'data': {'id': 'VIDAalbum'}},
        build_album_url('VIDAalbum', 1, cctv_series._DEFAULT_PAGE_SIZE):
            {'data': {'total': len(eps), 'list': eps}},
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SERIES
    assert result.source == 'album-api'


async def test_resolve_falls_back_to_html_when_apis_fail():
    """Level 3: column + album APIs return None; HTML contains related
    episodes via inline anchors / JS 'url' fields."""
    # 2 <a href="..."> related episodes + a duplicate
    other = 'https://tv.cctv.com/2024/03/02/VIDE2.shtml'
    third = 'https://tv.cctv.com/2024/03/03/VIDE3.shtml'
    html = (episode_html(column_id='TOPC1', item_id='VIDEcur')
            + f'<a href="{other}">next</a>'
            + f'<a href="{other}">dup</a>'
            + f'<a href="{third}">third</a>')
    fetch = FakeFetch({
        PAGE: html,
        # Column API fails
        build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE): None,
        # Album-info lookup fails
        build_album_info_url('VIDEcur'): None,
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SERIES
    assert result.source == 'html-fallback'
    assert other in result.episodes
    assert third in result.episodes


async def test_resolve_unknown_when_all_levels_fail():
    """Column + album + html fallback all yield nothing -> UNKNOWN."""
    html = episode_html(column_id='TOPC1', item_id='VIDEcur')
    fetch = FakeFetch({
        PAGE: html,
        build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE): None,
        build_album_info_url('VIDEcur'): None,
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.UNKNOWN


async def test_resolve_html_fallback_only_current_means_unknown():
    """HTML lists only the current episode -> Level 3 alone yields SINGLE
    -> caller treats as single, no expansion."""
    html = (episode_html(column_id='TOPC1', item_id='VIDEcur')
            + '<a href="https://tv.cctv.com/2024/03/01/VIDEcur.shtml">self</a>')
    fetch = FakeFetch({
        PAGE: html,
        build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE): None,
        build_album_info_url('VIDEcur'): None,
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    # Level 1 fails, Level 2 fails, Level 3 finds only current -> downgrade
    # to UNKNOWN at the bottom (HTML fallback downgrades SINGLE to nothing
    # because we require >=1 other episode; single-episode HTML is not a
    # reliable indicator of a series).
    assert result.kind is SeriesKind.UNKNOWN


async def test_resolve_retries_truncated_api_json():
    """API returns truncated JSON first, then a valid one."""
    item_id = 'VIDEcur'
    page_url = build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE)
    valid_eps = [
        episode('a'*32, 'https://tv.cctv.com/2024/03/01/VIDEcur.shtml'),
        episode('b'*32, 'https://tv.cctv.com/2024/03/02/VIDE2.shtml'),
    ]
    html = episode_html(column_id='TOPC1', item_id=item_id)
    fetch = FakeFetch({
        PAGE: html,
        page_url: ['{"data": {"total": 2, "list": [{"guid":',  # truncated
                   {'data': {'total': 2, 'list': valid_eps}}],
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SERIES
    assert fetch.requests.count(page_url) == 2


async def test_resolve_caches_column_page_across_calls():
    """Second call with the same column_id reuses cached JSON."""
    page_url = build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE)
    eps = [
        episode('a'*32, 'https://tv.cctv.com/2024/03/01/VIDEcur.shtml'),
        episode('b'*32, 'https://tv.cctv.com/2024/03/02/VIDE2.shtml'),
    ]
    html = episode_html(column_id='TOPC1', item_id='VIDEcur')

    first = FakeFetch({
        PAGE: html,
        page_url: {'data': {'total': len(eps), 'list': eps}},
    })
    r1 = await fetch_series_episodes(PAGE, _fetch=first)
    assert r1.kind is SeriesKind.SERIES

    second = FakeFetch({
        PAGE: html,
        page_url: {'data': {'total': len(eps), 'list': eps}},
    })
    r2 = await fetch_series_episodes(PAGE, _fetch=second)
    assert r2.kind is SeriesKind.SERIES
    # second request to the API URL was served from cache, not fetch
    assert second.requests.count(page_url) == 0


async def test_resolve_handles_column_id_response_with_no_data(monkeypatch):
    """API returns HTTP 200 with an unparseable payload -> tries 3x -> UNKNOWN."""
    html = episode_html(column_id='TOPC1', item_id='VIDEcur')
    fetch = FakeFetch({
        PAGE: html,
        build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE): 'not json',
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.UNKNOWN


async def test_resolve_handles_column_id_response_with_list_only_current():
    """API returns total=1 list=[current] -> classify -> SINGLE -> UNKNOWN
    path: caller treats as single download, no expansion."""
    item_id = 'VIDEcur'
    html = episode_html(column_id='TOPC1', item_id=item_id)
    eps = [episode('a'*32, f'https://tv.cctv.com/2024/03/01/{item_id}.shtml')]
    fetch = FakeFetch({
        PAGE: html,
        build_column_url('TOPC1', cctv_series._today_yyyymm(), 1, cctv_series._DEFAULT_PAGE_SIZE):
            {'data': {'total': 1, 'list': eps}},
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SINGLE


async def test_resolve_never_raises_on_fetch_exception():
    async def exploding_fetch(session, url):
        raise RuntimeError('boom')
    result = await fetch_series_episodes(PAGE, _fetch=exploding_fetch)
    assert result.kind is SeriesKind.UNKNOWN


async def test_resolve_handles_html_fetch_failure():
    fetch = FakeFetch({PAGE: None})
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.UNKNOWN


async def test_resolve_handles_html_without_any_ids():
    fetch = FakeFetch({PAGE: '<html>no column id here</html>'})
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.UNKNOWN


async def test_fetch_rejects_disallowed_url(monkeypatch):
    monkeypatch.setattr(cctv_series, 'validate_url',
                        lambda url, allow_private=False: 'blocked by test')
    import aiohttp
    async with aiohttp.ClientSession() as session:
        result = await cctv_series._aiohttp_fetch_checked(
            session, 'https://example.com/x', allow_private=False)
    assert result is None


async def test_resolve_cross_month_dedupes():
    """Same episode appears in two adjacent months -> dedupe keeps one."""
    # No pub_date on the page so the start anchor is _five_years_ago_yyyymm().
    html = (f'<html><script>var column_id = "TOPC1"; '
            f'var itemid1 = "VIDEcur";</script></html>')
    today_yyyymm = cctv_series._today_yyyymm()
    this_month = today_yyyymm
    prev_yyyymm = cctv_series._five_years_ago_yyyymm()  # any past month
    shared = episode('z'*32, 'https://tv.cctv.com/2024/01/15/VIDEshared.shtml')
    fetch = FakeFetch({
        PAGE: html,
        build_column_url('TOPC1', prev_yyyymm, 1, cctv_series._DEFAULT_PAGE_SIZE):
            {'data': {'total': 2, 'list': [
                shared,
                episode('a'*32, 'https://tv.cctv.com/2024/01/02/VIDEa.shtml')]}},
        build_column_url('TOPC1', this_month, 1, cctv_series._DEFAULT_PAGE_SIZE):
            {'data': {'total': 2, 'list': [
                shared,
                episode('b'*32, 'https://tv.cctv.com/2026/09/03/VIDEb.shtml')]}},
    })
    result = await fetch_series_episodes(PAGE, _fetch=fetch)
    assert result.kind is SeriesKind.SERIES
    # shared episode appears once even though two months listed it
    assert result.episodes.count('https://tv.cctv.com/2024/01/15/VIDEshared.shtml') == 1
    # both 'other' episodes present
    assert 'https://tv.cctv.com/2024/01/02/VIDEa.shtml' in result.episodes
    assert 'https://tv.cctv.com/2026/09/03/VIDEb.shtml' in result.episodes


def test_no_cctv_url_match_returns_unknown(monkeypatch):
    """When fetch returns HTML that doesn't even contain the URL pattern
    (e.g. block-page), we still don't raise."""
    async def run():
        fetch = FakeFetch({PAGE: '<html>blocked</html>'})
        return await fetch_series_episodes(PAGE, _fetch=fetch)
    result = asyncio.run(run())
    assert result.kind is SeriesKind.UNKNOWN


def test_cap_constants_are_sane():
    assert 100 <= cctv_series._MAX_EPISODES <= 2000
    assert 12 <= cctv_series._MAX_MONTHS <= 120
    assert 3600.0 <= cctv_series._CACHE_TTL <= 24 * 3600.0


def test_series_result_default_episodes_is_list():
    r = SeriesResult(kind=SeriesKind.UNKNOWN)
    assert r.episodes == []
    r.episodes.append('x')
    # new instance must not share the same list (no mutable-default bug)
    r2 = SeriesResult(kind=SeriesKind.UNKNOWN)
    assert r2.episodes == []