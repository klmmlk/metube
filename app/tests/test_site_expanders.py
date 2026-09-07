"""Unit tests for the CCTV series-page expander (app/site_expanders.py)."""

import pytest

import site_expanders
from site_expanders import _extract_episodes, expand_url, is_cctv_url

EP = 'https://tv.cctv.com/2024/03/01/VIDAb2Xx123.shtml'
EP_CN = 'https://tv.cctv.cn/2024/03/01/VIDAb2Xx123.shtml'
EP2 = 'https://tv.cctv.com/2024/03/02/VIDCd3Yy456.shtml'
SERIES = 'https://tv.cctv.com/2024/03/01/VIDAb2Xx123.shtml?spm=series'


@pytest.mark.parametrize('url,expected', [
    ('https://tv.cctv.com/lm/xwlb/videoset/', True),
    ('https://tv.cctv.cn/2024/03/01/VIDAb2Xx123.shtml', True),
    ('https://tv.cctv.com/2024/03/01/VIDAb2Xx123.shtml', True),
    ('https://news.cctv.com/some/page', True),
    ('https://cntv.cctv.cn/page', True),
    ('https://www.youtube.com/watch?v=x', False),
    ('https://cctv.com.evil.example.com/', False),
    ('not a url', False),
])
def test_is_cctv_url(url, expected):
    assert is_cctv_url(url) is expected


def test_extract_episodes_from_anchors_resolves_relative():
    html = f'<a href="{EP}">第1集</a><a href="/2024/03/02/VIDCd3Yy456.shtml">第2集</a>'
    assert _extract_episodes(html, SERIES) == [EP, EP2]


def test_extract_episodes_from_inline_js_url_fields():
    # the feda55c intent: series pages embed episode lists in inline JSON
    html = f"""<script>window.__INITIAL_STATE__ = {{"list": [
        {{"url": "{EP}", "title": "第1集"}},
        {{"url": "{EP_CN}", "title": "第2集"}}
    ]}}</script>"""
    assert _extract_episodes(html, SERIES) == [EP, EP_CN]


def test_extract_episodes_deduplicates_in_document_order():
    html = f'<a href="{EP}">a</a> <p>text {EP}</p> <a href="{EP2}">b</a> <a href="{EP}">c</a>'
    assert _extract_episodes(html, SERIES) == [EP, EP2]


def test_extract_episodes_ignores_non_episode_links():
    html = ('<a href="https://tv.cctv.com/lm/xwlb/">栏目</a>'
            '<a href="/2024/03/01/ARTI123.shtml">news</a>'
            '<a href="/2024/03/01/index.shtml">index</a>')
    assert _extract_episodes(html, SERIES) == []


def test_extract_episodes_empty_html():
    assert _extract_episodes('', SERIES) == []


async def test_expand_url_episode_page_short_circuits():
    # an episode page must never expand; this returns before any network use
    assert await expand_url(EP) == []


async def test_expand_url_non_cctv_short_circuits():
    assert await expand_url('https://www.youtube.com/watch?v=x') == []


async def test_expand_url_fetch_failure_returns_empty(monkeypatch):
    # any aiohttp use (ClientSession construction, the GET) explodes -> []
    monkeypatch.setattr(site_expanders, 'aiohttp', None)
    assert await expand_url('https://tv.cctv.com/lm/xwlb/videoset/') == []
