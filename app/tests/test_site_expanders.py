import pytest

from app.site_expanders import expand_url


SAMPLE_HTML = '''
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Test CCTV List</title></head>
<body>
<script>
var jsonData2 = [{"title":"ep1","url":"https://tv.cctv.cn/2024/02/11/VIDEJMBU1.shtml"},
{"title":"ep2","url":"https://tv.cctv.cn/2024/02/12/VIDEJMBU2.shtml"},
{"title":"ep3","url":"https://tv.cctv.cn/2024/02/13/VIDEJMBU3.shtml"}];
</script>
<a href="https://tv.cctv.cn/2024/02/14/VIDEJMBU4.shtml">link4</a>
</body>
</html>
'''


def test_expand_from_inline_script(monkeypatch):
    class DummyResp:
        def __init__(self, text):
            self.text = text
        def raise_for_status(self):
            return None

    class DummySession:
        def __init__(self, text):
            self._text = text
            self.headers = {}
        def get(self, url, timeout=10):
            return DummyResp(self._text)

    def dummy_get_session():
        return DummySession(SAMPLE_HTML)

    monkeypatch.setattr('app.site_expanders._get_session', lambda: dummy_get_session())
    urls = expand_url('https://tv.cctv.cn/some/series/page.html')
    assert isinstance(urls, list)
    assert 'https://tv.cctv.cn/2024/02/11/VIDEJMBU1.shtml' in urls
    assert 'https://tv.cctv.cn/2024/02/12/VIDEJMBU2.shtml' in urls
    assert 'https://tv.cctv.cn/2024/02/13/VIDEJMBU3.shtml' in urls
    assert 'https://tv.cctv.cn/2024/02/14/VIDEJMBU4.shtml' in urls
