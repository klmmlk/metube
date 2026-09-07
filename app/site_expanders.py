----
*** Begin Patch
*** Update File: app/site_expanders.py
@@
 _CCTV_EPISODE_RE = re.compile(r"https?://(?:tv\.)?cctv\.cn/\d{4}/\d{2}/\d{2}/VID[0-9A-Za-z]+\.shtml")
+
+# 匹配 JS/JSON 里 'url': 'https://...VID....shtml' 或 "url":"..."
+_JS_URL_FIELD_RE = re.compile(
+    r"['\"]url['\"]\s*:\s*['\"](https?://(?:tv\.)?cctv\.cn/\d{4}/\d{2}/\d{2}/VID[0-9A-Za-z]+\.shtml)['\"]",
+    re.IGNORECASE,
+)
@@
     # 2) fallback: regex search in HTML for absolute matches
     for m in _CCTV_EPISODE_RE.finditer(html):
         add(m.group(0))
+
+    # 3) extra: search inline <script> content / JSON-like structures for "url": "..."
+    for script in soup.find_all("script"):
+        # script.string is None when the tag contains nested nodes or is empty;
+        # fall back to .text which concatenates child text nodes but is always safe.
+        script_text = script.string if script.string is not None else script.get_text()
+        if not script_text:
+            continue
+        for m in _JS_URL_FIELD_RE.finditer(script_text):
+            add(m.group(1))
*** End Patch
