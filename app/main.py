*** Begin Patch
*** Update File: app/main.py
@@
     try:
         o = parse_download_options(post)
     except web.HTTPBadRequest as e:
         log.error("Bad request: %s", e.reason)
         raise
+    # helper: probe available formats with yt-dlp and pick the best format id
+    def _pick_best_format_id(url: str) -> str | None:
+        try:
+            from yt_dlp import YoutubeDL
+        except Exception as e:
+            log.debug("yt_dlp not available for probing: %s", e)
+            return None
+
+        ydl_opts = {"quiet": True, "no_warnings": True}
+        try:
+            with YoutubeDL(ydl_opts) as ydl:
+                info = ydl.extract_info(url, download=False)
+        except Exception as e:
+            log.debug("yt-dlp probe failed for %s: %s", url, e)
+            return None
+
+        formats = info.get('formats') or []
+        best = None
+        for f in formats:
+            # skip audio-only
+            if f.get('vcodec') == 'none':
+                continue
+            score = (f.get('height') or 0, f.get('tbr') or 0, f.get('filesize') or 0)
+            if best is None:
+                best = f
+            else:
+                best_score = (best.get('height') or 0, best.get('tbr') or 0, best.get('filesize') or 0)
+                if score > best_score:
+                    best = f
+
+        if not best:
+            return None
+        # If selected format already includes audio, use it directly; otherwise combine with bestaudio
+        if best.get('acodec') and best.get('acodec') != 'none':
+            return str(best.get('format_id'))
+        return f"{best.get('format_id')}+bestaudio/best"
*** End Patch
