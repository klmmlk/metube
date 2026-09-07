diff --git a/app/main.py b/app/main.py
index 5ba2ba4..0000000 100644
--- a/app/main.py
+++ b/app/main.py
@@
 @routes.post(config.URL_PREFIX + 'add')
 async def add(request):
     log.info("Received request to add download")
     post = await _read_json_request(request)
     try:
         o = parse_download_options(post)
     except web.HTTPBadRequest as e:
         log.error("Bad request: %s", e.reason)
         raise
+    # Attempt to expand series/index pages into individual episode URLs.
+    expanded = []
+    try:
+        from app.site_expanders import expand_url
+        expanded = await asyncio.get_running_loop().run_in_executor(None, expand_url, o['url'])
+        if expanded is None:
+            expanded = []
+    except Exception as exc:
+        log.debug("expand_url failed or not available: %s", exc)
+        expanded = []
+
+    # If expansion yields multiple episode URLs, add each individually using the same options.
+    if expanded and len(expanded) > 1:
+        try:
+            limit = int(o.get('playlist_item_limit') or 0)
+        except Exception:
+            limit = 0
+        if limit > 0:
+            expanded = expanded[:limit]
+
+        log.info("URL expanded into %d items; adding them individually", len(expanded))
+        results = []
+        for eps_url in expanded:
+            try:
+                status = await dqueue.add(
+                    eps_url,
+                    o['download_type'],
+                    o['codec'],
+                    o['format'],
+                    o['quality'],
+                    o['folder'],
+                    o['custom_name_prefix'],
+                    o['playlist_item_limit'],
+                    o['auto_start'],
+                    o['split_by_chapters'],
+                    o['chapter_template'],
+                    o['subtitle_language'],
+                    o['subtitle_mode'],
+                    o['ytdl_options_presets'],
+                    o['ytdl_options_overrides'],
+                    o['clip_start'],
+                    o['clip_end'],
+                    sponsorblock=o['sponsorblock'],
+                )
+            except Exception as e:
+                log.warning("Failed to add expanded URL %s: %s", eps_url, e)
+                status = {'status': 'error', 'msg': str(e)}
+            results.append({'url': eps_url, 'result': status})
+        return web.Response(text=serializer.encode({'status': 'ok', 'expanded_count': len(results), 'results': results}))
@@
-    status = await dqueue.add(
-        o['url'],
-        o['download_type'],
-        o['codec'],
-        o['format'],
-        o['quality'],
-        o['folder'],
-        o['custom_name_prefix'],
-        o['playlist_item_limit'],
-        o['auto_start'],
-        o['split_by_chapters'],
-        o['chapter_template'],
-        o['subtitle_language'],
-        o['subtitle_mode'],
-        o['ytdl_options_presets'],
-        o['ytdl_options_overrides'],
-        o['clip_start'],
-        o['clip_end'],
-        sponsorblock=o['sponsorblock'],
-    )
+    status = await dqueue.add(
+        o['url'],
+        o['download_type'],
+        o['codec'],
+        o['format'],
+        o['quality'],
+        o['folder'],
+        o['custom_name_prefix'],
+        o['playlist_item_limit'],
+        o['auto_start'],
+        o['split_by_chapters'],
+        o['chapter_template'],
+        o['subtitle_language'],
+        o['subtitle_mode'],
+        o['ytdl_options_presets'],
+        o['ytdl_options_overrides'],
+        o['clip_start'],
+        o['clip_end'],
+        sponsorblock=o['sponsorblock'],
+    )
     return web.Response(text=serializer.encode(status))
