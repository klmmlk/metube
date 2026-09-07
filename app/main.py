*** Begin Patch
*** Update File: app/main.py
@@
     # If expansion yields multiple episode URLs, add each individually using the same options.
     if expanded and len(expanded) > 1:
@@
-        for eps_url in expanded:
-            try:
-                status = await dqueue.add(
-                    eps_url,
-                    o['download_type'],
-                    o['codec'],
-                    o['format'],
-                    o['quality'],
-                    o['folder'],
-                    o['custom_name_prefix'],
-                    o['playlist_item_limit'],
-                    o['auto_start'],
-                    o['split_by_chapters'],
-                    o['chapter_template'],
-                    o['subtitle_language'],
-                    o['subtitle_mode'],
-                    o['ytdl_options_presets'],
-                    o['ytdl_options_overrides'],
-                    o['clip_start'],
-                    o['clip_end'],
-                    sponsorblock=o['sponsorblock'],
-                )
-            except Exception as e:
-                log.warning("Failed to add expanded URL %s: %s", eps_url, e)
-                status = {'status': 'error', 'msg': str(e)}
-            results.append({'url': eps_url, 'result': status})
+        for eps_url in expanded:
+            try:
+                # Probe best format for this episode URL (non-blocking via threadpool)
+                try:
+                    chosen_format = await asyncio.get_running_loop().run_in_executor(None, _pick_best_format_id, eps_url)
+                except Exception as e:
+                    log.debug("format probe raised for %s: %s", eps_url, e)
+                    chosen_format = None
+
+                item_overrides = (o.get('ytdl_options_overrides') or {}).copy()
+                if chosen_format:
+                    item_overrides['format'] = chosen_format
+
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
+                    item_overrides,
+                    o['clip_start'],
+                    o['clip_end'],
+                    sponsorblock=o['sponsorblock'],
+                )
+            except Exception as e:
+                log.warning("Failed to add expanded URL %s: %s", eps_url, e)
+                status = {'status': 'error', 'msg': str(e)}
+            results.append({'url': eps_url, 'result': status})
*** End Patch
