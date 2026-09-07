"""Tests for ``DownloadQueue`` with mocked yt-dlp extraction."""

from __future__ import annotations

import copy
import os
import re
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import time

from ytdl import Download, DownloadInfo, DownloadQueue


@pytest.fixture
def dq_env():
    with tempfile.TemporaryDirectory() as tmp:
        dl = os.path.join(tmp, "downloads")
        st = os.path.join(tmp, "state")
        os.makedirs(dl, exist_ok=True)
        os.makedirs(st, exist_ok=True)
        cfg = MagicMock()
        cfg.STATE_DIR = st
        cfg.DOWNLOAD_DIR = dl
        cfg.AUDIO_DOWNLOAD_DIR = dl
        cfg.TEMP_DIR = dl
        cfg.MAX_CONCURRENT_DOWNLOADS = "3"
        cfg.YTDL_OPTIONS = {}
        cfg.YTDL_OPTIONS_PRESETS = {}
        cfg.CUSTOM_DIRS = True
        cfg.CREATE_CUSTOM_DIRS = True
        cfg.CLEAR_COMPLETED_AFTER = "0"
        cfg.DELETE_FILE_ON_TRASHCAN = False
        cfg.OUTPUT_TEMPLATE = "%(title)s.%(ext)s"
        cfg.OUTPUT_TEMPLATE_CHAPTER = "%(title)s.%(ext)s"
        cfg.OUTPUT_TEMPLATE_PLAYLIST = ""
        cfg.OUTPUT_TEMPLATE_CHANNEL = ""
        yield cfg


def test_cancel_add_increments_generation(dq_env):
    notifier = MagicMock()
    dq = DownloadQueue(dq_env, notifier)
    before = dq._add_generation
    dq.cancel_add()
    assert dq._add_generation == before + 1


def test_download_queue_has_dedicated_executor_sized_from_config(dq_env):
    notifier = MagicMock()
    dq = DownloadQueue(dq_env, notifier)
    assert dq._download_executor is not None
    assert dq._download_executor._max_workers == 2 * int(dq_env.MAX_CONCURRENT_DOWNLOADS) + 2
    dq.close()


def test_close_cancels_running_downloads_before_shutdown(dq_env):
    notifier = MagicMock()
    dq = DownloadQueue(dq_env, notifier)

    running = MagicMock()
    running.started.return_value = True
    running.running.return_value = True
    idle = MagicMock()
    idle.started.return_value = False
    idle.running.return_value = False

    dq.queue.dict["u-running"] = running
    dq.queue.dict["u-idle"] = idle

    dq.close()

    # The active download's subprocess group is killed; the not-started one is
    # left alone. Executor is shut down afterwards.
    running.cancel.assert_called_once()
    idle.cancel.assert_not_called()
    assert dq._download_executor._shutdown


def test_get_returns_tuple_of_lists(dq_env):
    notifier = MagicMock()
    dq = DownloadQueue(dq_env, notifier)
    q, done = dq.get()
    assert q == [] and done == []


@pytest.mark.asyncio
async def test_add_single_video_goes_to_pending_when_auto_start_false(dq_env):
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": url,
            "webpage_url": url,
        }

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        result = await dq.add(
            "https://example.com/watch?v=1",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
        )
    assert result["status"] == "ok"
    assert dq.pending.exists("https://example.com/watch?v=1")


@pytest.mark.asyncio
async def test_add_unsupported_url_recorded_as_failed_entry(dq_env):
    """An unsupported/unextractable URL must show up as a red-cross entry in the
    done list, not just a transient toast and a server log line."""
    import ytdl

    notifier = AsyncMock()
    url = "https://example.com/not-a-video"

    def boom(self, url, *_args, **_kwargs):
        raise ytdl.yt_dlp.utils.YoutubeDLError(f'Unsupported URL: {url}')

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", boom):
        result = await dq.add(
            url, "video", "auto", "any", "best", "", "", 0, auto_start=True,
        )
    assert result["status"] == "error"
    assert dq.done.exists(url)
    failed = dq.done.get(url)
    assert failed.info.status == "error"
    assert failed.info.error == result["msg"]
    assert failed.info.url == url
    # The full URL stays in .url/.error for the detail panel; the display
    # title is shortened to the hostname so the Completed row stays readable.
    assert failed.info.title == "example.com"
    notifier.completed.assert_awaited()


@pytest.mark.asyncio
async def test_add_ssrf_rejected_url_recorded_as_failed_entry(dq_env):
    """A URL rejected by the SSRF guard (before yt-dlp ever runs) must also
    surface as a failed entry, not just an error status returned to the caller."""
    notifier = AsyncMock()
    url = "file:///etc/passwd"

    dq = DownloadQueue(dq_env, notifier)
    result = await dq.add(
        url, "video", "auto", "any", "best", "", "", 0, auto_start=True,
    )
    assert result["status"] == "error"
    assert dq.done.exists(url)
    failed = dq.done.get(url)
    assert failed.info.status == "error"
    assert failed.info.error == result["msg"]
    notifier.completed.assert_awaited()


@pytest.mark.asyncio
async def test_cancel_removes_from_pending(dq_env):
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": url,
            "webpage_url": url,
        }

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        await dq.add(
            "https://example.com/pending",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
        )
    url = "https://example.com/pending"
    await dq.cancel([url])
    assert not dq.pending.exists(url)
    notifier.canceled.assert_awaited()


@pytest.mark.asyncio
async def test_cancel_before_start_marks_download_canceled(dq_env):
    """Regression test for the race condition where cancel() arrives after the
    download has been placed in the queue and ``__start_download`` has been
    scheduled via ``asyncio.create_task`` but has not yet executed. Without the
    fix, the pending task would run ``download.start()`` despite the user
    cancelling, because its ``download.canceled`` guard was never flipped."""
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": url,
            "webpage_url": url,
        }

    dq = DownloadQueue(dq_env, notifier)
    url = "https://example.com/race"
    start_mock = AsyncMock()
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq.add(
            url,
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=True,
        )
        assert dq.queue.exists(url)
        download = dq.queue.get(url)
        assert download.canceled is False
        await dq.cancel([url])
        assert not dq.queue.exists(url)
        assert download.canceled is True
        notifier.canceled.assert_awaited_with(url)


@pytest.mark.asyncio
async def test_start_pending_moves_to_queue(dq_env):
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": url,
            "webpage_url": url,
        }

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        await dq.add(
            "https://example.com/startme",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
        )
    url = "https://example.com/startme"
    # Starting will spawn real download — cancel immediately before worker runs much
    with patch.object(DownloadQueue, "_DownloadQueue__start_download", AsyncMock()):
        await dq.start_pending([url])
    assert not dq.pending.exists(url)


@pytest.mark.asyncio
async def test_add_entry_queues_single_video_without_reextracting(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    entry = {
        "_type": "video",
        "id": "vid1",
        "title": "Test Video",
        "url": "https://example.com/watch?v=1",
        "webpage_url": "https://example.com/watch?v=1",
        "playlist_index": "01",
        "playlist_title": "Playlist",
    }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", side_effect=AssertionError("should not re-extract")):
        result = await dq.add_entry(
            entry,
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
        )

    assert result["status"] == "ok"
    assert dq.pending.exists("https://example.com/watch?v=1")


@pytest.mark.asyncio
async def test_retry_restores_playlist_output_context(dq_env):
    notifier = AsyncMock()
    dq_env.OUTPUT_TEMPLATE_PLAYLIST = "%(playlist_title)s/%(title)s.%(ext)s"
    dq = DownloadQueue(dq_env, notifier)
    url = "https://example.com/watch?v=1"
    failed_info = DownloadInfo(
        id="vid1",
        title="Test Video",
        url=url,
        quality="best",
        download_type="video",
        codec="auto",
        format="any",
        folder="",
        custom_name_prefix="",
        error="temporary failure",
        entry={
            "playlist_index": "01",
            "playlist_title": "My Playlist",
            "playlist_count": 10,
        },
        playlist_item_limit=0,
        split_by_chapters=False,
        chapter_template="",
    )
    failed_info.status = "error"
    await dq.done.put(Download(None, None, None, None, "best", "any", {}, failed_info))

    def fake_extract(self, extracted_url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": extracted_url,
            "webpage_url": extracted_url,
        }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        result = await dq.retry(url)

    assert result["status"] == "ok"
    queued = dq.queue.get(url)
    assert queued.output_template == "My Playlist/%(title)s.%(ext)s"
    assert queued.info.entry["playlist_index"] == "01"
    assert queued.info.entry["playlist_title"] == "My Playlist"


def _failed_playlist_item(url, **overrides):
    """A done-list entry for a playlist item that failed mid-download."""
    info = DownloadInfo(
        id="vid1",
        title="Test Video",
        url=url,
        quality="best",
        download_type="video",
        codec="auto",
        format="any",
        folder="",
        custom_name_prefix="",
        error="temporary failure",
        entry={
            "playlist_index": "01",
            "playlist_title": "My Playlist",
            "playlist_count": 10,
        },
        playlist_item_limit=0,
        split_by_chapters=False,
        chapter_template="",
        **overrides,
    )
    info.status = "error"
    return info


@pytest.mark.asyncio
async def test_retry_keeps_playlist_context_through_url_indirection(dq_env):
    # extract_flat=True makes yt-dlp hand back url/url_transparent results
    # unprocessed, so __add_entry recurses into add() a second time. The retry
    # context has to survive that hop or the item lands in the root directory.
    notifier = AsyncMock()
    dq_env.OUTPUT_TEMPLATE_PLAYLIST = "%(playlist_title)s/%(title)s.%(ext)s"
    dq = DownloadQueue(dq_env, notifier)
    url = "https://example.com/watch?v=1"
    resolved = "https://example.com/resolved?v=1"
    await dq.done.put(Download(None, None, None, None, "best", "any", {}, _failed_playlist_item(url)))

    def fake_extract(self, extracted_url, *_args, **_kwargs):
        if extracted_url == url:
            return {"_type": "url", "url": resolved, "id": "vid1"}
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": extracted_url,
            "webpage_url": extracted_url,
        }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        result = await dq.retry(url)

    assert result["status"] == "ok"
    queued = dq.queue.get(resolved)
    assert queued.output_template == "My Playlist/%(title)s.%(ext)s"
    assert queued.info.entry["playlist_title"] == "My Playlist"


@pytest.mark.asyncio
async def test_retry_reapplies_current_options_gates(dq_env):
    # The stored options passed parse_download_options when first submitted, but
    # the configuration can have changed since; retry must not resurrect
    # overrides or presets the current configuration no longer allows.
    notifier = AsyncMock()
    dq_env.ALLOW_YTDL_OPTIONS_OVERRIDES = False
    dq_env.YTDL_OPTIONS_PRESETS = {"Still There": {"writesubtitles": True}}
    dq = DownloadQueue(dq_env, notifier)
    url = "https://example.com/watch?v=1"
    info = _failed_playlist_item(
        url,
        ytdl_options_presets=["Still There", "Removed Preset"],
        ytdl_options_overrides={"paths": {"home": "/etc"}},
    )
    await dq.done.put(Download(None, None, None, None, "best", "any", {}, info))

    def fake_extract(self, extracted_url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": extracted_url,
            "webpage_url": extracted_url,
        }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        result = await dq.retry(url)

    assert result["status"] == "ok"
    queued = dq.queue.get(url)
    assert queued.info.ytdl_options_overrides == {}
    assert queued.info.ytdl_options_presets == ["Still There"]
    assert queued.ytdl_opts.get("paths", {}).get("home") != "/etc"


@pytest.mark.asyncio
async def test_retry_keeps_overrides_while_still_allowed(dq_env):
    notifier = AsyncMock()
    dq_env.ALLOW_YTDL_OPTIONS_OVERRIDES = True
    dq_env.YTDL_OPTIONS_PRESETS = {}
    dq = DownloadQueue(dq_env, notifier)
    url = "https://example.com/watch?v=1"
    info = _failed_playlist_item(url, ytdl_options_overrides={"writesubtitles": True})
    await dq.done.put(Download(None, None, None, None, "best", "any", {}, info))

    def fake_extract(self, extracted_url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": extracted_url,
            "webpage_url": extracted_url,
        }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        result = await dq.retry(url)

    assert result["status"] == "ok"
    assert dq.queue.get(url).info.ytdl_options_overrides == {"writesubtitles": True}


@pytest.mark.asyncio
async def test_retry_carries_the_sponsorblock_flag(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    url = "https://example.com/watch?v=1"
    await dq.done.put(
        Download(None, None, None, None, "best", "any", {}, _failed_playlist_item(url, sponsorblock=True))
    )

    def fake_extract(self, extracted_url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": extracted_url,
            "webpage_url": extracted_url,
        }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        result = await dq.retry(url)

    assert result["status"] == "ok"
    assert dq.queue.get(url).info.sponsorblock is True


@pytest.mark.asyncio
async def test_add_entry_duplicate_while_pending_is_skipped_not_clobbered(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    entry = {
        "_type": "video",
        "id": "vid1",
        "title": "Original Title",
        "url": "https://example.com/watch?v=1",
        "webpage_url": "https://example.com/watch?v=1",
    }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", side_effect=AssertionError("should not re-extract")):
        first = await dq.add_entry(entry, "video", "auto", "any", "best", "", "", 0, auto_start=False)
        assert first["status"] == "ok"
        assert "msg" not in first

        dupe_entry = {**entry, "title": "Different Title"}
        second = await dq.add_entry(dupe_entry, "audio", "auto", "mp3", "best", "", "", 0, auto_start=False)

    assert second["status"] == "ok"
    assert "Already in queue" in second["msg"]
    # The original pending download's options must survive untouched.
    pending_dl = dq.pending.get("https://example.com/watch?v=1")
    assert pending_dl.info.download_type == "video"
    assert pending_dl.info.title == "Original Title"


@pytest.mark.asyncio
async def test_add_entry_duplicate_while_queued_is_skipped(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    entry = {
        "_type": "video",
        "id": "vid1",
        "title": "Test Video",
        "url": "https://example.com/watch?v=1",
        "webpage_url": "https://example.com/watch?v=1",
    }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", side_effect=AssertionError("should not re-extract")), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        first = await dq.add_entry(entry, "video", "auto", "any", "best", "", "", 0, auto_start=True)
        assert first["status"] == "ok"
        assert dq.queue.exists("https://example.com/watch?v=1")

        second = await dq.add_entry(entry, "video", "auto", "any", "best", "", "", 0, auto_start=True)

    assert second["status"] == "ok"
    assert "Already in queue" in second["msg"]


@pytest.mark.asyncio
async def test_channel_download_uses_output_template_when_channel_template_empty(dq_env):
    """Channel tabs reported as playlists must honor OUTPUT_TEMPLATE when OUTPUT_TEMPLATE_CHANNEL is empty."""
    notifier = AsyncMock()
    dq_env.OUTPUT_TEMPLATE = "%(channel)s [YT]/%(title)s.%(ext)s"
    dq_env.OUTPUT_TEMPLATE_CHANNEL = ""
    dq_env.OUTPUT_TEMPLATE_PLAYLIST = ""

    channel_id = "UCabcd123"

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "playlist",
            "id": channel_id,
            "channel_id": channel_id,
            "channel": "Odin",
            "title": "Odin - Videos",
            "entries": [
                {
                    "id": "vid1",
                    "title": "Salvia Plath - Pondering",
                    "url": "https://example.com/watch?v=1",
                    "webpage_url": "https://example.com/watch?v=1",
                    "channel": "Odin",
                    "upload_date": "20130804",
                },
            ],
        }

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        result = await dq.add(
            "https://www.youtube.com/@odin/videos",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
        )

    assert result["status"] == "ok"
    url = "https://example.com/watch?v=1"
    assert dq.pending.exists(url)
    download = dq.pending.get(url)
    assert download.output_template.startswith("Odin [YT]/")
    assert "Odin - Videos" not in download.output_template


@pytest.mark.asyncio
async def test_playlist_download_not_treated_as_channel(dq_env):
    """Real playlists (id != channel_id) must not be promoted to channel downloads."""
    notifier = AsyncMock()
    dq_env.OUTPUT_TEMPLATE = "%(channel)s [YT]/%(title)s.%(ext)s"
    dq_env.OUTPUT_TEMPLATE_CHANNEL = ""
    dq_env.OUTPUT_TEMPLATE_PLAYLIST = "%(playlist_title)s/%(title)s.%(ext)s"

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "playlist",
            "id": "PLxyz789",
            "channel_id": "UCabcd123",
            "channel": "Odin",
            "title": "My Playlist",
            "entries": [
                {
                    "id": "vid1",
                    "title": "Test Video",
                    "url": "https://example.com/watch?v=1",
                    "webpage_url": "https://example.com/watch?v=1",
                },
            ],
        }

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        result = await dq.add(
            "https://www.youtube.com/playlist?list=PLxyz789",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
        )

    assert result["status"] == "ok"
    url = "https://example.com/watch?v=1"
    assert dq.pending.exists(url)
    download = dq.pending.get(url)
    assert download.output_template.startswith("My Playlist/")


def _channel_extraction(entry_id, **extra):
    """A channel yt-dlp reported as a playlist, addressed by *entry_id*."""
    return {
        "_type": "playlist",
        "id": entry_id,
        "channel_id": "UCabcd123",
        "channel": "Odin",
        "title": "Odin",
        **extra,
        "entries": [
            {
                "id": "vid1",
                "title": "Salvia Plath - Pondering",
                "url": "https://example.com/watch?v=1",
                "webpage_url": "https://example.com/watch?v=1",
                "channel": "Odin",
                "upload_date": "20130804",
            },
        ],
    }


async def _add_and_get_template(dq_env, extraction, url):
    dq_env.OUTPUT_TEMPLATE = "%(channel)s [YT]/%(title)s.%(ext)s"
    dq_env.OUTPUT_TEMPLATE_CHANNEL = ""
    dq_env.OUTPUT_TEMPLATE_PLAYLIST = "%(playlist_title)s/%(title)s.%(ext)s"

    def fake_extract(self, _url, *_args, **_kwargs):
        return extraction

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        result = await dq.add(url, "video", "auto", "any", "best", "", "", 0, auto_start=False)
    assert result["status"] == "ok"
    return dq.pending.get("https://example.com/watch?v=1").output_template


@pytest.mark.asyncio
async def test_bare_handle_channel_url_is_treated_as_a_channel(dq_env):
    """A channel addressed as /@handle reports its id as the handle, not the
    channel id, and was falling through to OUTPUT_TEMPLATE_PLAYLIST."""
    template = await _add_and_get_template(
        dq_env,
        _channel_extraction("@odin", uploader_id="@odin"),
        "https://www.youtube.com/@odin",
    )

    assert template.startswith("Odin [YT]/")


@pytest.mark.asyncio
async def test_legacy_vanity_channel_url_is_treated_as_a_channel(dq_env):
    """A legacy /c/Name URL reports the vanity name as its id, while
    uploader_id is still the handle."""
    template = await _add_and_get_template(
        dq_env,
        _channel_extraction("Odin", uploader_id="@odin"),
        "https://www.youtube.com/c/Odin",
    )

    assert template.startswith("Odin [YT]/")


@pytest.mark.asyncio
async def test_playlist_with_owner_uploader_id_is_still_a_playlist(dq_env):
    """A real playlist carries its owner's channel_id and uploader_id, but its
    own id matches neither, so it must keep the playlist template."""
    template = await _add_and_get_template(
        dq_env,
        _channel_extraction("PLxyz789", uploader_id="@odin", title="My Playlist"),
        "https://www.youtube.com/playlist?list=PLxyz789",
    )

    assert template.startswith("My Playlist/")


@pytest.mark.asyncio
async def test_add_merges_global_preset_and_override_options(dq_env):
    notifier = AsyncMock()
    dq_env.YTDL_OPTIONS = {"writesubtitles": False, "cookiefile": "/tmp/global.txt"}
    dq_env.YTDL_OPTIONS_PRESETS = {
        "Preset A": {"writesubtitles": True, "proxy": "http://preset-a"},
        "Preset B": {"writesubtitles": False, "ratelimit": 1000},
    }

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid2",
            "title": "Preset Video",
            "url": url,
            "webpage_url": url,
        }

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        result = await dq.add(
            "https://example.com/preset",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
            ytdl_options_presets=["Preset A", "Preset B"],
            ytdl_options_overrides={"proxy": "http://override", "embed_thumbnail": True},
        )

    assert result["status"] == "ok"
    queued = dq.pending.get("https://example.com/preset")
    assert queued.ytdl_opts["cookiefile"] == "/tmp/global.txt"
    assert queued.ytdl_opts["writesubtitles"] is False
    assert queued.ytdl_opts["ratelimit"] == 1000
    assert queued.ytdl_opts["proxy"] == "http://override"
    assert queued.ytdl_opts["embed_thumbnail"] is True


@pytest.mark.asyncio
async def test_extract_info_preset_null_download_archive_overrides_global(dq_env):
    """Preset download_archive:null must apply during extract_info (global archive otherwise wins first)."""
    dq_env.YTDL_OPTIONS = {"download_archive": "/tmp/archive.txt"}
    dq_env.YTDL_OPTIONS_PRESETS = {"NoArchive": {"download_archive": None}}

    captured_params: list = []

    class FakeYoutubeDL:
        def __init__(self, params=None):
            captured_params.append(params)

        def extract_info(self, url, download=False):
            return {
                "_type": "video",
                "id": "vid-archive",
                "title": "Archive Test",
                "url": url,
                "webpage_url": url,
            }

    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    with patch("ytdl.yt_dlp.YoutubeDL", FakeYoutubeDL):
        result = await dq.add(
            "https://example.com/archive-test",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
            ytdl_options_presets=["NoArchive"],
        )

    assert result["status"] == "ok"
    assert len(captured_params) == 1
    extract_params = captured_params[0]
    assert extract_params.get("download_archive") is None
    assert extract_params["extract_flat"] is True
    assert extract_params["noplaylist"] is True


@pytest.mark.asyncio
async def test_extract_info_metube_extract_keys_win_over_preset(dq_env):
    """MeTube's flat-extract settings must not be overridden by presets."""
    dq_env.YTDL_OPTIONS = {}
    dq_env.YTDL_OPTIONS_PRESETS = {
        "TryOverride": {"extract_flat": False, "noplaylist": False},
    }

    captured_params: list = []

    class FakeYoutubeDL:
        def __init__(self, params=None):
            captured_params.append(params)

        def extract_info(self, url, download=False):
            return {
                "_type": "video",
                "id": "vid-flat",
                "title": "Flat Test",
                "url": url,
                "webpage_url": url,
            }

    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    with patch("ytdl.yt_dlp.YoutubeDL", FakeYoutubeDL):
        result = await dq.add(
            "https://example.com/flat-test",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
            ytdl_options_presets=["TryOverride"],
        )

    assert result["status"] == "ok"
    assert captured_params[0]["extract_flat"] is True
    assert captured_params[0]["noplaylist"] is True


def _feed_extract(feed):
    """Patch for __extract_info that returns a playlist/channel feed dict."""

    def fake_extract(self, url, *_args, **_kwargs):
        return copy.deepcopy(feed)

    return fake_extract


_CHANNEL_FEED = {
    "_type": "playlist",
    "id": "UC123",
    "title": "Vanessa - Videos",
    "channel": "Vanessa",
    "channel_id": "UC123",
    "uploader": "Vanessa",
    "extractor": "youtube:tab",
    "extractor_key": "YoutubeTab",
    "webpage_url": "https://example.com/@vanessa/videos",
    "entries": [
        {"id": "v1", "title": "One", "url": "https://example.com/v1",
         "webpage_url": "https://example.com/v1", "_type": "url"},
    ],
}

_PLAYLIST_FEED = {
    "_type": "playlist",
    "id": "PL123",
    "title": "My Playlist",
    "extractor": "generic",
    "extractor_key": "Generic",
    "webpage_url": "https://example.com/playlist?list=PL123",
    "entries": [
        {"id": "v1", "title": "One", "url": "https://example.com/v1",
         "webpage_url": "https://example.com/v1", "_type": "url"},
    ],
}


def _written_files(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            found.append(os.path.relpath(os.path.join(dirpath, f), root))
    return sorted(found)


@pytest.mark.asyncio
async def test_channel_feed_metadata_lands_beside_its_items(dq_env):
    """Issues #660/#1040: the feed-level .info.json follows the same template
    the items use, so it sits in the channel's own folder rather than in
    DOWNLOAD_DIR under yt-dlp's pl_* default name."""
    dq_env.YTDL_OPTIONS = {"writeinfojson": True}
    dq_env.OUTPUT_TEMPLATE_CHANNEL = "%(channel)s/%(title)s.%(ext)s"

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", _feed_extract(_CHANNEL_FEED)), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        result = await dq.add(
            "https://example.com/@vanessa/videos", "video", "auto", "any", "best",
            "", "", 0, auto_start=False,
        )

    assert result["status"] == "ok"
    assert _written_files(dq_env.DOWNLOAD_DIR) == [
        os.path.join("Vanessa", "Vanessa - Videos.info.json")
    ]


@pytest.mark.asyncio
async def test_playlist_feed_metadata_uses_the_playlist_template(dq_env):
    dq_env.YTDL_OPTIONS = {"writeinfojson": True}
    dq_env.OUTPUT_TEMPLATE_PLAYLIST = "%(playlist_title)s/%(title)s.%(ext)s"

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", _feed_extract(_PLAYLIST_FEED)), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        await dq.add(
            "https://example.com/playlist?list=PL123", "video", "auto", "any", "best",
            "", "", 0, auto_start=False,
        )

    assert _written_files(dq_env.DOWNLOAD_DIR) == [
        os.path.join("My Playlist", "My Playlist.info.json")
    ]


@pytest.mark.asyncio
async def test_feed_metadata_honours_custom_folder(dq_env):
    dq_env.YTDL_OPTIONS = {"writeinfojson": True}
    dq_env.OUTPUT_TEMPLATE_PLAYLIST = "%(playlist_title)s/%(title)s.%(ext)s"

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", _feed_extract(_PLAYLIST_FEED)), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        await dq.add(
            "https://example.com/playlist?list=PL123", "video", "auto", "any", "best",
            "Music", "", 0, auto_start=False,
        )

    assert _written_files(dq_env.DOWNLOAD_DIR) == [
        os.path.join("Music", "My Playlist", "My Playlist.info.json")
    ]


@pytest.mark.asyncio
async def test_no_feed_metadata_without_writeinfojson(dq_env):
    """Nothing new appears for users who never asked for these files."""
    dq_env.YTDL_OPTIONS = {}

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", _feed_extract(_PLAYLIST_FEED)), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        await dq.add(
            "https://example.com/playlist?list=PL123", "video", "auto", "any", "best",
            "", "", 0, auto_start=False,
        )

    assert _written_files(dq_env.DOWNLOAD_DIR) == []


@pytest.mark.asyncio
async def test_feed_metadata_can_be_turned_off_by_the_user(dq_env):
    dq_env.YTDL_OPTIONS = {"writeinfojson": True, "allow_playlist_files": False}

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", _feed_extract(_PLAYLIST_FEED)), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()):
        await dq.add(
            "https://example.com/playlist?list=PL123", "video", "auto", "any", "best",
            "", "", 0, auto_start=False,
        )

    assert _written_files(dq_env.DOWNLOAD_DIR) == []


@pytest.mark.asyncio
async def test_feed_metadata_failure_does_not_fail_the_add(dq_env):
    dq_env.YTDL_OPTIONS = {"writeinfojson": True}

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", _feed_extract(_PLAYLIST_FEED)), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", new=AsyncMock()), \
         patch.object(
             DownloadQueue, "_DownloadQueue__write_feed_metadata_sync",
             side_effect=OSError("read-only filesystem"),
         ):
        result = await dq.add(
            "https://example.com/playlist?list=PL123", "video", "auto", "any", "best",
            "", "", 0, auto_start=False,
        )

    assert result["status"] == "ok"
    assert dq.pending.exists("https://example.com/v1")


@pytest.mark.asyncio
async def test_extraction_pass_never_writes_feed_metadata(dq_env):
    """The classification pass must not produce files: it runs before the add is
    known to succeed, and yt-dlp writes playlist files regardless of `download`."""
    dq_env.YTDL_OPTIONS = {"writeinfojson": True, "allow_playlist_files": True}
    captured: list = []

    class FakeYoutubeDL:
        def __init__(self, params=None):
            captured.append(params)

        def extract_info(self, url, download=False):
            return {"_type": "video", "id": "v", "title": "V", "url": url, "webpage_url": url}

    dq = DownloadQueue(dq_env, AsyncMock())
    with patch("ytdl.yt_dlp.YoutubeDL", FakeYoutubeDL):
        await dq.add(
            "https://example.com/watch?v=1", "video", "auto", "any", "best",
            "", "", 0, auto_start=False,
        )

    assert captured[0]["allow_playlist_files"] is False


@pytest.mark.asyncio
async def test_add_sets_clip_bounds_on_download_info(dq_env):
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {
            "_type": "video",
            "id": "vid1",
            "title": "Test Video",
            "url": url,
            "webpage_url": url,
        }

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_extract):
        result = await dq.add(
            "https://example.com/clip",
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=False,
            clip_start=10.0,
            clip_end=99.5,
        )

    assert result["status"] == "ok"
    download = dq.pending.get("https://example.com/clip")
    assert download.info.clip_start == 10.0
    assert download.info.clip_end == 99.5


def _upcoming_entry(url: str, *, release_timestamp: float | None = None) -> dict:
    return {
        "_type": "video",
        "id": "live1",
        "title": "Upcoming Stream",
        "url": url,
        "webpage_url": url,
        "live_status": "is_upcoming",
        "release_timestamp": release_timestamp if release_timestamp is not None else time.time() + 3600,
    }


@pytest.mark.asyncio
async def test_add_upcoming_stream_scheduled_without_starting(dq_env):
    notifier = AsyncMock()
    url = "https://example.com/live-upcoming"
    start_mock = AsyncMock()

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        result = await dq.add_entry(
            _upcoming_entry(url),
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=True,
        )

    assert result["status"] == "ok"
    assert dq.queue.exists(url)
    download = dq.queue.get(url)
    assert download.info.status == "scheduled"
    assert download.info.live_status == "is_upcoming"
    assert download.info.live_release_timestamp is not None
    start_mock.assert_not_called()
    assert url in dq._scheduled_probe_at
    # The "scheduled to start at ..." message must include a UTC offset
    # (a naive datetime's %z would render as an empty string here).
    assert re.search(r"[+-]\d{4}$", download.info.error)


@pytest.mark.asyncio
async def test_probe_scheduled_starts_when_live(dq_env):
    notifier = AsyncMock()
    url = "https://example.com/live-upcoming"
    start_mock = AsyncMock()

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq.add_entry(
            _upcoming_entry(url),
            "video",
            "auto",
            "any",
            "best",
            "",
            "",
            0,
            auto_start=True,
        )

    download = dq.queue.get(url)

    def fake_probe_extract(self, probe_url, ytdl_options_presets=None, ytdl_options_overrides=None):
        assert probe_url == url
        return {
            "_type": "video",
            "id": "live1",
            "title": "Live Now",
            "url": url,
            "webpage_url": url,
            "live_status": "is_live",
            "formats": [{"format_id": "22"}],
        }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", fake_probe_extract), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq._probe_scheduled_download(download)

    assert url not in dq._scheduled_probe_at
    assert download.info.live_status == "is_live"
    assert download.info.status == "pending"
    start_mock.assert_called_once_with(download)


@pytest.mark.asyncio
async def test_import_scheduled_re_registers_monitor(dq_env):
    notifier = AsyncMock()
    url = "https://example.com/live-restart"
    release = time.time() + 7200

    info = DownloadInfo(
        id="live1",
        title="Upcoming Stream",
        url=url,
        quality="best",
        download_type="video",
        codec="auto",
        format="any",
        folder="",
        custom_name_prefix="",
        error=None,
        entry=None,
        playlist_item_limit=0,
        split_by_chapters=False,
        chapter_template="",
        live_status="is_upcoming",
        live_release_timestamp=release,
    )
    info.status = "scheduled"

    dq = DownloadQueue(dq_env, notifier)
    start_mock = AsyncMock()
    with patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq._DownloadQueue__add_download(info, True)

    assert dq.queue.exists(url)
    assert dq.queue.get(url).info.status == "scheduled"
    assert url in dq._scheduled_probe_at
    start_mock.assert_not_called()


@pytest.mark.asyncio
async def test_probe_transient_error_retries_without_failing(dq_env):
    """A single probe failure must not abandon the scheduled stream."""
    import ytdl

    notifier = AsyncMock()
    url = "https://example.com/live-transient"
    start_mock = AsyncMock()

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq.add_entry(
            _upcoming_entry(url),
            "video", "auto", "any", "best", "", "", 0,
            auto_start=True,
        )
    download = dq.queue.get(url)

    def boom(self, *args, **kwargs):
        raise ytdl.yt_dlp.utils.YoutubeDLError("temporary network glitch")

    before = time.time()
    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", boom):
        await dq._probe_scheduled_download(download)

    # Still scheduled, still monitored, probe rescheduled into the future.
    assert download.info.status == "scheduled"
    assert url in dq._scheduled_probe_at
    assert dq._scheduled_probe_at[url] >= before
    assert dq._scheduled_probe_failures[url] == 1
    notifier.completed.assert_not_called()


@pytest.mark.asyncio
async def test_probe_gives_up_after_max_failures(dq_env):
    import ytdl

    notifier = AsyncMock()
    url = "https://example.com/live-dead"
    start_mock = AsyncMock()

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq.add_entry(
            _upcoming_entry(url),
            "video", "auto", "any", "best", "", "", 0,
            auto_start=True,
        )
    download = dq.queue.get(url)

    def boom(self, *args, **kwargs):
        raise ytdl.yt_dlp.utils.YoutubeDLError("stream was deleted")

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", boom):
        for _ in range(ytdl._LIVE_PROBE_MAX_FAILURES):
            await dq._probe_scheduled_download(download)

    assert url not in dq._scheduled_probe_at
    assert not dq.queue.exists(url)
    assert dq.done.exists(url)
    assert download.info.status == "error"
    notifier.completed.assert_awaited()


@pytest.mark.asyncio
async def test_probe_recovers_after_transient_then_starts(dq_env):
    """A transient failure followed by a successful live probe should start the download."""
    import ytdl

    notifier = AsyncMock()
    url = "https://example.com/live-recover"
    start_mock = AsyncMock()

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq.add_entry(
            _upcoming_entry(url),
            "video", "auto", "any", "best", "", "", 0,
            auto_start=True,
        )
    download = dq.queue.get(url)
    # The scheduling placeholder error is set on add.
    assert download.info.error

    def boom(self, *args, **kwargs):
        raise ytdl.yt_dlp.utils.YoutubeDLError("temporary glitch")

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", boom):
        await dq._probe_scheduled_download(download)
    assert dq._scheduled_probe_failures[url] == 1

    def live_now(self, *args, **kwargs):
        return {
            "_type": "video", "id": "live1", "title": "Live Now",
            "url": url, "webpage_url": url, "live_status": "is_live",
            "formats": [{"format_id": "22"}],
        }

    with patch.object(DownloadQueue, "_DownloadQueue__extract_info", live_now), \
         patch.object(DownloadQueue, "_DownloadQueue__start_download", start_mock):
        await dq._probe_scheduled_download(download)

    assert url not in dq._scheduled_probe_at
    assert url not in dq._scheduled_probe_failures
    assert download.info.status == "pending"
    # Placeholder error/msg cleared now that a real download is starting.
    assert download.info.error is None
    assert download.info.msg is None
    start_mock.assert_called_once_with(download)


def test_seconds_until_next_probe_none_when_empty(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    assert dq._seconds_until_next_probe() is None


def test_calc_download_path_allows_subfolder(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    path, err = dq._DownloadQueue__calc_download_path("video", "sub/dir")
    assert err is None
    assert os.path.realpath(path) == os.path.join(os.path.realpath(dq_env.DOWNLOAD_DIR), "sub", "dir")


def test_calc_download_path_rejects_sibling_prefix_escape(dq_env):
    """A folder resolving to a sibling sharing a name prefix must be rejected.

    Regression test: ``startswith`` would have accepted ``../downloads-secret``
    when the base directory is ``.../downloads``.
    """
    notifier = AsyncMock()
    base = os.path.realpath(dq_env.DOWNLOAD_DIR)
    sibling = base + "-secret"
    os.makedirs(sibling, exist_ok=True)
    dq = DownloadQueue(dq_env, notifier)
    escape_folder = os.path.join("..", os.path.basename(sibling), "x")
    path, err = dq._DownloadQueue__calc_download_path("video", escape_folder)
    assert path is None
    assert err is not None and err["status"] == "error"


def test_calc_download_path_rejects_parent_escape(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    path, err = dq._DownloadQueue__calc_download_path("video", "../../etc")
    assert path is None
    assert err is not None and err["status"] == "error"


def test_download_info_to_public_dict_excludes_server_only_fields():
    info = DownloadInfo(
        id="vid1",
        title="Test Video",
        url="https://example.com/watch?v=1",
        quality="best",
        download_type="video",
        codec="auto",
        format="any",
        folder="",
        custom_name_prefix="",
        error=None,
        entry={"id": "vid1", "huge": "x" * 100000},
        playlist_item_limit=0,
        split_by_chapters=False,
        chapter_template="",
    )
    info.subtitle_files = [{"filename": "a.srt", "size": 10}]
    public = info.to_public_dict()
    assert "entry" not in public
    assert "subtitle_files" not in public
    # Client-facing fields are still present.
    assert public["url"] == "https://example.com/watch?v=1"
    assert public["title"] == "Test Video"
    assert public["status"] == "pending"


def _make_download(dq_env, *, download_type="video", status="downloading", filename=None):
    info = DownloadInfo(
        id="id1",
        title="t",
        url="http://example.com/v",
        quality="best",
        download_type=download_type,
        codec="auto",
        format="any",
        folder="",
        custom_name_prefix="",
        error=None,
        entry=None,
        playlist_item_limit=0,
        split_by_chapters=False,
        chapter_template="",
    )
    info.status = status
    info.filename = filename
    info.size = 123 if filename else None
    return Download(
        dq_env.DOWNLOAD_DIR, dq_env.TEMP_DIR, "%(title)s.%(ext)s", "%(title)s.%(ext)s", "best", "any", {}, info
    )


def test_download_close_releases_status_queue(dq_env):
    download = _make_download(dq_env)
    status_queue = MagicMock()
    proc = MagicMock()
    download.status_queue = status_queue
    download.proc = proc

    download.close()

    proc.close.assert_called_once()
    assert download.status_queue is None


def test_download_close_releases_status_queue_without_process(dq_env):
    download = _make_download(dq_env)
    download.status_queue = MagicMock()

    download.close()

    assert download.status_queue is None


def test_download_close_releases_status_queue_when_process_close_fails(dq_env):
    download = _make_download(dq_env)
    download.status_queue = MagicMock()
    download.proc = MagicMock()
    download.proc.close.side_effect = RuntimeError('close failed')

    with pytest.raises(RuntimeError, match='close failed'):
        download.close()

    assert download.status_queue is None


@pytest.mark.asyncio
async def test_post_download_cleanup_clears_filename_on_error(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    download = _make_download(dq_env, status="downloading", filename="../tmp/partial.mp4")
    await dq.queue.put(download)

    await dq._post_download_cleanup(download)

    assert download.info.status == "error"
    assert download.info.filename is None
    assert download.info.size is None


@pytest.mark.asyncio
async def test_post_download_cleanup_keeps_captured_subtitles_on_error(dq_env):
    notifier = AsyncMock()
    dq = DownloadQueue(dq_env, notifier)
    download = _make_download(dq_env, download_type="captions", status="downloading", filename="en.srt")
    download.info.subtitle_files = [{"filename": "en.srt", "size": 42}]
    await dq.queue.put(download)

    await dq._post_download_cleanup(download)

    assert download.info.status == "error"
    assert download.info.filename == "en.srt"


@pytest.mark.asyncio
async def test_clear_skips_deletion_outside_download_directory(dq_env):
    notifier = AsyncMock()
    dq_env.DELETE_FILE_ON_TRASHCAN = True
    dq = DownloadQueue(dq_env, notifier)

    outside_dir = tempfile.mkdtemp()
    outside_file = os.path.join(outside_dir, "outside.txt")
    with open(outside_file, "w") as f:
        f.write("do not delete me")

    # A crafted/legacy relative filename that escapes DOWNLOAD_DIR via '..'.
    escaping_filename = os.path.relpath(outside_file, dq_env.DOWNLOAD_DIR)
    download = _make_download(dq_env, status="finished", filename=escaping_filename)
    await dq.done.put(download)

    await dq.clear([download.info.url])

    assert os.path.exists(outside_file)


# --- CCTV integration -------------------------------------------------------

import cctv as _cctv  # noqa: E402  (used to construct fake CctvStream instances)
import ytdl as _ytdl  # noqa: E402  (used by monkeypatch targets below)


CCTV_EP = 'https://tv.cctv.com/2024/02/21/VIDEAbcdEf0123456789.shtml'
CCTV_SERIES = 'https://tv.cctv.com/lm/xwlb/'


async def test_cctv_episode_url_rewrites_to_generic_with_forced_format(dq_env, monkeypatch):
    """The episode URL is rewritten to the resolved generic m3u8 and the
    forced format selector bypasses the normal height-based choice."""
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                'title': '节目名', 'url': url, 'webpage_url': url}

    resolved = _cctv.CctvStream(
        url='generic:https://dh5.cntv.myhwcdn.cn/asp/hls/2000/M/2000.m3u8',
        forced_format=_cctv.FORCED_FORMAT,
        source='clear-ladder', probed_quality='2000', title='节目名')

    async def fake_resolve(url, quality, *, entry=None, allow_private=False):
        return resolved

    monkeypatch.setattr(_ytdl, 'resolve_episode', fake_resolve)

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        result = await dq.add(
            CCTV_EP, 'video', 'auto', 'any', 'best', '', '', 0, auto_start=False)
    assert result['status'] == 'ok'
    # The download lands in pending under the REWRITTEN url (generic:...)
    dl = dq.pending.get(resolved.url)
    assert dl.info.url == resolved.url
    assert dl.info.forced_format == _cctv.FORCED_FORMAT
    assert dl.format == _cctv.FORCED_FORMAT


async def test_cctv_episode_url_pre_resolves_title_in_outtmpl(dq_env, monkeypatch):
    """The rewritten URL is a bare m3u8 whose generic extraction would
    produce a basename like '2000'. Pre-resolving %(title)s here keeps the
    real episode title in the output filename; %(ext)s is left dynamic."""
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                'title': '《新闻联播》 20240221', 'url': url, 'webpage_url': url}

    async def fake_resolve(url, quality, *, entry=None, allow_private=False):
        return _cctv.CctvStream(
            url='generic:https://dh5.cntv.myhwcdn.cn/asp/hls/2000/M/2000.m3u8',
            forced_format=_cctv.FORCED_FORMAT,
            source='clear-ladder', probed_quality='2000', title='《新闻联播》 20240221')

    monkeypatch.setattr(_ytdl, 'resolve_episode', fake_resolve)

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best', '', '', 0, auto_start=False)

    rewritten = 'generic:https://dh5.cntv.myhwcdn.cn/asp/hls/2000/M/2000.m3u8'
    dl = dq.pending.get(rewritten)
    assert dl.output_template == '《新闻联播》 20240221.%(ext)s'


async def test_cctv_resolve_failure_falls_back_to_original_behavior(dq_env, monkeypatch):
    """If the resolver returns None (any failure), the download must look
    exactly like it would have without the resolver wired in."""
    from dl_formats import get_format
    notifier = AsyncMock()

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                'title': '节目名', 'url': url, 'webpage_url': url}

    async def fake_resolve(url, quality, *, entry=None, allow_private=False):
        return None  # the resolver gave up

    monkeypatch.setattr(_ytdl, 'resolve_episode', fake_resolve)

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best', '', '', 0, auto_start=False)

    dl = dq.pending.get(CCTV_EP)
    assert dl.info.url == CCTV_EP                       # not rewritten
    assert dl.info.forced_format is None
    assert dl.format == get_format('video', 'auto', 'any', 'best')
    assert dl.output_template == '%(title)s.%(ext)s'    # %(title)s still dynamic


async def test_cctv_series_page_expands_to_episodes(dq_env, monkeypatch):
    """A CCTV column/index URL is expanded into its episode URLs; each
    episode goes through add() recursively with the resolver."""
    notifier = AsyncMock()
    episodes = [CCTV_EP, CCTV_EP.replace('AbcdEf', 'BcdeFg')]

    async def fake_expand(url):
        return episodes

    async def fake_resolve(url, quality, *, entry=None, allow_private=False):
        return _cctv.CctvStream(
            url=f'generic:https://x/{url.rsplit("/", 1)[-1]}',
            forced_format=_cctv.FORCED_FORMAT,
            source='clear-ladder', probed_quality='2000', title='t')

    monkeypatch.setattr(_ytdl, 'expand_url', fake_expand)
    monkeypatch.setattr(_ytdl, 'resolve_episode', fake_resolve)

    # for the per-episode add(): __extract_info just echoes the URL back
    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        result = await dq.add(CCTV_SERIES, 'video', 'auto', 'any', 'best',
                              '', '', 0, auto_start=False)

    assert result['status'] == 'ok'
    # both episodes are now in the queue, each rewritten by the resolver
    # and stored under its rewritten (generic:...) URL
    for ep in episodes:
        rewritten = f'generic:https://x/{ep.rsplit("/", 1)[-1]}'
        assert dq.pending.exists(rewritten)
        dl = dq.pending.get(rewritten)
        assert dl.info.url.startswith('generic:')


def test_cctv_old_persisted_record_without_forced_format_loads_cleanly(dq_env):
    """Backward compatibility: records saved before forced_format existed
    must load with forced_format=None and round-trip safely."""
    info = DownloadInfo(
        id='x', title='t', url=CCTV_EP, quality='best', download_type='video',
        codec='auto', format='any', folder='', custom_name_prefix='', error=None,
        entry=None, playlist_item_limit=0, split_by_chapters=False, chapter_template=None,
    )
    # Simulate an old record: drop forced_format if present
    info.__dict__.pop('forced_format', None)
    record = _ytdl._download_info_to_record(info, include_entry=False)
    assert 'forced_format' not in record

    reloaded = _ytdl._download_info_from_record(record)
    assert reloaded.forced_format is None
    # The reload must produce a working download object
    dl = Download(dq_env.DOWNLOAD_DIR, dq_env.TEMP_DIR,
                  dq_env.OUTPUT_TEMPLATE, dq_env.OUTPUT_TEMPLATE_CHAPTER,
                  reloaded.quality, reloaded.format, {}, reloaded,
                  allow_private=False)
    assert dl.format == 'bestvideo+bestaudio/best'


# --- CCTV whole-series (single-episode URL -> full season) -----------------

import cctv_series as _cctv_series  # noqa: E402


def _patch_cctv_resolver(monkeypatch, fake_resolve):
    """Default fake resolver: every episode gets rewritten to a generic m3u8
    URL with forced_format. Tests that want a different per-episode result
    can pass their own."""
    if fake_resolve is None:
        async def fake_resolve(url, quality, *, entry=None, allow_private=False):
            return _cctv.CctvStream(
                url=f'generic:https://x/{url.rsplit("/", 1)[-1]}',
                forced_format=_cctv.FORCED_FORMAT,
                source='clear-ladder', probed_quality='2000', title='t')
    else:
        fake_resolve = fake_resolve
    monkeypatch.setattr(_ytdl, 'resolve_episode', fake_resolve)


async def test_cctv_whole_series_episode_expands_to_siblings(dq_env, monkeypatch):
    """checkbox=true on a CCTV single-episode URL triggers whole-series
    detection; the queue ends up with every returned sibling episode."""
    notifier = AsyncMock()
    sibling_eps = [
        'https://tv.cctv.cn/2024/03/02/VIDE0000000000000000000000000000002.shtml',
        'https://tv.cctv.cn/2024/03/03/VIDE0000000000000000000000000000003.shtml',
        'https://tv.cctv.cn/2024/03/04/VIDE0000000000000000000000000000004.shtml',
    ]

    async def fake_series(url, *, allow_private=False, _fetch=None):
        return _cctv_series.SeriesResult(
            kind=_cctv_series.SeriesKind.SERIES,
            episodes=sibling_eps,
            column_id='TOPC1', source='html-fallback')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', fake_series)
    monkeypatch.setattr(_ytdl, 'expand_url', AsyncMock(return_value=[]))

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        result = await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best',
                              '', '', 0, auto_start=False,
                              download_whole_series=True)

    assert result['status'] == 'ok'
    # Each sibling is queued as its rewritten generic URL; the original
    # episode URL is not queued on its own (it is among the siblings).
    for ep in sibling_eps:
        rewritten = f'generic:https://x/{ep.rsplit("/", 1)[-1]}'
        assert dq.pending.exists(rewritten), f'missing {ep}'


async def test_cctv_whole_series_url_marker_is_equivalent_to_checkbox(dq_env, monkeypatch):
    """?cctv_all=true on the URL is normalised to download_whole_series=True
    without the caller having to pass the flag explicitly."""
    notifier = AsyncMock()
    sibling_eps = [CCTV_EP.replace('AbcdEf', 'BcdeFg')]

    async def fake_series(url, *, allow_private=False, _fetch=None):
        # url carries the marker -- it must be passed through unchanged
        assert 'cctv_all=true' in url
        return _cctv_series.SeriesResult(
            kind=_cctv_series.SeriesKind.SERIES, episodes=sibling_eps,
            column_id='TOPC1', source='column-api')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', fake_series)
    monkeypatch.setattr(_ytdl, 'expand_url', AsyncMock(return_value=[]))

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        result = await dq.add(CCTV_EP + '?cctv_all=true', 'video', 'auto', 'any', 'best',
                              '', '', 0, auto_start=False)  # no kwarg!

    assert result['status'] == 'ok'
    rewritten = f'generic:https://x/{sibling_eps[0].rsplit("/", 1)[-1]}'
    assert dq.pending.exists(rewritten)


async def test_cctv_whole_series_falls_back_to_single_when_kind_is_single(dq_env, monkeypatch):
    """fetch_series_episodes reports SINGLE (API confirmed this is not a
    series): the URL enters the queue as a single download, same as no
    checkbox."""
    notifier = AsyncMock()

    async def fake_series(url, *, allow_private=False, _fetch=None):
        return _cctv_series.SeriesResult(
            kind=_cctv_series.SeriesKind.SINGLE,
            column_id='TOPC1', source='column-api')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', fake_series)

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        result = await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best',
                              '', '', 0, auto_start=False,
                              download_whole_series=True)

    assert result['status'] == 'ok'
    # Single-episode path: rewritten generic URL present, no extra siblings.
    rewritten = f'generic:https://x/{CCTV_EP.rsplit("/", 1)[-1]}'
    assert dq.pending.exists(rewritten)


async def test_cctv_whole_series_swallows_detection_exception(dq_env, monkeypatch):
    """If fetch_series_episodes raises, the URL must still enter the queue
    exactly like an un-checked single-episode submission."""
    notifier = AsyncMock()

    async def exploding_series(url, *, allow_private=False, _fetch=None):
        raise RuntimeError('boom')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', exploding_series)

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        result = await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best',
                              '', '', 0, auto_start=False,
                              download_whole_series=True)

    assert result['status'] == 'ok'
    rewritten = f'generic:https://x/{CCTV_EP.rsplit("/", 1)[-1]}'
    assert dq.pending.exists(rewritten)


async def test_cctv_whole_series_recursion_never_double_calls(dq_env, monkeypatch):
    """When the recursive per-sibling add() runs, download_whole_series is
    forced to False so a sibling cannot trigger another series detection
    API call. The series detector must be called exactly once."""
    notifier = AsyncMock()
    sibling_eps = [
        'https://tv.cctv.cn/2024/03/02/VIDE0000000000000000000000000000002.shtml',
        'https://tv.cctv.cn/2024/03/03/VIDE0000000000000000000000000000003.shtml',
    ]
    call_count = {'n': 0}

    async def fake_series(url, *, allow_private=False, _fetch=None):
        call_count['n'] += 1
        return _cctv_series.SeriesResult(
            kind=_cctv_series.SeriesKind.SERIES, episodes=sibling_eps,
            column_id='TOPC1', source='html-fallback')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', fake_series)
    monkeypatch.setattr(_ytdl, 'expand_url', AsyncMock(return_value=[]))

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best',
                     '', '', 0, auto_start=False,
                     download_whole_series=True)

    # Exactly one detector call (the original) -- never re-entered.
    assert call_count['n'] == 1


async def test_cctv_whole_series_unknown_source_no_expansion(dq_env, monkeypatch):
    """Unknown result (detection simply couldn't tell) -> single-episode
    fallback. The detector must NOT be invoked a second time when the
    recursive add() resolves that single URL."""
    notifier = AsyncMock()

    async def fake_series(url, *, allow_private=False, _fetch=None):
        return _cctv_series.SeriesResult(
            kind=_cctv_series.SeriesKind.UNKNOWN, source='none')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', fake_series)

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        result = await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best',
                              '', '', 0, auto_start=False,
                              download_whole_series=True)

    assert result['status'] == 'ok'
    rewritten = f'generic:https://x/{CCTV_EP.rsplit("/", 1)[-1]}'
    assert dq.pending.exists(rewritten)


async def test_cctv_whole_series_playlist_item_limit_caps_expansion(dq_env, monkeypatch):
    """playlist_item_limit clips the episode list before recursion."""
    notifier = AsyncMock()
    sibling_eps = [
        f'https://tv.cctv.cn/2024/03/0{i+2}/VIDE000000000000000000000000000000{i+2}.shtml'
        for i in range(5)
    ]

    async def fake_series(url, *, allow_private=False, _fetch=None):
        return _cctv_series.SeriesResult(
            kind=_cctv_series.SeriesKind.SERIES, episodes=sibling_eps,
            column_id='TOPC1', source='html-fallback')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', fake_series)
    monkeypatch.setattr(_ytdl, 'expand_url', AsyncMock(return_value=[]))

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        await dq.add(CCTV_EP, 'video', 'auto', 'any', 'best',
                     '', '', 2, auto_start=False,  # limit to 2
                     download_whole_series=True)

    # Only the first two siblings were queued; the other three weren't.
    for ep in sibling_eps[:2]:
        rewritten = f'generic:https://x/{ep.rsplit("/", 1)[-1]}'
        assert dq.pending.exists(rewritten)
    for ep in sibling_eps[2:]:
        rewritten = f'generic:https://x/{ep.rsplit("/", 1)[-1]}'
        assert not dq.pending.exists(rewritten)


async def test_cctv_non_episode_url_ignores_whole_series_flag(dq_env, monkeypatch):
    """download_whole_series=True has no effect on non-CCTV URLs: the
    detector is never called and the regular yt-dlp path runs."""
    notifier = AsyncMock()
    calls = {'n': 0}

    async def fake_series(url, *, allow_private=False, _fetch=None):
        calls['n'] += 1
        return _cctv_series.SeriesResult(
            kind=_cctv_series.SeriesKind.SERIES, episodes=['x'],
            column_id='X', source='column-api')

    _patch_cctv_resolver(monkeypatch, None)
    monkeypatch.setattr(_ytdl, 'fetch_series_episodes', fake_series)
    monkeypatch.setattr(_ytdl, 'expand_url', AsyncMock(return_value=[]))

    yt_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'

    def fake_extract(self, url, *_args, **_kwargs):
        return {'_type': 'video', 'id': 'a' * 32, 'title': 't',
                'url': url, 'webpage_url': url}

    dq = DownloadQueue(dq_env, notifier)
    with patch.object(DownloadQueue, '_DownloadQueue__extract_info', fake_extract):
        await dq.add(yt_url, 'video', 'auto', 'any', 'best',
                     '', '', 0, auto_start=False,
                     download_whole_series=True)

    assert calls['n'] == 0
