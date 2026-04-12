import sys
import os
import asyncio
from unittest.mock import MagicMock

mock_config = MagicMock()
mock_config.RVC_SERVICE_URL = "https://test.trycloudflare.com"
mock_config.RVC_DEFAULT_MODEL = "Sebastian"
mock_config.MAX_HISTORY = 20
mock_config.MINIMAX_API_KEY = "test"
sys.modules["config"] = mock_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot import COVER_PATTERN, extract_cover_url


def test_cover_pattern_matches_翻唱():
    assert COVER_PATTERN.search("帮我翻唱这首歌")


def test_cover_pattern_matches_你来唱():
    assert COVER_PATTERN.search("你来唱一下")


def test_cover_pattern_matches_想听你唱():
    assert COVER_PATTERN.search("我想听你唱这首")


def test_cover_pattern_matches_唱一下():
    assert COVER_PATTERN.search("唱一下这个 https://youtu.be/xxx")


def test_cover_pattern_no_match():
    assert not COVER_PATTERN.search("今天天气真好")


def test_extract_cover_url_youtube():
    url = extract_cover_url("翻唱一下 https://youtu.be/dQw4w9WgXcQ 谢谢")
    assert url == "https://youtu.be/dQw4w9WgXcQ"


def test_extract_cover_url_youtube_full():
    url = extract_cover_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ 帮我唱")
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_extract_cover_url_bilibili():
    url = extract_cover_url("用你的声音唱 https://www.bilibili.com/video/BV1xx411c7mD")
    assert url == "https://www.bilibili.com/video/BV1xx411c7mD"


def test_extract_cover_url_no_url():
    assert extract_cover_url("翻唱一下但没有链接") is None


def test_extract_cover_url_non_audio_url():
    assert extract_cover_url("翻唱 https://google.com") is None


import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_process_cover_sends_audio_on_success():
    fake_ogg = b"OggS" + b"\x00" * 100

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.is_success = True
    mock_resp.content = fake_ogg

    async def mock_post(*args, **kwargs):
        return mock_resp

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"fake_mp3_bytes", b""))

    with patch("bot.httpx.AsyncClient") as MockClient, \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_subproc:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.post = mock_post

        bot_mock = MagicMock()
        bot_mock.send_audio = AsyncMock()

        from bot import process_cover
        await process_cover(
            url="https://youtu.be/test",
            model_name="Sebastian",
            chat_id=12345,
            bot=bot_mock,
        )

    bot_mock.send_audio.assert_called_once()
    assert bot_mock.send_audio.call_args.kwargs["chat_id"] == 12345


@pytest.mark.asyncio
async def test_process_cover_notifies_on_rvc_offline():
    import httpx as real_httpx

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.communicate = AsyncMock(return_value=(b"fake_mp3_bytes", b""))

    with patch("bot.httpx.AsyncClient") as MockClient, \
         patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.post = AsyncMock(side_effect=real_httpx.ConnectError("refused"))

        bot_mock = MagicMock()
        bot_mock.send_message = AsyncMock()

        from bot import process_cover
        await process_cover(
            url="https://youtu.be/test",
            model_name="Sebastian",
            chat_id=12345,
            bot=bot_mock,
        )

    bot_mock.send_message.assert_called_once()
    msg = bot_mock.send_message.call_args.kwargs["text"]
    assert "GPU" in msg or "没开机" in msg


@pytest.mark.asyncio
async def test_process_cover_notifies_on_download_failure():
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        bot_mock = MagicMock()
        bot_mock.send_message = AsyncMock()

        from bot import process_cover
        await process_cover(
            url="https://youtu.be/test",
            model_name="Sebastian",
            chat_id=12345,
            bot=bot_mock,
        )

    bot_mock.send_message.assert_called_once()
    msg = bot_mock.send_message.call_args.kwargs["text"]
    assert "下载" in msg or "链接" in msg


@pytest.mark.asyncio
async def test_handle_message_triggers_cover_not_ai():
    with patch("bot.asyncio.create_task") as mock_task, \
         patch("bot.process_cover", new_callable=lambda: lambda *a, **kw: None):

        mock_task.return_value = None

        from bot import handle_message
        update = MagicMock()
        update.effective_chat.id = 12345
        update.message.text = "翻唱一下 https://youtu.be/dQw4w9WgXcQ"
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await handle_message(update, context)

    update.message.reply_text.assert_called_once()
    reply = update.message.reply_text.call_args[0][0]
    assert "⏳" in reply
    mock_task.assert_called_once()
