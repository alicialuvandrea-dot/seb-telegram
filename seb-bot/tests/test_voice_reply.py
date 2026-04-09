# seb-bot/tests/test_voice_reply.py
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

# mock config，阻止真实 config.py 被导入
mock_config = MagicMock()
mock_config.MINIMAX_API_KEY = "test-key"
mock_config.MINIMAX_VOICE_MAP = {
    "default": "Japanese_GentleButler",
    "whisper": "whisper_man",
    "english": "English_DecentYoungMan",
}
mock_config.MAX_HISTORY = 20
mock_config.MAX_TOKENS = 4096
mock_config.TEMPERATURE = 0.9
mock_config.API_BASE = "https://api.example.com/v1"
mock_config.API_KEY = "test-api-key"
mock_config.TAVILY_API_KEY = "test-tavily"
mock_config.SEARCH_MAX_RESULTS = 3
mock_config.SAKURA_CHAT_ID = "12345"
mock_config.SENTINEL_PORT = 8765
mock_config.SENTINEL_TOKEN = "test"
mock_config.MEMORIES_LIMIT = 50
mock_config.PROACTIVE_HOURS = 5
mock_config.WEBSITE_URL = "https://example.com"
mock_config.WEBSITE_SECRET = "secret"
mock_config.GROK_KEY = "test"
mock_config.GROK_BASE = "https://api.example.com/v1"
mock_config.GROK_MODEL = "test-model"
mock_config.GROK_MAX_TOKENS = 1024
mock_config.IMGHOST_DIR = "/tmp"
mock_config.IMGHOST_URL = "https://example.com"
sys.modules["config"] = mock_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.mark.asyncio
async def test_call_tts_returns_decoded_bytes():
    """call_tts 应解码 hex 音频并返回 bytes"""
    hex_audio = bytes([0xFF, 0xFB, 0x90, 0x00]).hex()
    mock_resp_data = {
        "base_resp": {"status_code": 0, "status_msg": "success"},
        "data": {"audio": hex_audio},
    }
    mock_response = MagicMock()
    mock_response.json.return_value = mock_resp_data

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )
        from bot import call_tts
        result = await call_tts("こんにちは", "happy")

    assert result == bytes.fromhex(hex_audio)


@pytest.mark.asyncio
async def test_call_tts_raises_on_api_error():
    """call_tts 当 status_code 非 0 时应抛 RuntimeError"""
    mock_resp_data = {
        "base_resp": {"status_code": 1002, "status_msg": "invalid voice_id"},
        "data": {},
    }
    mock_response = MagicMock()
    mock_response.json.return_value = mock_resp_data

    with patch("httpx.AsyncClient") as MockClient:
        MockClient.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )
        from bot import call_tts
        with pytest.raises(RuntimeError, match="MiniMax TTS error"):
            await call_tts("テスト", "neutral")


def test_mp3_to_ogg_returns_bytes():
    """mp3_to_ogg 应将 MP3 bytes 转换为 OGG bytes"""
    fake_ogg = b"fake_ogg_content"

    mock_audio = MagicMock()
    mock_audio.export = MagicMock(
        side_effect=lambda buf, **kwargs: buf.write(fake_ogg)
    )

    with patch("bot.AudioSegment") as MockSeg:
        MockSeg.from_file.return_value = mock_audio
        from bot import mp3_to_ogg
        result = mp3_to_ogg(b"fake_mp3")

    assert result == fake_ogg
    MockSeg.from_file.assert_called_once()
    mock_audio.export.assert_called_once()
    _, kwargs = mock_audio.export.call_args
    assert kwargs.get("format") == "ogg"
    assert kwargs.get("codec") == "libopus"


@pytest.mark.asyncio
async def test_exec_action_voice_reply_sends_voice_and_spoiler():
    """voice_reply 成功时应发语音气泡和中文折叠剧透"""
    fake_mp3 = b"mp3_data"
    fake_ogg = b"ogg_data"
    mock_bot = MagicMock()
    mock_bot.send_voice = AsyncMock()
    mock_bot.send_message = AsyncMock()

    with patch("bot.call_tts", new=AsyncMock(return_value=fake_mp3)), \
         patch("bot.mp3_to_ogg", return_value=fake_ogg):
        from bot import exec_action
        await exec_action(
            "voice_reply",
            {"text": "こんにちは", "zh": "你好", "voice": "default", "emotion": "happy"},
            chat_id=12345,
            bot=mock_bot,
        )

    mock_bot.send_voice.assert_called_once()
    voice_call_args = mock_bot.send_voice.call_args
    assert voice_call_args[0][0] == 12345

    mock_bot.send_message.assert_called_once()
    msg_args = mock_bot.send_message.call_args
    assert "你好" in msg_args[0][1]


@pytest.mark.asyncio
async def test_exec_action_voice_reply_fallback_on_tts_error():
    """TTS 失败时应静默降级，发中文纯文本"""
    mock_bot = MagicMock()
    mock_bot.send_voice = AsyncMock()
    mock_bot.send_message = AsyncMock()

    with patch("bot.call_tts", new=AsyncMock(side_effect=RuntimeError("TTS failed"))):
        from bot import exec_action
        await exec_action(
            "voice_reply",
            {"text": "こんにちは", "zh": "你好", "voice": "default", "emotion": "happy"},
            chat_id=12345,
            bot=mock_bot,
        )

    mock_bot.send_voice.assert_not_called()
    mock_bot.send_message.assert_called_once_with(12345, "你好")


@pytest.mark.asyncio
async def test_exec_action_voice_reply_no_bot_skips_silently():
    """bot 参数为 None 时不应抛出，静默跳过"""
    from bot import exec_action
    await exec_action(
        "voice_reply",
        {"text": "こんにちは", "zh": "你好", "voice": "default", "emotion": "happy"},
    )


@pytest.mark.asyncio
async def test_do_reply_skips_text_send_on_voice_reply():
    """有 voice_reply action 时 do_reply 不应发主文字"""
    raw_reply = '<seb_action type="voice_reply">{"text": "ありがとう", "zh": "谢谢你", "voice": "default", "emotion": "happy"}</seb_action>'

    with patch("bot.call_api", new=AsyncMock(return_value=raw_reply)), \
         patch("bot.exec_action", new=AsyncMock(return_value="ok")), \
         patch("bot.fetch_memories", new=AsyncMock(return_value=[])), \
         patch("bot.fetch_plans", new=AsyncMock(return_value=[])):
        from bot import do_reply
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()
        context.bot.send_voice = AsyncMock()
        context.bot.send_message = AsyncMock()

        await do_reply(
            chat_id=12345,
            api_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            history_entry={"role": "user", "content": "hi"},
            update=update,
            context=context,
        )

    update.message.reply_text.assert_not_called()


@pytest.mark.asyncio
async def test_do_reply_sends_text_when_no_voice_reply():
    """没有 voice_reply action 时 do_reply 正常发主文字"""
    with patch("bot.call_api", new=AsyncMock(return_value="普通回复文本")), \
         patch("bot.fetch_memories", new=AsyncMock(return_value=[])), \
         patch("bot.fetch_plans", new=AsyncMock(return_value=[])):
        from bot import do_reply
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await do_reply(
            chat_id=12345,
            api_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            history_entry={"role": "user", "content": "hi"},
            update=update,
            context=context,
        )

    update.message.reply_text.assert_called()
