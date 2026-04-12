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
