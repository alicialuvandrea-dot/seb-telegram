import sys
import os
from unittest.mock import MagicMock

# 用 mock config 阻止真实 config.py 被导入
mock_config = MagicMock()
mock_config.TAVILY_API_KEY = "test-key"
mock_config.SEARCH_MAX_RESULTS = 3
sys.modules["config"] = mock_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot import extract_search_query


def test_extract_with_sou_yi_xia():
    assert extract_search_query("帮我搜一下最新iPhone价格") == "帮我最新iPhone价格"


def test_extract_with_cha_cha():
    assert extract_search_query("查查明天天气") == "明天天气"


def test_extract_no_keyword():
    assert extract_search_query("你好啊") is None


def test_extract_only_keyword():
    result = extract_search_query("搜一下")
    # 只有关键词时返回原文本（去掉关键词后为空，fallback 到原文）
    assert result == "搜一下"


import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_web_search_formats_results():
    mock_response = {
        "results": [
            {"title": "标题一", "url": "https://example.com/1", "content": "摘要内容一"},
            {"title": "标题二", "url": "https://example.com/2", "content": "摘要内容二"},
        ]
    }
    with patch("bot.AsyncTavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search = AsyncMock(return_value=mock_response)
        from bot import web_search
        result = await web_search("测试关键词")

    assert "【搜索结果：测试关键词】" in result
    assert "标题一" in result
    assert "https://example.com/1" in result
    assert "摘要内容一" in result
    assert "标题二" in result


@pytest.mark.asyncio
async def test_web_search_empty_results():
    with patch("bot.AsyncTavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search = AsyncMock(return_value={"results": []})
        from bot import web_search
        result = await web_search("无结果查询")

    assert "【搜索结果：无结果查询】" in result
    assert "未找到相关结果" in result
