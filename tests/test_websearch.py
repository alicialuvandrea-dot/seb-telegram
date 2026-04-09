import sys
import os
import httpx
from unittest.mock import MagicMock

# 用 mock config 阻止真实 config.py 被导入
mock_config = MagicMock()
mock_config.TAVILY_API_KEY = "test-key"
mock_config.SEARCH_MAX_RESULTS = 3
mock_config.MAX_HISTORY = 20
mock_config.MINIMAX_API_KEY = "test-minimax-key"
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
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_web_search_formats_results():
    """MiniMax 返回正常结果"""
    mock_minimax_data = {
        "organic": [
            {"title": "标题一", "link": "https://example.com/1", "snippet": "摘要内容一"},
            {"title": "标题二", "link": "https://example.com/2", "snippet": "摘要内容二"},
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = lambda: mock_minimax_data

    async def mock_post(*args, **kwargs):
        return mock_resp

    with patch("bot.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.post = mock_post
        from bot import web_search
        result = await web_search("测试关键词")

    assert "【搜索结果：测试关键词】" in result
    assert "标题一" in result
    assert "https://example.com/1" in result
    assert "摘要内容一" in result
    assert "标题二" in result


@pytest.mark.asyncio
async def test_web_search_empty_results():
    """MiniMax 返回空结果"""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = lambda: {"organic": []}

    async def mock_post(*args, **kwargs):
        return mock_resp

    with patch("bot.httpx.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.__aenter__ = AsyncMock(return_value=instance)
        instance.__aexit__ = AsyncMock(return_value=None)
        instance.post = mock_post
        from bot import web_search
        result = await web_search("无结果查询")

    assert "【搜索结果：无结果查询】" in result
    assert "未找到相关结果" in result


@pytest.mark.asyncio
async def test_keyword_trigger_injects_search_results():
    """关键词触发时，search_results 应被注入 system prompt"""
    injected_systems = []

    async def fake_call_api(messages, model_key="haiku"):
        injected_systems.append(messages[0]["content"])
        return "搜到了，是这样的"

    mock_search_result = "【搜索结果：明天天气】\n1. 晴 | weather.com\n   明天晴转多云"

    with patch("bot.call_api", side_effect=fake_call_api), \
         patch("bot.web_search", new=AsyncMock(return_value=mock_search_result)), \
         patch("bot.fetch_memories", new=AsyncMock(return_value=[])), \
         patch("bot.fetch_plans", new=AsyncMock(return_value=[])):

        from bot import handle_message
        update = MagicMock()
        update.effective_chat.id = 12345
        update.message.text = "查查明天天气"
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await handle_message(update, context)

    assert any("搜索结果" in s for s in injected_systems), \
        "搜索结果未注入 system prompt"


@pytest.mark.asyncio
async def test_search_command_calls_web_search():
    mock_search_result = "【搜索结果：崩铁新角色】\n1. 结果 | url\n   内容"

    async def fake_call_api(messages, model_key="haiku"):
        return "根据搜索结果，崩铁新角色是..."

    with patch("bot.call_api", side_effect=fake_call_api), \
         patch("bot.web_search", new=AsyncMock(return_value=mock_search_result)), \
         patch("bot.fetch_memories", new=AsyncMock(return_value=[])), \
         patch("bot.fetch_plans", new=AsyncMock(return_value=[])):

        from bot import handle_search
        update = MagicMock()
        update.effective_chat.id = 12345
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.args = ["崩铁", "新角色"]
        context.bot.send_chat_action = AsyncMock()

        await handle_search(update, context)

    update.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_search_command_no_args():
    from bot import handle_search
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.args = []

    await handle_search(update, context)

    update.message.reply_text.assert_called_once_with("用法：/search <关键词>")


@pytest.mark.asyncio
async def test_do_reply_two_pass_on_web_search_action():
    """do_reply 检测到 web_search action 时应执行二次调用"""
    call_count = 0
    injected_second_messages = []

    async def fake_call_api(messages, model_key="haiku"):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '<seb_action type="web_search">{"query": "今日黄金价格"}</seb_action>'
        injected_second_messages.extend(messages)
        return "今天黄金价格是每克XXX元"

    mock_search_result = "【搜索结果：今日黄金价格】\n1. 黄金 | gold.com\n   每克600元"

    with patch("bot.call_api", side_effect=fake_call_api), \
         patch("bot.web_search", new=AsyncMock(return_value=mock_search_result)):

        from bot import do_reply
        update = MagicMock()
        update.message.reply_text = AsyncMock()
        context = MagicMock()
        context.bot.send_chat_action = AsyncMock()

        await do_reply(
            chat_id=99999,
            api_messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "今日黄金价格"}],
            history_entry={"role": "user", "content": "今日黄金价格"},
            update=update,
            context=context,
        )

    assert call_count == 2, f"期望 2 次 API 调用，实际 {call_count} 次"
    second_contents = [m["content"] for m in injected_second_messages]
    assert any("搜索结果" in c for c in second_contents), "第二次调用未注入搜索结果"
    reply_text = update.message.reply_text.call_args[0][0]
    assert "黄金" in reply_text
