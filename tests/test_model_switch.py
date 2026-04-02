import sys
import os
import json
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

# mock config
mock_config = MagicMock()
mock_config.TAVILY_API_KEY = "test-key"
mock_config.SEARCH_MAX_RESULTS = 3
mock_config.MAX_HISTORY = 20
mock_config.MAX_TOKENS = 4096
mock_config.TEMPERATURE = 0.9
mock_config.API_KEY = "dzzi-key"
mock_config.API_BASE = "https://api.dzzi.ai/v1"
mock_config.MODEL = "claude-opus"
mock_config.MINIMAX_KEY = "minimax-key"
mock_config.MINIMAX_BASE = "https://api.minimaxi.com/v1"
mock_config.MINIMAX_MODEL = "MiniMax-M2.7"
mock_config.DEEPSEEK_KEY = "deepseek-key"
mock_config.DEEPSEEK_BASE = "https://api.deepseek.com/v1"
mock_config.DEEPSEEK_MODEL = "deepseek-chat"
sys.modules["config"] = mock_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bot


# ── load / save model state ────────────────────────────────────────────────────

def test_load_model_state_default(tmp_path):
    with patch("bot.MODEL_STATE_PATH", str(tmp_path / "model_state.json")):
        result = bot.load_model_state()
    assert result == "opus"


def test_load_model_state_saved(tmp_path):
    state_file = tmp_path / "model_state.json"
    state_file.write_text(json.dumps({"model_key": "minimax"}))
    with patch("bot.MODEL_STATE_PATH", str(state_file)):
        result = bot.load_model_state()
    assert result == "minimax"


def test_save_model_state(tmp_path):
    state_file = tmp_path / "model_state.json"
    with patch("bot.MODEL_STATE_PATH", str(state_file)):
        bot.save_model_state("deepseek")
    data = json.loads(state_file.read_text())
    assert data["model_key"] == "deepseek"


# ── call_api 使用当前模型配置 ───────────────────────────────────────────────────

import pytest

@pytest.mark.asyncio
async def test_call_api_uses_current_model():
    captured = {}

    async def fake_post(url, headers, json, **kwargs):
        captured["url"] = url
        captured["auth"] = headers.get("Authorization", "")
        captured["model"] = json.get("model", "")
        resp = MagicMock()
        resp.is_success = True
        resp.json.return_value = {"choices": [{"message": {"content": "回复"}}]}
        return resp

    original_key = bot.current_model_key
    try:
        bot.current_model_key = "minimax"
        with patch("httpx.AsyncClient") as mock_client:
            instance = mock_client.return_value.__aenter__.return_value
            instance.post = AsyncMock(side_effect=fake_post)
            result = await bot.call_api([{"role": "user", "content": "hi"}])

        assert "minimax" in captured["url"]
        assert "minimax-key" in captured["auth"]
        assert captured["model"] == "MiniMax-M2.7"
        assert result == "回复"
    finally:
        bot.current_model_key = original_key


@pytest.mark.asyncio
async def test_call_api_uses_deepseek():
    captured = {}

    async def fake_post(url, headers, json, **kwargs):
        captured["url"] = url
        captured["model"] = json.get("model", "")
        resp = MagicMock()
        resp.is_success = True
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        return resp

    original_key = bot.current_model_key
    try:
        bot.current_model_key = "deepseek"
        with patch("httpx.AsyncClient") as mock_client:
            instance = mock_client.return_value.__aenter__.return_value
            instance.post = AsyncMock(side_effect=fake_post)
            await bot.call_api([{"role": "user", "content": "hi"}])

        assert "deepseek" in captured["url"]
        assert captured["model"] == "deepseek-chat"
    finally:
        bot.current_model_key = original_key


# ── _model_keyboard 标记当前模型 ───────────────────────────────────────────────

def test_model_keyboard_marks_current():
    original_key = bot.current_model_key
    try:
        bot.current_model_key = "minimax"
        kb = bot._model_keyboard()
        labels = [row[0].text for row in kb.inline_keyboard]
        checked = [l for l in labels if l.startswith("✓")]
        unchecked = [l for l in labels if not l.startswith("✓")]
        assert len(checked) == 1
        assert "MiniMax" in checked[0]
        assert len(unchecked) == 2
    finally:
        bot.current_model_key = original_key


# ── handle_model_callback 切换模型并持久化 ─────────────────────────────────────

@pytest.mark.asyncio
async def test_handle_model_callback_switches_and_saves(tmp_path):
    state_file = tmp_path / "model_state.json"
    original_key = bot.current_model_key

    try:
        bot.current_model_key = "opus"
        query = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.data = "model:deepseek"

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        with patch("bot.MODEL_STATE_PATH", str(state_file)):
            await bot.handle_model_callback(update, context)

        assert bot.current_model_key == "deepseek"
        saved = json.loads(state_file.read_text())
        assert saved["model_key"] == "deepseek"
        query.edit_message_text.assert_called_once()
        assert "DeepSeek" in query.edit_message_text.call_args[0][0]
    finally:
        bot.current_model_key = original_key


@pytest.mark.asyncio
async def test_handle_model_callback_invalid_key():
    original_key = bot.current_model_key
    try:
        bot.current_model_key = "opus"
        query = MagicMock()
        query.answer = AsyncMock()
        query.edit_message_text = AsyncMock()
        query.data = "model:nonexistent"

        update = MagicMock()
        update.callback_query = query
        context = MagicMock()

        await bot.handle_model_callback(update, context)

        assert bot.current_model_key == "opus"
        query.edit_message_text.assert_not_called()
    finally:
        bot.current_model_key = original_key
