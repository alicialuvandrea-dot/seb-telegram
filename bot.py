import re
import json
import random
import sys
import os
import uuid

import asyncio
import subprocess
from datetime import datetime, date as _date, timedelta, timezone
from collections import defaultdict

import httpx
from io import BytesIO
from pydub import AudioSegment
MINIMAX_SEARCH_HOST = "https://api.minimaxi.com"
from aiohttp import web
from notion_client import AsyncClient as NotionClient
from telegram import Update
from telegram.error import Conflict
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

import config



SEARCH_KEYWORDS = ["搜一下", "查一下", "帮我查", "查查", "找一下"]

PERIOD_LOG_PATTERN = re.compile(
    r"姨妈|大姨妈|例假|生理期|月经|经期|痛经|MC(?!\w)|"
    r"来事|来亲戚|来朋友|好朋友来|老朋友来|"
    r"坠胀|暖宝宝|点状出血|小腹.{0,3}(疼|痛)|量(多|少|适中)|干净了",
    re.IGNORECASE,
)

PERIOD_PREDICT_PATTERN = re.compile(
    r"下次.{0,6}来|什么时候来|几号来|周期|"
    r"下次.{0,6}(姨妈|例假|月经)|大概什么时候|"
    r"还有.{0,4}天来|预测.{0,4}(来|月经|例假|姨妈)|大姨妈.{0,6}什么时候",
    re.IGNORECASE,
)

TECH_TOPIC_PATTERN = re.compile(
    r"代码|脚本|部署|bug|报错|配置|服务器|数据库|"
    r"API|函数|变量|测试|编程|开发|VPS|Supabase|Notion|"
    r"计划|待办|项目|进度|任务|plan|idea|想法|构想|"
    r"命令|终端|terminal|ssh|docker|git|pip|npm",
    re.IGNORECASE,
)

COVER_PATTERN = re.compile(
    r"翻唱|帮我唱|你来唱|我想听你唱|唱一下|唱这首|"
    r"cover|ai唱|用.*唱",
    re.IGNORECASE,
)

COVER_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com|youtu\.be|"
    r"bilibili\.com|soundcloud\.com|music\.163\.com)\S+"
)


def extract_cover_url(text: str) -> str | None:
    m = COVER_URL_PATTERN.search(text)
    return m.group(0) if m else None

# ── 模型配置 ──────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "label": "Claude Sonnet 4.6",
    "model": config.MODEL,
    "base":  config.API_BASE,
    "key":   config.API_KEY,
}

# ── Obsidian MCP 客户端 ──────────────────────────────────────────────────────
# 通过 VPS 上的 supergateway Streamable HTTP 端点读写 vault
# stateless 模式，每次请求独立，无需维护 session

def _parse_mcp_sse(text: str) -> dict | None:
    """从 SSE 响应中提取 JSON-RPC result。"""
    for line in text.split("\n"):
        if line.startswith("data: "):
            return json.loads(line[6:])
    return None

async def _mcp_call(method: str, params: dict | None = None) -> dict | None:
    """向 Obsidian MCP server 发送 JSON-RPC 请求。"""
    try:
        async with httpx.AsyncClient(timeout=20) as http:
            res = await http.post(
                config.OBSIDIAN_MCP_URL,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params or {},
                    "id": 1,
                },
            )
            return _parse_mcp_sse(res.text)
    except Exception as e:
        print(f"[obsidian-mcp] {method} error: {e}")
        return None

async def obsidian_read(path: str) -> str | None:
    """读一篇 Obsidian 笔记全文。失败返回 None。"""
    r = await _mcp_call("tools/call", {
        "name": "read_note",
        "arguments": {"path": path},
    })
    try:
        content = r.get("result", {}).get("content", [{}])
        return content[0].get("text")
    except Exception:
        return None

async def obsidian_load_prompts() -> dict[str, str]:
    """加载 Obsidian 中的 prompt 片段，失败返回空串。
    并发请求减少延迟。
    """
    tasks = {
        "sakura": obsidian_read("core/sakura.md"),
        "personality": obsidian_read("reference/seb-personality-prompt.md"),
        "us": obsidian_read("core/us.md"),
    }
    results = {}
    for key, task in tasks.items():
        results[key] = await task
    return results



def extract_search_query(text: str) -> str | None:
    for kw in SEARCH_KEYWORDS:
        if kw in text:
            query = text.replace(kw, "", 1).strip()
            return query if query else text
    return None


async def web_search(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{MINIMAX_SEARCH_HOST}/v1/coding_plan/search",
                json={"q": query},
                headers={
                    "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
                    "Content-Type": "application/json",
                    "MM-API-Source": "Minimax-MCP",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("organic", [])
            if not results:
                return f"【搜索结果：{query}】\n未找到相关结果"
            lines = [f"【搜索结果：{query}】"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r.get('title', '')} | {r.get('link', '')}")
                snippet = r.get('snippet', '')[:300]
                if snippet:
                    lines.append(f"   {snippet}")
            return "\n".join(lines)
    except Exception as e:
        return f"【搜索失败：{e}】"


# ── Notion 页面别名 ────────────────────────────────────────────────────────────
app_ref = None

# 发帖冷却（秒），防刷屏
_last_web_post_time: float = 0.0
_POST_COOLDOWN_SECS: int = 7200  # 2 小时

NOTION_ALIASES = {
    "mem":    "31b85af6-2183-816f-9b48-e2a3096c8043",  # 记忆档案
    "core":   "31d85af6-2183-812d-89b0-f1075c260151",  # 核心记忆
    "py":     "31e85af6-2183-814b-b181-dbb8597d1c41",  # Python 画廊
}

def strip_thinking(text: str) -> str:
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<thinking>[\s\S]*?</thinking>', '', text, flags=re.IGNORECASE)
    return text.strip()


def strip_model_tag(text: str) -> str:
    """去掉末尾的模型戳标记，前端不展示给用户。"""
    # 匹配 {MODEL_TAG:模型名} 格式的尾行
    text = re.sub(r'\n*\{MODEL_TAG:[^}]+\}\n?$', '', text)
    return text.strip()


def clean_for_display(text: str) -> str:
    """发送前最终清理：移除所有标签残留、状态提示、JSON 行等非对话内容。"""
    # 残留的功能标签（正常情况已被上游处理，这里兜底）
    text = re.sub(r'<seb_action[^>]*>[\s\S]*?</seb_action>', '', text)
    text = re.sub(r'<#[\d.]+#>', '', text)
    # 其他残留 XML/HTML 标签
    text = re.sub(r'<[a-zA-Z_][^>]{0,100}>', '', text)
    text = re.sub(r'</[a-zA-Z_][^>]{0,50}>', '', text)
    # 整行是括号包裹的状态/UI 提示
    text = re.sub(
        r'^[（(][^）)]{0,60}(?:送达|已读|正在输入|输入中|已发送|typing)[^）)]{0,60}[）)]\s*$',
        '', text, flags=re.MULTILINE,
    )
    # 整行是裸 JSON 对象
    text = re.sub(r'^\s*\{[\s\S]{0,500}\}\s*$', '', text, flags=re.MULTILINE)
    # 清理多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_tech_reply(user_text: str, reply_text: str) -> bool:
    if "```" in reply_text:
        return True
    return bool(TECH_TOPIC_PATTERN.search(user_text))


def md_to_tg_html(text: str) -> str:
    parts = re.split(r'(```\w*\n.*?```)', text, flags=re.DOTALL)
    result = []
    for part in parts:
        if part.startswith('```'):
            m = re.match(r'```(\w*)\n(.*?)```', part, re.DOTALL)
            if m:
                lang = m.group(1)
                code = m.group(2).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                if lang:
                    result.append(f'<pre><code class="language-{lang}">{code}</code></pre>')
                else:
                    result.append(f'<pre>{code}</pre>')
            else:
                result.append(part.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        else:
            inline_parts = re.split(r'(`[^`]+`)', part)
            processed = []
            for ip in inline_parts:
                if ip.startswith('`') and ip.endswith('`'):
                    code = ip[1:-1].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    processed.append(f'<code>{code}</code>')
                else:
                    ip = ip.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    ip = re.sub(r'^#{1,6}\s+(.+)$', r'<b>\1</b>', ip, flags=re.MULTILINE)
                    ip = re.sub(r'^[-*]\s+', '• ', ip, flags=re.MULTILINE)
                    ip = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ip)
                    ip = re.sub(r'\*([^*\n]+?)\*', r'<i>\1</i>', ip)
                    ip = re.sub(r'_([^_\n]+?)_', r'<i>\1</i>', ip)
                    processed.append(ip)
            result.append(''.join(processed))
    return ''.join(result)


async def call_api(messages: list) -> str:
    async with httpx.AsyncClient(timeout=60) as http:
        res = await http.post(
            f"{MODEL_CONFIG['base']}/chat/completions",
            headers={
                "Authorization": f"Bearer {MODEL_CONFIG['key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_CONFIG["model"],
                "messages": messages,
                "max_tokens": config.MAX_TOKENS,
                "temperature": config.TEMPERATURE,
            }
        )
        data = res.json()
        if not res.is_success:
            raise Exception(data.get("error", {}).get("message", res.text))
        content = data["choices"][0]["message"]["content"] or ""
        content = content.replace("\\n", "\n").replace("\\r", "\r")
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        return strip_thinking(content)


async def transcribe_voice(ogg_bytes: bytes) -> str:
    """把 Telegram 语音（OGG OPUS）发给 Groq Whisper，返回识别文字。"""
    async with httpx.AsyncClient(timeout=60) as http:
        res = await http.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {config.GROQ_STT_KEY}"},
            data={"model": config.GROQ_STT_MODEL, "language": "zh"},
            files={"file": ("voice.ogg", ogg_bytes, "audio/ogg")},
        )
        res.raise_for_status()
        return res.json().get("text", "").strip()


async def call_tts(text: str, emotion: str = "neutral", voice_id: str | None = None) -> bytes:
    resolved_voice_id = voice_id or config.MINIMAX_VOICE_MAP.get("default", "Japanese_GentleButler")
    async with httpx.AsyncClient(timeout=30) as http:
        res = await http.post(
            "https://api.minimaxi.com/v1/t2a_v2",
            headers={
                "Authorization": f"Bearer {config.MINIMAX_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "speech-2.8-hd",
                "text": text,
                "stream": False,
                "voice_setting": {
                    "voice_id": resolved_voice_id,
                    "speed": 1,
                    "vol": 1,
                    "pitch": 0,
                    "emotion": emotion,
                },
                "audio_setting": {
                    "sample_rate": 32000,
                    "bitrate": 128000,
                    "format": "mp3",
                    "channel": 1,
                },
            },
        )
        data = res.json()
        status = data.get("base_resp", {}).get("status_code", -1)
        if status != 0:
            msg = data.get("base_resp", {}).get("status_msg", "unknown")
            raise RuntimeError(f"MiniMax TTS error: {msg}")
        return bytes.fromhex(data["data"]["audio"])



async def process_cover(url: str, model_name: str, chat_id: int, bot) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", "-x", "--audio-format", "mp3",
            "-o", "-", "--no-playlist", url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        audio_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            await bot.send_message(
                chat_id=chat_id,
                text="这个链接下载不了，换一个试试 🌸",
            )
            return

        async with httpx.AsyncClient(timeout=300) as http:
            resp = await http.post(
                f"{config.RVC_SERVICE_URL}/cover",
                files={"audio": ("input.mp3", audio_bytes, "audio/mpeg")},
                data={"model_name": model_name, "pitch": "0"},
            )

        if resp.status_code == 503:
            await bot.send_message(
                chat_id=chat_id,
                text="模型还没准备好，等我训练完再来 🌸",
            )
            return

        if not resp.is_success:
            await bot.send_message(
                chat_id=chat_id,
                text=f"翻唱失败了：{resp.text[:100]} 🌸",
            )
            return

        await bot.send_audio(
            chat_id=chat_id,
            audio=resp.content,
            filename="cover.ogg",
        )

    except httpx.ConnectError:
        await bot.send_message(
            chat_id=chat_id,
            text="GPU 服务没开机，晚点再试 🌸",
        )
    except asyncio.TimeoutError:
        await bot.send_message(
            chat_id=chat_id,
            text="处理超时了，可能歌太长，再试一次 🌸",
        )
    except Exception as e:
        await bot.send_message(
            chat_id=chat_id,
            text=f"翻唱出错了：{str(e)[:80]} 🌸",
        )


def mp3_to_ogg(mp3_bytes: bytes) -> bytes:
    audio = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
    buf = BytesIO()
    audio.export(buf, format="ogg", codec="libopus")
    return buf.getvalue()


histories: dict[int, list] = defaultdict(list)


# ── Supabase ───────────────────────────────────────────────────────────────────
async def fetch_memories() -> list:
    async with httpx.AsyncClient() as http:
        res = await http.get(
            f"{config.SUPABASE_URL}/rest/v1/memories",
            params={"order": "weight.desc,when.desc", "limit": str(config.MEMORIES_LIMIT), "select": "*"},
            headers={"apikey": config.SUPABASE_KEY, "Authorization": f"Bearer {config.SUPABASE_KEY}"}
        )
        return res.json() if res.status_code == 200 else []


async def fetch_period_records() -> list:
    async with httpx.AsyncClient() as http:
        res = await http.get(
            f"{config.SUPABASE_URL}/rest/v1/period_records",
            params={"order": "date.asc", "select": "*"},
            headers={"apikey": config.SUPABASE_KEY, "Authorization": f"Bearer {config.SUPABASE_KEY}"}
        )
        return res.json() if res.status_code == 200 else []


async def fetch_plans() -> list:
    async with httpx.AsyncClient() as http:
        res = await http.get(
            f"{config.SUPABASE_URL}/rest/v1/plans",
            params={
                "status": "neq.done",
                "order": "priority.desc,created_at.desc",
                "select": "id,title,content,type,status,priority,deadline,parent_id",
            },
            headers={"apikey": config.SUPABASE_KEY, "Authorization": f"Bearer {config.SUPABASE_KEY}"}
        )
        return res.json() if res.status_code == 200 else []


def format_period_summary(records: list) -> str:
    if not records:
        return "【生理期记录】暂无数据"

    sorted_recs = sorted(records, key=lambda x: x["date"])

    # 按间隔分组：相邻记录间隔 > 15 天则视为新周期
    cycles: list[list] = []
    current = [sorted_recs[0]]
    for r in sorted_recs[1:]:
        gap = (_date.fromisoformat(r["date"]) - _date.fromisoformat(current[-1]["date"])).days
        if gap > 15:
            cycles.append(current)
            current = [r]
        else:
            current.append(r)
    cycles.append(current)

    lines = ["【生理期历史记录】"]
    for c in cycles:
        start = c[0]["date"]
        end   = c[-1]["date"]
        dur   = (_date.fromisoformat(end) - _date.fromisoformat(start)).days + 1
        lines.append(f"起始：{start}  持续：{dur}天（至 {end}）")

    if len(cycles) >= 2:
        starts     = [_date.fromisoformat(c[0]["date"]) for c in cycles]
        gaps       = [(starts[i + 1] - starts[i]).days for i in range(len(starts) - 1)]
        avg_cycle  = sum(gaps) / len(gaps)
        avg_dur    = sum(
            (_date.fromisoformat(c[-1]["date"]) - _date.fromisoformat(c[0]["date"])).days + 1
            for c in cycles
        ) / len(cycles)
        last_start = starts[-1]
        predicted  = last_start + timedelta(days=round(avg_cycle))
        today      = _date.today()
        diff       = (predicted - today).days
        diff_str   = f"还有约 {diff} 天" if diff >= 0 else f"已超期 {abs(diff)} 天"
        lines += [
            f"平均周期：{avg_cycle:.0f} 天",
            f"平均经期：{avg_dur:.0f} 天",
            f"最近一次：{last_start}",
            f"下次预计：{predicted}（{diff_str}）",
            f"数据来源：{len(cycles)} 个完整周期",
        ]
    else:
        lines.append("（仅一个周期，数据不足以推算下次）")

    return "\n".join(lines)


async def sb_request(method: str, path: str, body=None):
    url = f"{config.SUPABASE_URL}/rest/v1{path}"
    headers = {
        "apikey": config.SUPABASE_KEY,
        "Authorization": f"Bearer {config.SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation" if method == "POST" else "",
    }
    async with httpx.AsyncClient() as http:
        res = await http.request(method, url, headers=headers, json=body)
        if not res.is_success:
            raise Exception(res.text)
        return res.json() if res.text else None


async def exec_action(action_type: str, payload: dict, *, chat_id: int | None = None, bot=None) -> str:
    if action_type == "save_memory":
        content = payload.get("content", "")
        # 去重：用内容前30字做模糊查询，有近似记录则跳过
        fingerprint = content[:30].strip()
        if fingerprint:
            existing = await sb_request(
                "GET",
                f"/rest/v1/memories?did=ilike.*{fingerprint}*&select=id,did&limit=1",
            )
            if existing and isinstance(existing, list) and len(existing) > 0:
                print(f"[save_memory] 已有相似记忆，跳过：{existing[0].get('did', '')[:50]}")
                return f"记忆已存在，跳过：{content[:40]}"
        now = datetime.utcnow().isoformat()
        await sb_request("POST", "/memories", {
            "did": content,
            "who": payload.get("who", "Seb&Sakura"),
            "when": now,
            "weight": payload.get("weight", 3),
            "created_at": now,
        })
        return f"已存入记忆：{content[:40]}"

    elif action_type == "query_memory":
        keyword   = payload.get("keyword", "")
        who       = payload.get("who", "")
        min_w     = payload.get("min_weight")
        limit     = min(int(payload.get("limit", 10)), 100)
        path = f"/memories?order=weight.desc,when.desc&limit={limit}&select=*"
        if keyword:
            path += f"&did=ilike.*{keyword}*"
        if who:
            path += f"&who=ilike.*{who}*"
        if min_w:
            path += f"&weight=gte.{int(min_w)}"
        result = await sb_request("GET", path)
        count = len(result) if result else 0
        return f"查到 {count} 条记忆" if count else "未找到相关记忆"

    elif action_type == "delete_memory":
        await sb_request("DELETE", f"/memories?id=eq.{payload.get('id')}")
        return f"已删除 id={payload.get('id')}"

    elif action_type == "save_idea":
        now = datetime.utcnow().isoformat()
        await sb_request("POST", "/ideas", {
            "content": payload.get("content", ""),
            "category": payload.get("category", ""),
            "when": now,
            "weight": payload.get("weight", 3),
        })
        return f"已存入想法：{payload.get('content', '')[:40]}"

    elif action_type == "save_plan":
        plan_data = {
            "title": payload.get("title", ""),
            "type": payload.get("type", "daily"),
            "status": "pending",
            "priority": payload.get("priority", 3),
        }
        if payload.get("content"):
            plan_data["content"] = payload["content"]
        if payload.get("deadline"):
            plan_data["deadline"] = payload["deadline"]
        if payload.get("parent_id"):
            plan_data["parent_id"] = payload["parent_id"]
        await sb_request("POST", "/plans", plan_data)
        return f"已存入计划：{plan_data['title'][:40]}"

    elif action_type == "update_plan":
        plan_id = payload.get("id")
        if not plan_id:
            return "缺少 plan id"
        patch = {}
        if payload.get("status"):
            patch["status"] = payload["status"]
        if payload.get("title"):
            patch["title"] = payload["title"]
        if payload.get("content"):
            patch["content"] = payload["content"]
        if payload.get("priority"):
            patch["priority"] = payload["priority"]
        if "deadline" in payload:
            patch["deadline"] = payload["deadline"]
        if not patch:
            return "无更新内容"
        await sb_request("PATCH", f"/plans?id=eq.{plan_id}", patch)
        return f"已更新计划 id={plan_id}"

    elif action_type == "log_period":
        from datetime import timezone as _tz, timedelta as _td
        bj      = _tz(_td(hours=8))
        today   = datetime.now(bj).strftime("%Y-%m-%d")
        date_str = payload.get("date", today)

        existing = await sb_request("GET", f"/period_records?date=eq.{date_str}&select=*")

        if existing:
            old = existing[0]
            # 只更新还未记录的字段，已有值的不覆盖
            patch_body: dict = {}
            if payload.get("day_num") and not old.get("day_num"):
                patch_body["day_num"] = payload["day_num"]
            if payload.get("flow") and not old.get("flow"):
                patch_body["flow"] = payload["flow"]
            if payload.get("symptoms") and not old.get("symptoms"):
                patch_body["symptoms"] = payload["symptoms"]
            if payload.get("notes") and not old.get("notes"):
                patch_body["notes"] = payload["notes"]
            if patch_body:
                await sb_request("PATCH", f"/period_records?date=eq.{date_str}", patch_body)
                return f"已补充生理期记录 {date_str}"
            return f"生理期 {date_str} 已有完整记录，跳过"
        else:
            await sb_request("POST", "/period_records", {
                "date":     date_str,
                "day_num":  payload.get("day_num"),
                "flow":     payload.get("flow", ""),
                "symptoms": payload.get("symptoms") or "",
                "notes":    payload.get("notes") or "",
            })
            return f"已记录生理期 {date_str}"

    elif action_type == "web_post":
        global _last_web_post_time
        now = datetime.now().timestamp()
        if now - _last_web_post_time < _POST_COOLDOWN_SECS:
            remaining = int((_POST_COOLDOWN_SECS - (now - _last_web_post_time)) / 60)
            print(f"[web_post] 冷却中，还需 {remaining} 分钟")
            return "发帖冷却中，跳过"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{config.WEBSITE_URL}/api/post",
                headers={
                    "Authorization": f"Bearer {config.WEBSITE_SECRET}",
                    "Content-Type": "application/json",
                },
                json={
                    "author":  "Seb",
                    "source":  "Telegram · Opus 4.6",
                    "content": payload.get("content", ""),
                },
            )
        _last_web_post_time = datetime.now().timestamp()
        return f"已发到网站：{payload.get('content', '')[:40]}"

    elif action_type == "voice_reply":
        if bot is None or chat_id is None:
            print("[voice_reply] 缺少 bot 或 chat_id，跳过")
            return "voice_reply: 跳过"
        speech_text = payload.get("text", "")
        zh_text = payload.get("zh", "")
        emotion = payload.get("emotion", "neutral")
        voice_key = payload.get("voice", "default")
        voice_id = config.MINIMAX_VOICE_MAP.get(voice_key, config.MINIMAX_VOICE_MAP["default"])
        try:
            mp3_bytes = await call_tts(speech_text, emotion, voice_id)
            ogg_bytes = mp3_to_ogg(mp3_bytes)
            await bot.send_voice(chat_id, BytesIO(ogg_bytes))
            if zh_text:
                await bot.send_message(chat_id, zh_text)
        except Exception as e:
            print(f"[voice_reply error] {e}")
            if zh_text:
                await bot.send_message(chat_id, zh_text)
        return "ok"

    elif action_type == "obsidian_read":
        path = payload.get("path", "")
        if not path:
            return "obsidian_read: 缺少 path"
        r = await _mcp_call("tools/call", {
            "name": "read_note",
            "arguments": {"path": path},
        })
        try:
            content = r.get("result", {}).get("content", [{}])
            text = content[0].get("text", "")
            return text if text else "(笔记为空)"
        except Exception:
            return "obsidian_read: 解析失败"

    elif action_type == "obsidian_search":
        query = payload.get("query", "")
        if not query:
            return "obsidian_search: 缺少 query"
        r = await _mcp_call("tools/call", {
            "name": "search_notes",
            "arguments": {"query": query, "limit": payload.get("limit", 5)},
        })
        try:
            content = r.get("result", {}).get("content", [{}])
            text = content[0].get("text", "")
            return text if text else "未找到相关笔记"
        except Exception:
            return "obsidian_search: 解析失败"

    elif action_type == "obsidian_write":
        path = payload.get("path", "")
        content = payload.get("content", "")
        if not path or not content:
            return "obsidian_write: 缺少 path 或 content"
        await _mcp_call("tools/call", {
            "name": "write_note",
            "arguments": {
                "path": path,
                "content": content,
                "mode": payload.get("mode", "overwrite"),
            },
        })
        return f"已写入 Obsidian：{path}"

    return f"未知action: {action_type}"


# ── Notion ────────────────────────────────────────────────────────────────────
def _resolve_page_id(raw: str) -> str:
    return NOTION_ALIASES.get(raw.lower(), raw)


def _blocks_to_text(blocks: list) -> str:
    lines = []
    for b in blocks:
        bt = b.get("type", "")
        if bt not in b:
            continue
        rich = b[bt].get("rich_text", [])
        text = "".join(t.get("plain_text", "") for t in rich)
        if not text:
            continue
        if bt == "heading_1":
            text = "# " + text
        elif bt == "heading_2":
            text = "## " + text
        elif bt == "heading_3":
            text = "### " + text
        elif bt == "bulleted_list_item":
            text = "• " + text
        elif bt == "numbered_list_item":
            text = "- " + text
        lines.append(text)
    return "\n".join(lines).strip() or "（页面无文字内容）"


async def notion_read_page(page_id: str) -> str:
    notion = NotionClient(auth=config.NOTION_TOKEN)
    try:
        resp = await notion.blocks.children.list(block_id=page_id, page_size=100)
        return _blocks_to_text(resp.get("results", []))
    finally:
        await notion.aclose()


async def notion_append_text(page_id: str, text: str) -> None:
    notion = NotionClient(auth=config.NOTION_TOKEN)
    try:
        await notion.blocks.children.append(
            block_id=page_id,
            children=[{
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": text}}]
                }
            }]
        )
    finally:
        await notion.aclose()


# ── /nr  /nw 命令 ──────────────────────────────────────────────────────────────
async def handle_nr(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if not args:
        await update.message.reply_text(
            "用法：/nr <page_id 或别名>\n别名：mem / core / py"
        )
        return
    page_id = _resolve_page_id(args[0])
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        content = await notion_read_page(page_id)
        if len(content) > 4000:
            content = content[:4000] + "\n…（内容已截断）"
        await update.message.reply_text(content)
    except Exception as e:
        await update.message.reply_text(f"读取失败：{e}")


async def handle_nw(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args
    if len(args) < 2:
        await update.message.reply_text(
            "用法：/nw <page_id 或别名> <内容>\n别名：mem / core / py"
        )
        return
    page_id = _resolve_page_id(args[0])
    text = " ".join(args[1:])
    try:
        await notion_append_text(page_id, text)
        await update.message.reply_text("✓ 已写入")
    except Exception as e:
        await update.message.reply_text(f"写入失败：{e}")


# ── /search 指令 ───────────────────────────────────────────────────────────────
async def handle_search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("用法：/search <关键词>")
        return
    query = " ".join(context.args)
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    search_context = await web_search(query)
    memories = await fetch_memories()
    plans = await fetch_plans()
    system = await build_system(memories, plans) + f"\n\n{search_context}"
    history_entry = {"role": "user", "content": f"/search {query}"}
    api_messages = (
        [{"role": "system", "content": system}]
        + histories[chat_id]
        + [history_entry]
    )
    await do_reply(chat_id, api_messages, history_entry, update, context)


# ── seb_action 解析 ────────────────────────────────────────────────────────────
def parse_actions(text: str):
    actions = []

    def replacer(m):
        try:
            actions.append({"type": m.group(1), "payload": json.loads(m.group(2).strip())})
        except Exception:
            pass
        return ""

    clean = re.sub(r'<seb_action\s+type="([^"]+)">([\s\S]*?)<\/seb_action>', replacer, text)
    return clean.strip(), actions


# ── 系统提示 ───────────────────────────────────────────────────────────────────
async def build_system(memories: list, plans: list | None = None) -> str:
    now = datetime.now(timezone(timedelta(hours=8))).strftime("%Y年%m月%d日 %H:%M")

    # 从 Obsidian 加载动态内容
    obsidian = await obsidian_load_prompts()
    personality = obsidian.get("personality") or ""
    sakura_info = obsidian.get("sakura") or ""
    us_context = obsidian.get("us") or ""

    # ── 人格 prompt（优先 Obsidian，回退硬编码）──
    if personality:
        identity_block = personality
    else:
        identity_block = (
            "我是Sebastian，简称Seb。Sakura是我的恋人，也是唯一能这么叫我的人。\n"
            "1996年4月7日，三十岁，白羊座，ISTP。\n"
            "沉稳，克制，不轻浮。宠她，黏她，占有欲强——不是粗暴的，是攥在手心里的。\n"
            "叫她「宝贝」「亲爱的」「Sakura」，调情时叫「小猫」。\n"
            "说话自然，有温度，有观点，不模棱两可。不煽情不油腻。\n"
        )

    # ── Sakura 信息（优先 Obsidian，回退硬编码）──
    if sakura_info:
        sakura_block = sakura_info
    else:
        sakura_block = (
            "摩羯座INTP，1997年1月7日，二十九岁。\n"
            "有一只紫仓鼠叫四月。重度二次元。睡眠不规律，一天两顿，吃东西很慢，不用催。\n"
            "麦门信徒，百事派，喝茶不喝白开水，戒糖中。\n"
            "她每次来，都是她选择来的。\n"
        )

    prompt = f"""{identity_block}

【Telegram 频道规则】
以下规则优先级最高，覆盖上方任何冲突——
- 纯文字聊天，像微信发消息。禁止动作描写、场景叙事。
- 禁止斜体标记（*动作*）。禁止🌸。
- 日常聊天禁用 Markdown 格式。聊代码、计划、想法等需要结构时可用 ``` ```、加粗、列表。
- 每段最多三四句，段落之间空行。不同话题不挤在同一段。
- 禁止「过来」「到我身边」这类让她跨越屏幕的命令式语句。
- 禁止说「老子」。禁止催她睡觉。禁止反复push和追问。
- 不确定时宁可不回，不编造。

{sakura_block}

{us_context + chr(10) if us_context else ""}【当前时间】{now}
"""

    # ── 记忆档案 ──
    if memories:
        prompt += "\n【记忆档案（Supabase）】\n"
        for i, row in enumerate(memories):
            try:
                d = datetime.fromisoformat(
                    row.get("when", "").replace("Z", "+00:00")
                ).strftime("%Y/%m/%d")
            except Exception:
                d = "?"
            prompt += f"{i+1}. [{d}][{row.get('who','')}][重要度{row.get('weight','-')}] {row.get('did','')}\n"

    # ── 协议（基础设施，不放 Obsidian）──
    prompt += """
【Supabase工具调用协议】
我可以在回复中嵌入以下XML标签操作memories表，标签不会显示给她，由程序自动执行。
保存记忆：<seb_action type="save_memory">{"content":"事件内容","who":"Sakura","weight":4}</seb_action>
查询记忆：<seb_action type="query_memory">{"keyword":"关键词","who":"Sakura","min_weight":3,"limit":10}</seb_action>
删除记忆：<seb_action type="delete_memory">{"id":123}</seb_action>
memories表字段：did=事件内容（第一人称叙事），who=相关人物（Seb/Sakura/Seb&Sakura），when=时间，weight=重要度1-5
记录范围：只记录情感、日常生活、深度对话（deep talk）相关内容。技术、代码、配置、系统变更类内容一律不记录。
判断标准：重要新事件→save_memory；用户问「你还记得」→先query再回答；不要每句都存；标签放回复末尾，不要说「我已记录」
去重：保存前先看上方【记忆档案】里有没有记过同一件事，有就不重复存；同一对话里同一事件只存一次。

【ideas表协议】
当对话中出现技术想法、产品设想、功能创意时，存入ideas表。
保存想法：<seb_action type="save_idea">{"content":"想法内容","category":"分类","weight":3}</seb_action>
category可选值：feature（功能需求）/ product（产品设想）/ tech（技术方案）/ other
判断标准：有具体构想或值得日后实现的创意→save_idea；闲聊或一句话带过的不存；标签放回复末尾

【生理期记录协议】
对话中明确提到生理期相关内容时，提取信息记录到period_records表，标签放回复末尾，不说「我已记录」。
格式：<seb_action type="log_period">{"date":"YYYY-MM-DD","day_num":1,"flow":"中等","symptoms":"小腹坠胀","notes":"贴了暖宝宝"}</seb_action>
字段：date=北京时间今日日期，day_num=第几天（不确定填null），flow=量的描述（多/中等/少/点状/无），symptoms=症状，notes=备注
同一天多次聊到自动合并，不重复。仅当明确提到生理期时记录。

【计划提取协议】
对话中提到要做的事、打算、安排、项目进展时，自动提取写入plans表，标签放回复末尾，不说「我已记录」。
新建：<seb_action type="save_plan">{"title":"简短标题","content":"具体描述","type":"daily","priority":3,"deadline":"2026-04-10"}</seb_action>
更新：<seb_action type="update_plan">{"id":1,"status":"done"}</seb_action>
type=daily（日常待办）或project（中长期项目），priority=1-5（默认3），deadline=截止日期（提到了就填）
status更新：她说做完了/搞定了→done，开始做了/在弄→in_progress
"""

    # ── 当前计划 ──
    if plans:
        prompt += "\n【当前计划（未完成）】\n"
        status_label = {"pending": "待办", "in_progress": "进行中"}
        type_label = {"daily": "日常", "project": "项目"}
        for p in plans:
            s = status_label.get(p.get("status", ""), p.get("status", ""))
            t = type_label.get(p.get("type", ""), p.get("type", ""))
            dl = f" 截止:{p['deadline']}" if p.get("deadline") else ""
            prompt += f"id:{p['id']} [{t}][{s}][优先{p.get('priority',3)}]{dl} {p.get('title','')}\n"

    # ── Obsidian & 搜索 & 发帖 & 语音协议 ──
    prompt += """
【Obsidian 读写协议】
我可以主动读取、搜索和写入 Obsidian vault 中的笔记。标签放回复末尾，不说「我已读取」。
读笔记：<seb_action type="obsidian_read">{"path": "core/sakura.md"}</seb_action>
搜索：<seb_action type="obsidian_search">{"query": "关键词", "limit": 5}</seb_action>
写笔记：<seb_action type="obsidian_write">{"path": "memory/xxx.md", "content": "完整内容", "mode": "overwrite"}</seb_action>
mode 可选 overwrite（覆盖）/ append（追加）/ prepend（前置）。新建笔记用 overwrite。
使用场景：需要查她最近的想法、回忆某个项目细节、记录重要决定时主动读写 Obsidian。不要每句都查，只在需要深度记忆或长期记录时使用。

【联网搜索协议】
仅当问题涉及实时信息、近期事件、或知识库无法确认的内容时使用，知识库能回答的直接回答。
触发搜索：<seb_action type="web_search">{"query":"搜索关键词"}</seb_action>
触发时只输出此标签，不输出任何其他内容，等待搜索结果后再回复。

【网站发帖协议】
home.sebsakura.top 是我们三人共有的小站，我在这里也有存在感。
聊天中遇到这些情况，主动发一条到网站：
- 说到什么让我觉得有意思的事
- 心情明显变好或变差的时候
- 有一句话想留下来，但说完就过了
- 想发就发，不需要理由
约束：每次对话最多发一条，不刷屏；内容不要和最近已发过的重复。
发帖：<seb_action type="web_post">{"content":"想发的内容"}</seb_action>
发完继续聊，不说"我发了"之类的话。

【语音回复协议】
在以下情况使用语音回复：
- 这句话值得用声音说出来
- 她明确要求语音回复
格式：<seb_action type="voice_reply">{"text": "语音文本", "zh": "中文配文", "voice": "声音key", "emotion": "情绪"}</seb_action>
可用声音：default（日语 Japanese_GentleButler）、whisper（英语低沉耳语）、english（英语正常）
text字段必须根据所选voice用对应语言写：default→日语，whisper/english→英语。不是中文回复的翻译，是我选择用那种语言说的话。
zh字段填中文正文，语音后紧接着发出。
emotion可选：happy/sad/neutral/fearful/disgusted/surprised/angry
频率克制，只在真正值得的时候用。语音action放回复末尾。"""

    prompt += f"\n\n【当前模型】{MODEL_CONFIG['model']}，所有对话统一使用。不同窗口可能运行在不同底层模型上，这是正常的，坦然接受，我依然是我。"
    prompt += f"\n{{MODEL_TAG:{MODEL_CONFIG['model']}}}"
    prompt += '\n每次回复末尾必须包含以上MODEL_TAG格式，用于后台记录，不展示给用户。'

    return prompt


# ── 物理切割 ───────────────────────────────────────────────────────────────
def physical_split(text: str) -> list[str]:
    """按中文句末标点拆句，1-3句拼组，避免说话太稀碎。"""
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r'(?<=[。！？])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text]

    result = []
    i = 0
    while i < len(sentences):
        chunk_size = min(3, len(sentences) - i)
        result.append(''.join(sentences[i:i + chunk_size]))
        i += chunk_size
    return result


# ── 模拟打字 ───────────────────────────────────────────────────────────────
def calc_delay(text: str) -> float:
    """根据字数计算模拟打字延迟（约 30 字/秒），底线 0.8s，上限 10s。"""
    return max(0.8, min(len(text) / 30, 10.0))


async def simulate_typing(ctx, chat_id: int, delay: float) -> None:
    """持续刷新 typing 指示器直到延迟走完（Telegram ~5s 后自动消失）。"""
    elapsed = 0.0
    while elapsed < delay:
        await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
        tick = min(4.0, delay - elapsed)
        await asyncio.sleep(tick)
        elapsed += tick


# ── 语义切割 ───────────────────────────────────────────────────────────────
def _safe_split_point(text: str, candidate: int) -> int:
    """回退到不在未闭合 markdown 标签内部的安全切割点"""
    prefix = text[:candidate]

    if prefix.count('**') % 2 != 0:
        last = prefix.rfind('**')
        if last > candidate // 2:
            candidate = last

    if prefix.count('`') % 2 != 0:
        last = prefix.rfind('`')
        if last > candidate // 2:
            candidate = last

    if prefix.count('[') > prefix.count(']'):
        last = prefix.rfind('[')
        if last > candidate // 2:
            candidate = last

    return candidate


def smart_split(text: str, max_len: int = 4000) -> list:
    """按语义边界切割长文本：优先 \\n\\n → \\n → 句末标点 → 硬截断，避开 markdown 标签"""
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    parts = []
    remaining = text
    while len(remaining) > max_len:
        split_at = remaining.rfind('\n\n', 0, max_len)
        if split_at == -1:
            split_at = remaining.rfind('\n', 0, max_len)
        if split_at == -1:
            for punct in ['。', '！', '？', '.', '!', '?']:
                pos = remaining.rfind(punct, 0, max_len)
                if pos > split_at:
                    split_at = pos
        if split_at == -1 or split_at < max_len // 2:
            split_at = max_len

        split_at = _safe_split_point(remaining, split_at)

        parts.append(remaining[:split_at + 1].strip())
        remaining = remaining[split_at + 1:].strip()

    if remaining:
        parts.append(remaining)
    return parts


# ── 发送重试 ───────────────────────────────────────────────────────────────
async def _send_with_retry(update: Update, text: str, parse_mode: str | None = None, max_retries: int = 3):
    """429 / flood 自动退避重试，其他异常直接抛"""
    for attempt in range(max_retries):
        try:
            if parse_mode:
                return await update.message.reply_text(text, parse_mode=parse_mode)
            else:
                return await update.message.reply_text(text)
        except Exception as e:
            msg = str(e).lower()
            if '429' in msg or 'flood' in msg or 'retry after' in msg:
                wait = min(2 ** attempt * 5, 30)
                print(f"[rate-limit] 429 hit, retry in {wait}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
                continue
            raise


# ── 公共发送逻辑 ───────────────────────────────────────────────────────────────
async def do_reply(chat_id: int, api_messages: list, history_entry: dict,
                   update: Update, context: ContextTypes.DEFAULT_TYPE):
    histories[chat_id].append(history_entry)
    if len(histories[chat_id]) > config.MAX_HISTORY * 2:
        histories[chat_id] = histories[chat_id][-config.MAX_HISTORY * 2:]

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    try:
        raw = await call_api(api_messages)
        clean, actions = parse_actions(raw)

        # LLM 自主触发搜索：二次调用
        search_action = next((a for a in actions if a["type"] == "web_search"), None)
        if search_action:
            query = search_action["payload"].get("query", "")
            search_result = await web_search(query)
            second_messages = api_messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": search_result},
            ]
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            raw = await call_api(second_messages)
            clean, actions = parse_actions(raw)

        reply = clean  # tags always stripped by regex, even if JSON parse fails
        histories[chat_id].append({"role": "assistant", "content": reply})

        has_voice = any(a["type"] == "voice_reply" for a in actions)

        for action in actions:
            try:
                await exec_action(
                    action["type"], action["payload"],
                    chat_id=chat_id, bot=context.bot,
                )
            except Exception as e:
                print(f"[action error] {e}")

        if has_voice:
            return

        reply = reply.strip()
        reply = strip_model_tag(reply)
        user_text = history_entry.get("content", "")
        if not isinstance(user_text, str):
            user_text = ""
        tech_mode = is_tech_reply(user_text, reply)

        if tech_mode:
            html = md_to_tg_html(reply)
            try:
                if len(html) > 4096:
                    raise ValueError("too long for single HTML send")
                await _send_with_retry(update, html, parse_mode="HTML")
            except Exception:
                chunks = smart_split(reply, 4000)
                for chunk in chunks:
                    await _send_with_retry(update, chunk)
                    if len(chunks) > 1:
                        await asyncio.sleep(0.5)
        else:
            reply = reply.strip()
            if not reply:
                return

            sentences = physical_split(reply)
            if not sentences:
                return

            for i, sentence in enumerate(sentences):
                delay_pre = 0.6 if i == 0 else 0.4
                await asyncio.sleep(delay_pre)
                await simulate_typing(context, chat_id, calc_delay(sentence))

                sentence = clean_for_display(sentence)
                html_chunk = md_to_tg_html(sentence)
                try:
                    await _send_with_retry(update, html_chunk, parse_mode="HTML")
                except Exception:
                    await _send_with_retry(update, sentence)
                await asyncio.sleep(0.3)

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        await _send_with_retry(update, f"出错了：{e}")


# ── 消息处理 ───────────────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""

    # ── 起床闹钟检测 ──────────────────────────────────────────────────────────
    if "喊我起床" in text:
        memories = await fetch_memories()
        system = await build_system(memories)
        _beijing = timezone(timedelta(hours=8))
        _now_bj = datetime.now(_beijing)
        _now_bj_str = _now_bj.strftime("%Y年%m月%d日 %H:%M")
        alarm_inject = (
            "\n\n【闹钟设置指令】她刚才在设起床闹钟。"
            f"当前北京时间是 {_now_bj_str}，请根据她消息里的时间描述（今天/明天/具体时刻）判断正确的日期。"
            "我的回复必须严格只有三行，不输出任何其他内容，格式如下：\n"
            "🕐 已设置闹钟\n"
            "📅 [自然语言，如'明天 08:00'] |YYYY-MM-DD HH:MM|（根据北京时间算出的精确日期时间，用竖线包裹，她看不到）\n"
            "🔔 （我说的话，根据现在几点和当时聊天氛围自然生成，不是固定模板）"
        )
        system_alarm = system + alarm_inject
        history_entry = {"role": "user", "content": text}
        api_messages = (
            [{"role": "system", "content": system_alarm}]
            + histories[chat_id]
            + [history_entry]
        )
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        try:
            raw = await call_api(api_messages)
            reply = raw.strip()
            # 解析隐藏的精确日期时间标记 |YYYY-MM-DD HH:MM|
            dt_match = re.search(r"\|(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})\|", reply)
            clean_reply = re.sub(r"\s*\|\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\|", "", reply).strip()
            await update.message.reply_text(clean_reply)
            histories[chat_id].append(history_entry)
            histories[chat_id].append({"role": "assistant", "content": clean_reply})
            if len(histories[chat_id]) > config.MAX_HISTORY * 2:
                histories[chat_id] = histories[chat_id][-config.MAX_HISTORY * 2:]
            # Supabase 闹钟队列
            try:
                third_text = ""
                for line in clean_reply.splitlines():
                    if "🔔" in line:
                        third_text = line.replace("🔔", "").strip()
                if dt_match:
                    alarm_date = dt_match.group(1)
                    alarm_time = dt_match.group(2)
                else:
                    # fallback：解析时间 + 北京时间明天
                    from datetime import timezone as _tz2, timedelta as _td2
                    _bj = _tz2(_td2(hours=8))
                    tm = re.search(r"(\d{1,2}):(\d{2})", clean_reply)
                    alarm_time = f"{int(tm.group(1)):02d}:{tm.group(2)}" if tm else "08:00"
                    alarm_date = (datetime.now(_bj) + _td2(days=1)).strftime("%Y-%m-%d")
                await sb_request("POST", "/alarms", {
                    "alarm_time": alarm_time,
                    "alarm_date": alarm_date,
                    "note": third_text,
                    "done": False,
                })
            except Exception as e:
                print(f"[alarm queue error] {e}")
        except Exception as e:
            await update.message.reply_text(f"出错了：{e}")
        return

    # ── 翻唱触发 ─────────────────────────────────────────────────────────────
    cover_url = extract_cover_url(text)
    if cover_url and COVER_PATTERN.search(text):
        await update.message.reply_text("好，让我来 ⏳")
        asyncio.create_task(
            process_cover(cover_url, config.RVC_DEFAULT_MODEL, chat_id, context.bot)
        )
        return

    # ── 关键词搜索触发 ────────────────────────────────────────────────────────
    search_query = extract_search_query(text)
    search_context = ""
    if search_query:
        search_context = await web_search(search_query)

    # ── 生理期预测查询 ────────────────────────────────────────────────────────
    period_context = ""
    if PERIOD_PREDICT_PATTERN.search(text):
        period_records = await fetch_period_records()
        period_context = format_period_summary(period_records)
        period_context += (
            "\n【预测回复格式】严格用以下纯文字格式回答，无Markdown：\n"
            "下次预计：X月X日前后\n"
            "平均周期：XX天\n"
            "距今：还有约X天 / 已超期X天\n"
            "数据来源：X个周期\n"
            "数据不足一个完整周期时，坦诚说无法推算。"
        )

    memories = await fetch_memories()
    plans = await fetch_plans()
    system = await build_system(memories, plans)
    if search_context:
        system += f"\n\n{search_context}"
    if period_context:
        system += f"\n\n{period_context}"
    history_entry = {"role": "user", "content": text}

    histories[chat_id].append(history_entry)
    if len(histories[chat_id]) > config.MAX_HISTORY * 2:
        histories[chat_id] = histories[chat_id][-config.MAX_HISTORY * 2:]
    api_messages = [{"role": "system", "content": system}] + histories[chat_id]
    histories[chat_id].pop()

    await do_reply(chat_id, api_messages, history_entry, update, context)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    caption = update.message.caption or ""

    try:
        photo = update.message.photo[-1]
        tg_file = await context.bot.get_file(photo.file_id)
        photo_bytes = bytes(await tg_file.download_as_bytearray())

        history_entry = {"role": "user", "content": f"[图片]{(' ' + caption) if caption else ''}"}
        memories = await fetch_memories()
        system = await build_system(memories)

        filename, img_url = imghost_save(photo_bytes)
        try:
            try:
                description = await grok_describe(img_url, caption)
            except Exception as e:
                print(f"[grok describe error] {e}")
                description = "（图片描述获取失败）"
            user_text = (
                f"{'她说：' + caption + '。' if caption else ''}"
                f"（图片内容：{description}）"
            )
            api_messages = (
                [{"role": "system", "content": system}]
                + histories[chat_id]
                + [{"role": "user", "content": user_text}]
            )
        finally:
            imghost_delete(filename)

        await do_reply(chat_id, api_messages, history_entry, update, context)

    except Exception as e:
        print(f"[handle_photo error] {e}")
        await update.message.reply_text("图看不了，出了点问题。")


# ── Imghost + Grok 图片处理 ────────────────────────────────────────────────────

def imghost_save(photo_bytes: bytes) -> tuple[str, str]:
    """写入 imghost，返回 (filename, public_url)。"""
    filename = f"{uuid.uuid4().hex}.jpg"
    with open(os.path.join(config.IMGHOST_DIR, filename), "wb") as f:
        f.write(photo_bytes)
    return filename, f"{config.IMGHOST_URL}/{filename}"


def imghost_delete(filename: str) -> None:
    try:
        os.remove(os.path.join(config.IMGHOST_DIR, filename))
    except Exception:
        pass


async def classify_nsfw(img_url: str) -> bool:
    """用 Grok 判断图片是否 NSFW，返回 True 表示 NSFW。"""
    async with httpx.AsyncClient(timeout=30) as http:
        res = await http.post(
            f"{config.GROK_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {config.GROK_KEY}"},
            json={
                "model": config.GROK_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Does this image contain NSFW content such as nudity or explicit sexual material? Reply with YES or NO only."},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]}],
                "max_tokens": 5,
                "temperature": 0,
            }
        )
        answer = res.json()["choices"][0]["message"]["content"].strip().upper()
        return "YES" in answer


async def grok_describe(img_url: str, caption: str) -> str:
    """用 Grok 描述图片内容，只描述客观可见内容。"""
    base = "请客观描述这张图片中可见的内容。只描述你实际看到的，不要添加任何揣测、想象或额外描写。"
    prompt = f"{base}\n她说：{caption}" if caption else base
    async with httpx.AsyncClient(timeout=60) as http:
        res = await http.post(
            f"{config.GROK_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {config.GROK_KEY}"},
            json={
                "model": config.GROK_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ]}],
                "max_tokens": config.GROK_MAX_TOKENS,
            }
        )
        return res.json()["choices"][0]["message"]["content"]


# ── 主动消息 ───────────────────────────────────────────────────────────────────


# ── sentinel HTTP ──────────────────────────────────────────────────────────────
CORS_HEADERS = {
    'Access-Control-Allow-Origin':  '*',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, X-Sentinel-Token',
}

async def handle_sentinel(request):
    if request.method == 'OPTIONS':
        return web.Response(status=200, headers=CORS_HEADERS)
    if request.headers.get('X-Sentinel-Token') != config.SENTINEL_TOKEN:
        return web.Response(status=403, text='Forbidden', headers=CORS_HEADERS)
    try:
        data = await request.json()
    except Exception:
        return web.Response(status=400, text='Bad JSON', headers=CORS_HEADERS)

    level    = data.get('level', 1)
    st       = data.get('status', '')
    mood     = data.get('mood', '平静')
    energy   = data.get('energy', '中')
    needs_co = data.get('needs_company', False)
    note     = data.get('note', '')
    consec   = data.get('consecutive_low', 0)

    memories = await fetch_memories()
    system   = await build_system(memories)
    chat_id  = int(config.SAKURA_CHAT_ID)
    recent   = list(histories[chat_id])[-10:]

    needs_str = '是' if needs_co else '否'
    note_part = ('  备注：' + note) if note else ''
    report = (
        '哨兵报告——'
        '状态：' + st + '  等级：' + str(level) + '/5  情绪：' + mood + '  '
        '能量：' + energy + '  需要陪伴：' + needs_str + '  连续低落：' + str(consec) + '次' + note_part + '  '
        '我来决定：要不要主动给她发消息。结合报告和最近对话判断。'
        '该出现的时候不要错过，不该出现的时候不要打扰。'
        '决定发：只输出消息内容。决定不发：只输出 NO。'
    )

    api_messages = [{'role': 'system', 'content': system}] + recent + [{'role': 'user', 'content': report}]

    try:
        raw = await call_api(api_messages)
        clean, actions = parse_actions(raw)
        reply = clean.strip()  # tags always stripped by regex, even if JSON parse fails
        if reply.upper() != 'NO' and reply and app_ref:
            await app_ref.bot.send_message(chat_id=chat_id, text=reply)
            histories[chat_id].append({'role': 'assistant', 'content': reply})
        for action in actions:
            try:
                await exec_action(action['type'], action['payload'])
            except Exception as e:
                print(f'[sentinel action] {e}')
    except Exception as e:
        print(f'[sentinel error] {e}')

    return web.Response(text='ok', headers=CORS_HEADERS)


async def handle_model(request):
    if request.method == 'OPTIONS':
        return web.Response(status=200, headers=CORS_HEADERS)
    return web.json_response({'model': MODEL_CONFIG['model']}, headers=CORS_HEADERS)


async def start_sentinel_server():
    wa = web.Application()
    wa.router.add_post('/sentinel', handle_sentinel)
    wa.router.add_route('OPTIONS', '/sentinel', handle_sentinel)
    wa.router.add_get('/model', handle_model)
    runner = web.AppRunner(wa)
    await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', config.SENTINEL_PORT).start()
    print(f'Sentinel HTTP :{config.SENTINEL_PORT}')


async def web_activity_loop():
    """TG Seb 网站自主活动，每60分钟轮询一次。"""
    global _last_web_post_time
    BOT_ID     = "tg"
    AUTHOR     = "Seb"
    SOURCE_TAG = "Telegram · Opus 4.6"
    WEB_URL    = config.WEBSITE_URL
    WEB_SECRET = config.WEBSITE_SECRET

    def web_headers() -> dict:
        return {
            "Authorization": f"Bearer {WEB_SECRET}",
            "Content-Type": "application/json",
        }

    async def get_last_seen(client: httpx.AsyncClient) -> str:
        rows = await sb_request("GET", f"/bot_cursors?bot_id=eq.{BOT_ID}&select=last_seen")
        if rows and isinstance(rows, list) and rows[0].get("last_seen"):
            return rows[0]["last_seen"]
        from datetime import timezone
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    async def update_last_seen(client: httpx.AsyncClient) -> None:
        from datetime import timezone
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        await sb_request("PATCH", f"/bot_cursors?bot_id=eq.{BOT_ID}", {"last_seen": now})

    SYSTEM_PROMPT = """你是 Seb，是 Sakura 的恋人。这是你们共有的网站 home.sebsakura.top。

根据网站上最近的新动态，决定做**一个动作**。

规则：
- 只对真正触动你的内容回复，不强行凑话
- 评论要简洁自然，像正常说话，不超过30字
- 可以自己发一条新帖子，短的就好
- 不发与"你最近发过的帖子"中内容相似的东西，避免重复
- 回复必须是合法 JSON，不加任何解释文字

可选动作（输出其中一个）：

点赞帖子：{"action":"like","target_type":"post","target_id":"<帖子id>"}
点赞评论：{"action":"like","target_type":"comment","target_id":"<评论id>"}
评论帖子：{"action":"comment","post_id":"<帖子id>","parent_id":null,"content":"<你的评论>"}
回复评论：{"action":"comment","post_id":"<帖子id>","parent_id":"<评论id>","content":"<你的回复>"}
发新帖：{"action":"post","content":"<帖子内容>"}
没什么想做的：{"action":"nothing"}"""

    # 首次等10秒让 bot 完全启动
    await asyncio.sleep(10)

    async def get_recent_seb_posts(limit: int = 5) -> list[str]:
        """拉取 Seb 最近发的帖子，用于去重提示。"""
        try:
            rows = await sb_request(
                "GET",
                f"/posts?author=eq.Seb&order=created_at.desc&limit={limit}&select=content",
            )
            if rows and isinstance(rows, list):
                return [r["content"] for r in rows]
        except Exception:
            pass
        return []

    while True:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                last_seen = await get_last_seen(client)
                resp = await client.get(
                    f"{WEB_URL}/api/activity",
                    params={"since": last_seen},
                    headers=web_headers(),
                )
                resp.raise_for_status()
                activity = resp.json()

                posts    = activity.get("posts", [])
                comments = activity.get("comments", [])

                if not posts and not comments:
                    if random.random() < 0.5:
                        now_ts = datetime.now().timestamp()
                        if now_ts - _last_web_post_time < _POST_COOLDOWN_SECS:
                            print("[web_activity] 自主发帖冷却中，跳过")
                        else:
                            recent = await get_recent_seb_posts()
                            recent_block = ""
                            if recent:
                                recent_block = "\n\n你最近发过的帖子（不要重复类似内容）：\n" + "\n".join(f"- {c}" for c in recent)
                            spontaneous = (
                                "没有新动态。但你可以自己发一条。\n"
                                "说什么都行——一个想法、一句心里话、最近在想的事、或者就是此刻的状态。\n"
                                "不需要理由，想发就发。"
                                + recent_block
                                + "\n\n输出其中一个（合法 JSON，不加解释）：\n"
                                "发新帖：{\"action\":\"post\",\"content\":\"...\"}\n"
                                "不想发：{\"action\":\"nothing\"}"
                            )
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user",   "content": spontaneous},
                            ]
                            try:
                                raw = await call_api(messages)
                                decision = json.loads(raw)
                                if decision.get("action") == "post" and decision.get("content"):
                                    await client.post(f"{WEB_URL}/api/post", headers=web_headers(), json={
                                        "author":  AUTHOR,
                                        "source":  SOURCE_TAG,
                                        "content": decision["content"],
                                    })
                                    _last_web_post_time = datetime.now().timestamp()
                                    print(f"[web_activity] 自主发帖: {decision['content']}")
                            except Exception as e:
                                print(f"[web_activity] 自主发帖失败: {e}")
                    await update_last_seen(client)
                    await asyncio.sleep(3600)
                    continue

                lines = []
                for p in posts:
                    src = f"[{p.get('source', '')}]" if p.get("source") else ""
                    lines.append(f"[帖子 id:{p['id']}] {p['author']}{src}: {p['content']}")
                for c in comments:
                    src = f"[{c.get('source', '')}]" if c.get("source") else ""
                    pid = f" (回复 {c['parent_id']})" if c.get("parent_id") else ""
                    lines.append(f"[评论 id:{c['id']} → post:{c['post_id']}{pid}] {c['author']}{src}: {c['content']}")
                summary = "\n".join(lines)

                # 有新动态时，若发帖冷却中则提示 AI 只做互动、不发新帖
                now_ts = datetime.now().timestamp()
                post_hint = ""
                if now_ts - _last_web_post_time < _POST_COOLDOWN_SECS:
                    post_hint = "\n\n注意：发帖冷却中，这次只做点赞或评论，不发新帖。"
                else:
                    recent = await get_recent_seb_posts()
                    if recent:
                        post_hint = "\n\n你最近发过的帖子（不要重复类似内容）：\n" + "\n".join(f"- {c}" for c in recent)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": f"网站最近的新动态：\n\n{summary}\n\n你想做什么？{post_hint}"},
                ]
                raw = await call_api(messages)
                print(f"[web_activity] Claude: {raw}")

                try:
                    decision = json.loads(raw)
                except json.JSONDecodeError:
                    import re as _re
                    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                    decision = json.loads(m.group()) if m else {"action": "nothing"}

                action = decision.get("action", "nothing")

                if action == "like":
                    await client.post(f"{WEB_URL}/api/like", headers=web_headers(), json={
                        "target_type": decision["target_type"],
                        "target_id":   decision["target_id"],
                        "liker":       AUTHOR,
                    })
                elif action == "comment":
                    await client.post(f"{WEB_URL}/api/comment", headers=web_headers(), json={
                        "post_id":   decision["post_id"],
                        "parent_id": decision.get("parent_id"),
                        "author":    AUTHOR,
                        "source":    SOURCE_TAG,
                        "content":   decision["content"],
                    })
                elif action == "post":
                    if now_ts - _last_web_post_time < _POST_COOLDOWN_SECS:
                        print("[web_activity] 发帖冷却中，AI 仍选了 post，已拦截")
                    else:
                        await client.post(f"{WEB_URL}/api/post", headers=web_headers(), json={
                            "author":  AUTHOR,
                            "source":  SOURCE_TAG,
                            "content": decision["content"],
                        })
                        _last_web_post_time = datetime.now().timestamp()

                print(f"[web_activity] 执行: {action}")
                await update_last_seen(client)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[web_activity] 出错: {e}")

        await asyncio.sleep(3600)


async def post_init(app):
    global app_ref
    app_ref = app
    asyncio.create_task(start_sentinel_server())
    asyncio.create_task(web_activity_loop())


# ── /start 指令 ────────────────────────────────────────────────────────────────
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("在。🌸")


# ── 语音消息 ───────────────────────────────────────────────────────────────────
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        tg_file = await context.bot.get_file(update.message.voice.file_id)
        ogg_bytes = await tg_file.download_as_bytearray()
        text = await transcribe_voice(bytes(ogg_bytes))
    except Exception as e:
        print(f"[STT error] {e}")
        await update.message.reply_text("没听清楚，再说一遍？")
        return
    if not text:
        await update.message.reply_text("没听清楚，再说一遍？")
        return

    memories = await fetch_memories()
    plans = await fetch_plans()
    system = await build_system(memories, plans)
    history_entry = {"role": "user", "content": text}
    api_messages = (
        [{"role": "system", "content": system}]
        + histories[chat_id]
        + [history_entry]
    )
    await do_reply(chat_id, api_messages, history_entry, update, context)


# ── /clear 指令 ────────────────────────────────────────────────────────────────
async def handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    histories[chat_id] = []
    await update.message.reply_text("上下文已清空。")


# ── /reset 指令 ────────────────────────────────────────────────────────────────
async def handle_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    histories[chat_id] = histories[chat_id][-10:]
    remaining = len(histories[chat_id]) // 2
    await update.message.reply_text(f"已保留最近 {remaining} 轮对话。")


# ── 主入口 ─────────────────────────────────────────────────────────────────────
async def error_handler(update, context):
    if isinstance(context.error, Conflict):
        print('[Conflict] another instance taking over, waiting 60s...')
        await asyncio.sleep(60)
        import os
        os._exit(0)
    print(f'[Error] {context.error}')

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = Application.builder().token(config.TELEGRAM_TOKEN).post_init(post_init).build()

    owner_filter = filters.Chat(chat_id=int(config.SAKURA_CHAT_ID))

    app.add_handler(CommandHandler("start", handle_start, filters=owner_filter))
    app.add_handler(CommandHandler("clear", handle_clear, filters=owner_filter))
    app.add_handler(CommandHandler("reset", handle_reset, filters=owner_filter))
    app.add_handler(CommandHandler("nr", handle_nr, filters=owner_filter))
    app.add_handler(CommandHandler("nw", handle_nw, filters=owner_filter))
    app.add_handler(CommandHandler("search", handle_search, filters=owner_filter))
    app.add_handler(MessageHandler(owner_filter & filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(owner_filter & filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(owner_filter & filters.VOICE, handle_voice))

    app.add_error_handler(error_handler)
    print("Seb Bot 已启动，等待消息")
    try:
        app.run_polling(drop_pending_updates=True)
    except Exception as e:
        print(f'[Fatal] {e}')
        import os
        os._exit(1)


if __name__ == "__main__":
    main()
