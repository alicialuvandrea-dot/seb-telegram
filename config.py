import os
import json

# config.py — Seb Bot 配置
# 敏感密钥必须通过环境变量注入，不设硬编码回退值

TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]
API_KEY         = os.environ["API_KEY"]
API_BASE        = os.environ.get("API_BASE", "https://aihubmix.com/v1")
MODEL           = os.environ.get("MODEL", "claude-sonnet-4-6")

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "https://yqfxcebqzqgzjcwffppr.supabase.co")
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]

SAKURA_CHAT_ID  = os.environ.get("SAKURA_CHAT_ID", "5138140796")
SENTINEL_TOKEN  = os.environ["SENTINEL_TOKEN"]
SENTINEL_PORT   = int(os.environ.get("SENTINEL_PORT", "8765"))

MEMORIES_LIMIT  = int(os.environ.get("MEMORIES_LIMIT", "50"))
MAX_HISTORY     = int(os.environ.get("MAX_HISTORY", "20"))
TEMPERATURE     = float(os.environ.get("TEMPERATURE", "1.0"))
MAX_TOKENS      = int(os.environ.get("MAX_TOKENS", "4096"))

NOTION_TOKEN    = os.environ["NOTION_TOKEN"]
PROACTIVE_HOURS = int(os.environ.get("PROACTIVE_HOURS", "5"))

TAVILY_API_KEY     = os.environ.get("TAVILY_API_KEY", "tvly-placeholder")
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))

GROK_KEY        = os.environ["GROK_KEY"]
GROK_BASE       = os.environ.get("GROK_BASE", "https://aihubmix.com/v1")
GROK_MODEL      = os.environ.get("GROK_MODEL", "grok-4-1-fast-non-reasoning")
GROK_MAX_TOKENS = int(os.environ.get("GROK_MAX_TOKENS", "1024"))
IMGHOST_DIR     = os.environ.get("IMGHOST_DIR", "/home/ubuntu/imghost/files")
IMGHOST_URL     = os.environ.get("IMGHOST_URL", "https://imghost.sebsakura.top")

WEBSITE_URL     = os.environ.get("WEBSITE_URL", "https://home.sebsakura.top")
WEBSITE_SECRET  = os.environ["WEBSITE_SECRET"]

MINIMAX_API_KEY  = os.environ["MINIMAX_API_KEY"]
MINIMAX_VOICE_MAP = json.loads(os.environ.get("MINIMAX_VOICE_MAP", '{"default": "Japanese_GentleButler", "whisper": "whisper_man", "english": "English_DecentYoungMan"}'))

GROQ_STT_KEY   = os.environ["GROQ_STT_KEY"]
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")

RVC_SERVICE_URL    = os.environ.get("RVC_SERVICE_URL", "https://studio.sebsakura.top")
RVC_DEFAULT_MODEL  = os.environ.get("RVC_DEFAULT_MODEL", "Sebastian")
