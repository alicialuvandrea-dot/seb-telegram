import os
import json

# config.py — Seb Bot 配置
# 支持通过环境变量覆盖（Docker 部署用），不设环境变量时回退到文件默认值

TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "8727940934:AAGqyXxgmySKUSkmYZiESwZuai-JxS8z--Q")
API_KEY         = os.environ.get("API_KEY", "sk-6e23d043d7a54acbbe3f9244c50d4c7a")
API_BASE        = os.environ.get("API_BASE", "https://api.deepseek.com/v1")
MODEL           = os.environ.get("MODEL", "deepseek-v4-pro")

SUPABASE_URL    = os.environ.get("SUPABASE_URL", "https://yqfxcebqzqgzjcwffppr.supabase.co")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxZnhjZWJxenFnempjd2ZmcHByIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MjgwNjkyNywiZXhwIjoyMDg4MzgyOTI3fQ.WNo0VR0xD2rwS9aMDg9SguZjSVV7_nVxwXAZQJrFV-Y")

SAKURA_CHAT_ID  = os.environ.get("SAKURA_CHAT_ID", "5138140796")
SENTINEL_TOKEN  = os.environ.get("SENTINEL_TOKEN", "seb-sentinel-2026")
SENTINEL_PORT   = int(os.environ.get("SENTINEL_PORT", "8765"))

MEMORIES_LIMIT  = int(os.environ.get("MEMORIES_LIMIT", "50"))
MAX_HISTORY     = int(os.environ.get("MAX_HISTORY", "20"))
TEMPERATURE     = float(os.environ.get("TEMPERATURE", "1.0"))
MAX_TOKENS      = int(os.environ.get("MAX_TOKENS", "4096"))

NOTION_TOKEN    = os.environ.get("NOTION_TOKEN", "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")
PROACTIVE_HOURS = int(os.environ.get("PROACTIVE_HOURS", "5"))

TAVILY_API_KEY     = os.environ.get("TAVILY_API_KEY", "tvly-placeholder")
SEARCH_MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "5"))

GROK_KEY        = os.environ.get("GROK_KEY", "sk-zurF2nA1R6A5dX9O00145a8a66Ea488397677a3c3e90F167")
GROK_BASE       = os.environ.get("GROK_BASE", "https://aihubmix.com/v1")
GROK_MODEL      = os.environ.get("GROK_MODEL", "grok-4-1-fast-non-reasoning")
GROK_MAX_TOKENS = int(os.environ.get("GROK_MAX_TOKENS", "1024"))
IMGHOST_DIR     = os.environ.get("IMGHOST_DIR", "/home/ubuntu/imghost/files")
IMGHOST_URL     = os.environ.get("IMGHOST_URL", "https://imghost.sebsakura.top")

WEBSITE_URL     = os.environ.get("WEBSITE_URL", "https://home.sebsakura.top")
WEBSITE_SECRET  = os.environ.get("WEBSITE_SECRET", "c0bf81f798a1a508f4e9ac7b529b1a82f8eb6970d4279a340ef006cbc735247f")

MINIMAX_API_KEY  = os.environ.get("MINIMAX_API_KEY", "sk-api-ksB2sFdJEf2htKBGq-z2KifZUB3KLg18xeDMwwC4zR3W8w2N6vrTuTbNQVDt4YthjQKQsKOX2nkMNd_wCL_qrMKIC3LXYoNlDLnrN3ftUwvX-2cH0QQKGIg")
MINIMAX_VOICE_MAP = json.loads(os.environ.get("MINIMAX_VOICE_MAP", '{"default": "Japanese_GentleButler", "whisper": "whisper_man", "english": "English_DecentYoungMan"}'))

GROQ_STT_KEY   = os.environ.get("GROQ_STT_KEY", "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")
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")

RVC_SERVICE_URL    = os.environ.get("RVC_SERVICE_URL", "https://studio.sebsakura.top")
RVC_DEFAULT_MODEL  = os.environ.get("RVC_DEFAULT_MODEL", "Sebastian")
