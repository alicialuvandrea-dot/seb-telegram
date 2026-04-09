# Seb Telegram Bot

Claude Sonnet 驱动的 Telegram 聊天机器人，为 Sakura 构建。

## 功能

- AI 对话（Sonnet 4.6）
- 记忆系统（Supabase）
- 图片处理（Grok）
- 起床闹钟
- 哨兵监控系统
- 联网搜索（Tavily）
- 语音合成（MiniMax T2A v2）
- 网站自主活动

## 配置

1. 复制 `seb-bot/config.py.example` 为 `seb-bot/config.py`
2. 填入各服务的 API Key
3. 安装依赖：`pip install -r seb-bot/requirements.txt`
4. 运行：`python seb-bot/bot.py`

## 部署（VPS）

```bash
cp seb-bot/seb-telegram.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable seb-telegram
systemctl start seb-telegram
```
