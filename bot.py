import asyncio
import logging
import os
import subprocess

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CLAUDE_BIN = os.getenv("CLAUDE_BIN", "claude")

def get_allowed_ids() -> set[int]:
    raw = os.getenv("ALLOWED_USER_IDS", "")
    ids = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids


def is_allowed(user_id: int) -> bool:
    return user_id in get_allowed_ids()


async def run_claude(prompt: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [CLAUDE_BIN, "--print", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        ),
    )
    if result.returncode != 0 and result.stderr:
        logger.error("claude stderr: %s", result.stderr)
    return result.stdout.strip() or result.stderr.strip() or "No response."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("You are not authorised to use this bot.")
        return
    await update.message.reply_text(
        f"Hi {user.first_name}! Send me any message and I'll pass it to Claude."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("You are not authorised to use this bot.")
        return

    user_text = update.message.text
    logger.info("User %s (%d): %s", user.username or user.first_name, user.id, user_text)

    thinking_msg = await update.message.reply_text("Thinking...")

    try:
        response = await run_claude(user_text)
    except subprocess.TimeoutExpired:
        response = "Request timed out. Please try a shorter or simpler prompt."
    except Exception as e:
        logger.exception("Error running claude")
        response = f"Error: {e}"

    await thinking_msg.delete()

    # Telegram message limit is 4096 chars; split if needed
    for chunk in split_message(response):
        await update.message.reply_text(chunk)


def split_message(text: str, limit: int = 4096) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    if not get_allowed_ids():
        raise RuntimeError("ALLOWED_USER_IDS is empty — no one would be able to use the bot")

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Allowed user IDs: %s", get_allowed_ids())
    app.run_polling()


if __name__ == "__main__":
    main()
