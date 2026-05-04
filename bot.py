import asyncio
import logging
import os
import re
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
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Model aliases users can type as a prefix, e.g. "!haiku what is 2+2"
MODEL_ALIASES: dict[str, str] = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}

_FORCE_PREFIX_RE = re.compile(
    r"^!(" + "|".join(MODEL_ALIASES.keys()) + r")\s+",
    re.IGNORECASE,
)

_TECHNICAL_TERMS = re.compile(
    r"\b(algorithm|architecture|implement|refactor|debug|optimize|analyse|analyze|"
    r"function|class|database|api|server|deploy|kubernetes|docker|regex|recursion|"
    r"async|concurrent|performance|security|vulnerability|migration|schema)\b",
    re.IGNORECASE,
)

_DEEP_ANALYSIS_TRIGGERS = re.compile(
    r"\b(explain in depth|deep dive|detailed analysis|full implementation|"
    r"step by step|walk me through|design a system|write a complete|"
    r"compare and contrast|pros and cons)\b",
    re.IGNORECASE,
)

_CODE_PATTERN = re.compile(r"```|`[^`]+`|\bdef\b|\bclass\b|\bimport\b|\bfunction\b")

_MULTI_STEP = re.compile(r"\b(first|then|also|and then|finally|additionally|furthermore)\b", re.IGNORECASE)


def classify_prompt(text: str) -> str:
    words = text.split()
    word_count = len(words)
    question_marks = text.count("?")

    has_code = bool(_CODE_PATTERN.search(text))
    has_deep = bool(_DEEP_ANALYSIS_TRIGGERS.search(text))
    tech_hits = len(_TECHNICAL_TERMS.findall(text))
    multi_step_hits = len(_MULTI_STEP.findall(text))

    score = 0
    score += min(word_count // 10, 4)       # 0-4: length
    score += min(question_marks, 2)          # 0-2: multiple questions
    score += 3 if has_code else 0
    score += 4 if has_deep else 0
    score += min(tech_hits, 3)               # 0-3: technical vocabulary
    score += min(multi_step_hits, 2)         # 0-2: multi-step reasoning

    if score <= 3:
        return MODEL_ALIASES["haiku"]
    elif score <= 9:
        return MODEL_ALIASES["sonnet"]
    else:
        return MODEL_ALIASES["opus"]


def parse_message(text: str) -> tuple[str, str]:
    """Return (model, prompt) after stripping any !<alias> prefix."""
    m = _FORCE_PREFIX_RE.match(text)
    if m:
        alias = m.group(1).lower()
        model = MODEL_ALIASES[alias]
        prompt = text[m.end():]
        return model, prompt
    model = classify_prompt(text)
    return model, text


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


async def run_claude(prompt: str, model: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [CLAUDE_BIN, "--print", "--dangerously-skip-permissions", "--model", model, prompt],
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
    aliases = ", ".join(f"!{a}" for a in MODEL_ALIASES)
    await update.message.reply_text(
        f"Hi {user.first_name}! Send me any message and I'll pass it to Claude.\n\n"
        f"Model is chosen automatically by prompt complexity. "
        f"Force a model with a prefix: {aliases}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("You are not authorised to use this bot.")
        return

    user_text = update.message.text
    model, prompt = parse_message(user_text)

    logger.info(
        "User %s (%d) [model=%s]: %s",
        user.username or user.first_name,
        user.id,
        model,
        prompt,
    )

    thinking_msg = await update.message.reply_text(f"Thinking... (model: {model})")

    try:
        response = await run_claude(prompt, model)
    except subprocess.TimeoutExpired:
        response = "Request timed out. Please try a shorter or simpler prompt."
    except Exception as e:
        logger.exception("Error running claude")
        response = f"Error: {e}"

    await thinking_msg.delete()

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
