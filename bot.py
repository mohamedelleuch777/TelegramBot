import asyncio
import logging
import os
import re
import sqlite3
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

# ── Config ────────────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CLAUDE_BIN         = os.getenv("CLAUDE_BIN", "claude")
GEMINI_BIN         = os.getenv("GEMINI_BIN", "gemini")
OPENAI_BIN         = os.getenv("OPENAI_BIN", "python openai_cli.py") # Path to our OpenAI CLI script
PROVIDER           = os.getenv("PROVIDER", "claude").lower()   # claude | gemini | openai
DB_PATH            = os.getenv("DB_PATH", "history.db")
HISTORY_WINDOW     = int(os.getenv("HISTORY_WINDOW", "20"))    # messages kept per user

CLAUDE_MODELS: dict[str, str] = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-7",
}
GEMINI_MODELS: dict[str, str] = {
    "haiku":  "gemini-2.5-flash-lite",
    "sonnet": "gemini-2.5-flash",
    "opus":   "gemini-2.5-pro",
}
OPENAI_MODELS: dict[str, str] = {
    "haiku":  "gpt-3.5-turbo", # Mapping to a faster/cheaper model
    "sonnet": "gpt-4o",        # Mapping to a balanced model
    "opus":   "gpt-4o",      # Mapping to a powerful model
}

# ── Prefix parsing ────────────────────────────────────────────────────────────

# Matches !claude, !gemini, !openai, !haiku, !sonnet, !opus at the start of a message
_PREFIX_RE = re.compile(
    r"^!(claude|gemini|openai|haiku|sonnet|opus)\s+",
    re.IGNORECASE,
)

# ── Complexity classifier ─────────────────────────────────────────────────────

_TECHNICAL_TERMS = re.compile(
    r"\b(algorithm|architecture|implement|refactor|debug|optimize|analyse|analyze|"
    r"function|class|database|api|server|deploy|kubernetes|docker|regex|recursion|"
    r"async|concurrent|performance|security|vulnerability|migration|schema|ai|ml|code|programming|system design)\b",
    re.IGNORECASE,
)
_DEEP_ANALYSIS = re.compile(
    r"\b(explain in depth|deep dive|detailed analysis|full implementation|"
    r"step by step|walk me through|design a system|write a complete|"
    r"compare and contrast|pros and cons)\b",
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(r"```|`[^`]+`|\bdef\b|\bclass\b|\bimport\b|\bfunction\b")
_MULTI_STEP    = re.compile(
    r"\b(first|then|also|and then|finally|additionally|furthermore)\b",
    re.IGNORECASE,
)


def classify_prompt(text: str) -> str:
    """Return tier: 'haiku' | 'sonnet' | 'opus'."""
    words = len(text.split())
    score = 0
    score += min(words // 3, 4)           # Even more weight for word count
    score += min(text.count("?"), 2)
    score += 3 if _CODE_PATTERN.search(text) else 0
    score += 4 if _DEEP_ANALYSIS.search(text) else 0
    score += min(len(_TECHNICAL_TERMS.findall(text)), 7) # Even more weight for technical terms
    score += min(len(_MULTI_STEP.findall(text)), 2)

    if score <= 5:
        return "haiku"
    elif score <= 14:
        return "sonnet"
    return "opus"


def parse_message(text: str) -> tuple[str, str, str]:
    """Return (provider, tier, prompt).

    Prefix rules:
      !claude / !gemini  → force provider, auto-classify tier
      !haiku/sonnet/opus → force tier, use default provider
    """
    m = _PREFIX_RE.match(text)
    if m:
        token = m.group(1).lower()
        prompt = text[m.end():]
        if token in ("claude", "gemini", "openai"):
            return token, classify_prompt(prompt), prompt
        else:  # model tier
            current_provider = os.getenv("PROVIDER", "claude").lower()
            return current_provider, token, prompt
    return PROVIDER, classify_prompt(text), text


# ── SQLite history ────────────────────────────────────────────────────────────

def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role    TEXT NOT NULL,
            content TEXT NOT NULL,
            ts      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()


def get_history(user_id: int) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT role, content FROM history WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, HISTORY_WINDOW),
    ).fetchall()
    con.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def save_turn(user_id: int, user_msg: str, assistant_msg: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.executemany(
        "INSERT INTO history (user_id, role, content) VALUES (?,?,?)",
        [(user_id, "user", user_msg), (user_id, "assistant", assistant_msg)],
    )
    con.commit()
    con.close()


def clear_history(user_id: int) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM history WHERE user_id=?", (user_id,))
    con.commit()
    con.close()


# ── Providers ─────────────────────────────────────────────────────────────────

def _build_transcript(history: list[dict], prompt: str) -> str:
    if not history:
        return prompt
    transcript = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    )
    return f"[Conversation so far]\n{transcript}\n\nUser: {prompt}\nAssistant:"


async def run_claude(prompt: str, tier: str, history: list[dict]) -> str:
    model = CLAUDE_MODELS[tier]
    full_prompt = _build_transcript(history, prompt)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [CLAUDE_BIN, "--print", "--dangerously-skip-permissions", "--model", model, full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        ),
    )
    if result.returncode != 0 and result.stderr:
        logger.error("claude stderr: %s", result.stderr)
    return result.stdout.strip() or result.stderr.strip() or "No response."


async def run_gemini(prompt: str, tier: str, history: list[dict]) -> str:
    model = GEMINI_MODELS[tier]
    full_prompt = _build_transcript(history, prompt)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            [GEMINI_BIN, "--yolo", "--model", model, "--prompt", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        ),
    )
    if result.returncode != 0 and result.stderr:
        logger.error("gemini stderr: %s", result.stderr)
    return result.stdout.strip() or result.stderr.strip() or "No response."


async def run_openai(prompt: str, tier: str, history: list[dict]) -> str:
    model = OPENAI_MODELS[tier]
    full_prompt = _build_transcript(history, prompt)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: subprocess.run(
            OPENAI_BIN.split() + ["--model", model, "--prompt", full_prompt],
            capture_output=True,
            text=True,
            timeout=120,
        ),
    )
    if result.returncode != 0 and result.stderr:
        logger.error("openai stderr: %s", result.stderr)
    return result.stdout.strip() or result.stderr.strip() or "No response."


async def run_ai(provider: str, tier: str, prompt: str, history: list[dict]) -> str:
    if provider == "gemini":
        return await run_gemini(prompt, tier, history)
    elif provider == "openai":
        return await run_openai(prompt, tier, history)
    return await run_claude(prompt, tier, history)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def split_message(text: str, limit: int = 4096) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


# ── Handlers ──────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("You are not authorised to use this bot.")
        return
    await update.message.reply_text(
        f"Hi {user.first_name}! Send me any message and I'll pass it to Claude.\n\n"
        f"*Provider prefixes:* !claude, !gemini, !openai\n"
        f"*Model prefixes:* !haiku, !sonnet, !opus\n"
        f"*Commands:* /clear — reset your conversation history\n\n"
        f"Default provider: `{PROVIDER}`",
        parse_mode="Markdown",
    )


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("You are not authorised to use this bot.")
        return
    clear_history(user.id)
    await update.message.reply_text("Conversation history cleared.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("You are not authorised to use this bot.")
        return

    user_text = update.message.text
    provider, tier, prompt = parse_message(user_text)
    history = get_history(user.id)

    logger.info(
        "User %s (%d) [%s/%s, history=%d]: %s",
        user.username or user.first_name,
        user.id,
        provider,
        tier,
        len(history),
        prompt,
    )

    thinking_msg = await update.message.reply_text(
        f"Thinking... (`{provider}` / `{tier}`)", parse_mode="Markdown"
    )

    try:
        response = await run_ai(provider, tier, prompt, history)
    except subprocess.TimeoutExpired:
        response = "Request timed out. Please try a shorter or simpler prompt."
    except Exception as e:
        logger.exception("Error running AI provider")
        response = f"Error: {e}"

    await thinking_msg.delete()

    save_turn(user.id, prompt, response)

    for chunk in split_message(response):
        await update.message.reply_text(chunk)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
    if not get_allowed_ids():
        raise RuntimeError("ALLOWED_USER_IDS is empty — no one would be able to use the bot")

    init_db()

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started. Provider=%s, Allowed IDs=%s", PROVIDER, get_allowed_ids())
    app.run_polling()


if __name__ == "__main__":
    pid_file = "/tmp/telegram-bot.pid"
    try:
        if os.path.exists(pid_file):
            old_pid = int(open(pid_file).read().strip())
            try:
                os.kill(old_pid, 0)
                raise SystemExit(f"Bot already running (PID {old_pid}). Stop it first or delete {pid_file}.")
            except ProcessLookupError:
                pass  # stale PID file — previous process is gone
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        main()
    finally:
        if os.path.exists(pid_file):
            os.remove(pid_file)
