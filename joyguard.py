from __future__ import annotations

import asyncio
import html
import random
import re
import time
from typing import Any

from aiogram import F, types
from aiogram.enums import ContentType
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import ChatMemberUpdatedFilter, Command, CommandObject, CommandStart, IS_MEMBER, IS_NOT_MEMBER
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

from joyguard_app.database import db
from joyguard_app.memory import (
    build_user_memory_context,
    choose_varied_entries,
    get_chat_history_entries,
    get_display_name,
    normalize_message_text,
    schedule_memory_capture,
    store_chat_history,
)
from joyguard_app.openrouter import call_openrouter
from joyguard_app.styles import (
    add_saved_style,
    delete_saved_style,
    extract_saved_style_id,
    get_active_saved_style_id,
    get_effective_ai_style,
    get_saved_style,
    get_saved_styles,
    get_user_custom_prompt,
    get_user_style,
    is_saved_style_key,
    reset_user_style,
    set_user_custom_prompt,
    set_user_style,
    validate_custom_style_prompt,
    validate_saved_style_name,
)
from joyguard_app.settings import (
    ADMIN_ID,
    AI_STYLE_PRESETS,
    CHAT_HISTORY_CHAR_LIMIT,
    CHAT_HISTORY_LIMIT,
    CHAT_MEMORY_CONTEXT_LIMIT,
    CHAT_MEMORY_DB_LIMIT,
    CHAT_MEMORY_MESSAGE_CHAR_LIMIT,
    CUSTOM_STYLE_KEY,
    CUSTOM_STYLE_MIN_LENGTH,
    CUSTOM_STYLE_PROMPT_LIMIT,
    DEFAULT_AI_STYLE,
    ECHO_REPLY_RESPONSES,
    MAX_RANK_ENTRIES,
    MAX_SAVED_STYLES,
    REQUIRED_CHANNEL,
    REQUIRED_CHANNEL_URL,
    SAVED_STYLE_PREFIX,
    SUBSCRIPTION_CACHE_TTL_FAIL,
    SUBSCRIPTION_CACHE_TTL_OK,
    SUPPORT_MEDIA_TYPES,
    SWEAR_RANK_ENTRIES,
    SWEAR_WORDS,
    TASK_REJECT_RESPONSES,
    TASK_REQUEST_KEYWORDS,
    USER_MEMORY_CONTEXT_LIMIT,
    WELCOME_TEXT,
    WORD_PATTERN,
    bot,
    dp,
    logger,
    subscription_cache,
)

BOT_ID: int | None = None
BOT_USERNAME: str | None = None


async def generate_ai_reply(
    message: types.Message,
    chat_history: list[str],
    chat_memories: list[str],
    user_memories: list[str]
) -> str | None:
    text = message.text or message.caption
    if not text:
        return None

    sections: list[str] = []
    if chat_history:
        trimmed = chat_history[-CHAT_HISTORY_LIMIT:]
        history_text = "\n".join(f"- {line}" for line in trimmed)
        sections.append(f"Внутренний пересказ беседы (не цитируй это напрямую):\n{history_text}")
    if chat_memories:
        varied_chat_memories = choose_varied_entries(chat_memories, CHAT_MEMORY_CONTEXT_LIMIT)
        memories_text = "\n".join(f"- {line}" for line in varied_chat_memories)
        sections.append(f"Мои наблюдения о теме разговора (держи в голове, но не раскрывай):\n{memories_text}")
    if user_memories:
        varied_user_memories = choose_varied_entries(user_memories, USER_MEMORY_CONTEXT_LIMIT)
        user_text = "\n".join(f"- {line}" for line in varied_user_memories)
        sections.append(f"Мои личные заметки о собеседниках (не рассказывай о них):\n{user_text}")

    context_block = "\n\n".join(sections) if sections else "Нет дополнительного контекста."
    author_name = get_display_name(message.from_user)
    payload_user = (
        f"Используй следующие внутренние заметки только мысленно, не проговаривай их:\n{context_block}\n\n"
        f"Сейчас тебе написали: {author_name} (ID {message.from_user.id if message.from_user else 'unknown'})."
        f" Ответь на это сообщение своим обычным язвительным стилем:\n{text}"
    )
    user_id = message.from_user.id if message.from_user else None
    style_key = get_effective_ai_style(user_id, default=DEFAULT_AI_STYLE)
    style_prompt: str | None = None
    if style_key == CUSTOM_STYLE_KEY:
        custom_prompt = get_user_custom_prompt(user_id)
        if not custom_prompt:
            logger.warning("Пользователь выбрал кастомный стиль, но описание пустое. Возвращаю стиль по умолчанию.")
            style_key = DEFAULT_AI_STYLE
        else:
            style_prompt = custom_prompt
    elif is_saved_style_key(style_key):
        saved_id = extract_saved_style_id(style_key)
        saved_style = get_saved_style(user_id, saved_id) if user_id else None
        if saved_style:
            style_prompt = saved_style["prompt"]
        else:
            logger.warning("Сохранённый стиль %s не найден, возвращаюсь к дефолту.", style_key)
            set_user_style(user_id, DEFAULT_AI_STYLE)
            style_key = DEFAULT_AI_STYLE
    style_prompt = style_prompt or AI_STYLE_PRESETS.get(style_key, AI_STYLE_PRESETS[DEFAULT_AI_STYLE])["prompt"]
    messages = [
        {"role": "system", "content": style_prompt},
        {"role": "user", "content": payload_user}
    ]
    return await call_openrouter(messages)


def summarize_message_text(message: types.Message) -> str:
    text = (message.text or message.caption or "").strip()
    if text:
        return text[:CHAT_MEMORY_MESSAGE_CHAR_LIMIT]
    return f"<{message.content_type}>"


def normalize_message_text(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"\s+", " ", value).strip()
    return normalized.lower() if normalized else None


def choose_varied_entries(entries: list[str], limit: int) -> list[str]:
    if limit <= 0 or len(entries) <= limit:
        return entries
    recent_keep = entries[:max(MEMORY_MIN_RECENT_SHARE, min(limit // 2, len(entries)))]
    remaining = entries[len(recent_keep):]
    to_pick = limit - len(recent_keep)
    if remaining and to_pick > 0:
        sampled = random.sample(remaining, min(to_pick, len(remaining)))
        recent_keep += sampled
    return recent_keep


def is_echo_of_bot_message(message: types.Message) -> bool:
    if not message.reply_to_message or not BOT_ID:
        return False
    replied = message.reply_to_message
    if not replied.from_user or replied.from_user.id != BOT_ID:
        return False
    current_text = normalize_message_text(message.text or message.caption)
    replied_text = normalize_message_text(replied.text or replied.caption)
    if not current_text or not replied_text:
        return False
    return current_text == replied_text


async def send_echo_response(message: types.Message) -> None:
    response = random.choice(ECHO_REPLY_RESPONSES)
    try:
        await message.reply(response)
    except Exception as exc:
        logger.error(f"Не удалось отправить ответ на копию сообщения: {exc}")


def is_task_request(message: types.Message) -> bool:
    text = (message.text or message.caption or "").lower()
    if not text:
        return False
    if any(keyword in text for keyword in TASK_REQUEST_KEYWORDS):
        return True
    return False


async def send_task_reject(message: types.Message) -> None:
    response = random.choice(TASK_REJECT_RESPONSES)
    try:
        await message.reply(response)
    except Exception as exc:
        logger.error(f"Не удалось отправить отказ на запрос: {exc}")


def message_mentions_bot(message: types.Message) -> bool:
    if not BOT_USERNAME:
        return False
    tag = f"@{BOT_USERNAME.lower()}"

    def text_has_tag(text: str | None, entities: list[types.MessageEntity] | None) -> bool:
        if not text:
            return False
        lowered = text.lower()
        if tag in lowered:
            return True
        if not entities:
            return False
        for entity in entities:
            if entity.type == "mention":
                mention_text = text[entity.offset: entity.offset + entity.length]
                if mention_text.lower() == tag:
                    return True
        return False

    return text_has_tag(message.text, message.entities) or text_has_tag(message.caption, message.caption_entities)
# ==================== FSM States ====================
class BotStates(StatesGroup):
    waiting_global_autoresponder = State()
    waiting_support_message = State()
    waiting_admin_reply = State()  # Ожидание ответа админа
    waiting_custom_style = State()
    waiting_saved_style_name = State()
    waiting_saved_style_prompt = State()

# ==================== Клавиатуры ====================
def get_main_keyboard():
    """Главная клавиатура в личных сообщениях"""
    keyboard = ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="✍️ Глобальный автоответчик")],
            [KeyboardButton(text="🎭 Стиль общения"), KeyboardButton(text="👨‍🔧 Тех.поддержка")],
            [KeyboardButton(text="❓ Помощь")]
        ],
        resize_keyboard=True
    )
    return keyboard


SUBSCRIBE_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔔 Подписаться", url=REQUIRED_CHANNEL_URL)],
    [InlineKeyboardButton(text="✅ Проверить подписку", callback_data="check_subscription")]
])

SUBSCRIBE_GROUP_KEYBOARD = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text="🔔 Подписаться на Silent Power", url=REQUIRED_CHANNEL_URL)]
])


async def is_user_subscribed(user_id: int) -> bool:
    if not REQUIRED_CHANNEL:
        return True
    now = time.time()
    cached = subscription_cache.get(user_id)
    if cached and cached[1] > now:
        return cached[0]
    try:
        member = await bot.get_chat_member(REQUIRED_CHANNEL, user_id)
        status = member.status in {"member", "administrator", "creator"}
    except TelegramBadRequest:
        status = False

    ttl = SUBSCRIPTION_CACHE_TTL_OK if status else SUBSCRIPTION_CACHE_TTL_FAIL
    subscription_cache[user_id] = (status, now + ttl)
    return status


async def ensure_channel_subscription(message: types.Message) -> bool:
    if message.chat.type != "private" or not REQUIRED_CHANNEL:
        return True
    if await is_user_subscribed(message.from_user.id):
        return True
    await message.answer(
        "Чтобы пользоваться ботом, подпишитесь на канал Silent и вернитесь сюда.",
        reply_markup=SUBSCRIBE_KEYBOARD
    )
    return False


async def ensure_group_subscription(message: types.Message) -> bool:
    if message.chat.type not in {"group", "supergroup"} or not REQUIRED_CHANNEL or not message.from_user:
        return True
    if await is_user_subscribed(message.from_user.id):
        return True
    await send_temp_answer(
        message,
        "Чтобы пользоваться командами SpringtrapSilent, подпишитесь на канал Silent Power и повторите команду.",
        reply_markup=SUBSCRIBE_GROUP_KEYBOARD
    )
    return False


def build_support_admin_keyboard(user_id: int) -> InlineKeyboardMarkup:
    ban_info = db.get_support_ban(user_id) or {"block_media": False, "block_all": False}
    media_text = "🚫 Запретить медиа" if not ban_info["block_media"] else "♻️ Разрешить медиа"
    full_text = "⛔️ Полный бан" if not ban_info["block_all"] else "♻️ Разрешить пользователя"
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="💬 Ответить", callback_data=f"reply_{user_id}")],
        [
            InlineKeyboardButton(text=media_text, callback_data=f"support_media_{user_id}"),
            InlineKeyboardButton(text=full_text, callback_data=f"support_full_{user_id}")
        ]
    ])


async def send_temp_answer(message: types.Message, text: str, *, delay: int = 20, **kwargs) -> None:
    """Отправляет ответ, который автоматически удалится через delay секунд."""
    sent_message = await message.answer(text, **kwargs)

    async def _delete_later():
        try:
            await asyncio.sleep(delay)
            await sent_message.delete()
        except Exception as e:
            logger.debug(f"Не удалось удалить временное сообщение: {e}")

    asyncio.create_task(_delete_later())


def record_user_profiles_from_message(message: types.Message):
    """Сохранить информацию об участвующих пользователях для поиска по username."""
    if message.from_user:
        db.upsert_user_profile(message.from_user)
    if message.reply_to_message and message.reply_to_message.from_user:
        db.upsert_user_profile(message.reply_to_message.from_user)


def extract_mentioned_usernames(message: types.Message) -> list[str]:
    usernames: list[str] = []

    def _extract_from(text: str | None, entities: list[types.MessageEntity] | None):
        if not text or not entities:
            return
        for entity in entities:
            if entity.type == "mention":
                mention_text = text[entity.offset: entity.offset + entity.length]
                if mention_text.startswith("@"):
                    usernames.append(mention_text[1:])

    _extract_from(message.text, message.entities)
    _extract_from(message.caption, message.caption_entities)
    return usernames


def gather_targets_from_message(message: types.Message) -> list[dict]:
    """Возвращает список пользователей, которых мог адресовать отправитель (ответ или упоминание)."""
    targets: list[dict] = []
    seen_ids: set[int] = set()
    seen_usernames: set[str] = set()

    def add_target(user_id: int | None, display_name: str | None, username: str | None = None):
        if user_id:
            if user_id in seen_ids:
                return
            seen_ids.add(user_id)
        elif username:
            uname = username.lower()
            if uname in seen_usernames:
                return
            seen_usernames.add(uname)
        else:
            return

        name = display_name or (f"@{username}" if username else (f"ID{user_id}" if user_id else ""))
        targets.append({"user_id": user_id, "name": name or None, "username": username})

    # Адресат из ответа
    if message.reply_to_message and message.reply_to_message.from_user:
        target_user = message.reply_to_message.from_user
        db.upsert_user_profile(target_user)
        add_target(target_user.id, target_user.first_name, target_user.username)

    def process_entities(text: str | None, entities: list[types.MessageEntity] | None):
        if not text or not entities:
            return
        for entity in entities:
            if entity.type == "text_mention" and entity.user:
                db.upsert_user_profile(entity.user)
                add_target(entity.user.id, entity.user.first_name, entity.user.username)
            elif entity.type == "mention":
                mention_text = text[entity.offset: entity.offset + entity.length]
                if mention_text.startswith("@"):
                    username = mention_text[1:]
                    profile = db.get_user_by_username(username)
                    if profile:
                        add_target(
                            profile["user_id"],
                            profile.get("first_name"),
                            profile.get("username")
                        )
                    else:
                        add_target(None, mention_text, username)

    process_entities(message.text, message.entities)
    process_entities(message.caption, message.caption_entities)

    return targets


def count_swears_in_text(text: str | None) -> int:
    if not text:
        return 0
    lower_text = text.lower()
    tokens = WORD_PATTERN.findall(lower_text)
    return sum(1 for token in tokens if token in SWEAR_WORDS)


async def process_swear_stats(message: types.Message):
    if message.chat.type in {"group", "supergroup"} and message.from_user:
        combined_text_parts = [part for part in (message.text, message.caption) if part]
        if not combined_text_parts:
            return
        joined_text = " \n".join(combined_text_parts)
        lower_joined = joined_text.lower()
        if not any(word in lower_joined for word in SWEAR_WORDS):
            return
        swear_count = count_swears_in_text(joined_text)
        if swear_count > 0:
            record_user_profiles_from_message(message)
            db.increment_swear(message.chat.id, message.from_user.id, swear_count)


async def send_swear_ranking(message: types.Message):
    ranking = db.get_swear_ranking(message.chat.id, SWEAR_RANK_ENTRIES)
    if not ranking:
        await message.answer("📊 В этом чате пока нет данных по матам.")
        return

    lines = ["🤬 Топ по матюкам:\n"]
    for idx, (user_id, count) in enumerate(ranking, start=1):
        name = await get_chat_user_name(message.chat.id, user_id)
        lines.append(f"{idx}. {name} — {count}")

    await message.answer("\n".join(lines))


async def get_chat_user_name(chat_id: int, user_id: int) -> str:
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        user = member.user
        if user.full_name:
            return user.full_name
        if user.username:
            return f"@{user.username}"
    except Exception:
        pass
    return f"ID{user_id}"


async def send_block_profile(message: types.Message, target_user_id: int, title_name: str | None = None):
    blocked_ids = db.get_blocks_by_blocker(message.chat.id, target_user_id)
    display_name = title_name or await get_chat_user_name(message.chat.id, target_user_id)
    text_lines = [
        f"📊 Профиль блокировок: {display_name}",
        f"Всего заблокировано: {len(blocked_ids)}"
    ]

    if blocked_ids:
        text_lines.append("\nЗаблокированы:")
        for idx, blocked_id in enumerate(blocked_ids, start=1):
            blocked_name = await get_chat_user_name(message.chat.id, blocked_id)
            text_lines.append(f"{idx}. {blocked_name}")
    else:
        text_lines.append("\nПока никого не заблокировал(а).")

    await message.answer("\n".join(text_lines))


async def send_block_ranking(message: types.Message):
    blocks = db.get_chat_blocks(message.chat.id)
    if not blocks:
        await message.answer("📋 В этом чате нет активных блокировок.")
        return

    stats: dict[int, int] = {}
    for blocker_id, _ in blocks:
        stats[blocker_id] = stats.get(blocker_id, 0) + 1

    ranking = sorted(stats.items(), key=lambda item: (-item[1], item[0]))[:MAX_RANK_ENTRIES]

    lines = ["🏆 Рейтинг блокировок чата:\n"]
    for idx, (user_id, count) in enumerate(ranking, start=1):
        name = await get_chat_user_name(message.chat.id, user_id)
        lines.append(f"{idx}. {name} — {count}")

    await message.answer("\n".join(lines))


def remove_target_mentions(text: str, targets: list[dict]) -> str:
    if not text:
        return text
    result = text
    for target in targets:
        username = target.get("username")
        if username:
            pattern = rf"@{re.escape(username)}\b"
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    # Удаляем лишние пробелы
    result = re.sub(r"\s+", " ", result)
    return result.strip()


def extract_personal_message(after_command_text: str, targets: list[dict]) -> str | None:
    if not after_command_text:
        return None

    candidate = after_command_text.strip()

    newline_index = candidate.find('\n')
    if newline_index != -1:
        candidate = candidate[newline_index + 1:]

    candidate = candidate.lstrip("-—:").strip()
    candidate = remove_target_mentions(candidate, targets)
    return candidate or None


async def resolve_targets_with_fetch(chat_id: int, targets: list[dict]):
    for target in targets:
        if target.get("user_id") or not target.get("username"):
            continue
        username = target["username"]
        resolved_user = None

        username_with_at = username if username.startswith("@") else f"@{username}"
        try:
            chat_obj = await bot.get_chat(username_with_at)
            if chat_obj and getattr(chat_obj, "type", None) == "private":
                resolved_user = chat_obj
        except TelegramBadRequest:
            resolved_user = None

        if resolved_user is None:
            continue

        target["user_id"] = resolved_user.id
        target["name"] = resolved_user.first_name or getattr(resolved_user, "full_name", None) or target.get("name") or username_with_at
        target["username"] = resolved_user.username or username
        db.upsert_user_profile(resolved_user)

# ==================== Обработчики команд ====================

@dp.my_chat_member(ChatMemberUpdatedFilter(IS_NOT_MEMBER >> IS_MEMBER))
async def on_bot_added(event: types.ChatMemberUpdated):
    """Обработчик добавления бота в группу"""
    if event.chat.type not in {"group", "supergroup"}:
        return
    added_by = getattr(event, "from_user", None)
    await event.answer(
        "👋 Спасибо за добавление SpringtrapSilent!\n\n"
        "📝 Доступные команды:\n"
        "• Ответьте на сообщение пользователя командой 'Спринг стоп' для блокировки\n"
        "• 'Спринг стоп' + текст для установки персонального автоответчика\n"
        "• 'Спринг список' для просмотра блокировок в чате\n"
        "• 'Топ маты' / 'Топ матов' для рейтинга по количеству матов\n"
        "• 'Спринг стоп все' для включения/выключения режима и указания персонального автоответчика (либо глобального в ЛС)\n"
        "• Командой 'Спринг стоп' по конкретному пользователю можно убрать его из общего блок-листа\n"
        "• Можно упоминать бота или отвечать ему, чтобы пообщаться с встроенным ИИ (стиль переключается в личке)\n\n"
        "⚠️ ВАЖНО: Сделайте бота администратором с правом удаления сообщений!\n\n"
        + ("ℹ️ Чтобы пользоваться командами бота, подпишитесь на [канал](https://t.me/silentpower_V).\n\n"
           if REQUIRED_CHANNEL else "")
        + "💬 Напишите мне в личку для настройки глобального автоответчика.",
        parse_mode="Markdown",
        disable_web_page_preview=True
    )

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    """Обработчик команды /start (только в личных сообщениях)"""
    if message.chat.type != "private":
        return
    if not await ensure_channel_subscription(message):
        return

    await message.answer(WELCOME_TEXT, reply_markup=get_main_keyboard())

@dp.message(F.text.func(lambda text: isinstance(text, str) and text.lower().startswith("спринг список")))
async def cmd_list(message: types.Message):
    """Команда 'Спринг список' - рейтинг и профили блокировок"""
    if message.chat.type == "private":
        await message.answer("Эта команда работает только в групповых чатах.")
        return
    if not await ensure_group_subscription(message):
        return

    text = message.text.strip()
    lower_text = text.lower()

    record_user_profiles_from_message(message)
    targets = gather_targets_from_message(message)
    await resolve_targets_with_fetch(message.chat.id, targets)

    if lower_text.startswith("спринг список мой"):
        await send_block_profile(message, message.from_user.id)
        return

    if targets:
        target = targets[0]
        target_id = target.get("user_id")
        target_name = target.get("name")
        if target_id:
            await send_block_profile(message, target_id, target_name)
        else:
            await send_temp_answer(
                message,
                "❌ Не удалось определить пользователя. Убедитесь, что он ранее писал в чате."
            )
        return

    await send_block_ranking(message)

@dp.message(F.text.func(lambda text: isinstance(text, str) and text.strip().lower() == "бот"))
async def ping_bot(message: types.Message):
    """Простая проверка активности по слову 'бот'"""
    await message.answer("Че надо")

@dp.message(F.text.func(lambda text: isinstance(text, str) and "спринг стоп" in text.lower()))
async def cmd_joy_stop(message: types.Message):
    """Команда 'Спринг стоп' - блокировка/разблокировка"""
    if message.chat.type == "private":
        await message.answer("Эта команда работает только в групповых чатах.")
        return
    if not await ensure_group_subscription(message):
        return
    
    blocker_id = message.from_user.id
    record_user_profiles_from_message(message)
    targets = gather_targets_from_message(message)
    await resolve_targets_with_fetch(message.chat.id, targets)
    text = message.text
    text_lower = text.lower()
    cmd_pos = text_lower.find("спринг стоп")
    if cmd_pos == -1:
        return

    after_command_text = text[cmd_pos + len("спринг стоп"):]
    tail_lower = text_lower[cmd_pos:].lstrip()

    # Обработка режима "Спринг стоп все"
    global_block_enabled, global_block_message = db.get_global_block(message.chat.id, blocker_id)

    if tail_lower.startswith("спринг стоп все"):
        remaining_text = text[cmd_pos + len("спринг стоп все"):]
        global_message = extract_personal_message(remaining_text, targets)
        enabled = db.toggle_global_block(message.chat.id, blocker_id, global_message)
        blocker_name = message.from_user.first_name
        if enabled:
            if global_message:
                response = (
                    f"🔒 {blocker_name} включил(а) режим 'Спринг стоп все'. Никто не может отвечать на его сообщения.\n\n"
                    f"Персональный ответ:\n{global_message}"
                )
            else:
                response = f"🔒 {blocker_name} включил(а) режим 'Спринг стоп все'. Никто не может отвечать на его сообщения."
        else:
            response = f"🔓 {blocker_name} отключил(а) режим 'Спринг стоп все'. Теперь пользователи снова могут отвечать."
        await send_temp_answer(message, response)
        return

    personal_message = extract_personal_message(after_command_text, targets)

    # Обычный режим требует указать пользователя (ответом или @username)
    if not targets:
        await send_temp_answer(
            message,
            "❌ Укажите пользователя: ответьте на его сообщение или добавьте @username в команду."
        )
        return

    target = targets[0]
    blocked_id = target.get("user_id")
    blocked_name = target.get("name") or "пользователь"

    if not blocked_id:
        await send_temp_answer(
            message,
            "❌ Не удалось определить пользователя. Убедитесь, что он ранее писал в чате."
        )
        return

    # Нельзя заблокировать самого себя
    if blocker_id == blocked_id:
        await message.answer("❌ Вы не можете заблокировать самого себя.")
        return

    # Если включен "Спринг стоп все", то команда работает как исключение
    if global_block_enabled:
        allowed = db.toggle_global_block_exception(message.chat.id, blocker_id, blocked_id)
        blocker_name = message.from_user.first_name
        if allowed:
            response = (
                f"🔓 {blocker_name} разрешил(а) пользователю {blocked_name} отвечать, даже когда включён режим 'Спринг стоп все'."
            )
        else:
            response = (
                f"🔒 {blocker_name} снова запретил(а) пользователю {blocked_name} отвечать в режиме 'Спринг стоп все'."
            )
        await send_temp_answer(message, response)
        return

    # Переключаем блокировку
    is_blocked = db.toggle_block(
        message.chat.id,
        blocker_id,
        blocked_id,
        personal_message
    )

    blocker_name = message.from_user.first_name
    blocked_name = target.get("name") or "пользователь"

    if is_blocked:
        if personal_message:
            response = f"🔒 {blocker_name} запретил(а) пользователю {blocked_name} отвечать на свои сообщения и установил(а) персональный автоответчик."
        else:
            response = f"🔒 {blocker_name} запретил(а) пользователю {blocked_name} отвечать на свои сообщения."
    else:
        response = f"🔓 {blocker_name} разрешил(а) пользователю {blocked_name} снова отвечать на свои сообщения."

    await send_temp_answer(message, response)


@dp.message(F.text.func(
    lambda text: isinstance(text, str) and text.strip().lower().startswith(("топ маты", "топ матов"))
))
async def cmd_swear_top(message: types.Message):
    if message.chat.type != "private":
        if not await ensure_group_subscription(message):
            return
        await send_swear_ranking(message)
    else:
        await message.answer("Команда работает только в групповых чатах.")
        return

async def maybe_reply_with_ai(message: types.Message, targets: list[dict] | None = None) -> None:
    if not message.from_user or message.from_user.is_bot:
        return
    if not (message.text or message.caption):
        return

    replied_to_bot = (
        message.reply_to_message
        and message.reply_to_message.from_user
        and BOT_ID is not None
        and message.reply_to_message.from_user.id == BOT_ID
    )
    mentioned_bot = message_mentions_bot(message)

    if not replied_to_bot and not mentioned_bot:
        return

    if not await ensure_group_subscription(message):
        return

    if replied_to_bot and is_echo_of_bot_message(message):
        await send_echo_response(message)
        return

    if is_task_request(message):
        await send_task_reject(message)
        return

    if targets is None:
        targets = gather_targets_from_message(message)

    history_entries = get_chat_history_entries(message.chat.id)
    chat_memories = db.get_chat_memories(message.chat.id, CHAT_MEMORY_CONTEXT_LIMIT)
    user_memory_context = build_user_memory_context(message.chat.id, targets)
    reply_text = await generate_ai_reply(message, history_entries, chat_memories, user_memory_context)
    if not reply_text:
        return

    try:
        await message.reply(reply_text)
    except Exception as exc:
        logger.error(f"Не удалось отправить ответ через Grok: {exc}")


@dp.message((F.chat.type == "group") | (F.chat.type == "supergroup"))
@dp.message((F.chat.type == "group") | (F.chat.type == "supergroup"))
async def check_reply_block(message: types.Message):
    """Проверка сообщений на попытку связаться с пользователем, который ограничил ответы."""
    if not message.from_user:
        return

    store_chat_history(message)

    await process_swear_stats(message)

    record_user_profiles_from_message(message)
    targets = gather_targets_from_message(message)
    schedule_memory_capture(message, targets)

    await maybe_reply_with_ai(message, targets)

    replier_id = message.from_user.id

    if not targets:
        return

    blocked_target = None
    blocker_id = None
    personal_message = None

    for target in targets:
        target_id = target.get("user_id")
        if not target_id:
            continue

        global_block_enabled, global_block_message = db.get_global_block(message.chat.id, target_id)
        if global_block_enabled and not db.is_global_block_exception(message.chat.id, target_id, replier_id):
            blocked_target = target
            blocker_id = target_id
            personal_message = global_block_message
            break

        is_blocked, personal_msg = db.is_blocked(message.chat.id, target_id, replier_id)
        if is_blocked:
            blocked_target = target
            blocker_id = target_id
            personal_message = personal_msg
            break

    if not blocked_target:
        return

    try:
        await message.delete()

        autoresponder = personal_message or db.get_global_autoresponder(blocker_id)
        if not autoresponder:
            autoresponder = "Пользователь установил ограничение на ответы к своим сообщениям."

        replier_mention = message.from_user.mention_html()
        target_name = blocked_target.get("name") or "этот пользователь"
        text = (
            f"{replier_mention}, {html.escape(target_name)} установил(а) для вас следующий ответ:\n\n"
            f"\"{html.escape(autoresponder)}\""
        )

        await send_temp_answer(message, text, parse_mode="HTML")

    except Exception as e:
        logger.error(f"Ошибка при обработке заблокированного сообщения: {e}")
        await message.answer(
            "⚠️ Не удалось удалить сообщение. Убедитесь, что бот является администратором с правом удаления сообщений."
        )

# ==================== Обработчики для личных сообщений ====================

@dp.message(F.text == "✍️ Глобальный автоответчик")
async def global_autoresponder_menu(message: types.Message, state: FSMContext):
    """Меню глобального автоответчика"""
    if message.chat.type != "private":
        return
    if not await ensure_channel_subscription(message):
        return
    
    # Очищаем любое предыдущее состояние
    await state.clear()
    
    current = db.get_global_autoresponder(message.from_user.id)
    
    text = "✍️ Глобальный автоответчик\n\n"
    if current:
        text += f"Текущий автоответчик:\n\"{current}\"\n\n"
    else:
        text += "У вас пока не установлен глобальный автоответчик.\n\n"
    
    text += "Отправьте мне новый текст автоответчика или /cancel для отмены."
    
    await message.answer(text)
    await state.set_state(BotStates.waiting_global_autoresponder)

@dp.message(BotStates.waiting_global_autoresponder)
async def save_global_autoresponder(message: types.Message, state: FSMContext):
    """Сохранение глобального автоответчика"""
    if not await ensure_channel_subscription(message):
        await state.clear()
        return
    # Проверяем, нажата ли кнопка меню
    if message.text == "👨‍🔧 Тех.поддержка":
        await state.clear()
        await support_menu(message, state)
        return
    
    if message.text == "❓ Помощь":
        await state.clear()
        await help_menu(message, state)
        return
    
    if message.text == "/cancel":
        await state.clear()
        await message.answer("❌ Отменено.", reply_markup=get_main_keyboard())
        return
    
    db.set_global_autoresponder(message.from_user.id, message.text)
    await state.clear()
    await message.answer(
        "✅ Глобальный автоответчик успешно установлен!",
        reply_markup=get_main_keyboard()
    )


@dp.callback_query(F.data == "style_saved_menu")
async def open_saved_styles_menu(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    if callback.message.chat.type != "private":
        await callback.answer("Доступно только в личке", show_alert=True)
        return

    styles = get_saved_styles(user_id)
    active_id = get_active_saved_style_id(user_id)
    lines = ["💾 Твои сохранённые стили:"]
    if styles:
        for style in styles:
            mark = " (активен)" if style["id"] == active_id else ""
            lines.append(f"• {style['name']}{mark}")
    else:
        lines.append("Пока пусто. Нажми '➕ Новый стиль', чтобы сохранить первый пресет.")

    await callback.message.answer("\n".join(lines), reply_markup=build_saved_styles_keyboard(user_id))
    await callback.answer()


@dp.callback_query(F.data == "style_saved_back")
async def saved_styles_back(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    text, keyboard = build_style_menu_view(user_id)
    await callback.message.answer(text, reply_markup=keyboard)
    await callback.answer()


@dp.callback_query(F.data == "style_saved_add")
async def saved_styles_add(callback: types.CallbackQuery, state: FSMContext):
    user_id = callback.from_user.id
    styles = get_saved_styles(user_id)
    if len(styles) >= MAX_SAVED_STYLES:
        await callback.answer("Лимит 5 стилей уже достигнут.", show_alert=True)
        return
    await state.set_state(BotStates.waiting_saved_style_name)
    await state.update_data(saved_style_user_id=user_id)
    await callback.message.answer(
        "🆕 Придумай название для нового стиля (2-40 символов).\n"
        "Команда /cancel отменяет сохранение."
    )
    await callback.answer()


@dp.callback_query(F.data.startswith("style_saved_use_"))
async def saved_style_activate(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    try:
        style_id = int(callback.data.split("_")[-1])
    except ValueError:
        await callback.answer("Не удалось понять стиль", show_alert=True)
        return
    style = get_saved_style(user_id, style_id)
    if not style:
        await callback.answer("Этот стиль уже удалён", show_alert=True)
        return
    set_user_style(user_id, f"{SAVED_STYLE_PREFIX}{style_id}", saved_style_id=style_id)
    await refresh_style_menu_message(callback.message, user_id)
    await callback.answer(f"Стиль '{style['name']}' активирован")


@dp.callback_query(F.data.startswith("style_saved_del_"))
async def saved_style_delete(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    try:
        style_id = int(callback.data.split("_")[-1])
    except ValueError:
        await callback.answer("Некорректный стиль", show_alert=True)
        return
    style = get_saved_style(user_id, style_id)
    if not style:
        await callback.answer("Уже удалён")
        return
    deleted = delete_saved_style(user_id, style_id)
    if deleted:
        current = get_user_style(user_id)
        if current == f"{SAVED_STYLE_PREFIX}{style_id}":
            set_user_style(user_id, DEFAULT_AI_STYLE)
        await refresh_style_menu_message(callback.message, user_id)
        await callback.answer(f"Стиль '{style['name']}' удалён")
    else:
        await callback.answer("Не удалось удалить", show_alert=True)


@dp.message(BotStates.waiting_saved_style_name)
async def handle_saved_style_name(message: types.Message, state: FSMContext):
    if message.chat.type != "private":
        await message.answer("Создавать стиль можно только в личке.")
        return
    if message.text and message.text.strip() == "/cancel":
        await state.clear()
        await message.answer("❌ Сохранение стиля отменено.")
        return

    data = await state.get_data()
    user_id = data.get("saved_style_user_id") or message.from_user.id
    name = (message.text or "").strip()
    is_valid, error = validate_saved_style_name(name, user_id)
    if not is_valid:
        await message.answer(error)
        return

    await state.update_data(saved_style_name=name)
    await state.set_state(BotStates.waiting_saved_style_prompt)
    await message.answer(
        "✍️ Теперь опиши поведение стиля (15-600 символов).\n"
        "Команда /cancel отменяет сохранение."
    )


@dp.message(BotStates.waiting_saved_style_prompt)
async def handle_saved_style_prompt(message: types.Message, state: FSMContext):
    if message.chat.type != "private":
        await message.answer("Создавать стиль можно только в личке.")
        return
    if message.text and message.text.strip() == "/cancel":
        await state.clear()
        await message.answer("❌ Сохранение стиля отменено.")
        return

    data = await state.get_data()
    user_id = data.get("saved_style_user_id") or message.from_user.id
    name = data.get("saved_style_name")
    if not name:
        await state.clear()
        await message.answer("Что-то пошло не так, попробуй снова." )
        return

    prompt = (message.text or "").strip()
    is_valid, error = await validate_custom_style_prompt(prompt)
    if not is_valid:
        await message.answer(error)
        return

    style = add_saved_style(user_id, name, prompt)
    set_user_style(user_id, f"{SAVED_STYLE_PREFIX}{style['id']}", saved_style_id=style["id"])
    await state.clear()
    await message.answer(
        f"✅ Стиль '{name}' сохранён и активирован!",
        reply_markup=get_main_keyboard()
    )

@dp.message(F.text == "👨‍🔧 Тех.поддержка")
async def support_menu(message: types.Message, state: FSMContext):
    """Меню тех.поддержки"""
    if message.chat.type != "private":
        return
    if not await ensure_channel_subscription(message):
        return
    
    # Очищаем любое предыдущее состояние
    await state.clear()
    
    await message.answer(
        "👨‍🔧 Тех.поддержка\n\n"
        "Опишите вашу проблему или вопрос, и я передам его администраторам. Можете приложить медиа.\n\n"
        "Отправьте /cancel для отмены."
    )
    await state.set_state(BotStates.waiting_support_message)


@dp.message(BotStates.waiting_support_message)
async def save_support_message(message: types.Message, state: FSMContext):
    """Сохранение сообщения в тех.поддержку"""
    if not await ensure_channel_subscription(message):
        await state.clear()
        return
    # Проверяем, нажата ли кнопка меню
    if message.text == "✍️ Глобальный автоответчик":
        await state.clear()
        await global_autoresponder_menu(message, state)
        return

    if message.text == "❓ Помощь":
        await state.clear()
        await help_menu(message, state)
        return
    
    if message.text == "/cancel":
        await state.clear()
        await message.answer("❌ Отменено.", reply_markup=get_main_keyboard())
        return
    
    ban_info = db.get_support_ban(message.from_user.id)
    if ban_info and ban_info["block_all"]:
        await message.answer(
            "",
            reply_markup=get_main_keyboard()
        )
        await state.clear()
        return

    if ban_info and ban_info["block_media"] and message.content_type in SUPPORT_MEDIA_TYPES:
        await message.answer("🚫 Вам запрещено отправлять медиа в техподдержку. Опишите проблему текстом.")
        return

    # Проверка антиспама
    can_send, wait_time = db.can_send_support_message(message.from_user.id, cooldown_seconds=30)
    if not can_send:
        await message.answer(
            f"⏰ Пожалуйста, подождите {wait_time} сек. перед отправкой следующего сообщения.",
            reply_markup=get_main_keyboard()
        )
        await state.clear()
        return

    # Сохраняем в БД
    stored_text = message.text or message.caption or f"<{message.content_type}>"
    db.save_support_message(message.from_user.id, stored_text)

    # Отправляем администратору, если ID указан
    if ADMIN_ID:
        try:
            admin_id = int(ADMIN_ID)
            user_info = f"От: {message.from_user.first_name}"
            if message.from_user.username:
                user_info += f" (@{message.from_user.username})"
            user_info += f"\nID: {message.from_user.id}"

            keyboard = build_support_admin_keyboard(message.from_user.id)

            header_lines = ["📩 Новое сообщение в тех.поддержку:", "", user_info]
            if message.text:
                header_lines.append("\nСообщение:\n" + message.text)
            elif message.caption:
                header_lines.append("\nПодпись:\n" + message.caption)
            else:
                header_lines.append(f"\nТип контента: {message.content_type}")

            await bot.send_message(
                admin_id,
                "\n".join(header_lines),
                reply_markup=keyboard
            )

            if message.content_type in SUPPORT_MEDIA_TYPES:
                await message.copy_to(admin_id)

            success_text = "✅ Ваше сообщение отправлено администратору!\n" \
                          "Он свяжется с вами в ближайшее время."
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения админу: {e}")
            success_text = "✅ Ваше сообщение сохранено!\n" \
                          "Администраторы увидят его при следующей проверке."
    else:
        success_text = "✅ Ваше сообщение сохранено в базу данных!\n" \
                      "Для прямой отправки администратору добавьте ADMIN_ID в .env файл."
    
    await state.clear()
    await message.answer(success_text, reply_markup=get_main_keyboard())

@dp.message(F.text == "❓ Помощь")
async def help_menu(message: types.Message, state: FSMContext):
    """Меню помощи"""
    if message.chat.type != "private":
        return
    if not await ensure_channel_subscription(message):
        await state.clear()
        return
    
    # Очищаем состояние при переходе в помощь
    await state.clear()
    
    await message.answer(
        "❓ Помощь по SpringtrapSilent\n\n"
        "📝 Команды в групповых чатах:\n\n"
        "1️⃣ Спринг стоп\n"
        "Ответьте на сообщение пользователя этой командой, чтобы заблокировать/разблокировать ему возможность отвечать на ваши сообщения.\n\n"
        "1️⃣➕ Спринг стоп все\n"
        "Останавливает всех: никто не сможет отвечать на ваши сообщения до повторного выключения.\n\n"
        "2️⃣ Спринг стоп + текст\n"
        "Напишите команду 'Спринг стоп' и с новой строки ваш текст автоответчика. "
        "Этот текст будет показываться заблокированному пользователю при попытке ответить вам.\n\n"
        "3️⃣ Спринг список\n"
        "Показывает список всех блокировок в текущем чате.\n\n"
        "4️⃣ Топ маты / Топ матов\n"
        "Выводит рейтинг пользователей чата по количеству зафиксированных матов.\n\n"
        "⚙️ Настройки и ИИ в личных сообщениях:\n\n"
        "• Глобальный автоответчик — текст по умолчанию для блокировок\n"
        "• Стиль общения — восемь готовых режимов (токсик, фембой, клерк, гот, батя, аристократ, хулиган) или ваш кастом\n"
        "• ИИ-агент — ответит, если упомянуть бота или написать ему в ответ, учитывает память чата\n"
        "• Тех.поддержка — связь с администраторами\n"
        "• Помощь — это сообщение\n\n"
        "⚠️ Важно: Бот должен быть администратором чата с правом удаления сообщений!",
        reply_markup=get_main_keyboard()
    )


def build_personal_style_keyboard(current: str | None, saved_count: int) -> InlineKeyboardMarkup:
    buttons = []
    for key, preset in AI_STYLE_PRESETS.items():
        suffix = " ✅" if current == key else ""
        buttons.append([
            InlineKeyboardButton(text=f"{preset['title']}{suffix}", callback_data=f"style_me_{key}")
        ])
    custom_suffix = " ✅" if current == CUSTOM_STYLE_KEY else ""
    buttons.append([
        InlineKeyboardButton(text=f"📝 Свой стиль{custom_suffix}", callback_data="style_me_custom")
    ])
    if current:
        buttons.append([
            InlineKeyboardButton(text="Сбросить на стиль бота", callback_data="style_me_reset")
        ])
    buttons.append([
        InlineKeyboardButton(
            text=f"💾 Сохранённые стили ({saved_count}/{MAX_SAVED_STYLES})",
            callback_data="style_saved_menu"
        )
    ])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def build_saved_styles_keyboard(user_id: int) -> InlineKeyboardMarkup:
    styles = get_saved_styles(user_id)
    active_id = get_active_saved_style_id(user_id)
    rows: list[list[InlineKeyboardButton]] = []
    for style in styles:
        mark = " ✅" if style["id"] == active_id else ""
        rows.append(
            [
                InlineKeyboardButton(
                    text=f"{style['name']}{mark}",
                    callback_data=f"style_saved_use_{style['id']}"
                ),
                InlineKeyboardButton(
                    text="🗑",
                    callback_data=f"style_saved_del_{style['id']}"
                )
            ]
        )
    if len(styles) < MAX_SAVED_STYLES:
        rows.append([InlineKeyboardButton(text="➕ Новый стиль", callback_data="style_saved_add")])
    rows.append([InlineKeyboardButton(text="⬅️ Назад", callback_data="style_saved_back")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_style_menu_view(user_id: int) -> tuple[str, InlineKeyboardMarkup]:
    personal = get_user_style(user_id)
    default_style = DEFAULT_AI_STYLE
    saved_styles = get_saved_styles(user_id)
    personal_prompt = get_user_custom_prompt(user_id) if personal == CUSTOM_STYLE_KEY else None
    status_text: str
    if personal == CUSTOM_STYLE_KEY:
        snippet = (personal_prompt or "не задан").strip()
        preview = (snippet[:120] + "…") if len(snippet) > 120 else snippet
        status_text = f"Твой личный стиль: 📝 Свой.\nОписание: {preview}"
    elif personal and is_saved_style_key(personal):
        saved_id = extract_saved_style_id(personal)
        saved = get_saved_style(user_id, saved_id)
        if saved:
            snippet = saved["prompt"].strip()
            preview = (snippet[:120] + "…") if len(snippet) > 120 else snippet
            status_text = f"Твой личный стиль: � {saved['name']}.\nОписание: {preview}"
        else:
            status_text = "Твой личный стиль: 💾 (не найден). Использую стиль бота."
    elif personal:
        title = AI_STYLE_PRESETS.get(personal, {"title": "Неизвестно"})["title"]
        status_text = f"Твой личный стиль: {title}"
    else:
        title = AI_STYLE_PRESETS.get(default_style, {"title": "Неизвестно"})["title"]
        status_text = f"Ты используешь стиль бота: {title}"

    text = (
        "🎭 Настройка твоего стиля\n\n"
        f"{status_text}.\nВыбери готовый вариант или нажми 'Свой стиль', чтобы описать характер."
    )
    keyboard = build_personal_style_keyboard(personal, len(saved_styles))
    return text, keyboard


async def refresh_style_menu_message(message: types.Message, user_id: int) -> None:
    text, keyboard = build_style_menu_view(user_id)
    try:
        await message.edit_text(text, reply_markup=keyboard)
    except TelegramBadRequest:
        await message.edit_reply_markup(reply_markup=keyboard)


@dp.message(F.text == "🎭 Стиль общения")
async def style_menu(message: types.Message):
    if message.chat.type != "private":
        return
    if not await ensure_channel_subscription(message):
        return

    user_id = message.from_user.id
    text, keyboard = build_style_menu_view(user_id)
    await message.answer(text, reply_markup=keyboard)

@dp.callback_query(F.data.startswith("style_"))
async def change_style(callback: types.CallbackQuery, state: FSMContext):
    data = callback.data
    user_id = callback.from_user.id

    if data == "style_me_reset":
        reset_user_style(user_id)
        await refresh_style_menu_message(callback.message, user_id)
        await callback.answer("Личный стиль сброшен")
        return

    if data.startswith("style_me_"):
        style_key = data.split("_", maxsplit=2)[2]
        if style_key == "custom":
            await callback.answer()
            await state.set_state(BotStates.waiting_custom_style)
            await callback.message.answer(
                "📝 Опиши, как я должен разговаривать лично с тобой."
                f" Минимум {CUSTOM_STYLE_MIN_LENGTH} символов, максимум {CUSTOM_STYLE_PROMPT_LIMIT}.\n"
                "Команда /cancel отменяет настройку."
            )
            return
        if style_key not in AI_STYLE_PRESETS:
            await callback.answer("Неизвестный стиль")
            return
        set_user_style(user_id, style_key)
        await refresh_style_menu_message(callback.message, user_id)
        await callback.answer("Личный стиль обновлён")
        return


@dp.message(BotStates.waiting_custom_style)
async def handle_custom_style_input(message: types.Message, state: FSMContext):
    if message.chat.type != "private":
        await message.answer("Настройку стиля можно делать только в личке.")
        return
    if message.text and message.text.strip() == "/cancel":
        await state.clear()
        await message.answer("❌ Настройка кастомного стиля отменена.")
        return

    text = (message.text or "").strip()
    is_valid, error = await validate_custom_style_prompt(text)
    if not is_valid:
        await message.answer(error)
        return

    set_user_custom_prompt(message.from_user.id, text)
    set_user_style(message.from_user.id, CUSTOM_STYLE_KEY)
    await state.clear()
    await message.answer(
        "✅ Кастомный стиль сохранён. Теперь я буду отвечать по твоим правилам.",
        reply_markup=get_main_keyboard()
    )

# ==================== Команды администратора ====================

@dp.callback_query(F.data.startswith("reply_"))
async def admin_reply_button(callback: types.CallbackQuery, state: FSMContext):
    """Обработка нажатия на кнопку 'Ответить'"""
    # Проверяем, что это администратор
    if not ADMIN_ID or str(callback.from_user.id) != str(ADMIN_ID):
        await callback.answer("У вас нет прав администратора", show_alert=True)
        return
    
    # Извлекаем ID пользователя
    user_id = int(callback.data.split("_")[1])
    
    # Сохраняем ID в состоянии
    await state.update_data(reply_to_user_id=user_id)
    await state.set_state(BotStates.waiting_admin_reply)
    
    await callback.message.answer(
        f"✏️ Напишите ваш ответ пользователю {user_id}:\n\n"
        "Отправьте /cancel для отмены."
    )
    await callback.answer()

@dp.message(BotStates.waiting_admin_reply)
async def send_admin_reply(message: types.Message, state: FSMContext):
    """Отправка ответа админа пользователю"""
    if message.text == "/cancel":
        await state.clear()
        await message.answer("❌ Отменено.")
        return
    
    # Получаем ID пользователя
    data = await state.get_data()
    user_id = data.get("reply_to_user_id")
    
    if not user_id:
        await message.answer("❌ Ошибка: ID пользователя не найден.")
        await state.clear()
        return
    
    try:
        if message.text:
            await bot.send_message(
                user_id,
                f"💬 Ответ от администратора:\n\n{message.text}"
            )
        else:
            await bot.send_message(user_id, "💬 Ответ от администратора:")
            await message.copy_to(user_id)

        await message.answer(
            f"✅ Ответ отправлен пользователю {user_id}!\n\n"
            + (f"Текст ответа:\n{message.text}" if message.text else "Медиа-файл отправлен.")
        )

    except Exception as e:
        logger.error(f"Ошибка отправки ответа: {e}")
        await message.answer(f"❌ Ошибка при отправке: {e}")
    
    await state.clear()


@dp.callback_query(F.data.startswith("support_media_"))
async def toggle_support_media(callback: types.CallbackQuery):
    if not ADMIN_ID or str(callback.from_user.id) != str(ADMIN_ID):
        await callback.answer("У вас нет прав", show_alert=True)
        return

    user_id = int(callback.data.split("_")[-1])
    new_state = db.toggle_support_media_ban(user_id)
    text = "Медиа запрещены" if new_state else "Медиа снова разрешены"
    await callback.answer(text)
    await callback.message.edit_reply_markup(reply_markup=build_support_admin_keyboard(user_id))


@dp.callback_query(F.data == "check_subscription")
async def check_subscription(callback: types.CallbackQuery):
    if not REQUIRED_CHANNEL:
        await callback.answer("Подписка не требуется", show_alert=True)
        return
    if await is_user_subscribed(callback.from_user.id):
        await callback.answer("Подписка подтверждена!", show_alert=True)
        await callback.message.answer(WELCOME_TEXT, reply_markup=get_main_keyboard())
    else:
        await callback.answer("Подписка не найдена. Проверьте, что подписаны на канал.", show_alert=True)


@dp.callback_query(F.data.startswith("support_full_"))
async def toggle_support_full(callback: types.CallbackQuery):
    if not ADMIN_ID or str(callback.from_user.id) != str(ADMIN_ID):
        await callback.answer("У вас нет прав", show_alert=True)
        return

    user_id = int(callback.data.split("_")[-1])
    new_state = db.toggle_support_full_ban(user_id)
    text = "Пользователь заблокирован в поддержке" if new_state else "Пользователь снова может писать"
    await callback.answer(text)
    await callback.message.edit_reply_markup(reply_markup=build_support_admin_keyboard(user_id))

# ==================== Запуск бота ====================
async def init_bot_identity():
    global BOT_ID, BOT_USERNAME
    if BOT_ID and BOT_USERNAME:
        return
    me = await bot.get_me()
    BOT_ID = me.id
    BOT_USERNAME = me.username


async def main():
    logger.info("Запуск JoyGuard...")
    await init_bot_identity()
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())