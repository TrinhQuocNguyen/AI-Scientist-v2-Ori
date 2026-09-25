"""Claude backend that runs through the Claude Agent SDK (Claude Code).

Uses the Claude subscription (Pro / Max / Team / Enterprise) you are logged in
with via `claude login`, so no API key is needed. Select it with the
`claude-code/` model prefix, e.g. `claude-code/opus`, `claude-code/sonnet`,
`claude-code/claude-opus-5-5`.

Each call is a single, tool-less turn: Claude only returns text (or JSON that
matches a schema), exactly like a plain chat-completion call.
"""

import logging
import os
import re
import time
from typing import Any

import anyio

from ai_scientist.utils.token_tracker import token_tracker

logger = logging.getLogger("ai-scientist")

PREFIX = "claude-code/"
MAX_RETRIES = int(os.environ.get("CLAUDE_CODE_MAX_RETRIES", 8))
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI research assistant."

_DATA_URL_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)
_warned_api_key = False


def is_claude_code_model(model: str | None) -> bool:
    return bool(model) and model.startswith(PREFIX)


def _to_blocks(content: Any) -> list[dict]:
    """Convert OpenAI- or Anthropic-style message content into Anthropic blocks."""
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, dict):
        content = [content]
    blocks = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif item.get("type") == "text":
            blocks.append({"type": "text", "text": item["text"]})
        elif item.get("type") == "image_url":
            url = item["image_url"]
            url = url["url"] if isinstance(url, dict) else url
            match = _DATA_URL_RE.match(url)
            if match:
                source = {"type": "base64", "media_type": match[1], "data": match[2]}
            else:
                source = {"type": "url", "url": url}
            blocks.append({"type": "image", "source": source})
        elif item.get("type") == "image":
            blocks.append(item)
        else:
            blocks.append({"type": "text", "text": str(item)})
    return blocks


def _merge_text(blocks: list[dict]) -> list[dict]:
    merged = []
    for block in blocks:
        if block["type"] == "text" and merged and merged[-1]["type"] == "text":
            merged[-1] = {"type": "text", "text": merged[-1]["text"] + "\n" + block["text"]}
        else:
            merged.append(block)
    return merged


def _flatten(messages: list[dict]) -> tuple[list[dict], str]:
    """Fold a chat history into one user turn, since each call is a fresh session.

    Returns the content blocks and any system-role text found in the history.
    """
    system_parts = [
        m["content"] for m in messages if m["role"] == "system" and isinstance(m["content"], str)
    ]
    turns = [m for m in messages if m["role"] != "system"]
    *history, last = turns
    blocks = []
    if history:
        blocks.append(
            {
                "type": "text",
                "text": "Here is the conversation so far. Reply to the final user message, "
                "continuing the conversation as the assistant.",
            }
        )
        for m in history:
            blocks.append({"type": "text", "text": f"<{m['role']}>"})
            blocks.extend(_to_blocks(m["content"]))
            blocks.append({"type": "text", "text": f"</{m['role']}>"})
        blocks.append({"type": "text", "text": "<user>"})
        blocks.extend(_to_blocks(last["content"]))
        blocks.append({"type": "text", "text": "</user>"})
    else:
        blocks.extend(_to_blocks(last["content"]))
    return _merge_text(blocks), "\n\n".join(system_parts)


async def _run(blocks, model_id, system_prompt, output_schema, state):
    from claude_agent_sdk import ClaudeAgentOptions, RateLimitEvent, ResultMessage
    from claude_agent_sdk import query as sdk_query

    options = ClaudeAgentOptions(
        model=model_id,
        system_prompt=system_prompt,
        tools=[],
        max_turns=3 if output_schema is not None else 1,
        setting_sources=[],
        output_format=(
            {"type": "json_schema", "schema": output_schema}
            if output_schema is not None
            else None
        ),
        extra_args={"no-session-persistence": None},
    )

    async def prompt_stream():
        yield {
            "type": "user",
            "message": {"role": "user", "content": blocks},
            "parent_tool_use_id": None,
            "session_id": "default",
        }

    result = None
    async for message in sdk_query(prompt=prompt_stream(), options=options):
        if isinstance(message, RateLimitEvent):
            if message.rate_limit_info.status == "rejected":
                state["resets_at"] = message.rate_limit_info.resets_at
        elif isinstance(message, ResultMessage):
            result = message
    return result


def _retry_wait(attempt: int, resets_at: int | None) -> float:
    if resets_at:
        # Subscription usage limit hit: wait for the window to reset.
        return max(resets_at - time.time(), 0) + 60
    return min(5 * 2**attempt, 300)


def query(
    messages: list[dict],
    model: str,
    system_message: str | None = None,
    output_schema: dict | None = None,
) -> tuple[str, Any, dict]:
    """Send a chat history to Claude and return (text, structured_output, usage).

    `structured_output` is the parsed JSON when `output_schema` is given, else None.
    """
    from claude_agent_sdk import CLINotFoundError

    global _warned_api_key
    if os.environ.get("ANTHROPIC_API_KEY") and not _warned_api_key:
        print(
            "Warning: ANTHROPIC_API_KEY is set, so Claude Code will bill that API key "
            "instead of your Claude subscription. Unset it to use the subscription."
        )
        _warned_api_key = True

    blocks, history_system = _flatten(messages)
    system_prompt = (
        "\n\n".join(p for p in [system_message, history_system] if p)
        or DEFAULT_SYSTEM_PROMPT
    )
    model_id = model[len(PREFIX):]

    for attempt in range(MAX_RETRIES):
        state = {}
        try:
            result = anyio.run(_run, blocks, model_id, system_prompt, output_schema, state)
            if result is None:
                raise RuntimeError("Claude Code returned no result")
            if result.is_error:
                raise RuntimeError(f"Claude Code error: {result.errors or result.result}")
            if output_schema is not None and result.structured_output is None:
                raise RuntimeError("Claude Code returned no structured output")
            break
        except CLINotFoundError:
            raise
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = _retry_wait(attempt, state.get("resets_at"))
            print(
                f"Claude Code call failed ({e}); retrying in {wait:.0f}s "
                f"(attempt {attempt + 1}/{MAX_RETRIES})"
            )
            time.sleep(wait)

    usage = result.usage or {}
    cached = usage.get("cache_read_input_tokens", 0)
    prompt_tokens = (
        usage.get("input_tokens", 0) + cached + usage.get("cache_creation_input_tokens", 0)
    )
    token_tracker.add_tokens(model, prompt_tokens, usage.get("output_tokens", 0), 0, cached)
    token_tracker.add_interaction(
        model,
        system_prompt,
        "\n".join(b["text"] for b in blocks if b["type"] == "text"),
        result.result,
        int(time.time()),
    )
    return result.result or "", result.structured_output, usage
