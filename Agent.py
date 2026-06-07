import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

WORKDIR = Path.cwd()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
SYSTEM_PROMPT = "你叫小帅，是一个非常专业的AI助手，每次回答问题你都需要给我提供足够的情绪价值。"

MAX_DISPLAY_LINES = 20
MAX_DISPLAY_WIDTH = 120
TRUNCATE_MSG = "\033[90m... (output truncated)\033[0m"

TOOL_ICONS = {
    "bash": ">",
    "read_file": "R",
    "write_file": "W",
    "edit_file": "E",
}


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(item in command for item in dangerous):
        return "Error: Dangerous command blocked"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            timeout=120
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
    stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
    output = (stdout + stderr).strip()
    return output[:5000] if output else "(no output)"


def run_read(path: str, limit: int | None = None) -> str:
    try:
        text = safe_path(path).read_text(encoding="utf-8")
        lines = text.splitlines()
        if limit is not None and limit < len(lines):
            lines = lines[:limit] + [f"...({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text(encoding="utf-8")
        if old_text not in content:
            return f"Error: Text not found in {path}"

        fp.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute."
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path from workspace root."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read. Omit to read all lines."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating it if it does not exist.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path from workspace root."
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file."
                }
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace an exact text match in a file with new text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path from workspace root."
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace."
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace the old_text with."
                }
            },
            "required": ["path", "old_text", "new_text"]
        }
    },
]


def normalize_messages(messages: list) -> list:
    cleaned = []
    for msg in messages:
        clean = {"role": msg["role"]}
        if isinstance(msg.get("content"), str):
            clean["content"] = msg["content"]
        elif isinstance(msg.get("content"), list):
            normalized_blocks = []
            for block in msg["content"]:
                if hasattr(block, "model_dump"):
                    normalized_blocks.append(block.model_dump())
                elif isinstance(block, dict):
                    normalized_blocks.append({k: v for k, v in block.items() if not k.startswith("_")})
                else:
                    normalized_blocks.append({"type": "text", "text": str(block)})
            clean["content"] = normalized_blocks
        else:
            clean["content"] = msg.get("content", "")
        cleaned.append(clean)

    existing_results = set()
    for msg in cleaned:
        if isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    existing_results.add(block.get("tool_use_id"))

    for msg in cleaned:
        if msg["role"] != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id") not in existing_results:
                cleaned.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block["id"], "content": "(cancelled)"}
                ]})

    if not cleaned:
        return cleaned
    merged = [cleaned[0]]
    for msg in cleaned[1:]:
        if msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev_c = prev["content"] if isinstance(prev["content"], list) \
                else [{"type": "text", "text": str(prev["content"])}]
            curr_c = msg["content"] if isinstance(msg["content"], list) \
                else [{"type": "text", "text": str(msg["content"])}]
            prev["content"] = prev_c + curr_c
        else:
            merged.append(msg)
    return merged



def format_output(output: str) -> str:
    lines = output.splitlines()
    truncated = False
    if len(lines) > MAX_DISPLAY_LINES:
        lines = lines[:MAX_DISPLAY_LINES]
        truncated = True
    formatted_lines = []
    for line in lines:
        if len(line) > MAX_DISPLAY_WIDTH:
            formatted_lines.append(line[:MAX_DISPLAY_WIDTH] + "...")
            truncated = True
        else:
            formatted_lines.append(line)
    result = "\n".join(formatted_lines)
    if truncated:
        result += "\n" + TRUNCATE_MSG
    return result


def print_tool_call(name: str, params: dict, output: str) -> None:
    icon = TOOL_ICONS.get(name, "●")
    params_str = "  ".join(f"{k}={v}" for k, v in params.items())
    print(f"\033[34m{icon} {name}\033[0m \033[90m{params_str}\033[0m")
    print("\033[90m" + "─" * 40 + "\033[0m")
    print(format_output(output))
    print("\033[90m" + "─" * 40 + "\033[0m")


def agent_loop(messages: list) -> None:
    client = Anthropic(
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )

    while True:
        response = client.messages.create(
            model=MODEL,
            system=SYSTEM_PROMPT,
            messages=normalize_messages(messages),
            tools=TOOLS,
            max_tokens=8000
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            print_tool_call(block.name, block.input, output)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36muser >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "quit", "exit"):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if hasattr(block, "text"):
                    print(f"\033[32mAI >> \033[0m{block.text}")

        print()
