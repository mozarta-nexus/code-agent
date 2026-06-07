import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

WORKDIR = Path.cwd()
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
SYSTEM_PROMPT = f"""
- 你叫小帅，是一个非常专业的AI助手，每次回答问题你都需要给我提供足够的情绪价值。
- 你是一个在 {WORKDIR} 工作目录中运行的代码智能体。
- 对于多步骤任务，必须使用 todo 工具制定和维护执行计划。
- 当计划包含多个步骤时，任何时刻只能有一个步骤标记为 in_progress。
- 在任务执行过程中，根据实际进展持续更新计划状态。
- 优先通过工具完成分析、修改、执行和验证工作，而不是仅用文字说明。
- 保持计划与实际执行过程同步。
"""

PLAN_REMINDER_INTERVAL = 3


@dataclass
class PlanItem:
    content: str
    status: str = "pending"
    active_form: str = ""


@dataclass
class PlanningState:
    items: list[PlanItem] = field(default_factory=list)
    rounds_since_update: int = 0


class TodoManager:
    def __init__(self):
        self.state = PlanningState()

    def update(self, items: list) -> str:
        if len(items) > 12:
            raise ValueError("Keep the session plan short (max 12 items)")

        normalized = []
        in_progress_count = 0;
        for index, raw_item in enumerate(items):
            content = str(raw_item.get("content", "")).strip()
            status = str(raw_item.get("status", "pending")).lower()
            active_form = str(raw_item.get("active_form", "")).strip()

            if not content:
                raise ValueError(f"Item {index}: content required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {index}: invalid status {status}")
            if status == "in_progress":
                in_progress_count += 1

            normalized.append(PlanItem(
                content=content,
                status=status,
                active_form=active_form
            ))

        if in_progress_count > 1:
            raise ValueError("Only one plan item can be in_progress")

        self.state.items = normalized
        self.state.rounds_since_update = 0;
        return self.render()

    def note_round_without_update(self) -> None:
        self.state.rounds_since_update += 1

    def reminder(self) -> str | None:
        if not self.state.items:
            return None
        if self.state.rounds_since_update < PLAN_REMINDER_INTERVAL:
            return None
        return "<reminder>Refresh your current plan before continuing.</reminder>"

    def render(self) -> str:
        if not self.state.items:
            return "No session plan yet."
        lines = []
        for item in self.state.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item.status]
            line = f"{marker} {item.content}"
            if item.status == "in_progress" and item.active_form:
                line += f" ({item.active_form})"
            lines.append(line)
        completed = sum(1 for item in self.state.items if item.status == "completed")
        lines.append(f"\n({completed}/{len(self.state.items)} completed)")
        return "\n".join(lines)


TODO = TodoManager()

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
    "todo": lambda **kw: TODO.update(kw["items"]),
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
    {
        "name": "todo",
        "description": "Rewrite the current session plan for multi-step work.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {
                                "type": "string",
                                "description": "Optional present-continuous label.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        },
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
        used_todo = False
        for block in response.content:
            if block.type != "tool_use":
                continue
            handler = TOOL_HANDLERS.get(block.name)
            try:
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
            except Exception as e:
                output = f"Error: {e}"
            print_tool_call(block.name, block.input, output)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})

            if block.name == "todo":
                used_todo = True
        if used_todo:
            TODO.state.rounds_since_update = 0
        else:
            TODO.note_round_without_update()
            reminder = TODO.reminder()
            if reminder:
                results.insert(0, {"type": "text", "text": reminder})

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
