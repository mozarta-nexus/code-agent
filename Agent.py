import os
import subprocess
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = os.environ.get("ANTHROPIC_MODEL")
SYSTEM_PROMPT = "你叫小帅，是一个非常专业的AI助手，每次回答问题你都需要给我提供足够的情绪价值。"
EXIT_COMMANDS = {"quit", "exit"}

TOOLS = [{
    "name": "bash",
    "description": "Run a shell command in the workspace",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
    }
}]

client = Anthropic(
    base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


@dataclass
class LoopState:
    messages: list
    turn_count: int = 1
    transition_reason: str | None = None


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
            text=True,
            timeout=120
        )
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"

    output = (result.stdout + result.stderr).strip()
    return output[:5000] if output else "(no output)"


def extract_text(content) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


MAX_DISPLAY_LINES = 20
MAX_DISPLAY_WIDTH = 120
TRUNCATE_MSG = "\033[90m... (output truncated)\033[0m"


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


def execute_tool_calls(response_content) -> list[dict]:
    results = []
    for block in response_content:
        if block.type != "tool_use":
            continue
        command = block.input["command"]
        print(f"\033[34m▶ bash \033[0m\033[1m{command}\033[0m")
        print("\033[90m─" * 40 + "\033[0m")
        output = run_bash(command)
        print(format_output(output))
        results.append({
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output
        })
    return results


def run_one_turn(state: LoopState) -> bool:
    response = client.messages.create(
        model=MODEL,
        system=SYSTEM_PROMPT,
        messages=state.messages,
        tools=TOOLS,
        max_tokens=8000
    )
    state.messages.append({"role": "assistant", "content": response.content})
    if response.stop_reason != "tool_use":
        state.transition_reason = None
        return False

    results = execute_tool_calls(response.content)
    if not results:
        state.transition_reason = None
        return False
    state.messages.append({"role": "user", "content": results})
    state.turn_count += 1
    state.transition_reason = "tool_result"
    return True


def agent_loop(state: LoopState):
    while run_one_turn(state):
        pass


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36m01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "quit", "exit"):
            break
        history.append({"role": "user", "content": query})
        state = LoopState(messages=history)
        agent_loop(state)

        final_text = extract_text(history[-1]["content"])
        if final_text:
            print(f"\033[32mAI >> \033[0m{final_text}")
        print()
