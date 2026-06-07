# Code Agent

从零到一实现一个 Claude Code 风格的代码智能体。

## 项目目标

通过学习 Claude Code 的设计理念与实现方式，从零构建一个具备代码理解、编辑、执行等能力的 AI 代码助手。

## 已实现功能

- **Agent Loop**：多轮工具调用循环，自动执行直到任务完成
- **工具系统**：
  - `bash` - 执行 shell 命令（含危险命令拦截）
  - `read_file` - 读取文件内容（支持行数限制）
  - `write_file` - 写入文件（自动创建目录）
  - `edit_file` - 精确文本替换编辑
  - `todo` - 任务计划管理（多步骤任务规划与跟踪）
- **消息规范化**：自动处理 Anthropic 对象序列化、孤立 tool_use 补全、同角色消息合并
- **安全机制**：工作目录路径限制，防止路径逃逸
- **Windows 兼容**：UTF-8 编码处理，解决 GBK 编码问题
- **输出格式化**：工具调用图标 + 参数显示 + 分隔线 + 截断提示

## 技术栈

- Python 3.10+
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-python-sdk)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

## 快速开始

```bash
# 克隆项目
git clone https://github.com/mozarta-nexus/code-agent.git
cd code-agent

# 安装依赖
pip install anthropic python-dotenv

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 运行
python Agent.py
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `ANTHROPIC_BASE_URL` | API 基础地址 | - |
| `ANTHROPIC_API_KEY` | API 密钥 | - |
| `ANTHROPIC_MODEL` | 模型名称 | `claude-sonnet-4-20250514` |
| `ANTHROPIC_MAX_TOKENS` | 最大 token 数 | `1024` |

## 项目结构

```
code-agent/
├── Agent.py           # 主程序（Agent Loop + 工具系统）
├── .env.example       # 环境变量样例
├── .gitignore
├── LICENSE
└── README.md
```

## 版本历史

- **v0.3.0** - 新增 todo 工具与任务计划管理
- **v0.2.0** - 多工具支持（read/write/edit）+ Windows UTF-8 编码
- **v0.1.0** - Agent Loop + bash 工具

## 许可证

MIT
