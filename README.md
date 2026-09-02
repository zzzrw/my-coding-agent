# my-coding-agent

一个极简但完整的编程智能体（Coding Agent）：通过与大型语言模型交互，自主读写文件、执行命令，完成编程任务。

仓库地址：https://github.com/zzzrw/my-coding-agent

## 快速开始

依赖：Python 3.11+，推荐使用 uv。

```bash
# 安装依赖
uv sync

# 环境变量配置（亦兼容 OPENAI_*、DEEPSEEK_*）
export CODING_AGENT_MODEL="deepseek-chat"
export CODING_AGENT_API_KEY="sk-..."
export CODING_AGENT_BASE_URL="https://api.deepseek.com/v1"

# 启动
uv run coding-agent --workspace .
```

配置同样可写入本地配置文件 `~/.config/coding-agent/config.toml`（或工作区 `.coding-agent.toml`，可设置模型、密钥、语言偏好等）；二者都未配置时，首次启动会进入引导界面填写模型与密钥。密钥仅经环境变量或未入库的本地配置提供，绝不写入仓库。

启动方式：直接运行上述 `uv run coding-agent --workspace .`；`--help` 可查看全部参数且无需密钥。

## 实现与核心（均自行实现）

- 对话历史与上下文管理：JSONL 会话持久化、会话恢复（resume）、超窗摘要压缩。
- 工具定义与本地执行：运行命令、写/改/读/列/搜索/删除文件等，含参数校验、输出截断与失败重试。
- 模型输出解析：OpenAI 兼容流式响应与工具调用（tool call）的增量、容错解析。
- 循环终止：默认不设步数上限；内置"空转"检测与模型无响应看门狗；Ctrl+C 可中止运行，并保留中止前已完成工具操作的上下文。
- 错误处理：工具失败结果回传模型、有界重试、运行时错误事件化。

以上核心逻辑均为自行编写：未使用任何 Agent 框架 / SDK，也未调用服务端托管的代码执行（如 Code Interpreter、Files API）。

## 权限与安全

- 三级权限 default / workspace / full，可随时用 `/permission` 切换，默认 workspace。
- 写/改文件与命令执行前提供审批与 diff 预览；审批决策可记忆（一次/本轮/会话/始终），并以本地私有文件保存、跨会话生效；拒绝时可附原因回传模型。
- 高危命令硬拒绝；删除操作限定于工作区内；`/undo` 撤销最近一次写入。

## 架构与交互

- 架构上 Textual 终端界面与 Agent Runtime 解耦：界面只消费运行时事件流，不直接接触模型、工具或会话存储。
- 工具可并行调度，运行输出实时流式展示；等待模型回复时有明确指示。
- 历史回溯：空闲时双击 Esc，可从任意历史用户消息处 fork 出一个新会话继续工作，原会话不受影响。
- Skills：支持以 SKILL.md 组织的技能目录（工作区与用户两级），通过 `load_skill` 按需加载，并可执行技能自带的辅助脚本。

## 其它说明

- 提交材料（本说明与演示视频）已自查不含姓名、院校等可识别个人信息。
- 已按考核要求完成并自行实现：对话历史与上下文管理、工具的定义与本地执行、模型输出的解析、循环终止条件、错误处理。
