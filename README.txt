# coding-agent

项目地址：https://github.com/zzzrw/my-coding-agent

## 安装

需要 Python 3.11+。推荐使用 uv：

```bash
uv sync
```

也可以安装本项目及开发依赖：

```bash
python -m pip install -e .
```

## 配置与运行

先设置兼容 OpenAI API 的模型和密钥。密钥只从环境变量读取，不要写入仓库：

```bash
export CODING_AGENT_MODEL="你的模型名"
export CODING_AGENT_API_KEY="你的密钥"
# 可选：export CODING_AGENT_BASE_URL="https://兼容接口/v1"
python -m coding_agent.app --workspace .
```

也支持 `OPENAI_API_KEY`、`DEEPSEEK_API_KEY` 及对应的模型和 Base URL 环境变量。运行参数：`--workspace`、`--model`、`--base-url`、`--session-dir`、`--context-window`。可用 `python -m coding_agent.app --help` 查看帮助，此操作不需要密钥。

## MVP 功能

提供单一会话 transcript、提示输入框和底部状态栏；支持新建、恢复、压缩会话及权限模式切换。内置读文件、列文件、搜索、写文件、编辑文件和运行命令六个工具，默认权限策略会对修改与命令操作请求确认，并将会话持久化到本地。

核心 Agent Runtime、工具执行器、权限策略、上下文裁剪、会话存储和 TUI 均为本项目自行实现，未用外部 Agent 框架替代核心逻辑。
