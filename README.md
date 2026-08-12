# Company Research Agent（公司研究智能体）

一个以“证据优先”为核心的公司研究服务，目前重点面向医药与生命科学企业。

它会依次完成公司身份确认、研究规划、Exa 搜索、逐条事实提取、引用校验、信息缺口评估和结构化报告生成。系统不会直接相信大模型输出：每条事实必须能够追溯到已登记的来源和原文引用；无效 Claim 会被单独拒绝，不会拖垮同批次中的其他有效事实。

## 主要能力

- 使用 FastAPI 提供独立的研究 API。
- 支持 OpenAI-compatible 与 Anthropic-compatible LLM 接口。
- 通过 Exa MCP 搜索公司公开资料。
- 对公司名称、别名、法律后缀和同名实体进行保守解析。
- 集中保存 Source 元数据，Evidence 只通过 `source_ids` 引用来源。
- 支持 SEC、公司官网、监管机构等权威来源的表格型引用。
- 将未知信息表示为独立的 Research Gap，而不是伪造 Evidence。
- 单条 Claim 校验失败不会清空整个证据批次。
- 返回完整状态转换、预算使用、停止原因和处理错误。
- 内置查询、Token、成本、轮次、并发和运行时长限制。

## 工作流程

```text
ResolveCompany
  -> PlanResearch
  -> GenerateSearchQueries
  -> ParallelSearch
  -> ProcessEvidence
  -> Summarize
  -> AssessInformation
  -> RefineCompanyIdentity
  -> GenerateFollowupQueries（需要时）
  -> Finalize
  -> ReturnResult
```

每次运行都有独立的预算、Source Registry、Evidence Processor 和 Run Trace，不会在不同请求之间共享研究证据。

## 环境要求

- Windows、macOS 或 Linux
- Python 3.11 或更高版本
- DeepSeek、OpenAI 或其他兼容 LLM 的 API Key
- Exa API Key 及可用的 Exa MCP 地址
- 可选：Docker 与 Docker Compose

## Windows PowerShell 快速开始

先进入项目目录。不要在 `C:\Users\你的用户名` 下直接执行安装命令。

```powershell
cd C:\你的路径\CompanyResearchAgent
```

确认 Python 版本：

```powershell
python --version
```

创建并激活虚拟环境：

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

安装项目和开发依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，填入自己的密钥和提供商地址。`.env` 已被 Git 忽略，禁止把真实 API Key 提交到仓库。

## 核心配置

最少需要配置：

```dotenv
CRA_ENVIRONMENT=development

# 服务自身的访问密码，由你自行生成，不是 DeepSeek 或 Exa 的 Key
CRA_SERVICE_API_KEY=replace-with-a-long-random-value

# LLM
CRA_LLM_PROVIDER=openai
CRA_LLM_API_KEY=replace-with-your-llm-key
CRA_LLM_BASE_URL=https://your-openai-compatible-endpoint.example/v1
CRA_LLM_MODEL=replace-with-your-model-name

# Exa MCP
CRA_EXA_MCP_URL=https://your-exa-mcp-endpoint.example/mcp?tools=web_search_advanced_exa
CRA_EXA_API_KEY=replace-with-your-exa-key
```

如果使用 DeepSeek 的 OpenAI-compatible API，通常保持：

```dotenv
CRA_LLM_PROVIDER=openai
CRA_LLM_MODEL=deepseek-chat
```

`CRA_LLM_BASE_URL` 请填写 DeepSeek 当前文档给出的兼容接口地址。不要把 API Key 放入 URL 查询参数。

常用运行限制：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `CRA_MAX_SEARCH_ROUNDS` | `3` | 最大研究轮数 |
| `CRA_MAX_TOTAL_QUERIES` | `3` | 单次运行查询预算 |
| `CRA_MAX_QUERIES_PER_ROUND` | `3` | 每轮最大查询数 |
| `CRA_TOKEN_BUDGET` | `100000` | 单次运行 Token 上限 |
| `CRA_COST_BUDGET_USD` | `25` | 估算 LLM 成本上限 |
| `CRA_EVIDENCE_LIMIT` | `500` | 最大 Evidence 数量 |
| `CRA_RUN_TIMEOUT_SECONDS` | `900` | 端到端超时时间 |
| `CRA_SEARCH_MAX_CONCURRENCY` | `3` | Exa 并发搜索数 |
| `CRA_EXTRACTION_BATCH_SIZE` | `1` | 每次 Evidence 提取的来源数；提高前需验证引用准确率 |
| `CRA_CACHE_ENABLED` | `true` | 是否启用本地缓存 |

来源可信度由应用控制。可在 `.env` 中配置可信域名：

```dotenv
CRA_SOURCE_TYPE_DOMAIN_RULES={"pfizer.com":"official","fda.gov":"regulatory","sec.gov":"regulatory","reuters.com":"news"}
```

不要把 `com`、`org` 或 `co.uk` 这种公共后缀配置为可信来源。

## 启动服务

```powershell
python -m uvicorn app.main:app --reload
```

启动后可访问：

- Swagger API 文档：<http://127.0.0.1:8000/docs>
- 存活检查：<http://127.0.0.1:8000/health/live>
- 配置就绪检查：<http://127.0.0.1:8000/health/ready>

直接访问 `http://127.0.0.1:8000/` 返回 `404 Not Found` 是正常的，因为项目没有定义根路径。

## 调用研究 API

PowerShell 示例：

```powershell
$headers = @{
    "X-API-Key" = "你在 CRA_SERVICE_API_KEY 中设置的值"
}

$body = @{
    canonical_name = "Pfizer Inc."
    country = "United States"
    industry = "Pharmaceuticals"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/v1/research" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body
```

长时间运行或需要显示进度时，推荐使用异步任务接口：

```powershell
$job = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/v1/research-jobs" `
    -Method Post `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body

# 查询任务状态和最终结果
Invoke-RestMethod -Uri "http://127.0.0.1:8000$($job.status_url)" -Headers $headers

# SSE 进度流；curl.exe 支持持续显示事件
curl.exe -N `
    -H "X-API-Key: 你在 CRA_SERVICE_API_KEY 中设置的值" `
    "http://127.0.0.1:8000$($job.events_url)"
```

SSE 会发送 `queued`、`running`、`company_resolved`、`search_progress`、
`evidence_batch_ready`、`completed` 或 `failed`。断线重连时可传 `Last-Event-ID`。
当前任务队列保存在服务进程内，进程重启后任务记录不会保留；合入生产宿主项目时可在不改变
HTTP 契约的情况下替换为其持久化队列。

只有 `canonical_name` 是必填字段。`website`、`linkedin`、`country`、`industry` 和 `summary` 都只是待验证提示，不能直接成为研究事实。

响应的主要结构：

```json
{
  "execution_status": "completed_with_gaps",
  "company": {
    "canonical_name": "Pfizer Inc.",
    "identity_status": "confirmed",
    "aliases": ["Pfizer"],
    "claim_ids": ["clm_..."]
  },
  "research": {
    "overview": {"findings": []},
    "unknowns": {"gaps": []}
  },
  "evidence": {
    "sources": [],
    "evidence": [],
    "conflicts": [],
    "rejected_claims": [],
    "processing_errors": []
  },
  "run_trace": {
    "logical_query_count": 3,
    "exa_tool_call_count": 3,
    "llm_provider_call_count": 8,
    "llm_calls_by_stage": {},
    "rounds": [],
    "transitions": [],
    "stop_reason": "sufficient_information"
  }
}
```

## Evidence 语义

- `verified_fact`：由官方、监管或明确匹配目标公司的权威来源直接支持，或者得到多个可靠独立来源支持。
- `single_source`：只有一个可靠但非权威来源支持。
- `inference`：保留给可复现的确定性推导；自由文本模型推理不会直接升级为 Evidence。
- `unknown`：仅用于来源本身明确说明某事实尚未确定的少数场景。
- 普通的信息缺失使用 `research.unknowns.gaps` 表示。

每条 Evidence：

- 必须包含可解析的 `source_ids`；
- 必须包含能够在来源正文中定位的 `supporting_quotes`；
- 不允许把 URL、引用编号或 Source ID 写进 Claim 文本；
- 不重复保存 URL、标题、发布日期和来源类型；
- 可以由多个独立 Source 共同支持。

LLM 返回的一条 Claim 含 URL 或格式错误时，系统只拒绝该条记录，并在 `rejected_claims` 和 `processing_errors` 中保留具体原因。

## 执行状态和停止原因

顶层 `execution_status` 可能为：

- `completed`
- `completed_with_gaps`
- `partial_failure`
- `failed`

常见停止原因包括：

- `sufficient_information`
- `max_rounds`
- `query_budget_reached`
- `token_budget_reached`
- `cost_budget_reached`
- `time_budget_reached`
- `no_new_evidence`
- `search_unavailable`
- `extraction_failure`

`partial_failure` 不代表整个响应无效。它通常表示部分来源、Claim 或提供商调用失败，但仍然保留了可验证结果。

## 测试

测试不需要真实 API Key，网络依赖均使用 Fake 或 Mock。

```powershell
python -m pytest -q
python -m ruff format --check app tests
python -m ruff check app tests
python -m mypy app
python -m compileall -q app tests
```

当前自动化测试覆盖：

- API 认证和完整响应 Schema；
- 状态机合法转换；
- Exa MCP 解析和错误处理；
- Source URL/内容去重；
- Claim 逐条校验和部分失败；
- SEC 表格、总部、注册地和 ticker；
- Pfizer Pipeline 引用；
- 公司法律名称和简称归一化；
- Evidence 双向引用和计数一致性；
- Research Gap、预算和停止原因；
- 多轮 Orchestrator 集成流程。

## 项目结构

```text
app/api/          FastAPI 路由、依赖和错误响应
app/core/         状态机、预算、停止策略、协议和枚举
app/resolver/     公司身份解析与名称归一化
app/planner/      研究主题和搜索查询规划
app/search/       Exa MCP 适配器与并发执行器
app/evidence/     Source Registry、Claim 校验和证据策略
app/reflection/   覆盖度评估和后续查询建议
app/llm/          OpenAI/Anthropic-compatible 客户端
app/prompts/      可版本化 Prompt 模板
app/cache/        内存/SQLite TTL 缓存
app/services/     依赖组装、编排和最终报告
app/schemas/      Pydantic v2 API 与领域模型
tests/            单元测试、集成测试和回归 Fixture
artifacts/        已验证的示例研究输出
```

Pfizer 修复后的示例输出位于：

```text
artifacts/pfizer_response_post_fix_final.json
```

该样例包含 8 个 Source、32 个 supported claims、31 个 verified facts，并能通过项目自身的 `ResearchResponse.model_validate()`。

## Docker

```powershell
docker compose up --build
```

容器使用非 root 用户运行，并将 SQLite 缓存保存在配置的数据卷中。生产环境应在可信入口终止 TLS，并使用 Secret Manager 管理 API Key。

## 常见问题

### 浏览器显示 `{"detail":"Not Found"}`

这是访问了未定义的 `/`。请打开 `/docs` 或 `/health/live`。

### 返回 `authentication_required`

请求头中的 `X-API-Key` 必须与 `.env` 中的 `CRA_SERVICE_API_KEY` 完全一致。

### 返回 `research_provider_error`

检查：

1. `CRA_LLM_PROVIDER` 是否与接口格式一致；
2. `CRA_LLM_MODEL` 是否是提供商支持的模型名称；
3. `CRA_LLM_BASE_URL` 是否正确；
4. DeepSeek/LLM 和 Exa Key 是否有效且有可用额度；
5. Exa MCP URL 是否能暴露 `web_search_advanced_exa` 工具。

### 为什么结果是 `partial_failure`

可能原因包括查询预算已用完、部分 Claim 未通过引用校验、LLM 输出不符合 Schema、搜索提供商临时失败，或部分研究维度仍存在缺口。请结合 `run_trace.stop_reason`、`rejected_claims` 和 `processing_errors` 判断。

## 已知限制

- 搜索质量和实时性取决于 Exa 返回的网页正文。
- DeepSeek 等模型的结构化输出存在波动，同一公司重复运行可能得到不同数量的有效 Claim。
- 引用校验有意偏保守，复杂句、截断内容或不明确的主语可能被拒绝。
- Exa 成本尚未计入项目的 LLM 美元预算。
- SQLite 缓存适合单机部署；多副本部署需要共享缓存和统一限流。
- 当前共享 API Key 是服务级认证，不提供租户隔离和租户级配额。

## 安全说明

- 永远不要提交 `.env`、真实 API Key、Cookie 或访问令牌。
- 生产环境只使用 HTTPS 提供商地址。
- 不要在 URL 查询参数中放置密钥。
- 设置合理的查询、Token、成本和并发限制。
- 对外部署时使用反向代理、TLS、访问日志和速率限制。
