# 车辆故障智能诊断系统

> 输入一句自然语言描述的故障现象，系统同时检索 **OBD 故障码知识库（Milvus）**、**同车型投诉案例库（MySQL）** 和 **公开网络信息（Tavily）**，
> 由大模型融合成一份「可能原因 + 故障码 + 检查步骤 + 是否通病 + 证据来源」的可溯源诊断报告，全过程 SSE 流式推送。

技术栈：FastAPI · SSE · LangGraph · BGE-M3 · Milvus · MySQL · MongoDB · Tavily · 原生 HTML/JS

---

## 一、快速开始

### 1. 装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配环境变量

```bash
copy .env.example .env
```
然后填 `LLM_APIKEY` / `LLM_URL` / `LLM_MODEL`、`MYSQL_URL`、`TAVILY_APIKEY` 等。

### 3. 起服务

```bash
python -m app.main
# 或者：uvicorn app.main:app --reload --port 8000
```

打开 <http://127.0.0.1:8000> 就是前端页面，<http://127.0.0.1:8000/docs> 是自动生成的接口文档。

### 4. 没配好数据库也能先看效果

页面左下角勾上 **「演示模式（不连库）」**，或者请求时带 `"demo": true`：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"怠速抖动加速无力\",\"demo\":true}"
```

演示模式用内置样例数据走完整条流水线（进度条、流式输出、报告落盘都真实发生），不依赖任何外部服务。

---

## 二、目录结构

```
car_fault_agent/
├── app/                       ← FastAPI 服务层（本次新增）
│   ├── main.py                入口：lifespan / CORS / 静态页 / 兜底异常
│   ├── pipeline.py            诊断流水线：分阶段上报 + 流式 token
│   ├── sse.py                 SSE 帧封装 + 心跳
│   ├── schemas.py             请求响应模型
│   ├── demo_data.py           演示模式样例数据
│   ├── routes/
│   │   ├── chat.py            POST /api/chat/stream、POST /api/chat
│   │   ├── reports.py         报告列表 / 预览 / 下载
│   │   └── meta.py            健康检查 / 阶段定义 / 会话历史
│   └── static/index.html      前端单页（无 CDN 依赖）
│
├── agent/                     ← 业务编排层
│   ├── graph.py               retrieval_graph（检索）+ agent_graph（完整）
│   ├── answer.py              Prompt 构造 + 流式生成 + 落盘
│   ├── state.py               AgentState 定义
│   ├── node/                  LangGraph 四个节点
│   │   ├── splitter_and_search_node.py   ① 切分 + 双路检索
│   │   ├── rerank_node.py                ② 证据重排
│   │   ├── fusion_node.py                ③ 融合（知识库 + 网络）
│   │   └── get_answer_and_txt_to_md_node.py  ④ 生成回答 + 落盘
│   ├── database_agents/       MySQL / Milvus 检索与入库
│   ├── internet_agent/        Tavily 网络检索
│   └── merge_agent/           证据融合 + pub/sub 订阅注册
│
├── client/                    基础设施客户端（LLM / 嵌入 / Milvus / Mongo / Tavily / pub-sub）
├── config/                    配置中心（pydantic-settings 读 .env）
├── logger_set/                loguru 配置
├── prompt/                    Prompt 模板
├── data/                      原始数据（OBDex yaml、投诉 csv）
├── model/                     本地模型（bge-m3、bge-reranker-large）
├── output/                    诊断报告落盘目录
├── tests/                     自检脚本
├── tools/                     一次性工具（权重格式转换）
├── REVIEW.md                  代码审查报告
└── requirements.txt
```

---

## 三、诊断流水线

```
用户描述
   │
   ├─① 语义切分      jieba TF-IDF 抽关键词
   ├─② 双路检索      ┌ Milvus：BGE-M3 稠密+稀疏混合检索（权重 0.5/0.5）→ 故障码知识
   │                 └ MySQL ：problem_desc / car_model LIKE 关键词 → 同车型投诉
   ├─③ 证据重排      bge-reranker 交叉打分 → 断崖截断 → 投诉按相关度过滤
   ├─④ 网络检索      Tavily：同车型维修形势 / 该故障的公开报道
   ├─⑤ 证据融合      三路证据归一成统一结构（故障码 + 投诉 + 网页）
   └─⑥ 生成报告      LLM 流式输出 → Markdown 落盘 → 写 Mongo 历史
```

**为什么是「固定编排」而不是 ReAct 自主 Agent**：诊断类任务要求结论可复现、每一步可解释、成本可预估。
把控制流写死在图里，换来的是「同一个输入永远走同样的路径」，方便评估和排障。
（这也是 Anthropic《Building effective agents》里对生产环境的建议：能用工作流就不要放任自主决策。）

---

## 四、接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat/stream` | **SSE 流式诊断**（前端用这个） |
| POST | `/api/chat` | 一次性诊断，等全部跑完再返回（脚本/测试用） |
| GET | `/api/reports` | 报告列表 |
| GET | `/api/reports/{name}` | 下载报告（`.md`） |
| GET | `/api/reports/{name}/raw` | 预览 Markdown 原文 |
| GET | `/api/health` | 依赖连通性 + 缺包检查 |
| GET | `/api/stages` | 阶段定义（前端据此画步骤条） |
| GET | `/api/sessions` | 历史会话列表（Mongo） |
| GET | `/api/sessions/{id}` | 单个会话历史 |

### SSE 事件协议

`POST /api/chat/stream`，请求体：

```json
{ "query": "我的速腾怠速抖动加速无力", "session_id": "", "demo": false }
```

响应 `Content-Type: text/event-stream`，事件序列：

| 事件 | 载荷 | 说明 |
|---|---|---|
| `session` | `{session_id, query, demo, stages, started_at}` | 流开始，告诉前端要跑哪几个阶段 |
| `stage` | `{key, status, detail, elapsed_ms}` | `status` ∈ `running` / `done` / `error` / `skip` |
| `evidence` | `{vector, sql, web}` | 三路证据，用于证据面板 |
| `token` | `{text}` | 大模型增量文本 |
| `done` | `{md_name, md_path, chars, elapsed_ms, warnings}` | 结束，带报告文件名 |
| `error` | `{stage, message}` | 某阶段失败（不一定中断整条流） |
| `: ping` | — | 每 10 秒一次心跳注释行，防代理断连 |

一行一个 SSE 帧，格式是标准的：

```
event: token
data: {"text": "## 一、结论"}

```

### 为什么前端不用 `EventSource`

`EventSource` 只能发 GET，故障描述动辄上百字，塞 URL 里又要 encode 又有长度风险。
所以前端用 `fetch` + `ReadableStream` 自己读 `text/event-stream` —— **协议完全一样**，只是可以带 JSON body、可以中途 abort。
解析逻辑见 `app/static/index.html` 里的 `streamDiagnosis()`。

---

## 五、已验证 / 未验证

**已实际跑通：**

- 服务启动、`/`、`/docs`、`/api/health`、`/api/stages`、`/api/reports`（含下载与目录穿越防护）
- 演示模式完整 SSE 流：`session` → 12 个 `stage` → `evidence` → 127 个 `token` → `done`，报告正常落盘
- 前端逻辑（Node + DOM stub，灌入真实 SSE 帧）：SSE 解析 → Markdown 渲染（标题/表格/列表/引用/代码块）→ 步骤条状态 → 会话回填
- 核心逻辑自检 `python tests/test_core.py`：6 项全通过
- 投诉库实测：`car_fault_01` 297 行、`car_fault_02` 300 行，字段与 SQLAlchemy 模型一致，`LIKE '%抖动%'` 能召回

**暂未端到端验证（受本机环境限制，不是代码问题）：**

- Milvus 未启动 → 向量检索走降级分支（返回空，链路继续）
- 缺 `aiomysql` / `pymongo` / `tavily` → 对应能力降级，`/api/health` 会明确列出
- BGE-M3 加载需要 torch ≥ 2.6（或先转 safetensors），见 `REVIEW.md` 第五节

---

## 六、自检与排障

```bash
python tests/test_core.py              # 核心逻辑自检
python tests/check_mysql.py            # 投诉库表结构 + LIKE 召回
node tests/frontend_check.js           # 前端逻辑（需先 export 一份 SSE 输出到 SSE_FILE）
```

排障顺序建议：

1. `GET /api/health` —— 先看哪一项是红的（数据库连不上 / 缺包 / 没配 key）
2. 看控制台日志 —— 每个阶段都有 `[xxx_node] 开始 / 结束`，卡在哪一步一目了然
3. 想跳过所有外部依赖，就用 `demo: true` 先确认「前端 + SSE + 落盘」这条链路是通的
