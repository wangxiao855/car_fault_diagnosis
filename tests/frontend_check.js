/* 前端逻辑无浏览器测试：用最小 DOM stub 跑真实的 index.html <script>，并灌入服务端真实 SSE 帧 */
const fs = require("fs");

const html = fs.readFileSync("D:/python/car_fault_agent/app/static/index.html", "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];

/* ---------------- 最小 DOM stub ---------------- */
function makeEl(tag) {
  const el = {
    tagName: (tag || "div").toUpperCase(),
    children: [],
    _html: "",
    _qs: new Map(),
    className: "",
    textContent: "",
    value: "",
    checked: false,
    disabled: false,
    href: "",
    title: "",
    style: {},
    dataset: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    appendChild(child) { this.children.push(child); return child; },
    remove() {},
    addEventListener() {},
    setAttribute() {},
    setAttributeNS() {},
    focus() {},
    get innerHTML() { return this._html; },
    // 与浏览器一致：设置 innerHTML 会清空原有子节点
    set innerHTML(v) { this._html = String(v); this.children.length = 0; },
    // 永不为 null 的查询：真实浏览器里 innerHTML 会生成节点，这里无需解析
    querySelector(sel) {
      if (!this._qs.has(sel)) this._qs.set(sel, makeEl("div"));
      return this._qs.get(sel);
    },
    querySelectorAll() { return []; }
  };
  return el;
}

const byId = new Map();
global.document = {
  getElementById(id) {
    if (!byId.has(id)) byId.set(id, makeEl("div"));
    return byId.get(id);
  },
  createElement(tag) { return makeEl(tag); },
  querySelector() { return makeEl("div"); },
  addEventListener() {}
};
global.window = global;
global.navigator = { clipboard: { writeText: async () => {} } };
global.requestAnimationFrame = (cb) => setTimeout(cb, 0);
global.setInterval = () => 0;
global.clearInterval = () => {};
global.location = { href: "http://127.0.0.1:8011/" };

/* ---------------- fetch stub ---------------- */
const sseText = fs.readFileSync(process.env.SSE_FILE, "utf8");
global.fetch = async (url, opts) => {
  if (String(url).includes("/api/chat/stream")) {
    // 把真实 SSE 文本切成 3 段，模拟分块到达
    const chunks = [sseText.slice(0, 4096), sseText.slice(4096, 8192), sseText.slice(8192)];
    let i = 0;
    const body = {
      getReader() {
        return {
          async read() {
            if (i >= chunks.length) return { done: true, value: undefined };
            const value = new TextEncoder().encode(chunks[i++]);
            return { done: false, value };
          }
        };
      }
    };
    return { ok: true, status: 200, statusText: "OK", body };
  }
  if (String(url).includes("/api/health")) {
    return {
      ok: true, status: 200,
      json: async () => ({
        ok: true, demo_mode: false, version: "test",
        services: [{ name: "MySQL（投诉库）", ok: true, detail: "" }, { name: "Milvus", ok: false, detail: "" }]
      })
    };
  }
  return { ok: true, status: 200, json: async () => ({ sessions: [], available: true }) };
};

/* ---------------- 执行真实脚本 ---------------- */
(async () => {
  try {
    eval(script);
  } catch (e) {
    console.log("[FAIL] 初始化脚本抛异常:", e.message);
    process.exit(1);
  }
  console.log("[PASS] 脚本初始化无异常");

  // 触发一次完整流式诊断
  await streamDiagnosis("我的速腾怠速抖动加速无力");

  // 在 DOM 树里找 answer-body / steps
  function walk(node, hits) {
    if (!node || typeof node !== "object") return;
    if (node.className === "answer-body") hits.body.push(node);
    if (node.className === "steps") hits.steps.push(node);
    if (node.className && String(node.className).startsWith("step ")) hits.stepNodes.push(node.className);
    (node.children || []).forEach((c) => walk(c, hits));
  }
  const hits = { body: [], steps: [], stepNodes: [] };
  walk(document.getElementById("inner"), hits);

  const bodyHtml = hits.body.map((b) => b.innerHTML).join("\n");
  const checks = {
    有回答正文: bodyHtml.length > 500,
    渲染出标题: /<h1>车辆故障诊断报告<\/h1>/.test(bodyHtml),
    渲染出表格: /<table>/.test(bodyHtml),
    渲染出列表: /<li>/.test(bodyHtml),
    步骤条已创建: hits.steps.length > 0 && hits.steps[0].children.length === 6,
    有步骤进入完成态: hits.stepNodes.some((c) => c.includes("done")),
    会话id已回填: document.getElementById("sessionPill").textContent.includes("会话")
  };

  let bad = [];
  for (const [k, v] of Object.entries(checks)) { if (!v) bad.push(k); }
  console.log(JSON.stringify(checks, null, 0));
  console.log(bad.length ? "[FAIL] " + bad.join(", ") : "[PASS] 前端全流程（SSE 解析 + Markdown 渲染 + 步骤条）通过");
  process.exit(bad.length ? 1 : 0);
})();
