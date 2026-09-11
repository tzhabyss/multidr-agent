const state = {
  sessionId: localStorage.getItem("dr-session-id") || createId("session"),
  sessionTitle: "新研究",
  sessions: [],
  sessionOffset: 0,
  sessionHasMore: false,
  messages: [],
  messageOffset: 0,
  messageHasMore: false,
  messageCount: 0,
  runId: null,
  running: false,
  startedAt: null,
  timer: null,
  report: "",
  events: [],
};

const phaseOrder = ["memory", "planning", "research", "synthesis", "writing", "reviewing", "artifact"];
const phaseAliases = {
  session: "memory", clarification: "memory", clarifying: "memory", clarifier: "memory",
  plan: "planning", planner: "planning",
  worker: "research", worker_loop: "research", worker_strategy: "research", worker_reflection: "research",
  strategy: "research", reflection: "research", search: "research", searching: "research",
  selection: "research", fetch: "research", fetching: "research", extraction: "research",
  synthesizer: "synthesis", writer: "writing", reviewer: "reviewing",
  citation: "reviewing", finalization: "artifact", artifacts: "artifact",
};
const modeNames = { quick: "快速", standard: "标准", deep: "深入", max: "最大" };
const byId = (id) => document.getElementById(id);

function createId(prefix) {
  if (crypto.randomUUID) return `${prefix}-${crypto.randomUUID().replaceAll("-", "").slice(0, 16)}`;
  return `${prefix}-${Date.now().toString(36)}${Math.random().toString(36).slice(2, 9)}`;
}

function toast(message, kind = "normal") {
  const element = byId("toast");
  element.textContent = message;
  element.className = `toast visible ${kind === "error" ? "error" : ""}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.className = "toast"; }, 2600);
}

function setSession(sessionId, title = "新研究") {
  state.sessionId = sessionId;
  state.sessionTitle = title;
  localStorage.setItem("dr-session-id", sessionId);
  byId("currentSessionTitle").textContent = title;
  document.title = `${title} — Deep Research`;
  renderSessionList();
}

function formatSessionTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  if (date.toDateString() === now.toDateString()) return date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

function resetRunView() {
  state.runId = null;
  state.report = "";
  state.events = [];
  byId("pipelineSection").hidden = true;
  byId("reportSection").hidden = true;
  byId("activityList").replaceChildren();
  byId("activityEmpty").hidden = false;
  byId("reportBody").replaceChildren();
  byId("reportState").textContent = "尚未生成";
  byId("runStatus").textContent = "空闲";
  byId("currentAction").textContent = "等待新的研究问题。";
  byId("runIdValue").textContent = "尚未创建";
  byId("resumeButton").hidden = true;
  byId("copyReportButton").disabled = true;
  byId("downloadReportButton").disabled = true;
  ["roundMetric", "workerMetric", "sourceMetric", "searchMetric", "fetchMetric"].forEach((id) => { byId(id).textContent = "—"; });
  byId("claimCount").textContent = "0";
  byId("citationState").textContent = "—";
  byId("sourceList").innerHTML = '<p class="empty-note">研究完成后显示来源。</p>';
  document.querySelectorAll(".dag-node, .dag-edge").forEach((node) => node.classList.remove("active", "complete", "failed"));
  stopTimer();
}

function resetConversation() {
  resetRunView();
  state.messages = [];
  state.messageOffset = 0;
  state.messageHasMore = false;
  state.messageCount = 0;
  byId("conversationHistory").replaceChildren();
  byId("conversationHistory").hidden = true;
  byId("welcomeState").hidden = false;
  byId("currentSessionMeta").textContent = "普通研究 · 持久记忆";
  byId("questionInput").value = "";
  resizeComposer();
  byId("mainScroller").scrollTop = 0;
}

function startTimer() {
  stopTimer();
  state.startedAt = Date.now();
  const update = () => {
    const seconds = Math.floor((Date.now() - state.startedAt) / 1000);
    byId("elapsedTime").textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  };
  update();
  state.timer = setInterval(update, 1000);
}

function stopTimer() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
}

function canonicalPhase(raw) { return phaseOrder.includes(raw) ? raw : (phaseAliases[raw] || raw); }

function updateDag(event) {
  const index = phaseOrder.indexOf(canonicalPhase(event.phase));
  if (index < 0) return;
  phaseOrder.forEach((name, position) => {
    const node = document.querySelector(`.dag-node[data-phase="${name}"]`);
    node.classList.toggle("complete", position < index || (position === index && event.status === "completed"));
    node.classList.toggle("active", position === index && ["started", "progress", "heartbeat"].includes(event.status));
    node.classList.toggle("failed", position === index && event.status === "failed");
  });
  document.querySelectorAll(".dag-edge").forEach((edge, position) => edge.classList.toggle("complete", position < index));
}

function updateMetrics(event) {
  const metrics = event.metrics || {};
  if (event.round_no) byId("roundMetric").textContent = String(event.round_no);
  if (metrics.total_workers !== undefined) byId("workerMetric").textContent = `${metrics.finished_workers || 0}/${metrics.total_workers}`;
  const mappings = [
    ["sourceMetric", ["source_count", "sources", "effective_sources"]],
    ["searchMetric", ["searches", "search_count", "total_searches"]],
    ["fetchMetric", ["fetches", "fetch_count", "total_fetches"]],
  ];
  mappings.forEach(([id, keys]) => {
    const key = keys.find((candidate) => metrics[candidate] !== undefined);
    if (key) byId(id).textContent = String(metrics[key]);
  });
}

function addActivity(event) {
  byId("activityEmpty").hidden = true;
  const item = document.createElement("li");
  item.className = "activity-item";
  item.dataset.status = event.status;
  const time = document.createElement("span");
  time.className = "activity-time";
  time.textContent = `${Number(event.elapsed_seconds || 0).toFixed(1)}s`;
  const marker = document.createElement("i");
  marker.className = "activity-marker";
  const message = document.createElement("span");
  message.className = "activity-message";
  message.textContent = event.message || event.event_type;
  const role = document.createElement("span");
  role.className = "activity-role";
  role.textContent = event.worker_id || event.agent_role || event.phase;
  item.append(time, marker, message, role);
  byId("activityList").append(item);
  state.events.push(event);
  if (state.events.length > 80) {
    byId("activityList").firstElementChild?.remove();
    state.events.shift();
  }
}

function handleProgress(event) {
  byId("pipelineSection").hidden = false;
  if (event.run_id) {
    state.runId = event.run_id;
    byId("runIdValue").textContent = event.run_id;
    byId("resumeButton").hidden = false;
  }
  byId("runStatus").textContent = event.status === "heartbeat" ? "运行中" : statusName(event.status);
  byId("headerStatus").textContent = event.status === "failed" ? "失败" : "运行中";
  byId("currentAction").textContent = event.message || "正在处理";
  byId("runEyebrow").textContent = event.round_no ? `第 ${event.round_no} 轮` : "实时运行";
  updateDag(event);
  updateMetrics(event);
  if (event.visibility !== "quiet" || ["failed", "warning"].includes(event.status)) addActivity(event);
}

function statusName(status) {
  return ({ started: "启动", progress: "运行中", completed: "完成", failed: "失败", warning: "注意", heartbeat: "运行中" })[status] || status;
}

function filterOperationalSections(lines) {
  const hiddenTitles = new Set(["证据边界", "研究假设"]);
  const visible = [];
  let hiding = false;
  lines.forEach((line) => {
    const heading = line.trimEnd().match(/^(#{1,6})\s+(.+)$/);
    if (heading && heading[1].length <= 2) {
      if (heading[1].length === 2 && hiddenTitles.has(heading[2].trim())) {
        hiding = true;
        return;
      }
      hiding = false;
    }
    if (!hiding) visible.push(line);
  });
  return visible;
}

function parseTableRow(line) {
  let value = line.trim();
  if (!value.includes("|")) return null;
  if (value.startsWith("|")) value = value.slice(1);
  if (value.endsWith("|") && !value.endsWith("\\|")) value = value.slice(0, -1);
  const cells = [];
  let cell = "";
  for (let index = 0; index < value.length; index += 1) {
    if (value[index] === "\\" && value[index + 1] === "|") {
      cell += "|";
      index += 1;
    } else if (value[index] === "|") {
      cells.push(cell.trim());
      cell = "";
    } else {
      cell += value[index];
    }
  }
  cells.push(cell.trim());
  return cells;
}

function isTableSeparator(cells) {
  return Boolean(cells?.length) && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function appendTableCellText(cell, value) {
  String(value).split(/<br\s*\/?>/i).forEach((part, index) => {
    if (index) cell.append(document.createElement("br"));
    cell.append(document.createTextNode(part));
  });
}

function createTable(header, rows) {
  const wrapper = document.createElement("div");
  wrapper.className = "table-scroll";
  wrapper.setAttribute("role", "region");
  wrapper.setAttribute("aria-label", "可横向滚动的表格");
  wrapper.tabIndex = 0;
  const table = document.createElement("table");
  table.className = "report-table";
  const headRow = document.createElement("tr");
  header.forEach((value) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    appendTableCellText(cell, value);
    headRow.append(cell);
  });
  const head = document.createElement("thead");
  head.append(headRow);
  const body = document.createElement("tbody");
  rows.forEach((values) => {
    const row = document.createElement("tr");
    values.forEach((value) => {
      const cell = document.createElement("td");
      appendTableCellText(cell, value);
      row.append(cell);
    });
    body.append(row);
  });
  table.append(head, body);
  wrapper.append(table);
  return wrapper;
}

function renderMarkdown(root, markdown, { hideOperationalSections = false } = {}) {
  let list = null;
  let code = null;
  const rawLines = String(markdown || "").split(/\r?\n/);
  const lines = hideOperationalSections ? filterOperationalSections(rawLines) : rawLines;
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const rawLine = lines[lineIndex];
    const line = rawLine.trimEnd();
    if (line.startsWith("```")) {
      if (code) { root.append(code); code = null; } else { code = document.createElement("pre"); }
      list = null;
      continue;
    }
    if (code) { code.textContent += `${rawLine}\n`; continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      list = null;
      const element = document.createElement(`h${heading[1].length}`);
      element.textContent = heading[2];
      root.append(element);
      continue;
    }
    const headerCells = parseTableRow(line);
    const separatorCells = parseTableRow(lines[lineIndex + 1] || "");
    if (headerCells && separatorCells && headerCells.length === separatorCells.length && isTableSeparator(separatorCells)) {
      list = null;
      const rows = [];
      lineIndex += 2;
      while (lineIndex < lines.length) {
        const rowCells = parseTableRow(lines[lineIndex]);
        if (!rowCells || rowCells.length !== headerCells.length || !lines[lineIndex].trim()) break;
        rows.push(rowCells);
        lineIndex += 1;
      }
      lineIndex -= 1;
      root.append(createTable(headerCells, rows));
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      if (!list) { list = document.createElement("ul"); root.append(list); }
      const item = document.createElement("li");
      item.textContent = line.replace(/^[-*]\s+/, "");
      list.append(item);
      continue;
    }
    list = null;
    if (!line.trim()) continue;
    if (/^---+$/.test(line.trim())) {
      const divider = document.createElement("div");
      divider.className = "report-divider";
      root.append(divider);
      continue;
    }
    const paragraph = document.createElement("p");
    paragraph.textContent = line;
    root.append(paragraph);
  }
  if (code) root.append(code);
}

function researchMessagePreview(content) {
  const withoutTrace = String(content || "").split(/\n\nResearch run:/i)[0].trim();
  if (withoutTrace.length <= 1400) return withoutTrace;
  return `${withoutTrace.slice(0, 1400).trimEnd()}…`;
}

function createConversationMessage(role, content, runId = null) {
  const item = document.createElement("article");
  item.className = `message message-${role}${runId ? " message-research" : ""}`;
  if (runId) item.dataset.runId = runId;
  if (role === "assistant") {
    const mark = document.createElement("span");
    mark.className = "assistant-mark";
    mark.textContent = "DR";
    item.append(mark);
  }
  const label = document.createElement("span");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "Deep Research";
  const body = document.createElement("div");
  body.className = "message-content";
  renderMarkdown(body, runId ? researchMessagePreview(content) : content);
  item.append(label, body);
  if (runId) {
    const action = document.createElement("button");
    action.type = "button";
    action.className = "message-run-action";
    action.dataset.runId = runId;
    action.textContent = runId === state.runId ? "当前报告" : "查看此报告";
    action.addEventListener("click", async () => {
      if (runId === state.runId) {
        byId("reportSection").scrollIntoView({ block: "start" });
        return;
      }
      action.disabled = true;
      action.textContent = "正在读取…";
      await inspectRun(runId, { scroll: false });
      document.querySelectorAll(".message-run-action").forEach((button) => {
        button.textContent = button.dataset.runId === state.runId ? "当前报告" : "查看此报告";
        button.disabled = false;
      });
      byId("reportSection").scrollIntoView({ block: "start" });
    });
    item.append(action);
  }
  return item;
}

function appendConversationMessage(role, content, runId = null) {
  byId("welcomeState").hidden = true;
  const stream = byId("conversationHistory");
  stream.hidden = false;
  stream.append(createConversationMessage(role, content, runId));
}

function renderConversation() {
  const stream = byId("conversationHistory");
  stream.replaceChildren();
  if (state.messageHasMore) {
    const loader = document.createElement("button");
    loader.type = "button";
    loader.className = "history-load-button";
    loader.textContent = "加载更早消息";
    loader.addEventListener("click", loadOlderMessages);
    stream.append(loader);
  }
  state.messages.forEach((message) => {
    stream.append(createConversationMessage(message.role, message.content, message.run_id));
  });
  if (!state.messages.length) {
    stream.hidden = true;
    byId("welcomeState").hidden = false;
  } else {
    stream.hidden = false;
    byId("welcomeState").hidden = true;
  }
}

function renderReport(markdown) {
  const root = byId("reportBody");
  root.replaceChildren();
  renderMarkdown(root, markdown, { hideOperationalSections: true });
}

function handleResult(result, { appendReply = true, scroll = true } = {}) {
  state.running = false;
  stopTimer();
  const run = result.run || (result.run_id ? result : null);
  if (result.session_id) setSession(result.session_id, state.sessionTitle);
  if (run?.run_id) {
    state.runId = run.run_id;
    byId("runIdValue").textContent = run.run_id;
    byId("resumeButton").hidden = false;
  }
  if (!run && appendReply && result.message) {
    state.messages.push({ role: "assistant", content: result.message, run_id: null });
    state.messageCount += 1;
    appendConversationMessage("assistant", result.message);
  }
  if (run && appendReply && result.message) {
    state.messages.push({ role: "assistant", content: result.message, run_id: run.run_id });
    state.messageCount += 1;
    appendConversationMessage("assistant", result.message, run.run_id);
  }
  const markdown = run?.markdown || "";
  if (markdown) {
    state.report = markdown;
    renderReport(markdown);
    byId("reportSection").hidden = false;
  }
  const status = run?.status || result.action || "completed";
  byId("runStatus").textContent = status;
  byId("headerStatus").textContent = "就绪";
  byId("reportState").textContent = status === "completed_verified" ? "引用已验证" : status === "completed_partial" ? "部分完成" : "已生成";
  byId("copyReportButton").disabled = !markdown;
  byId("downloadReportButton").disabled = !markdown;
  byId("claimCount").textContent = String(run?.claims?.length || 0);
  byId("citationState").textContent = run?.gate ? (run.gate.passed ? "通过" : "未通过") : "—";
  const sources = run?.sources || [];
  byId("sourceMetric").textContent = String(sources.length || 0);
  renderSources(sources);
  if (run?.usage) {
    byId("searchMetric").textContent = String(run.usage.searches ?? byId("searchMetric").textContent);
    byId("fetchMetric").textContent = String(run.usage.fetches ?? byId("fetchMetric").textContent);
  }
  markDagComplete(status);
  byId("submitButton").disabled = false;
  byId("submitButton").querySelector("span").textContent = "开始研究";
  loadSessions();
  if (scroll) scrollMainToBottom();
}

function markDagComplete(status) {
  if (["completed_verified", "completed_partial", "research"].includes(status)) {
    document.querySelectorAll(".dag-node, .dag-edge").forEach((element) => element.classList.add("complete"));
  } else if (["respond_from_context", "clarify"].includes(status)) {
    document.querySelector('.dag-node[data-phase="memory"]')?.classList.add("complete");
  }
  document.querySelectorAll(".dag-node").forEach((element) => element.classList.remove("active"));
}

function renderSources(sources) {
  const root = byId("sourceList");
  root.replaceChildren();
  if (!sources.length) { root.innerHTML = '<p class="empty-note">本次没有返回来源。</p>'; return; }
  sources.slice(0, 24).forEach((source) => {
    const item = document.createElement("div");
    item.className = "source-item";
    const title = document.createElement("strong");
    title.textContent = source.title || "未命名来源";
    const url = document.createElement("span");
    url.textContent = source.canonical_uri || source.url || "";
    item.append(title, url);
    root.append(item);
  });
}

async function parseEventStream(response, handlers) {
  if (!response.ok || !response.body) throw new Error(`请求失败：HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data = [];
      block.split("\n").forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) data.push(line.slice(5).trim());
      });
      if (data.length) handlers[event]?.(JSON.parse(data.join("\n")));
    }
    if (done) break;
  }
}

async function beginResearch(message, mode) {
  resetRunView();
  state.running = true;
  if (state.sessionTitle === "新研究") setSession(state.sessionId, message.replace(/\s+/g, " ").slice(0, 48));
  state.messages.push({ role: "user", content: message, run_id: null });
  state.messageCount += 1;
  appendConversationMessage("user", message);
  byId("pipelineSection").hidden = false;
  startTimer();
  byId("submitButton").disabled = true;
  byId("submitButton").querySelector("span").textContent = "研究中";
  byId("runStatus").textContent = "启动";
  byId("headerStatus").textContent = "运行中";
  byId("modeDetail").textContent = modeNames[mode];
  byId("questionInput").value = "";
  resizeComposer();
  scrollMainToBottom();
  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: state.sessionId, mode, profile_id: "general_research", locale: "zh-CN" }),
    });
    await parseEventStream(response, {
      ready: (data) => { state.sessionId = data.session_id; localStorage.setItem("dr-session-id", data.session_id); },
      progress: handleProgress,
      result: (data) => handleResult(data),
      error: handleStreamError,
    });
  } catch (error) {
    handleStreamError({ message: error.message, error_type: error.name });
  }
}

async function resumeRun(runId) {
  if (!runId || state.running) return;
  resetRunView();
  state.runId = runId;
  state.running = true;
  byId("pipelineSection").hidden = false;
  byId("runIdValue").textContent = runId;
  byId("headerStatus").textContent = "运行中";
  startTimer();
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/resume/stream`, { method: "POST" });
    await parseEventStream(response, { progress: handleProgress, result: handleResult, error: handleStreamError });
  } catch (error) {
    handleStreamError({ message: error.message, error_type: error.name });
  }
}

async function inspectRun(runId, { scroll = false } = {}) {
  if (!runId || state.running) return;
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    if (!response.ok) throw new Error(`无法读取任务：HTTP ${response.status}`);
    handleResult(await response.json(), { appendReply: false, scroll });
  } catch (error) {
    toast(error.message, "error");
  }
}

function handleStreamError(error) {
  state.running = false;
  stopTimer();
  byId("submitButton").disabled = false;
  byId("submitButton").querySelector("span").textContent = "重新开始";
  byId("runStatus").textContent = "失败";
  byId("headerStatus").textContent = "需要处理";
  byId("currentAction").textContent = `${error.error_type || "Error"}：${error.message || "运行失败"}`;
  const active = document.querySelector(".dag-node.active");
  active?.classList.add("failed");
  active?.classList.remove("active");
  toast(error.message || "研究任务失败", "error");
}

async function loadSessions({ append = false } = {}) {
  try {
    const offset = append ? state.sessionOffset : 0;
    const response = await fetch(`/api/sessions?limit=50&offset=${offset}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (append) {
      const known = new Set(state.sessions.map((session) => session.session_id));
      state.sessions.push(...payload.sessions.filter((session) => !known.has(session.session_id)));
    } else {
      state.sessions = payload.sessions;
    }
    state.sessionOffset = payload.next_offset ?? state.sessions.length;
    state.sessionHasMore = Boolean(payload.has_more);
    const active = state.sessions.find((session) => session.session_id === state.sessionId);
    if (active) state.sessionTitle = active.title;
    renderSessionList();
  } catch (error) {
    byId("sessionList").innerHTML = '<p class="sidebar-empty">无法读取对话。</p>';
  }
}

function renderSessionList() {
  const root = byId("sessionList");
  const query = byId("sessionSearch").value.trim().toLowerCase();
  const sessions = state.sessions.filter((session) => session.title.toLowerCase().includes(query));
  byId("sessionCount").textContent = `${state.sessions.length}${state.sessionHasMore ? "+" : ""}`;
  root.replaceChildren();
  if (!sessions.length) {
    root.innerHTML = `<p class="sidebar-empty">${query ? "没有匹配的对话。" : "开始第一次研究后，对话会出现在这里。"}</p>`;
    return;
  }
  sessions.forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `session-item ${session.session_id === state.sessionId ? "active" : ""}`;
    button.setAttribute("aria-current", session.session_id === state.sessionId ? "page" : "false");
    const title = document.createElement("strong");
    title.textContent = session.title;
    const meta = document.createElement("small");
    meta.textContent = `${session.message_count} 条消息 · ${formatSessionTime(session.updated_at)}`;
    button.append(title, meta);
    button.addEventListener("click", () => selectSession(session));
    root.append(button);
  });
  if (state.sessionHasMore && !query) {
    const more = document.createElement("button");
    more.type = "button";
    more.className = "session-load-more";
    more.textContent = "加载更多对话";
    more.addEventListener("click", () => loadSessions({ append: true }));
    root.append(more);
  }
}

async function loadOlderMessages() {
  if (!state.messageHasMore || !state.sessionId) return;
  const stream = byId("conversationHistory");
  const previousHeight = stream.scrollHeight;
  const loader = stream.querySelector(".history-load-button");
  if (loader) {
    loader.disabled = true;
    loader.textContent = "正在读取…";
  }
  try {
    const response = await fetch(
      `/api/sessions/${encodeURIComponent(state.sessionId)}?limit=80&offset=${state.messageOffset}`,
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const detail = await response.json();
    const known = new Set(
      state.messages.map((message) => message.message_id || `${message.role}:${message.created_at}:${message.content}`),
    );
    const older = detail.messages.filter((message) => {
      const key = message.message_id || `${message.role}:${message.created_at}:${message.content}`;
      return !known.has(key);
    });
    state.messages = [...older, ...state.messages];
    state.messageOffset = detail.next_offset ?? state.messageOffset;
    state.messageHasMore = Boolean(detail.has_more);
    renderConversation();
    byId("mainScroller").scrollTop += stream.scrollHeight - previousHeight;
  } catch (error) {
    if (loader) {
      loader.disabled = false;
      loader.textContent = "重试加载更早消息";
    }
    toast("无法读取更早的消息。", "error");
  }
}

async function selectSession(session, { force = false } = {}) {
  if (state.running || (!force && session.session_id === state.sessionId)) return;
  resetConversation();
  setSession(session.session_id, session.title);
  closeSidebar();
  byId("currentSessionMeta").textContent = `正在读取 · ${session.message_count} 条消息`;
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(session.session_id)}?limit=80&offset=0`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const detail = await response.json();
    state.messages = detail.messages;
    state.messageOffset = detail.next_offset ?? detail.messages.length;
    state.messageHasMore = Boolean(detail.has_more);
    state.messageCount = detail.message_count ?? detail.messages.length;
    renderConversation();
    byId("currentSessionMeta").textContent = `${state.messageCount} 条消息 · ${formatSessionTime(detail.updated_at)} 更新`;
    const lastRun = detail.related_run_ids.at(-1);
    if (lastRun) await inspectRun(lastRun, { scroll: false });
    byId("mainScroller").scrollTop = 0;
  } catch (error) {
    toast("无法打开该对话。", "error");
  }
}

async function loadRuntime() {
  try {
    const [healthResponse, runtimeResponse, skillResponse] = await Promise.all([
      fetch("/api/health"), fetch("/api/runtime"), fetch("/api/skills"),
    ]);
    if (!healthResponse.ok || !runtimeResponse.ok || !skillResponse.ok) throw new Error("控制台初始化失败");
    const [health, runtime, skills] = await Promise.all([healthResponse.json(), runtimeResponse.json(), skillResponse.json()]);
    byId("apiDot").className = "status-dot ready";
    byId("apiStatus").textContent = "运行就绪";
    const memoryCount = (health.memory?.indexed_documents || 0) + (health.memory?.active_facts || 0);
    byId("runtimeSummary").textContent = `${memoryCount} 条记忆 · ${skills.skills.length} 个 Skills`;
    renderSkills(skills.skills);
    const normal = runtime.profiles.general_research;
    document.querySelector(".skill-state strong").textContent = normal.skill_evolution_mode === "observe" ? "观察模式已开启" : "观察模式未开启";
  } catch (error) {
    byId("apiDot").className = "status-dot error";
    byId("apiStatus").textContent = "连接失败";
    byId("headerStatus").textContent = "离线";
    toast(error.message, "error");
  }
}

function renderSkills(skills) {
  const root = byId("skillList");
  root.replaceChildren();
  skills.forEach((skill) => {
    const item = document.createElement("div");
    item.className = "skill-item";
    const title = document.createElement("strong");
    title.textContent = `${skill.skill_id} · ${skill.version}`;
    const description = document.createElement("p");
    description.textContent = skill.description;
    item.append(title, description);
    root.append(item);
  });
}

function newConversation() {
  if (state.running) { toast("当前研究完成后再新建对话。", "error"); return; }
  resetConversation();
  setSession(createId("session"), "新研究");
  closeSidebar();
  byId("questionInput").focus();
}

function resizeComposer() {
  const input = byId("questionInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
}

function scrollMainToBottom() {
  requestAnimationFrame(() => { byId("mainScroller").scrollTop = byId("mainScroller").scrollHeight; });
}

function toggleInspector(force) {
  const shell = byId("appShell");
  const hidden = force === undefined ? !shell.classList.contains("inspector-hidden") : !force;
  shell.classList.toggle("inspector-hidden", hidden);
  byId("inspectorToggle").setAttribute("aria-expanded", String(!hidden));
}

function openSidebar() {
  byId("appShell").classList.add("sidebar-open");
  byId("scrim").hidden = false;
}
function closeSidebar() {
  byId("appShell").classList.remove("sidebar-open");
  byId("scrim").hidden = true;
}

function downloadReport() {
  if (!state.report) return;
  const blob = new Blob([state.report], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.runId || "deep-research-report"}.md`;
  link.click();
  URL.revokeObjectURL(url);
}

byId("researchForm").addEventListener("submit", (event) => {
  event.preventDefault();
  if (state.running) return;
  const message = byId("questionInput").value.trim();
  const mode = byId("modeSelect").value;
  if (!message) { toast("请输入需要研究的问题。", "error"); return; }
  beginResearch(message, mode);
});
byId("questionInput").addEventListener("input", resizeComposer);
byId("questionInput").addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") byId("researchForm").requestSubmit();
});
byId("newSessionButton").addEventListener("click", newConversation);
byId("sessionSearch").addEventListener("input", renderSessionList);
byId("runIdValue").addEventListener("click", () => state.runId && navigator.clipboard.writeText(state.runId).then(() => toast("运行 ID 已复制")));
byId("resumeButton").addEventListener("click", () => resumeRun(state.runId));
byId("copyReportButton").addEventListener("click", () => navigator.clipboard.writeText(state.report).then(() => toast("报告已复制")));
byId("downloadReportButton").addEventListener("click", downloadReport);
byId("inspectorToggle").addEventListener("click", () => toggleInspector());
byId("inspectorClose").addEventListener("click", () => toggleInspector(false));
byId("sidebarToggle").addEventListener("click", openSidebar);
byId("sidebarClose").addEventListener("click", closeSidebar);
byId("scrim").addEventListener("click", closeSidebar);
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") { event.preventDefault(); newConversation(); }
  if (event.key === "Escape") closeSidebar();
});
document.querySelectorAll(".inspector-tab").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll(".inspector-tab").forEach((item) => { item.classList.remove("active"); item.setAttribute("aria-selected", "false"); });
  document.querySelectorAll(".inspector-panel").forEach((panel) => { panel.classList.remove("active"); panel.hidden = true; });
  button.classList.add("active");
  button.setAttribute("aria-selected", "true");
  const panel = byId(button.dataset.panel);
  panel.hidden = false;
  panel.classList.add("active");
}));

setSession(state.sessionId);
resizeComposer();
Promise.all([loadRuntime(), loadSessions()]).then(async () => {
  const active = state.sessions.find((session) => session.session_id === state.sessionId);
  if (active) await selectSession(active, { force: true });
});
