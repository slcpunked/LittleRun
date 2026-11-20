/* LittleRun UI. Talks only to the host app through window.chrome.webview messages. */
"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const icon = (name) => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", "#i-" + name);
  s.appendChild(u);
  return s;
};

// ------------------------------------------------------------------ host bridge

const host = (() => {
  const wv = window.chrome && window.chrome.webview;
  const pending = new Map();
  const handlers = {};
  let seq = 0;
  if (wv) {
    wv.addEventListener("message", (e) => {
      const m = e.data;
      if (m.type === "reply") {
        const p = pending.get(m.id);
        if (!p) return;
        pending.delete(m.id);
        m.ok ? p.resolve(m.data) : p.reject(new Error(m.error || "request failed"));
      } else if (m.type === "event") {
        (handlers[m.name] || []).forEach((h) => h(m.data));
      }
    });
  }
  return {
    call(type, data) {
      if (!wv) return Promise.reject(new Error("Not running inside LittleRun"));
      const id = "m" + ++seq;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        wv.postMessage({ id, type, data });
      });
    },
    on(name, fn) { (handlers[name] = handlers[name] || []).push(fn); },
  };
})();

// ------------------------------------------------------------------ state

const S = {
  init: null,
  settings: null,
  chats: [],
  currentId: null,
  mode: "chat",
  thinking: false,
  backend: { state: "stopped", message: "" },
  gen: null, // { reqId, chatId, msg?, span? }
  setupFromApp: false,
};

const model = () => (S.init && S.init.models ? S.init.models[S.settings.modelName || S.init.defaultModel] : null);
const current = () => S.chats.find((c) => c.id === S.currentId) || null;
const storyOn = () => LR.useStory(S.settings, model());
const presets = () => (S.init && S.init.presets) || [];
function characterFor(id) {
  if (id === "custom" && S.settings.customCharacter && S.settings.customCharacter.self.trim()) return S.settings.customCharacter;
  return presets().find((p) => p.id === id) || presets()[0] || { id: "ada", name: "Ada" };
}
// A chat keeps the character it started with; a new chat uses the current setting.
const chatCharacter = (c) => characterFor((c && c.character) || S.settings.character);
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

function toast(text, bad) {
  const t = el("div", "toast" + (bad ? " bad" : ""), text);
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), bad ? 6000 : 2600);
}

// ------------------------------------------------------------------ boot

async function boot() {
  let init;
  try {
    init = await host.call("app.init");
  } catch (e) {
    $("boot").textContent = e.message;
    return;
  }
  applyInit(init);
  $("boot").hidden = true;
  if (init.payloadError) {
    showFatal(init.payloadError);
  } else if (init.setUp) {
    await showApp();
  } else {
    showSetup(false);
  }
}

function applyInit(init) {
  S.init = init;
  S.settings = init.settings;
  S.thinking = !!init.settings.thinking;
  if (init.backend) setBackend(init.backend);
}

function showFatal(msg) {
  $("setup").hidden = false;
  $("setup-form").hidden = true;
  $("setup-progress").hidden = false;
  $("steps").innerHTML = "";
  $("setup-error").hidden = false;
  $("setup-error").textContent = msg;
  $("setup-cancel").hidden = true;
}

// ------------------------------------------------------------------ setup

const STEPS = [
  ["runtime", "Python runtime"],
  ["torch", "PyTorch"],
  ["backend", "LittleRun backend"],
  ["kernel", "Fast CPU kernel"],
  ["verify", "Check the runtime"],
  ["model", "Model files"],
  ["validate", "Verify the model"],
];

function showSetup(fromApp) {
  S.setupFromApp = fromApp;
  $("app").hidden = true;
  $("setup").hidden = false;
  $("setup-form").hidden = false;
  $("setup-progress").hidden = true;
  renderSystem();
  renderModelCard();
  const dev = S.settings.device || "cpu";
  document.querySelector(`input[name=device][value=${dev}]`).checked = !document.querySelector(
    `input[name=device][value=${dev}]`).disabled;
  $("accept").checked = !!S.settings.licenseAccepted;
  $("setup-hint").textContent = fromApp ? "Your chats are kept." : "";
  $("setup-back").hidden = !fromApp;
  $("install").textContent = fromApp ? "Run setup" : "Install";
  updateInstallButton();
}

function renderSystem() {
  const s = S.init.system;
  const m = model();
  const box = $("sys");
  box.innerHTML = "";
  const add = (label, value, warn) => {
    const d = el("div");
    d.appendChild(el("b", null, label));
    const v = el("span", warn ? "warn" : null, value);
    v.title = value;
    d.appendChild(v);
    box.appendChild(d);
  };
  add("Processor", `${s.cpu} (${s.logicalCores} threads)`);
  const availGb = s.availableRamBytes / 2 ** 30;
  add("Memory", `${LR.formatBytes(s.totalRamBytes)} (${LR.formatBytes(s.availableRamBytes)} free)`, availGb < 5);
  const g = s.nvidiaGpu;
  add("Graphics", g ? `${g.name}, ${(g.memoryMiB / 1024).toFixed(0)} GB, driver ${g.driverVersion}` : "No NVIDIA GPU found");
  const need = (m ? m.sizeBytes : 0) + 4 * 2 ** 30;
  add("Disk free", LR.formatBytes(s.freeDiskBytes), s.freeDiskBytes < need);

  const gpu = document.querySelector("input[name=device][value=cuda]");
  const minDriver = S.init.cudaMinDriver || 0;
  if (!g) {
    gpu.disabled = true;
    $("gpu-note").textContent = "Needs an NVIDIA graphics card.";
  } else if (g.driverVersion < minDriver) {
    gpu.disabled = true;
    $("gpu-note").textContent = `Needs NVIDIA driver ${minDriver} or newer (you have ${g.driverVersion}). Update the driver to use the GPU.`;
  } else if (g.memoryMiB < (S.init.minGpuMemoryMiB || 15000)) {
    gpu.disabled = true;
    $("gpu-note").textContent = `Needs a GPU with 16 GB of memory; your ${g.name} has ${(g.memoryMiB / 1024).toFixed(0)} GB.`;
  } else {
    gpu.disabled = false;
    $("gpu-note").textContent = `Uses your ${g.name}: reads prompts about 10× faster, but writes at about CPU speed for now. ` +
      "Uses about 11 GB of its memory (16 GB card required) and installs about 3 GB of CUDA libraries.";
  }
}

function renderModelCard() {
  const m = model();
  const card = $("model-card");
  card.innerHTML = "";
  card.appendChild(el("h3", null, m.displayName));
  const meta = el("div", "meta");
  meta.appendChild(el("span", null, LR.formatBytes(m.sizeBytes)));
  meta.appendChild(el("span", null, m.repo));
  meta.appendChild(el("span", null, "revision " + m.revision.slice(0, 7)));
  meta.appendChild(el("span", null, "License " + m.license.toUpperCase()));
  card.appendChild(meta);
  if (m.note) card.appendChild(el("div", "note", m.note));
  $("download-note").textContent = `${LR.formatBytes(m.sizeBytes)} from Hugging Face at a pinned revision, checked with sha256.`;
  $("license-link").textContent = m.license.toUpperCase();
}

let importDir = null;
let importOk = false;

async function pickFolder() {
  const dir = await host.call("setup.pickFolder");
  if (!dir) return;
  importDir = dir;
  document.querySelector("input[name=source][value=import]").checked = true;
  $("import-path").textContent = dir;
  const r = await host.call("setup.checkFolder", { dir, model: S.settings.modelName || S.init.defaultModel });
  importOk = r.missing.length === 0;
  $("import-note").textContent = importOk
    ? "All model files found. They will be checked against the pinned hashes."
    : "Missing: " + r.missing.join(", ");
  updateInstallButton();
}

function updateInstallButton() {
  const source = document.querySelector("input[name=source]:checked").value;
  const ok = $("accept").checked && (source === "download" || (importDir && importOk));
  $("install").disabled = !ok;
}

function stepRow(id) {
  return $("steps").querySelector(`li[data-step="${id}"]`);
}

async function runSetup() {
  const device = document.querySelector("input[name=device]:checked").value;
  const source = document.querySelector("input[name=source]:checked").value;
  $("setup-form").hidden = true;
  $("setup-progress").hidden = false;
  $("setup-error").hidden = true;
  $("setup-retry").hidden = true;
  $("setup-cancel").hidden = false;
  $("setup-cancel").disabled = false;
  $("dl").hidden = true;
  $("setup-log").textContent = "";
  const ol = $("steps");
  ol.innerHTML = "";
  for (const [id, label] of STEPS) {
    const li = el("li", "pending");
    li.dataset.step = id;
    const ico = el("span", "ico");
    li.appendChild(ico);
    li.appendChild(el("span", null, id === "model" && source === "import" ? "Model folder" : label));
    li.appendChild(el("span", "msg"));
    ol.appendChild(li);
  }
  try {
    const init = await host.call("setup.start", {
      device,
      model: S.settings.modelName || S.init.defaultModel,
      importDir: source === "import" ? importDir : null,
      acceptLicense: $("accept").checked,
    });
    applyInit(init);
    await showApp();
    toast("Setup complete. The model is loading.");
  } catch (e) {
    $("setup-error").hidden = false;
    $("setup-error").textContent = e.message;
    $("setup-cancel").hidden = true;
    $("setup-retry").hidden = false;
  }
}

host.on("setup", (e) => {
  if (e.state === "log") {
    const pre = $("setup-log");
    const lines = (pre.textContent + e.message + "\n").split("\n");
    pre.textContent = lines.slice(-400).join("\n");
    pre.scrollTop = pre.scrollHeight;
    return;
  }
  if (e.step === "finish") return;
  const li = stepRow(e.step);
  if (!li) return;
  li.className = e.state;
  const ico = li.querySelector(".ico");
  ico.innerHTML = "";
  if (e.state === "done") ico.appendChild(icon("check"));
  if (e.state === "failed") ico.appendChild(icon("x"));
  li.querySelector(".msg").textContent = e.state === "running" && e.bytes == null ? e.message : e.state === "done" ? e.message : "";
  li.querySelector(".msg").title = e.message;
  // Mark earlier steps that were skipped as done once a later one starts.
  if (e.state === "running" || e.state === "done") {
    for (const [id] of STEPS) {
      if (id === e.step) break;
      const prev = stepRow(id);
      if (prev && prev.className === "pending") prev.className = "done";
    }
  }
  if (e.step === "model" && e.bytes != null) {
    $("dl").hidden = false;
    $("dl-fill").style.width = ((e.fraction || 0) * 100).toFixed(1) + "%";
    $("dl-bytes").textContent = `${LR.formatBytes(e.bytes)} of ${LR.formatBytes(e.total)}`;
    const eta = e.bytesPerSecond > 0 ? (e.total - e.bytes) / e.bytesPerSecond : null;
    $("dl-rate").textContent = e.bytesPerSecond ? `${LR.formatBytes(e.bytesPerSecond)}/s · ${LR.formatDuration(eta)} left` : "";
  }
  if (e.step === "model" && e.state === "done") $("dl").hidden = true;
});

// ------------------------------------------------------------------ app

async function showApp() {
  $("setup").hidden = true;
  $("app").hidden = false;
  try {
    S.chats = (await host.call("chats.load")) || [];
  } catch (e) {
    S.chats = [];
    toast("Could not read saved chats: " + e.message, true);
  }
  S.chats.sort((a, b) => b.updatedAt - a.updatedAt);
  setMode(S.mode, true);
  renderList();
  renderCurrent();
  updateThink();
  $("input").focus();
}

function setBackend(b) {
  S.backend = b;
  const st = $("status");
  st.className = "status " + b.state;
  const label = b.state === "ready" ? "Ready" + (b.device === "cuda" ? " · GPU" : " · CPU") : b.message || b.state;
  $("status-text").textContent = label;
  st.title = b.message || "";
  $("status-restart").hidden = b.state !== "error" && b.state !== "stopped";
  updateComposer();
}

host.on("backend", setBackend);
host.on("warmup", (w) => {
  S.warming = w.running ? w.name : null;
  updateComposer();
});

function updateComposer() {
  const ready = S.backend.state === "ready";
  const busy = !!S.gen;
  const send = $("send");
  send.replaceChildren(icon(busy ? "stop" : "send"));
  send.title = busy ? "Stop (Esc)" : "Send (Enter)";
  send.disabled = !busy && (!ready || !$("input").value.trim());
  $("doc-go").disabled = !busy && (!ready || !$("doc").textContent.trim());
  $("doc-go").replaceChildren(document.createTextNode(busy ? "Stop" : "Continue"),
    Object.assign(el("kbd", null, busy ? "Esc" : "Ctrl+Enter")));
  const m = model();
  const story = storyOn();
  const who = chatCharacter(current()).name;
  $("composer-hint").textContent = !ready
    ? (S.backend.state === "error" ? "Model stopped" : "Model loading…")
    : story && S.warming && !busy ? `Getting ${S.warming} ready…`
    : story ? `Story chat · answers can be wrong`
    : m && !m.chatTuned ? "Not chat-tuned: replies may be incoherent" : "";
  // The pill names the character in story chat; thinking only applies to the model's own chat format.
  $("model-pill").replaceChildren(icon("chip"),
    document.createTextNode((story ? who + " · " : "") + (m ? m.displayName : "")));
  $("think").hidden = story;
}

function setMode(mode, keepChat) {
  S.mode = mode;
  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  const c = current();
  if (!keepChat && c && c.mode !== mode) S.currentId = null;
  $("scroll").hidden = mode !== "chat";
  $("composer").hidden = mode !== "chat";
  $("doc-wrap").hidden = mode !== "complete";
  $("complete-bar").hidden = mode !== "complete";
  if (mode === "chat" && storyOn() && S.backend.state === "ready") host.call("story.warm").catch(() => {});
  renderCurrent();
  renderList();
  (mode === "chat" ? $("input") : $("doc")).focus();
}

function newChat() {
  if (S.gen) return;
  S.currentId = null;
  renderCurrent();
  renderList();
  (S.mode === "chat" ? $("input") : $("doc")).focus();
}

function renderList() {
  const nav = $("chat-list");
  nav.innerHTML = "";
  let group = null;
  for (const c of S.chats) {
    const g = LR.dateGroup(c.updatedAt);
    if (g !== group) {
      nav.appendChild(el("div", "chat-group", g));
      group = g;
    }
    const b = el("button", "chat-item" + (c.id === S.currentId ? " on" : ""));
    b.appendChild(el("span", "t", c.title));
    if (c.mode === "complete") b.appendChild(el("span", "mode", "TEXT"));
    const del = el("span", "icon-btn del");
    del.title = "Delete";
    del.appendChild(icon("trash"));
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(c.id);
    });
    b.appendChild(del);
    b.addEventListener("click", () => openChat(c.id));
    nav.appendChild(b);
  }
}

function openChat(id) {
  if (S.gen) {
    toast("Wait for the current reply to finish, or stop it first.");
    return;
  }
  const c = S.chats.find((x) => x.id === id);
  if (!c) return;
  S.currentId = id;
  if (c.mode !== S.mode) setMode(c.mode, true);
  else {
    renderCurrent();
    renderList();
  }
}

function deleteChat(id) {
  if (S.gen && S.gen.chatId === id) return;
  S.chats = S.chats.filter((c) => c.id !== id);
  if (S.currentId === id) S.currentId = null;
  saveChats(true);
  renderList();
  renderCurrent();
}

let saveTimer = null;
function saveChats(now) {
  clearTimeout(saveTimer);
  const go = () => host.call("chats.save", { chats: S.chats }).catch((e) => toast("Could not save chats: " + e.message, true));
  if (now) go();
  else saveTimer = setTimeout(go, 400);
}

function ensureChat(mode, titleSource) {
  let c = current();
  if (!c) {
    c = { id: uid(), title: LR.chatTitle(titleSource), mode, createdAt: Date.now(), updatedAt: Date.now(), messages: [], doc: "" };
    if (mode === "chat" && storyOn()) c.character = S.settings.character;
    S.chats.unshift(c);
    S.currentId = c.id;
  }
  return c;
}

function touch(c) {
  c.updatedAt = Date.now();
  S.chats.sort((a, b) => b.updatedAt - a.updatedAt);
}

// ------------------------------------------------------------------ chat rendering

function renderCurrent() {
  const c = current();
  if (S.mode === "complete") {
    const doc = $("doc");
    doc.textContent = c && c.mode === "complete" ? c.doc || "" : "";
    $("complete-stats").textContent = c && c.lastStats ? LR.statsLine(c.lastStats) : "";
    updateComposer();
    return;
  }
  const thread = $("thread");
  thread.innerHTML = "";
  const msgs = c && c.mode === "chat" ? c.messages : [];
  $("empty").hidden = msgs.length > 0;
  const m = model();
  const story = storyOn();
  const who = chatCharacter(c).name;
  $("empty-title").textContent = story ? `Chat with ${who}` : "What's on your mind?";
  $("model-notice").hidden = !(m && !m.chatTuned);
  if (m && !m.chatTuned) {
    const n = $("model-notice");
    n.innerHTML = "";
    if (story) {
      n.appendChild(document.createTextNode(
        `${m.displayName} wasn't trained for chat, so LittleRun writes the conversation as a story in which ${who} answers. ` +
        "Replies usually make sense, but facts are often wrong. "));
      const a = el("a", null, "Change the character");
      a.addEventListener("click", () => { openSettings(); showTab("character"); });
      n.appendChild(a);
      n.appendChild(document.createTextNode("."));
    } else {
      n.appendChild(document.createTextNode(`${m.displayName} isn't trained for chat, so replies are often incoherent. `));
      const a = el("a", null, "Try Complete mode");
      a.addEventListener("click", () => setMode("complete"));
      n.appendChild(a);
      n.appendChild(document.createTextNode(", which continues text."));
    }
  }
  msgs.forEach((msg, i) => thread.appendChild(renderMessage(msg, i === msgs.length - 1)));
  scrollToEnd(true);
  updateComposer();
}

function renderMessage(msg, last) {
  const row = el("div", "msg " + msg.role);
  row._msg = msg;
  if (msg.role === "user") {
    row.appendChild(el("div", "bubble", msg.content));
    return row;
  }
  fillAssistant(row, msg, last);
  return row;
}

function fillAssistant(row, msg, last) {
  row.innerHTML = "";
  if (msg.who) row.appendChild(el("div", "who", msg.who));
  const { thinking, answer, inProgress } = LR.splitThinking(msg.content);
  if (thinking || inProgress) {
    const d = el("details", "think");
    if (inProgress) d.open = true;
    const s = el("summary");
    s.appendChild(icon("bulb"));
    s.appendChild(document.createTextNode(inProgress ? "Thinking…" : "Thought"));
    d.appendChild(s);
    d.appendChild(el("div", "think-body", thinking));
    row.appendChild(d);
  }
  const body = el("div", "body");
  body.innerHTML = LR.renderMarkdown(answer);
  if (msg.pending && !inProgress) body.classList.add("cursor");
  if (!answer && msg.pending && !inProgress) body.classList.add("cursor");
  row.appendChild(body);
  if (msg.error) row.appendChild(el("div", "err", msg.error));
  if (!msg.pending) {
    const foot = el("div", "msg-foot");
    const copy = el("button", "icon-btn");
    copy.title = "Copy";
    copy.appendChild(icon("copy"));
    copy.addEventListener("click", () => host.call("clipboard.copy", { text: answer || msg.content }).then(() => toast("Copied")));
    foot.appendChild(copy);
    if (last) {
      const retry = el("button", "icon-btn");
      retry.title = "Regenerate";
      retry.appendChild(icon("retry"));
      retry.addEventListener("click", regenerate);
      foot.appendChild(retry);
    }
    if (msg.stats) foot.appendChild(el("span", "stats", LR.statsLine(msg.stats)));
    row.appendChild(foot);
  }
}

let stick = true;
$("scroll").addEventListener("scroll", () => {
  const s = $("scroll");
  stick = s.scrollHeight - s.scrollTop - s.clientHeight < 80;
});
function scrollToEnd(force) {
  const s = $("scroll");
  if (force || stick) s.scrollTop = s.scrollHeight;
}

// ------------------------------------------------------------------ generation

function sendChat() {
  if (S.gen) return stop();
  const text = $("input").value.trim();
  if (!text || S.backend.state !== "ready") return;
  const c = ensureChat("chat", text);
  c.messages.push({ role: "user", content: text });
  $("input").value = "";
  autosize();
  generateReply(c);
}

function regenerate() {
  const c = current();
  if (!c || S.gen || S.backend.state !== "ready") return;
  if (c.messages.length && c.messages[c.messages.length - 1].role === "assistant") c.messages.pop();
  generateReply(c);
}

async function generateReply(c) {
  const history = LR.historyForModel(c.messages);
  const story = storyOn();
  if (story && !c.character) c.character = S.settings.character;
  const msg = { role: "assistant", content: "", pending: true };
  if (story) msg.who = chatCharacter(c).name;
  c.messages.push(msg);
  touch(c);
  renderList();
  renderCurrent();
  const reqId = uid();
  S.gen = { reqId, chatId: c.id, msg };
  updateComposer();
  try {
    const r = await host.call("gen.start", { reqId, mode: "chat", messages: history, thinking: S.thinking,
                                             character: story ? c.character : null });
    msg.stats = r;
  } catch (e) {
    msg.error = e.message;
  } finally {
    if (story) msg.content = LR.tidyStory(msg.content);
    if (!msg.error && !msg.content.trim() && !(msg.stats && msg.stats.stopped))
      msg.error = "The model gave an empty reply. Try again.";
    msg.pending = false;
    S.gen = null;
    touch(c);
    saveChats(true);
    if (S.currentId === c.id && S.mode === "chat") {
      const rows = $("thread").children;
      if (rows.length) fillAssistant(rows[rows.length - 1], msg, true);
    }
    updateComposer();
    scrollToEnd();
  }
}

let frame = 0;
host.on("gen.delta", (d) => {
  const g = S.gen;
  if (!g || g.reqId !== d.reqId) return;
  if (g.msg) {
    g.msg.content += d.text;
    if (!frame) {
      frame = requestAnimationFrame(() => {
        frame = 0;
        if (!S.gen || S.currentId !== S.gen.chatId || S.mode !== "chat") return;
        const rows = $("thread").children;
        if (rows.length) fillAssistant(rows[rows.length - 1], S.gen.msg, true);
        scrollToEnd();
      });
    }
  } else if (g.span) {
    g.span.textContent += d.text;
    g.text += d.text;
  }
});

function stop() {
  if (S.gen) host.call("gen.stop", { reqId: S.gen.reqId });
}

async function continueDoc() {
  if (S.gen) return stop();
  const doc = $("doc");
  const prompt = doc.textContent;
  if (!prompt.trim() || S.backend.state !== "ready") return;
  const c = ensureChat("complete", prompt.split("\n")[0]);
  c.doc = prompt;
  const span = el("span", "gen");
  doc.appendChild(span);
  const reqId = uid();
  S.gen = { reqId, chatId: c.id, span, text: "" };
  doc.contentEditable = "false";
  updateComposer();
  renderList();
  try {
    const r = await host.call("gen.start", { reqId, mode: "complete", prompt });
    c.lastStats = r;
    $("complete-stats").textContent = LR.statsLine(r);
  } catch (e) {
    toast(e.message, true);
  } finally {
    c.doc = prompt + S.gen.text;
    S.gen = null;
    doc.contentEditable = "plaintext-only";
    touch(c);
    saveChats(true);
    renderList();
    updateComposer();
    // Keep the highlight until the text is edited.
    doc.addEventListener("input", function unmark() {
      doc.querySelectorAll(".gen").forEach((s) => s.replaceWith(document.createTextNode(s.textContent)));
      doc.removeEventListener("input", unmark);
    });
  }
}

// ------------------------------------------------------------------ settings

function openSettings() {
  const f = $("settings-form");
  const s = S.settings;
  for (const name of ["systemPrompt", "temperature", "maxTokens", "topP", "topK", "repetitionPenalty", "maxContext", "port"])
    f.elements[name].value = s[name];
  f.elements.thinking.checked = !!s.thinking;
  f.elements.closeToTray.checked = !!s.closeToTray;
  f.elements.chatStyle.value = s.chatStyle || "auto";
  f.elements.userName.value = s.userName || "";
  fillCharacterTab();
  fillApiTab();
  const m = model();
  const rt = S.init.runtime || {};
  const facts = $("model-facts");
  facts.innerHTML = "";
  const fact = (k, v) => { facts.appendChild(el("dt", null, k)); facts.appendChild(el("dd", null, v || "–")); };
  fact("Model", m ? m.displayName : "");
  fact("Source", m ? `${m.repo} @ ${m.revision.slice(0, 7)} (${m.license})` : "");
  fact("Folder", s.modelDir);
  fact("Device", s.device === "cuda" ? "NVIDIA GPU" : "CPU");
  fact("Runtime", rt.torch ? `PyTorch ${rt.torch}, Python ${rt.python}` : "");
  fact("1-bit kernel", rt.kernelOk ? "Ready" : "Not available (slower path)");
  fact("Status", S.backend.message || S.backend.state);
  $("about").textContent = `LittleRun ${S.init.version}. Licensed under the PolyForm Noncommercial License 1.0.0. ` +
    `Model weights keep their own license (${m ? m.license.toUpperCase() : ""}).`;
  $("settings-msg").textContent = "";
  showTab("gen");
  $("settings").showModal();
}

function fillCharacterTab() {
  const s = S.settings;
  const list = $("char-list");
  list.innerHTML = "";
  const options = [...presets(), { id: "custom", name: "Custom", summary: "Write your own character" }];
  for (const p of options) {
    const label = el("label", "choice char-pick");
    const input = el("input");
    input.type = "radio";
    input.name = "character";
    input.value = p.id;
    input.checked = (s.character || "ada") === p.id;
    input.addEventListener("change", showCustom);
    const body = el("span", "choice-body");
    body.appendChild(el("strong", null, p.name));
    body.appendChild(el("span", "muted", p.summary));
    label.appendChild(input);
    label.appendChild(body);
    list.appendChild(label);
  }
  const cc = s.customCharacter || {};
  $("cc-name").value = cc.name && cc.name !== "Custom" ? cc.name : "";
  $("cc-self").value = cc.self || "";
  $("cc-scene").value = cc.scene || "";
  $("cc-examples").value = LR.formatExamples(cc.examples);
  const from = $("cc-from");
  from.innerHTML = "";
  from.appendChild(Object.assign(el("option", null, "Keep what is here"), { value: "" }));
  for (const p of presets()) from.appendChild(Object.assign(el("option", null, p.name), { value: p.id }));
  showCustom();
}

function showCustom() {
  const picked = document.querySelector("#char-list input:checked");
  $("custom-char").hidden = !picked || picked.value !== "custom";
}

function copyPresetToCustom() {
  const p = presets().find((x) => x.id === $("cc-from").value);
  if (!p) return;
  $("cc-name").value = p.name;
  $("cc-self").value = p.self;
  $("cc-scene").value = p.scene;
  $("cc-examples").value = LR.formatExamples(p.examples);
  $("cc-from").value = "";
}

function fillApiTab() {
  const port = S.backend.port || S.settings.port;
  $("api-url").value = `http://127.0.0.1:${port}/v1`;
  $("api-key").value = S.settings.apiKey;
  $("api-model").value = model() ? model().modelId : "";
}

function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.dataset.tab !== name));
}

function collectSettings() {
  const f = $("settings-form");
  const num = (n) => Number(f.elements[n].value);
  return {
    ...S.settings,
    systemPrompt: f.elements.systemPrompt.value,
    temperature: num("temperature"),
    maxTokens: num("maxTokens"),
    topP: num("topP"),
    topK: num("topK"),
    repetitionPenalty: num("repetitionPenalty"),
    maxContext: num("maxContext"),
    port: num("port"),
    thinking: f.elements.thinking.checked,
    closeToTray: f.elements.closeToTray.checked,
    chatStyle: f.elements.chatStyle.value,
    userName: f.elements.userName.value.trim() || "Sam",
    character: (document.querySelector("#char-list input:checked") || { value: "ada" }).value,
    customCharacter: {
      id: "custom",
      name: $("cc-name").value.trim() || "Custom",
      summary: "Your own character",
      self: $("cc-self").value.trim(),
      scene: $("cc-scene").value.trim(),
      examples: LR.parseExamples($("cc-examples").value),
    },
  };
}

async function saveSettings(e) {
  e.preventDefault();
  if (!$("settings-form").reportValidity()) return;
  const next = collectSettings();
  if (next.character === "custom" && !next.customCharacter.self) {
    showTab("character");
    $("settings-msg").textContent = "Describe your custom character in “Who they are” first.";
    return;
  }
  try {
    const r = await host.call("settings.save", next);
    S.settings = r.settings;
    S.thinking = !!r.settings.thinking;
    updateThink();
    $("settings").close();
    renderCurrent();
    toast(r.restarted ? "Saved. The model is restarting." : "Saved");
  } catch (err) {
    $("settings-msg").textContent = err.message;
  }
}

function updateThink() {
  $("think").classList.toggle("on", S.thinking);
}

// ------------------------------------------------------------------ input

function autosize() {
  const t = $("input");
  t.style.height = "auto";
  t.style.height = Math.min(t.scrollHeight, 240) + "px";
  updateComposer();
}

function wire() {
  $("pick-folder").addEventListener("click", pickFolder);
  document.querySelectorAll("input[name=source]").forEach((r) => r.addEventListener("change", updateInstallButton));
  $("accept").addEventListener("change", updateInstallButton);
  $("license-link").addEventListener("click", (e) => {
    e.preventDefault();
    const m = model();
    host.call("open.url", { url: m.licenseUrl || `https://huggingface.co/${m.repo}` });
  });
  $("install").addEventListener("click", runSetup);
  $("setup-cancel").addEventListener("click", () => {
    $("setup-cancel").disabled = true;
    host.call("setup.cancel");
  });
  $("setup-retry").addEventListener("click", () => showSetup(S.setupFromApp));
  $("setup-back").addEventListener("click", () => {
    $("setup").hidden = true;
    $("app").hidden = false;
  });
  $("open-logs-setup").addEventListener("click", () => host.call("open.logs"));

  $("new-chat").addEventListener("click", newChat);
  $("toggle-side").addEventListener("click", () => {
    $("app").classList.add("side-hidden");
    $("show-side").hidden = false;
  });
  $("show-side").addEventListener("click", () => {
    $("app").classList.remove("side-hidden");
    $("show-side").hidden = true;
  });
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    if (S.gen) return toast("Stop the current generation first.");
    if (b.dataset.mode !== S.mode) setMode(b.dataset.mode);
  }));
  $("status-restart").addEventListener("click", () => host.call("backend.restart"));
  $("think").addEventListener("click", () => {
    S.thinking = !S.thinking;
    updateThink();
  });

  const input = $("input");
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendChat();
    }
  });
  $("send").addEventListener("click", sendChat);
  $("doc-go").addEventListener("click", continueDoc);
  $("doc").addEventListener("input", () => {
    const c = current();
    if (c && c.mode === "complete" && !S.gen) {
      c.doc = $("doc").textContent;
      touch(c);
      saveChats();
    }
    updateComposer();
  });
  $("doc").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.ctrlKey) {
      e.preventDefault();
      continueDoc();
    }
  });
  $("doc-clear").addEventListener("click", () => {
    if (S.gen) return;
    $("doc").textContent = "";
    S.currentId = null;
    $("complete-stats").textContent = "";
    renderList();
    updateComposer();
    $("doc").focus();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && S.gen) {
      e.preventDefault();
      stop();
    } else if (e.key === "n" && e.ctrlKey && !$("app").hidden) {
      e.preventDefault();
      newChat();
    } else if (e.key === "," && e.ctrlKey && !$("app").hidden) {
      e.preventDefault();
      openSettings();
    }
  });
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-external]");
    if (a) {
      e.preventDefault();
      host.call("open.url", { url: a.href });
    }
  });

  $("open-settings").addEventListener("click", openSettings);
  $("cc-from").addEventListener("change", copyPresetToCustom);
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("settings-save").addEventListener("click", saveSettings);
  document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", () => {
    host.call("clipboard.copy", { text: $(b.dataset.copy).value }).then(() => toast("Copied"));
  }));
  $("new-key").addEventListener("click", async () => {
    const r = await host.call("settings.newKey", collectSettings());
    S.settings = r.settings;
    fillApiTab();
    $("settings-msg").textContent = "New key saved. The model is restarting.";
  });
  $("restart-model").addEventListener("click", () => {
    host.call("backend.restart");
    $("settings-msg").textContent = "Restarting the model…";
  });
  $("rerun-setup").addEventListener("click", () => {
    if (S.gen) return toast("Stop the current generation first.");
    $("settings").close();
    showSetup(true);
  });
  $("open-data").addEventListener("click", () => host.call("open.data"));
  $("open-logs").addEventListener("click", () => host.call("open.logs"));
}

wire();
boot();
