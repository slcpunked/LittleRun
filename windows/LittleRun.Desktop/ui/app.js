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
const iconBtn = (name, title, onClick, cls = "icon-btn") => {
  const b = el("button", cls);
  b.type = "button";
  b.title = title;
  b.appendChild(icon(name));
  b.addEventListener("click", onClick);
  return b;
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
  gen: null,          // { reqId, chatId, msg?, span?, text? }
  setupFromApp: false,
  warming: null,
  search: "",
  editing: null,      // { chatId, index }
  speaking: null,     // key of what is being read aloud
  listening: null,    // { base } while offline dictation runs
  renaming: null,     // chat id
};

const model = () => (S.init && S.init.models ? S.init.models[S.settings.modelName || S.init.defaultModel] : null);
const current = () => S.chats.find((c) => c.id === S.currentId) || null;
const storyOn = () => LR.useStory(S.settings, model());
const presets = () => (S.init && S.init.presets) || [];
const customReady = () => !!(S.settings.customCharacter && S.settings.customCharacter.self && S.settings.customCharacter.self.trim());
function characterFor(id) {
  if (id === "custom" && customReady()) return S.settings.customCharacter;
  return presets().find((p) => p.id === id) || presets()[0] || { id: "ada", name: "Ada" };
}
// A chat keeps the character it started with; a new chat uses the current setting.
const chatCharacter = (c) => characterFor((c && c.character) || S.settings.character);
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

function avatar(name, cls = "avatar") {
  const a = el("span", cls, (name || "?").trim().charAt(0).toUpperCase());
  let h = 0;
  for (const ch of name || "") h = (h * 31 + ch.charCodeAt(0)) % 360;
  a.style.setProperty("--av", `hsl(${h} 52% 52%)`);
  return a;
}

function toast(text, bad, action) {
  const t = el("div", "toast" + (bad ? " bad" : ""));
  t.appendChild(el("span", null, text));
  if (action) {
    const b = el("button", null, action.label);
    b.addEventListener("click", () => { action.run(); t.remove(); });
    t.appendChild(b);
  }
  $("toasts").appendChild(t);
  setTimeout(() => t.remove(), bad ? 6000 : action ? 5000 : 2600);
}

// ------------------------------------------------------------------ theme

const darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

function applyTheme() {
  const pref = (S.settings && S.settings.theme) || "system";
  const dark = pref === "dark" || (pref === "system" && darkQuery.matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  document.documentElement.style.fontSize = 15 * ((S.settings && S.settings.fontScale) || 1) + "px";
  const css = getComputedStyle(document.documentElement);
  host.call("window.theme", {
    dark,
    bg: css.getPropertyValue("--bg-side").trim(),
    border: css.getPropertyValue("--line-strong").trim(),
  }).catch(() => {});
}
darkQuery.addEventListener("change", applyTheme);

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
  setMaximized(!!init.maximized);
  $("boot").hidden = true;
  if (init.payloadError) showFatal(init.payloadError);
  else if (init.setUp) await showApp();
  else if (init.needsUpdate) await runUpdate();
  else showSetup(false);
}

// A new version with a changed backend: reinstall the runtime with the same choices, no questions asked.
async function runUpdate() {
  showSetup(false);
  $("setup-title").textContent = "Updating LittleRun";
  $("setup-sub").textContent = "Installing the new runtime. Your model, settings and chats are kept.";
  importDir = S.settings.modelDir;
  importOk = true;
  document.querySelector("input[name=source][value=import]").checked = true;
  document.querySelector(`input[name=device][value=${S.settings.device === "cuda" ? "cuda" : "cpu"}]`).checked = true;
  $("accept").checked = true;
  await runSetup();
}

function applyInit(init) {
  S.init = init;
  S.settings = init.settings;
  S.thinking = !!init.settings.thinking;
  applyTheme();
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

function setMaximized(max) {
  const b = $("win-max");
  b.replaceChildren(icon(max ? "restore" : "max"));
  b.title = max ? "Restore" : "Maximize";
}
host.on("window", (w) => setMaximized(w.maximized));

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
  $("setup-title").textContent = "Set up LittleRun";
  $("setup-sub").textContent = "Runs LittleBit sub-1-bit models on this PC. Everything stays on your computer.";
  $("app").hidden = true;
  $("toggle-side").hidden = true;
  $("status").hidden = true;
  $("setup").hidden = false;
  $("setup-form").hidden = false;
  $("setup-progress").hidden = true;
  renderSystem();
  renderModelCard();
  const dev = S.settings.device || "cpu";
  const radio = document.querySelector(`input[name=device][value=${dev}]`);
  radio.checked = !radio.disabled;
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
  add("Memory", `${LR.formatBytes(s.totalRamBytes)} (${LR.formatBytes(s.availableRamBytes)} free)`, s.availableRamBytes / 2 ** 30 < 5);
  const g = s.nvidiaGpu;
  add("Graphics", g ? `${g.name}, ${(g.memoryMiB / 1024).toFixed(0)} GB, driver ${g.driverVersion}` : "No NVIDIA GPU found");
  add("Disk free", LR.formatBytes(s.freeDiskBytes), s.freeDiskBytes < (m ? m.sizeBytes : 0) + 4 * 2 ** 30);

  const gpu = document.querySelector("input[name=device][value=cuda]");
  const minDriver = S.init.cudaMinDriver || 0;
  if (!g) {
    gpu.disabled = true;
    $("gpu-note").textContent = "Needs an NVIDIA graphics card.";
  } else if (g.driverVersion < minDriver) {
    gpu.disabled = true;
    $("gpu-note").textContent = `Needs NVIDIA driver ${minDriver} or newer (you have ${g.driverVersion}).`;
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
  for (const t of [LR.formatBytes(m.sizeBytes), m.repo, "revision " + m.revision.slice(0, 7), "License " + m.license.toUpperCase()])
    meta.appendChild(el("span", null, t));
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
  $("install").disabled = !($("accept").checked && (source === "download" || (importDir && importOk)));
}

const stepRow = (id) => $("steps").querySelector(`li[data-step="${id}"]`);

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
    li.appendChild(el("span", "ico"));
    li.appendChild(el("span", null, id === "model" && source === "import" ? "Model folder" : label));
    li.appendChild(el("span", "step-msg"));
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
    pre.textContent = (pre.textContent + e.message + "\n").split("\n").slice(-400).join("\n");
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
  const msg = li.querySelector(".step-msg");
  msg.textContent = e.state === "running" && e.bytes == null ? e.message : e.state === "done" ? e.message : "";
  msg.title = e.message;
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

// ------------------------------------------------------------------ app shell

async function showApp() {
  $("setup").hidden = true;
  $("app").hidden = false;
  $("toggle-side").hidden = false;
  $("status").hidden = false;
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
  $("status-text").textContent = b.state === "ready" ? "Ready · " + (b.device === "cuda" ? "GPU" : "CPU") : b.message || b.state;
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
  if (!S.settings) return;
  const ready = S.backend.state === "ready";
  const busy = !!S.gen;
  const send = $("send");
  send.replaceChildren(icon(busy ? "stop" : "send"));
  send.title = busy ? "Stop (Esc)" : "Send (Enter)";
  send.disabled = !busy && (!ready || !$("input").value.trim());
  const go = $("doc-go");
  go.textContent = busy ? "Stop" : "Continue";
  go.disabled = !busy && (!ready || !$("doc").textContent.trim());
  const m = model();
  const story = storyOn();
  const who = chatCharacter(current()).name;
  $("composer-hint").textContent = S.listening ? "Listening… press Esc to stop"
    : !ready ? (S.backend.state === "error" ? "Model stopped" : "Model loading…")
    : story && S.warming && !busy ? `Getting ${S.warming} ready…`
    : story ? "Story chat · answers can be wrong"
    : m && !m.chatTuned ? "Not chat-tuned: replies may be incoherent" : "";
  $("pill-avatar").replaceWith(Object.assign(avatar(story ? who : "LR", "avatar avatar-xs"), { id: "pill-avatar" }));
  $("pill-text").textContent = (story ? who + " · " : "") + (m ? m.displayName : "");
  $("char-pill").title = story ? "Character for new chats" : m ? m.displayName : "";
  $("think").hidden = story;
  $("mic").classList.toggle("on", !!S.listening);
  $("composer").classList.toggle("listening", !!S.listening);
  $("export-chat").hidden = !current();
}

function setMode(mode, keepChat) {
  S.mode = mode;
  document.querySelectorAll(".main-top .seg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  const c = current();
  if (!keepChat && c && c.mode !== mode) S.currentId = null;
  $("scroll").hidden = mode !== "chat";
  $("composer").hidden = mode !== "chat";
  $("doc-wrap").hidden = mode !== "complete";
  $("complete-bar").hidden = mode !== "complete";
  $("jump").hidden = true;
  if (mode === "chat" && storyOn() && S.backend.state === "ready") host.call("story.warm").catch(() => {});
  cancelEdit();
  renderCurrent();
  renderList();
  (mode === "chat" ? $("input") : $("doc")).focus();
}

function newChat() {
  if (S.gen) return;
  cancelEdit();
  S.currentId = null;
  renderCurrent();
  renderList();
  (S.mode === "chat" ? $("input") : $("doc")).focus();
}

function toggleSidebar() {
  $("app").classList.toggle("side-hidden");
}

// ------------------------------------------------------------------ sidebar

function renderList() {
  const nav = $("chat-list");
  nav.innerHTML = "";
  const list = LR.searchChats(S.chats, S.search);
  if (!list.length) {
    nav.appendChild(el("div", "chat-empty", S.search ? "No chats match your search." : "Your chats will appear here."));
    return;
  }
  let group = null;
  for (const c of list) {
    const g = LR.dateGroup(c.updatedAt);
    if (g !== group) {
      nav.appendChild(el("div", "chat-group", g));
      group = g;
    }
    const b = el("div", "chat-item" + (c.id === S.currentId ? " on" : ""));
    b.tabIndex = 0;
    b.title = c.title;
    if (S.renaming === c.id) {
      const inp = el("input", "t-input");
      inp.value = c.title;
      const done = (save) => {
        if (S.renaming !== c.id) return;
        S.renaming = null;
        if (save && inp.value.trim()) {
          c.title = inp.value.trim().slice(0, 120);
          saveChats(true);
        }
        renderList();
      };
      inp.addEventListener("keydown", (e) => {
        if (e.key === "Enter") done(true);
        if (e.key === "Escape") { e.stopPropagation(); done(false); }
      });
      inp.addEventListener("blur", () => done(true));
      b.appendChild(inp);
      setTimeout(() => { inp.focus(); inp.select(); }, 0);
    } else {
      b.appendChild(el("span", "t", c.title));
      if (c.mode === "complete") b.appendChild(el("span", "mode", "TEXT"));
      const acts = el("span", "acts");
      acts.appendChild(iconBtn("pencil", "Rename", (e) => { e.stopPropagation(); S.renaming = c.id; renderList(); }));
      acts.appendChild(iconBtn("download", "Export", (e) => { e.stopPropagation(); exportChat(c); }));
      acts.appendChild(iconBtn("trash", "Delete", (e) => { e.stopPropagation(); deleteChat(c.id); }));
      b.appendChild(acts);
      b.addEventListener("click", () => openChat(c.id));
      b.addEventListener("dblclick", () => { S.renaming = c.id; renderList(); });
      b.addEventListener("keydown", (e) => {
        if (e.key === "Enter") openChat(c.id);
        if (e.key === "F2") { S.renaming = c.id; renderList(); }
        if (e.key === "Delete") deleteChat(c.id);
      });
    }
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
  cancelEdit();
  S.currentId = id;
  if (c.mode !== S.mode) setMode(c.mode, true);
  else {
    renderCurrent();
    renderList();
  }
}

let lastDeleted = null;
function deleteChat(id) {
  if (S.gen && S.gen.chatId === id) return;
  const i = S.chats.findIndex((c) => c.id === id);
  if (i < 0) return;
  lastDeleted = { chat: S.chats[i] };
  S.chats.splice(i, 1);
  if (S.currentId === id) S.currentId = null;
  saveChats(true);
  renderList();
  renderCurrent();
  toast("Chat deleted.", false, {
    label: "Undo",
    run: () => {
      if (!lastDeleted) return;
      S.chats.push(lastDeleted.chat);
      S.chats.sort((a, b) => b.updatedAt - a.updatedAt);
      lastDeleted = null;
      saveChats(true);
      renderList();
    },
  });
}

async function exportChat(c) {
  c = c || current();
  if (!c) return;
  try {
    const r = await host.call("chat.export", { name: c.title, markdown: LR.chatToMarkdown(c, S.settings.userName) });
    if (r && r.saved) toast("Chat exported.", false, { label: "Show", run: () => host.call("open.file", { path: r.saved }) });
  } catch (e) {
    toast("Could not export: " + e.message, true);
  }
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
    $("doc").textContent = c && c.mode === "complete" ? c.doc || "" : "";
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
  $("empty-avatar").replaceWith(Object.assign(avatar(story ? who : "LittleRun", "avatar avatar-lg"), { id: "empty-avatar" }));
  $("empty-title").textContent = story ? `Chat with ${who}` : "What's on your mind?";
  const n = $("model-notice");
  n.hidden = !(m && !m.chatTuned);
  n.innerHTML = "";
  if (m && !m.chatTuned) {
    if (story) {
      n.appendChild(document.createTextNode(
        `${m.displayName} wasn't trained for chat, so LittleRun writes the conversation as a story in which ${who} answers. ` +
        "Replies usually make sense, but facts are often wrong. "));
      const a = el("a", null, "Choose a character");
      a.addEventListener("click", () => openSettings("character"));
      n.appendChild(a);
      n.appendChild(document.createTextNode("."));
    } else {
      n.appendChild(document.createTextNode(`${m.displayName} isn't trained for chat, so replies are often incoherent. `));
      const a = el("a", null, "Try Complete mode");
      a.addEventListener("click", () => setMode("complete"));
      n.appendChild(a);
      n.appendChild(document.createTextNode("."));
    }
  }
  msgs.forEach((msg, i) => thread.appendChild(renderMessage(c, msg, i, i === msgs.length - 1)));
  scrollToEnd(true);
  updateComposer();
}

const speakKey = (c, i) => `${c ? c.id : "x"}:${i}`;

function renderMessage(c, msg, index, last) {
  const row = el("div", "msg " + msg.role);
  if (msg.role === "user") {
    const b = el("div", "bubble", msg.content);
    if (msg.at) b.title = new Date(msg.at).toLocaleString();
    row.appendChild(b);
    const foot = el("div", "msg-foot");
    foot.appendChild(iconBtn("pencil", "Edit and resend", () => startEdit(c, index)));
    foot.appendChild(iconBtn("copy", "Copy", () => copy(msg.content)));
    row.appendChild(foot);
    return row;
  }
  fillAssistant(row, c, msg, index, last);
  return row;
}

function fillAssistant(row, c, msg, index, last) {
  row.innerHTML = "";
  const name = msg.who || (storyOn() ? chatCharacter(c).name : model() ? "LittleRun" : "AI");
  row.appendChild(avatar(name));
  const content = el("div", "content");
  row.appendChild(content);
  if (msg.who) content.appendChild(el("div", "who", msg.who));
  const { thinking, answer, inProgress } = LR.splitThinking(msg.content);
  if (thinking || inProgress) {
    const d = el("details", "think");
    if (inProgress) d.open = true;
    const s = el("summary");
    s.appendChild(icon("bulb"));
    s.appendChild(document.createTextNode(inProgress ? "Thinking…" : "Thought"));
    d.appendChild(s);
    d.appendChild(el("div", "think-body", thinking));
    content.appendChild(d);
  }
  const body = el("div", "body");
  body.innerHTML = LR.renderMarkdown(answer);
  for (const pre of body.querySelectorAll("pre")) {
    const head = el("div", "code-head");
    head.appendChild(el("span", null, pre.dataset.lang || "code"));
    const b = el("button");
    b.type = "button";
    b.appendChild(icon("copy"));
    b.appendChild(document.createTextNode("Copy"));
    b.addEventListener("click", () => copy(pre.querySelector("code").textContent));
    head.appendChild(b);
    pre.prepend(head);
  }
  if (msg.pending && !inProgress) body.classList.add("cursor");
  content.appendChild(body);
  if (msg.error) content.appendChild(el("div", "err", msg.error));
  if (!msg.pending) {
    const key = speakKey(c, index);
    const foot = el("div", "msg-foot" + (S.speaking === key ? " speaking" : ""));
    foot.appendChild(iconBtn("copy", "Copy", () => copy(answer || msg.content)));
    const sp = iconBtn("speaker", S.speaking === key ? "Stop reading" : "Read aloud",
                       () => (S.speaking === key ? stopSpeaking() : speak(key, msg.content)));
    if (S.speaking === key) sp.classList.add("on");
    foot.appendChild(sp);
    if (last) foot.appendChild(iconBtn("retry", "Regenerate", regenerate));
    if (msg.stats) {
      const st = el("span", "stats", LR.statsLine(msg.stats));
      if (msg.at) st.title = new Date(msg.at).toLocaleString();
      foot.appendChild(st);
    }
    content.appendChild(foot);
  }
}

function rerenderMessage(c, index) {
  if (!c || S.currentId !== c.id || S.mode !== "chat") return;
  const rows = $("thread").children;
  const msg = c.messages[index];
  if (rows[index] && msg) {
    if (msg.role === "assistant") fillAssistant(rows[index], c, msg, index, index === c.messages.length - 1);
  }
}

let stick = true;
$("scroll").addEventListener("scroll", () => {
  const s = $("scroll");
  const gap = s.scrollHeight - s.scrollTop - s.clientHeight;
  stick = gap < 80;
  $("jump").hidden = gap < 240;
});
function scrollToEnd(force) {
  const s = $("scroll");
  if (force || stick) {
    s.style.scrollBehavior = "auto";
    s.scrollTop = s.scrollHeight;
    s.style.scrollBehavior = "";
  }
}

async function copy(text) {
  try {
    await host.call("clipboard.copy", { text });
    toast("Copied");
  } catch (e) {
    toast("Could not copy: " + e.message, true);
  }
}

// ------------------------------------------------------------------ editing

function startEdit(c, index) {
  if (S.gen || !c) return;
  S.editing = { chatId: c.id, index };
  $("input").value = c.messages[index].content;
  $("edit-banner").hidden = false;
  autosize();
  $("input").focus();
  $("input").setSelectionRange($("input").value.length, $("input").value.length);
}

function cancelEdit() {
  if (!S.editing) return;
  S.editing = null;
  $("edit-banner").hidden = true;
  $("input").value = "";
  autosize();
}

// ------------------------------------------------------------------ generation

function sendChat() {
  if (S.gen) return stop();
  const text = $("input").value.trim();
  if (!text || S.backend.state !== "ready") return;
  if (S.listening) host.call("speech.stop").catch(() => {});
  let c;
  if (S.editing && S.editing.chatId === S.currentId && current()) {
    c = current();
    c.messages.splice(S.editing.index);
    S.editing = null;
    $("edit-banner").hidden = true;
  } else {
    cancelEdit();
    c = ensureChat("chat", text);
  }
  c.messages.push({ role: "user", content: text, at: Date.now() });
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
  stopSpeaking();
  const history = LR.historyForModel(c.messages);
  const story = storyOn();
  if (story && !c.character) c.character = S.settings.character;
  const msg = { role: "assistant", content: "", pending: true, at: Date.now() };
  if (story) msg.who = chatCharacter(c).name;
  c.messages.push(msg);
  const index = c.messages.length - 1;
  touch(c);
  renderList();
  renderCurrent();
  const reqId = uid();
  S.gen = { reqId, chatId: c.id, msg, index };
  updateComposer();
  try {
    msg.stats = await host.call("gen.start", { reqId, mode: "chat", messages: history, thinking: S.thinking,
                                               character: story ? c.character : null });
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
    rerenderMessage(c, index);
    updateComposer();
    scrollToEnd();
    if (S.settings.autoRead && !msg.error && msg.content.trim() && !(msg.stats && msg.stats.stopped))
      speak(speakKey(c, index), msg.content);
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
        if (!S.gen) return;
        rerenderMessage(current(), S.gen.index);
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
  stopSpeaking();
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
    doc.addEventListener("input", function unmark() {
      doc.querySelectorAll(".gen").forEach((s) => s.replaceWith(document.createTextNode(s.textContent)));
      doc.removeEventListener("input", unmark);
    });
  }
}

// ------------------------------------------------------------------ read aloud (Windows voices via WebView2)

function voices() {
  return window.speechSynthesis ? speechSynthesis.getVoices() : [];
}

function pickVoice() {
  const all = voices();
  return all.find((v) => v.name === S.settings.ttsVoice) ||
         all.find((v) => v.lang === navigator.language) ||
         all.find((v) => v.lang.startsWith("en")) || all[0] || null;
}

function speak(key, text) {
  if (!window.speechSynthesis) return toast("Read aloud is not available.", true);
  const words = key === "sample" ? text : LR.speakableText(text);
  if (!words) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(words);
  const v = pickVoice();
  if (v) { u.voice = v; u.lang = v.lang; }
  u.rate = S.settings.ttsRate || 1;
  u.onend = u.onerror = () => {
    if (S.speaking === key) {
      S.speaking = null;
      refreshSpeaking();
    }
  };
  S.speaking = key;
  refreshSpeaking();
  speechSynthesis.speak(u);
}

function stopSpeaking() {
  if (window.speechSynthesis) speechSynthesis.cancel();
  if (S.speaking) {
    S.speaking = null;
    refreshSpeaking();
  }
}

function refreshSpeaking() {
  const c = current();
  if (c && S.mode === "chat") c.messages.forEach((m, i) => { if (m.role === "assistant" && !m.pending) rerenderMessage(c, i); });
  $("doc-read").classList.toggle("on", S.speaking === "doc");
}

// ------------------------------------------------------------------ dictation (Windows speech recognizer)

async function toggleMic() {
  if (S.mode !== "chat") return;
  if (S.listening) {
    host.call("speech.stop").catch(() => {});
    return;
  }
  stopSpeaking();
  $("input").focus();
  try {
    const r = await host.call("speech.start");
    if (r && r.engine === "voicetyping") {
      toast("Windows Voice Typing is open. Speak, then press Send.");
    }
  } catch (e) {
    toast(e.message, true);
  }
}

host.on("stt", (e) => {
  const input = $("input");
  if (e.kind === "state") {
    if (e.text === "listening") {
      S.listening = { base: input.value };
    } else {
      const had = S.listening;
      S.listening = null;
      if (had) input.value = had.base;
      autosize();
      if (had && S.settings.sttAutoSend && input.value.trim()) sendChat();
    }
    updateComposer();
  } else if (e.kind === "partial" && S.listening) {
    input.value = LR.appendDictation(S.listening.base, e.text);
    autosize();
  } else if (e.kind === "final" && S.listening) {
    S.listening.base = LR.appendDictation(S.listening.base, e.text);
    input.value = S.listening.base;
    autosize();
  } else if (e.kind === "error") {
    S.listening = null;
    updateComposer();
    toast(e.text, true);
  }
});

// ------------------------------------------------------------------ character menu

function openCharMenu() {
  if (!storyOn()) return openSettings("model");
  const menu = $("char-menu");
  if (!menu.hidden) return closeMenu();
  menu.innerHTML = "";
  menu.appendChild(el("div", "menu-title", "Character for new chats"));
  const opts = [...presets()];
  if (customReady()) opts.push(S.settings.customCharacter);
  for (const p of opts) {
    const b = el("button");
    b.type = "button";
    b.appendChild(avatar(p.name, "avatar avatar-xs"));
    const t = el("span", "mt");
    t.appendChild(el("b", null, p.name));
    t.appendChild(el("small", null, p.id === "custom" ? "Your character" : p.summary));
    b.appendChild(t);
    if (S.settings.character === p.id) b.appendChild(Object.assign(icon("check"), { classList: "tick" }));
    b.addEventListener("click", () => chooseCharacter(p.id));
    menu.appendChild(b);
  }
  menu.appendChild(el("hr"));
  const edit = el("button");
  edit.type = "button";
  edit.appendChild(icon("pencil"));
  edit.appendChild(el("span", "mt", "Edit characters…"));
  edit.addEventListener("click", () => { closeMenu(); openSettings("character"); });
  menu.appendChild(edit);
  const pill = $("char-pill").getBoundingClientRect();
  const main = document.querySelector(".main").getBoundingClientRect();
  menu.hidden = false;
  menu.style.left = Math.max(8, pill.left - main.left) + "px";
  menu.style.top = pill.top - main.top - menu.offsetHeight - 8 + "px";
}

function closeMenu() {
  $("char-menu").hidden = true;
}

async function chooseCharacter(id) {
  closeMenu();
  if (id === S.settings.character) return;
  const r = await host.call("settings.save", { ...S.settings, character: id });
  S.settings = r.settings;
  const c = current();
  if (c && c.messages && c.messages.length) newChat();
  else renderCurrent();
  toast(`New chats now use ${characterFor(id).name}.`);
}

// ------------------------------------------------------------------ settings

const TITLES = { general: "General", appearance: "Appearance", character: "Character", gen: "Generation",
                 voice: "Voice", model: "Model", api: "Local API", about: "About" };

function openSettings(tab) {
  closeMenu();
  const f = $("settings-form");
  const s = S.settings;
  for (const name of ["systemPrompt", "temperature", "maxTokens", "topP", "topK", "repetitionPenalty", "maxContext", "port",
                      "userName", "chatStyle", "fontScale", "ttsRate", "sttEngine"])
    if (f.elements[name]) f.elements[name].value = s[name];
  for (const name of ["thinking", "closeToTray", "startWithWindows", "autoRead", "sttAutoSend"])
    f.elements[name].checked = !!s[name];
  document.querySelectorAll("#theme-seg button").forEach((b) => b.classList.toggle("on", b.dataset.value === s.theme));
  fillVoices();
  fillSttLanguages();
  fillCharacterTab();
  fillApiTab();
  fillModelTab();
  $("data-path").textContent = S.init.paths.root;
  $("about-version").textContent = S.init.version;
  const m = model();
  $("about").textContent = `Runs LittleBit sub-1-bit models on this PC. Licensed under the PolyForm Noncommercial License 1.0.0. ` +
    `Model weights keep their own license (${m ? m.license.toUpperCase() : ""}).`;
  document.querySelectorAll(".slider").forEach(paintSlider);
  $("settings-msg").textContent = "";
  showTab(tab || "general");
  if (!$("settings").open) $("settings").showModal();
}

function showTab(name) {
  document.querySelectorAll("#settings-nav button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll("#settings-form .tab").forEach((t) => (t.hidden = t.dataset.tab !== name));
  $("settings-title").textContent = TITLES[name] || "";
}

function paintSlider(r) {
  const pct = ((r.value - r.min) / (r.max - r.min)) * 100;
  r.style.setProperty("--fill", pct + "%");
  const out = r.parentElement.querySelector("output");
  const v = Number(r.value);
  if (out) out.textContent = r.name === "ttsRate" ? v.toFixed(2) + "×" : v.toFixed(2);
  if (r.name === "fontScale") $("font-scale-label").textContent = Math.round(v * 100) + "%";
}

function fillVoices() {
  const sel = $("tts-voice");
  sel.innerHTML = "";
  sel.appendChild(Object.assign(el("option", null, "Automatic"), { value: "" }));
  for (const v of voices()) sel.appendChild(Object.assign(el("option", null, `${v.name.replace(/^Microsoft /, "")}`), { value: v.name }));
  sel.value = S.settings.ttsVoice || "";
}
if (window.speechSynthesis) speechSynthesis.addEventListener("voiceschanged", () => { if ($("settings").open) fillVoices(); });

function fillSttLanguages() {
  const sel = $("stt-lang");
  sel.innerHTML = "";
  const langs = S.init.speechLanguages || [];
  for (const l of langs) {
    let label = l;
    try { label = new Intl.DisplayNames([navigator.language], { type: "language" }).of(l) + ` (${l})`; } catch {}
    sel.appendChild(Object.assign(el("option", null, label), { value: l }));
  }
  sel.value = S.settings.sttLanguage;
  const offline = $("settings-form").elements.sttEngine.value === "offline";
  $("stt-lang-row").hidden = !offline;
  $("stt-note").textContent = offline
    ? (langs.length ? "Runs on this PC. Accuracy is basic; Voice Typing is usually better." : "No Windows speech language is installed.")
    : "Opens the Windows dictation bar (needs online speech recognition in Windows privacy settings).";
}

function fillCharacterTab() {
  const s = S.settings;
  const grid = $("char-list");
  grid.innerHTML = "";
  for (const p of [...presets(), { id: "custom", name: s.customCharacter && s.customCharacter.name !== "Custom" ? s.customCharacter.name : "Custom",
                                   summary: "Write your own character" }]) {
    const card = el("button", "char-card" + ((s.character || "ada") === p.id ? " on" : ""));
    card.type = "button";
    card.dataset.id = p.id;
    card.appendChild(avatar(p.id === "custom" && p.name === "Custom" ? "+" : p.name));
    card.appendChild(el("b", null, p.name));
    card.appendChild(el("small", null, p.summary));
    card.appendChild(Object.assign(icon("check"), { classList: "tick" }));
    card.addEventListener("click", () => {
      grid.querySelectorAll(".char-card").forEach((x) => x.classList.toggle("on", x === card));
      $("custom-char").hidden = p.id !== "custom";
      scheduleSave();
    });
    grid.appendChild(card);
  }
  const cc = s.customCharacter || {};
  $("cc-name").value = cc.name && cc.name !== "Custom" ? cc.name : "";
  $("cc-self").value = cc.self || "";
  $("cc-scene").value = cc.scene || "";
  $("cc-examples").value = LR.formatExamples(cc.examples);
  const from = $("cc-from");
  from.innerHTML = "";
  from.appendChild(Object.assign(el("option", null, "Keep what is here"), { value: "" }));
  for (const p of presets()) from.appendChild(Object.assign(el("option", null, "Copy " + p.name), { value: p.id }));
  $("custom-char").hidden = (s.character || "ada") !== "custom";
}

function copyPresetToCustom() {
  const p = presets().find((x) => x.id === $("cc-from").value);
  if (!p) return;
  $("cc-name").value = p.name + " (copy)";
  $("cc-self").value = p.self;
  $("cc-scene").value = p.scene;
  $("cc-examples").value = LR.formatExamples(p.examples);
  $("cc-from").value = "";
  scheduleSave();
}

function fillApiTab() {
  $("api-url").value = `http://127.0.0.1:${S.backend.port || S.settings.port}/v1`;
  $("api-key").value = S.settings.apiKey;
  $("api-model").value = model() ? model().modelId : "";
}

function fillModelTab() {
  const s = S.settings;
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
}

function collectSettings() {
  const f = $("settings-form");
  const num = (n) => Number(f.elements[n].value);
  const picked = document.querySelector("#char-list .char-card.on");
  const custom = {
    id: "custom",
    name: $("cc-name").value.trim() || "Custom",
    summary: "Your own character",
    self: $("cc-self").value.trim(),
    scene: $("cc-scene").value.trim(),
    examples: LR.parseExamples($("cc-examples").value),
  };
  let character = picked ? picked.dataset.id : S.settings.character;
  if (character === "custom" && !custom.self) character = S.settings.character === "custom" ? "ada" : S.settings.character;
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
    startWithWindows: f.elements.startWithWindows.checked,
    chatStyle: f.elements.chatStyle.value,
    userName: f.elements.userName.value.trim() || "Sam",
    character,
    customCharacter: custom,
    theme: (document.querySelector("#theme-seg button.on") || { dataset: { value: "system" } }).dataset.value,
    fontScale: num("fontScale"),
    autoRead: f.elements.autoRead.checked,
    ttsVoice: f.elements.ttsVoice.value,
    ttsRate: num("ttsRate"),
    sttEngine: f.elements.sttEngine.value,
    sttLanguage: f.elements.sttLanguage.value || S.settings.sttLanguage,
    sttAutoSend: f.elements.sttAutoSend.checked,
  };
}

// Settings apply as you change them, like Windows Settings.
let settingsTimer = null;
let savedTimer = null;
function scheduleSave(delay = 450) {
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(saveSettingsNow, delay);
}

async function saveSettingsNow() {
  clearTimeout(settingsTimer);
  const f = $("settings-form");
  if (!f.checkValidity()) return;
  const next = collectSettings();
  try {
    const r = await host.call("settings.save", next);
    S.settings = r.settings;
    S.thinking = !!r.settings.thinking;
    applyTheme();
    updateThink();
    renderCurrent();
    const msg = $("settings-msg");
    msg.textContent = r.restarted ? "Saved · model restarting" : "Saved";
    msg.style.opacity = 1;
    clearTimeout(savedTimer);
    savedTimer = setTimeout(() => (msg.style.opacity = 0), 1600);
    if (r.restarted) fillApiTab();
    const picked = document.querySelector("#char-list .char-card.on");
    if (picked && picked.dataset.id === "custom" && !next.customCharacter.self)
      msg.textContent = "Describe your character in “Who they are” to use it.";
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

function anyDialogOpen() {
  return $("settings").open || $("shortcuts").open;
}

function wire() {
  // window
  $("win-min").addEventListener("click", () => host.call("window.minimize"));
  $("win-max").addEventListener("click", () => host.call("window.maximize").then((r) => setMaximized(r.maximized)));
  $("win-close").addEventListener("click", () => host.call("window.close"));
  $("toggle-side").addEventListener("click", toggleSidebar);

  // setup
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
    $("toggle-side").hidden = false;
    $("status").hidden = false;
  });
  $("open-logs-setup").addEventListener("click", () => host.call("open.logs"));

  // app
  $("new-chat").addEventListener("click", newChat);
  $("search").addEventListener("input", (e) => {
    S.search = e.target.value;
    renderList();
  });
  $("search").addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.stopPropagation(); e.target.value = ""; S.search = ""; renderList(); $("input").focus(); }
  });
  document.querySelectorAll(".main-top .seg button").forEach((b) => b.addEventListener("click", () => {
    if (S.gen) return toast("Stop the current generation first.");
    if (b.dataset.mode !== S.mode) setMode(b.dataset.mode);
  }));
  $("status-restart").addEventListener("click", () => host.call("backend.restart"));
  $("think").addEventListener("click", () => {
    S.thinking = !S.thinking;
    updateThink();
  });
  $("char-pill").addEventListener("click", (e) => { e.stopPropagation(); openCharMenu(); });
  document.addEventListener("click", (e) => { if (!e.target.closest("#char-menu")) closeMenu(); });
  $("mic").addEventListener("click", toggleMic);
  $("jump").addEventListener("click", () => { stick = true; $("scroll").scrollTo({ top: $("scroll").scrollHeight, behavior: "smooth" }); });
  $("export-chat").addEventListener("click", () => exportChat());
  $("edit-cancel").addEventListener("click", cancelEdit);
  $("open-shortcuts").addEventListener("click", () => $("shortcuts").showModal());
  $("shortcuts-close").addEventListener("click", () => $("shortcuts").close());

  const input = $("input");
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendChat();
    } else if (e.key === "ArrowUp" && !input.value && !S.gen) {
      const c = current();
      if (c) {
        const i = c.messages.map((m) => m.role).lastIndexOf("user");
        if (i >= 0) { e.preventDefault(); startEdit(c, i); }
      }
    }
  });
  $("send").addEventListener("click", sendChat);
  $("doc-go").addEventListener("click", continueDoc);
  $("doc-read").addEventListener("click", () => (S.speaking === "doc" ? stopSpeaking() : speak("doc", $("doc").textContent)));
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

  // keyboard
  document.addEventListener("keydown", (e) => {
    const inApp = !$("app").hidden;
    const k = e.key.toLowerCase();
    if (e.key === "Escape") {
      if (!$("char-menu").hidden) return closeMenu();
      if (anyDialogOpen()) return;
      if (S.gen) { e.preventDefault(); return stop(); }
      if (S.listening) { e.preventDefault(); return host.call("speech.stop"); }
      if (S.speaking) { e.preventDefault(); return stopSpeaking(); }
      if (S.editing) { e.preventDefault(); return cancelEdit(); }
      return;
    }
    if (!e.ctrlKey || !inApp) return;
    if (k === "n") { e.preventDefault(); if (!anyDialogOpen()) newChat(); }
    else if (k === "k") { e.preventDefault(); $("app").classList.remove("side-hidden"); $("search").focus(); }
    else if (k === "b") { e.preventDefault(); toggleSidebar(); }
    else if (k === "d") { e.preventDefault(); toggleMic(); }
    else if (k === "s") { e.preventDefault(); exportChat(); }
    else if (k === ",") { e.preventDefault(); openSettings(); }
    else if (k === "/") { e.preventDefault(); $("shortcuts").open ? $("shortcuts").close() : $("shortcuts").showModal(); }
  });
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-external]");
    if (a) {
      e.preventDefault();
      host.call("open.url", { url: a.href });
    }
  });

  // settings
  $("open-settings").addEventListener("click", () => openSettings());
  $("settings-close").addEventListener("click", () => { saveSettingsNow(); $("settings").close(); });
  $("settings").addEventListener("close", () => { saveSettingsNow(); $("input").focus(); });
  $("settings").addEventListener("click", (e) => { if (e.target === $("settings")) $("settings").close(); });
  $("shortcuts").addEventListener("click", (e) => { if (e.target === $("shortcuts")) $("shortcuts").close(); });
  document.querySelectorAll("#settings-nav button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  const form = $("settings-form");
  form.addEventListener("submit", (e) => e.preventDefault());
  form.addEventListener("input", (e) => {
    const t = e.target;
    if (t.classList.contains("slider")) paintSlider(t);
    if (t.dataset.apply === "restart") return; // applied on change, not on every keystroke
    if (t.name === "fontScale") {
      S.settings.fontScale = Number(t.value);
      applyTheme();
    }
    scheduleSave(t.tagName === "TEXTAREA" || t.type === "text" ? 700 : 300);
  });
  form.addEventListener("change", (e) => {
    if (e.target.name === "sttEngine") fillSttLanguages();
    if (e.target.dataset.apply === "restart") saveSettingsNow();
  });
  document.querySelectorAll("#theme-seg button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#theme-seg button").forEach((x) => x.classList.toggle("on", x === b));
    S.settings.theme = b.dataset.value;
    applyTheme();
    saveSettingsNow();
  }));
  $("cc-from").addEventListener("change", copyPresetToCustom);
  document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", () => copy($(b.dataset.copy).value)));
  $("new-key").addEventListener("click", async () => {
    const r = await host.call("settings.newKey", collectSettings());
    S.settings = r.settings;
    fillApiTab();
    $("settings-msg").textContent = "New key saved · model restarting";
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
  $("tts-test").addEventListener("click", () => {
    const f = $("settings-form");
    S.settings.ttsVoice = f.elements.ttsVoice.value;
    S.settings.ttsRate = Number(f.elements.ttsRate.value);
    speak("sample", "Hello! This is how replies will sound when LittleRun reads them aloud.");
  });
  $("open-data").addEventListener("click", () => host.call("open.data"));
  $("open-logs").addEventListener("click", () => host.call("open.logs"));
  $("about-shortcuts").addEventListener("click", () => { $("settings").close(); $("shortcuts").showModal(); });
}

wire();
boot();
