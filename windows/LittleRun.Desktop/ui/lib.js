// Pure helpers for the LittleRun UI. No DOM access, so they run under `node --test` too.
(function (root, factory) {
  const lib = factory();
  if (typeof module === "object" && module.exports) module.exports = lib;
  else root.LR = lib;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Inline markdown on already-escaped text: code, bold, italic, links (http/https only).
  function inline(escaped) {
    const codes = [];
    let s = escaped.replace(/`([^`\n]+)`/g, (_, c) => {
      codes.push(c);
      return "\u0000" + (codes.length - 1) + "\u0000";
    });
    s = s
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/(^|[^_\w])_([^_\n]+)_(?![_\w])/g, "$1<em>$2</em>")
      .replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g,
        (_, t, u) => `<a href="${u}" data-external="1">${t}</a>`);
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[Number(i)]}</code>`);
  }

  // Small, safe markdown renderer: everything is escaped first, then a fixed set of tags is added.
  function renderMarkdown(md) {
    const lines = String(md).replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;
    let para = [];
    const flush = () => {
      if (para.length) {
        out.push("<p>" + para.map((l) => inline(escapeHtml(l))).join("<br>") + "</p>");
        para = [];
      }
    };
    while (i < lines.length) {
      const line = lines[i];
      const fence = line.match(/^\s*```\s*([\w+#.-]*)\s*$/);
      if (fence) {
        flush();
        const body = [];
        i++;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
        i++; // closing fence (or end of a still-streaming block)
        const lang = fence[1] ? ` data-lang="${escapeHtml(fence[1])}"` : "";
        out.push(`<pre${lang}><code>${escapeHtml(body.join("\n"))}</code></pre>`);
        continue;
      }
      const h = line.match(/^(#{1,6})\s+(.*)$/);
      if (h) {
        flush();
        const n = Math.min(h[1].length + 1, 6); // h1 in a reply looks too loud
        out.push(`<h${n}>${inline(escapeHtml(h[2]))}</h${n}>`);
        i++;
        continue;
      }
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        flush();
        out.push("<hr>");
        i++;
        continue;
      }
      if (/^\s*>/.test(line)) {
        flush();
        const q = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push("<blockquote>" + renderMarkdown(q.join("\n")) + "</blockquote>");
        continue;
      }
      const li = line.match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
      if (li) {
        flush();
        const ordered = /\d/.test(li[1]);
        const items = [];
        while (i < lines.length) {
          const m = lines[i].match(/^\s*([-*+]|\d+[.)])\s+(.*)$/);
          if (!m || /\d/.test(m[1]) !== ordered) break;
          items.push("<li>" + inline(escapeHtml(m[2])) + "</li>");
          i++;
        }
        out.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
        continue;
      }
      if (!line.trim()) {
        flush();
        i++;
        continue;
      }
      para.push(line);
      i++;
    }
    flush();
    return out.join("");
  }

  // Qwen3 thinking output: "<think>…</think>answer". The opening tag may be part of the prompt instead.
  function splitThinking(text) {
    const s = String(text);
    const open = s.indexOf("<think>");
    const close = s.indexOf("</think>");
    if (open === -1 && close === -1) return { thinking: "", answer: s, inProgress: false };
    if (close === -1) return { thinking: s.slice(open + 7).trim(), answer: "", inProgress: true };
    const start = open !== -1 && open < close ? open + 7 : 0;
    return { thinking: s.slice(start, close).trim(), answer: s.slice(close + 8).replace(/^\s+/, ""), inProgress: false };
  }

  function chatTitle(text, max = 48) {
    const t = String(text).replace(/\s+/g, " ").trim();
    if (!t) return "New chat";
    return t.length <= max ? t : t.slice(0, max - 1).replace(/\s+\S*$/, "") + "…";
  }

  function formatBytes(n) {
    if (n == null || !isFinite(n)) return "";
    const u = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    let v = n;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return (i === 0 ? v : v.toFixed(v >= 100 ? 0 : 1)) + " " + u[i];
  }

  function formatDuration(sec) {
    if (sec == null || !isFinite(sec) || sec < 0) return "";
    if (sec < 60) return Math.round(sec) + "s";
    const m = Math.floor(sec / 60);
    if (m < 60) return m + "m " + Math.round(sec % 60) + "s";
    return Math.floor(m / 60) + "h " + (m % 60) + "m";
  }

  // Sidebar grouping label for a timestamp (ms).
  function dateGroup(ts, now = Date.now()) {
    const d = new Date(ts);
    const n = new Date(now);
    const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const diff = Math.round((day(n) - day(d)) / 86400000);
    if (diff <= 0) return "Today";
    if (diff === 1) return "Yesterday";
    if (diff < 7) return "Previous 7 days";
    if (diff < 30) return "Previous 30 days";
    return "Older";
  }

  // Messages to send: assistant turns without their thinking (the Qwen3 convention).
  function historyForModel(messages) {
    return messages
      .filter((m) => (m.role === "user" || m.role === "assistant") && !m.error)
      .map((m) => ({ role: m.role, content: m.role === "assistant" ? splitThinking(m.content).answer : m.content }));
  }

  function statsLine(r) {
    if (!r || r.stopped) return r && r.stopped ? "Stopped" : "";
    const parts = [];
    if (r.completionTokens) parts.push(r.completionTokens + " tokens");
    const decode = r.firstTokenSeconds != null ? r.seconds - r.firstTokenSeconds : r.seconds;
    if (r.completionTokens > 1 && decode > 0) parts.push(((r.completionTokens - 1) / decode).toFixed(1) + " tok/s");
    if (r.firstTokenSeconds != null) parts.push("first token " + r.firstTokenSeconds.toFixed(1) + "s");
    if (r.finishReason === "length") parts.push("hit the token limit");
    return parts.join(" · ");
  }

  // Story chat: used for models that were not trained on chat (or when chosen in settings).
  function useStory(settings, model) {
    if (!settings || !model) return false;
    return settings.chatStyle === "story" || (settings.chatStyle !== "template" && !model.chatTuned);
  }

  // A story reply often ends with a comma before the closing quote ("It's winter," she said), and when the model
  // forgets to close the quote it rambles on; keep the first three sentences.
  function tidyStory(text, maxSentences = 3) {
    const parts = String(text).trim().split(/(?<=[.!?])\s+/);
    const t = parts.slice(0, maxSentences).join(" ").trim();
    return /[,;:]$/.test(t) ? t.slice(0, -1) + "." : t;
  }

  // Example exchanges as edited in settings:  "Q: question" / "A: answer" lines, pairs in order.
  function parseExamples(text) {
    const out = [];
    let q = null;
    for (const raw of String(text).split(/\r?\n/)) {
      const line = raw.trim();
      const m = line.match(/^([QA])\s*[:.)-]\s*(.*)$/i);
      if (!m) continue;
      if (m[1].toUpperCase() === "Q") q = m[2].trim();
      else if (q && m[2].trim()) {
        out.push({ q, a: m[2].trim() });
        q = null;
      }
    }
    return out;
  }

  function formatExamples(list) {
    return (list || []).map((e) => `Q: ${e.q}\nA: ${e.a}`).join("\n\n");
  }

  // Plain text to read aloud: no thinking, no code blocks, no markdown symbols.
  function speakableText(md) {
    return splitThinking(md).answer
      .replace(/```[\s\S]*?(```|$)/g, " (code) ")
      .replace(/`([^`]*)`/g, "$1")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/^\s{0,3}(#{1,6}|>|[-*+]|\d+[.)])\s+/gm, "")
      .replace(/(\*\*|__|\*|_|~~)(?=\S)([^*_~]+?)\1/g, "$2")
      .replace(/\s+/g, " ")
      .trim();
  }

  // A chat as a Markdown document for export.
  function chatToMarkdown(chat, userName) {
    const when = new Date(chat.createdAt || Date.now());
    const lines = [`# ${chat.title || "Chat"}`, "", `_${when.toLocaleString()}_`, ""];
    if (chat.mode === "complete") {
      lines.push(chat.doc || "");
    } else {
      for (const m of chat.messages || []) {
        if (m.error && !m.content) continue;
        const who = m.role === "user" ? userName || "You" : m.who || "Assistant";
        lines.push(`**${who}:**`, "", m.role === "assistant" ? splitThinking(m.content).answer : m.content, "");
      }
    }
    return lines.join("\n").trimEnd() + "\n";
  }

  // Chats whose title or text contains every word of the query (case-insensitive).
  function searchChats(chats, query) {
    const words = String(query).toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return chats;
    return chats.filter((c) => {
      const hay = [c.title, c.doc, ...(c.messages || []).map((m) => m.content)].join("\n").toLowerCase();
      return words.every((w) => hay.includes(w));
    });
  }

  // Merge dictated text into what is already typed, with sensible spacing and a capital at sentence starts.
  function appendDictation(base, phrase) {
    const p = String(phrase).trim();
    if (!p) return base;
    const b = String(base);
    const startsSentence = !b.trim() || /[.!?]\s*$/.test(b);
    const text = startsSentence ? p.charAt(0).toUpperCase() + p.slice(1) : p;
    return !b ? text : b + (/\s$/.test(b) ? "" : " ") + text;
  }

  return { escapeHtml, renderMarkdown, splitThinking, chatTitle, formatBytes, formatDuration, dateGroup,
           historyForModel, statsLine, useStory, tidyStory, parseExamples, formatExamples, speakableText,
           chatToMarkdown, searchChats, appendDictation };
});
