// Run: node --test windows/LittleRun.Desktop/ui/tests
const test = require("node:test");
const assert = require("node:assert/strict");
const LR = require("../lib.js");

test("markdown escapes HTML before adding tags", () => {
  const html = LR.renderMarkdown('<img src=x onerror="alert(1)"> **bold**');
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;img"));
  assert.ok(html.includes("<strong>bold</strong>"));
});

test("markdown renders fenced code without touching its content", () => {
  const html = LR.renderMarkdown("Before\n```python\nx = 1 * 2 * 3\n<b>\n```\nAfter");
  assert.match(html, /<pre data-lang="python"><code>x = 1 \* 2 \* 3\n&lt;b&gt;<\/code><\/pre>/);
  assert.match(html, /<p>Before<\/p>/);
  assert.match(html, /<p>After<\/p>/);
});

test("an unclosed fence (still streaming) renders as code", () => {
  assert.match(LR.renderMarkdown("```js\nlet a"), /<pre data-lang="js"><code>let a<\/code><\/pre>/);
});

test("markdown lists, headings, inline code and quotes", () => {
  const html = LR.renderMarkdown("# Title\n- one\n- `two`\n\n1. a\n2. b\n\n> quoted");
  assert.match(html, /<h2>Title<\/h2>/);
  assert.match(html, /<ul><li>one<\/li><li><code>two<\/code><\/li><\/ul>/);
  assert.match(html, /<ol><li>a<\/li><li>b<\/li><\/ol>/);
  assert.match(html, /<blockquote><p>quoted<\/p><\/blockquote>/);
});

test("only http(s) links become anchors", () => {
  assert.match(LR.renderMarkdown("[site](https://example.com)"), /<a href="https:\/\/example.com" data-external="1">site<\/a>/);
  assert.ok(!LR.renderMarkdown("[x](javascript:alert(1))").includes("<a "));
});

test("bold and italic do not break code spans or snake_case", () => {
  const html = LR.renderMarkdown("use `a*b*c` and my_var_name");
  assert.ok(html.includes("<code>a*b*c</code>"));
  assert.ok(html.includes("my_var_name"));
  assert.ok(!html.includes("<em>"));
});

test("splitThinking handles complete, streaming and prompt-opened think blocks", () => {
  assert.deepEqual(LR.splitThinking("<think>\nhmm\n</think>\n\nAnswer"), { thinking: "hmm", answer: "Answer", inProgress: false });
  assert.deepEqual(LR.splitThinking("<think>still going"), { thinking: "still going", answer: "", inProgress: true });
  assert.deepEqual(LR.splitThinking("reasoning</think>Done"), { thinking: "reasoning", answer: "Done", inProgress: false });
  assert.deepEqual(LR.splitThinking("plain"), { thinking: "", answer: "plain", inProgress: false });
});

test("history drops thinking and failed turns", () => {
  const h = LR.historyForModel([
    { role: "user", content: "hi" },
    { role: "assistant", content: "<think>x</think>Hello" },
    { role: "user", content: "again" },
    { role: "assistant", content: "", error: "boom" },
  ]);
  assert.deepEqual(h, [
    { role: "user", content: "hi" },
    { role: "assistant", content: "Hello" },
    { role: "user", content: "again" },
  ]);
});

test("titles, sizes, durations and date groups", () => {
  assert.equal(LR.chatTitle("   "), "New chat");
  assert.equal(LR.chatTitle("short question"), "short question");
  assert.ok(LR.chatTitle("word ".repeat(40)).endsWith("…"));
  assert.equal(LR.formatBytes(2831300000), "2.6 GB");
  assert.equal(LR.formatBytes(512), "512 B");
  assert.equal(LR.formatDuration(75), "1m 15s");
  const now = new Date(2026, 10, 20, 12).getTime();
  assert.equal(LR.dateGroup(now - 3600e3, now), "Today");
  assert.equal(LR.dateGroup(now - 86400e3, now), "Yesterday");
  assert.equal(LR.dateGroup(now - 3 * 86400e3, now), "Previous 7 days");
  assert.equal(LR.dateGroup(now - 90 * 86400e3, now), "Older");
});

test("story chat is used for models without chat training unless the template is chosen", () => {
  const base = { chatTuned: false };
  assert.equal(LR.useStory({ chatStyle: "auto" }, base), true);
  assert.equal(LR.useStory({ chatStyle: "template" }, base), false);
  assert.equal(LR.useStory({ chatStyle: "auto" }, { chatTuned: true }), false);
  assert.equal(LR.useStory({ chatStyle: "story" }, { chatTuned: true }), true);
});

test("story replies lose a trailing speech comma", () => {
  assert.equal(LR.tidyStory(" It's winter, "), "It's winter.");
  assert.equal(LR.tidyStory("Spring!"), "Spring!");
  assert.equal(LR.tidyStory("One. Two! Three? Four. Five."), "One. Two! Three?");
  assert.equal(LR.tidyStory("It's 3.5 miles. Easy walk."), "It's 3.5 miles. Easy walk.");
});

test("example exchanges parse from Q:/A: lines and round-trip", () => {
  const text = "Q: How are you?\nA: Fine, thanks.\n\nq) Favorite food?\na - Soup.\nA: orphan answer\nQ: no answer";
  const list = LR.parseExamples(text);
  assert.deepEqual(list, [{ q: "How are you?", a: "Fine, thanks." }, { q: "Favorite food?", a: "Soup." }]);
  assert.deepEqual(LR.parseExamples(LR.formatExamples(list)), list);
});

test("speakable text drops thinking, code and markdown", () => {
  const t = LR.speakableText("<think>hmm</think>## Title\n**Bold** and `x` with [a link](https://e.com).\n```js\nlet a\n```\n- item");
  assert.equal(t, "Title Bold and x with a link. (code) item");
});

test("chat exports as markdown with speaker names", () => {
  const md = LR.chatToMarkdown({ title: "Dinner", createdAt: 0, mode: "chat", messages: [
    { role: "user", content: "What's for dinner?" },
    { role: "assistant", who: "Ada", content: "<think>x</think>Soup." },
    { role: "assistant", content: "", error: "failed" },
  ] }, "Sam");
  assert.match(md, /^# Dinner\n/);
  assert.match(md, /\*\*Sam:\*\*\n\nWhat's for dinner\?/);
  assert.match(md, /\*\*Ada:\*\*\n\nSoup\./);
  assert.ok(!md.includes("failed"));
  assert.match(LR.chatToMarkdown({ title: "T", mode: "complete", doc: "Once upon" }), /Once upon\n$/);
});

test("chat search matches every word in titles and messages", () => {
  const chats = [
    { title: "Dinner ideas", messages: [{ content: "pasta tonight" }] },
    { title: "Books", messages: [{ content: "The Hobbit" }] },
    { title: "Story", mode: "complete", doc: "The history of Rome" },
  ];
  assert.equal(LR.searchChats(chats, "").length, 3);
  assert.deepEqual(LR.searchChats(chats, "hobbit").map((c) => c.title), ["Books"]);
  assert.deepEqual(LR.searchChats(chats, "dinner pasta").map((c) => c.title), ["Dinner ideas"]);
  assert.deepEqual(LR.searchChats(chats, "rome").map((c) => c.title), ["Story"]);
});

test("dictation joins phrases with spacing and capitals", () => {
  assert.equal(LR.appendDictation("", "hello there"), "Hello there");
  assert.equal(LR.appendDictation("Hello there.", "how are you"), "Hello there. How are you");
  assert.equal(LR.appendDictation("I want", "pizza"), "I want pizza");
  assert.equal(LR.appendDictation("abc ", "  "), "abc ");
});

test("stats line", () => {
  assert.equal(LR.statsLine({ completionTokens: 11, seconds: 4, firstTokenSeconds: 2, finishReason: "stop" }),
    "11 tokens · 5.0 tok/s · first token 2.0s");
  assert.equal(LR.statsLine({ stopped: true }), "Stopped");
});
