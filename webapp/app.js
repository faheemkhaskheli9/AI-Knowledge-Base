"use strict";
const $ = (id) => document.getElementById(id);
let TOKEN = "", files = [], cur = null, dirty = false;

const esc = (s) => s.replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

async function api(path, opts = {}) {
  const o = {method: opts.method || "GET", headers: {}};
  if (opts.body) { o.body = JSON.stringify(opts.body); o.headers["Content-Type"] = "application/json"; }
  if (o.method !== "GET") o.headers["X-KB-Token"] = TOKEN;
  const r = await fetch(path, o);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { const e = new Error(j.error || r.statusText); e.status = r.status; throw e; }
  return j;
}
function status(msg, err) { const s = $("status"); s.textContent = msg; s.className = err ? "err" : ""; }

/* ---------- markdown ---------- */
const IMAGE_EXT = /\.(png|jpe?g|gif|webp)$/i;  // what /api/image serves
function safeDecode(s) { try { return decodeURIComponent(s); } catch (e) { return s; } }
// "#/<path>" opens a file; "#/<path>?line=N" also jumps to source line N.
function fileHash(path, line) {
  return "#/" + encodeURI(path).replace(/[?#]/g, encodeURIComponent) + (line ? "?line=" + line : "");
}
function resolveRel(base, href) {
  const parts = (href.startsWith("/") ? href.slice(1) : base.split("/").slice(0, -1).join("/") + "/" + href).split("/");
  const out = [];
  for (const p of parts) { if (p === "..") out.pop(); else if (p && p !== ".") out.push(p); }
  return out.join("/");
}
function wikiTarget(name) {
  const n = name.trim().toLowerCase().replace(/\.md$/, "");
  const f = files.find((f) => f.path.toLowerCase().replace(/\.md$/, "") === n) ||
            files.find((f) => f.path.toLowerCase().replace(/\.md$/, "").split("/").pop() === n);
  return f && f.path;
}
function inline(src, base) {
  const stash = [];
  const keep = (h) => "\u0000" + (stash.push(h) - 1) + "\u0000";
  let s = src.replace(/(`+)([\s\S]*?[^`])\1(?!`)/g, (_, __, c) => keep("<code>" + esc(c.trim()) + "</code>"));
  s = esc(s);
  s = s.replace(/\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (m, t, label) => {
    const p = wikiTarget(t.replace(/&amp;/g, "&"));
    return p ? keep('<a href="' + fileHash(p) + '">' + (label || t) + "</a>") : keep('<span class="muted" title="missing">' + (label || t) + "</span>");
  });
  // Images first, so the link rule below does not take "[alt](src)" out of "![alt](src)".
  s = s.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^)]*&quot;)?\)/g, (m, alt, src) => {
    alt = alt.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[+i].replace(/<[^>]*>/g, ""));  // already escaped
    src = src.replace(/&amp;/g, "&");
    // The CSP allows only same-origin images, so remote images become links.
    if (/^https?:/i.test(src)) return keep('<a href="' + esc(src) + '" target="_blank" rel="noopener noreferrer">' + (alt || esc(src)) + "</a>");
    if (/^[a-z][a-z0-9+.-]*:/i.test(src) || src.startsWith("#") || src.startsWith("//")) return alt;
    const p = resolveRel(base, safeDecode(src.split(/[?#]/)[0]));
    if (!IMAGE_EXT.test(p)) return alt;
    return keep('<img src="/api/image?path=' + esc(encodeURIComponent(p)) + '" alt="' + alt + '" loading="lazy">');
  });
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^)]*&quot;)?\)/g, (m, text, href) => {
    href = href.replace(/&amp;/g, "&");
    if (/^(https?:|mailto:)/i.test(href)) return keep('<a href="' + esc(href) + '" target="_blank" rel="noopener noreferrer">' + text + "</a>");
    if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("#")) return text;
    const p = resolveRel(base, safeDecode(href.split("#")[0]));
    return files.some((f) => f.path === p) ? keep('<a href="' + fileHash(p) + '">' + text + "</a>") : text;
  });
  s = s.replace(/&lt;(https?:\/\/[^\s&]+)&gt;/g, (m, u) => keep('<a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(u) + "</a>"));
  s = s.replace(/\*\*([^*]+)\*\*|__([^_]+)__/g, (m, a, b) => "<strong>" + (a || b) + "</strong>")
       .replace(/(^|[^*\w])\*([^*\s][^*]*)\*(?!\w)/g, "$1<em>$2</em>")
       .replace(/(^|[^_\w])_([^_\s][^_]*)_(?!\w)/g, "$1<em>$2</em>")
       .replace(/~~([^~]+)~~/g, "<del>$1</del>");
  // Stashed HTML can hold placeholders itself (a linked image, code in link text), so expand recursively.
  const unstash = (t) => t.replace(/\u0000(\d+)\u0000/g, (_, i) => unstash(stash[+i]));
  return unstash(s);
}
function cells(line) { return line.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|")); }
// Each top-level block (and list item / table row) carries data-line = its 1-based source line,
// so a search hit can scroll to the block that holds its line. `off` = lines before `text`.
function markdown(text, base, off = 0) {
  let fm = "";
  const m = off ? null : text.match(/^---\n([\s\S]*?)\n---\n?/);
  if (m) {
    fm = '<details class="fm" data-line="1"><summary>frontmatter</summary><pre>' + esc(m[1]) + "</pre></details>";
    text = text.slice(m[0].length); off = m[0].split("\n").length - 1;
  }
  const L = text.split("\n"), out = [];
  let i = 0, pStart = 0;
  const para = [];
  const at = (n) => ' data-line="' + (off + n + 1) + '"';
  const flush = () => { if (para.length) { out.push("<p" + at(pStart) + ">" + inline(para.join(" "), base) + "</p>"); para.length = 0; } };
  while (i < L.length) {
    const line = L[i], start = i;
    let f = line.match(/^\s*(```+|~~~+)\s*(\S*)/);
    if (f) {
      flush(); const buf = []; i++;
      while (i < L.length && !L[i].trim().startsWith(f[1])) buf.push(L[i++]);
      i++; out.push("<pre" + at(start) + "><code>" + esc(buf.join("\n")) + "</code></pre>"); continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    let h = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (h) { flush(); out.push("<h" + h[1].length + at(start) + ">" + inline(h[2], base) + "</h" + h[1].length + ">"); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr" + at(start) + ">"); i++; continue; }
    if (/^\s*>/.test(line)) {
      flush(); const buf = [];
      while (i < L.length && /^\s*>/.test(L[i])) buf.push(L[i++].replace(/^\s*> ?/, ""));
      out.push("<blockquote" + at(start) + ">" + markdown(buf.join("\n"), base, off + start) + "</blockquote>"); continue;
    }
    if (line.includes("|") && i + 1 < L.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(L[i + 1])) {
      flush(); const head = cells(line); i += 2; const rows = [];
      while (i < L.length && L[i].trim() && L[i].includes("|")) rows.push({n: i, c: cells(L[i++])});
      out.push("<table" + at(start) + "><thead><tr>" + head.map((c) => "<th>" + inline(c, base) + "</th>").join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr" + at(r.n) + ">" + r.c.map((c) => "<td>" + inline(c, base) + "</td>").join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      flush(); const items = [];
      while (i < L.length) {
        const mm = L[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)/);
        if (mm) items.push({ind: mm[1].replace(/\t/g, "  ").length, ol: /\d/.test(mm[2]), t: mm[3], n: off + i + 1});
        else if (L[i].trim() && /^\s+\S/.test(L[i]) && items.length) items[items.length - 1].t += " " + L[i].trim();
        else break;
        i++;
      }
      out.push(list(items, base)); continue;
    }
    if (!para.length) pStart = i;
    para.push(line.trim()); i++;
  }
  flush();
  return fm + out.join("\n");
}
function list(items, base) {
  let html = "", stack = [];
  for (const it of items) {
    while (stack.length && it.ind < stack[stack.length - 1].ind) { html += "</li></" + stack.pop().tag + ">"; }
    const tag = it.ol ? "ol" : "ul", top = stack[stack.length - 1];
    if (!top || it.ind > top.ind) { html += "<" + tag + ">"; stack.push({ind: it.ind, tag}); }
    else html += "</li>";
    let t = it.t, box = t.match(/^\[([ xX])\]\s+(.*)/);
    if (box) t = (box[1] === " " ? "☐ " : "☑ ") + box[2];
    html += '<li data-line="' + it.n + '">' + inline(t, base);
  }
  while (stack.length) html += "</li></" + stack.pop().tag + ">";
  return html;
}
function csvTable(text) {
  const rows = []; let row = [], f = "", q = false, ln = 1, start = 1;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "\n") ln++;
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(f); f = ""; }
    else if (c === "\n") { row.push(f); row.n = start; rows.push(row); row = []; f = ""; start = ln; }
    else if (c !== "\r") f += c;
  }
  if (f || row.length) { row.push(f); row.n = start; rows.push(row); }
  if (!rows.length) return "";
  const tr = (r) => '<tr data-line="' + r.n + '">';
  return "<table><thead>" + tr(rows[0]) + rows[0].map((c) => "<th>" + esc(c) + "</th>").join("") + "</tr></thead><tbody>" +
    rows.slice(1).map((r) => tr(r) + r.map((c) => "<td>" + esc(c) + "</td>").join("") + "</tr>").join("") + "</tbody></table>";
}
function render(path, text) {
  if (/\.md$/i.test(path)) return markdown(text, path);
  if (/\.csv$/i.test(path)) return csvTable(text);
  return "<pre><code>" + text.split("\n").map((l, n) => '<span data-line="' + (n + 1) + '">' + esc(l) + "</span>").join("\n") + "</code></pre>";
}

/* ---------- tree ---------- */
function buildTree() {
  const q = $("filter").value.trim().toLowerCase(), root = {d: {}, f: []};
  for (const f of files) {
    if (q && !f.path.toLowerCase().includes(q)) continue;
    let n = root; const parts = f.path.split("/");
    for (const p of parts.slice(0, -1)) n = n.d[p] || (n.d[p] = {d: {}, f: []});
    n.f.push(f);
  }
  const draw = (n, open) => {
    let h = "";
    for (const k of Object.keys(n.d).sort()) h += "<details" + (open ? " open" : "") + "><summary>" + esc(k) + "</summary>" + draw(n.d[k], open) + "</details>";
    for (const f of n.f) h += '<a href="' + fileHash(f.path) + '" data-p="' + esc(f.path) + '" class="' + (f.readonly ? "ro " : "") + '">' + esc(f.path.split("/").pop()) + "</a>";
    return h;
  };
  $("tree").innerHTML = draw(root, !!q);
  markCur();
}
function markCur() {
  document.querySelectorAll("#tree a").forEach((a) => {
    const on = cur && a.dataset.p === cur.path; a.classList.toggle("cur", on);
    if (on) for (let p = a.parentElement; p && p.id !== "tree"; p = p.parentElement) if (p.tagName === "DETAILS") p.open = true;
  });
}

/* ---------- file view/edit ---------- */
function setMode(edit) {
  if (!cur) return;  // no file open (start page, search results)
  $("view").hidden = edit; $("editor").hidden = !edit; $("save").hidden = !edit || cur.readonly;
  $("toggle").textContent = edit ? "View" : "Edit";
  if (edit) {
    $("editor").focus();
    if (cur.line) { caretToLine($("editor"), cur.line); cur.line = 0; }
  }
}
// Put the caret at the start of 1-based line n and scroll it into view (about a third down).
function caretToLine(ta, n) {
  const v = ta.value, lines = v.split("\n");
  const pos = lines.slice(0, Math.max(0, Math.min(n, lines.length) - 1)).reduce((a, l) => a + l.length + 1, 0);
  // Measure the height of the text before the line (wrapped lines included) by loading only that text.
  ta.value = v.slice(0, pos); ta.scrollTop = 0;
  const before = ta.scrollHeight; ta.value = v;
  ta.scrollTop = before > ta.clientHeight ? before - ta.clientHeight / 3 : 0;  // scrollHeight >= clientHeight
  ta.setSelectionRange(pos, pos);
}
// Scroll the view to the deepest rendered block whose source starts at or before line n, and flash it.
function showLine(n) {
  let hit = null;
  for (const el of $("view").querySelectorAll("[data-line]")) { if (+el.dataset.line <= n) hit = el; else break; }  // document order = line order
  if (!hit) return;
  if (hit.tagName === "DETAILS") hit.open = true;
  hit.scrollIntoView({block: "center"});
  hit.classList.remove("flash"); void hit.offsetWidth; hit.classList.add("flash");
  setTimeout(() => hit.classList.remove("flash"), 2600);
}
async function open(path, line) {
  if (dirty && !confirm("Discard unsaved changes?")) { location.hash = fileHash(cur.path); return; }
  try {
    const f = await api("/api/file?path=" + encodeURIComponent(path));
    cur = f; cur.line = line || 0; dirty = false;
    $("bar").hidden = false; $("crumb").textContent = path + (f.readonly ? "  (read-only)" : "");
    $("editor").value = f.content; $("editor").readOnly = f.readonly;
    $("view").innerHTML = render(path, f.content); $("view").scrollTop = 0;
    status(""); setMode(false); markCur();
    if (line) showLine(line);
  } catch (e) { $("view").innerHTML = '<p class="muted">' + esc(e.message) + "</p>"; }
}
async function save() {
  if (!cur || cur.readonly) return;
  try {
    const content = $("editor").value;
    const r = await api("/api/file", {method: "PUT", body: {path: cur.path, content, version: cur.version}});
    cur.version = r.version; cur.content = content; dirty = false;
    $("view").innerHTML = render(cur.path, content); status("Saved " + new Date().toLocaleTimeString());
  } catch (e) { status(e.message, true); }
}
function route() {
  const m = location.hash.match(/^#\/(.*?)(?:\?line=(\d+))?$/);
  const p = m && safeDecode(m[1]);
  if (p) open(p, m[2] ? +m[2] : 0);
}

/* ---------- search ---------- */
async function doSearch() {
  const q = $("search").value.trim();
  if (q.length < 2) return;
  const hits = await api("/api/search?q=" + encodeURIComponent(q));
  const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
  $("bar").hidden = true; $("editor").hidden = true; $("view").hidden = false; cur = null; dirty = false;
  $("view").innerHTML = "<h2>" + hits.length + " result(s) for “" + esc(q) + "”</h2>" + hits.map((h) =>
    '<a class="hit" href="' + esc(fileHash(h.path, h.line)) + '"><code>' + esc(h.path) + (h.line ? ":" + h.line : "") + "</code><br>" +
    esc(h.text).replace(re, "<b>$1</b>") + "</a>").join("");
}

async function init() {
  const c = await api("/api/config");
  TOKEN = c.token; $("title").textContent = c.title; document.title = c.title;
  c.actions.forEach((a) => {
    const b = document.createElement("button"); b.textContent = a;
    b.onclick = async () => { try { const r = await api("/api/action", {method: "POST", body: {name: a}}); alert(a + " (exit " + r.code + ")\n\n" + r.output); } catch (e) { alert(e.message); } };
    $("actions").appendChild(b);
  });
  files = await api("/api/tree"); buildTree(); route();
}
$("filter").oninput = buildTree;
$("search").onkeydown = (e) => { if (e.key === "Enter") doSearch(); };
$("menu").onclick = () => $("side").classList.toggle("hidden");
$("toggle").onclick = () => setMode(!$("editor").hidden ? false : true);
$("save").onclick = save;
$("editor").oninput = () => { dirty = true; status("Unsaved changes"); };
$("new").onclick = async () => {
  const p = prompt("New file path (e.g. folder/topic.md):"); if (!p) return;
  try { await api("/api/new", {method: "POST", body: {path: p.trim(), content: /\.md$/i.test(p) ? "# " + p.trim().split("/").pop().replace(/\.md$/i, "") + "\n\n" : ""}});
    files = await api("/api/tree"); buildTree(); location.hash = fileHash(p.trim()); setTimeout(() => cur && setMode(true), 300);
  } catch (e) { alert(e.message); }
};
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(); } });
window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
window.addEventListener("hashchange", route);
// A link to the hash already in the address bar fires no hashchange (e.g. the same search hit twice,
// or the last-opened file after a search), so open it directly.
document.addEventListener("click", (e) => {
  const a = e.target.closest && e.target.closest('a[href^="#/"]');
  if (a && a.getAttribute("href") === location.hash && !e.ctrlKey && !e.metaKey && !e.shiftKey) { e.preventDefault(); route(); }
});
init();
