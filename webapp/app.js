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
    return p ? keep('<a href="#/' + encodeURI(p) + '">' + (label || t) + "</a>") : keep('<span class="muted" title="missing">' + (label || t) + "</span>");
  });
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^)]*&quot;)?\)/g, (m, text, href) => {
    href = href.replace(/&amp;/g, "&");
    if (/^(https?:|mailto:)/i.test(href)) return keep('<a href="' + esc(href) + '" target="_blank" rel="noopener noreferrer">' + text + "</a>");
    if (/^[a-z][a-z0-9+.-]*:/i.test(href) || href.startsWith("#")) return text;
    const p = resolveRel(base, decodeURI(href.split("#")[0]));
    return files.some((f) => f.path === p) ? keep('<a href="#/' + encodeURI(p) + '">' + text + "</a>") : text;
  });
  s = s.replace(/&lt;(https?:\/\/[^\s&]+)&gt;/g, (m, u) => keep('<a href="' + esc(u) + '" target="_blank" rel="noopener noreferrer">' + esc(u) + "</a>"));
  s = s.replace(/\*\*([^*]+)\*\*|__([^_]+)__/g, (m, a, b) => "<strong>" + (a || b) + "</strong>")
       .replace(/(^|[^*\w])\*([^*\s][^*]*)\*(?!\w)/g, "$1<em>$2</em>")
       .replace(/(^|[^_\w])_([^_\s][^_]*)_(?!\w)/g, "$1<em>$2</em>")
       .replace(/~~([^~]+)~~/g, "<del>$1</del>");
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => stash[+i]);
}
function cells(line) { return line.trim().replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|")); }
function markdown(text, base) {
  let fm = "";
  const m = text.match(/^---\n([\s\S]*?)\n---\n?/);
  if (m) { fm = '<details class="fm"><summary>frontmatter</summary><pre>' + esc(m[1]) + "</pre></details>"; text = text.slice(m[0].length); }
  const L = text.split("\n"), out = [];
  let i = 0;
  const para = [];
  const flush = () => { if (para.length) { out.push("<p>" + inline(para.join(" "), base) + "</p>"); para.length = 0; } };
  while (i < L.length) {
    const line = L[i];
    let f = line.match(/^\s*(```+|~~~+)\s*(\S*)/);
    if (f) {
      flush(); const buf = []; i++;
      while (i < L.length && !L[i].trim().startsWith(f[1])) buf.push(L[i++]);
      i++; out.push("<pre><code>" + esc(buf.join("\n")) + "</code></pre>"); continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    let h = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (h) { flush(); out.push("<h" + h[1].length + ">" + inline(h[2], base) + "</h" + h[1].length + ">"); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr>"); i++; continue; }
    if (/^\s*>/.test(line)) {
      flush(); const buf = [];
      while (i < L.length && /^\s*>/.test(L[i])) buf.push(L[i++].replace(/^\s*> ?/, ""));
      out.push("<blockquote>" + markdown(buf.join("\n"), base) + "</blockquote>"); continue;
    }
    if (line.includes("|") && i + 1 < L.length && /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(L[i + 1])) {
      flush(); const head = cells(line); i += 2; const rows = [];
      while (i < L.length && L[i].trim() && L[i].includes("|")) rows.push(cells(L[i++]));
      out.push("<table><thead><tr>" + head.map((c) => "<th>" + inline(c, base) + "</th>").join("") + "</tr></thead><tbody>" +
        rows.map((r) => "<tr>" + r.map((c) => "<td>" + inline(c, base) + "</td>").join("") + "</tr>").join("") + "</tbody></table>");
      continue;
    }
    if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
      flush(); const items = [];
      while (i < L.length) {
        const mm = L[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)/);
        if (mm) items.push({ind: mm[1].replace(/\t/g, "  ").length, ol: /\d/.test(mm[2]), t: mm[3]});
        else if (L[i].trim() && /^\s+\S/.test(L[i]) && items.length) items[items.length - 1].t += " " + L[i].trim();
        else break;
        i++;
      }
      out.push(list(items, base)); continue;
    }
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
    html += "<li>" + inline(t, base);
  }
  while (stack.length) html += "</li></" + stack.pop().tag + ">";
  return html;
}
function csvTable(text) {
  const rows = []; let row = [], f = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(f); f = ""; }
    else if (c === "\n") { row.push(f); rows.push(row); row = []; f = ""; }
    else if (c !== "\r") f += c;
  }
  if (f || row.length) { row.push(f); rows.push(row); }
  if (!rows.length) return "";
  return "<table><thead><tr>" + rows[0].map((c) => "<th>" + esc(c) + "</th>").join("") + "</tr></thead><tbody>" +
    rows.slice(1).map((r) => "<tr>" + r.map((c) => "<td>" + esc(c) + "</td>").join("") + "</tr>").join("") + "</tbody></table>";
}
function render(path, text) {
  if (/\.md$/i.test(path)) return markdown(text, path);
  if (/\.csv$/i.test(path)) return csvTable(text);
  return "<pre><code>" + esc(text) + "</code></pre>";
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
    for (const f of n.f) h += '<a href="#/' + encodeURI(f.path) + '" data-p="' + esc(f.path) + '" class="' + (f.readonly ? "ro " : "") + '">' + esc(f.path.split("/").pop()) + "</a>";
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
  $("view").hidden = edit; $("editor").hidden = !edit; $("save").hidden = !edit || cur.readonly;
  $("toggle").textContent = edit ? "View" : "Edit";
  if (edit) $("editor").focus();
}
async function open(path) {
  if (dirty && !confirm("Discard unsaved changes?")) { location.hash = "#/" + encodeURI(cur.path); return; }
  try {
    const f = await api("/api/file?path=" + encodeURIComponent(path));
    cur = f; dirty = false;
    $("bar").hidden = false; $("crumb").textContent = path + (f.readonly ? "  (read-only)" : "");
    $("editor").value = f.content; $("editor").readOnly = f.readonly;
    $("view").innerHTML = render(path, f.content); $("view").scrollTop = 0;
    status(""); setMode(false); markCur();
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
  const p = decodeURI(location.hash.replace(/^#\//, ""));
  if (p) open(p);
}

/* ---------- search ---------- */
async function doSearch() {
  const q = $("search").value.trim();
  if (q.length < 2) return;
  const hits = await api("/api/search?q=" + encodeURIComponent(q));
  const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
  $("bar").hidden = true; $("editor").hidden = true; $("view").hidden = false; cur = null; dirty = false;
  $("view").innerHTML = "<h2>" + hits.length + " result(s) for “" + esc(q) + "”</h2>" + hits.map((h) =>
    '<a class="hit" href="#/' + encodeURI(h.path) + '"><code>' + esc(h.path) + (h.line ? ":" + h.line : "") + "</code><br>" +
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
    files = await api("/api/tree"); buildTree(); location.hash = "#/" + encodeURI(p.trim()); setTimeout(() => cur && setMode(true), 300);
  } catch (e) { alert(e.message); }
};
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "s") { e.preventDefault(); save(); } });
window.addEventListener("beforeunload", (e) => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
window.addEventListener("hashchange", route);
init();
