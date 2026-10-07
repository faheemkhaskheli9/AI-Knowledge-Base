"""Run the python code blocks in each topic, so "minimal and runnable" is tested.

    python check_code.py                 # run everything runnable here, exit 1 on a failure
    python check_code.py concepts/x.md   # only these files
    python check_code.py --promote       # also set status: passing -> stable, failing stable -> needs-review

The blocks of one file are joined in order and run as one script in an empty
temp dir (120 s limit). A file is skipped, not failed, when it needs a package
that is not installed, network access, credentials or a model/dataset download.
A slow benchmark can raise its limit with `<!-- check-timeout: 900 -->` anywhere in the file.
Mark a deliberate fragment with `<!-- skip-check: reason -->` on the line above
its fence. stdlib only; the topic code needs numpy, scikit-learn, etc. installed.
"""
import argparse
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from build_index import ROOT, SKIP, frontmatter

TIMEOUT = 120
NEEDS_NETWORK = re.compile(
    r"API_KEY|os\.environ|getenv|from_pretrained|load_dataset|fetch_\w+\(|hf_hub|"
    r"requests\.|httpx|urlopen|input\(|\.pull\(|snapshot_download|\.from_env\("
)
NETWORK_MODULES = {"openai", "anthropic", "requests", "httpx", "mcp", "huggingface_hub", "datasets",
                   "transformers", "sentence_transformers", "trl", "peft", "diffusers", "langchain",
                   "llama_index", "chromadb", "faiss", "ollama", "vllm", "whisper", "openai_whisper",
                   "gradio", "streamlit", "fastapi", "uvicorn", "cv2", "ultralytics", "timm"}
LIMIT = re.compile(r"<!--\s*check-timeout:\s*(\d+)\s*-->")
FENCE = re.compile(r"(<!--\s*skip-check:[^>]*-->\s*\n)?```(?:python|py)\n(.*?)```", re.S)


def imported(code):
    """Top-level module names imported anywhere in code; None if it does not parse."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.update(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            mods.add(n.module.split(".")[0])
    return mods


def blocks(text):
    return [m.group(2) for m in FENCE.finditer(text) if not m.group(1)]


def check(path):
    rel = path.relative_to(ROOT).as_posix()
    text = path.read_text(encoding="utf-8")
    code = "\n".join(blocks(text))
    limit = int(LIMIT.search(text).group(1)) if LIMIT.search(text) else TIMEOUT
    if not code.strip():
        return rel, "none", ""
    mods = imported(code)
    if mods is None:
        return rel, "fail", "does not parse (SyntaxError)"
    if mods & NETWORK_MODULES or NEEDS_NETWORK.search(code):
        return rel, "skip", "needs network, credentials or a download"
    missing = sorted(m for m in mods if m != "__future__" and importlib.util.find_spec(m) is None)
    if missing:
        return rel, "skip", "not installed: " + ", ".join(missing)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "topic.py"
        script.write_text(code, encoding="utf-8")
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0", "PYTHONWARNINGS": "ignore",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            r = subprocess.run([sys.executable, str(script)], cwd=tmp, env=env, capture_output=True,
                               text=True, timeout=limit)
        except subprocess.TimeoutExpired:
            return rel, "fail", f"timed out after {limit} s"
    if r.returncode:
        lines = [l for l in r.stderr.strip().splitlines() if l.strip()]
        return rel, "fail", lines[-1][:200] if lines else f"exit {r.returncode}"
    return rel, "pass", ""


def set_status(path, status):
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"^(status:[ \t]*)[A-Za-z-]+", lambda m: m.group(1) + status, text, count=1, flags=re.M)
    if new != text:
        path.write_text(new, encoding="utf-8", newline="\n")
    return new != text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.files:
        paths = [ROOT / f for f in a.files]
    else:
        paths = [p for p in sorted(ROOT.rglob("*.md")) if p.name not in SKIP and not p.relative_to(ROOT).as_posix().startswith(".")]
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(check, paths))
    count = {k: 0 for k in ("pass", "fail", "skip", "none")}
    changed = 0
    for (rel, state, why), path in zip(results, paths):
        count[state] += 1
        if state == "fail":
            print(f"[x] {rel}: {why}")
        if a.promote:
            meta = frontmatter(path.read_text(encoding="utf-8")) or {}
            cur = meta.get("status")
            if state == "pass" and cur == "draft" and meta.get("sources"):
                changed += set_status(path, "stable")
            elif state == "fail" and cur == "stable":
                changed += set_status(path, "needs-review")
    skips = {}
    for rel, state, why in results:
        if state == "skip":
            skips[why.split(":")[0]] = skips.get(why.split(":")[0], 0) + 1
    print(f"passed {count['pass']}, failed {count['fail']}, skipped {count['skip']} {skips or ''}, no code {count['none']}")
    if a.promote:
        print(f"status changed in {changed} file(s); run python build_index.py")
    return 1 if count["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
