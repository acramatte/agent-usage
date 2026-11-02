"""Amp threads, read from ampcode.com through the `amp` CLI (needs `amp login`).

Amp keeps threads on its server, not on disk, so this is not a registry
adapter: `agent-usage ingest --amp` writes them to events-ampcode.parquet, once
for every machine. Threads whose `updated` time is unchanged since the last run
are not fetched again.
"""
import json
import re
import shutil
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .ingest import W_CR, W_CW5, W_IN, W_OUT, elapsed_ms, ev, iso

PAGE = 500  # the CLI's maximum --limit
WORKERS = 8


class RateLimited(Exception):
    pass


def amp(*args):
    p = subprocess.run(["amp", *args], capture_output=True, text=True)
    if p.returncode:
        if "limited to" in p.stderr:
            raise RateLimited(p.stderr.strip())
        raise SystemExit(f"amp {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def list_threads():
    out, offset = [], 0
    while True:
        page = json.loads(amp("threads", "list", "--json", "--include-archived",
                              "--limit", str(PAGE), "--offset", str(offset)))
        out += page
        if len(page) < PAGE:
            return out
        offset += PAGE


def warn(tid, msg):
    print(f"amp          warning: {tid}: {msg}", file=sys.stderr)


def fetch(tid, limited):
    """The thread, and its usage report unless the report's rate limit (60 an
    hour) has been hit; `limited` is set on the first refusal."""
    try:
        thread = json.loads(amp("threads", "export", tid))
    except json.JSONDecodeError as e:
        raise ValueError(f"export is not JSON: {e}") from e
    if limited.is_set():
        return thread, None
    try:
        return thread, amp("threads", "usage", "--details", tid)
    except RateLimited:
        limited.set()
        return thread, None


# ------------------------------------------------------------------- parsing

def model_key(name):
    """'openai/gpt-6.1-sol', 'gpt-6.1-sol' and 'GPT-6.1 Sol' -> 'gpt-6-1-sol'."""
    return re.sub(r"[\s._]+", "-", name.rsplit("/", 1)[-1].strip().lower())


def num(s):
    s = re.sub(r"[^\d.]", "", s or "")
    return float(s) if s else 0.0


MODEL_COLS = {"Model", "Requests", "Input", "Output"}
COST_COLS = ("Amp cost", "Cost")  # 'Cost' when the table has no list-price column
TOTALS = dict(input=r"^Input tokens: ([\d,]+)", cache_read=r"^Input tokens: [\d,]+ \(([\d,]+) cache reads\)",
              output=r"^Output tokens: ([\d,]+)")


def usage_models(md, tid=None):
    """The usage report's Models table, by model key: requests, input (incl.
    cache), output, Amp cost and estimated list price; and the report's thread
    totals (None where a line is missing). Warns about a changed layout."""
    models = defaultdict(lambda: dict(requests=0, input=0, output=0, amp=0.0, list=0.0))
    section = cols = None
    has_table = False
    for line in md.splitlines():
        if line.startswith("## "):
            section, cols = line[3:].strip(), None
        elif section == "Models" and line.startswith("|") and not line.startswith("| -"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cols is None:  # the header; the list-price column is not always there
                cols, has_table = cells, True
                if MODEL_COLS - set(cols) or not set(COST_COLS) & set(cols):
                    warn(tid, f"usage report's Models table has unexpected columns {cols}")
                continue
            row = dict(zip(cols, cells))
            m = models[model_key(row["Model"])]
            m["name"] = re.sub(r"\s+", "-", row["Model"].lower())  # 'GPT-5.6 Terra' -> 'gpt-5.6-terra'
            m["requests"] += int(num(row.get("Requests")))
            m["input"] += int(num(row.get("Input")))
            m["output"] += int(num(row.get("Output")))
            m["amp"] += num(row.get(COST_COLS[0], row.get(COST_COLS[1])))
            m["list"] += num(row.get("Est. list price"))
    if not has_table:
        warn(tid, "usage report has no Models table")
    totals = {}
    for k, pat in TOTALS.items():
        m = re.search(pat, md, re.M)
        totals[k] = int(num(m.group(1))) if m else None
        if not m:
            warn(tid, f"usage report has no line matching {pat!r}")
    return dict(models), totals


def project_of(thread):
    trees = (thread.get("env") or {}).get("initial", {}).get("trees") or []
    if not trees:
        return None
    url = (trees[0].get("repository") or {}).get("url") or trees[0].get("uri") or ""
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git") or None


def units(e):
    return (e["input_tokens"] * W_IN + e["output_tokens"] * W_OUT
            + e["cache_read"] * W_CR + e["cache_write"] * W_CW5)


def thread_events(thread, usage_md):
    """Events for one exported thread. Model calls and tool calls come from the
    messages; requests the export does not show (subagents such as Task, finder
    and oracle, titles, compaction) from the usage report's Models table, as one
    aggregate row per model. Each model's cost (Amp's charge, plus the list-price
    estimate for your own provider keys or subscriptions) is split across its
    rows by price weight. Without a usage report, only the messages."""
    tid = thread["id"]
    common = dict(harness="amp", session_id=tid, project=project_of(thread))
    out, tools = [], {}
    for msg in thread.get("messages") or []:
        u = msg.get("usage") or {}
        ts = u.get("timestamp") or msg.get("createdAt") or iso((msg.get("meta") or {}).get("sentAt"))
        if msg.get("role") == "assistant" and u.get("model"):
            out.append(ev(**common, ts=ts, event_type="model_call", name=u["model"].rsplit("/", 1)[-1],
                          input_tokens=u.get("inputTokens", 0), output_tokens=u.get("outputTokens", 0),
                          cache_read=u.get("cacheReadInputTokens", 0),
                          cache_write=u.get("cacheCreationInputTokens", 0)))
        for b in msg.get("content") or []:
            if b.get("type") == "tool_use":
                tools[b["id"]] = e = ev(**common, ts=ts, event_type="tool_call", name=b["name"])
                out.append(e)
                name = (b.get("input") or {}).get("name")
                if b["name"] == "skill" and name:
                    out.append(ev(**common, ts=ts, event_type="skill", name=name, detail="model"))
            elif b.get("type") == "tool_result" and b.get("toolUseID") in tools:
                e = tools[b["toolUseID"]]
                e["is_error"] = (b.get("run") or {}).get("status") == "error"
                # Results of one turn arrive in one message: an upper bound per tool.
                e["duration_ms"] = elapsed_ms(e["ts"], msg.get("createdAt"))

    if usage_md is None:
        return out
    models, totals = usage_models(usage_md, tid)
    calls = defaultdict(list)
    for e in out:
        if e["event_type"] == "model_call":
            calls[model_key(e["name"])].append(e)
    # The report may name a model differently from the export: without its prefix
    # ('Sonnet 5.5' for 'claude-sonnet-5-5'), or as an alias for a dated release
    # ('~deepseek/deepseek-v4-flash-latest' for 'deepseek-v4-flash-0731').
    def base(k):
        return re.sub(r"-(latest|\d{4,8})$", "", k)

    for k in [k for k in models if k not in calls]:
        alias = [c for c in calls if c not in models
                 and (c.endswith("-" + k) or k.endswith("-" + c) or base(c) == base(k))]
        if len(alias) == 1:
            models[alias[0]] = models.pop(k)
    rest = {}
    for k, m in models.items():
        seen = calls.get(k, [])
        n = m["requests"] - len(seen)
        if n > 0:
            rest[k] = (n, max(0, m["input"] - sum(e["input_tokens"] + e["cache_read"] + e["cache_write"]
                                                  for e in seen)),
                       max(0, m["output"] - sum(e["output_tokens"] for e in seen)))
    # The report gives cache reads per thread only; share the unseen ones out.
    cache_left = max(0, (totals["cache_read"] or 0) - sum(e["cache_read"] for evs in calls.values() for e in evs))
    rest_in = sum(r[1] for r in rest.values())
    first = min((e["ts"] for e in out if e["ts"]), default=iso(thread.get("created")))
    for k, (n, inp, outp) in rest.items():
        cr = min(inp, round(cache_left * inp / rest_in)) if rest_in else 0
        name = calls[k][0]["name"] if calls.get(k) else models[k]["name"]
        e = ev(**common, ts=first, event_type="model_call", name=name, detail="subagent", n=n,
               input_tokens=inp - cr, cache_read=cr, output_tokens=outp, is_subagent=True)
        calls[k].append(e)
        out.append(e)
    for k, evs in calls.items():
        m = models.get(k)
        total = sum(units(e) for e in evs)
        if not m or not (m["amp"] or m["list"]):
            continue
        cost, src = m["amp"] + m["list"], "estimated" if m["list"] else "reported"
        for e in evs:
            e["cost_usd"] = cost * (units(e) / total if total else 1 / len(evs))
            e["cost_source"] = src

    # The rows should add up to the report: a mismatch means the export and the
    # report disagree (e.g. model names no longer match) or the report changed.
    mc = [e for e in out if e["event_type"] == "model_call"]
    got = dict(input=sum(e["input_tokens"] + e["cache_read"] + e["cache_write"] for e in mc),
               cache_read=sum(e["cache_read"] for e in mc), output=sum(e["output_tokens"] for e in mc))
    for k, want in totals.items():
        # Sharing out cache reads rounds once per subagent row.
        if want is not None and abs(got[k] - want) > len(rest):
            warn(tid, f"{k} tokens: events add up to {got[k]:,}, the usage report says {want:,}")
    return out


# --------------------------------------------------------------------- ingest

def collect(data_dir, known_sessions):
    """Events for threads new or updated since the last run, and the state to
    save once they are written."""
    if not shutil.which("amp"):
        raise SystemExit("ingest --amp needs the amp CLI: https://ampcode.com/manual")
    state_file = Path(data_dir) / "amp-threads.json"
    seen = json.loads(state_file.read_text()) if state_file.exists() else {}
    threads = list_threads()
    todo = [t["id"] for t in threads
            if t["id"] not in known_sessions or seen.get(t["id"]) != t["updated"]]
    print(f"amp          {len(threads)} threads, fetching {len(todo)} new or updated")
    rows, limited, done = [], threading.Event(), set()

    def one(tid):
        """A thread's events and whether its usage report was read; on an
        unexpected export or report, a warning and no events (retried next run)."""
        try:
            thread, usage_md = fetch(tid, limited)
            return thread_events(thread, usage_md), usage_md is not None
        except Exception as e:  # noqa: BLE001 -- one odd thread must not stop the rest
            warn(tid, f"skipped: {type(e).__name__}: {e}")
            return [], False

    with ThreadPoolExecutor(WORKERS) as pool:
        for tid, (events, reported) in zip(todo, pool.map(one, todo)):
            rows += events
            if reported:
                done.add(tid)
    if limited.is_set() and (missing := len(todo) - len(done)):
        print(f"amp          {missing} threads without cost or subagent usage: the usage report "
              "is limited to 60 an hour; run again later to fill them in")
    # Threads still missing their report stay out of the state, so the next run retries them.
    state = {t["id"]: t["updated"] for t in threads
             if t["id"] in done or (t["id"] not in todo and t["id"] in seen)}

    def save():
        state_file.write_text(json.dumps(state, indent=1))

    return rows, save
