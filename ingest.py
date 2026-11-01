# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1"]
# ///
"""Normalize local agent transcripts into one events table.

Sources: Claude Code (~/.claude/projects), Codex (~/.codex/sessions),
pi (~/.pi/agent/sessions), Hermes (~/.hermes/state.db).

Output: data/events-<machine>.parquet and data/skills-<machine>.json
"""
import argparse
import glob
import json
import os
import re
import socket
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb

HOME = Path.home()
ROOT = Path(__file__).parent

# Relative price weights shared by Anthropic models (input = 1). Used only to
# split a session's reported cost across its calls, and to estimate cost for
# sessions with no reported cost, calibrated from the sessions that have one.
W_IN, W_OUT, W_CR, W_CW5, W_CW1H = 1.0, 5.0, 0.1, 1.25, 2.0


def iso(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        if ts > 1e11:  # milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()
    return ts


def elapsed_ms(start, end):
    try:
        a, b = (datetime.fromisoformat(x.replace("Z", "+00:00")) for x in (start, end))
        return max(0.0, (b - a).total_seconds() * 1000)
    except (AttributeError, ValueError, TypeError):
        return None


def project_of(cwd):
    return Path(cwd).name if cwd else None


def ev(**kw):
    base = dict(ts=None, machine=None, harness=None, session_id=None, project=None,
                event_type=None, name=None, detail=None, n=1, input_tokens=0,
                output_tokens=0, cache_read=0, cache_write=0, reasoning_tokens=0,
                cost_usd=None, cost_source=None, duration_ms=None, is_error=None,
                is_subagent=False)
    base.update(kw)
    return base


def jsonl(path):
    with open(path, errors="replace") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


SKILL_MD = re.compile(r"/skills/(?:[^/\s'\"]+/)*?([^/\s'\"]+)/SKILL\.md")


def skill_from_text(s):
    """A skill loaded by reading its SKILL.md (pi, Codex)."""
    m = SKILL_MD.search(s or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------- Claude Code

def claude_code(skill_names):
    out = []
    cost_state = {}  # session -> modelUsage of the last cost-state record
    files = glob.glob(str(HOME / ".claude/projects/**/*.jsonl"), recursive=True)
    for f in files:
        sub = "/subagents/" in f
        calls = {}  # message.id -> event (usage repeats per content block; keep last)
        tools = {}  # tool_use id -> event
        for d in jsonl(f):
            t = d.get("type")
            sid = d.get("sessionId")
            if t == "cost-state":
                cost_state[sid] = d.get("modelUsage") or {}
                continue
            common = dict(harness="claude-code", session_id=sid, ts=d.get("timestamp"),
                          project=project_of(d.get("cwd")), is_subagent=sub or d.get("isSidechain", False))
            msg = d.get("message") or {}
            if t == "assistant":
                model = msg.get("model")
                u = msg.get("usage") or {}
                if model and model != "<synthetic>" and u:
                    cc = u.get("cache_creation") or {}
                    calls[msg["id"]] = ev(**common, event_type="model_call", name=model,
                        input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                        cache_read=u.get("cache_read_input_tokens", 0),
                        cache_write=u.get("cache_creation_input_tokens", 0),
                        reasoning_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                        _cw1h=cc.get("ephemeral_1h_input_tokens", 0))
                for b in msg.get("content") or []:
                    if b.get("type") != "tool_use" or b["id"] in tools:
                        continue
                    e = ev(**common, event_type="tool_call", name=b["name"])
                    tools[b["id"]] = e
                    inp = b.get("input") or {}
                    if b["name"] == "Skill" and inp.get("skill"):
                        out.append(ev(**common, event_type="skill", name=inp["skill"], detail="model"))
            elif t == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    m = re.search(r"<command-name>/?([^<]+)</command-name>", content)
                    if m:
                        name = m.group(1).strip()
                        kind = "skill" if name in skill_names else "command"
                        out.append(ev(**common, event_type=kind, name=name, detail="user"))
                elif isinstance(content, list):
                    r = d.get("toolUseResult") if isinstance(d.get("toolUseResult"), dict) else {}
                    for b in content:
                        if b.get("type") == "tool_result" and b.get("tool_use_id") in tools:
                            e = tools[b["tool_use_id"]]
                            e["is_error"] = bool(b.get("is_error"))
                            if "durationMs" in r:
                                e["duration_ms"] = r["durationMs"]
                            elif "durationSeconds" in r:
                                e["duration_ms"] = r["durationSeconds"] * 1000
                            else:
                                e["duration_ms"] = elapsed_ms(e["ts"], d.get("timestamp"))
        out += calls.values()
        out += tools.values()

    # Cost: split each session's reported per-model cost across its calls by
    # price weight; calibrate a $/unit rate per model for sessions without one.
    def units(e):
        cw1h = e.get("_cw1h", 0)
        return (e["input_tokens"] * W_IN + e["output_tokens"] * W_OUT + e["cache_read"] * W_CR
                + cw1h * W_CW1H + (e["cache_write"] - cw1h) * W_CW5)

    by_sm = defaultdict(list)
    for e in out:
        if e["event_type"] == "model_call":
            by_sm[(e["session_id"], e["name"])].append(e)
    rate_num, rate_den = defaultdict(float), defaultdict(float)
    for (sid, model), evs in by_sm.items():
        reported = (cost_state.get(sid) or {}).get(model, {}).get("costUSD")
        total = sum(units(e) for e in evs)
        if reported is None or total == 0:
            continue
        for e in evs:
            e["cost_usd"] = reported * units(e) / total
            e["cost_source"] = "reported"
        rate_num[model] += reported
        rate_den[model] += total
    for (sid, model), evs in by_sm.items():
        if evs[0]["cost_source"] or not rate_den[model]:
            continue
        rate = rate_num[model] / rate_den[model]
        for e in evs:
            e["cost_usd"] = units(e) * rate
            e["cost_source"] = "estimated"
    for e in out:
        e.pop("_cw1h", None)
    return out


# ---------------------------------------------------------------------- Codex

def codex():
    out = []
    for f in glob.glob(str(HOME / ".codex/sessions/**/*.jsonl"), recursive=True):
        sid = cwd = model = None
        prev_total = None
        tools = {}
        for d in jsonl(f):
            p = d.get("payload") if isinstance(d.get("payload"), dict) else {}
            t, pt = d.get("type"), p.get("type")
            if t == "session_meta":
                sid, cwd = p.get("id"), p.get("cwd")
                continue
            if t == "turn_context":
                model = p.get("model") or model
                cwd = p.get("cwd") or cwd
                continue
            common = dict(harness="codex", session_id=sid, ts=d.get("timestamp"), project=project_of(cwd))
            if t == "event_msg" and pt == "token_count":
                tot = ((p.get("info") or {}).get("total_token_usage")) or {}
                if prev_total is not None and tot.get("total_tokens", 0) > prev_total.get("total_tokens", 0):
                    dl = {k: tot.get(k, 0) - prev_total.get(k, 0) for k in tot}
                    cached = dl.get("cached_input_tokens", 0)
                    out.append(ev(**common, event_type="model_call", name=model,
                        input_tokens=dl.get("input_tokens", 0) - cached, cache_read=cached,
                        cache_write=dl.get("cache_write_input_tokens", 0),
                        output_tokens=dl.get("output_tokens", 0),
                        reasoning_tokens=dl.get("reasoning_output_tokens", 0)))
                if tot:
                    prev_total = tot
            elif t == "response_item" and pt in ("function_call", "custom_tool_call"):
                e = ev(**common, event_type="tool_call", name=p.get("name"))
                tools[p.get("call_id")] = e
                skill = skill_from_text(p.get("arguments") or p.get("input"))
                if skill:
                    out.append(ev(**common, event_type="skill", name=skill, detail="read"))
            elif t == "response_item" and pt in ("function_call_output", "custom_tool_call_output"):
                e = tools.get(p.get("call_id"))
                o = p.get("output") if isinstance(p.get("output"), str) else json.dumps(p.get("output"))
                if e:
                    m = re.search(r"(?:exited with code|Exit code:) (\d+)", o)
                    e["is_error"] = bool(m and m.group(1) != "0")
                    m = re.search(r"Wall time: ([\d.]+) seconds", o)
                    if m:
                        e["duration_ms"] = float(m.group(1)) * 1000
        out += tools.values()
    return out


# ------------------------------------------------------------------------- pi

def pi():
    out = []
    for f in glob.glob(str(HOME / ".pi/agent/sessions/**/*.jsonl"), recursive=True):
        sid = cwd = None
        tools = {}
        for d in jsonl(f):
            if d.get("type") == "session":
                sid, cwd = d.get("id"), d.get("cwd")
                continue
            m = d.get("message") or {}
            common = dict(harness="pi", session_id=sid, ts=d.get("timestamp"), project=project_of(cwd))
            if m.get("role") == "assistant":
                u = m.get("usage") or {}
                if u:
                    out.append(ev(**common, event_type="model_call", name=m.get("model"),
                        input_tokens=u.get("input", 0), output_tokens=u.get("output", 0),
                        cache_read=u.get("cacheRead", 0), cache_write=u.get("cacheWrite", 0),
                        reasoning_tokens=u.get("reasoning", 0),
                        cost_usd=(u.get("cost") or {}).get("total"), cost_source="reported"))
                for b in m.get("content") or []:
                    if b.get("type") == "toolCall":
                        e = ev(**common, event_type="tool_call", name=b.get("name"))
                        tools[b.get("id")] = e
                        skill = skill_from_text(json.dumps(b.get("arguments")))
                        if skill:
                            out.append(ev(**common, event_type="skill", name=skill, detail="read"))
            elif m.get("role") == "toolResult" and m.get("toolCallId") in tools:
                e = tools[m["toolCallId"]]
                e["is_error"] = bool(m.get("isError"))
                e["duration_ms"] = elapsed_ms(e["ts"], d.get("timestamp"))
        out += tools.values()
    return out


# --------------------------------------------------------------------- Hermes

def hermes():
    db = HOME / ".hermes/state.db"
    if not db.exists():
        return []
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sessions = {r[0]: r[1:] for r in c.execute("select id, cwd, parent_session_id from sessions")}
    out = []

    def common(sid, ts):
        cwd, parent = sessions.get(sid, (None, None))
        return dict(harness="hermes", session_id=sid, ts=iso(ts), project=project_of(cwd),
                    is_subagent=parent is not None)

    for (sid, model, task, n, i, o, cr, cw, r, est, act, first) in c.execute(
            """select session_id, model, task, api_call_count, input_tokens, output_tokens,
                      cache_read_tokens, cache_write_tokens, reasoning_tokens,
                      estimated_cost_usd, actual_cost_usd, first_seen from session_model_usage"""):
        out.append(ev(**common(sid, first), event_type="model_call", name=model, detail=task or "main",
            n=n, input_tokens=i, output_tokens=o, cache_read=cr, cache_write=cw, reasoning_tokens=r,
            cost_usd=act or est, cost_source="reported" if act else "estimated"))

    tools = {}
    for sid, ts, calls in c.execute(
            "select session_id, timestamp, tool_calls from messages where tool_calls is not null and tool_calls != '[]'"):
        for call in json.loads(calls):
            fn = call.get("function") or {}
            e = ev(**common(sid, ts), event_type="tool_call", name=fn.get("name"))
            tools[call.get("id")] = e
            if fn.get("name") == "skill_view":
                try:
                    name = json.loads(fn.get("arguments") or "{}").get("name") or ""
                    out.append(ev(**common(sid, ts), event_type="skill",
                                  name=name.split(":")[-1], detail="model"))
                except json.JSONDecodeError:
                    pass
    for call_id, ts, content in c.execute(
            "select tool_call_id, timestamp, content from messages where role = 'tool'"):
        e = tools.get(call_id)
        if not e:
            continue
        e["duration_ms"] = max(0.0, (ts - datetime.fromisoformat(e["ts"]).timestamp()) * 1000)
        try:
            res = json.loads(content or "")
            e["is_error"] = isinstance(res, dict) and (res.get("success") is False or bool(res.get("error")))
        except (json.JSONDecodeError, TypeError):
            e["is_error"] = False
    out += tools.values()
    return out


# ------------------------------------------------------------ skill inventory

def skill_inventory():
    """Installed skills per harness, so the dashboard can show unused ones."""
    inv = defaultdict(set)
    pats = {
        "claude-code": [".claude/skills/*/SKILL.md", ".claude/plugins/**/skills/*/SKILL.md",
                        "Development/**/.claude/skills/*/SKILL.md"],
        "hermes": [".hermes/skills/**/SKILL.md"],
        "pi": [".pi/agent/skills/**/SKILL.md"],
        "codex": [".codex/skills/**/SKILL.md"],
    }
    for harness, globs in pats.items():
        for g in globs:
            for p in glob.glob(str(HOME / g), recursive=True):
                if "/node_modules/" not in p:
                    inv[harness].add(Path(p).parent.name)
    return {k: sorted(v) for k, v in inv.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", default=socket.gethostname().split(".")[0])
    args = ap.parse_args()

    inv = skill_inventory()
    all_skills = set().union(*inv.values()) if inv else set()
    rows = []
    for name, fn in [("claude-code", lambda: claude_code(all_skills)), ("codex", codex), ("pi", pi), ("hermes", hermes)]:
        got = fn()
        print(f"{name:12} {len(got):6} events")
        rows += got
    for r in rows:
        r["machine"] = args.machine

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as tmp:
        for r in rows:
            tmp.write(json.dumps(r) + "\n")
    dest = data / f"events-{args.machine}.parquet"
    # Harnesses prune old transcripts (Claude Code after ~30 days), so keep
    # previously ingested sessions that are no longer on disk.
    keep = ""
    if dest.exists():
        keep = f"""union all by name
          select * from '{dest}' p
          where not exists (select 1 from fresh f where f.harness = p.harness
                            and f.session_id is not distinct from p.session_id)"""
    duckdb.sql(f"""
        create temp table fresh as
          select cast(ts as timestamptz) as ts, * exclude (ts)
          from read_json('{tmp.name}', format='newline_delimited', columns={{
            ts:'varchar', machine:'varchar', harness:'varchar', session_id:'varchar',
            project:'varchar', event_type:'varchar', name:'varchar', detail:'varchar',
            n:'bigint', input_tokens:'bigint', output_tokens:'bigint', cache_read:'bigint',
            cache_write:'bigint', reasoning_tokens:'bigint', cost_usd:'double',
            cost_source:'varchar', duration_ms:'double', is_error:'boolean', is_subagent:'boolean'}})
          where ts is not null
    """)
    duckdb.sql(f"copy (select * from fresh {keep} order by ts) to '{dest}.tmp' (format parquet)")
    os.replace(f"{dest}.tmp", dest)
    os.unlink(tmp.name)
    (data / f"skills-{args.machine}.json").write_text(json.dumps(inv, indent=2))
    total = duckdb.sql(f"select count(*) from '{dest}'").fetchall()[0][0]
    print(f"wrote {dest} ({len(rows)} fresh, {total} total events)")


if __name__ == "__main__":
    main()
