"""Normalize local agent transcripts into one events table.

Sources: Claude Code, Codex, pi, OMP and Hermes, at the store paths the vendored
registry lists for them (env overrides and profiles included).

Output: <data dir>/events-<machine>.parquet and <data dir>/skills-<machine>.json
"""
import glob
import json
import os
import re
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from . import registry

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


class ToolCalls:
    """Tool calls by id: issued, then ok or error at their first result, or no
    result (is_error None) when the transcript ends first. A call or result
    repeated under the same id counts once, the first time."""

    def __init__(self):
        self.calls, self.started = {}, {}

    def issue(self, cid, e):
        """Track e as call cid; False when cid was issued before (e is dropped)."""
        if cid is None:
            cid = object()  # cannot be matched to a result, but still counts
        if cid in self.calls:
            return False
        self.calls[cid] = e
        return True

    def start(self, cid, ts):
        """When the call began running, if the harness logs it apart from the call."""
        self.started.setdefault(cid, ts)

    def result(self, cid, is_error, end_ts=None, duration_ms=None):
        """The call's first result; duration_ms, else the time from start to end_ts."""
        e = self.calls.get(cid)
        if e is None or e["is_error"] is not None:
            return
        e["is_error"] = bool(is_error)
        if duration_ms is None and end_ts is not None:
            duration_ms = elapsed_ms(self.started.get(cid) or e["ts"], end_ts)
        e["duration_ms"] = duration_ms

    def events(self):
        return list(self.calls.values())


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

def units(e):
    """A Claude Code call's price-weighted token count."""
    cw1h = e.get("_cw1h", 0)
    return (e["input_tokens"] * W_IN + e["output_tokens"] * W_OUT + e["cache_read"] * W_CR
            + cw1h * W_CW1H + (e["cache_write"] - cw1h) * W_CW5)


def skill_name(name):
    """`<plugin>:<skill>` as the inventory lists it, by its directory name."""
    return name.split(":")[-1]


def model_family(key):
    """cost-state keys 1M-context use as `<model>[1m]`; messages log `<model>`."""
    return re.sub(r"\[[^\]]*\]$", "", key)


def claude_code(files, skill_names=(), prior_rates=None):
    """prior_rates: {(session, model): (reported $, units)} from earlier ingests,
    so estimates stay calibrated after the reporting sessions are pruned."""
    out = []
    cost_state = {}  # session -> modelUsage of the last cost-state record
    # Keyed across files: a session copied into a second store path (or a
    # resumed transcript repeating earlier lines) must not count twice.
    calls = {}  # message.id -> event (usage repeats per content block; keep last)
    tools = ToolCalls()
    seen = set()  # line uuids
    for f in files:
        sub = "/subagents/" in f
        for d in jsonl(f):
            if not isinstance(d, dict):
                continue
            if (u := d.get("uuid")) is not None:
                if u in seen:
                    continue
                seen.add(u)
            t = d.get("type")
            sid = d.get("sessionId")
            if t == "cost-state":
                cost_state[sid] = (d.get("timestamp"), d.get("modelUsage") or {})
                continue
            common = dict(harness="claude-code", session_id=sid, ts=d.get("timestamp"),
                          project=project_of(d.get("cwd")), is_subagent=sub or d.get("isSidechain", False))
            msg = d.get("message") if isinstance(d.get("message"), dict) else {}
            content = msg.get("content")
            blocks = [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []
            if t == "assistant":
                model = msg.get("model")
                u = msg.get("usage") or {}
                if model and model != "<synthetic>" and u:
                    cc = u.get("cache_creation") or {}
                    calls[msg.get("id") or d.get("uuid")] = ev(**common, event_type="model_call", name=model,
                        input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0),
                        cache_read=u.get("cache_read_input_tokens", 0),
                        cache_write=u.get("cache_creation_input_tokens", 0),
                        reasoning_tokens=(u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
                        _cw1h=cc.get("ephemeral_1h_input_tokens", 0))
                for b in blocks:
                    if b.get("type") != "tool_use" or not tools.issue(
                            b.get("id"), ev(**common, event_type="tool_call", name=b.get("name"))):
                        continue
                    inp = b.get("input") if isinstance(b.get("input"), dict) else {}
                    if b.get("name") == "Skill" and inp.get("skill"):
                        out.append(ev(**common, event_type="skill", name=skill_name(inp["skill"]), detail="model"))
            elif t == "user":
                if isinstance(content, str):
                    m = re.search(r"<command-name>/?([^<]+)</command-name>", content)
                    if m:
                        name = m.group(1).strip()
                        kind = "skill" if skill_name(name) in skill_names else "command"
                        out.append(ev(**common, event_type=kind,
                                      name=skill_name(name) if kind == "skill" else name, detail="user"))
                else:
                    r = d.get("toolUseResult") if isinstance(d.get("toolUseResult"), dict) else {}
                    dur = (r["durationMs"] if "durationMs" in r
                           else r["durationSeconds"] * 1000 if "durationSeconds" in r else None)
                    for b in blocks:
                        if b.get("type") == "tool_result":
                            tools.result(b.get("tool_use_id"), b.get("is_error"), d.get("timestamp"), dur)
    out += tools.events()

    # Cost: split each session's reported per-model cost across its calls by
    # price weight; calibrate a $/unit rate per model for sessions without one.
    by_sm = defaultdict(list)
    for e in calls.values():
        by_sm[(e["session_id"], e["name"])].append(e)
    project = {e["session_id"]: e["project"] for e in calls.values()}
    reported = {}  # (session, model) -> $
    for sid, (ts, usage) in cost_state.items():
        side = defaultdict(lambda: defaultdict(float))
        for key, u in usage.items():
            if not isinstance(u, dict) or u.get("costUSD") is None:
                continue
            model = model_family(key)
            reported[(sid, model)] = reported.get((sid, model), 0) + u["costUSD"]
            if (sid, model) not in by_sm:
                for k in ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheCreationInputTokens"):
                    side[model][k] += u.get(k) or 0
        # Models with no assistant message (background title generation, web
        # fetch summaries, ...): one event with the session's totals, counted as
        # one call since cost-state has no call count.
        for model, u in side.items():
            by_sm[(sid, model)].append(ev(harness="claude-code", session_id=sid, ts=ts, project=project.get(sid),
                event_type="model_call", name=model, detail="background",
                input_tokens=int(u["inputTokens"]), output_tokens=int(u["outputTokens"]),
                cache_read=int(u["cacheReadInputTokens"]), cache_write=int(u["cacheCreationInputTokens"])))
    for evs in by_sm.values():
        out += evs

    rate_num, rate_den = defaultdict(float), defaultdict(float)
    for (sid, model), evs in by_sm.items():
        cost = reported.get((sid, model))
        total = sum(units(e) for e in evs)
        if cost is None or total == 0:
            continue
        for e in evs:
            e["cost_usd"] = cost * units(e) / total
            e["cost_source"] = "reported"
        if cost > 0:  # subscription (OAuth) sessions report $0
            rate_num[model] += cost
            rate_den[model] += total
    on_disk = {sid for sid, _ in by_sm}
    for (sid, model), (cost, total) in (prior_rates or {}).items():
        if sid not in on_disk and cost > 0 and total > 0:
            rate_num[model] += cost
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

def codex(files):
    out = []
    for f in files:
        sid = cwd = model = None
        # The first token_count with usage already covers the first turn, so
        # deltas start from zero. Totals (not last_token_usage) make repeated
        # token_count events count once.
        prev_total = {}
        tools = ToolCalls()
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
                if tot.get("total_tokens", 0) > prev_total.get("total_tokens", 0):
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
                if not tools.issue(p.get("call_id"), ev(**common, event_type="tool_call", name=p.get("name"))):
                    continue
                args = p.get("arguments") or p.get("input")
                # Codex reads files through its shell tools; apply_patch writes them.
                skill = p.get("name") != "apply_patch" and skill_from_text(
                    args if isinstance(args, str) else json.dumps(args))
                if skill:
                    out.append(ev(**common, event_type="skill", name=skill, detail="read"))
            elif t == "response_item" and pt in ("function_call_output", "custom_tool_call_output"):
                o = p.get("output") if isinstance(p.get("output"), str) else json.dumps(p.get("output"))
                code = re.search(r"(?:exited with code|Exit code:) (\d+)", o)
                wall = re.search(r"Wall time: ([\d.]+) seconds", o)
                tools.result(p.get("call_id"), code and code.group(1) != "0",
                             duration_ms=float(wall.group(1)) * 1000 if wall else None)
        out += tools.events()
    return out


# -------------------------------------------------------------------- pi, OMP

SKILL_URI = re.compile(r"skill://([A-Za-z0-9_.:-]+)")


def pi_like(harness, files):
    """pi and its fork OMP share a session format; OMP adds side-model
    `model_usage` records, `tool_execution_start` timings and `parentSession`."""
    out = []
    for f in files:
        sid = cwd = None
        sub = False
        tools = ToolCalls()
        for d in jsonl(f):
            t = d.get("type")
            if t == "session":
                sid, cwd, sub = d.get("id"), d.get("cwd"), bool(d.get("parentSession"))
                continue
            m = d.get("message") or {}
            common = dict(harness=harness, session_id=sid, ts=d.get("timestamp"),
                          project=project_of(cwd), is_subagent=sub)

            def usage_event(model, u, detail=None):
                cost = (u.get("cost") or {}).get("total")
                return ev(**common, event_type="model_call", name=model, detail=detail,
                    input_tokens=u.get("input", 0), output_tokens=u.get("output", 0),
                    cache_read=u.get("cacheRead", 0), cache_write=u.get("cacheWrite", 0),
                    reasoning_tokens=u.get("reasoning", u.get("reasoningTokens", 0)),
                    cost_usd=cost, cost_source="reported" if cost is not None else None)

            if t == "model_usage" and d.get("usage"):
                out.append(usage_event(d.get("model"), d["usage"], d.get("purpose")))
            elif t == "custom" and d.get("customType") == "tool_execution_start":
                data = d.get("data") or {}
                tools.start(data.get("toolCallId"), data.get("startedAt"))
            elif m.get("role") == "assistant":
                if m.get("usage"):
                    out.append(usage_event(m.get("model"), m["usage"]))
                content = m.get("content")
                for b in content if isinstance(content, list) else []:
                    if isinstance(b, dict) and b.get("type") == "toolCall" and tools.issue(
                            b.get("id"), ev(**common, event_type="tool_call", name=b.get("name"))):
                        args = json.dumps(b.get("arguments"))
                        uri = SKILL_URI.search(args)
                        # A SKILL.md path counts only when read: bash, edit or
                        # write calls touching one do not load the skill.
                        skill = uri.group(1) if uri else b.get("name") == "read" and skill_from_text(args)
                        if skill:
                            out.append(ev(**common, event_type="skill", name=skill, detail="read"))
            elif m.get("role") == "toolResult":
                tools.result(m.get("toolCallId"), m.get("isError"), d.get("timestamp"))
        out += tools.events()
    return out


# --------------------------------------------------------------------- Hermes

def hermes(dbs):
    return [e for db in dbs for e in hermes_db(db)]


def hermes_db(db):
    c = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        return hermes_rows(c)
    finally:
        c.close()


def hermes_rows(c):
    sessions = {r[0]: r[1:] for r in c.execute("select id, cwd, parent_session_id from sessions")}
    out = []

    def common(sid, ts):
        cwd, parent = sessions.get(sid, (None, None))
        return dict(harness="hermes", session_id=sid, ts=iso(ts), project=project_of(cwd),
                    is_subagent=parent is not None)

    # actual_cost_usd defaults to 0 when the provider reports none.
    for (sid, model, task, n, i, o, cr, cw, r, est, act, first) in c.execute(
            """select session_id, model, task, api_call_count, input_tokens, output_tokens,
                      cache_read_tokens, cache_write_tokens, reasoning_tokens,
                      estimated_cost_usd, actual_cost_usd, first_seen from session_model_usage"""):
        out.append(ev(**common(sid, first), event_type="model_call", name=model, detail=task or "main",
            n=n, input_tokens=i, output_tokens=o, cache_read=cr, cache_write=cw, reasoning_tokens=r,
            cost_usd=act or est, cost_source="reported" if act else "estimated"))

    # Messages in order. Hermes writes the same call again with fresh timestamps
    # when compression starts a continuation session, when it rewrites a
    # session's messages, and as duplicate rows: ToolCalls counts the first call
    # and times it to the first result (the copies are written ms apart).
    tools = ToolCalls()
    for sid, role, ts, calls, call_id, content in c.execute(
            """select session_id, role, timestamp, tool_calls, tool_call_id, content from messages
               where role = 'tool' or (tool_calls is not null and tool_calls != '[]') order by id"""):
        if role == "tool":
            try:
                res = json.loads(content or "")
                err = isinstance(res, dict) and (res.get("success") is False or bool(res.get("error")))
            except (json.JSONDecodeError, TypeError):
                err = False
            tools.result(call_id, err, iso(ts))
            continue
        try:
            calls = json.loads(calls)
        except json.JSONDecodeError:
            continue
        for call in calls if isinstance(calls, list) else []:
            fn = call.get("function") or {}
            cid = call.get("id")
            if not tools.issue(cid, ev(**common(sid, ts), event_type="tool_call", name=fn.get("name"))):
                continue
            if fn.get("name") == "skill_view":
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    continue
                # With file_path it reads a supporting file of an already loaded skill.
                if isinstance(args, dict) and args.get("name") and not args.get("file_path"):
                    out.append(ev(**common(sid, ts), event_type="skill",
                                  name=skill_name(args["name"]), detail="model"))
    return out + tools.events()


# ------------------------------------------------------------ skill inventory

# Personal skill roots per harness, with the env overrides the harnesses honour.
SKILL_PATHS = {
    "claude-code": ["${CLAUDE_CONFIG_DIR:-~/.claude}/skills/*/SKILL.md",
                    "${CLAUDE_CONFIG_DIR:-~/.claude}/plugins/**/skills/*/SKILL.md"],
    "hermes": ["${HERMES_HOME:-~/.hermes}/skills/**/SKILL.md",
               "${HERMES_HOME:-~/.hermes}/profiles/*/skills/**/SKILL.md"],
    "pi": ["${PI_CODING_AGENT_DIR:-~/.pi/agent}/skills/**/SKILL.md"],
    "omp": ["~/.omp/agent/skills/**/SKILL.md", "~/.omp/profiles/*/agent/skills/**/SKILL.md"],
    "codex": ["${CODEX_HOME:-~/.codex}/skills/**/SKILL.md"],
}


def claude_projects():
    """Directories Claude Code has run in, from its global config."""
    cfg = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home()).expanduser() / ".claude.json"
    try:
        projects = json.loads(cfg.read_text()).get("projects")
    except (OSError, ValueError, AttributeError):
        return []
    return list(projects) if isinstance(projects, dict) else []


def skill_inventory():
    """Installed skills per harness, so the dashboard can show unused ones."""
    inv = defaultdict(set)
    found = {h: registry.find(pats) for h, pats in SKILL_PATHS.items()}
    found["claude-code"] += [p for d in claude_projects()
                             for p in glob.glob(glob.escape(d) + "/.claude/skills/*/SKILL.md")]
    for harness, paths in found.items():
        for p in paths:
            if "/node_modules/" not in p:
                inv[harness].add(Path(p).parent.name)
    return {k: sorted(v) for k, v in inv.items()}


# Registry id -> (harness name in the events table, store paths the adapter
# reads, adapter(files, installed skills of that harness, prior cost rates)).
# Claude Code's and Codex's prompt-history files and Codex's zstd rollouts are
# listed in the registry but carry no usage we can read.
ADAPTERS = {
    "claude": ("claude-code", lambda p: "history" not in p.lower(), claude_code),
    "codex": ("codex", lambda p: p.endswith("rollout-*.jsonl"), lambda f, *_: codex(f)),
    "pi": ("pi", lambda p: True, lambda f, *_: pi_like("pi", f)),
    "omp": ("omp", lambda p: True, lambda f, *_: pi_like("omp", f)),
    "hermes": ("hermes", lambda p: True, lambda f, *_: hermes(f)),
}


def collect(prior_rates=None):
    """Fresh events from every adapter, and the installed-skill inventory."""
    inv = skill_inventory()
    rows = []
    for rid, (name, keep, adapter) in ADAPTERS.items():
        files = registry.files_for(rid, keep)
        got = adapter(files, set(inv.get(name, ())), prior_rates)
        print(f"{name:12} {len(got):6} events from {len(files)} files")
        rows += got
    return rows, inv


def lit(s):
    """A SQL string literal."""
    return "'" + str(s).replace("'", "''") + "'"


def run(data_dir, machine):
    dest = Path(data_dir) / f"events-{machine}.parquet"
    prior = {}
    if dest.exists():
        # Reported Claude Code costs of earlier ingests keep calibrating the
        # estimates once their transcripts are pruned. 1h cache writes are not
        # stored apart, so they weigh as 5m ones here.
        prior = {(s, m): (c, u) for s, m, c, u in duckdb.sql(f"""
            select session_id, name, sum(cost_usd),
                   sum(input_tokens * {W_IN} + output_tokens * {W_OUT} + cache_read * {W_CR}
                       + cache_write * {W_CW5})::double
            from {lit(dest)} where harness = 'claude-code' and event_type = 'model_call'
              and cost_source = 'reported' group by 1, 2""").fetchall()}
    rows, inv = collect(prior)
    dest = write(data_dir, machine, rows)
    (Path(data_dir) / f"skills-{machine}.json").write_text(json.dumps(inv, indent=2))
    return dest


def run_amp(data_dir, machine):
    from . import amp

    dest = Path(data_dir) / f"events-{machine}.parquet"
    known = set()
    if dest.exists():
        known = {r[0] for r in duckdb.sql(f"select distinct session_id from {lit(dest)}").fetchall()}
    rows, save = amp.collect(data_dir, known)
    dest = write(data_dir, machine, rows)
    save()
    return dest


def write(data_dir, machine, rows):
    """Merge fresh rows into <data dir>/events-<machine>.parquet."""
    for r in rows:
        r["machine"] = machine

    data = Path(data_dir)
    dest = data / f"events-{machine}.parquet"
    con = duckdb.connect()
    data.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as tmp:
        for r in rows:
            tmp.write(json.dumps(r) + "\n")
    # Harnesses prune old transcripts (Claude Code after ~30 days), so keep
    # previously ingested sessions that are no longer on disk.
    keep = ""
    if dest.exists():
        keep = f"""union all by name
          select * from {lit(dest)} p
          where not exists (select 1 from fresh f where f.harness = p.harness
                            and f.session_id is not distinct from p.session_id)"""
    staged = f"{dest}.tmp"
    try:
        con.sql(f"""
            create temp table fresh as select * from (
              select try_cast(ts as timestamptz) as ts, * exclude (ts)
              from read_json({lit(tmp.name)}, format='newline_delimited', columns={{
                ts:'varchar', machine:'varchar', harness:'varchar', session_id:'varchar',
                project:'varchar', event_type:'varchar', name:'varchar', detail:'varchar',
                n:'bigint', input_tokens:'bigint', output_tokens:'bigint', cache_read:'bigint',
                cache_write:'bigint', reasoning_tokens:'bigint', cost_usd:'double',
                cost_source:'varchar', duration_ms:'double', is_error:'boolean', is_subagent:'boolean'}})
            ) where ts is not null
        """)
        con.sql(f"copy (select * from fresh {keep} order by ts) to {lit(staged)} (format parquet)")
        os.replace(staged, dest)
    finally:
        os.unlink(tmp.name)
        if os.path.exists(staged):
            os.unlink(staged)
    total = con.sql(f"select count(*) from {lit(dest)}").fetchall()[0][0]
    print(f"wrote {dest} ({len(rows)} fresh, {total} total events)")
    return dest
