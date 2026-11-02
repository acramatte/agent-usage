import json
import re
import shutil
import sqlite3
from pathlib import Path

import duckdb
import pytest

from agent_usage import amp, ingest, registry

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A copy of fixtures/home as $HOME, with every variable our adapters'
    registry paths read unset, so the defaults (~/.claude, ...) apply."""
    home = tmp_path / "home"
    shutil.copytree(FIXTURES / "home", home)
    db = sqlite3.connect(home / ".hermes/state.db")
    db.executescript((home / ".hermes/state.sql").read_text())
    db.commit()
    db.close()
    # A project Claude Code has run in, with a project-level skill.
    (home / "work/demo/.claude/skills/ship").mkdir(parents=True)
    (home / "work/demo/.claude/skills/ship/SKILL.md").write_text("---\nname: ship\n---\n")
    (home / ".claude.json").write_text(json.dumps({"projects": {str(home / "work/demo"): {}}}))
    monkeypatch.setenv("HOME", str(home))
    paths = [p for rid in ingest.ADAPTERS for p in registry.entry(rid)["store_paths"]]
    for p in paths + [p for ps in ingest.SKILL_PATHS.values() for p in ps]:
        for var in re.findall(r"\$\{(\w+)", p):
            if var != "HOME":
                monkeypatch.delenv(var, raising=False)
    return home


def E(harness, session_id, project, ts, event_type, name, **kw):
    return dict(ts=ts, machine=None, harness=harness, session_id=session_id, project=project,
                event_type=event_type, name=name, detail=None, n=1, input_tokens=0,
                output_tokens=0, cache_read=0, cache_write=0, reasoning_tokens=0,
                cost_usd=None, cost_source=None, duration_ms=None, is_error=None,
                is_subagent=False) | kw


def normalized(events):
    rows = [{k: round(v, 9) if isinstance(v, float) else v for k, v in e.items()} for e in events]
    return sorted(rows, key=lambda r: tuple(str(v) for v in r.values()))


# Claude Code cost: session A reports $0.50 for claude-sonnet-4-5 (keyed
# claude-sonnet-4-5[1m] in cost-state, as 1M-context sessions are), split by price
# weight (input 1, output 5, cache read 0.1, 5m write 1.25, 1h write 2):
#   msg_1: 10 + 40*5 + 100*0.1 + 30*1.25 + 20*2 = 297.5
#   msg_2: 4 + 2*5 = 14
# Session B has no cost-state; it is estimated at A's rate: 100 + 10*5 = 150 units.
A_RATE = 0.5 / 311.5

CC = dict(harness="claude-code", project="demo")
EXPECTED = [
    E(**CC, session_id="A", ts="2026-10-01T10:00:00.000Z", event_type="skill", name="review", detail="user"),
    E(**CC, session_id="A", ts="2026-10-01T10:00:01.000Z", event_type="command", name="clear", detail="user"),
    # A project-level skill; a Hermes skill's name is still a command here.
    E(**CC, session_id="A", ts="2026-10-01T10:00:01.200Z", event_type="skill", name="ship", detail="user"),
    E(**CC, session_id="A", ts="2026-10-01T10:00:01.300Z", event_type="command", name="triage", detail="user"),
    E(**CC, session_id="A", ts="2026-10-01T10:00:03.000Z", event_type="model_call", name="claude-sonnet-4-5",
      input_tokens=10, output_tokens=40, cache_read=100, cache_write=50,
      cost_usd=297.5 * A_RATE, cost_source="reported"),
    E(**CC, session_id="A", ts="2026-10-01T10:00:03.000Z", event_type="tool_call", name="Bash",
      is_error=False, duration_ms=1234),
    E(**CC, session_id="A", ts="2026-10-01T10:00:06.000Z", event_type="model_call", name="claude-sonnet-4-5",
      input_tokens=4, output_tokens=2, cost_usd=14 * A_RATE, cost_source="reported"),
    E(**CC, session_id="A", ts="2026-10-01T10:00:06.000Z", event_type="tool_call", name="Skill",
      is_error=True, duration_ms=2000.0),
    # Invoked as anthropic-skills:pdf; installed as .../skills/pdf.
    E(**CC, session_id="A", ts="2026-10-01T10:00:06.000Z", event_type="skill", name="pdf", detail="model"),
    # Background model in cost-state only: its totals, as one call.
    E(**CC, session_id="A", ts="2026-10-01T10:00:10.000Z", event_type="model_call", name="claude-haiku-4-5",
      detail="background", input_tokens=300, output_tokens=20, cost_usd=0.0004, cost_source="reported"),
    E(**CC, session_id="B", ts="2026-10-01T11:00:00.000Z", event_type="model_call", name="claude-sonnet-4-5",
      input_tokens=100, output_tokens=10, cost_usd=150 * A_RATE, cost_source="estimated", is_subagent=True),

    # Codex: deltas of the running totals, the first one included; a repeated
    # token_count adds nothing, and cached tokens are taken out of input_tokens.
    E("codex", "C1", "api", "2026-10-01T09:00:01.000Z", "model_call", "gpt-5-codex",
      input_tokens=60, cache_read=40, output_tokens=10, reasoning_tokens=5),
    E("codex", "C1", "api", "2026-10-01T09:00:02.000Z", "tool_call", "shell", is_error=False, duration_ms=500.0),
    E("codex", "C1", "api", "2026-10-01T09:00:02.000Z", "skill", "deploy", detail="read"),
    E("codex", "C1", "api", "2026-10-01T09:00:03.000Z", "model_call", "gpt-5-codex",
      input_tokens=40, cache_read=160, output_tokens=20, reasoning_tokens=10),
    E("codex", "C1", "api", "2026-10-01T09:00:04.000Z", "tool_call", "apply_patch", is_error=True),

    E("pi", "P1", "cli", "2026-10-02T08:00:01.000Z", "model_call", "claude-opus-4-1",
      input_tokens=7, output_tokens=3, cache_read=50, cache_write=5, cost_usd=0.01, cost_source="reported"),
    E("pi", "P1", "cli", "2026-10-02T08:00:01.000Z", "tool_call", "read", is_error=False, duration_ms=2000.0),
    E("pi", "P1", "cli", "2026-10-02T08:00:01.000Z", "skill", "brainstorm", detail="read"),

    E("omp", "O1", "cli", "2026-10-02T09:00:00.000Z", "model_call", "find-model", detail="find",
      input_tokens=20, output_tokens=2, reasoning_tokens=1, cost_usd=0.002, cost_source="reported",
      is_subagent=True),
    E("omp", "O1", "cli", "2026-10-02T09:00:01.000Z", "model_call", "gpt-5",
      input_tokens=30, output_tokens=6, reasoning_tokens=4, cost_usd=0.03, cost_source="reported",
      is_subagent=True),
    E("omp", "O1", "cli", "2026-10-02T09:00:01.000Z", "tool_call", "read", is_error=True,
      duration_ms=500.0, is_subagent=True),
    E("omp", "O1", "cli", "2026-10-02T09:00:01.000Z", "skill", "tdd", detail="read", is_subagent=True),

    E("hermes", "H1", "bot", "2026-09-21T14:13:20+00:00", "model_call", "claude-sonnet-4-5", detail="main",
      n=3, input_tokens=100, output_tokens=20, cache_read=10, cache_write=5,
      cost_usd=0.02, cost_source="estimated"),
    E("hermes", "H2", "bot", "2026-09-21T14:15:00+00:00", "model_call", "gpt-5", detail="compress",
      input_tokens=10, output_tokens=1, reasoning_tokens=2, cost_usd=0.0015, cost_source="reported",
      is_subagent=True),
    E("hermes", "H1", "bot", "2026-09-21T14:13:30+00:00", "tool_call", "skill_view",
      is_error=True, duration_ms=2500.0),
    E("hermes", "H1", "bot", "2026-09-21T14:13:30+00:00", "skill", "triage", detail="model"),
    E("hermes", "H1", "bot", "2026-09-21T14:13:33+00:00", "tool_call", "skill_view",
      is_error=False, duration_ms=1000.0),
]


def test_fixture_events(home):
    events, inventory = ingest.collect()
    assert normalized(events) == normalized(EXPECTED)
    assert inventory == {"claude-code": ["review", "ship"], "hermes": ["triage"]}


def test_ingest_keeps_sessions_pruned_from_disk(home, tmp_path):
    data = tmp_path / "data"
    dest = ingest.run(data, "m1")
    shutil.rmtree(home / ".codex/sessions")
    ingest.run(data, "m1")
    counts = dict(duckdb.sql(f"select harness, count(*) from '{dest}' group by 1").fetchall())
    # Codex's session is gone from disk but kept; the others are replaced, not doubled.
    assert counts == {"claude-code": 11, "codex": 5, "pi": 3, "omp": 4, "hermes": 5}


def test_estimates_survive_pruning_of_reporting_session(home, tmp_path):
    data = tmp_path / "data"
    ingest.run(data, "m1")
    # A (with the cost-state) is pruned; B is still on disk and re-ingested.
    (home / ".claude/projects/-work-demo/A.jsonl").unlink()
    shutil.rmtree(home / ".claude/transcripts")
    dest = ingest.run(data, "m1")
    # A's stored calls calibrate the rate; 1h cache writes weigh as 5m ones
    # there (msg_1: 297.5 - 20*(2 - 1.25) = 282.5).
    got = duckdb.sql(f"select cost_usd, cost_source from '{dest}' where session_id = 'B'").fetchall()
    assert got == [(pytest.approx(150 * 0.5 / 296.5), "estimated")]


def test_registry_paths(tmp_path, monkeypatch):
    for d in ["h/state.db", "h/profiles/work/state.db", "h/profiles/x/notes.txt"]:
        (tmp_path / d).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / d).touch()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "h"))
    monkeypatch.delenv("DEJA_HERMES_HOME", raising=False)
    monkeypatch.delenv("DEJA_HERMES_PROFILES_ROOT", raising=False)
    monkeypatch.delenv("DEJA_HERMES_DB", raising=False)
    assert registry.files_for("hermes") == [str(tmp_path / "h/state.db"),
                                            str(tmp_path / "h/profiles/work/state.db")]

    monkeypatch.setenv("HOME", "/u")
    monkeypatch.delenv("CLINE_SESSION_DATA_DIR", raising=False)
    monkeypatch.delenv("CLINE_DATA_DIR", raising=False)
    monkeypatch.setenv("CLINE_DIR", "/c")
    expand = registry.expand
    assert expand("${CLINE_SESSION_DATA_DIR:-${CLINE_DATA_DIR:-${CLINE_DIR:-~/.cline}/data}/sessions}/*/x") \
        == "/c/data/sessions/*/x"
    monkeypatch.delenv("CLINE_DIR")
    assert expand("${CLINE_DATA_DIR:-${CLINE_DIR:-~/.cline}/data}") == "/u/.cline/data"
    assert expand("${CLINE_DIR}/x") is None
    assert expand("<vscode-globalStorage>/a/tasks/*/h.json") is None
    assert expand("~/.minimax/v2/sessions/YYYY/MM/DD/<session>/messages.jsonl") \
        == "/u/.minimax/v2/sessions/*/*/*/*/messages.jsonl"
    assert expand("cli_sessions.db beside sessions.db (the legacy name)") is None
    assert registry.braces("/a/{json,jsonl}") == ["/a/json", "/a/jsonl"]


# Amp: model calls from the export; the usage report's extra requests per model
# become one subagent row each, with the thread's unseen cache reads (2,900 - 1,900)
# shared by their input (sol 2,890, terra 500). Each model's Amp cost plus list
# price is split by price weight across its rows.
SOL_UNITS = dict(c1=50 * 5 + 900 * 0.1 + 100 * 1.25, c2=10 + 20 * 5 + 1000 * 0.1,
                 rest=2037 + 20 * 5 + 853 * 0.1)
SOL = 0.15 / sum(SOL_UNITS.values())
AMP = dict(harness="amp", session_id="T-1", project="foglio")
AMP_EXPECTED = [
    E(**AMP, ts="2026-10-09T10:00:02.000Z", event_type="model_call", name="gpt-6.1-sol",
      output_tokens=50, cache_read=900, cache_write=100, cost_usd=SOL_UNITS["c1"] * SOL,
      cost_source="estimated"),
    E(**AMP, ts="2026-10-09T10:00:02.000Z", event_type="tool_call", name="shell_command",
      is_error=False, duration_ms=3000.0),
    E(**AMP, ts="2026-10-09T10:00:02.000Z", event_type="tool_call", name="skill",
      is_error=True, duration_ms=3000.0),
    E(**AMP, ts="2026-10-09T10:00:02.000Z", event_type="skill", name="funes", detail="model"),
    E(**AMP, ts="2026-10-09T10:00:06.000Z", event_type="model_call", name="gpt-6.1-sol",
      input_tokens=10, output_tokens=20, cache_read=1000, cost_usd=SOL_UNITS["c2"] * SOL,
      cost_source="estimated"),
    E(**AMP, ts="2026-10-09T10:00:02.000Z", event_type="model_call", name="gpt-6.1-sol",
      detail="subagent", n=3, input_tokens=2037, cache_read=853, output_tokens=20,
      cost_usd=SOL_UNITS["rest"] * SOL, cost_source="estimated", is_subagent=True),
    E(**AMP, ts="2026-10-09T10:00:02.000Z", event_type="model_call", name="gpt-5.6-terra",
      detail="subagent", n=2, input_tokens=353, cache_read=147, output_tokens=10,
      cost_usd=0.02, cost_source="estimated", is_subagent=True),
    E(**AMP, ts="2026-10-09T10:00:02.000Z", event_type="model_call", name="gpt-transcribe",
      detail="subagent", n=1, cost_usd=0.001, cost_source="reported", is_subagent=True),
]


def amp_fixture():
    thread = json.loads((FIXTURES / "amp/T-1.json").read_text())
    return thread, (FIXTURES / "amp/T-1.usage.md").read_text()


def test_amp_thread_events(capsys):
    assert normalized(amp.thread_events(*amp_fixture())) == normalized(AMP_EXPECTED)
    # Without the usage report: the messages alone, no cost.
    thread, _ = amp_fixture()
    assert normalized(amp.thread_events(thread, None)) == normalized(
        [e | dict(cost_usd=None, cost_source=None) for e in AMP_EXPECTED if e["detail"] != "subagent"])
    assert capsys.readouterr().err == ""

    # Report variants seen in practice: a model name without the export's prefix or
    # as a '-latest' alias, and 'Cost' for 'Amp cost'.
    thread, usage = amp_fixture()
    for name in ["6.1 Sol", "~openai/gpt-6.1-sol-latest"]:
        variant = usage.replace("| GPT-6.1 Sol |", f"| {name} |").replace("| Amp cost |", "| Cost |")
        assert normalized(amp.thread_events(thread, variant)) == normalized(AMP_EXPECTED)
    assert capsys.readouterr().err == ""

    # The report changed: a renamed column; a model name the export does not match.
    amp.thread_events(thread, usage.replace("| Requests |", "| Calls |"))
    assert "T-1: usage report's Models table has unexpected columns ['Model', 'Calls'," \
        in capsys.readouterr().err
    amp.thread_events(thread, usage.replace("| GPT-6.1 Sol |", "| GPT-6.1 Sol Pro |"))
    err = capsys.readouterr().err
    assert "T-1: input tokens: events add up to 7,410, the usage report says 5,400" in err


def test_amp_ingest_retries_rate_limited_threads(tmp_path, monkeypatch):
    thread, usage = amp_fixture()
    listed = [{"id": "T-1", "updated": "u1"}, {"id": "T-2", "updated": "u1"}]
    calls, limit = [], {"T-2"}

    def fake(*args):
        calls.append(args[:2] + args[-1:])
        if args[:2] == ("threads", "list"):
            return json.dumps(listed)
        if args[1] == "export":
            return json.dumps(thread | {"id": args[-1]})
        if args[-1] in limit:
            raise amp.RateLimited("Usage information is limited to 60 requests per hour.")
        return usage

    monkeypatch.setattr(amp, "amp", fake)
    monkeypatch.setattr(amp.shutil, "which", lambda _: "/bin/amp")
    monkeypatch.setattr(amp, "WORKERS", 1)  # T-1's report is fetched before T-2 hits the limit
    data = tmp_path / "data"

    def run():
        calls.clear()
        dest = ingest.run_amp(data, "ampcode")
        return {c[2] for c in calls if c[1] == "export"}, dict(duckdb.sql(
            f"select session_id, count(*) from '{dest}' group by 1").fetchall())

    # T-2 is ingested without its report, and fetched again next run; T-1 is not.
    assert run() == ({"T-1", "T-2"}, {"T-1": 8, "T-2": 5})
    limit.clear()
    assert run() == ({"T-2"}, {"T-1": 8, "T-2": 8})
    assert run() == (set(), {"T-1": 8, "T-2": 8})
    listed[0]["updated"] = "u2"
    assert run()[0] == {"T-1"}
