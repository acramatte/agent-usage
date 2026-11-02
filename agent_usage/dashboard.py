"""Aggregate <data dir>/events-*.parquet into one self-contained HTML page."""
import glob
import json
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path

import duckdb


def build(data_dir, out):
    data_dir = Path(data_dir)
    if not glob.glob(str(data_dir / "events-*.parquet")):
        raise SystemExit(f"no events-*.parquet in {data_dir}; run `agent-usage ingest` first")
    events = "'" + str(data_dir / "events-*.parquet").replace("'", "''") + "'"
    con = duckdb.connect()
    # Days are local to the machine building the page (DuckDB's session time
    # zone); the page's date math uses the same local calendar.
    agg = con.sql(f"""
        select strftime(ts::date, '%Y-%m-%d'), machine, harness, event_type, name,
               sum(n)::int, sum(input_tokens)::bigint, sum(output_tokens)::bigint,
               sum(cache_read)::bigint, sum(cache_write)::bigint,
               round(coalesce(sum(cost_usd), 0), 6),
               round(coalesce(sum(cost_usd) filter (where cost_source = 'estimated'), 0), 6),
               count(*) filter (where is_error)::int,
               case when event_type = 'tool_call'
                    then list(round(duration_ms)::int) filter (where duration_ms is not null) end
        from {events}
        group by 1, 2, 3, 4, 5 order by 1
    """).fetchall()
    # Every day each session was active, so a range counts the sessions active
    # in it; sessions are numbered rather than named.
    sessions = con.sql(f"""
        select distinct strftime(ts::date, '%Y-%m-%d'), machine, harness,
               dense_rank() over (order by machine, harness, session_id)
        from {events} where session_id is not null
    """).fetchall()
    inventory = {}  # machine -> harness -> installed skills
    for f in glob.glob(str(data_dir / "skills-*.json")):
        inventory[Path(f).stem.removeprefix("skills-")] = json.loads(Path(f).read_text())
    data = {
        "built": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "agg": agg,
        "sessions": sessions,
        "inventory": inventory,
    }
    html = (files(__package__) / "dashboard.template.html").read_text()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", blob))
    print(f"wrote {out} ({len(agg)} rows, {len({s[3] for s in sessions})} sessions)")
    return out
