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
    events = str(data_dir / "events-*.parquet")
    con = duckdb.connect()
    agg = con.sql(f"""
        select strftime(ts::date, '%Y-%m-%d'), machine, harness, event_type, name,
               sum(n)::int, sum(input_tokens)::bigint, sum(output_tokens)::bigint,
               sum(cache_read)::bigint, sum(cache_write)::bigint,
               round(coalesce(sum(cost_usd), 0), 6),
               round(coalesce(sum(cost_usd) filter (where cost_source = 'estimated'), 0), 6),
               count(*) filter (where is_error)::int,
               case when event_type = 'tool_call'
                    then list(round(duration_ms)::int) filter (where duration_ms is not null) end
        from '{events}'
        group by 1, 2, 3, 4, 5 order by 1
    """).fetchall()
    sessions = con.sql(f"""
        select strftime(min(ts)::date, '%Y-%m-%d'), machine, harness
        from '{events}' where session_id is not null group by session_id, machine, harness
    """).fetchall()
    inventory = {}
    for f in glob.glob(str(data_dir / "skills-*.json")):
        for h, names in json.loads(Path(f).read_text()).items():
            inventory.setdefault(h, set()).update(names)
    data = {
        "built": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "agg": agg,
        "sessions": sessions,
        "inventory": {h: sorted(v) for h, v in inventory.items()},
    }
    html = (files(__package__) / "dashboard.template.html").read_text()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("/*__DATA__*/null", blob))
    print(f"wrote {out} ({len(agg)} rows, {len(sessions)} sessions)")
    return out
