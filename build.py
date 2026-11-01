# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1"]
# ///
"""Aggregate data/events-*.parquet into a self-contained out/dashboard.html."""
import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).parent
EVENTS = str(ROOT / "data/events-*.parquet")


def rows(sql):
    rel = duckdb.sql(sql)
    return rel.fetchall()


def main():
    agg = rows(f"""
        select strftime(ts::date, '%Y-%m-%d'), machine, harness, event_type, name,
               sum(n)::int, sum(input_tokens)::bigint, sum(output_tokens)::bigint,
               sum(cache_read)::bigint, sum(cache_write)::bigint,
               round(coalesce(sum(cost_usd), 0), 6),
               round(coalesce(sum(cost_usd) filter (where cost_source = 'estimated'), 0), 6),
               count(*) filter (where is_error)::int,
               case when event_type = 'tool_call'
                    then list(round(duration_ms)::int) filter (where duration_ms is not null) end
        from '{EVENTS}'
        group by 1, 2, 3, 4, 5 order by 1
    """)
    sessions = rows(f"""
        select strftime(min(ts)::date, '%Y-%m-%d'), machine, harness
        from '{EVENTS}' where session_id is not null group by session_id, machine, harness
    """)
    inventory = {}
    for f in glob.glob(str(ROOT / "data/skills-*.json")):
        for h, names in json.loads(Path(f).read_text()).items():
            inventory.setdefault(h, set()).update(names)
    data = {
        "built": datetime.now(timezone.utc).isoformat(timespec="minutes"),
        "agg": agg,
        "sessions": sessions,
        "inventory": {h: sorted(v) for h, v in inventory.items()},
    }
    html = (ROOT / "dashboard.template.html").read_text()
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / "dashboard.html").write_text(html.replace("/*__DATA__*/null", blob))
    print(f"wrote {out / 'dashboard.html'} ({len(agg)} rows, {len(sessions)} sessions)")


if __name__ == "__main__":
    main()
