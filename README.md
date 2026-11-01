# agent-usage

Skills, tool calls and model usage across local coding agents, in one table and one page.

```sh
uv run ingest.py   # transcripts -> data/events-<machine>.parquet (+ skills-<machine>.json)
uv run build.py    # data/*.parquet -> out/dashboard.html (self-contained, open it directly)
```

## Sources

| Harness | Read from | Model usage | Cost | Tools | Skills |
|---|---|---|---|---|---|
| Claude Code | `~/.claude/projects/**/*.jsonl` (incl. subagents) | per API call | session `cost-state`, split by price weight; estimated when absent | `tool_use` / `tool_result` | `Skill` tool, slash commands matching an installed skill |
| Codex | `~/.codex/sessions/**/*.jsonl` | `token_count` deltas | none | `function_call`, `custom_tool_call` | reads of `…/skills/<name>/SKILL.md` |
| pi | `~/.pi/agent/sessions/**/*.jsonl` | per message | reported per message | `toolCall` / `toolResult` | reads of `SKILL.md` |
| Hermes | `~/.hermes/state.db` (read-only) | `session_model_usage` (per session × model × task) | actual, else estimated | `messages.tool_calls` | `skill_view` |

Hermes is read from its local database rather than Langfuse: same data, no API keys.

## Schema (`events-*.parquet`)

One row per event: `ts, machine, harness, session_id, project, event_type`
(`model_call | tool_call | skill | command`), `name` (model / tool / skill), `detail`,
`n` (call count; >1 only for Hermes aggregates), token columns, `cost_usd`, `cost_source`
(`reported | estimated`), `duration_ms`, `is_error`, `is_subagent`.

Re-running `ingest.py` replaces sessions still on disk and keeps the ones a harness has
since pruned (Claude Code deletes transcripts after ~30 days), so run it regularly.

## Multiple machines

Each machine writes its own `events-<hostname>.parquet`. Put them in one place
(e.g. an HF dataset next to `acramatte/funes-memory`) and `build.py` reads them all
from `data/`.
