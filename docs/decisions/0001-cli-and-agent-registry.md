# 0001 — Package as a CLI with a vendored agent registry

Status: accepted; work items 1–6 done, 7 open · 2026-11-01

Implementation notes:

- Claude Code's registry id is `claude` (not `claude-code`); the events table
  keeps `harness = 'claude-code'`.
- Adapters skip registry paths they cannot read: Claude Code / Codex
  `history.jsonl`, Codex `*.jsonl.zst`. New roots now read when present:
  Claude `transcripts/`, Codex `archived_sessions/`, OMP and Hermes profiles,
  and every env override.
- Inner placeholders (`<session-id>`, `YYYY/MM/DD`) glob as `*`; paths starting
  with one (`<vscode-globalStorage>`, `<app data>`, `<project>`) and `%VAR%`
  with the variable unset are skipped.
- Fixtures in `tests/fixtures/home` are synthetic, written to match the record
  shapes the adapters parse, not samples cut from real transcripts. Replace
  them with redacted real samples when convenient.
- Refactor check: `main`'s scripts and the CLI produce identical rows on the
  fixture home. On a real machine: ingest with both into separate data dirs and
  run `scripts/compare-events`; differences should be limited to the new roots.
- `push` became `sync` (push this machine's files, pull the others', never
  overwrite this machine's local copy); the extra is `agent-usage[sync]`.

This is a handoff. It is written so an agent that has not seen the original
conversation can pick the work up from here.

## Where things stand

`main` has a working prototype (see `README.md`):

- `ingest.py` reads transcripts for five harnesses (Claude Code, Codex, pi, OMP,
  Hermes) into one events table, `data/events-<host>.parquet`, and merges with
  the previous run so sessions the harness has since pruned are kept.
- `build.py` aggregates that table into a self-contained `out/dashboard.html`
  (model usage, tool calls, skills, filters by range / harness / machine).
- Both run as `uv run <script>` with inline dependencies (DuckDB). `data/` and
  `out/` are gitignored: they hold personal usage data.

Paths are hardcoded per harness, and only agents with an adapter are seen at all.

## Decision

1. **Ship it as a Python CLI installed with `uv tool install`** (`agent-usage`),
   not a compiled binary, for now. Adapters will churn as agent formats change,
   and iteration speed matters more than install friction for the current
   audience. Revisit a single binary (Rust, like funes) only if it is going to
   people without `uv`. The registry and fixtures carry over either way.
2. **Discover agents from a vendored copy of deja-vu's registry**, not by
   scraping its docs page and not by hand-maintained globs.
3. **Keep our own usage adapters.** deja-vu cannot supply usage data (below).

### What was checked in deja-vu

Repo `vshulcz/deja-vu`, MIT, Go, checked at commit
`1eae24f2cf46dadec9513ffcfd1d5b10dadede96` (2026-11-01, release v0.23.3).

- **Registry: reusable.** `docs/registry/registry.json`, `schema_version: 1`,
  43 harnesses. Per entry: `id`, `display_name`, `store_paths` (with env
  overrides like `${CODEX_HOME:-~/.codex}` and profile globs), `format_kind`,
  `parser_source`, `fixture_paths`, `last_verified`, `capabilities`, `gaps`,
  `surfaces`. Covers paths we currently miss: `~/.hermes/profiles/*/state.db`,
  `~/.omp/profiles/*/agent/sessions/**/*.jsonl`.
- **Normalized sessions: not usable for this.** `internal/model/model.go`
  `Message` is `{role, text, time}`; `Session` has no model, tokens, cost or
  tool-call structure. `internal/usage` and `deja stats` count deja's own
  recall/injection activity and session heatmaps, not agent token usage. A
  `--json` export would be the same flattened text funes keeps.
- **Fixtures: not usable as usage tests.** `fixtures/registry/<id>/` samples
  are trimmed to text; the Claude Code, Codex and OMP fixtures contain no usage
  or tool-call fields.

## Work to do

1. **Package.** `pyproject.toml` with a console entry point `agent-usage`;
   move `ingest.py` / `build.py` into a package; ship
   `dashboard.template.html` as package data. Commands:
   ```
   agent-usage agents      # detected harnesses, session counts, adapter yes/no
   agent-usage ingest      # → events-<host>.parquet (existing merge semantics)
   agent-usage dashboard   # build HTML and open it
   agent-usage push        # optional: upload parquet to an HF dataset
   ```
   Keep the output location configurable (default somewhere like
   `~/.local/share/agent-usage/`), not the package directory.
2. **Vendor the registry.** Copy `registry.json` to
   `agent_usage/registry/deja-vu-registry.json` with deja-vu's LICENSE text
   alongside (MIT: keep the copyright notice, "Copyright (c) 2026 Vladislav
   Shulcz"), and a `scripts/update-registry` that fetches a newer one and records
   the commit SHA. Do not depend on it at runtime over the network.
3. **Path expansion.** Implement `${VAR:-default}`, `${VAR}`, `~` and `**`
   globbing from `store_paths`. Drop entries whose variable is unset and has no
   default. Some paths are placeholders (`<vscode-globalStorage>`,
   `<app data>`, `<project>`): skip them for now, or resolve per OS.
4. **`agents` command.** For each registry entry: found or not, number of
   session files/DBs, and whether we have a usage adapter. Map our adapter names
   to registry ids (`claude-code`, `codex`, `pi`, `omp`, `hermes`; check the
   exact ids in the JSON).
5. **Adapters take their roots from the registry**, so profiles and env
   overrides work for free. Make adapter output byte-for-byte unchanged on the
   current machine before and after the refactor (compare event counts and sums
   per harness/event_type, as was done when pi and OMP were merged into
   `pi_like`).
6. **Fixtures.** Build small, redacted samples from real transcripts for each
   adapter (keep usage, model, tool_use/tool_result, skill invocation records;
   replace text content), and a test asserting the exact events produced.
7. **Next adapters, cheapest first:** use `format_kind` / `parser_source` to
   find agents sharing a shape with ours. Claude Code-shaped: Cherry Studio,
   CodeBuddy, Command Code. pi-shaped: gajae-code, Senpi, prime-agent. Verify
   each against a real file before claiming support.

## Things that are easy to get wrong

- **Claude Code deletes transcripts after ~30 days.** Ingest must keep merging
  with the previous parquet; never rebuild from scratch.
- **Claude Code usage repeats per content block** with the same `message.id`;
  keep one per id. Cost comes from the session's last `cost-state` record and is
  split across calls by price weight; sessions without one are estimated from a
  per-model rate calibrated on the others (`cost_source = estimated`).
- **Codex** reports cumulative `total_token_usage`; take deltas. `input_tokens`
  includes cached tokens. No cost at all.
- **Hermes** usage is per session × model × task (`n` = call count), not per call.
  Read `state.db` read-only (`mode=ro`).
- **OMP** has extra `model_usage` records (side models such as `find`) that are
  not in assistant messages, and stores installed skills only as hashed keys, so
  there is no "installed but unused" inventory for it.
- **Subscription (OAuth) models report $0**; that is not free usage. The
  dashboard says so; keep that caveat.
- Never commit `data/`, `out/` or unredacted transcripts.
