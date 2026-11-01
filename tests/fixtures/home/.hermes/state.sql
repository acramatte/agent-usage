-- Built into state.db by the tests (and by hand: sqlite3 state.db < state.sql).
create table sessions (id text primary key, cwd text, parent_session_id text);
insert into sessions values ('H1', '/work/bot', null), ('H2', '/work/bot', 'H1');
create table session_model_usage (
  session_id text, model text, task text, api_call_count integer,
  input_tokens integer, output_tokens integer, cache_read_tokens integer,
  cache_write_tokens integer, reasoning_tokens integer,
  estimated_cost_usd real, actual_cost_usd real, first_seen real);
insert into session_model_usage values
  ('H1', 'claude-sonnet-4-5', null, 3, 100, 20, 10, 5, 0, 0.02, null, 1790000000.0),
  ('H2', 'gpt-5', 'compress', 1, 10, 1, 0, 0, 2, 0.001, 0.0015, 1790000100.0);
create table messages (
  id integer primary key autoincrement, session_id text not null, role text not null,
  content text, tool_call_id text, tool_calls text, timestamp real not null);
insert into messages (session_id, role, content, tool_call_id, tool_calls, timestamp) values
  ('H1', 'user', '(redacted)', null, null, 1790000005.0),
  ('H1', 'assistant', '(redacted)', null, '[]', 1790000006.0),
  ('H1', 'assistant', null, null, '[{"id":"h1","type":"function","function":{"name":"skill_view","arguments":"{\"name\":\"ops:triage\"}"}}]', 1790000010.0),
  ('H1', 'tool', '{"success": false, "error": "(redacted)"}', 'h1', null, 1790000012.5);
