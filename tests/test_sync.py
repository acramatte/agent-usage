from agent_usage.cli import to_pull

REMOTE = [".gitattributes", "README.md", "events-mac.parquet", "skills-mac.json",
          "events-box.parquet", "skills-box.json", "sub/events-x.parquet"]


def test_pulls_other_machines_but_never_overwrites_own_local_files(tmp_path):
    (tmp_path / "events-box.parquet").touch()
    (tmp_path / "skills-box.json").touch()
    assert to_pull(REMOTE, "box", tmp_path) == ["events-mac.parquet", "skills-mac.json"]


def test_restores_own_files_missing_locally(tmp_path):
    (tmp_path / "skills-box.json").touch()
    assert to_pull(REMOTE, "box", tmp_path) == [
        "events-mac.parquet", "skills-mac.json", "events-box.parquet"]


def test_amp_files_are_own_on_the_machine_that_ingests_amp(tmp_path):
    remote = REMOTE + ["events-ampcode.parquet"]
    (tmp_path / "events-box.parquet").touch()
    (tmp_path / "skills-box.json").touch()
    (tmp_path / "events-ampcode.parquet").touch()
    assert "events-ampcode.parquet" in to_pull(remote, "box", tmp_path)  # another machine's
    (tmp_path / "amp-threads.json").touch()
    assert to_pull(remote, "box", tmp_path) == ["events-mac.parquet", "skills-mac.json"]
