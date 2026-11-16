import json
import shutil
import sys
import types

from agent_usage import amp, cli
from agent_usage.cli import own_files, to_pull

REMOTE = [".gitattributes", "README.md", "events-mac.parquet", "skills-mac.json",
          "events-box.parquet", "skills-box.json", "sub/events-x.parquet"]
AMP_FILES = ["events-ampcode.parquet", "skills-ampcode.json"]


def plan(machine, data_dir, owners, take_over=False):
    """What sync would do: files to push, files to pull, owners to write back."""
    own, owners = own_files(machine, data_dir, owners, take_over)
    return [f for f in own if (data_dir / f).exists()], to_pull(REMOTE + AMP_FILES[:1], own, data_dir), owners


def test_pulls_other_machines_but_never_overwrites_own_local_files(tmp_path):
    (tmp_path / "events-box.parquet").touch()
    (tmp_path / "skills-box.json").touch()
    own, _ = own_files("box", tmp_path, {})
    assert to_pull(REMOTE, own, tmp_path) == ["events-mac.parquet", "skills-mac.json"]


def test_restores_own_files_missing_locally(tmp_path):
    (tmp_path / "skills-box.json").touch()
    own, _ = own_files("box", tmp_path, {})
    assert to_pull(REMOTE, own, tmp_path) == [
        "events-mac.parquet", "skills-mac.json", "events-box.parquet"]


def test_one_machine_owns_amp_files(tmp_path):
    box, mac = tmp_path / "box", tmp_path / "mac"
    for d in (box, mac):
        d.mkdir()
        (d / "amp-threads.json").touch()
        (d / "events-ampcode.parquet").touch()

    # The first machine that ingests Amp and syncs claims Amp's files: it pushes
    # them, with the thread records that describe them.
    push, pull, owners = plan("box", box, {})
    assert owners == dict.fromkeys(AMP_FILES, "box")
    assert {"events-ampcode.parquet", "amp-threads.json"} <= set(push)
    assert "events-ampcode.parquet" not in pull

    # Another machine that also ingests Amp neither pushes them nor keeps its copy...
    push, pull, after = plan("mac", mac, owners)
    assert after == owners
    assert not {"events-ampcode.parquet", "amp-threads.json"} & set(push)
    assert "events-ampcode.parquet" in pull
    # ...unless it takes them over.
    push, pull, after = plan("mac", mac, owners, take_over=True)
    assert after == dict.fromkeys(AMP_FILES, "mac") and "events-ampcode.parquet" in push

    # A machine that never ingested Amp can take over too (sync adopts the records).
    (tmp_path / "new").mkdir()
    assert plan("new", tmp_path / "new", owners, take_over=True)[2] == dict.fromkeys(AMP_FILES, "new")

    # The owner stays the owner without amp-threads.json: its newer copy is not pulled over.
    (box / "amp-threads.json").unlink()
    push, pull, after = plan("box", box, owners)
    assert after == owners
    assert "events-ampcode.parquet" in push and "events-ampcode.parquet" not in pull


def fake_hub(monkeypatch, remote):
    """huggingface_hub backed by the directory `remote`."""
    class HfApi:
        def create_repo(self, *a, **k): pass
        def repo_info(self, *a, **k): return types.SimpleNamespace(private=True)
        def list_repo_files(self, *a, **k): return sorted(p.name for p in remote.iterdir())
        def upload_file(self, path_or_fileobj, path_in_repo, **k):
            data = path_or_fileobj if isinstance(path_or_fileobj, bytes) else path_or_fileobj.read_bytes()
            (remote / path_in_repo).write_bytes(data)

    def hf_hub_download(repo, f, repo_type=None, local_dir=None, force_download=False):
        dest = (local_dir or remote.parent / "cache") / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(remote / f, dest)
        return str(dest)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=HfApi, hf_hub_download=hf_hub_download))


def test_take_over_adopts_the_previous_owners_events_and_records(tmp_path, monkeypatch):
    remote, desk = tmp_path / "remote", tmp_path / "desk"
    remote.mkdir(), desk.mkdir()
    fake_hub(monkeypatch, remote)
    (remote / "owners.json").write_text(json.dumps(dict.fromkeys(AMP_FILES, "laptop")))
    (remote / "events-ampcode.parquet").write_text("laptop's events")
    laptop_records = {"T-1": dict(updated="u1", state="complete", parser=amp.PARSER)}
    (remote / "amp-threads.json").write_text(json.dumps(laptop_records))
    (desk / "events-ampcode.parquet").write_text("desk's events")
    (desk / "amp-threads.json").write_text(json.dumps({"T-2": dict(updated="u1", state="complete",
                                                                  parser=amp.PARSER)}))

    def sync(*extra):
        cli.main(["sync", "--repo", "u/d", "--machine", "desk", "--data-dir", str(desk), *extra])

    sync()  # not the owner: nothing of Amp's is pushed
    assert (remote / "amp-threads.json").read_text() == json.dumps(laptop_records)
    (desk / "events-ampcode.parquet").write_text("desk's events")  # desk ingests Amp again
    sync("--take-over")
    assert (desk / "events-ampcode.parquet").read_text() == "laptop's events"
    assert json.loads((desk / "amp-threads.json").read_text()) == laptop_records
    assert json.loads((remote / "owners.json").read_text()) == dict.fromkeys(AMP_FILES, "desk")


def test_take_over_without_pushed_records_reads_every_thread_again(tmp_path, monkeypatch):
    remote, desk = tmp_path / "remote", tmp_path / "desk"
    remote.mkdir(), desk.mkdir()
    fake_hub(monkeypatch, remote)
    (remote / "owners.json").write_text(json.dumps(dict.fromkeys(AMP_FILES, "laptop")))
    (remote / "events-ampcode.parquet").write_text("laptop's events")
    (desk / "amp-threads.json").write_text(json.dumps({"T-1": "u1"}))  # also the pre-record format
    cli.main(["sync", "--repo", "u/d", "--machine", "desk", "--data-dir", str(desk), "--take-over"])
    records = amp.load_records(desk / "amp-threads.json")
    assert amp.thread_state(records["T-1"], {"updated": "u1"}) == "stale"
    assert records["T-1"]["state"] == "complete"  # its rows keep their cost until it is read
