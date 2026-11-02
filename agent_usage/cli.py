"""agent-usage: skills, tool calls and model usage across local coding agents."""
import argparse
import json
import os
import re
import socket
import webbrowser
from pathlib import Path

from . import registry


def default_data_dir():
    if d := os.environ.get("AGENT_USAGE_DATA"):
        return Path(d).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(xdg).expanduser() / "agent-usage"


def agents(args):
    from .ingest import ADAPTERS

    rows = []
    for e in registry.load():
        keep = ADAPTERS[e["id"]][1] if e["id"] in ADAPTERS else (lambda p: True)
        n = len(registry.files_for(e["id"], keep))
        rows.append(dict(id=e["id"], name=e["display_name"], found=n > 0, files=n,
                         ingest=e["id"] in ADAPTERS))
    rows.sort(key=lambda r: (not r["found"], not r["ingest"], r["id"]))
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(f"{'id':14} {'agent':24} {'found':5} {'files':>6}  ingest")
    for r in rows:
        print(f"{r['id']:14} {r['name'][:24]:24} {'yes' if r['found'] else 'no':5} "
              f"{r['files']:>6}  {'yes' if r['ingest'] else 'no'}")


def ingest(args):
    from .ingest import run, run_amp

    (run_amp if args.amp else run)(args.data_dir, args.machine)


def dashboard(args):
    from .dashboard import build

    out = build(args.data_dir, args.out or args.data_dir / "dashboard.html")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


def machine_name(s):
    """--machine names files in the data dir and the dataset: no path parts."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", s):
        raise argparse.ArgumentTypeError(f"not a valid machine name: {s!r}")
    return s


AMP = "ampcode"


def own_files(machine, data_dir):
    """This machine's files, and Amp's on the machine that runs `ingest --amp`
    (it keeps amp-threads.json), so that one pushes them and never pulls them."""
    machines = [machine] + ([AMP] if (data_dir / "amp-threads.json").exists() else [])
    return [f"{kind}-{m}.{ext}" for m in machines
            for kind, ext in (("events", "parquet"), ("skills", "json"))]


SHARED = re.compile(r"(events-[^/]+\.parquet|skills-[^/]+\.json)")


def to_pull(remote, machine, data_dir):
    """Dataset files to download: every other machine's, and this machine's own
    only when missing here (a reinstall). An existing local copy is never older
    than the pushed one, and replacing it would lose sessions since pruned."""
    own = set(own_files(machine, data_dir))
    return [f for f in remote
            if SHARED.fullmatch(f) and not (f in own and (data_dir / f).exists())]


def sync(args):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit("sync needs huggingface_hub: uv tool install 'agent-usage[sync]'")
    if not args.repo:
        raise SystemExit("sync needs --repo <user/dataset> or AGENT_USAGE_HF_REPO")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    # private=True only applies when the dataset is new; an existing one keeps its visibility.
    if not api.repo_info(args.repo, repo_type="dataset").private and not args.allow_public:
        raise SystemExit(f"{args.repo} is public; refusing to sync (pass --allow-public to override)")
    # Push only this machine's files: other machines' copies here may be stale.
    paths = [args.data_dir / n for n in own_files(args.machine, args.data_dir)
             if (args.data_dir / n).exists()]
    if not paths:
        print(f"nothing to push for {args.machine} in {args.data_dir} (run ingest first)")
    for p in paths:
        api.upload_file(path_or_fileobj=p, path_in_repo=p.name, repo_id=args.repo,
                        repo_type="dataset", commit_message=f"agent-usage: {p.name}")
        print(f"pushed {p.name} to {args.repo}")
    remote = api.list_repo_files(args.repo, repo_type="dataset")
    for f in to_pull(remote, args.machine, args.data_dir):
        hf_hub_download(args.repo, f, repo_type="dataset", local_dir=args.data_dir)
        print(f"pulled {f} from {args.repo}")


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="where events-*.parquet live (default: $AGENT_USAGE_DATA "
                             "or $XDG_DATA_HOME/agent-usage)")
    machine = argparse.ArgumentParser(add_help=False)
    machine.add_argument("--machine", type=machine_name,
                         help=f"default: this host's name ('{AMP}' for ingest --amp)")

    ap = argparse.ArgumentParser(prog="agent-usage", description=__doc__)
    sub = ap.add_subparsers(required=True, metavar="command")
    p = sub.add_parser("agents", help="detected agents, session files, ingested yes/no")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=agents)
    p = sub.add_parser("ingest", parents=[common, machine],
                       help="transcripts -> events-<machine>.parquet")
    p.add_argument("--amp", action="store_true",
                   help=f"Amp threads from ampcode.com (needs the amp CLI) -> events-{AMP}.parquet")
    p.set_defaults(fn=ingest)
    p = sub.add_parser("dashboard", parents=[common], help="build the HTML page and open it")
    p.add_argument("--out", type=Path, help="default: <data dir>/dashboard.html")
    p.add_argument("--no-open", action="store_true")
    p.set_defaults(fn=dashboard)
    p = sub.add_parser("sync", parents=[common, machine],
                       help="push this machine's parquet to a Hugging Face dataset, "
                            "pull the other machines'")
    p.add_argument("--repo", default=os.environ.get("AGENT_USAGE_HF_REPO"))
    p.add_argument("--allow-public", action="store_true",
                   help="sync even if the dataset already exists and is public")
    p.set_defaults(fn=sync)
    args = ap.parse_args(argv)
    if "machine" in args and not args.machine:
        args.machine = AMP if getattr(args, "amp", False) else socket.gethostname().split(".")[0]
    args.fn(args)


if __name__ == "__main__":
    main()
