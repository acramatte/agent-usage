"""Agent discovery from the vendored deja-vu registry (see registry/SOURCE)."""
import glob
import json
import os
import re
from importlib.resources import files

REGISTRY = files(__package__) / "registry" / "deja-vu-registry.json"

# Innermost ${VAR} or ${VAR:-default}; nested defaults resolve inside out.
VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^${}]*))?\}")
WINVAR = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")
PLACEHOLDER = re.compile(r"(?:<[^<>]*>)+")
DATE_SEGMENT = re.compile(r"(?<=/)(?:YYYY|MM|DD)(?=/)")
BRACES = re.compile(r"\{([^{}]*)\}")


def load():
    return json.loads(REGISTRY.read_text())["harnesses"]


def entry(harness_id):
    return next(e for e in load() if e["id"] == harness_id)


def expand(pattern):
    """A registry store path as a glob pattern for this machine, or None when it
    names an unset variable without a default or an unresolved location."""
    # Defaults are rescanned (they nest); environment values are not, so a
    # value containing ${...} or %...% cannot expand again or loop forever.
    env = lambda v: v.replace("$", "\0").replace("%", "\1")
    s = pattern
    while m := VAR.search(s):
        value = env(os.environ.get(m.group(1)) or "") or m.group(2)
        if not value:
            return None
        s = s[:m.start()] + value + s[m.end():]
    while m := WINVAR.search(s):
        value = env(os.environ.get(m.group(1)) or "")
        if not value:
            return None
        s = s[:m.start()] + value + s[m.end():]
    s = s.replace("\0", "$").replace("\1", "%")
    if s.startswith("<"):  # <vscode-globalStorage>, <app data>, <project>, ...
        return None
    s = os.path.expanduser(s)
    # Placeholders inside a path stand for a name: <session-id>.jsonl, YYYY/MM/DD.
    s = DATE_SEGMENT.sub("*", PLACEHOLDER.sub("*", s))
    return s if os.path.isabs(s) else None


def braces(pattern):
    m = BRACES.search(pattern)
    if not m:
        return [pattern]
    return [p for alt in m.group(1).split(",")
            for p in braces(pattern[:m.start()] + alt + pattern[m.end():])]


def find(patterns):
    """Files matching any of the registry store paths, in pattern order, once each."""
    seen, out = set(), []
    for pattern in patterns:
        expanded = expand(pattern)
        if not expanded:
            continue
        for g in braces(expanded):
            for f in glob.glob(g, recursive=True):
                key = os.path.realpath(f)
                if key not in seen and os.path.isfile(f):
                    seen.add(key)
                    out.append(f)
    return out


def files_for(harness_id, keep=lambda pattern: True):
    return find(p for p in entry(harness_id)["store_paths"] if keep(p))
