#!/usr/bin/env python3
"""Make an extra instance of a benchmark service, so two runs on the same service can overlap
(Decision Log D-84). The original harness is never touched: the copy is a separate folder.

    .venv/bin/python scripts/make_service_copy.py fastapi b --offset 100

Creates service_copies/harness-<service>-<name>/ (a copy of harness/) and changes only:
  - the Docker Compose project name (fastapi -> fastapi-b), fixed container names and named volumes
    (suffix _b), so containers/volumes do not collide;
  - localhost:<port> in the service's own scripts (start/reset health checks), to the shifted ports;
  - every host port in the service's compose files (+offset), container ports unchanged;
  - the service's port in the copy's src/config.py (+offset).
Runs use it with PRETRAINED_HARNESS_ROOT=<copy> (the queue sets this per job).
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "harness"


def shift_ports(text: str, offset: int, mapping: dict | None = None) -> str:
    """'- "8000:8000"' / '- 9323:9323' -> host side + offset (only inside ports: lists).
    Records old -> new host ports in `mapping`."""
    out, in_ports = [], False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("ports:"):
            in_ports = True
        elif in_ports and not stripped.startswith("-"):
            in_ports = False
        if in_ports and stripped.startswith("-"):
            def repl(m):
                if mapping is not None:
                    mapping[int(m.group(2))] = int(m.group(2)) + offset
                return f"{m.group(1)}{int(m.group(2)) + offset}:{m.group(3)}"
            line = re.sub(r'(["\']?)(\d+):(\d+)', repl, line, count=1)
        out.append(line)
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("service")
    ap.add_argument("name")
    ap.add_argument("--offset", type=int, default=100)
    a = ap.parse_args()
    dest = ROOT / "service_copies" / f"harness-{a.service}-{a.name}"
    if dest.exists():
        print(f"exists: {dest}")
        return 0
    dest.parent.mkdir(exist_ok=True)
    # working/ copies can contain root-owned build files; give them back to us before copying
    for sub in ("working", "original"):
        p = HARNESS / "services" / a.service / sub
        subprocess.run(["docker", "run", "--rm", "-v", f"{p}:/w", "--entrypoint", "chown", "postgres:17", "-R",
                        f"{__import__('os').getuid()}:{__import__('os').getgid()}", "/w"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.copytree(HARNESS, dest, symlinks=True, ignore=shutil.ignore_patterns("__pycache__", ".git", "results"))
    svc = dest / "services" / a.service
    changed, ports, containers = [], {}, {}
    for sub in ("original", "working", "."):            # TodoApp keeps its compose file at the service root
        for f in sorted((svc / sub).glob("docker-compose*.yml")):
            t = f.read_text()
            containers.update((n, f"{n}_{a.name}") for n in re.findall(r"^\s+container_name:\s*(\S+)\s*$", t, flags=re.M))
            new = re.sub(rf"^name:\s*{re.escape(a.service)}\s*$", f"name: {a.service}-{a.name}", t, flags=re.M)
            # fixed container names and named volumes would be shared with the original: suffix them
            new = re.sub(r"^(\s+container_name:\s*)(\S+)\s*$", rf"\g<1>\g<2>_{a.name}", new, flags=re.M)
            new = re.sub(r"^(\s{4,}name:\s*)(\S+)\s*$", rf"\g<1>\g<2>_{a.name}", new, flags=re.M)
            new = shift_ports(new, a.offset, ports)
            if new != t:
                f.write_text(new)
                changed.append(str(f.relative_to(dest)))
    # the service's own scripts may wait for localhost:<port> (health checks): point them at the copy
    for f in sorted(svc.glob("*.sh")):
        t = f.read_text()
        new = re.sub(r"localhost:(\d+)", lambda m: f"localhost:{ports.get(int(m.group(1)), int(m.group(1)))}", t)
        for old, renamed in containers.items():         # scripts that name a container (TodoApp's reset.sh)
            new = re.sub(rf"\b{re.escape(old)}\b(?!_)", renamed, new)
        if new != t:
            f.write_text(new)
            changed.append(str(f.relative_to(dest)))
    cfg = dest / "src" / "config.py"
    t = cfg.read_text()
    m = re.search(rf'("{re.escape(a.service)}": ServiceConfig\(\s*name="{re.escape(a.service)}",\s*port=)(\d+)', t)
    if not m:
        print("could not find the service port in config.py", file=sys.stderr)
        return 1
    port = int(m.group(2)) + a.offset
    cfg.write_text(t[:m.start(2)] + str(port) + t[m.end(2):])
    changed.append(f"src/config.py ({a.service} port {m.group(2)} -> {port})")
    print(f"created {dest}\nchanged:\n  " + "\n  ".join(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
