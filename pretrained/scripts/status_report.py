#!/usr/bin/env python3
"""One-screen status of the benchmark runs (times in IST).

    .venv/bin/python scripts/status_report.py [--since-hours 1]

Shows: running jobs (instance, elapsed, progress, ETA of the last generated test and of the
whole job), the server session end, jobs finished in the last N hours with their status and score,
and any job that cannot finish before the session ends.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IST = timezone(timedelta(hours=5, minutes=30))
TOTAL = {"fastapi": 204, "nestjs-realworld": 94, "todoapp": 36}
EXEC_H = {"fastapi": 0.75, "nestjs-realworld": 0.4, "todoapp": 0.2}   # scoring after generation (no LLM)


def ist(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).strftime("%H:%M")


def n_tests(svc: str, reqs) -> int:
    if not reqs:
        return TOTAL[svc]
    return sum(1 + len(json.load(open(ROOT / f"harness/data/requirements/{svc}/requirement_{r}.json")).get("mutants", []))
               for r in reqs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-hours", type=float, default=1.0)
    a = ap.parse_args()
    now = time.time()
    print(f"Status at {ist(now)} IST, {datetime.fromtimestamp(now, IST):%a %d %b}")
    end = None
    ep = ROOT.parent / "secrets" / "queue_endpoints.json"
    bf = ROOT.parent / "secrets" / "llm_backends.json"
    if bf.exists():                                       # dispatcher mode (D-101): the last server's end
        bk = json.loads(bf.read_text()).get("backends", [])
        ends = [datetime.fromisoformat(b["valid_until"].replace("Z", "+00:00")).timestamp() for b in bk if b.get("valid_until")]
        end = max(ends) if ends else None
        if end:
            left = (end - now) / 3600
            print(f"Servers: {len(bk)} in the dispatcher; last session ends ~{ist(end)} IST "
                  f"({'%.1f h left' % left if left > 0 else 'ENDED'})")
        ep = Path("/nonexistent")
    if ep.exists():
        cfg = json.loads(ep.read_text())
        end = datetime.fromisoformat(cfg["valid_until"].replace("Z", "+00:00")).timestamp()
        left = (end - now) / 3600
        print(f"Servers: {len(cfg.get('services', {}))} mapped; session ends ~{ist(end)} IST "
              f"({'%.1f h left' % left if left > 0 else 'ENDED'})")
    try:                                                    # dispatcher mode (D-101)
        import urllib.request
        d = json.loads(urllib.request.urlopen("http://127.0.0.1:11500/status", timeout=3).read())
        print(f"Dispatcher: queue {d['queue']} (oldest {d['oldest_wait_s']} s), busy {d['busy_total']}/{d['slots_total']} slots, "
              f"requests ok {d['totals']['ok']}, retried {d['totals']['retries']}, failed {d['totals']['failed']}, "
              f"timed out {d['totals']['timeouts']}")
        for sv in d["servers"]:
            up = d["uptime_s"] or 1
            print(f"  {sv['name']:14} {'up  ' if sv['alive'] else 'DOWN'} busy {sv['busy']}/{sv['slots']}  served {sv['served']:5}"
                  f"  errors {sv['errors']}  busy {100 * sv['busy_s'] / (up * sv['slots']):3.0f}% of slot-time"
                  f"  ends in {sv['ends_in_h']} h")
    except Exception:
        pass
    print("\nRunning:")
    pids = subprocess.run(["pgrep", "-f", "[.]venv/bin/python.*scripts/run_benchmark.py"],
                          capture_output=True, text=True).stdout.split()
    busy = set()
    for pid in pids:
        try:
            args = open(f"/proc/{pid}/cmdline").read().split("\0")
            env = open(f"/proc/{pid}/environ").read().split("\0")
        except OSError:
            continue
        if "--exp-id" not in args:
            continue
        eid, svc = args[args.index("--exp-id") + 1], args[args.index("--services") + 1]
        root = next((e.split("=", 1)[1] for e in env if e.startswith("PRETRAINED_HARNESS_ROOT=")), "")
        inst = svc + ("@" + root.rsplit("-", 1)[-1] if root else "")
        busy.add(inst)
        metas = glob.glob(str(ROOT / f"results/*/{eid}/run_meta.json"))
        if not metas:
            continue
        meta = json.load(open(metas[0]))
        start = datetime.fromisoformat(meta["started"]).replace(tzinfo=IST).timestamp()
        tot = n_tests(svc, meta.get("requirements"))
        d = Path(metas[0]).parent / svc
        done = len([p for p in glob.glob(str(d / "generated_tests/*/test.py")) if os.path.getmtime(p) >= start])
        last = max([os.path.getmtime(p) for p in glob.glob(str(d / "generated_tests/*/*"))] + [start])
        man = d / "generation_manifest.json"
        gen_done = man.exists() and os.path.getmtime(man) >= start
        el = (now - start) / 3600
        exec_h = EXEC_H[svc] * tot / TOTAL[svc]
        if gen_done:
            gen_end, job_end = os.path.getmtime(man), os.path.getmtime(man) + exec_h * 3600
            phase = "scoring"
        elif done:
            gen_end = now + (tot - done) / (done / el) * 3600
            job_end, phase = gen_end + exec_h * 3600, "generating"
        else:
            gen_end = job_end = None
            phase = "starting"
        risk = " ⚠ last test after session end" if end and gen_end and not gen_done and gen_end > end else ""
        print(f"  {eid:52} {inst:20} {el:4.1f} h  {phase:10} {done:3}/{tot}  last write {int((now - last) / 60)} min ago"
              f"  last test ≈{ist(gen_end) if gen_end else '?'}  done ≈{ist(job_end) if job_end else '?'} IST{risk}")
    if not busy:
        print("  (nothing running)")
    if ep.exists():
        idle = sorted(set(json.loads(ep.read_text()).get("services", {})) - busy)
        if idle and end and end > now:
            print(f"  idle servers: {', '.join(idle)}")
    print(f"\nFinished in the last {a.since_hours:g} h:")
    sys.path.insert(0, str(ROOT))
    from common.wandb_log import summarize
    rows = []
    for m in glob.glob(str(ROOT / "results/*/*/run_meta.json")):
        meta = json.load(open(m))
        fin = meta.get("finished")
        if not fin:
            continue
        ts = datetime.fromisoformat(fin).replace(tzinfo=IST).timestamp()
        if now - ts > a.since_hours * 3600:
            continue
        try:
            s = summarize(Path(m).parent)["pooled"]
            score = f"caught {s['valid_pct']:.1f}%" + (f", bug-planted {s['mutated_pct']:.1f}%" if s.get("mutated_pct") else "")
        except Exception:
            score = "-"
        rows.append((ts, meta["exp_id"], meta.get("status", "?")[:50], score))
    for ts, eid, st, score in sorted(rows):
        print(f"  {ist(ts)} {eid:52} {st:10} {score}")
    if not rows:
        print("  (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
