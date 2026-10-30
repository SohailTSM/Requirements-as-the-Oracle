#!/usr/bin/env python3
"""Run benchmark jobs from a YAML queue (one at a time by default).

    .venv/bin/python scripts/run_queue.py configs/queue_baselines.yaml

- Skips jobs whose run_meta.json already says status "ok" (safe to restart).
- Before each job: waits while results/queue.pause exists (lets development use the
  services) and while available RAM is below MIN_FREE_GB (protects VS Code).
- Each job runs in its own process, so its memory is released afterwards.
- The YAML is re-read before every job, so jobs can be added or reordered while it runs.
- `max_parallel` in the YAML (default 1) allows several jobs at once, but only on DISJOINT
  services (each service has one working copy and one port). The GPU is shared through Ollama,
  which serves one request at a time; overlapping runs fill the GPU's idle time during the
  harness's service resets and test executions. run_benchmark.py processes started outside
  the queue count too (their services are read from their command line).
- A job that failed twice is skipped. A job ended by a signal (stopped by hand) is not a failure.
- results/queue.reserve lists services kept free for development (re-read every loop).
- secrets/queue_endpoints.json (optional) sends each service's jobs to a remote LLM server.
- Appends one line per job to results/queue_status.jsonl.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
PAUSE_FILE = RESULTS / "queue.pause"
RESERVE_FILE = RESULTS / "queue.reserve"      # services (one per line) the queue must not start jobs on
# Remote LLM servers per service (D-82). Kept in secrets/ because the links contain a secret:
#   {"valid_until": "2026-10-24T05:38:00Z", "min_hours_left": 3,
#    "services": {"nestjs-realworld": "https://.../<secret>/v1", "todoapp": "..."}}
# When the file exists, a job starts only if its service has a server with enough time left;
# nothing falls back to the local GPU.
ENDPOINTS_FILE = PROJECT_ROOT.parent / "secrets" / "queue_endpoints.json"
STATUS_FILE = RESULTS / "queue_status.jsonl"
MIN_FREE_GB = 4.0


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 99.0


def log_status(**kw) -> None:
    kw["time"] = datetime.now().isoformat(timespec="seconds")
    with STATUS_FILE.open("a") as f:
        f.write(json.dumps(kw) + "\n")


def exp_id(job: dict) -> str:
    return job.get("exp_id") or f"{job['tool']}__{job['variant']}__rep{job['rep']}"


def already_done(job: dict) -> bool:
    meta = RESULTS / job["group"] / exp_id(job) / "run_meta.json"
    return meta.exists() and json.loads(meta.read_text()).get("status") == "ok"


def _instance_of(pid: str) -> str:
    """'' for the original harness, '@<name>' for a service copy (PRETRAINED_HARNESS_ROOT)."""
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return ""
    for kv in env:
        if kv.startswith(b"PRETRAINED_HARNESS_ROOT="):
            name = Path(kv.split(b"=", 1)[1].decode()).name          # harness-<service>-<name>
            return "@" + name.rsplit("-", 1)[-1] if name.startswith("harness-") else ""
    return ""


def external_busy_services(own_pids: set[int]) -> tuple[int, set[str]]:
    """run_benchmark.py processes not started by this queue: (count, instance keys they use)."""
    out = subprocess.run(["pgrep", "-af", "scripts/run_benchmark.py"], capture_output=True, text=True)
    n, busy = 0, set()
    for line in out.stdout.splitlines():
        pid, _, cmd = line.partition(" ")
        first = cmd.split()[0] if cmd.split() else ""
        if int(pid) in own_pids or first.endswith(("zsh", "bash", "/sh", "timeout")) or first == "sh":
            continue                               # wrappers and shells, not the benchmark itself
        n += 1
        inst = _instance_of(pid)
        args = cmd.split()
        if "--services" in args:
            for a in args[args.index("--services") + 1:]:
                if a.startswith("--"):
                    break
                busy.add(a + inst)
        else:
            busy |= {"fastapi", "nestjs-realworld", "todoapp"}
    return n, busy


# Run lengths on one T4 (hours), measured on the 23-24 Oct Kaggle sessions (D-90); used so that a job is
# only started if it can finish before the remote server's session ends.
EST_HOURS = {("b1_single_step", "fastapi"): 0.7, ("b1_single_step", "nestjs-realworld"): 0.5,
             ("b1_single_step", "todoapp"): 0.3, ("b2_refinement", "fastapi"): 10.5,
             ("b2_refinement", "nestjs-realworld"): 8.0, ("b2_refinement", "todoapp"): 2.1,
             ("b3_prompt_fixed", "fastapi"): 4.0, ("b3_prompt_fixed", "nestjs-realworld"): 2.0,
             ("b3_prompt_fixed", "todoapp"): 0.8, ("ours", "fastapi"): 3.5, ("ours", "nestjs-realworld"): 1.6,
             ("ours", "todoapp"): 0.6, ("e_fuzzer", "fastapi"): 0.5, ("e_fuzzer", "nestjs-realworld"): 0.3,
             ("e_fuzzer", "todoapp"): 0.2, ("b4_response_judge", "fastapi"): 1.5,
             ("b4_response_judge", "nestjs-realworld"): 0.8, ("b4_response_judge", "todoapp"): 0.4,
             ("golden", "fastapi"): 0.8, ("golden", "nestjs-realworld"): 0.5, ("golden", "todoapp"): 0.3}


# Our agent's variants differ a lot in LLM calls per test. Factors vs the default, measured per test on
# the same 5 FastAPI requirements (smoke v4/v5, dedicated server); untimed variants get cautious guesses (D-96).
CONFIG_FACTOR = {"architecture__blackboard": 3.6, "architecture__multi_persona": 1.85, "architecture__react": 2.3,
                 "communication__mcp": 1.1, "communication__a2a": 1.05, "retrieval__full": 1.3,
                 "retrieval__query_rewrite": 1.15, "precision__self_consistency": 1.4, "generation__raw": 0.6,
                 "content__full": 1.1, "refinement__none": 0.7}


def est_hours(job: dict) -> float:
    full = sum(EST_HOURS.get((job["tool"], s), 4.0) for s in job["services"])
    if SLOTS > 1 and job["tool"] not in NO_LLM_TOOLS:
        full *= SHARED_SLOWDOWN                   # it may share its server with another generating job
    cfg = str(job.get("agent_config", ""))
    full *= next((f for k, f in CONFIG_FACTOR.items() if k in cfg), 1.0)
    if job.get("requirements"):
        n = sum(len(ALL_REQS.get(s, [])) for s in job["services"]) or 1
        return max(0.2, full * len(job["requirements"]) / n)
    return full


def failures(group: str, eid: str) -> int:
    if not STATUS_FILE.exists():
        return 0
    n = 0
    for line in STATUS_FILE.read_text().splitlines():
        r = json.loads(line)
        if (r.get("event") == "end" and r.get("group") == group and r.get("exp_id") == eid
                and r.get("returncode") not in (0, -15, -9, -2)):
            n += 1
    return n


# FastAPI requirement 15 names a fixed host port ("mailcatcher at localhost:1080"), so it can only be
# tested faithfully on the original instance (D-85). Every full FastAPI job is therefore split: the
# main part (all other requirements, any instance) and an "__r15" part (requirement 15, original only).
PORT_BOUND = {"fastapi": [15]}
ALL_REQS = {s: sorted(int(f.stem.split("_")[1]) for f in (PROJECT_ROOT / "harness/data/requirements" / s).glob("requirement_*.json"))
            for s in ("fastapi", "nestjs-realworld", "todoapp")}


def _meta(job: dict):
    m = RESULTS / job["group"] / exp_id(job) / "run_meta.json"
    return json.loads(m.read_text()) if m.exists() else None


MAX_PART_HOURS = 1.5      # runs are split into requirement chunks of at most ~1.5 h (D-94, D-99)
MIN_PART_HOURS = 0.25


def part_hours() -> float:
    """Chunk size for parts not started yet. In dispatcher mode it shrinks near the session end so
    the last hours still get parts that finish in time (D-108), with more margin than the fit rule."""
    if BACKENDS_FILE.exists():
        left = dispatcher_hours_left()
        if 0 < left < float("inf"):
            return max(MIN_PART_HOURS, min(MAX_PART_HOURS, (left - 0.4) / (SHARED_SLOWDOWN * 1.4)))
    return MAX_PART_HOURS


def _chunks(job: dict, reqs: list) -> list[dict]:
    """A job as requirement chunks (exp_id __partN), merged in analysis. Short parts cannot be cut
    off by a server session ending and fill a session's last hours. Parts that already exist (run,
    running or failed) keep their requirements exactly; only uncovered requirements are chunked anew,
    numbered after the existing parts, so changing the chunk size never reshuffles a job (D-99)."""
    existing = []
    for m in sorted((RESULTS / job["group"]).glob(f"{exp_id(job)}__part*/run_meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("requirements"):
            existing.append(dict(job, requirements=meta["requirements"], exp_id=meta["exp_id"]))
    if existing:
        covered = {r for p in existing for r in p["requirements"]}
        rest = [r for r in reqs if r not in covered]
        n0 = max(int(p["exp_id"].rsplit("__part", 1)[1]) for p in existing)
        new = []
        if rest:
            k = max(1, int(-(-est_hours(dict(job, requirements=rest)) // part_hours())))
            size = -(-len(rest) // k)
            new = [dict(job, requirements=rest[i * size:(i + 1) * size], exp_id=f"{exp_id(job)}__part{n0 + i + 1}")
                   for i in range(k) if rest[i * size:(i + 1) * size]]
        return existing + new
    whole = dict(job, requirements=reqs)
    k = max(1, int(-(-est_hours(whole) // part_hours())))
    if k == 1:
        return [whole]
    size = -(-len(reqs) // k)
    return [dict(job, requirements=reqs[i * size:(i + 1) * size], exp_id=f"{exp_id(job)}__part{i + 1}")
            for i in range(k) if reqs[i * size:(i + 1) * size]]


def split_port_bound(job: dict) -> list[dict]:
    if len(job["services"]) != 1 or job.get("requirements"):
        return [job]
    svc = job["services"][0]
    bound = PORT_BOUND.get(svc, [])
    meta = _meta(job)
    if meta and not meta.get("finished") and not meta.get("status"):
        return [job]                                # running right now as one whole job: leave it whole
    if not bound:                                   # no port-bound requirement: chunk if long and not done
        if meta and meta.get("status") == "ok":
            return [job]
        parts = _chunks(job, ALL_REQS[svc])
        return parts if len(parts) > 1 else [job]
    if meta and meta.get("status") == "ok":
        covered = (meta.get("requirements") is None and meta.get("harness_instance", "original") == "original"
                   and not set(bound) & set(meta.get("excluded_requirements", [])))
        if covered:
            return [job]                                # an earlier full run on the original instance
        main = []
    else:
        main = _chunks(job, [r for r in ALL_REQS[svc] if r not in bound])
    part = dict(job, requirements=list(bound), exp_id=exp_id(job) + "__r15", original_only=True)
    return main + [part]


def pending_jobs(queue_path: Path, running: set[str]):
    queue = yaml.safe_load(queue_path.read_text())
    defaults = queue.get("defaults", {})
    jobs = []
    for j in queue["jobs"]:
        for job in split_port_bound({**defaults, **j}):
            key = f"{job['group']}/{exp_id(job)}"
            meta = _meta(job)
            in_progress = bool(meta) and not meta.get("finished") and not meta.get("status")   # started by another runner
            if key not in running and not in_progress and not already_done(job) and failures(job["group"], exp_id(job)) < 2:
                jobs.append(job)
    return jobs, int(queue.get("max_parallel", 1)), set(queue.get("skip_instances", []))


NO_LLM_TOOLS = {"golden", "e_fuzzer"}          # need a service instance but no model server
SLOTS = 1                                       # set from the endpoints file (slots_per_server)
SHARED_SLOWDOWN = 1.5                           # a job's run time when it shares a 2-slot server (D-100; to be measured)


def generating_urls():
    """Counter {server URL: number of jobs still generating on it} (see _generating_list)."""
    from collections import Counter
    return Counter(_generating_list())


def _generating_list() -> list:
    """LLM servers that a running job is still generating tests on (scoring needs no server, D-97).
    Read from every run_benchmark.py process: its PRETRAINED_LLM_URL and whether its run has
    written generation_manifest.json yet."""
    out = subprocess.run(["pgrep", "-f", "[.]venv/bin/python.*scripts/run_benchmark.py"], capture_output=True, text=True)
    urls = []
    for pid in out.stdout.split():
        try:
            args = Path(f"/proc/{pid}/cmdline").read_text().split("\0")
            env = Path(f"/proc/{pid}/environ").read_text().split("\0")
        except OSError:
            continue
        url = next((e.split("=", 1)[1] for e in env if e.startswith("PRETRAINED_LLM_URL=")), None)
        if not url or "--exp-id" not in args:
            continue
        eid, svc = args[args.index("--exp-id") + 1], args[args.index("--services") + 1]
        tool = args[args.index("--tool") + 1] if "--tool" in args else ""
        if tool in NO_LLM_TOOLS:
            continue
        metas = list(RESULTS.glob(f"*/{eid}/run_meta.json"))
        if not metas:
            urls.append(url)
            continue
        meta = json.loads(metas[0].read_text())
        from datetime import datetime
        try:
            started = datetime.fromisoformat(meta["started"]).timestamp()
        except Exception:
            started = 0
        man = metas[0].parent / svc / "generation_manifest.json"
        if not (man.exists() and man.stat().st_mtime >= started):
            urls.append(url)
    return urls


# ---------------------------------------------------------------- dispatcher mode (D-101)
BACKENDS_FILE = PROJECT_ROOT.parent / "secrets" / "llm_backends.json"
DISPATCH_URL = "http://127.0.0.1:11500/v1"
TARGET_QUEUE = 1                 # start more runs while the dispatcher's queue is at most this long
EXTRA_RUNS = 2                   # at most slots + EXTRA_RUNS runs generating at once
RAMP_S = 150                     # after starting a run, wait this long before judging the queue again


def dispatcher_status():
    """The dispatcher's /status as a dict, or None if it is not reachable."""
    import urllib.request
    try:
        return json.loads(urllib.request.urlopen("http://127.0.0.1:11500/status", timeout=5).read())
    except Exception:
        return None


def all_instances() -> list:
    """Every service instance on this machine: originals plus service_copies/harness-<svc>-<x>."""
    inst = ["fastapi", "nestjs-realworld", "todoapp"]
    for d in sorted((PROJECT_ROOT / "service_copies").glob("harness-*")):
        svc, name = d.name[len("harness-"):].rsplit("-", 1)
        inst.append(f"{svc}@{name}")
    return inst


def dispatcher_hours_left() -> float:
    """Hours until the last usable server's session ends (from the backends file)."""
    from datetime import datetime, timezone
    try:
        cfg = json.loads(BACKENDS_FILE.read_text())
    except Exception:
        return 0.0
    ends = [datetime.fromisoformat(b["valid_until"].replace("Z", "+00:00")).timestamp()
            for b in cfg.get("backends", []) if b.get("valid_until")]
    return (max(ends) - time.time()) / 3600 - 2 / 60 if ends else float("inf")


def endpoints():
    """({instance key: url}, hours left) usable now, or (None, inf) without an endpoints file.
    Instance keys: "fastapi" (original harness) or "fastapi@b" (service copy b, D-84)."""
    if not ENDPOINTS_FILE.exists():
        return None, float("inf")
    from datetime import datetime, timezone
    cfg = json.loads(ENDPOINTS_FILE.read_text())
    until = datetime.fromisoformat(cfg["valid_until"].replace("Z", "+00:00"))
    left = (until - datetime.now(timezone.utc)).total_seconds() / 3600
    global SLOTS
    SLOTS = int(cfg.get("slots_per_server", 1))   # parallel requests per model server (D-100)
    return cfg.get("services", {}), left


def copy_root(instance: str) -> str | None:
    """Harness folder for an instance key ("fastapi@b" -> service_copies/harness-fastapi-b)."""
    if "@" not in instance:
        return None
    svc, name = instance.split("@", 1)
    return str(PROJECT_ROOT / "service_copies" / f"harness-{svc}-{name}")


def start(job: dict):
    eid = exp_id(job)
    out_dir = RESULTS / job["group"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PROJECT_ROOT / ".venv/bin/python"), str(PROJECT_ROOT / "scripts/run_benchmark.py"),
           "--tool", job["tool"], "--variant", job["variant"], "--rep", str(job["rep"]),
           "--group", job["group"], "--exp-id", eid, "--services", *job["services"]]
    if job.get("agent_config"):
        cmd += ["--agent-config", str(PROJECT_ROOT / job["agent_config"])]
    if job.get("requirements"):
        cmd += ["--requirements", *map(str, job["requirements"])]
    if job.get("timeout"):                           # per-test timeout (B4 tests wait for an LLM verdict)
        cmd += ["--timeout", str(job["timeout"])]
    print(f"[queue] start: {job['group']}/{eid} on {job.get('instance') or job['services']}", flush=True)
    log_status(event="start", group=job["group"], exp_id=eid)
    logf = (out_dir / f"{eid}.log").open("a")
    env = dict(os.environ)
    if job.get("instance") and copy_root(job["instance"]):   # service copy for this job (D-84)
        env["PRETRAINED_HARNESS_ROOT"] = copy_root(job["instance"])
    if job.get("llm_url"):                           # remote LLM server for this job (D-82)
        env["PRETRAINED_LLM_URL"] = job["llm_url"]
        env["AZURE_AI_FOUNDRY_ENDPOINT_OPENAI"] = job["llm_url"]
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.STDOUT, env=env)
    return {"job": job, "proc": proc, "log": logf, "t0": time.time()}


def dispatch_mode_step(jobs, running, busy, can_start) -> bool:
    """Start at most one job; True if one was started."""
    if not can_start:
        return False
    st = dispatcher_status()
    hours_left = dispatcher_hours_left()
    generating = sum(_generating_list_any())
    instances = [k for k in all_instances() if k not in busy]
    for j in jobs:
        needs_llm = j["tool"] not in NO_LLM_TOOLS
        if needs_llm:
            if st is None or not st.get("slots_total"):
                continue                              # no dispatcher or no live server
            if st["queue"] > TARGET_QUEUE or generating >= st["slots_total"] + EXTRA_RUNS:
                continue                              # the models are already kept busy
            if hours_left < est_hours(j) * SHARED_SLOWDOWN * 1.2 + 0.25:
                continue                              # would not finish before the last server ends
        svc = j["services"][0]
        free = [k for k in instances if k.split("@")[0] == svc
                and not (j.get("original_only") and "@" in k) and (not j.get("pin") or k == j["pin"])]
        if len(j["services"]) > 1 or not free:
            continue
        job = dict(j, instance=free[0], llm_url=DISPATCH_URL if needs_llm else None)
        if not job["llm_url"]:
            job.pop("llm_url")
        running[f"{job['group']}/{exp_id(job)}"] = start(job)
        return True
    return False


def _generating_list_any() -> list:
    """1 per run_benchmark process of a model-using tool that is still generating (any server)."""
    return [1 for _ in _generating_list()]


def main() -> int:
    queue_path = Path(sys.argv[1])
    RESULTS.mkdir(exist_ok=True)
    running = {}                                   # "group/exp_id" -> info
    while True:
        for key, r in list(running.items()):       # reap finished jobs
            rc = r["proc"].poll()
            if rc is None:
                continue
            r["log"].close()
            job = r["job"]
            log_status(event="end", group=job["group"], exp_id=exp_id(job), returncode=rc,
                       seconds=round(time.time() - r["t0"], 1))
            print(f"[queue] end: {key} rc={rc} after {time.time() - r['t0']:.0f}s", flush=True)
            del running[key]
        jobs, max_parallel, skip = pending_jobs(queue_path, set(running))
        if not jobs and not running:
            break
        n_ext, busy = external_busy_services({r["proc"].pid for r in running.values()})
        for r in running.values():
            busy |= {r["job"].get("instance")} if r["job"].get("instance") else set(r["job"]["services"])
        if RESERVE_FILE.exists():
            busy |= {line.strip() for line in RESERVE_FILE.read_text().splitlines() if line.strip()}
        busy |= skip                                  # instances this queue leaves to another queue (D-95)
        can_start = (not PAUSE_FILE.exists() and len(running) + n_ext < max_parallel
                     and mem_available_gb() >= MIN_FREE_GB)
        if BACKENDS_FILE.exists():                   # dispatcher mode: one queue for all model servers
            started = dispatch_mode_step(jobs, running, busy, can_start)
            if started:
                time.sleep(RAMP_S)
                continue
            time.sleep(30)
            continue
        eps, hours_left = endpoints()
        from collections import Counter
        gen_urls = generating_urls() if eps else Counter()
        if can_start:
            chosen = None
            for j in jobs:
                if hours_left < est_hours(j) * 1.2 + 0.25:     # would not finish before the session ends
                    continue
                if len(j["services"]) > 1 or eps is None:      # original instances only
                    if not (set(j["services"]) & busy) and (eps is None or all(sv in eps for sv in j["services"])):
                        chosen = dict(j, llm_url=j.get("llm_url") or (eps or {}).get(j["services"][0]))
                        break
                    continue
                svc = j["services"][0]
                needs_llm = j["tool"] not in NO_LLM_TOOLS
                free = [k for k in eps if k.split("@")[0] == svc and k not in busy
                        and not (needs_llm and gen_urls[eps[k]] >= SLOTS)
                        and not (j.get("original_only") and "@" in k) and (not j.get("pin") or k == j["pin"])]
                if free:
                    chosen = dict(j, instance=free[0], llm_url=j.get("llm_url") or eps[free[0]])
                    break
            if chosen is not None:
                if not chosen.get("llm_url"):
                    chosen.pop("llm_url", None)
                running[f"{chosen['group']}/{exp_id(chosen)}"] = start(chosen)
                time.sleep(60)                     # let it load models before judging RAM again
                continue
        time.sleep(30)
    print("[queue] all jobs processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
