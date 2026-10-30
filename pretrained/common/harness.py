"""Helpers for driving RESTestBench services from our own scripts (without modifying the harness)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .paths import HARNESS_ROOT, PROJECT_ROOT

if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")
SERVICE_PORTS = {"fastapi": 8000, "nestjs-realworld": 3000, "todoapp": 5000}


def patch_working_copy_ownership() -> None:
    """Containers write build files into mounted working copies as root, which breaks the
    harness's reset copy. Give them back to this user before every reset."""
    from src.service_manager import ServiceManager
    if getattr(ServiceManager.reset_working_copy, "_pretrained_patched", False):
        return
    original = ServiceManager.reset_working_copy
    uid, gid = os.getuid(), os.getgid()

    def reset_with_chown(self, service_name, source_subdir=None):
        working = self.get_working_path(service_name)
        if working.exists():
            subprocess.run(["docker", "run", "--rm", "-v", f"{working}:/w", "--entrypoint", "chown",
                            "postgres:17", "-R", f"{uid}:{gid}", "/w"],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return original(self, service_name, source_subdir)

    reset_with_chown._pretrained_patched = True
    ServiceManager.reset_working_copy = reset_with_chown


def patch_health_wait(timeout: float = 180.0, interval: float = 5.0) -> None:
    """The harness checks a service's health once, right after it copies original/ over working/.
    That copy makes watch-mode services (NestJS) recompile; on a busy machine the single check
    then fails and the harness silently skips the service. Poll for up to `timeout` seconds."""
    import time
    from src.service_manager import ServiceManager
    if getattr(ServiceManager.is_service_healthy, "_pretrained_patched", False):
        return
    original = ServiceManager.is_service_healthy

    def healthy_with_wait(self, port, health_endpoint="/docs", timeout_s=5.0):
        deadline = time.time() + timeout
        while True:
            if original(self, port, health_endpoint, timeout_s):
                return True
            if time.time() >= deadline:
                return False
            time.sleep(interval)

    healthy_with_wait._pretrained_patched = True
    ServiceManager.is_service_healthy = healthy_with_wait


def service_manager():
    from src.config import SERVICES_DIR
    from src.service_manager import ServiceManager
    patch_working_copy_ownership()
    return ServiceManager(SERVICES_DIR)


def load_requirement(service: str, rid: int):
    from src.mutator import Requirement
    return Requirement.from_file(HARNESS_ROOT / "data/requirements" / service / f"requirement_{rid}.json")


def run_pytest(test_file: Path, timeout: int = 120) -> tuple[bool, str, dict | None]:
    """Run one generated test in a fresh process; return (passed, output, runtime report)."""
    report = Path(test_file).with_suffix(".report.json")
    report.unlink(missing_ok=True)
    env = dict(os.environ, PRETRAINED_REPORT=str(report))
    t0 = time.time()
    try:
        p = subprocess.run([PYTHON, "-m", "pytest", str(test_file), "-q", "--tb=short", "-p", "no:cacheprovider"],
                           capture_output=True, text=True, timeout=timeout, env=env, cwd=Path(test_file).parent)
        passed, output = p.returncode == 0, (p.stdout + p.stderr)[-4000:]
    except subprocess.TimeoutExpired:
        passed, output = False, f"timeout after {timeout}s"
    data = json.loads(report.read_text()) if report.exists() else None
    if data is not None:
        data["seconds"] = round(time.time() - t0, 2)
    return passed, output, data


def prepare_valid(sm, service: str) -> None:
    """Valid implementation, healthy, clean database (as the harness does before each run)."""
    port = SERVICE_PORTS[service]
    sm.reset_working_copy(service)
    sm.wait_for_service_health(service, port)
    sm.call_reset_script(service, port)


def prepare_mutant(sm, service: str, mutant) -> bool:
    port = SERVICE_PORTS[service]
    sm.reset_working_copy(service)
    applied = sm.apply_mutation(service, mutant)
    sm.wait_for_service_health(service, port)
    sm.call_reset_script(service, port)
    return applied
