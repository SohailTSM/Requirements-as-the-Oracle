"""RESTestBench's refinement tool with every round recorded.

Behaviour is identical to `tools.refinement_tool.RefinementTool`; this subclass only
records each round (test code, run output, HTTP trace, judge decision) to
`refinement_history.json` next to the generated test. The history is needed to
measure value copying / weakening (RQ2) and to build DPO preference pairs.
"""
import json
import time
from pathlib import Path

from tools.refinement_tool.refinement_tool import RefinementTool


class LoggedRefinementTool(RefinementTool):

    def generate_test(self, scenario_description, test_file_path, test_dir_path,
                      spec_path, reset_script_path, service_base_url="http://localhost:8000"):
        self._history = []
        started = time.time()
        error = None
        try:
            return super().generate_test(scenario_description, test_file_path, test_dir_path,
                                         spec_path, reset_script_path, service_base_url)
        except Exception as e:  # recorded, then re-raised unchanged
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            record = {
                "scenario_description": scenario_description,
                "service_base_url": service_base_url,
                "rounds_used": getattr(self, "refinement_rounds_used", None),
                "seconds": round(time.time() - started, 2),
                "error": error,
                "events": self._history,
            }
            Path(test_dir_path).mkdir(parents=True, exist_ok=True)
            (Path(test_dir_path) / "refinement_history.json").write_text(
                json.dumps(record, indent=2, default=str), encoding="utf-8")

    def _generate_test(self, scenario_description, spec_content, service_base_url,
                       previous_test=None, test_output=None, http_trace=None, judge_feedback=None):
        code, cost = super()._generate_test(scenario_description, spec_content, service_base_url,
                                            previous_test, test_output, http_trace, judge_feedback)
        self._history.append({"type": "generate", "refine": previous_test is not None,
                              "test_code": code, "t": time.time()})
        return code, cost

    def _execute_test_with_trace(self, test_file_path, service_base_url, timeout=60):
        passed, output, trace = super()._execute_test_with_trace(test_file_path, service_base_url, timeout)
        self._history.append({"type": "execute", "passed": passed, "output": output,
                              "http_trace": trace, "t": time.time()})
        return passed, output, trace

    def _judge_test(self, scenario_description, spec_content, test_code, test_output, http_trace):
        decision, cost = super()._judge_test(scenario_description, spec_content,
                                             test_code, test_output, http_trace)
        self._history.append({"type": "judge", "decision": decision, "t": time.time()})
        return decision, cost
