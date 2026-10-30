#!/usr/bin/env python3
"""SFT data for the fine-tuned precision layer (leave-one-service-out).

For fold <held_out>, train on the other two services. Each example is the exact catalogue
prompt our precision layer sends (requirement + retrieved operations) and the target is that
requirement's gold checks (gold/checks.json) in the precision layer's JSON format.
Both the vague and the precise wording of training-service requirements are used as inputs;
nothing from the held-out service is included.

Output: finetune/data/sft_fold_<held_out>.jsonl  ({"messages": [...]} chat format)
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from agent.precision import CATALOGUE_PROMPT, FEW_SHOT, PROMPT_VERSION, PROPERTY_CATALOGUE  # noqa: E402
from agent.retrieval import Retriever  # noqa: E402
from common.paths import HARNESS_ROOT, SERVICES  # noqa: E402

KIND_TO_PROPERTY = {"status": "success_status", "field": "explicit", "message": "explicit",
                    "side_effect": "persistence", "presence": "explicit", "membership": "filter"}
RETRIEVAL_FOR = {"vague": "hybrid_rerank", "precise": "hybrid"}   # best variant per wording (offline eval)


def target_json(gold_checks: list[dict]) -> str:
    items = [{"description": c["check"], "property": KIND_TO_PROPERTY.get(c["kind"], "explicit"), "kind": "meaning"}
             for c in gold_checks]
    return "[" + ",\n ".join(json.dumps(i) for i in items) + "]"


def main() -> int:
    gold = json.loads((PROJECT_ROOT / "gold/checks.json").read_text())
    examples = {s: [] for s in SERVICES}
    for service in SERVICES:
        retriever = Retriever(service)
        for f in sorted((HARNESS_ROOT / "data/requirements" / service).glob("requirement_*.json")):
            req = json.loads(f.read_text())
            for wording in ("vague", "precise"):
                text = req[f"requirement_{wording}"]
                ops = retriever.rank(text, RETRIEVAL_FOR[wording])[:5]
                prompt = CATALOGUE_PROMPT.format(catalogue=PROPERTY_CATALOGUE, few_shot=FEW_SHOT,
                                                 operations="\n\n".join(o.to_text() for o in ops), requirement=text)
                examples[service].append({
                    "service": service, "requirement": req["id"], "wording": wording, "prompt_version": PROMPT_VERSION,
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": target_json(gold[service][str(req["id"])])}]})
    out = PROJECT_ROOT / "finetune" / "data"
    out.mkdir(parents=True, exist_ok=True)
    for held_out in SERVICES:
        train = [e for s in SERVICES if s != held_out for e in examples[s]]
        with (out / f"sft_fold_{held_out}.jsonl").open("w") as f:
            for e in train:
                f.write(json.dumps(e) + "\n")
        print(f"fold {held_out}: {len(train)} training examples (held out: {len(examples[held_out])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
