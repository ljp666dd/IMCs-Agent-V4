import argparse
import json
import os

from src.config.config import config
from src.services.task.eval_suite import (
    EvalSuiteConfig,
    build_eval_report,
    load_ids_set_from_csv,
    write_report_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="IMCs evaluation suite (P1-2)")
    parser.add_argument("--db", default=config.DB_PATH, help="SQLite DB path (default: data/imcs.db)")
    parser.add_argument("--knowledge-dir", default=os.path.join("data", "tasks"), help="Knowledge pack directory")
    parser.add_argument("--out-dir", default=os.path.join("data", "analysis"), help="Report output directory")
    parser.add_argument("--recent", type=int, default=20, help="Evaluate most recent N plans")
    parser.add_argument("--plan-id", action="append", default=None, help="Evaluate specific plan id (repeatable)")
    parser.add_argument("--ks", default="5,10,20", help="K values (comma-separated)")
    parser.add_argument("--ground-truth-csv", default=None, help="CSV path containing ground truth ids")
    args = parser.parse_args()

    ks = []
    for part in str(args.ks or "").split(","):
        part = part.strip()
        if part.isdigit():
            ks.append(int(part))
    if not ks:
        ks = [5, 10, 20]

    gt_ids = load_ids_set_from_csv(args.ground_truth_csv) if args.ground_truth_csv else None
    plan_ids = tuple(args.plan_id) if args.plan_id else None

    cfg = EvalSuiteConfig(
        db_path=args.db,
        knowledge_dir=args.knowledge_dir,
        recent=args.recent,
        ks=tuple(ks),
        ground_truth_ids=gt_ids,
        plan_ids=plan_ids,
    )
    report = build_eval_report(cfg)
    paths = write_report_files(report, out_dir=args.out_dir, prefix="eval_report")

    print("Eval suite done:")
    print(json.dumps(report.get("summary") or {}, ensure_ascii=False, indent=2))
    print("Report paths:")
    print(json.dumps(paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

