import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from src.agents.core.experiment_agent import ExperimentDataAgent
from src.agents.core.ml_agent import MLAgent, MLAgentConfig

from src.services.db.database import DatabaseService


def main():
    parser = argparse.ArgumentParser(description="Iterate with experiment data -> retrain ML -> update evidence.")
    parser.add_argument("--data-dir", default="data/experimental/rde_lsv", help="RDE LSV directory")
    parser.add_argument("--metric", default="exchange_current_density", help="Activity metric name")
    parser.add_argument("--reference-potential", type=float, default=0.2)
    parser.add_argument("--loading", type=float, default=0.25)
    parser.add_argument("--precious-fraction", type=float, default=0.20)
    parser.add_argument("--out", default="data/experimental/iteration_report.json")
    args = parser.parse_args()

    exp_agent = ExperimentDataAgent()
    exp_summary = exp_agent.process_rde_directory(
        data_dir=args.data_dir,
        reference_potential=args.reference_potential,
        loading_mg_cm2=args.loading,
        precious_fraction=args.precious_fraction,
    )

    ml_agent = MLAgent(MLAgentConfig(output_dir="data/ml_agent"))
    ml_agent.load_activity_metrics_from_db(args.metric)
    results = ml_agent.train_traditional_models() if ml_agent.X_train is not None else []
    preds = ml_agent.predict_best()

    db = DatabaseService()
    ranked = []
    if preds:
        try:
            ranked = sorted(preds.items(), key=lambda kv: kv[1], reverse=True)[:10]
        except Exception:
            ranked = list(preds.items())[:10]
        for mid, score in preds.items():
            try:
                db.save_evidence(
                    material_id=str(mid),
                    source_type="ml_prediction",
                    source_id=f"auto_iter_{args.metric}",
                    score=0.6,
                    metadata={
                        "prediction": float(score),
                        "metric": args.metric,
                        "origin": "auto_iterate_experiment",
                    },
                )
            except Exception:
                continue

    report = {
        "experiment": exp_summary,
        "metric": args.metric,
        "trained_models": [r.name for r in results],
        "predictions": preds,
        "ranking_top_n": [{"material_id": mid, "score": score} for mid, score in ranked],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Report saved: {args.out}")


if __name__ == "__main__":
    main()
