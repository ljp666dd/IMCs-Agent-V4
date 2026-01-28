import argparse
import json
import os
import subprocess
import sys
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)


def _split_channels(raw: str) -> List[str]:
    if not raw:
        return ["total"]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or ["total"]


def main():
    parser = argparse.ArgumentParser(description="Auto train + predict DOS curves for multiple channels.")
    parser.add_argument("--channels", default="total,s,p,d")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--update-db", action="store_true")
    args = parser.parse_args()

    channels = _split_channels(args.channels)
    reports = {}

    train_script = os.path.join(BASE_DIR, "src", "tools", "train_dos_curve_model.py")
    pred_script = os.path.join(BASE_DIR, "src", "tools", "predict_dos_curves.py")

    for ch in channels:
        model_out = os.path.join(BASE_DIR, "data", "ml_agent", f"dos_curve_model_{ch}.pkl")
        report_out = os.path.join(BASE_DIR, "data", "ml_agent", f"dos_curve_report_{ch}.json")
        cmd = [
            sys.executable, train_script,
            "--channel", ch,
            "--n-components", str(args.n_components),
            "--model-out", model_out,
            "--report", report_out,
            "--test-size", str(args.test_size),
        ]
        print("Training:", " ".join(cmd))
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"Training failed for channel {ch}")
            return ret.returncode

        pred_curves = os.path.join(BASE_DIR, "data", "theory", "dos_curve_predictions", ch)
        pred_plots = os.path.join(BASE_DIR, "data", "theory", "dos_curve_pred_plots", ch)
        cmd = [
            sys.executable, pred_script,
            "--model", model_out,
            "--out-curves", pred_curves,
            "--out-plots", pred_plots,
        ]
        if args.only_missing:
            cmd.append("--only-missing")
        if args.plot:
            cmd.append("--plot")
        if args.update_db:
            cmd.append("--update-db")
        print("Predict:", " ".join(cmd))
        ret = subprocess.run(cmd, check=False)
        if ret.returncode != 0:
            print(f"Prediction failed for channel {ch}")
            return ret.returncode

        if os.path.exists(report_out):
            with open(report_out, "r") as f:
                reports[ch] = json.load(f)

    summary_path = os.path.join(BASE_DIR, "data", "ml_agent", "dos_curve_report_all.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print("Summary report:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
