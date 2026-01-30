import argparse
import itertools
import json
import math
import os
import sys
import time

from src.config.config import config
from src.agents.core.theory_agent import TheoryDataConfig
from src.services.db.database import DatabaseService

try:
    from mp_api.client import MPRester
    HAS_MP = True
except ImportError:
    HAS_MP = False


def comb_count(n: int, k: int) -> int:
    try:
        return math.comb(n, k)
    except Exception:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Materials Project for 2-5 element alloys using TheoryDataConfig elements."
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimum number of elements in alloy.")
    parser.add_argument("--k-max", type=int, default=5, help="Maximum number of elements in alloy.")
    parser.add_argument("--max-combos", type=int, default=0, help="Max combinations per k (0 = no limit).")
    parser.add_argument("--per-combo-limit", type=int, default=50, help="Max docs per combo.")
    parser.add_argument("--stable-only", action="store_true", help="Only stable materials.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between combos (seconds).")
    parser.add_argument("--out", default="data/theory/mp_alloys_k2_k5.jsonl", help="Output JSONL path.")
    parser.add_argument("--resume", action="store_true", help="Resume from state file if present.")
    parser.add_argument("--state-file", default="data/theory/mp_alloys_state.json", help="Progress state file.")
    parser.add_argument("--slice-index", type=int, default=0, help="Slice index (0-based).")
    parser.add_argument("--slice-count", type=int, default=1, help="Total slices (default 1 = no slicing).")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no MP calls).")
    parser.add_argument("--store-db", action="store_true", help="Store minimal records into SQLite.")
    return parser.parse_args()


def load_processed_combos(path: str) -> set:
    if not os.path.exists(path):
        return set()
    processed = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                combo = rec.get("combo")
                if isinstance(combo, list):
                    processed.add(tuple(combo))
    except Exception:
        pass
    return processed


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path: str, state: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    if not HAS_MP:
        print("mp_api not installed. Please install mp-api first.")
        return 1
    api_key = config.MP_API_KEY
    if not api_key:
        print("MP_API_KEY missing. Set it in environment or .env.")
        return 1

    elements = TheoryDataConfig().elements
    if args.k_min < 2:
        print("k-min must be >= 2 for alloys.")
        return 1
    if args.k_max < args.k_min:
        print("k-max must be >= k-min.")
        return 1
    if args.slice_count < 1:
        print("slice-count must be >= 1.")
        return 1
    if args.slice_index < 0 or args.slice_index >= args.slice_count:
        print("slice-index must be in [0, slice-count).")
        return 1

    total_est = 0
    for k in range(args.k_min, args.k_max + 1):
        total_est += comb_count(len(elements), k)
    print(f"Elements: {len(elements)}")
    print(f"Estimated combinations (k={args.k_min}..{args.k_max}): {total_est}")
    if args.max_combos > 0:
        print(f"Limiting to {args.max_combos} combos per k.")
    if args.slice_count > 1:
        print(f"Slicing enabled: slice {args.slice_index + 1}/{args.slice_count}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    processed = load_processed_combos(args.out) if args.resume else set()
    state = load_state(args.state_file) if args.resume else {}
    resume_k = int(state.get("k", args.k_min)) if args.resume else args.k_min
    resume_index = int(state.get("combo_index", 0)) if args.resume else 0
    db = DatabaseService() if args.store_db else None

    with MPRester(api_key) as mpr, open(args.out, "a", encoding="utf-8") as out:
        for k in range(args.k_min, args.k_max + 1):
            if args.resume and k < resume_k:
                continue
            count = 0
            combo_index = 0
            for combo in itertools.combinations(elements, k):
                if args.resume and k == resume_k and combo_index < resume_index:
                    combo_index += 1
                    continue
                if args.max_combos > 0 and count >= args.max_combos:
                    break
                if args.slice_count > 1 and (combo_index % args.slice_count) != args.slice_index:
                    combo_index += 1
                    continue
                if combo in processed:
                    combo_index += 1
                    continue

                state = {"k": k, "combo_index": combo_index}
                save_state(args.state_file, state)

                if args.dry_run:
                    processed.add(combo)
                    count += 1
                    combo_index += 1
                    continue

                try:
                    docs = mpr.materials.summary.search(
                        elements=list(combo),
                        num_elements=k,
                        is_stable=True if args.stable_only else None,
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "elements",
                            "formation_energy_per_atom",
                            "energy_above_hull",
                            "is_stable",
                        ],
                        chunk_size=1,
                    )
                    for d in docs:
                        record = {
                            "combo": list(combo),
                            "num_elements": k,
                            "material_id": str(d.material_id),
                            "formula": getattr(d, "formula_pretty", None),
                            "elements": [el.symbol for el in getattr(d, "elements", [])] if getattr(d, "elements", None) else None,
                            "formation_energy_per_atom": getattr(d, "formation_energy_per_atom", None),
                            "energy_above_hull": getattr(d, "energy_above_hull", None),
                            "is_stable": getattr(d, "is_stable", None),
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if db and record.get("material_id") and record.get("formula"):
                            db.save_material(
                                material_id=record["material_id"],
                                formula=record["formula"],
                                energy=record.get("formation_energy_per_atom"),
                                cif_path=None,
                            )
                    processed.add(combo)
                    count += 1
                except Exception as exc:
                    print(f"Combo {combo} failed: {exc}")
                combo_index += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)

    print(f"Done. Output: {args.out}")
    save_state(args.state_file, {"k": args.k_max, "combo_index": 0, "done": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
