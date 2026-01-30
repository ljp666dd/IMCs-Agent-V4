import argparse
import csv
from typing import List, Set

# Recommendation eval: Top-K Recall

def load_ids(path: str) -> List[str]:
    ids = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row.get('material_id') or row.get('id') or row.get('mid'))
    return [i for i in ids if i]


def topk_recall(candidates: List[str], ground_truth: Set[str], k: int) -> float:
    topk = set(candidates[:k])
    if not ground_truth:
        return 0.0
    return len(topk & ground_truth) / len(ground_truth)


def main(candidates_csv: str, gt_csv: str, ks: List[int]):
    candidates = load_ids(candidates_csv)
    gt = set(load_ids(gt_csv))
    for k in ks:
        r = topk_recall(candidates, gt, k)
        print(f"Top-{k} recall: {r:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', required=True, help='CSV with material_id and score (sorted)')
    parser.add_argument('--ground_truth', required=True, help='CSV with material_id ground truth')
    parser.add_argument('--k', default='5,10,20', help='Comma-separated K values')
    args = parser.parse_args()
    ks = [int(x.strip()) for x in args.k.split(',') if x.strip()]
    main(args.candidates, args.ground_truth, ks)
