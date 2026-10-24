"""Run inspection over labelled samples and print a confusion matrix.

    venv/bin/python tools/evaluate.py     # uses data/good/ and data/bad/<defect>/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pin_inspection.accuracy import evaluate_samples  # noqa: E402
from pin_inspection.config import load_config  # noqa: E402
from pin_inspection.model import PinModel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main():
    cfg = load_config()
    mp = ROOT / cfg["model"]["path"]
    model = PinModel.load(mp) if mp.exists() else None
    print(f"Model: {'loaded' if model else 'none (tolerances only)'}\n")
    r = evaluate_samples(ROOT / "data", cfg, model)
    for i in r["items"]:
        print(f"  {'ok   ' if i['correct'] else 'WRONG'} {i['file']}: {i['verdict']} {' '.join(i['failed'])}")
    g, b = r["good"], r["bad"]
    print(f"\n              predicted PASS  predicted FAIL")
    print(f"  good ({g['n']:3d})   {g['passed']:14d}  {g['n'] - g['passed']:14d}   <- FAIL here = false reject")
    print(f"  bad  ({b['n']:3d})   {b['n'] - b['caught']:14d}  {b['caught']:14d}   <- PASS here = missed defect")
    print(f"\nAccuracy {r['accuracy_pct']}% ({r['correct']}/{r['n']})")
    for d, v in r["defects"].items():
        print(f"  {d:12s} caught {v['caught']}/{v['n']}")


if __name__ == "__main__":
    main()
