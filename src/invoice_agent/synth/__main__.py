"""CLI: python -m invoice_agent.synth --n 300 --seed 42 --out data/synthetic"""

import argparse
import json

from invoice_agent.synth import DatasetGenerator, write_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a labelled synthetic invoice dataset.")
    p.add_argument("--n", type=int, default=300, help="number of invoice documents")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vendors", type=int, default=25)
    p.add_argument("--scan-ratio", type=float, default=0.3)
    p.add_argument("--out", default="data/synthetic")
    a = p.parse_args()
    ds = DatasetGenerator(seed=a.seed, n_vendors=a.vendors, scan_ratio=a.scan_ratio).generate(a.n)
    stats = write_dataset(ds, a.out)
    print(json.dumps(stats, indent=2))
    print(f"Wrote dataset to {a.out}")


if __name__ == "__main__":
    main()
