"""CLI: python -m invoiceguard.synth --n 300 --seed 42 --out data/synthetic"""

import argparse
import json

from invoiceguard.synth import DatasetGenerator, write_dataset


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a labelled synthetic invoice dataset.")
    p.add_argument("--n", type=int, default=300, help="number of invoice documents")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vendors", type=int, default=25)
    p.add_argument("--scan-ratio", type=float, default=0.3)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--no-variety", action="store_true", help="fixed per-template look (dataset v1)")
    p.add_argument("--adversarial", action="store_true",
                   help="generate the adversarial set (prompt injection, look-alikes, splits) instead")
    p.add_argument("--per-kind", type=int, default=6, help="adversarial cases per kind")
    a = p.parse_args()
    if a.adversarial:
        from invoiceguard.synth.adversarial import AdversarialGenerator
        ds = AdversarialGenerator(seed=a.seed, n_vendors=a.vendors, per_kind=a.per_kind).generate()
    else:
        ds = DatasetGenerator(seed=a.seed, n_vendors=a.vendors, scan_ratio=a.scan_ratio,
                              layout_variety=not a.no_variety).generate(a.n)
    stats = write_dataset(ds, a.out)
    print(json.dumps(stats, indent=2))
    print(f"Wrote dataset to {a.out}")


if __name__ == "__main__":
    main()
