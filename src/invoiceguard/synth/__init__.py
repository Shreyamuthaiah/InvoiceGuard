"""Synthetic invoice dataset with ground-truth labels."""

from invoiceguard.synth.generator import Dataset, DatasetGenerator
from invoiceguard.synth.writer import write_dataset

__all__ = ["Dataset", "DatasetGenerator", "write_dataset"]
