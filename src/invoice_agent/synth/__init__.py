"""Synthetic invoice dataset with ground-truth labels."""

from invoice_agent.synth.generator import Dataset, DatasetGenerator
from invoice_agent.synth.writer import write_dataset

__all__ = ["Dataset", "DatasetGenerator", "write_dataset"]
