"""Training-side model classes for the fresh-model multi-modal upgrade.

These modules live separately from ``mcrs/retrieval_modules`` (inference)
and ``mcrs/embedders`` (catalog encoding) because they're only loaded by
training scripts under ``scripts/`` — keeping them out of the inference
import path avoids pulling ``peft`` / heavy training-only deps into
Blind-A inference.
"""
