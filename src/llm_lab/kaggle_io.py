"""Restore a Kaggle Dataset into a directory, one file at a time.

Kept for the notebooks that call download_dataset(); the implementation, and why it works
this way, is llm_lab.storage.KaggleStore. Notebooks import this after Cell 1 puts src/ on
sys.path.
"""
from llm_lab.storage import KaggleStore


def download_dataset(slug, dest, required=("latest.pt",), attempts=6, wait=20, only=None):
    """Download every file of `slug` into `dest` (or just the names in `only`).

    Returns the list of file names restored, or None if the Dataset does not exist.
    Raises (StoreError, a RuntimeError) if the Dataset lacks a file in `required`, or if any
    file fails `attempts` times. Every file is mandatory because the checkpoint notebooks push
    `dest` back as the next version: a file skipped here would be silently dropped from the
    Dataset. Pass `only` solely when reading from a Dataset this caller never pushes back.
    """
    return KaggleStore(slug, attempts=attempts, wait=wait).download(dest, only=only, required=required)
