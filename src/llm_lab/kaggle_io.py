"""Restore a Kaggle Dataset into a directory, one file at a time.

Why not `kaggle datasets download -d <slug> --unzip`: that asks Kaggle for the
whole-dataset zip, and for the multi-GB checkpoint Datasets (a new version every
~500 steps) Kaggle often never builds one. The API answers 404 "No gcs url found"
for every version, forever, so the notebook's retry loop could never succeed
(aion-transformer-tpu-large v97/v98, 2026-09-22). Per-file downloads (`-f name`)
redirect straight to the stored file and work. They also skip the unzip, so the
disk never holds the archive and its contents at the same time.

Notebooks import this after Cell 1 puts src/ on sys.path.
"""
import subprocess
import time
from pathlib import Path

# The listing call answers 403 (not 404) for a Dataset that doesn't exist; a bad
# credential is a 401, which deliberately doesn't match.
_MISSING_MARKERS = ("403", "404", "not found", "doesn't exist", "does not exist")


def _kaggle(*args):
    res = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    return res.returncode, ((res.stdout or "") + (res.stderr or "")).strip()


def _list_files(slug):
    """{name: size} of the current version, or (None, output) when the call failed."""
    rc, out = _kaggle("datasets", "files", slug, "--csv")
    if rc != 0:
        return None, out
    files, seen_header = {}, False
    for line in out.splitlines():  # the CLI prints an "outdated version" warning first
        if line.startswith("name,size"):
            seen_header = True
        elif seen_header and line.strip():
            name, size = line.split(",")[:2]
            files[name] = int(size)
    return (files if seen_header else None), out


def download_dataset(slug, dest, required=("latest.pt",), attempts=6, wait=20):
    """Download every file of `slug` into `dest`.

    Returns the list of file names restored, or None if the Dataset does not exist.
    Raises RuntimeError if the Dataset lacks a file in `required`, or if ANY file fails
    `attempts` times (a single file can 404 transiently). Every file is mandatory because
    the checkpoint notebooks push `dest` back as the next version: a file skipped here
    would be silently dropped from the Dataset.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = out = None
    for attempt in range(1, attempts + 1):
        files, out = _list_files(slug)
        if files is not None:
            break
        if any(m in out.lower() for m in _MISSING_MARKERS):
            return None
        print(f"Listing {slug} failed (attempt {attempt}/{attempts}); retrying in {wait}s...\n{out[-400:]}")
        time.sleep(wait)
    else:
        raise RuntimeError(f"Could not list files of {slug}.\nLast output:\n{out}")

    missing = [r for r in required if r not in files]
    if missing:
        raise RuntimeError(f"{slug} has no {missing} (files: {sorted(files)}). Is the Dataset populated?")

    restored = []
    for name, size in files.items():
        path = dest / name
        for attempt in range(1, attempts + 1):
            # --force: the CLI otherwise skips a file whose local copy looks newer.
            rc, out = _kaggle("datasets", "download", "-d", slug, "-f", name, "-p", str(dest), "--force")
            if rc == 0 and path.exists() and path.stat().st_size == size:
                restored.append(name)
                print(f"  {name} ({size / 1e6:,.1f} MB)")
                break
            print(f"  {name}: attempt {attempt}/{attempts} failed (rc={rc}); retrying in {wait}s...\n{out[-400:]}")
            time.sleep(wait)
        else:
            raise RuntimeError(f"Could not download {name} from {slug}.\nLast output:\n{out}")
    return restored
