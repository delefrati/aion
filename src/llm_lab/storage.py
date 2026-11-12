"""Where a run's checkpoints and data live between sessions: one interface, three backends.

A Store is a flat set of files (no subdirectories) that changes as a whole:

    files()                     {name: size}, or None if the store doesn't exist
    download(dest, only, required)  copy files into dest; None if the store doesn't exist
    push(paths, message)        make the store's content exactly these files

"Doesn't exist" is a definite answer, never a guess: a lookup that fails for any other
reason (an unmounted Drive, a network error, a bad credential) raises StoreError. Callers
decide what a missing store means; seeding a run over a lookup that merely failed is how
real progress got overwritten before.

Backends:
    LocalStore(path)                  a directory
    DriveStore(path, mount)           a directory on a mounted Google Drive (Colab)
    KaggleStore(slug, title)          a Kaggle Dataset, through the kaggle CLI
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable


class StoreError(RuntimeError):
    """The store couldn't be read or written. Not the same as the store being missing."""


def _check_required(name: str, files: dict[str, int], required: Iterable[str]) -> None:
    missing = [r for r in required if r not in files]
    if missing:
        raise StoreError(f"{name} has no {missing} (files: {sorted(files)}). Is it populated?")


def _order(paths: list[Path]) -> list[Path]:
    """Markers (no suffix, all caps: COMPLETE, BEST) go last, after the files they name."""
    return sorted(paths, key=lambda p: p.name.isupper() and "." not in p.name)


class Store:
    name: str

    def files(self) -> dict[str, int] | None:
        raise NotImplementedError

    def download(self, dest, only: Iterable[str] | None = None,
                 required: Iterable[str] = ()) -> list[str] | None:
        raise NotImplementedError

    def push(self, paths: Iterable, message: str = "") -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.name


class LocalStore(Store):
    """A directory. A push copies each file to a temp name and renames it into place (markers
    last), then deletes the files that weren't pushed, so a reader never sees half a file."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = str(self.path)

    def _available(self) -> None:
        pass

    def files(self) -> dict[str, int] | None:
        self._available()
        if not self.path.is_dir():
            return None
        return {p.name: p.stat().st_size for p in self.path.iterdir()
                if p.is_file() and not p.name.endswith(".tmp")}

    def download(self, dest, only=None, required=()):
        files = self.files()
        if files is None:
            return None
        _check_required(self.name, files, required)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        names = [n for n in files if only is None or n in set(only)]
        for n in names:
            shutil.copyfile(self.path / n, dest / n)
            if (dest / n).stat().st_size != files[n]:
                raise StoreError(f"{self.name}/{n}: copied size differs from the store's")
        return names

    def push(self, paths, message=""):
        self._available()
        paths = _order([Path(p) for p in paths])
        self.path.mkdir(parents=True, exist_ok=True)
        try:
            for p in paths:
                tmp = self.path / (p.name + ".tmp")
                shutil.copyfile(p, tmp)
                os.replace(tmp, self.path / p.name)
            keep = {p.name for p in paths}
            for old in self.path.iterdir():
                if old.is_file() and old.name not in keep:
                    old.unlink()
        except OSError as e:
            raise StoreError(f"push to {self.name} failed: {e}") from e


class DriveStore(LocalStore):
    """A directory on Google Drive mounted at `mount` (Colab: /content/drive). An unmounted
    Drive raises instead of looking like an empty or missing run."""

    def __init__(self, path, mount="/content/drive"):
        super().__init__(path)
        self.mount = Path(mount)
        self.name = f"drive:{self.path}"

    def _available(self) -> None:
        if not (os.path.ismount(self.mount) or (self.mount / "MyDrive").is_dir()):
            raise StoreError(f"Google Drive isn't mounted at {self.mount}: "
                             "run drive.mount() first.")


class KaggleStore(Store):
    """A Kaggle Dataset, `owner/name`, through the kaggle CLI.

    Files are downloaded one at a time (`-f name`): Kaggle often never builds the whole-
    Dataset zip for a multi-GB checkpoint Dataset, and the API then answers 404 "No gcs url
    found" for every version, forever (aion-transformer-tpu-large v97/v98, 2026-09-22).
    Per-file downloads redirect straight to the stored file, and skip the unzip.

    Inside a notebook the CLI authenticates with the notebook-scoped token named by
    $KAGGLE_API_V1_TOKEN before KAGGLE_USERNAME/KAGGLE_KEY. That token 401s on uploads (the
    CLI prints "Authentication required" for any 401) and can't see private Datasets, which
    then list as 403, i.e. missing (kaggle-pretrain-tpu v358732932, 2026-10-16). Every CLI
    call here drops it, so the account key is used.

    The listing answers 403 (not 404) for a Dataset that doesn't exist, so a 403 means
    missing. A credential without access gets a 403 too; the listing output is printed so
    that case can be told apart. A 401 is a bad credential and raises.
    """

    NOTEBOOK_TOKENS = ("KAGGLE_API_V1_TOKEN", "KAGGLE_API_TOKEN")
    MISSING = ("403", "404", "not found", "doesn't exist", "does not exist")

    def __init__(self, slug: str, title: str | None = None, attempts: int = 6, wait: float = 20,
                 push_attempts: int = 3, push_wait: float = 30, cli: str = "kaggle"):
        self.slug = slug
        self.name = f"kaggle:{slug}"
        self.title = title or slug.split("/")[-1]
        self.attempts, self.wait = attempts, wait
        self.push_attempts, self.push_wait = push_attempts, push_wait
        self.cli = cli
        self._exists: bool | None = None

    def _run(self, *args) -> tuple[int, str]:
        env = {k: v for k, v in os.environ.items() if k not in self.NOTEBOOK_TOKENS}
        try:
            res = subprocess.run([self.cli, *args], capture_output=True, text=True, env=env,
                                 timeout=3600)
        except (OSError, subprocess.TimeoutExpired) as e:
            return 1, f"{type(e).__name__}: {e}"
        return res.returncode, ((res.stdout or "") + (res.stderr or "")).strip()

    def _list(self) -> tuple[dict[str, int] | None, str]:
        rc, out = self._run("datasets", "files", self.slug, "--csv")
        if rc != 0:
            return None, out
        files, seen_header = {}, False
        for line in out.splitlines():  # the CLI may print an "outdated version" warning first
            if line.startswith("name,size"):
                seen_header = True
            elif seen_header and line.strip():
                name, size = line.split(",")[:2]
                files[name] = int(size)
        return (files if seen_header else None), out

    def files(self) -> dict[str, int] | None:
        out = ""
        for attempt in range(1, self.attempts + 1):
            files, out = self._list()
            if files is not None:
                self._exists = True
                return files
            low = out.lower()
            bad_credential = any(a in low for a in ("401", "unauthorized", "authentication required"))
            if not bad_credential and any(m in low for m in self.MISSING):
                print(f"{self.slug}: treating as missing. The listing answered:\n{out[-400:]}")
                self._exists = False
                return None
            print(f"Listing {self.slug} failed (attempt {attempt}/{self.attempts}); "
                  f"retrying in {self.wait}s...\n{out[-400:]}")
            time.sleep(self.wait)
        raise StoreError(f"Could not list files of {self.slug}.\nLast output:\n{out}")

    def download(self, dest, only=None, required=()):
        files = self.files()
        if files is None:
            return None
        _check_required(self.slug, files, required)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if only is not None:
            files = {n: s for n, s in files.items() if n in set(only)}
        restored = []
        for name, size in files.items():
            path, out, rc = dest / name, "", 0
            for attempt in range(1, self.attempts + 1):
                # --force: the CLI otherwise skips a file whose local copy looks newer.
                rc, out = self._run("datasets", "download", "-d", self.slug, "-f", name,
                                    "-p", str(dest), "--force")
                if rc == 0 and path.exists() and path.stat().st_size == size:
                    restored.append(name)
                    print(f"  {name} ({size / 1e6:,.1f} MB)")
                    break
                print(f"  {name}: attempt {attempt}/{self.attempts} failed (rc={rc}); "
                      f"retrying in {self.wait}s...\n{out[-400:]}")
                time.sleep(self.wait)
            else:
                raise StoreError(f"Could not download {name} from {self.slug}.\nLast output:\n{out}")
        return restored

    def push(self, paths, message=""):
        """One new Dataset version holding exactly `paths` (created on the first push).
        Raises StoreError unless Kaggle accepted it."""
        import json
        if self._exists is None:
            self._exists = self.files() is not None
        paths = [Path(p) for p in paths]
        with tempfile.TemporaryDirectory(prefix="aion_push_", dir=_staging_root(paths)) as stage:
            stage = Path(stage)
            for p in paths:
                try:
                    os.link(p, stage / p.name)  # no copy of a multi-GB file when it can be helped
                except OSError:
                    shutil.copy2(p, stage / p.name)
            (stage / "dataset-metadata.json").write_text(json.dumps(
                {"title": self.title, "id": self.slug, "licenses": [{"name": "CC0-1.0"}]}))
            out = ""
            for attempt in range(1, self.push_attempts + 1):
                if self._exists:
                    cmd = ("datasets", "version", "-p", str(stage), "-m", message or "update",
                           "--dir-mode", "zip")
                else:
                    cmd = ("datasets", "create", "-p", str(stage), "--dir-mode", "zip")
                rc, out = self._run(*cmd)
                print(out)
                # The CLI prints its auth help and exits 1, but check the text too, in case
                # a future version degrades to a zero exit.
                if rc == 0 and "Authentication required" not in out:
                    self._exists = True
                    return
                print(f"[push] {self.slug}: FAILED (rc={rc}, attempt {attempt}/{self.push_attempts})")
                if attempt < self.push_attempts:
                    time.sleep(self.push_wait)
        raise StoreError(f"Push to {self.slug} failed {self.push_attempts} times.\nLast output:\n{out}")


def _staging_root(paths: list[Path]) -> str | None:
    """Stage beside the files' directory (same filesystem, so the hard links work), not in
    it: a staging dir left by a kill must not look like part of the run."""
    return str(paths[0].resolve().parent.parent) if paths else None
