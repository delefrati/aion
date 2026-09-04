"""Publish and fetch tokenized data caches via GitHub Releases.

The tokenized cache (train.bin + val.bin + tokenizer.json) is stored as assets on a
GitHub Release so Colab and Kaggle can pull the *same* data from one public URL with no
auth, no per-file download quotas, and no Kaggle/Drive credentials.

GitHub Releases allow up to 2 GB per asset and unlimited assets, and don't count against
repo size. Files larger than the part size are split into ``<name>.partNN`` chunks and a
``manifest.json`` records how to reassemble and verify them.

Publish (needs the `gh` CLI, authenticated once with `gh auth login`):
    python -m llm_lab.data.remote publish \
        --repo <owner>/<repo> --tag pretrain-tokenized-v1 \
        --dir /tmp/data --files train.bin val.bin tokenizer.json

Fetch (pure Python, no gh/auth needed for a public repo):
    python -m llm_lab.data.remote fetch \
        --repo <owner>/<repo> --tag pretrain-tokenized-v1 --dir /tmp/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MANIFEST_NAME = "manifest.json"
# Stay safely under GitHub's 2 GB per-asset hard limit.
DEFAULT_PART_SIZE_MB = 1900
_CHUNK = 8 * 1024 * 1024  # 8 MB streaming buffer


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _split(path: Path, part_size: int) -> list[Path]:
    """Split a file into <name>.partNN chunks; return the chunk paths in order."""
    parts: list[Path] = []
    with path.open("rb") as f:
        idx = 0
        while True:
            data = f.read(part_size)
            if not data:
                break
            part = path.with_name(f"{path.name}.part{idx:02d}")
            part.write_bytes(data)
            parts.append(part)
            idx += 1
    return parts


# --------------------------------------------------------------------------- publish


def publish(
    repo: str,
    tag: str,
    src_dir: Path,
    files: list[str],
    title: str | None = None,
    notes: str | None = None,
    part_size_mb: int = DEFAULT_PART_SIZE_MB,
) -> None:
    """Split large files, build a manifest, and upload everything as a GitHub Release."""
    src_dir = Path(src_dir)
    part_size = part_size_mb * 1024 * 1024
    assets: list[Path] = []
    manifest: dict[str, dict] = {"files": {}}

    for name in files:
        path = src_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found")
        size = path.stat().st_size
        entry = {"size": size, "sha256": _sha256(path)}
        if size > part_size:
            parts = _split(path, part_size)
            entry["parts"] = [p.name for p in parts]
            assets.extend(parts)
            print(f"{name}: {size/1e9:.2f} GB -> {len(parts)} parts")
        else:
            entry["parts"] = [name]
            assets.append(path)
            print(f"{name}: {size/1e6:.1f} MB")
        manifest["files"][name] = entry

    manifest_path = src_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    assets.append(manifest_path)

    title = title or f"Tokenized cache {tag}"
    notes = notes or f"AION tokenized data cache: {', '.join(files)}"

    # Create the (empty) release if it doesn't exist yet — assets are uploaded one at a
    # time below so a dropped connection only costs the in-flight asset, not a re-upload
    # of everything: a single `gh release create <assets...>` call has no way to record
    # which assets got through before it died, so a retry re-sends the lot with --clobber.
    create = subprocess.run(
        ["gh", "release", "create", tag, "--repo", repo, "--title", title, "--notes", notes],
        capture_output=True, text=True,
    )
    if create.returncode != 0:
        stderr = (create.stderr or "").lower()
        if "already exists" not in stderr and "release.tag_name already_exists" not in stderr:
            raise RuntimeError(f"gh release create failed:\n{create.stderr}")
        print(f"Release {tag} already exists; resuming asset upload.")

    # Skip assets already on the release at the right size, so a re-run after a partial
    # failure only sends what didn't make it last time.
    existing: dict[str, int] = {}
    view = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
        capture_output=True, text=True,
    )
    if view.returncode == 0:
        existing = {a["name"]: a["size"] for a in json.loads(view.stdout)["assets"]}

    for asset in assets:
        if existing.get(asset.name) == asset.stat().st_size:
            print(f"{asset.name}: already on release, skipping")
            continue
        print(f"Uploading {asset.name}...")
        up = subprocess.run(
            ["gh", "release", "upload", tag, str(asset), "--repo", repo, "--clobber"],
            capture_output=True, text=True,
        )
        if up.returncode != 0:
            raise RuntimeError(
                f"gh release upload failed on {asset.name}:\n{up.stderr}\n"
                "Already-uploaded assets are kept — re-run publish() to resume."
            )

    # Chunk files are throwaway once uploaded.
    for entry in manifest["files"].values():
        if len(entry["parts"]) > 1:
            for part in entry["parts"]:
                (src_dir / part).unlink(missing_ok=True)

    print(f"Published {len(files)} file(s) to {repo}@{tag}")


# ----------------------------------------------------------------------------- fetch


def _asset_url(repo: str, tag: str, name: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Drop the GitHub Authorization header when redirected to a different host.

    Private release-asset downloads 302 to a pre-signed storage URL that carries its own
    auth; forwarding the bearer token makes storage reject the request.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and (
            urllib.parse.urlsplit(newurl).netloc
            != urllib.parse.urlsplit(req.full_url).netloc
        ):
            for h in ("Authorization", "authorization"):
                new.headers.pop(h, None)
        return new


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_StripAuthOnRedirect)


def _download(
    url: str,
    dest: Path,
    headers: dict | None = None,
    retries: int = 4,
    timeout: int = 60,
) -> None:
    # `timeout` bounds each individual socket operation, not the whole transfer, so a
    # multi-GB part still downloads fine as long as bytes keep arriving. Without it
    # urllib blocks forever on a stalled connection: the retry loop below never fires
    # (a hang raises nothing), and a notebook cell just sits there with no error.
    opener = _opener()
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "aion-llm-lab", **(headers or {})}
            )
            with opener.open(req, timeout=timeout) as resp, dest.open("wb") as out:
                while True:
                    block = resp.read(_CHUNK)
                    if not block:
                        break
                    out.write(block)
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            if attempt == retries:
                raise
            print(f"  download attempt {attempt} failed ({exc}); retrying in 5s...")
            time.sleep(5)


def _release_assets(repo: str, tag: str, token: str) -> dict[str, str]:
    """Map asset name -> authenticated API download URL for a release (private-repo safe)."""
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "aion-llm-lab",
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    })
    with _opener().open(req) as resp:
        data = json.load(resp)
    return {a["name"]: a["url"] for a in data.get("assets", [])}


def fetch(repo: str, tag: str, dest_dir: Path, token: str | None = None) -> bool:
    """Download + reassemble the cache from a release. Returns True on success.

    Public repo: anonymous download from the browser URL. Private repo: pass ``token``
    (or set GITHUB_TOKEN / GH_TOKEN) to download through the authenticated GitHub API.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if token:
        try:
            assets = _release_assets(repo, tag, token)
        except Exception as exc:  # noqa: BLE001 - caller falls back to rebuild
            print(f"No release {repo}@{tag} ({exc}).")
            return False

        def _get(name: str, dest: Path) -> None:
            if name not in assets:
                raise FileNotFoundError(f"asset {name!r} not in release {tag}")
            _download(assets[name], dest, headers={
                "Accept": "application/octet-stream",
                "Authorization": f"Bearer {token}",
            })
    else:
        def _get(name: str, dest: Path) -> None:
            _download(_asset_url(repo, tag, name), dest)

    manifest_path = dest_dir / MANIFEST_NAME
    try:
        _get(MANIFEST_NAME, manifest_path)
    except Exception as exc:  # noqa: BLE001 - fall back to whatever the caller does next
        print(f"No release manifest for {repo}@{tag} ({exc}).")
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        target = dest_dir / name
        if target.is_file() and target.stat().st_size == entry["size"]:
            print(f"{name}: already present, skipping")
            continue

        parts = entry["parts"]
        if len(parts) == 1:
            print(f"{name}: downloading ({entry['size']/1e6:.1f} MB)")
            _get(parts[0], target)
        else:
            print(f"{name}: downloading {len(parts)} parts ({entry['size']/1e9:.2f} GB)")
            with target.open("wb") as out:
                for part in parts:
                    tmp = dest_dir / part
                    _get(part, tmp)
                    out.write(tmp.read_bytes())
                    tmp.unlink(missing_ok=True)

        digest = _sha256(target)
        if digest != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch for {name}: {digest} != {entry['sha256']}")
        print(f"{name}: OK")

    print(f"Fetched {len(manifest['files'])} file(s) from {repo}@{tag}")
    return True


# ------------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm_lab.data.remote")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("publish", help="Split, manifest, and upload a cache as a release")
    p.add_argument("--repo", required=True, help="owner/repo, e.g. <owner>/aion-datasets")
    p.add_argument("--tag", required=True, help="release tag, e.g. pretrain-tokenized-v1")
    p.add_argument("--dir", required=True, help="directory containing the files")
    p.add_argument("--files", nargs="+", default=["train.bin", "val.bin", "tokenizer.json"])
    p.add_argument("--title", default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--part-size-mb", type=int, default=DEFAULT_PART_SIZE_MB)

    p = sub.add_parser("fetch", help="Download + reassemble a cache from a release")
    p.add_argument("--repo", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dir", required=True, help="destination directory")
    p.add_argument("--token", default=None,
                   help="GitHub token for a private repo (else GITHUB_TOKEN/GH_TOKEN env)")

    args = parser.parse_args(argv)

    if args.command == "publish":
        publish(args.repo, args.tag, Path(args.dir), args.files,
                title=args.title, notes=args.notes, part_size_mb=args.part_size_mb)
        return 0
    if args.command == "fetch":
        ok = fetch(args.repo, args.tag, Path(args.dir), token=args.token)
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
