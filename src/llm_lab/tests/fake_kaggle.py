"""A stand-in for the `kaggle` CLI, for tests: Datasets are directories under $FAKE_KAGGLE_ROOT.

It answers the four calls KaggleStore makes, the way Kaggle does:
    datasets files SLUG --csv                      403 when the Dataset doesn't exist
    datasets download -d SLUG -f NAME -p DIR --force
    datasets version -p DIR -m MSG --dir-mode zip  a new version = exactly DIR's files
    datasets create  -p DIR --dir-mode zip

Like the real CLI in a notebook, it fails when $KAGGLE_API_V1_TOKEN is set: listings get a
403 and uploads a 401 ("Authentication required").

$FAKE_KAGGLE_ROOT/script.json queues failures, consumed one per call:
    {"files": ["500", "401"], "download:NAME": ["500"], "push": ["401", "auth0"]}
"500" is a server error, "401" a bad credential, "auth0" the auth message with exit 0.
Every call is appended to $FAKE_KAGGLE_ROOT/calls.jsonl.
"""
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(os.environ["FAKE_KAGGLE_ROOT"])


def dataset_dir(slug: str) -> Path:
    return ROOT / "datasets" / slug.replace("/", "__")


def scripted(key: str) -> str | None:
    f = ROOT / "script.json"
    if not f.exists():
        return None
    script = json.loads(f.read_text())
    queue = script.get(key) or []
    if not queue:
        return None
    action = queue.pop(0)
    f.write_text(json.dumps(script))
    return action


def fail(action: str) -> None:
    if action == "500":
        print("500 Server Error: Internal Server Error")
        sys.exit(1)
    if action == "401":
        print("401 Client Error: Unauthorized\nAuthentication required to call the Kaggle API")
        sys.exit(1)
    if action == "auth0":
        print("Authentication required to call the Kaggle API")
        sys.exit(0)


def main(argv: list[str]) -> None:
    with open(ROOT / "calls.jsonl", "a") as f:
        f.write(json.dumps({"argv": argv, "token": "KAGGLE_API_V1_TOKEN" in os.environ}) + "\n")
    token = "KAGGLE_API_V1_TOKEN" in os.environ
    assert argv[0] == "datasets", argv
    cmd, args = argv[1], argv[2:]
    opt = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith("-")}

    if cmd == "files":
        fail(scripted("files") or "")
        d = dataset_dir(args[0])
        if token or not d.is_dir():
            print("403 Client Error: Forbidden for url: https://api.kaggle.com/v1/datasets.DatasetApiService/ListDatasetFiles")
            sys.exit(1)
        print("Warning: Looks like you're using an outdated `kaggle` version")
        print("name,size,creationDate")
        for p in sorted(d.iterdir()):
            print(f"{p.name},{p.stat().st_size},2026-11-11 00:00:00")
    elif cmd == "download":
        name = opt["-f"]
        fail(scripted(f"download:{name}") or "")
        src = dataset_dir(opt["-d"]) / name
        if token or not src.is_file():
            print("404 Client Error: Not Found")
            sys.exit(1)
        Path(opt["-p"]).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, Path(opt["-p"]) / name)
    elif cmd in ("version", "create"):
        fail(scripted("push") or "")
        if token:
            print("Authentication required to call the Kaggle API")
            sys.exit(1)
        stage = Path(opt["-p"])
        meta = json.loads((stage / "dataset-metadata.json").read_text())
        d = dataset_dir(meta["id"])
        if cmd == "create" and d.exists():
            print("409 Conflict: Dataset already exists")
            sys.exit(1)
        if cmd == "version" and not d.exists():
            print("404 Client Error: Not Found")
            sys.exit(1)
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        for p in stage.iterdir():
            if p.name != "dataset-metadata.json":
                shutil.copyfile(p, d / p.name)
        print(f"Dataset {'version' if cmd == 'version' else 'creation'} successful: {meta['id']}")
    else:
        print(f"fake kaggle: unknown command {argv}")
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
