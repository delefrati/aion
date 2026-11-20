"""AION-1: one Store interface; the local-directory, Google Drive and Kaggle backends pass
the same restore/push tests, and the Kaggle one keeps kaggle_io's rules."""
from __future__ import annotations

from pathlib import Path

import pytest

from llm_lab.storage import DriveStore, KaggleStore, LocalStore, StoreError


@pytest.fixture(params=["local", "drive", "kaggle"])
def make_store(request, tmp_path):
    """make_store(run_name) -> a Store for that run, on the backend under test."""
    if request.param == "local":
        return lambda run: LocalStore(tmp_path / "stores" / run)
    if request.param == "drive":
        mount = tmp_path / "drive"
        (mount / "MyDrive").mkdir(parents=True)
        return lambda run: DriveStore(mount / "MyDrive" / "aion_checkpoints" / run, mount=mount)
    request.getfixturevalue("fake_kaggle")
    return lambda run: KaggleStore(f"someone/{run}", wait=0, push_wait=0)


def _files(d: Path, **contents) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, body in contents.items():
        p = d / name.replace("__", ".")
        p.write_bytes(body if isinstance(body, bytes) else body.encode())
        out.append(p)
    return out


# --- the contract every backend meets -------------------------------------------------

def test_a_missing_store_is_none(make_store, tmp_path):
    store = make_store("never-pushed")
    assert store.files() is None
    assert store.download(tmp_path / "dest") is None


def test_push_then_restore_round_trips(make_store, tmp_path):
    store = make_store("run")
    pushed = _files(tmp_path / "src", step_5__safetensors=b"\x00\x01" * 1000,
                    step_5__state__json='{"step": 5}', COMPLETE='{"step": 5}')
    store.push(pushed, "step 5")
    assert store.files() == {p.name: p.stat().st_size for p in pushed}
    restored = store.download(tmp_path / "dest")
    assert sorted(restored) == sorted(p.name for p in pushed)
    for p in pushed:
        assert (tmp_path / "dest" / p.name).read_bytes() == p.read_bytes()


def test_a_push_replaces_the_whole_content(make_store, tmp_path):
    store = make_store("run")
    store.push(_files(tmp_path / "a", step_5__safetensors="old", COMPLETE="5"), "step 5")
    store.push(_files(tmp_path / "b", step_9__safetensors="new", COMPLETE="9"), "step 9")
    assert set(store.files()) == {"step_9.safetensors", "COMPLETE"}


def test_download_only_and_required(make_store, tmp_path):
    store = make_store("run")
    store.push(_files(tmp_path / "src", a__bin="a", b__bin="bb"), "x")
    assert store.download(tmp_path / "d1", only=["b.bin"]) == ["b.bin"]
    assert not (tmp_path / "d1" / "a.bin").exists()
    with pytest.raises(StoreError, match="c.bin"):
        store.download(tmp_path / "d2", required=["a.bin", "c.bin"])


# --- backend-specific -------------------------------------------------------------------

def test_an_unmounted_drive_is_an_error_not_a_missing_run(tmp_path):
    store = DriveStore(tmp_path / "drive" / "MyDrive" / "run", mount=tmp_path / "drive")
    with pytest.raises(StoreError, match="mounted"):
        store.files()
    with pytest.raises(StoreError, match="mounted"):
        store.push([], "x")


def test_local_push_leaves_no_temp_files_and_writes_markers_last(tmp_path, monkeypatch):
    import os
    order = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append(Path(b).name), real(a, b)))
    store = LocalStore(tmp_path / "store")
    store.push(_files(tmp_path / "src", COMPLETE="1", step_1__safetensors="w", BEST="1"), "x")
    assert order[-2:] in (["COMPLETE", "BEST"], ["BEST", "COMPLETE"])
    assert not list((tmp_path / "store").glob("*.tmp"))


def test_kaggle_strips_the_notebook_token(fake_kaggle, tmp_path, monkeypatch):
    """The real CLI in a notebook would 403 the listing and 401 the upload with this set."""
    monkeypatch.setenv("KAGGLE_API_V1_TOKEN", "/kaggle/notebook-token")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    store = KaggleStore("someone/run", wait=0, push_wait=0)
    store.push(_files(tmp_path / "src", a__bin="a"), "x")  # creates it
    assert store.files() == {"a.bin": 1}
    assert store.download(tmp_path / "dest") == ["a.bin"]
    assert all(not c["token"] for c in fake_kaggle.calls())


def test_kaggle_403_is_missing_but_401_and_server_errors_raise(fake_kaggle, tmp_path):
    store = KaggleStore("someone/run", attempts=2, wait=0, push_wait=0)
    assert store.files() is None  # 403: no such Dataset
    store.push(_files(tmp_path / "src", a__bin="a"), "x")

    fake_kaggle.script(files=["401", "401"])
    with pytest.raises(StoreError):
        store.files()
    fake_kaggle.script(files=["500", "500"])
    with pytest.raises(StoreError):
        store.files()
    fake_kaggle.script(files=["500"])  # a transient error is retried
    assert store.files() == {"a.bin": 1}


def test_kaggle_retries_a_failed_file(fake_kaggle, tmp_path):
    store = KaggleStore("someone/run", attempts=3, wait=0, push_wait=0)
    store.push(_files(tmp_path / "src", a__bin="aaaa", b__bin="b"), "x")
    fake_kaggle.script(**{"download:a.bin": ["500", "500"]})
    assert sorted(store.download(tmp_path / "dest")) == ["a.bin", "b.bin"]
    fake_kaggle.script(**{"download:a.bin": ["500"] * 3})
    with pytest.raises(StoreError, match="a.bin"):
        store.download(tmp_path / "dest2")


def test_kaggle_push_retries_and_never_reports_a_failed_push(fake_kaggle, tmp_path):
    store = KaggleStore("someone/run", wait=0, push_attempts=3, push_wait=0)
    store.push(_files(tmp_path / "v1", a__bin="1"), "v1")

    fake_kaggle.script(push=["401", "auth0"])  # two failures, then accepted
    store.push(_files(tmp_path / "v2", a__bin="22"), "v2")
    assert store.files() == {"a.bin": 2}

    fake_kaggle.script(push=["401", "auth0", "500"])
    with pytest.raises(StoreError):
        store.push(_files(tmp_path / "v3", a__bin="333"), "v3")
    assert store.files() == {"a.bin": 2}  # still the last accepted version


def test_kaggle_io_download_dataset_still_works(fake_kaggle, tmp_path):
    from llm_lab.kaggle_io import download_dataset
    assert download_dataset("someone/run", tmp_path / "d", wait=0) is None
    KaggleStore("someone/run", wait=0).push(_files(tmp_path / "src", latest__pt="w"), "x")
    assert download_dataset("someone/run", tmp_path / "d", wait=0) == ["latest.pt"]


def _attached(fake_kaggle, tmp_path, layout="flat"):
    """A Dataset me/data in the fake Kaggle, and a copy of it as an attached input."""
    from llm_lab.storage import KaggleStore
    src = tmp_path / "src"
    src.mkdir()
    (src / "train.bin").write_bytes(b"x" * 100)
    (src / "val.bin").write_bytes(b"y" * 10)
    KaggleStore("me/data", wait=0, push_wait=0).push(sorted(src.iterdir()))
    root = tmp_path / "input"
    mount = root / "data" if layout == "flat" else root / "datasets" / "me" / "data"
    mount.mkdir(parents=True)
    for f in src.iterdir():
        (mount / f.name).write_bytes(f.read_bytes())
    return root, mount


@pytest.mark.parametrize("layout", ["flat", "nested"])
def test_kaggle_links_an_attached_input_instead_of_downloading(fake_kaggle, tmp_path, layout):
    from llm_lab.storage import KaggleStore
    root, mount = _attached(fake_kaggle, tmp_path, layout)
    store = KaggleStore("me/data", wait=0, use_mounted=True, mount_roots=[root])
    assert sorted(store.download(tmp_path / "dest")) == ["train.bin", "val.bin"]
    assert (tmp_path / "dest" / "train.bin").resolve() == (mount / "train.bin").resolve()
    assert not [c for c in fake_kaggle.calls() if "download" in c["argv"]]


def test_kaggle_downloads_what_the_attached_input_lacks_or_differs_in(fake_kaggle, tmp_path):
    from llm_lab.storage import KaggleStore
    root, mount = _attached(fake_kaggle, tmp_path)
    (mount / "val.bin").write_bytes(b"y" * 9)  # an older version: the size differs
    (mount / "train.bin").unlink()
    store = KaggleStore("me/data", wait=0, use_mounted=True, mount_roots=[root])
    store.download(tmp_path / "dest")
    dest = tmp_path / "dest"
    assert not (dest / "val.bin").is_symlink() and (dest / "val.bin").stat().st_size == 10
    assert not (dest / "train.bin").is_symlink() and (dest / "train.bin").stat().st_size == 100


def test_kaggle_ignores_attached_inputs_unless_asked(fake_kaggle, tmp_path):
    from llm_lab.storage import KaggleStore
    root, _ = _attached(fake_kaggle, tmp_path)
    KaggleStore("me/data", wait=0, mount_roots=[root]).download(tmp_path / "dest")
    assert not (tmp_path / "dest" / "train.bin").is_symlink()
