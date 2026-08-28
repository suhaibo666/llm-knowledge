from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class TreeSnapshot:
    digest: str
    files: tuple[PurePosixPath, ...]


def snapshot_tree(root: Path) -> TreeSnapshot:
    """Snapshot every source file using stable relative-path metadata."""
    resolved = root.resolve()
    entries = sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: (
            path.relative_to(resolved).as_posix().casefold(),
            path.relative_to(resolved).as_posix(),
        ),
    )
    digest = hashlib.sha256()
    root_metadata = resolved.stat()
    digest.update(b".\0")
    digest.update(str(root_metadata.st_size).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(root_metadata.st_mtime_ns).encode("ascii"))
    digest.update(b"\0")
    files: list[PurePosixPath] = []
    for path in entries:
        relative = PurePosixPath(path.relative_to(resolved).as_posix())
        metadata = path.stat()
        files.append(relative)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(metadata.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
    return TreeSnapshot(digest.hexdigest(), tuple(files))


def watch_changes(
    root: Path,
    callback: Callable[[], None],
    *,
    stop_requested: Callable[[], bool] = lambda: False,
    poll_interval: float = 0.5,
    stable_debounce: float = 0.25,
    snapshot: Callable[[Path], TreeSnapshot] = snapshot_tree,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Invoke *callback* once after each stable source-tree change."""
    current = snapshot(root)
    while not stop_requested():
        sleep(poll_interval)
        if stop_requested():
            return
        observed = snapshot(root)
        if observed.digest == current.digest:
            continue

        while True:
            sleep(stable_debounce)
            if stop_requested():
                return
            stable = snapshot(root)
            if stable.digest == observed.digest:
                break
            observed = stable

        current = stable
        try:
            callback()
        except Exception as error:
            print(f"MkDocs preview refresh failed: {error}", file=sys.stderr)
