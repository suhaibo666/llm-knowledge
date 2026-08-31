from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class TreeSnapshot:
    digest: str
    files: tuple[PurePosixPath, ...]


WatchInputs = Path | Sequence[Path]


def snapshot_tree(root: WatchInputs) -> TreeSnapshot:
    """Snapshot one source tree or an ordered set of source files/trees."""
    single_root = isinstance(root, Path)
    roots = (root,) if single_root else tuple(root)
    if not roots:
        raise ValueError("watch input set cannot be empty")
    digest = hashlib.sha256()
    files: list[PurePosixPath] = []
    for index, source in enumerate(roots):
        resolved = source.resolve()
        if single_root:
            logical_root = PurePosixPath(".")
        else:
            logical_root = PurePosixPath(f"input-{index:03d}") / source.name
        digest.update(logical_root.as_posix().encode("utf-8"))
        digest.update(b"\0")
        if not resolved.exists():
            digest.update(b"missing\0")
            continue
        root_metadata = resolved.stat()
        digest.update(str(root_metadata.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(root_metadata.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        if resolved.is_file():
            entries = (resolved,)
        else:
            entries = tuple(sorted(
                (path for path in resolved.rglob("*") if path.is_file()),
                key=lambda path: (
                    path.relative_to(resolved).as_posix().casefold(),
                    path.relative_to(resolved).as_posix(),
                ),
            ))
        for path in entries:
            if resolved.is_file():
                relative = (
                    PurePosixPath(resolved.name) if single_root else logical_root
                )
            else:
                nested = PurePosixPath(path.relative_to(resolved).as_posix())
                relative = nested if single_root else logical_root / nested
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
    root: WatchInputs,
    callback: Callable[[], None],
    *,
    stop_requested: Callable[[], bool] = lambda: False,
    poll_interval: float = 0.5,
    stable_debounce: float = 0.25,
    snapshot: Callable[[WatchInputs], TreeSnapshot] = snapshot_tree,
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
