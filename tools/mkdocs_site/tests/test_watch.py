import os
from pathlib import Path, PurePosixPath

from tools.mkdocs_site.watch import TreeSnapshot, snapshot_tree, watch_changes


def test_snapshot_is_sorted_and_uses_relative_paths(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    (wiki / "nested").mkdir(parents=True)
    (wiki / "z.md").write_text("# Z", encoding="utf-8")
    (wiki / "nested" / "a.svg").write_text("<svg/>", encoding="utf-8")

    snapshot = snapshot_tree(wiki)

    assert snapshot.files == (
        PurePosixPath("nested/a.svg"),
        PurePosixPath("z.md"),
    )


def test_snapshot_changes_for_create_modify_move_and_delete(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    first = snapshot_tree(wiki)
    page = wiki / "a.md"
    page.write_text("# A", encoding="utf-8")
    second = snapshot_tree(wiki)
    page.write_text("# A changed", encoding="utf-8")
    stat = page.stat()
    os.utime(page, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    third = snapshot_tree(wiki)
    page.rename(wiki / "b.md")
    fourth = snapshot_tree(wiki)
    (wiki / "b.md").unlink()
    fifth = snapshot_tree(wiki)

    assert len({item.digest for item in (first, second, third, fourth, fifth)}) == 5


def test_watch_debounces_a_stable_change_once() -> None:
    snapshots = iter(
        [
            TreeSnapshot("old", ()),
            TreeSnapshot("changing", (PurePosixPath("a.md"),)),
            TreeSnapshot("stable", (PurePosixPath("a.md"),)),
            TreeSnapshot("stable", (PurePosixPath("a.md"),)),
        ]
    )
    sleeps: list[float] = []
    callbacks: list[str] = []
    stopped = False

    def callback() -> None:
        nonlocal stopped
        callbacks.append("refresh")
        stopped = True

    watch_changes(
        Path("wiki"),
        callback,
        stop_requested=lambda: stopped,
        snapshot=lambda _root: next(snapshots),
        sleep=sleeps.append,
    )

    assert callbacks == ["refresh"]
    assert sleeps == [0.5, 0.25, 0.25]


def test_watch_prints_callback_failure_and_continues_to_next_change(
    capsys,
) -> None:
    snapshots = iter(
        [
            TreeSnapshot("old", ()),
            TreeSnapshot("first", (PurePosixPath("a.md"),)),
            TreeSnapshot("first", (PurePosixPath("a.md"),)),
            TreeSnapshot("second", (PurePosixPath("b.md"),)),
            TreeSnapshot("second", (PurePosixPath("b.md"),)),
        ]
    )
    attempts = 0
    stopped = False

    def callback() -> None:
        nonlocal attempts, stopped
        attempts += 1
        if attempts == 1:
            raise RuntimeError("conversion exploded")
        stopped = True

    watch_changes(
        Path("wiki"),
        callback,
        stop_requested=lambda: stopped,
        snapshot=lambda _root: next(snapshots),
        sleep=lambda _seconds: None,
    )

    assert attempts == 2
    assert "conversion exploded" in capsys.readouterr().err
