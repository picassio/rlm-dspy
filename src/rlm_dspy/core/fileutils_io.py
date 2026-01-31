"""File I/O utilities - linking, deletion, atomic writes, sync."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from .fileutils_path import is_windows, is_macos

logger = logging.getLogger(__name__)


def smart_link(source: Path, target: Path, force: bool = False) -> None:
    """Create a link from source to target, preferring hard links.

    Falls back to copy on Windows or cross-device scenarios.

    Args:
        source: Source file/directory
        target: Target link location  
        force: Remove target if exists
    """
    source = source.resolve()
    target = target.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if target.exists():
        if force:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        else:
            raise FileExistsError(f"Target already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)

    if source.is_file():
        try:
            os.link(source, target)
            return
        except OSError:
            pass
        shutil.copy2(source, target)
    else:
        try:
            _recursive_hard_link(source, target)
        except OSError:
            shutil.copytree(source, target)


def _recursive_hard_link(source: Path, target: Path) -> None:
    """Recursively hard link a directory tree."""
    target.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        src_item = item
        tgt_item = target / item.name

        if src_item.is_dir():
            _recursive_hard_link(src_item, tgt_item)
        else:
            os.link(src_item, tgt_item)


def smart_rmtree(path: Path, aggressive: bool = False) -> bool:
    """Remove a directory tree with better error handling.

    Args:
        path: Directory to remove
        aggressive: If True, try harder (kill blocking processes on macOS)

    Returns:
        True if successfully removed
    """
    if not path.exists():
        return True

    def onerror(func, path_str, exc_info):
        p = Path(path_str)
        try:
            p.chmod(0o777)
            func(path_str)
        except Exception:
            logger.warning("Cannot remove: %s", path_str)

    try:
        shutil.rmtree(path, onerror=onerror)
        return True
    except Exception as e:
        logger.warning("Initial rmtree failed: %s", e)

    if aggressive and is_macos():
        _kill_blocking_processes(path)
        try:
            shutil.rmtree(path, onerror=onerror)
            return True
        except Exception:
            pass

    return not path.exists()


def _kill_blocking_processes(path: Path) -> None:
    """Kill processes blocking a path (macOS only)."""
    import subprocess
    try:
        result = subprocess.run(
            ["lsof", "+D", str(path)],
            capture_output=True, text=True, timeout=10
        )
        pids = set()
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) > 1:
                try:
                    pids.add(int(parts[1]))
                except ValueError:
                    pass
        for pid in pids:
            if pid != os.getpid():
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
    except Exception:
        pass


def sync_directory(
    source: Path,
    target: Path,
    delete_extra: bool = True,
    dry_run: bool = False,
) -> dict:
    """Sync source directory to target.

    Args:
        source: Source directory
        target: Target directory
        delete_extra: Delete files in target not in source
        dry_run: Don't actually modify files

    Returns:
        Dict with counts: created, updated, deleted, unchanged
    """
    source = source.resolve()
    target = target.resolve()

    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source}")

    stats = {"created": 0, "updated": 0, "deleted": 0, "unchanged": 0}

    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)

    source_files = {f.relative_to(source) for f in source.rglob("*") if f.is_file()}
    target_files = {f.relative_to(target) for f in target.rglob("*") if f.is_file()} if target.exists() else set()

    for rel_path in source_files:
        src_file = source / rel_path
        tgt_file = target / rel_path

        if not tgt_file.exists():
            if not dry_run:
                tgt_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, tgt_file)
            stats["created"] += 1
        elif src_file.stat().st_mtime > tgt_file.stat().st_mtime:
            if not dry_run:
                shutil.copy2(src_file, tgt_file)
            stats["updated"] += 1
        else:
            stats["unchanged"] += 1

    if delete_extra:
        extra = target_files - source_files
        for rel_path in extra:
            tgt_file = target / rel_path
            if not dry_run:
                tgt_file.unlink()
            stats["deleted"] += 1

    return stats


@contextmanager
def atomic_write(
    path: Path,
    mode: str = "w",
    encoding: str | None = "utf-8",
    backup: bool = False,
    fsync: bool = True,
) -> Generator:
    """Context manager for atomic file writes.

    Args:
        path: Target file path
        mode: Write mode ('w' or 'wb')
        encoding: Text encoding (None for binary)
        backup: Keep backup of original file
        fsync: Call fsync before closing

    Yields:
        File handle to write to

    Example:
        with atomic_write(Path("config.json")) as f:
            json.dump(data, f)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )
    tmp_file = Path(tmp_path)

    try:
        if "b" in mode:
            f = os.fdopen(fd, mode)
        else:
            f = os.fdopen(fd, mode, encoding=encoding)

        try:
            yield f

            if fsync:
                f.flush()
                os.fsync(f.fileno())
        finally:
            f.close()

        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)

        tmp_file.replace(path)

    except Exception:
        try:
            tmp_file.unlink()
        except Exception:
            pass
        raise
