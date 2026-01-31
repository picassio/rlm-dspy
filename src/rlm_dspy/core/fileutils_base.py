"""Base file utilities - path safety, OS detection, linking, deletion."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)


class PathTraversalError(Exception):
    """Raised when a path traversal attack is detected."""
    pass


def validate_path_safety(path: Path, base_dir: Path | None = None) -> Path:
    """Validate that a path is safe (no traversal attacks)."""
    resolved = path.resolve()
    path_str = str(path)
    
    if ".." in path_str:
        logger.warning("Path traversal attempt detected: %s", path_str)
        raise PathTraversalError(f"Path contains traversal sequence: {path_str}")

    if base_dir is not None:
        base_resolved = base_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            logger.warning("Path escapes base directory: %s not in %s", resolved, base_resolved)
            raise PathTraversalError(f"Path {resolved} is outside base directory {base_resolved}")

        try:
            common = os.path.commonpath([str(base_resolved), str(resolved)])
            if common != str(base_resolved):
                raise PathTraversalError(f"Path {resolved} escapes base directory via symlink")
        except ValueError:
            raise PathTraversalError(f"Path {resolved} is on different drive than {base_resolved}")

    return resolved


def is_windows() -> bool:
    return sys.platform == "win32"


def is_macos() -> bool:
    return sys.platform == "darwin"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def get_cache_dir(app_name: str = "rlm_dspy") -> Path:
    """Get platform-specific cache directory."""
    if is_windows():
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif is_macos():
        base = Path.home() / "Library" / "Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache) if xdg_cache else Path.home() / ".cache"

    cache_dir = base / app_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def smart_link(source: Path, target: Path, force: bool = False) -> None:
    """Create a link, preferring hard links, falling back to copy."""
    source = source.resolve()
    target = target.resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")

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
        tgt = target / item.name
        if item.is_dir():
            _recursive_hard_link(item, tgt)
        else:
            os.link(item, tgt)


def smart_rmtree(path: Path, aggressive: bool = False) -> bool:
    """Remove directory tree with better error handling."""
    if not path.exists():
        return True

    def onerror(func, path_str, exc_info):
        try:
            Path(path_str).chmod(0o777)
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


def _kill_blocking_processes(path: Path, graceful_timeout: float = 2.0) -> None:
    """Kill processes blocking a path (Unix only).
    
    Uses SIGTERM first for graceful shutdown, then SIGKILL after timeout.
    Only kills processes that are children of the current process or
    processes explicitly blocking the target path.
    
    Args:
        path: Path being blocked
        graceful_timeout: Seconds to wait after SIGTERM before SIGKILL
    """
    import signal
    import time
    
    try:
        result = subprocess.run(
            ["lsof", "+D", str(path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        pids = set()
        current_pid = os.getpid()
        
        for line in result.stdout.strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) > 1:
                try:
                    pid = int(parts[1])
                    # Don't kill self or init/system processes
                    if pid != current_pid and pid > 1:
                        pids.add(pid)
                except ValueError:
                    pass
        
        if not pids:
            return
        
        # Try graceful termination first (SIGTERM)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.debug("Sent SIGTERM to PID %d blocking %s", pid, path)
            except ProcessLookupError:
                pids.discard(pid)
            except PermissionError:
                logger.warning("No permission to kill PID %d", pid)
                pids.discard(pid)
        
        # Wait for graceful shutdown
        if pids:
            time.sleep(graceful_timeout)
        
        # Force kill remaining processes (SIGKILL)
        for pid in pids:
            try:
                # Check if still running
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
                logger.warning("Sent SIGKILL to PID %d (didn't respond to SIGTERM)", pid)
            except ProcessLookupError:
                pass  # Already dead
            except PermissionError:
                pass
                
    except FileNotFoundError:
        # lsof not available
        pass
    except Exception as e:
        logger.debug("Failed to kill blocking processes: %s", e)


def sync_directory(source: Path, target: Path, delete_extra: bool = True, dry_run: bool = False) -> dict:
    """Sync source directory to target."""
    source = source.resolve()
    target = target.resolve()

    if not source.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source}")

    stats = {"created": 0, "updated": 0, "deleted": 0, "unchanged": 0, "errors": 0}

    if not dry_run:
        target.mkdir(parents=True, exist_ok=True)

    source_files = {f.relative_to(source): f for f in source.rglob("*") if f.is_file()}
    target_files = {f.relative_to(target): f for f in target.rglob("*") if f.is_file()} if target.exists() else {}

    for rel_path, src_file in source_files.items():
        tgt_file = target / rel_path
        try:
            if rel_path not in target_files:
                if not dry_run:
                    tgt_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, tgt_file)
                stats["created"] += 1
            elif src_file.stat().st_mtime > target_files[rel_path].stat().st_mtime:
                if not dry_run:
                    shutil.copy2(src_file, tgt_file)
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1
        except Exception as e:
            logger.warning("Failed to sync %s: %s", rel_path, e)
            stats["errors"] += 1

    if delete_extra:
        extra = set(target_files.keys()) - set(source_files.keys())
        for rel_path in extra:
            try:
                if not dry_run:
                    (target / rel_path).unlink()
                stats["deleted"] += 1
            except Exception as e:
                logger.warning("Failed to delete %s: %s", rel_path, e)
                stats["errors"] += 1

    return stats


def path_to_module(path: Path, root: Path | None = None) -> str:
    """Convert file path to Python module path."""
    if root:
        try:
            rel_path = path.resolve().relative_to(root.resolve())
        except ValueError:
            rel_path = path
    else:
        rel_path = path

    parts = list(rel_path.parts)
    if parts and parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]
        if parts[-1] == '__init__':
            parts = parts[:-1]

    return '.'.join(parts)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def atomic_write(
    path: Path,
    mode: str = "w",
    encoding: str | None = "utf-8",
    backup: bool = False,
    backup_suffix: str = ".bak",
    fsync: bool = True,
) -> Generator:
    """Context manager for atomic file writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
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
            backup_path = path.with_suffix(path.suffix + backup_suffix)
            shutil.copy2(path, backup_path)

        tmp_file.replace(path)

    except Exception:
        try:
            tmp_file.unlink()
        except Exception:
            pass
        raise
