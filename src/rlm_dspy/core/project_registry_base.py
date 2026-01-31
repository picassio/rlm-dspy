"""Project registry base types - Project, RegistryConfig, and file locking."""

from __future__ import annotations

import fcntl
import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Generator


@dataclass
class Project:
    """A registered project with metadata."""
    name: str
    path: str
    alias: str | None = None
    indexed_at: str | None = None
    snippet_count: int = 0
    file_count: int = 0
    auto_watch: bool = False
    tags: list[str] = field(default_factory=list)
    path_hash: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.path_hash:
            self.path_hash = hashlib.md5(self.path.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(**data)

    def matches(self, query: str) -> bool:
        """Check if project matches a name, alias, or tag query."""
        query = query.lower()
        if self.name.lower() == query:
            return True
        if self.alias and self.alias.lower() == query:
            return True
        if any(t.lower() == query for t in self.tags):
            return True
        return False


@dataclass
class RegistryConfig:
    """Configuration for project registry."""
    registry_file: Path = field(default_factory=lambda: Path.home() / ".rlm" / "projects.json")
    index_dir: Path = field(default_factory=lambda: Path.home() / ".rlm" / "indexes")
    auto_register: bool = True
    name_from_path: bool = True
    max_projects: int = 50
    cleanup_orphaned_days: int = 30


@contextmanager
def file_lock(lock_path: Path, timeout: float = 10.0) -> Generator[None, None, None]:
    """Cross-process file lock using flock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    lock_fd = None

    try:
        lock_fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT)
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > timeout:
                    raise TimeoutError(f"Could not acquire lock on {lock_path} within {timeout}s")
                time.sleep(0.1)
        yield
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(lock_fd)
