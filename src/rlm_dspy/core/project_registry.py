"""Project registry for managing multiple indexed codebases."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Generator
from contextlib import contextmanager

from .project_registry_base import Project, RegistryConfig, file_lock

logger = logging.getLogger(__name__)


class ProjectRegistry:
    """Manages registered projects and their indexes."""

    def __init__(self, config: RegistryConfig | None = None):
        self.config = config or RegistryConfig()
        self.config.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self._projects: dict[str, Project] = {}
        self._default_project: str | None = None
        self._lock = threading.RLock()
        self._file_mtime: float = 0
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if not self.config.registry_file.exists():
            return
        try:
            data = json.loads(self.config.registry_file.read_text(encoding='utf-8'))
            self._projects = {name: Project.from_dict(p) for name, p in data.get("projects", {}).items()}
            self._default_project = data.get("default")
            self._file_mtime = self.config.registry_file.stat().st_mtime
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load registry: %s", e)

    def _reload_if_stale(self) -> None:
        """Reload from disk if file changed externally."""
        if not self.config.registry_file.exists():
            return
        try:
            current_mtime = self.config.registry_file.stat().st_mtime
            if current_mtime > self._file_mtime:
                self._load()
        except OSError:
            pass

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        """Context manager for atomic read-modify-write operations."""
        lock_path = self.config.registry_file.with_suffix(".lock")
        with file_lock(lock_path):
            self._reload_if_stale()
            yield
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        """Save registry (caller must hold file lock)."""
        data = {
            "projects": {name: p.to_dict() for name, p in self._projects.items()},
            "default": self._default_project,
            "version": 1,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.config.registry_file.parent, suffix=".tmp")
        try:
            import os
            with os.fdopen(tmp_fd, 'w') as f:
                json.dump(data, f, indent=2)
            Path(tmp_path).replace(self.config.registry_file)
            self._file_mtime = self.config.registry_file.stat().st_mtime
        except Exception:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass
            raise

    def _save(self) -> None:
        """Save registry with file locking."""
        lock_path = self.config.registry_file.with_suffix(".lock")
        with file_lock(lock_path):
            self._save_unlocked()

    def add(self, path: str | Path, name: str | None = None, alias: str | None = None,
            auto_watch: bool = False, tags: list[str] | None = None) -> Project:
        """Add or update a project."""
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Generate name from path if not provided
        if not name:
            name = path.name
            base_name = name
            counter = 1
            while name in self._projects:
                existing = self._projects[name]
                if existing.path == str(path):
                    name = existing.name
                    break
                name = f"{base_name}-{counter}"
                counter += 1

        # Check for max projects
        if name not in self._projects and len(self._projects) >= self.config.max_projects:
            raise ValueError(f"Maximum projects ({self.config.max_projects}) reached")

        with self._transaction():
            existing = self._projects.get(name)
            if existing and existing.path != str(path):
                raise ValueError(f"Project '{name}' already exists with different path")

            project = Project(
                name=name,
                path=str(path),
                alias=alias or (existing.alias if existing else None),
                indexed_at=existing.indexed_at if existing else None,
                snippet_count=existing.snippet_count if existing else 0,
                file_count=existing.file_count if existing else 0,
                auto_watch=auto_watch,
                tags=tags or (existing.tags if existing else []),
            )
            self._projects[name] = project
            logger.info("Added project: %s (%s)", name, path)
            return project

    def remove(self, name: str, delete_index: bool = False) -> bool:
        """Remove a project."""
        with self._transaction():
            project = self._projects.pop(name, None)
            if not project:
                return False
            if self._default_project == name:
                self._default_project = None
            if delete_index:
                idx_path = self.config.index_dir / name
                if idx_path.exists():
                    shutil.rmtree(idx_path)
            logger.info("Removed project: %s", name)
            return True

    def get(self, name: str) -> Project | None:
        """Get a project by name, alias, or tag."""
        with self._lock:
            self._reload_if_stale()
            if name in self._projects:
                return self._projects[name]
            for project in self._projects.values():
                if project.matches(name):
                    return project
            return None

    def list(self, tags: list[str] | None = None, sort_by: str = "name") -> list[Project]:
        """List projects, optionally filtered by tags."""
        with self._lock:
            self._reload_if_stale()
            projects = list(self._projects.values())
            if tags:
                projects = [p for p in projects if any(t in p.tags for t in tags)]
            if sort_by == "name":
                projects.sort(key=lambda p: p.name.lower())
            elif sort_by == "indexed_at":
                projects.sort(key=lambda p: p.indexed_at or "", reverse=True)
            elif sort_by == "snippet_count":
                projects.sort(key=lambda p: p.snippet_count, reverse=True)
            return projects

    def set_default(self, name: str) -> None:
        """Set the default project."""
        with self._transaction():
            if name not in self._projects:
                raise ValueError(f"Project not found: {name}")
            self._default_project = name

    def get_default(self) -> Project | None:
        """Get the default project."""
        with self._lock:
            self._reload_if_stale()
            if self._default_project and self._default_project in self._projects:
                return self._projects[self._default_project]
            return None

    def _find_overlapping_projects(self, path: Path) -> list[Project]:
        """Find projects that overlap with the given path."""
        path = Path(path).resolve()
        overlaps = []
        for project in self._projects.values():
            project_path = Path(project.path)
            try:
                project_path.relative_to(path)
                overlaps.append(project)
                continue
            except ValueError:
                pass
            try:
                path.relative_to(project_path)
                overlaps.append(project)
            except ValueError:
                pass
        return overlaps

    def find_overlaps(self, name: str | None = None) -> dict[str, list[Project]]:
        """Find all overlapping project paths."""
        with self._lock:
            overlaps = {}
            projects_to_check = [self._projects[name]] if name and name in self._projects else list(self._projects.values())
            for project in projects_to_check:
                path = Path(project.path)
                project_overlaps = []
                for other in self._projects.values():
                    if other.name == project.name:
                        continue
                    other_path = Path(other.path)
                    try:
                        other_path.relative_to(path)
                        project_overlaps.append(other)
                        continue
                    except ValueError:
                        pass
                    try:
                        path.relative_to(other_path)
                        project_overlaps.append(other)
                    except ValueError:
                        pass
                if project_overlaps:
                    overlaps[project.name] = project_overlaps
            return overlaps

    def find_best_match(self, path: str | Path) -> Project | None:
        """Find the most specific project for a given path."""
        path = Path(path).resolve()
        best_match = None
        best_depth = -1
        with self._lock:
            self._reload_if_stale()
            for project in self._projects.values():
                project_path = Path(project.path)
                try:
                    path.relative_to(project_path)
                    depth = len(project_path.parts)
                    if depth > best_depth:
                        best_depth = depth
                        best_match = project
                except ValueError:
                    pass
        return best_match

    def update_stats(self, name: str, snippet_count: int | None = None,
                     file_count: int | None = None, indexed_at: str | None = None) -> None:
        """Update project statistics."""
        with self._transaction():
            project = self._projects.get(name)
            if not project:
                return
            if snippet_count is not None:
                project.snippet_count = snippet_count
            if file_count is not None:
                project.file_count = file_count
            if indexed_at is not None:
                project.indexed_at = indexed_at

    def tag(self, name: str, tags: list[str]) -> None:
        """Add tags to a project."""
        with self._transaction():
            project = self._projects.get(name)
            if project:
                project.tags = list(set(project.tags) | set(tags))

    def untag(self, name: str, tags: list[str]) -> None:
        """Remove tags from a project."""
        with self._transaction():
            project = self._projects.get(name)
            if project:
                project.tags = [t for t in project.tags if t not in tags]

    def get_index_path(self, name: str) -> Path:
        """Get the index directory path for a project."""
        return self.config.index_dir / name

    def find_orphaned(self) -> list[Path]:
        """Find index directories without registered projects."""
        orphaned = []
        if not self.config.index_dir.exists():
            return orphaned
        with self._lock:
            registered_names = set(self._projects.keys())
            registered_hashes = {p.path_hash for p in self._projects.values()}
        for child in self.config.index_dir.iterdir():
            if child.is_dir() and child.name not in registered_names and child.name not in registered_hashes:
                orphaned.append(child)
        return orphaned

    def cleanup_orphaned(self, dry_run: bool = True) -> list[Path]:
        """Remove orphaned index directories."""
        orphaned = self.find_orphaned()
        if not dry_run:
            for path in orphaned:
                shutil.rmtree(path)
                logger.info("Removed orphaned index: %s", path)
        return orphaned

    def migrate_legacy(self) -> list[tuple[str, str]]:
        """Migrate legacy hash-based indexes to named projects."""
        migrated = []
        if not self.config.index_dir.exists():
            return migrated
        for child in self.config.index_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if len(name) != 12 or not all(c in '0123456789abcdef' for c in name):
                continue
            if any(p.path_hash == name for p in self._projects.values()):
                continue
            manifest_path = child / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                repo_path = manifest.get("repo_path", "")
                if not repo_path:
                    continue
                project_name = Path(repo_path).name
                base_name = project_name
                counter = 1
                while project_name in self._projects:
                    project_name = f"{base_name}-{counter}"
                    counter += 1
                project = Project(
                    name=project_name, path=repo_path,
                    snippet_count=manifest.get("snippet_count", 0),
                    file_count=len(manifest.get("files", {})),
                    indexed_at=datetime.fromtimestamp(manifest.get("updated", 0)).isoformat() if manifest.get("updated") else None,
                    path_hash=name,
                )
                new_path = self.config.index_dir / project_name
                if not new_path.exists():
                    child.rename(new_path)
                    self._projects[project_name] = project
                    migrated.append((name, project_name))
                    logger.info("Migrated index %s → %s", name, project_name)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to migrate %s: %s", name, e)
        if migrated:
            self._save()
        return migrated

    def auto_register(self, path: str | Path) -> Project | None:
        """Auto-register a project if config allows."""
        if not self.config.auto_register:
            return None
        path = Path(path).expanduser().resolve()
        for project in self._projects.values():
            if project.path == str(path):
                return project
        name = path.name if self.config.name_from_path else hashlib.md5(str(path).encode()).hexdigest()[:12]
        base_name = name
        counter = 1
        while name in self._projects:
            name = f"{base_name}-{counter}"
            counter += 1
        try:
            return self.add(name, path)
        except ValueError:
            return None

    def __len__(self) -> int:
        return len(self._projects)

    def __contains__(self, name: str) -> bool:
        return name in self._projects


# Global registry instance
_registry: ProjectRegistry | None = None
_registry_lock = threading.Lock()


def get_project_registry(config: RegistryConfig | None = None) -> ProjectRegistry:
    """Get the global project registry instance."""
    global _registry
    if _registry is not None and config is None:
        return _registry
    with _registry_lock:
        if _registry is None or config is not None:
            _registry = ProjectRegistry(config)
        return _registry


__all__ = ["Project", "RegistryConfig", "ProjectRegistry", "get_project_registry"]
