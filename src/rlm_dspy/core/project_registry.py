"""Project registry for managing multiple indexed codebases.

Provides named project management instead of hash-based directories,
with support for cross-project search and project metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    
    # Internal: hash of the path for backwards compatibility
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
        if query in [t.lower() for t in self.tags]:
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


class ProjectRegistry:
    """Manages registered projects and their indexes.
    
    Features:
    - Named projects instead of hash directories
    - Cross-project search
    - Project tagging and aliasing
    - Orphaned index cleanup
    - Migration from legacy hash-based indexes
    
    Example:
        ```python
        registry = ProjectRegistry()
        
        # Register a project
        registry.add("my-app", "~/projects/my-app")
        
        # List projects
        for project in registry.list():
            print(f"{project.name}: {project.snippet_count} snippets")
        
        # Search across projects
        results = registry.search_all("authentication", projects=["my-app", "backend"])
        ```
    """
    
    def __init__(self, config: RegistryConfig | None = None):
        self.config = config or RegistryConfig()
        self.config.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        
        self._projects: dict[str, Project] = {}
        self._default_project: str | None = None
        self._load()
    
    def _load(self) -> None:
        """Load registry from disk."""
        if not self.config.registry_file.exists():
            return
        
        try:
            data = json.loads(self.config.registry_file.read_text())
            self._default_project = data.get("default_project")
            
            for name, proj_data in data.get("projects", {}).items():
                self._projects[name] = Project.from_dict(proj_data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load project registry: %s", e)
    
    def _save(self) -> None:
        """Save registry to disk."""
        data = {
            "default_project": self._default_project,
            "projects": {
                name: proj.to_dict() 
                for name, proj in self._projects.items()
            },
            "updated_at": datetime.now().isoformat(),
        }
        self.config.registry_file.write_text(json.dumps(data, indent=2))
    
    def add(
        self,
        name: str,
        path: str | Path,
        alias: str | None = None,
        tags: list[str] | None = None,
        auto_watch: bool = False,
    ) -> Project:
        """Register a new project.
        
        Args:
            name: Project name (used in CLI and search)
            path: Path to the project directory
            alias: Optional short alias
            tags: Optional tags for grouping
            auto_watch: Whether to auto-watch for changes (daemon)
            
        Returns:
            The registered Project
            
        Raises:
            ValueError: If name already exists or path doesn't exist
        """
        path = Path(path).expanduser().resolve()
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if name in self._projects:
            raise ValueError(f"Project '{name}' already exists. Use update() or remove() first.")
        
        # Check for duplicate paths
        for existing in self._projects.values():
            if existing.path == str(path):
                raise ValueError(
                    f"Path already registered as '{existing.name}'. "
                    f"Use 'rlm-dspy project remove {existing.name}' first."
                )
        
        project = Project(
            name=name,
            path=str(path),
            alias=alias,
            tags=tags or [],
            auto_watch=auto_watch,
        )
        
        self._projects[name] = project
        self._save()
        
        logger.info("Registered project '%s' at %s", name, path)
        return project
    
    def remove(self, name: str, delete_index: bool = False) -> bool:
        """Remove a project from the registry.
        
        Args:
            name: Project name
            delete_index: Also delete the index directory
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._projects:
            return False
        
        project = self._projects[name]
        
        if delete_index:
            index_path = self.config.index_dir / name
            if index_path.exists():
                shutil.rmtree(index_path)
                logger.info("Deleted index for '%s'", name)
            
            # Also try legacy hash path
            legacy_path = self.config.index_dir / project.path_hash
            if legacy_path.exists():
                shutil.rmtree(legacy_path)
        
        del self._projects[name]
        
        if self._default_project == name:
            self._default_project = None
        
        self._save()
        logger.info("Removed project '%s'", name)
        return True
    
    def get(self, name: str) -> Project | None:
        """Get a project by name or alias."""
        if name in self._projects:
            return self._projects[name]
        
        # Try alias
        for project in self._projects.values():
            if project.alias and project.alias.lower() == name.lower():
                return project
        
        return None
    
    def list(
        self,
        tags: list[str] | None = None,
        sort_by: str = "name",
    ) -> list[Project]:
        """List all registered projects.
        
        Args:
            tags: Filter by tags (OR logic)
            sort_by: Sort by 'name', 'indexed_at', or 'snippet_count'
            
        Returns:
            List of Project objects
        """
        projects = list(self._projects.values())
        
        if tags:
            tags_lower = [t.lower() for t in tags]
            projects = [
                p for p in projects
                if any(t.lower() in tags_lower for t in p.tags)
            ]
        
        if sort_by == "name":
            projects.sort(key=lambda p: p.name.lower())
        elif sort_by == "indexed_at":
            projects.sort(key=lambda p: p.indexed_at or "", reverse=True)
        elif sort_by == "snippet_count":
            projects.sort(key=lambda p: p.snippet_count, reverse=True)
        
        return projects
    
    def set_default(self, name: str) -> None:
        """Set the default project for searches."""
        if name not in self._projects:
            raise ValueError(f"Project '{name}' not found")
        
        self._default_project = name
        self._save()
    
    def get_default(self) -> Project | None:
        """Get the default project."""
        if self._default_project:
            return self._projects.get(self._default_project)
        return None
    
    def update_stats(
        self,
        name: str,
        snippet_count: int,
        file_count: int,
    ) -> None:
        """Update project statistics after indexing."""
        if name not in self._projects:
            return
        
        project = self._projects[name]
        project.snippet_count = snippet_count
        project.file_count = file_count
        project.indexed_at = datetime.now().isoformat()
        self._save()
    
    def tag(self, name: str, tags: list[str]) -> None:
        """Add tags to a project."""
        if name not in self._projects:
            raise ValueError(f"Project '{name}' not found")
        
        project = self._projects[name]
        for tag in tags:
            if tag not in project.tags:
                project.tags.append(tag)
        self._save()
    
    def untag(self, name: str, tags: list[str]) -> None:
        """Remove tags from a project."""
        if name not in self._projects:
            raise ValueError(f"Project '{name}' not found")
        
        project = self._projects[name]
        project.tags = [t for t in project.tags if t not in tags]
        self._save()
    
    def get_index_path(self, name: str) -> Path:
        """Get the index directory path for a project."""
        return self.config.index_dir / name
    
    def find_orphaned(self) -> list[Path]:
        """Find index directories without registered projects."""
        orphaned = []
        
        if not self.config.index_dir.exists():
            return orphaned
        
        registered_names = set(self._projects.keys())
        registered_hashes = {p.path_hash for p in self._projects.values()}
        
        for child in self.config.index_dir.iterdir():
            if not child.is_dir():
                continue
            
            name = child.name
            
            # Check if it's a registered project name or hash
            if name not in registered_names and name not in registered_hashes:
                orphaned.append(child)
        
        return orphaned
    
    def cleanup_orphaned(self, dry_run: bool = True) -> list[Path]:
        """Remove orphaned index directories.
        
        Args:
            dry_run: If True, only return list without deleting
            
        Returns:
            List of removed (or would-be-removed) paths
        """
        orphaned = self.find_orphaned()
        
        if not dry_run:
            for path in orphaned:
                shutil.rmtree(path)
                logger.info("Removed orphaned index: %s", path)
        
        return orphaned
    
    def migrate_legacy(self) -> list[tuple[str, str]]:
        """Migrate legacy hash-based indexes to named projects.
        
        Scans for index directories with hash names and registers
        them as projects using the directory name from their manifest.
        
        Returns:
            List of (old_hash, new_name) tuples for migrated projects
        """
        migrated = []
        
        if not self.config.index_dir.exists():
            return migrated
        
        for child in self.config.index_dir.iterdir():
            if not child.is_dir():
                continue
            
            # Check if it looks like a hash (12 hex chars)
            name = child.name
            if len(name) != 12 or not all(c in '0123456789abcdef' for c in name):
                continue
            
            # Already registered?
            if any(p.path_hash == name for p in self._projects.values()):
                continue
            
            # Try to load manifest
            manifest_path = child / "manifest.json"
            if not manifest_path.exists():
                continue
            
            try:
                manifest = json.loads(manifest_path.read_text())
                repo_path = manifest.get("repo_path", "")
                
                if not repo_path:
                    continue
                
                # Generate project name from path
                project_name = Path(repo_path).name
                
                # Handle duplicates
                base_name = project_name
                counter = 1
                while project_name in self._projects:
                    project_name = f"{base_name}-{counter}"
                    counter += 1
                
                # Register the project
                project = Project(
                    name=project_name,
                    path=repo_path,
                    snippet_count=manifest.get("snippet_count", 0),
                    file_count=len(manifest.get("files", {})),
                    indexed_at=datetime.fromtimestamp(
                        manifest.get("updated", 0)
                    ).isoformat() if manifest.get("updated") else None,
                    path_hash=name,
                )
                
                # Rename directory
                new_path = self.config.index_dir / project_name
                if not new_path.exists():
                    child.rename(new_path)
                    self._projects[project_name] = project
                    migrated.append((name, project_name))
                    logger.info("Migrated index %s → %s", name, project_name)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to migrate %s: %s", name, e)
                continue
        
        if migrated:
            self._save()
        
        return migrated
    
    def auto_register(self, path: str | Path) -> Project | None:
        """Auto-register a project if config allows.
        
        Used by CodeIndex when building an index for an unregistered path.
        
        Args:
            path: Path to the project
            
        Returns:
            The registered Project, or None if auto-register is disabled
        """
        if not self.config.auto_register:
            return None
        
        path = Path(path).expanduser().resolve()
        
        # Already registered?
        for project in self._projects.values():
            if project.path == str(path):
                return project
        
        # Generate name
        if self.config.name_from_path:
            name = path.name
        else:
            name = hashlib.md5(str(path).encode()).hexdigest()[:12]
        
        # Handle duplicates
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


def get_project_registry(config: RegistryConfig | None = None) -> ProjectRegistry:
    """Get the global project registry instance."""
    global _registry
    if _registry is None or config is not None:
        _registry = ProjectRegistry(config)
    return _registry


__all__ = [
    "Project",
    "RegistryConfig", 
    "ProjectRegistry",
    "get_project_registry",
]
