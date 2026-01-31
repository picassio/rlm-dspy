"""Index daemon workers - event handlers and background indexing threads."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Callable, TYPE_CHECKING

from watchdog.events import FileSystemEventHandler, FileSystemEvent

if TYPE_CHECKING:
    from .daemon import DaemonConfig

logger = logging.getLogger(__name__)


class IndexEventHandler(FileSystemEventHandler):
    """Handles file system events and queues index updates."""

    def __init__(
        self,
        project_name: str,
        queue: Queue,
        watch_patterns: list[str],
        ignore_patterns: list[str],
    ):
        self.project_name = project_name
        self.queue = queue
        self.watch_patterns = watch_patterns
        self.ignore_patterns = ignore_patterns

    def _should_process(self, path: str) -> bool:
        """Check if path should trigger re-indexing."""
        path_obj = Path(path)

        # Check ignore patterns
        for pattern in self.ignore_patterns:
            if pattern in str(path_obj):
                return False

        # Check watch patterns (file extension match)
        for pattern in self.watch_patterns:
            if pattern.startswith("*"):
                if path_obj.suffix == pattern[1:]:
                    return True
            elif path_obj.match(pattern):
                return True

        return False

    def _queue_event(self, path: str) -> None:
        """Queue an index update for the project."""
        if not self._should_process(path):
            return
        try:
            self.queue.put_nowait((self.project_name, time.time()))
        except Full:
            logger.debug("[%s] Queue full, dropping event for %s",
                        self.project_name, Path(path).name)
        logger.info("[%s] %s queued", self.project_name, Path(path).name)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._queue_event(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._queue_event(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._queue_event(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._queue_event(event.src_path)
            if hasattr(event, 'dest_path'):
                self._queue_event(event.dest_path)


class IndexWorker(threading.Thread):
    """Worker thread that processes index update requests."""

    def __init__(
        self,
        queue: Queue,
        config: "DaemonConfig",
        on_index_complete: Callable[[str, int], None] | None = None,
    ):
        super().__init__(daemon=True)
        self.queue = queue
        self.config = config
        self.on_index_complete = on_index_complete
        self._stop_event = threading.Event()
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Process index update requests with debouncing."""
        consecutive_errors = 0
        max_backoff = 60.0

        while not self._stop_event.is_set():
            try:
                try:
                    project_name, event_time = self.queue.get(timeout=1.0)
                    with self._lock:
                        self._pending[project_name] = event_time
                except Empty:
                    pass

                now = time.time()
                ready = []

                with self._lock:
                    for project_name, last_event in list(self._pending.items()):
                        if now - last_event >= self.config.debounce_seconds:
                            ready.append(project_name)
                            del self._pending[project_name]

                for project_name in ready:
                    self._index_project(project_name)

                consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                backoff = min(2 ** (consecutive_errors - 1), max_backoff)
                logger.error("Worker error (backoff %.1fs): %s", backoff, e)
                if self._stop_event.wait(timeout=backoff):
                    break

    def _index_project(self, project_name: str) -> None:
        """Re-index a project."""
        try:
            from .project_registry import get_project_registry
            from .vector_index import get_index_manager

            registry = get_project_registry()
            project = registry.get(project_name)

            if not project:
                logger.warning("Project %s not found in registry", project_name)
                return

            logger.info("Re-indexing project: %s", project_name)
            manager = get_index_manager()
            count = manager.build(project.path, force=False)
            logger.info("Indexed %s: %d snippets", project_name, count)

            if self.on_index_complete:
                self.on_index_complete(project_name, count)

        except Exception as e:
            logger.error("Failed to index %s: %s", project_name, e)
