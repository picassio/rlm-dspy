"""Index daemon for automatic background indexing.

Provides a background service that watches registered projects
for file changes and automatically updates their indexes.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from watchdog.observers import Observer

from .daemon_worker import IndexEventHandler, IndexWorker

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for the index daemon."""
    pid_file: Path = field(default_factory=lambda: Path.home() / ".rlm" / "daemon.pid")
    log_file: Path = field(default_factory=lambda: Path.home() / ".rlm" / "daemon.log")

    debounce_seconds: float = 5.0
    watch_patterns: list[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.tsx", "*.go", "*.rs",
        "*.java", "*.c", "*.h", "*.cpp", "*.hpp", "*.rb", "*.cs",
    ])
    ignore_patterns: list[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", "node_modules", ".venv", "venv",
        "dist", "build", ".tox", ".pytest_cache", "*.egg-info",
    ])
    max_concurrent_indexes: int = 2
    idle_timeout: int = 0

    @classmethod
    def from_user_config(cls) -> "DaemonConfig":
        """Load from user config."""
        from .user_config import load_config
        config = load_config()
        daemon_config = config.get("daemon", {})
        return cls(
            pid_file=Path(daemon_config.get("pid_file", "~/.rlm/daemon.pid")).expanduser(),
            log_file=Path(daemon_config.get("log_file", "~/.rlm/daemon.log")).expanduser(),
            debounce_seconds=daemon_config.get("watch_debounce", 5.0),
            max_concurrent_indexes=daemon_config.get("max_concurrent_indexes", 2),
            idle_timeout=daemon_config.get("idle_timeout", 0),
        )


class IndexDaemon:
    """Background daemon for automatic index updates."""

    def __init__(self, config: DaemonConfig | None = None):
        self.config = config or DaemonConfig.from_user_config()
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)

        self._observer = Observer()
        self._queue: Queue = Queue(maxsize=1000)
        self._worker: IndexWorker | None = None
        self._watches: dict[str, any] = {}
        self._running = False
        self._lock = threading.Lock()
        self._pid_fd: int | None = None

        self._start_time: float | None = None
        self._index_count = 0
        self._last_activity: float | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, daemonize: bool = False) -> None:
        """Start the daemon."""
        if self._running:
            logger.warning("Daemon already running")
            return

        if daemonize:
            self._daemonize()

        self._acquire_pid_lock()
        self._write_pid()
        self._setup_signal_handlers()

        self._worker = IndexWorker(
            self._queue,
            self.config,
            on_index_complete=self._on_index_complete,
        )
        self._worker.start()
        self._observer.start()
        self._running = True
        self._start_time = time.time()
        logger.info("Index daemon started (PID: %d)", os.getpid())

    def stop(self) -> None:
        """Stop the daemon."""
        if not self._running:
            return

        logger.info("Stopping index daemon...")
        self._running = False

        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=5.0)
            self._worker = None

        self._observer.stop()
        self._observer.join(timeout=5.0)

        for name in list(self._watches.keys()):
            self.unwatch(name)

        self._cleanup_pid()
        logger.info("Index daemon stopped")

    def watch(self, project_name: str) -> bool:
        """Start watching a project for changes."""
        from .project_registry import get_project_registry

        if project_name in self._watches:
            logger.debug("Already watching: %s", project_name)
            return True

        registry = get_project_registry()
        project = registry.get(project_name)
        if not project:
            logger.error("Project not found: %s", project_name)
            return False

        path = Path(project.path)
        if not path.exists():
            logger.error("Project path does not exist: %s", path)
            return False

        handler = IndexEventHandler(
            project_name=project_name,
            queue=self._queue,
            watch_patterns=self.config.watch_patterns,
            ignore_patterns=self.config.ignore_patterns,
        )

        watch = self._observer.schedule(handler, str(path), recursive=True)
        self._watches[project_name] = watch
        logger.info("Watching project: %s (%s)", project_name, path)
        return True

    def unwatch(self, project_name: str) -> bool:
        """Stop watching a project."""
        watch = self._watches.pop(project_name, None)
        if watch:
            self._observer.unschedule(watch)
            logger.info("Stopped watching: %s", project_name)
            return True
        return False

    def watch_all(self) -> int:
        """Watch all registered projects with auto_watch=True."""
        from .project_registry import get_project_registry
        registry = get_project_registry()
        count = 0
        for project in registry.list():
            if project.auto_watch and self.watch(project.name):
                count += 1
        return count

    def get_status(self) -> dict:
        """Get daemon status."""
        return {
            "running": self._running,
            "pid": os.getpid() if self._running else None,
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "watching": list(self._watches.keys()),
            "watch_count": len(self._watches),
            "index_count": self._index_count,
            "last_activity": self._last_activity,
            "queue_size": self._queue.qsize(),
        }

    def run_forever(self) -> None:
        """Run the daemon until stopped."""
        try:
            while self._running:
                if self.config.idle_timeout > 0:
                    if self._last_activity:
                        idle = time.time() - self._last_activity
                        if idle > self.config.idle_timeout:
                            logger.info("Idle timeout reached, stopping daemon")
                            break
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping...")
        finally:
            self.stop()

    def _on_index_complete(self, project_name: str, count: int) -> None:
        """Called when a project index is updated."""
        self._index_count += 1
        self._last_activity = time.time()

    def _daemonize(self) -> None:
        """Fork into background with proper error handling."""
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            raise RuntimeError(f"First fork failed: {e}") from e
        
        os.setsid()
        
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            raise RuntimeError(f"Second fork failed: {e}") from e

        # Redirect standard file descriptors
        # Store references to prevent garbage collection
        self._devnull = open(os.devnull, 'r')
        self._log_file = open(self.config.log_file, 'a')
        sys.stdin = self._devnull
        sys.stdout = self._log_file
        sys.stderr = self._log_file

    def _acquire_pid_lock(self) -> None:
        """Acquire exclusive lock on PID file."""
        import fcntl
        self._pid_fd = os.open(str(self.config.pid_file), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(self._pid_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._pid_fd)
            self._pid_fd = None
            raise RuntimeError("Another daemon instance is running")

    def _write_pid(self) -> None:
        """Write PID to file."""
        if self._pid_fd is not None:
            os.ftruncate(self._pid_fd, 0)
            os.lseek(self._pid_fd, 0, os.SEEK_SET)
            os.write(self._pid_fd, f"{os.getpid()}\n".encode())

    def _cleanup_pid(self) -> None:
        """Clean up PID file and file descriptors."""
        import fcntl
        if self._pid_fd is not None:
            try:
                fcntl.flock(self._pid_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(self._pid_fd)
            self._pid_fd = None
        try:
            self.config.pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        # Close redirected file descriptors
        if hasattr(self, '_devnull') and self._devnull:
            try:
                self._devnull.close()
            except OSError:
                pass
            self._devnull = None
        if hasattr(self, '_log_file') and self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info("Received signal %d, stopping...", signum)
            self.stop()
            sys.exit(0)
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)


def get_daemon_pid() -> int | None:
    """Get the PID of the running daemon."""
    config = DaemonConfig.from_user_config()
    if not config.pid_file.exists():
        return None
    try:
        pid = int(config.pid_file.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def is_daemon_running() -> bool:
    """Check if the daemon is running."""
    return get_daemon_pid() is not None


def stop_daemon() -> bool:
    """Stop the running daemon."""
    pid = get_daemon_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            time.sleep(0.1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
        os.kill(pid, signal.SIGKILL)
        return True
    except (ProcessLookupError, PermissionError):
        return False
