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
from queue import Queue, Empty, Full
from typing import Callable

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for the index daemon."""
    pid_file: Path = field(default_factory=lambda: Path.home() / ".rlm" / "daemon.pid")
    log_file: Path = field(default_factory=lambda: Path.home() / ".rlm" / "daemon.log")
    
    # File watching
    debounce_seconds: float = 5.0  # Wait after last change before re-indexing
    watch_patterns: list[str] = field(default_factory=lambda: [
        "*.py", "*.js", "*.ts", "*.tsx", "*.go", "*.rs",
        "*.java", "*.c", "*.h", "*.cpp", "*.hpp", "*.rb", "*.cs",
    ])
    ignore_patterns: list[str] = field(default_factory=lambda: [
        "*.pyc", "__pycache__", ".git", "node_modules", ".venv", "venv",
        "dist", "build", ".tox", ".pytest_cache", "*.egg-info",
    ])
    
    # Resource limits
    max_concurrent_indexes: int = 2
    idle_timeout: int = 0  # Seconds of no activity before stopping (0 = never)
    
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


class IndexEventHandler(FileSystemEventHandler):
    """Handles file system events and queues index updates."""
    
    def __init__(
        self, 
        project_name: str,
        queue: Queue,
        config: DaemonConfig,
    ):
        super().__init__()
        self.project_name = project_name
        self.queue = queue
        self.config = config
        self._last_event_time: dict[str, float] = {}
    
    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored."""
        from pathlib import Path
        p = Path(path)
        
        # Check ignore patterns
        for pattern in self.config.ignore_patterns:
            if pattern.startswith("*"):
                if p.suffix == pattern[1:] or p.name == pattern[1:]:
                    return True
            elif pattern in p.parts:
                return True
        
        return False
    
    def _should_watch(self, path: str) -> bool:
        """Check if path matches watch patterns."""
        from pathlib import Path
        p = Path(path)
        
        for pattern in self.config.watch_patterns:
            if pattern.startswith("*"):
                if p.suffix == pattern[1:]:
                    return True
            elif p.name == pattern:
                return True
        
        return False
    
    def on_any_event(self, event: FileSystemEvent) -> None:
        """Handle any file system event."""
        if event.is_directory:
            return
        
        # Only care about actual content changes (ignore open/close/access events)
        if event.event_type not in ('modified', 'created', 'deleted', 'moved'):
            return
        
        path = event.src_path
        
        if self._should_ignore(path):
            return
        
        if not self._should_watch(path):
            return
        
        # Queue the project for re-indexing (non-blocking to avoid hanging observer)
        try:
            self.queue.put_nowait((self.project_name, time.time()))
        except Full:
            # Queue full - event will be dropped but project will be re-indexed
            # when queue drains (debouncing means we don't need every event)
            logger.debug("[%s] Queue full, dropping event for %s", 
                        self.project_name, Path(path).name)
        logger.info("[%s] %s: %s", 
                   self.project_name, event.event_type.upper(), Path(path).name)


class IndexWorker(threading.Thread):
    """Worker thread that processes index update requests."""
    
    def __init__(
        self,
        queue: Queue,
        config: DaemonConfig,
        on_index_complete: Callable[[str, int], None] | None = None,
    ):
        super().__init__(daemon=True)
        self.queue = queue
        self.config = config
        self.on_index_complete = on_index_complete
        self._stop_event = threading.Event()
        self._pending: dict[str, float] = {}  # project -> last_event_time
        self._lock = threading.Lock()
    
    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()
    
    def run(self) -> None:
        """Process index update requests with debouncing."""
        consecutive_errors = 0
        max_backoff = 60.0  # Maximum backoff in seconds
        
        while not self._stop_event.is_set():
            try:
                # Get items from queue with timeout
                try:
                    project_name, event_time = self.queue.get(timeout=1.0)
                    with self._lock:
                        self._pending[project_name] = event_time
                except Empty:
                    pass
                
                # Check for projects ready to index (debounce period passed)
                now = time.time()
                ready = []
                
                with self._lock:
                    for project_name, last_event in list(self._pending.items()):
                        if now - last_event >= self.config.debounce_seconds:
                            ready.append(project_name)
                            del self._pending[project_name]
                
                # Index ready projects
                for project_name in ready:
                    self._index_project(project_name)
                
                # Reset error count on successful iteration
                consecutive_errors = 0
                    
            except Exception as e:
                consecutive_errors += 1
                # Exponential backoff: 1s, 2s, 4s, ..., max 60s
                backoff = min(2 ** (consecutive_errors - 1), max_backoff)
                logger.error("Worker error (backoff %.1fs): %s", backoff, e)
                # Use wait() instead of sleep() to respond to stop signal
                if self._stop_event.wait(timeout=backoff):
                    break  # Stop signal received during backoff
    
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


class IndexDaemon:
    """Background daemon for automatic index updates.
    
    Watches registered projects for file changes and automatically
    updates their indexes with debouncing.
    
    Example:
        ```python
        daemon = IndexDaemon()
        daemon.start()
        
        # Watch a project
        daemon.watch("my-app")
        
        # Stop daemon
        daemon.stop()
        ```
    """
    
    def __init__(self, config: DaemonConfig | None = None):
        self.config = config or DaemonConfig.from_user_config()
        self.config.pid_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._observer = Observer()
        # Bounded queue to prevent memory exhaustion under high file churn
        # maxsize=1000 allows ~1000 pending index requests before blocking
        self._queue: Queue = Queue(maxsize=1000)
        self._worker: IndexWorker | None = None
        self._watches: dict[str, any] = {}  # project_name -> watch handle
        self._running = False
        self._lock = threading.Lock()
        self._pid_fd: int | None = None  # File descriptor for PID file lock
        
        # Stats
        self._start_time: float | None = None
        self._index_count = 0
        self._last_activity: float | None = None
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def start(self, daemonize: bool = False) -> None:
        """Start the daemon.
        
        Args:
            daemonize: If True, fork to background (Unix only)
        """
        if self._running:
            logger.warning("Daemon already running")
            return
        
        if daemonize:
            self._daemonize()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Write PID file with locking
        self._write_pid_file()
        
        # Setup logging
        self._setup_logging()
        
        # Start worker
        self._worker = IndexWorker(
            self._queue, 
            self.config,
            on_index_complete=self._on_index_complete,
        )
        self._worker.start()
        
        # Start observer
        self._observer.start()
        
        self._running = True
        self._start_time = time.time()
        self._last_activity = time.time()
        
        logger.info("Index daemon started (PID: %d)", os.getpid())
        
        # Auto-watch projects with auto_watch=True
        self._auto_watch_projects()
    
    def stop(self) -> None:
        """Stop the daemon."""
        if not self._running:
            return
        
        logger.info("Stopping index daemon...")
        
        # Stop worker
        if self._worker:
            self._worker.stop()
            self._worker.join(timeout=5.0)
        
        # Stop observer
        self._observer.stop()
        self._observer.join(timeout=5.0)
        
        # Close PID file descriptor (releases lock) and remove file
        if hasattr(self, '_pid_fd') and self._pid_fd is not None:
            try:
                os.close(self._pid_fd)
            except OSError:
                pass
            self._pid_fd = None
        
        if self.config.pid_file.exists():
            try:
                self.config.pid_file.unlink()
            except OSError:
                pass
        
        self._running = False
        logger.info("Index daemon stopped")
    
    def watch(self, project_name: str) -> bool:
        """Start watching a project for changes.
        
        Args:
            project_name: Name of registered project to watch
            
        Returns:
            True if watch started, False if project not found
        """
        from .project_registry import get_project_registry
        
        if project_name in self._watches:
            logger.debug("Already watching %s", project_name)
            return True
        
        registry = get_project_registry()
        project = registry.get(project_name)
        
        if not project:
            logger.error("Project %s not found", project_name)
            return False
        
        if not Path(project.path).exists():
            logger.error("Project path does not exist: %s", project.path)
            return False
        
        # Create event handler
        handler = IndexEventHandler(project_name, self._queue, self.config)
        
        # Schedule watch
        watch = self._observer.schedule(
            handler,
            project.path,
            recursive=True,
        )
        
        with self._lock:
            self._watches[project_name] = watch
        
        # Update project auto_watch flag
        project.auto_watch = True
        registry._save()
        
        logger.info("Watching project: %s at %s", project_name, project.path)
        return True
    
    def unwatch(self, project_name: str) -> bool:
        """Stop watching a project.
        
        Args:
            project_name: Name of project to stop watching
            
        Returns:
            True if unwatch succeeded
        """
        from .project_registry import get_project_registry
        
        with self._lock:
            if project_name not in self._watches:
                return False
            
            watch = self._watches.pop(project_name)
            
            # Unschedule while holding lock to prevent race
            self._observer.unschedule(watch)
            
            # Update project auto_watch flag (registry has its own lock)
            registry = get_project_registry()
            project = registry.get(project_name)
            if project:
                project.auto_watch = False
                registry._save()
        
        logger.info("Stopped watching project: %s", project_name)
        return True
    
    def list_watches(self) -> list[str]:
        """List currently watched projects."""
        with self._lock:
            return list(self._watches.keys())
    
    def get_status(self) -> dict:
        """Get daemon status."""
        return {
            "running": self._running,
            "pid": os.getpid() if self._running else None,
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "watches": self.list_watches(),
            "index_count": self._index_count,
            "last_activity": self._last_activity,
            "queue_size": self._queue.qsize(),
        }
    
    def run_forever(self) -> None:
        """Run the daemon until interrupted."""
        if not self._running:
            self.start()
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self._running:
                try:
                    time.sleep(1.0)
                    
                    # Check worker health and restart if needed
                    self._check_worker_health()
                    
                    # Check idle timeout
                    if self.config.idle_timeout > 0:
                        if self._last_activity:
                            idle_time = time.time() - self._last_activity
                            if idle_time > self.config.idle_timeout:
                                logger.info("Idle timeout reached, stopping daemon")
                                break
                    
                    # Reset error count on successful iteration
                    consecutive_errors = 0
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error("Daemon loop error (%d/%d): %s", 
                                consecutive_errors, max_consecutive_errors, e)
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical("Too many consecutive errors, stopping daemon")
                        break
                    
                    # Brief backoff before retry
                    time.sleep(min(consecutive_errors, 5))
                            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def _check_worker_health(self) -> None:
        """Check if worker is alive and restart if crashed."""
        if self._worker and not self._worker.is_alive():
            logger.warning("Worker thread died, restarting...")
            
            # Clean up dead thread (non-blocking join)
            try:
                self._worker.join(timeout=0.1)
            except Exception:
                pass
            
            # Create new worker
            self._worker = IndexWorker(
                self._queue,
                self.config,
                on_index_complete=self._on_index_complete,
            )
            self._worker.start()
            logger.info("Worker thread restarted")
    
    def _setup_logging(self) -> None:
        """Setup logging to file and configure daemon loggers."""
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Close existing handler if any (prevents file descriptor leak)
        if hasattr(self, '_log_handler') and self._log_handler:
            self._log_handler.close()
            # Remove from loggers
            for logger_name in ['rlm_dspy.core.daemon', 'rlm_dspy.core.vector_index']:
                log = logging.getLogger(logger_name)
                log.removeHandler(self._log_handler)
        
        # Create file handler
        file_handler = logging.FileHandler(self.config.log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Configure daemon logger to use file handler
        daemon_logger = logging.getLogger('rlm_dspy.core.daemon')
        daemon_logger.setLevel(logging.DEBUG)
        daemon_logger.addHandler(file_handler)
        daemon_logger.propagate = False
        
        # Also log vector_index messages for indexing progress
        index_logger = logging.getLogger('rlm_dspy.core.vector_index')
        index_logger.setLevel(logging.INFO)
        index_logger.addHandler(file_handler)
        index_logger.propagate = False
        
        # Store reference for cleanup
        self._log_handler = file_handler
        
        # Write startup marker
        daemon_logger.info("=" * 50)
        daemon_logger.info("Daemon started (PID: %d)", os.getpid())
        daemon_logger.info("Log file: %s", self.config.log_file)
    
    def _daemonize(self) -> None:
        """Fork to background (Unix only)."""
        if sys.platform == "win32":
            raise RuntimeError("Daemonize not supported on Windows")
        
        # First fork
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
        
        # Decouple from parent
        os.chdir("/")
        os.setsid()
        os.umask(0)
        
        # Second fork
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
        
        # Redirect std file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Open files and keep them open for the daemon's lifetime
        # Using os.open to avoid Python file object lifecycle issues
        devnull_fd = os.open('/dev/null', os.O_RDONLY)
        log_fd = os.open(str(self.config.log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        
        os.dup2(devnull_fd, sys.stdin.fileno())
        os.dup2(log_fd, sys.stdout.fileno())
        os.dup2(log_fd, sys.stderr.fileno())
        
        # Close original fds (duplicates remain open)
        os.close(devnull_fd)
        os.close(log_fd)
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            logger.info("Received signal %d, initiating shutdown...", signum)
            self._running = False
        
        # Handle SIGTERM (kill) and SIGINT (Ctrl+C)
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
    
    def __del__(self) -> None:
        """Cleanup on garbage collection."""
        # Ensure PID fd is closed even if stop() wasn't called
        if hasattr(self, '_pid_fd') and self._pid_fd is not None:
            try:
                os.close(self._pid_fd)
            except OSError:
                pass
            self._pid_fd = None
    
    def _write_pid_file(self) -> None:
        """Write PID file with file locking to prevent races."""
        pid_file = self.config.pid_file
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Open with O_CLOEXEC to prevent fd inheritance after fork
            # This prevents the fd from leaking to child processes
            flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
            if hasattr(os, 'O_CLOEXEC'):  # Unix
                flags |= os.O_CLOEXEC
            fd = os.open(str(pid_file), flags)
            
            # Try to get exclusive lock (platform-specific)
            if sys.platform != "win32":
                import fcntl
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    raise RuntimeError("Another daemon instance is already running")
            else:
                # Windows: use msvcrt for file locking
                import msvcrt
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                except OSError:
                    os.close(fd)
                    raise RuntimeError("Another daemon instance is already running")
            
            os.write(fd, str(os.getpid()).encode())
            # Store fd so lock is held and can be closed on shutdown
            self._pid_fd = fd
        except OSError as e:
            # Close fd if write failed (fd was opened but not yet stored in self._pid_fd)
            if 'fd' in locals() and fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise RuntimeError(f"Failed to write PID file: {e}")
    
    def _auto_watch_projects(self) -> None:
        """Auto-watch projects with auto_watch=True."""
        from .project_registry import get_project_registry
        
        registry = get_project_registry()
        for project in registry.list():
            if project.auto_watch:
                self.watch(project.name)
    
    def _on_index_complete(self, project_name: str, count: int) -> None:
        """Called when indexing completes."""
        self._index_count += 1
        self._last_activity = time.time()


def get_daemon_pid() -> int | None:
    """Get the PID of the running daemon, if any."""
    config = DaemonConfig.from_user_config()
    
    if not config.pid_file.exists():
        return None
    
    try:
        pid = int(config.pid_file.read_text().strip())
        
        # Check if process is running
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            # Process not running - PID file is stale
            # Don't unlink here to avoid race with daemon's file lock
            # The daemon startup will handle cleanup via file locking
            logger.debug("Stale PID file found (process %d not running)", pid)
            return None
            
    except (ValueError, IOError):
        return None


def is_daemon_running() -> bool:
    """Check if daemon is running."""
    return get_daemon_pid() is not None


def stop_daemon() -> bool:
    """Stop the running daemon."""
    pid = get_daemon_pid()
    if not pid:
        return False
    
    try:
        os.kill(pid, signal.SIGTERM)
        
        # Wait for process to stop
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except OSError:
                return True
        
        # Force kill
        os.kill(pid, signal.SIGKILL)
        return True
        
    except OSError:
        return False


__all__ = [
    "DaemonConfig",
    "IndexDaemon",
    "get_daemon_pid",
    "is_daemon_running",
    "stop_daemon",
]
