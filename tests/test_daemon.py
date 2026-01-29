"""Tests for index daemon."""



class TestDaemonConfig:
    """Test DaemonConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from rlm_dspy.core.daemon import DaemonConfig

        config = DaemonConfig()

        assert config.debounce_seconds == 5.0
        assert config.max_concurrent_indexes == 2
        assert config.idle_timeout == 0
        assert len(config.watch_patterns) > 0
        assert len(config.ignore_patterns) > 0

    def test_watch_patterns(self):
        """Test watch patterns include common languages."""
        from rlm_dspy.core.daemon import DaemonConfig

        config = DaemonConfig()

        assert "*.py" in config.watch_patterns
        assert "*.js" in config.watch_patterns
        assert "*.ts" in config.watch_patterns
        assert "*.go" in config.watch_patterns
        assert "*.rs" in config.watch_patterns

    def test_ignore_patterns(self):
        """Test ignore patterns exclude common directories."""
        from rlm_dspy.core.daemon import DaemonConfig

        config = DaemonConfig()

        assert "__pycache__" in config.ignore_patterns
        assert ".git" in config.ignore_patterns
        assert "node_modules" in config.ignore_patterns
        assert ".venv" in config.ignore_patterns


class TestIndexEventHandler:
    """Test IndexEventHandler class."""

    def test_should_ignore(self, tmp_path):
        """Test file ignore logic."""
        from rlm_dspy.core.daemon import IndexEventHandler, DaemonConfig
        from queue import Queue

        config = DaemonConfig()
        queue = Queue()
        handler = IndexEventHandler("test", queue, config)

        # Should ignore
        assert handler._should_ignore(str(tmp_path / "test.pyc"))
        assert handler._should_ignore(str(tmp_path / "__pycache__" / "test.py"))
        assert handler._should_ignore(str(tmp_path / ".git" / "config"))
        assert handler._should_ignore(str(tmp_path / "node_modules" / "pkg" / "index.js"))

        # Should not ignore
        assert not handler._should_ignore(str(tmp_path / "test.py"))
        assert not handler._should_ignore(str(tmp_path / "src" / "main.py"))

    def test_should_watch(self, tmp_path):
        """Test file watch logic."""
        from rlm_dspy.core.daemon import IndexEventHandler, DaemonConfig
        from queue import Queue

        config = DaemonConfig()
        queue = Queue()
        handler = IndexEventHandler("test", queue, config)

        # Should watch
        assert handler._should_watch(str(tmp_path / "test.py"))
        assert handler._should_watch(str(tmp_path / "app.js"))
        assert handler._should_watch(str(tmp_path / "main.go"))
        assert handler._should_watch(str(tmp_path / "lib.rs"))

        # Should not watch
        assert not handler._should_watch(str(tmp_path / "README.md"))
        assert not handler._should_watch(str(tmp_path / "config.yaml"))
        assert not handler._should_watch(str(tmp_path / "image.png"))


class TestIndexDaemon:
    """Test IndexDaemon class."""

    def test_create_daemon(self, tmp_path):
        """Test daemon creation."""
        from rlm_dspy.core.daemon import IndexDaemon, DaemonConfig

        config = DaemonConfig(
            pid_file=tmp_path / "daemon.pid",
            log_file=tmp_path / "daemon.log",
        )
        daemon = IndexDaemon(config)

        assert not daemon.is_running
        assert daemon.list_watches() == []

    def test_get_status(self, tmp_path):
        """Test status retrieval."""
        from rlm_dspy.core.daemon import IndexDaemon, DaemonConfig

        config = DaemonConfig(
            pid_file=tmp_path / "daemon.pid",
            log_file=tmp_path / "daemon.log",
        )
        daemon = IndexDaemon(config)

        status = daemon.get_status()

        assert status["running"] is False
        assert status["watches"] == []
        assert status["index_count"] == 0


class TestDaemonHelpers:
    """Test daemon helper functions."""

    def test_is_daemon_running_no_pid_file(self, tmp_path, monkeypatch):
        """Test is_daemon_running when no PID file exists."""
        from rlm_dspy.core.daemon import is_daemon_running, DaemonConfig

        # Use temp path for PID file
        config = DaemonConfig(pid_file=tmp_path / "nonexistent.pid")
        monkeypatch.setattr("rlm_dspy.core.daemon.DaemonConfig.from_user_config", lambda: config)

        assert is_daemon_running() is False

    def test_get_daemon_pid_no_file(self, tmp_path, monkeypatch):
        """Test get_daemon_pid when no PID file exists."""
        from rlm_dspy.core.daemon import get_daemon_pid, DaemonConfig

        config = DaemonConfig(pid_file=tmp_path / "nonexistent.pid")
        monkeypatch.setattr("rlm_dspy.core.daemon.DaemonConfig.from_user_config", lambda: config)

        assert get_daemon_pid() is None

    def test_get_daemon_pid_stale_pid(self, tmp_path, monkeypatch):
        """Test get_daemon_pid with stale PID file."""
        from rlm_dspy.core.daemon import get_daemon_pid, DaemonConfig

        # Create PID file with non-existent process
        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("999999")  # Unlikely to exist

        config = DaemonConfig(pid_file=pid_file)
        monkeypatch.setattr("rlm_dspy.core.daemon.DaemonConfig.from_user_config", lambda: config)

        # Should return None (stale file is NOT deleted to avoid race with daemon lock)
        assert get_daemon_pid() is None
        # File still exists - cleanup happens during daemon startup via file locking
        assert pid_file.exists()


class TestIndexWorker:
    """Test IndexWorker class."""

    def test_worker_creation(self):
        """Test worker creation."""
        from rlm_dspy.core.daemon import IndexWorker, DaemonConfig
        from queue import Queue

        config = DaemonConfig()
        queue = Queue()
        worker = IndexWorker(queue, config)

        assert worker.config == config
        assert worker.queue == queue

    def test_worker_stop(self):
        """Test worker stop signal."""
        from rlm_dspy.core.daemon import IndexWorker, DaemonConfig
        from queue import Queue

        config = DaemonConfig()
        queue = Queue()
        worker = IndexWorker(queue, config)

        worker.start()
        worker.stop()
        worker.join(timeout=2.0)

        assert not worker.is_alive()
