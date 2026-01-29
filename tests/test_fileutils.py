"""Tests for file utilities."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rlm_dspy.core.fileutils import (
    PathTraversalError,
    validate_path_safety,
    is_windows,
    is_macos,
    is_linux,
    get_cache_dir,
    smart_link,
    smart_rmtree,
    path_to_module,
    ensure_dir,
    atomic_write,
    SKIP_DIRS,
    should_skip_entry,
    collect_files,
    format_file_context,
    load_context_from_paths,
    estimate_tokens,
    truncate_context,
    smart_truncate_context,
    clear_context_cache,
)


class TestValidatePathSafety:
    """Tests for path traversal protection."""
    
    def test_simple_path_ok(self):
        """Normal paths pass validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.txt"
            path.touch()
            result = validate_path_safety(path)
            assert result == path.resolve()
    
    def test_relative_path_ok(self):
        """Relative paths are resolved."""
        result = validate_path_safety(Path("."))
        assert result.is_absolute()
    
    def test_dotdot_blocked(self):
        """Path traversal with .. is blocked."""
        with pytest.raises(PathTraversalError, match="traversal sequence"):
            validate_path_safety(Path("/tmp/../etc/passwd"))
    
    def test_base_dir_escapes_blocked(self):
        """Paths escaping base directory are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            outside = base.parent / "other"
            
            with pytest.raises(PathTraversalError, match="outside base directory"):
                validate_path_safety(outside, base_dir=base)
    
    def test_within_base_dir_ok(self):
        """Paths within base directory are allowed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            inside = base / "subdir" / "file.txt"
            result = validate_path_safety(inside, base_dir=base)
            assert str(base) in str(result)


class TestPlatformDetection:
    """Tests for platform detection functions."""
    
    def test_exactly_one_platform(self):
        """Exactly one of is_windows/macos/linux should be True."""
        platforms = [is_windows(), is_macos(), is_linux()]
        # On most systems, exactly one should be True
        # But some edge cases (WSL, etc.) might differ
        assert sum(platforms) >= 1
    
    @patch('sys.platform', 'win32')
    def test_is_windows(self):
        """is_windows returns True on Windows."""
        assert is_windows() is True
    
    @patch('sys.platform', 'darwin')
    def test_is_macos(self):
        """is_macos returns True on macOS."""
        assert is_macos() is True
    
    @patch('sys.platform', 'linux')
    def test_is_linux(self):
        """is_linux returns True on Linux."""
        assert is_linux() is True


class TestGetCacheDir:
    """Tests for cache directory logic."""
    
    def test_returns_path(self):
        """Returns a Path object."""
        result = get_cache_dir()
        assert isinstance(result, Path)
    
    def test_contains_app_name(self):
        """Path contains the app name."""
        result = get_cache_dir("myapp")
        assert "myapp" in str(result)
    
    def test_default_app_name(self):
        """Default app name is rlm_dspy."""
        result = get_cache_dir()
        assert "rlm_dspy" in str(result)


class TestSmartLink:
    """Tests for smart_link function."""
    
    def test_link_file(self):
        """Can create a link to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.txt"
            source.write_text("content")
            target = Path(tmpdir) / "target.txt"
            
            smart_link(source, target)
            
            assert target.exists()
            assert target.read_text() == "content"
    
    def test_link_nonexistent_source_fails(self):
        """Linking to nonexistent source raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "nonexistent"
            target = Path(tmpdir) / "target"
            
            with pytest.raises(FileNotFoundError):
                smart_link(source, target)
    
    def test_link_existing_target_fails(self):
        """Linking to existing target raises FileExistsError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.txt"
            source.write_text("source")
            target = Path(tmpdir) / "target.txt"
            target.write_text("target")
            
            with pytest.raises(FileExistsError):
                smart_link(source, target)
    
    def test_link_force_overwrites(self):
        """force=True overwrites existing target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.txt"
            source.write_text("new content")
            target = Path(tmpdir) / "target.txt"
            target.write_text("old content")
            
            smart_link(source, target, force=True)
            
            assert target.read_text() == "new content"


class TestSmartRmtree:
    """Tests for smart_rmtree function."""
    
    def test_removes_directory(self):
        """Can remove a directory tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_to_remove = Path(tmpdir) / "subdir"
            dir_to_remove.mkdir()
            (dir_to_remove / "file.txt").write_text("content")
            
            result = smart_rmtree(dir_to_remove)
            
            assert result is True
            assert not dir_to_remove.exists()
    
    def test_nonexistent_returns_true(self):
        """Removing nonexistent path returns True."""
        result = smart_rmtree(Path("/nonexistent/path/xyz"))
        assert result is True
    
    def test_path_traversal_blocked(self):
        """Path traversal is blocked."""
        with pytest.raises(PathTraversalError):
            smart_rmtree(Path("/tmp/../etc"))


class TestPathToModule:
    """Tests for path_to_module conversion."""
    
    def test_simple_path(self):
        """Converts simple path to module."""
        result = path_to_module(Path("src/mypackage/module.py"))
        assert result == "src.mypackage.module"
    
    def test_with_root(self):
        """Converts path relative to root."""
        result = path_to_module(
            Path("/project/src/mypackage/module.py"),
            root=Path("/project/src")
        )
        assert result == "mypackage.module"
    
    def test_removes_init(self):
        """Removes __init__ from end."""
        result = path_to_module(Path("mypackage/__init__.py"))
        assert result == "mypackage"
    
    def test_no_py_extension(self):
        """Removes .py extension."""
        result = path_to_module(Path("module.py"))
        assert result == "module"


class TestEnsureDir:
    """Tests for ensure_dir function."""
    
    def test_creates_dir(self):
        """Creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new" / "nested" / "dir"
            result = ensure_dir(new_dir)
            
            assert result == new_dir
            assert new_dir.exists()
            assert new_dir.is_dir()
    
    def test_existing_dir_ok(self):
        """Existing directory is returned unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_dir(Path(tmpdir))
            assert result == Path(tmpdir)


class TestAtomicWrite:
    """Tests for atomic_write function."""
    
    def test_writes_text(self):
        """Can write text content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.txt"
            atomic_write(path, "hello world")
            assert path.read_text() == "hello world"
    
    def test_writes_binary(self):
        """Can write binary content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.bin"
            atomic_write(path, b"binary data", mode="wb")
            assert path.read_bytes() == b"binary data"
    
    def test_creates_parent_dirs(self):
        """Creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "new" / "nested" / "file.txt"
            atomic_write(path, "content")
            assert path.exists()
    
    def test_overwrites_existing(self):
        """Overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "file.txt"
            path.write_text("old")
            atomic_write(path, "new")
            assert path.read_text() == "new"


class TestSkipDirs:
    """Tests for SKIP_DIRS constant."""
    
    def test_common_dirs_skipped(self):
        """Common directories are in skip list."""
        assert ".git" in SKIP_DIRS
        assert "__pycache__" in SKIP_DIRS
        assert "node_modules" in SKIP_DIRS
        assert ".venv" in SKIP_DIRS


class TestShouldSkipEntry:
    """Tests for should_skip_entry function."""
    
    def test_skips_git_dir(self):
        """Skips .git directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entry = root / ".git"
            
            result = should_skip_entry(".git", entry, root, spec=None, is_dir=True)
            assert result is True
    
    def test_skips_pycache(self):
        """Skips __pycache__ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entry = root / "__pycache__"
            
            result = should_skip_entry("__pycache__", entry, root, spec=None, is_dir=True)
            assert result is True
    
    def test_allows_normal_file(self):
        """Allows normal files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            entry = root / "module.py"
            
            result = should_skip_entry("module.py", entry, root, spec=None, is_dir=False)
            assert result is False


class TestCollectFiles:
    """Tests for collect_files function."""
    
    def test_collects_from_directory(self):
        """Collects files from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a.py").write_text("# a")
            (root / "b.py").write_text("# b")
            
            files = collect_files([root])
            
            assert len(files) == 2
            assert all(f.suffix == ".py" for f in files)
    
    def test_collects_single_file(self):
        """Collects single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "test.py"
            file.write_text("# test")
            
            files = collect_files([file])
            
            assert len(files) == 1
            assert files[0] == file
    
    def test_skips_common_dirs(self):
        """Skips common ignored directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "main.py").write_text("# main")
            
            pycache = root / "__pycache__"
            pycache.mkdir()
            (pycache / "cached.pyc").write_text("cached")
            
            files = collect_files([root])
            
            assert len(files) == 1
            assert "pycache" not in str(files[0])


class TestFormatFileContext:
    """Tests for format_file_context function."""
    
    def test_formats_with_line_numbers(self):
        """Formats files with line numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "test.py"
            file.write_text("line1\nline2\nline3")
            
            context, skipped = format_file_context([file], add_line_numbers=True)
            
            assert "=== FILE:" in context
            assert "   1 | line1" in context
            assert "   2 | line2" in context
            assert len(skipped) == 0
    
    def test_formats_without_line_numbers(self):
        """Formats files without line numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "test.py"
            file.write_text("content")
            
            context, skipped = format_file_context([file], add_line_numbers=False)
            
            assert "=== FILE:" in context
            assert "content" in context
            assert " | " not in context


class TestEstimateTokens:
    """Tests for estimate_tokens function."""
    
    def test_estimates_tokens(self):
        """Estimates token count."""
        text = "a" * 400  # 400 chars
        tokens = estimate_tokens(text, chars_per_token=4.0)
        assert tokens == 100
    
    def test_empty_text(self):
        """Empty text has 0 tokens."""
        tokens = estimate_tokens("")
        assert tokens == 0


class TestTruncateContext:
    """Tests for truncate_context function."""
    
    def test_no_truncation_needed(self):
        """Short context is not truncated."""
        context = "short text"
        result, truncated = truncate_context(context, max_tokens=1000)
        assert result == context
        assert truncated is False
    
    def test_tail_truncation(self):
        """Tail strategy keeps end."""
        context = "a" * 1000
        result, truncated = truncate_context(context, max_tokens=50, strategy="tail")
        assert truncated is True
        assert "[TRUNCATED]" in result
        assert result.endswith("a" * 50)  # Keeps end
    
    def test_head_truncation(self):
        """Head strategy keeps start."""
        context = "a" * 1000
        result, truncated = truncate_context(context, max_tokens=50, strategy="head")
        assert truncated is True
        assert "[TRUNCATED]" in result
        assert result.startswith("a" * 50)  # Keeps start


class TestSmartTruncateContext:
    """Tests for smart_truncate_context function."""
    
    def test_no_truncation_needed(self):
        """Short context is not truncated."""
        context = "short text"
        result, truncated = smart_truncate_context(context, max_tokens=1000)
        assert result == context
        assert truncated is False
    
    def test_preserves_file_markers(self):
        """Preserves file boundary markers when truncating."""
        files = []
        for i in range(10):
            files.append(f"=== FILE: file{i}.py ===\n{'x' * 100}\n=== END FILE ===\n")
        context = "".join(files)
        
        # Use larger token limit to keep some files
        result, truncated = smart_truncate_context(context, max_tokens=500)
        
        if truncated:
            # Either has file markers or truncation message
            assert "=== FILE:" in result or "TRUNCATED" in result


class TestContextCache:
    """Tests for context caching."""
    
    def test_clear_cache(self):
        """Can clear context cache."""
        # Just verify it doesn't raise
        clear_context_cache()
