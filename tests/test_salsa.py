"""Tests for Salsa-style incremental computation."""

import tempfile
from pathlib import Path

from rlm_dspy.core.salsa import (
    SalsaDB,
    file_content,
    file_hash,
    file_line_count,
    file_lines,
    get_db,
    is_salsa_query,
    reset_db,
    salsa_query,
)


class TestSalsaQuery:
    """Tests for @salsa_query decorator."""

    def test_decorator_marks_function(self):
        """Decorator marks function as salsa query."""
        @salsa_query
        def my_query(db, x):
            return x * 2

        assert is_salsa_query(my_query)

    def test_unmarked_function(self):
        """Regular function is not a salsa query."""
        def regular_func(x):
            return x * 2

        assert not is_salsa_query(regular_func)

    def test_direct_call_works(self):
        """Decorated function can be called directly."""
        @salsa_query
        def my_query(db, x):
            return x * 2

        # Direct call bypasses caching
        result = my_query(None, 5)
        assert result == 10


class TestSalsaDBFiles:
    """Tests for file management."""

    def test_set_and_get_file(self):
        """Set and get file content."""
        db = SalsaDB()
        db.set_file("test.py", "print('hello')")

        content = db.get_file("test.py")
        assert content == "print('hello')"

    def test_file_revision_increments(self):
        """File revision increments on update."""
        db = SalsaDB()

        assert db.get_file_revision("test.py") == 0

        db.set_file("test.py", "v1")
        assert db.get_file_revision("test.py") == 1

        db.set_file("test.py", "v2")
        assert db.get_file_revision("test.py") == 2

    def test_get_missing_file(self):
        """Getting missing file returns None."""
        db = SalsaDB()
        assert db.get_file("missing.py") is None

    def test_load_file_from_disk(self):
        """Load file from disk."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# test file")
            f.flush()

            db = SalsaDB()
            content = db.load_file(f.name)

            assert content == "# test file"
            assert db.get_file(f.name) == "# test file"


class TestSalsaDBQueries:
    """Tests for query execution and caching."""

    def test_query_caches_result(self):
        """Query result is cached."""
        db = SalsaDB()
        call_count = 0

        @salsa_query
        def expensive_query(db, x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call computes
        result1 = db.query(expensive_query, 5)
        assert result1 == 10
        assert call_count == 1

        # Second call uses cache
        result2 = db.query(expensive_query, 5)
        assert result2 == 10
        assert call_count == 1  # Not called again

    def test_different_args_different_cache(self):
        """Different arguments have separate cache entries."""
        db = SalsaDB()
        call_count = 0

        @salsa_query
        def my_query(db, x):
            nonlocal call_count
            call_count += 1
            return x * 2

        db.query(my_query, 5)
        db.query(my_query, 10)

        assert call_count == 2

    def test_file_dependency_invalidation(self):
        """Query is invalidated when dependent file changes."""
        db = SalsaDB()
        call_count = 0

        @salsa_query
        def read_and_process(db, path):
            nonlocal call_count
            call_count += 1
            content = db.get_file(path)
            return f"processed: {content}"

        db.set_file("data.txt", "original")

        # First query
        result1 = db.query(read_and_process, "data.txt")
        assert result1 == "processed: original"
        assert call_count == 1

        # Cached
        _ = db.query(read_and_process, "data.txt")
        assert call_count == 1

        # Update file - should invalidate cache
        db.set_file("data.txt", "updated")

        # Query recomputes
        result3 = db.query(read_and_process, "data.txt")
        assert result3 == "processed: updated"
        assert call_count == 2

    def test_stats_tracking(self):
        """Query stats are tracked."""
        db = SalsaDB()

        @salsa_query
        def my_query(db, x):
            return x

        db.query(my_query, 1)  # miss
        db.query(my_query, 1)  # hit
        db.query(my_query, 2)  # miss
        db.query(my_query, 2)  # hit

        stats = db.stats
        assert stats.cache_hits == 2
        assert stats.cache_misses == 2


class TestSalsaDBInvalidation:
    """Tests for cache invalidation."""

    def test_manual_invalidation(self):
        """Manual invalidation works."""
        db = SalsaDB()
        call_count = 0

        @salsa_query
        def my_query(db, x):
            nonlocal call_count
            call_count += 1
            return x

        db.query(my_query, 5)
        assert call_count == 1

        db.invalidate(my_query, 5)

        db.query(my_query, 5)
        assert call_count == 2

    def test_invalidate_all(self):
        """Invalidate all clears cache."""
        db = SalsaDB()

        @salsa_query
        def my_query(db, x):
            return x

        db.query(my_query, 1)
        db.query(my_query, 2)
        db.query(my_query, 3)

        assert db.cache_size() == 3

        count = db.invalidate_all()

        assert count == 3
        assert db.cache_size() == 0


class TestSalsaDBPersistence:
    """Tests for save/load."""

    def test_save_and_load(self):
        """Save and load state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "salsa.json"

            # Create and populate DB
            db1 = SalsaDB()
            db1.set_file("a.py", "# file a")
            db1.set_file("b.py", "# file b")
            db1.save(path)

            # Load into new DB
            db2 = SalsaDB()
            loaded = db2.load(path)

            assert loaded
            assert db2.get_file("a.py") == "# file a"
            assert db2.get_file("b.py") == "# file b"

    def test_load_missing_file(self):
        """Load missing file returns False."""
        db = SalsaDB()
        result = db.load("/nonexistent/path.json")
        assert not result


class TestBuiltinQueries:
    """Tests for built-in queries."""

    def test_file_content(self):
        """file_content query works."""
        db = SalsaDB()
        db.set_file("test.py", "hello")

        content = db.query(file_content, "test.py")
        assert content == "hello"

    def test_file_hash(self):
        """file_hash query works."""
        db = SalsaDB()
        db.set_file("test.py", "hello")

        h = db.query(file_hash, "test.py")
        assert h is not None
        assert len(h) == 16  # Truncated SHA256

    def test_file_lines(self):
        """file_lines query works."""
        db = SalsaDB()
        db.set_file("test.py", "line1\nline2\nline3")

        lines = db.query(file_lines, "test.py")
        assert lines == ["line1", "line2", "line3"]

    def test_file_line_count(self):
        """file_line_count query works."""
        db = SalsaDB()
        db.set_file("test.py", "line1\nline2\nline3")

        count = db.query(file_line_count, "test.py")
        assert count == 3


class TestGlobalDB:
    """Tests for global DB instance."""

    def test_get_db_creates_instance(self):
        """get_db creates instance."""
        reset_db()
        db = get_db()
        assert db is not None

    def test_get_db_returns_same_instance(self):
        """get_db returns same instance."""
        reset_db()
        db1 = get_db()
        db2 = get_db()
        assert db1 is db2

    def test_reset_db(self):
        """reset_db clears instance."""
        reset_db()
        db1 = get_db()
        reset_db()
        db2 = get_db()
        assert db1 is not db2
