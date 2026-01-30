"""Tests for index compression module."""

import json
import pytest
import numpy as np
from pathlib import Path

from rlm_dspy.core.index_compression import (
    compress_file,
    decompress_file,
    compress_numpy_array,
    load_numpy_array,
    compress_json,
    load_json,
    compress_index,
    decompress_index,
    get_index_size,
    is_compressed,
    CompressionStats,
)


class TestCompressionStats:
    """Tests for CompressionStats."""

    def test_str_format(self):
        """Test string formatting."""
        stats = CompressionStats(
            original_size=1000000,
            compressed_size=250000,
            compression_ratio=4.0,
            files_compressed=5,
        )
        result = str(stats)
        assert "Compressed 5 files" in result
        assert "75.0% saved" in result
        assert "4.0x ratio" in result

    def test_format_size(self):
        """Test size formatting."""
        assert "B" in CompressionStats._format_size(500)
        assert "KB" in CompressionStats._format_size(5000)
        assert "MB" in CompressionStats._format_size(5000000)


class TestFileCompression:
    """Tests for file compression functions."""

    def test_compress_decompress_file(self, tmp_path):
        """Test compressing and decompressing a file."""
        # Create test file
        test_file = tmp_path / "test.txt"
        content = "Hello, World! " * 1000
        test_file.write_text(content)
        original_size = test_file.stat().st_size

        # Compress
        compressed_size = compress_file(test_file, remove_original=True)
        compressed_file = tmp_path / "test.txt.gz"

        assert compressed_file.exists()
        assert not test_file.exists()  # Original removed
        assert compressed_size < original_size  # Actually compressed

        # Decompress
        decompressed = decompress_file(compressed_file, remove_compressed=True)

        assert decompressed.exists()
        assert not compressed_file.exists()
        assert decompressed.read_text() == content

    def test_compress_keeps_original(self, tmp_path):
        """Test keeping original file after compression."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        compress_file(test_file, remove_original=False)

        assert test_file.exists()
        assert (tmp_path / "test.txt.gz").exists()


class TestNumpyCompression:
    """Tests for numpy array compression."""

    def test_compress_load_array(self, tmp_path):
        """Test compressing and loading numpy array."""
        arr = np.random.rand(100, 128).astype(np.float32)
        path = tmp_path / "embeddings.npz"

        # Compress
        size = compress_numpy_array(arr, path)
        assert path.exists()
        assert size > 0

        # Load
        loaded = load_numpy_array(path)
        
        # Should be close (quantization may reduce precision)
        np.testing.assert_array_almost_equal(arr, loaded, decimal=2)

    def test_quantization(self, tmp_path):
        """Test that float64 is quantized to float16."""
        arr = np.random.rand(50, 64).astype(np.float64)
        path = tmp_path / "test.npz"

        compress_numpy_array(arr, path)
        
        # Load raw to check dtype
        data = np.load(path)
        assert data['embeddings'].dtype == np.float16

        # Load with conversion
        loaded = load_numpy_array(path, dtype=np.float32)
        assert loaded.dtype == np.float32


class TestJsonCompression:
    """Tests for JSON compression."""

    def test_compress_load_json(self, tmp_path):
        """Test compressing and loading JSON."""
        data = {"key": "value", "numbers": [1, 2, 3], "nested": {"a": 1}}
        path = tmp_path / "data.json.gz"

        # Compress
        size = compress_json(data, path)
        assert path.exists()
        assert size > 0

        # Load
        loaded = load_json(path)
        assert loaded == data

    def test_load_uncompressed_json(self, tmp_path):
        """Test loading uncompressed JSON."""
        data = {"key": "value"}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data))

        loaded = load_json(path)
        assert loaded == data


class TestIndexCompression:
    """Tests for full index compression."""

    @pytest.fixture
    def mock_index(self, tmp_path):
        """Create a mock index directory."""
        index_dir = tmp_path / "index"
        index_dir.mkdir()

        # Create numpy file
        arr = np.random.rand(100, 128).astype(np.float32)
        np.save(index_dir / "embeddings.npy", arr)

        # Create JSON file
        metadata = {"snippets": [{"file": "test.py", "line": 1}] * 100}
        (index_dir / "metadata.json").write_text(json.dumps(metadata))

        # Create manifest (should stay uncompressed)
        manifest = {"version": 1, "snippet_count": 100}
        (index_dir / "manifest.json").write_text(json.dumps(manifest))

        # Create large text file
        (index_dir / "corpus.txt").write_text("test content " * 1000)

        return index_dir

    def test_compress_index(self, mock_index):
        """Test compressing an index."""
        original_size = get_index_size(mock_index)
        
        stats = compress_index(mock_index)
        
        compressed_size = get_index_size(mock_index)

        # Check compression happened
        assert stats.files_compressed >= 2  # At least numpy and json
        assert stats.compression_ratio > 1  # Actually compressed
        assert compressed_size < original_size

        # Check file types
        assert (mock_index / "embeddings.npz").exists()
        assert not (mock_index / "embeddings.npy").exists()
        assert (mock_index / "metadata.json.gz").exists()
        assert not (mock_index / "metadata.json").exists()
        
        # Manifest should stay uncompressed
        assert (mock_index / "manifest.json").exists()

    def test_decompress_index(self, mock_index):
        """Test decompressing an index."""
        # First compress
        compress_index(mock_index)
        
        # Then decompress
        count = decompress_index(mock_index)
        
        assert count >= 2
        assert (mock_index / "embeddings.npy").exists()
        assert (mock_index / "metadata.json").exists()
        assert not (mock_index / "embeddings.npz").exists()

    def test_is_compressed(self, mock_index):
        """Test checking if index is compressed."""
        assert not is_compressed(mock_index)
        
        compress_index(mock_index)
        assert is_compressed(mock_index)
        
        decompress_index(mock_index)
        assert not is_compressed(mock_index)

    def test_get_index_size(self, mock_index):
        """Test getting index size."""
        size = get_index_size(mock_index)
        assert size > 0

    def test_roundtrip_preserves_data(self, mock_index):
        """Test that compress/decompress preserves data."""
        # Read original data
        original_arr = np.load(mock_index / "embeddings.npy")
        original_meta = json.loads((mock_index / "metadata.json").read_text())

        # Compress and decompress
        compress_index(mock_index)
        decompress_index(mock_index)

        # Check data preserved
        loaded_arr = np.load(mock_index / "embeddings.npy")
        loaded_meta = json.loads((mock_index / "metadata.json").read_text())

        # Array should be close (float16 quantization may reduce precision)
        np.testing.assert_array_almost_equal(original_arr, loaded_arr, decimal=2)
        assert loaded_meta == original_meta
