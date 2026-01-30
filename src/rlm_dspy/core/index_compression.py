"""
Index Compression for reducing disk usage.

Provides compression utilities for vector indexes to reduce storage
requirements for large codebases.
"""

import gzip
import json
import logging
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics from compression operation."""
    
    original_size: int  # bytes
    compressed_size: int  # bytes
    compression_ratio: float  # original / compressed
    files_compressed: int
    
    def __str__(self) -> str:
        saved = self.original_size - self.compressed_size
        saved_pct = (saved / self.original_size * 100) if self.original_size > 0 else 0
        return (
            f"Compressed {self.files_compressed} files: "
            f"{self._format_size(self.original_size)} → {self._format_size(self.compressed_size)} "
            f"({saved_pct:.1f}% saved, {self.compression_ratio:.1f}x ratio)"
        )
    
    @staticmethod
    def _format_size(size: int) -> str:
        """Format size in human-readable form."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


def compress_file(path: Path, remove_original: bool = True) -> int:
    """
    Compress a file using gzip.
    
    Args:
        path: File to compress
        remove_original: Whether to remove the original file
        
    Returns:
        Size of compressed file in bytes
    """
    compressed_path = path.with_suffix(path.suffix + '.gz')
    
    with open(path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb', compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    if remove_original:
        path.unlink()
    
    return compressed_path.stat().st_size


def decompress_file(path: Path, remove_compressed: bool = False) -> Path:
    """
    Decompress a gzipped file.
    
    Args:
        path: Compressed file (.gz)
        remove_compressed: Whether to remove the compressed file
        
    Returns:
        Path to decompressed file
    """
    if not path.suffix == '.gz':
        raise ValueError(f"Expected .gz file, got {path}")
    
    decompressed_path = path.with_suffix('')
    
    with gzip.open(path, 'rb') as f_in:
        with open(decompressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    if remove_compressed:
        path.unlink()
    
    return decompressed_path


def compress_numpy_array(arr: np.ndarray, path: Path) -> int:
    """
    Compress and save a numpy array.
    
    Uses float16 quantization + gzip for ~4x compression.
    
    Args:
        arr: Numpy array to compress
        path: Output path (will add .npz extension)
        
    Returns:
        Size of compressed file in bytes
    """
    # Quantize to float16 if float32/64
    if arr.dtype in (np.float32, np.float64):
        arr = arr.astype(np.float16)
    
    # Save compressed
    np.savez_compressed(path, embeddings=arr)
    
    return path.stat().st_size


def load_numpy_array(path: Path, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Load a compressed numpy array.
    
    Args:
        path: Path to compressed file (.npz)
        dtype: Target dtype (default float32)
        
    Returns:
        Numpy array
    """
    data = np.load(path)
    arr = data['embeddings']
    
    # Convert back to target dtype
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    
    return arr


def compress_json(data: Any, path: Path) -> int:
    """
    Compress and save JSON data.
    
    Args:
        data: JSON-serializable data
        path: Output path (will add .json.gz extension)
        
    Returns:
        Size of compressed file in bytes
    """
    json_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
    
    with gzip.open(path, 'wb', compresslevel=6) as f:
        f.write(json_bytes)
    
    return path.stat().st_size


def load_json(path: Path) -> Any:
    """
    Load compressed or uncompressed JSON.
    
    Args:
        path: Path to JSON file (may be .json or .json.gz)
        
    Returns:
        Parsed JSON data
    """
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return json.loads(f.read().decode('utf-8'))
    else:
        return json.loads(path.read_text())


def compress_index(index_path: Path) -> CompressionStats:
    """
    Compress all files in an index directory.
    
    Compresses:
    - *.npy files (numpy arrays) -> *.npz (quantized + compressed)
    - *.json files -> *.json.gz
    - Large *.txt files -> *.txt.gz
    
    Args:
        index_path: Path to index directory
        
    Returns:
        CompressionStats with results
    """
    if not index_path.is_dir():
        raise ValueError(f"Not a directory: {index_path}")
    
    original_size = 0
    compressed_size = 0
    files_compressed = 0
    
    # Compress numpy files
    for npy_file in index_path.glob("*.npy"):
        original_size += npy_file.stat().st_size
        
        # Load and save compressed
        arr = np.load(npy_file)
        npz_path = npy_file.with_suffix('.npz')
        compressed_size += compress_numpy_array(arr, npz_path)
        
        # Remove original
        npy_file.unlink()
        files_compressed += 1
        logger.debug("Compressed %s", npy_file.name)
    
    # Compress JSON files (except manifest which should stay readable)
    for json_file in index_path.glob("*.json"):
        if json_file.name == "manifest.json":
            # Keep manifest uncompressed for quick reading
            original_size += json_file.stat().st_size
            compressed_size += json_file.stat().st_size
            continue
        
        original_size += json_file.stat().st_size
        
        data = json.loads(json_file.read_text())
        gz_path = json_file.with_suffix('.json.gz')
        compressed_size += compress_json(data, gz_path)
        
        json_file.unlink()
        files_compressed += 1
        logger.debug("Compressed %s", json_file.name)
    
    # Compress large text files
    for txt_file in index_path.glob("*.txt"):
        size = txt_file.stat().st_size
        if size < 10000:  # Skip small files
            original_size += size
            compressed_size += size
            continue
        
        original_size += size
        compressed_size += compress_file(txt_file, remove_original=True)
        files_compressed += 1
        logger.debug("Compressed %s", txt_file.name)
    
    ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    return CompressionStats(
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=ratio,
        files_compressed=files_compressed,
    )


def decompress_index(index_path: Path) -> int:
    """
    Decompress all files in an index directory.
    
    Args:
        index_path: Path to index directory
        
    Returns:
        Number of files decompressed
    """
    if not index_path.is_dir():
        raise ValueError(f"Not a directory: {index_path}")
    
    files_decompressed = 0
    
    # Decompress numpy files
    for npz_file in index_path.glob("*.npz"):
        arr = load_numpy_array(npz_file)
        npy_path = npz_file.with_suffix('.npy')
        np.save(npy_path, arr)
        npz_file.unlink()
        files_decompressed += 1
    
    # Decompress JSON files
    for gz_file in index_path.glob("*.json.gz"):
        data = load_json(gz_file)
        json_path = gz_file.with_suffix('')  # Remove .gz
        json_path.write_text(json.dumps(data, indent=2))
        gz_file.unlink()
        files_decompressed += 1
    
    # Decompress text files
    for gz_file in index_path.glob("*.txt.gz"):
        decompress_file(gz_file, remove_compressed=True)
        files_decompressed += 1
    
    return files_decompressed


def get_index_size(index_path: Path) -> int:
    """
    Get total size of an index directory.
    
    Args:
        index_path: Path to index directory
        
    Returns:
        Total size in bytes
    """
    if not index_path.is_dir():
        return 0
    
    total = 0
    for f in index_path.iterdir():
        if f.is_file():
            total += f.stat().st_size
    
    return total


def is_compressed(index_path: Path) -> bool:
    """
    Check if an index is compressed.
    
    Args:
        index_path: Path to index directory
        
    Returns:
        True if index appears to be compressed
    """
    if not index_path.is_dir():
        return False
    
    # Check for compressed files
    has_npz = any(index_path.glob("*.npz"))
    has_gz = any(index_path.glob("*.gz"))
    
    # Check for uncompressed files
    has_npy = any(index_path.glob("*.npy"))
    has_json = any(f for f in index_path.glob("*.json") if f.name != "manifest.json")
    
    # Compressed if has compressed files and no uncompressed
    return (has_npz or has_gz) and not (has_npy or has_json)
