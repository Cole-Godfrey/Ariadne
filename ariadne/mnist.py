"""download, verify, and read the original MNIST IDX files."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import struct
import urllib.request

import numpy as np


BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_file(directory: Path, name: str, expected_md5: str) -> Path:
    path = directory / name
    if path.exists() and _md5(path) == expected_md5:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / (name + ".tmp")
    errors = []
    for base_url in BASE_URLS:
        try:
            with urllib.request.urlopen(base_url + name, timeout=60) as source:
                with temporary.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
            if _md5(temporary) != expected_md5:
                raise ValueError(f"checksum mismatch for {name}")
            os.replace(temporary, path)
            return path
        except (OSError, ValueError) as error:
            errors.append(f"{base_url}: {error}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {name}: {'; '.join(errors)}")


def _images(path: Path, expected_count: int) -> np.ndarray:
    with gzip.open(path, "rb") as source:
        payload = source.read()
    magic, count, rows, columns = struct.unpack_from(">IIII", payload)
    if (magic, count, rows, columns) != (2051, expected_count, 28, 28):
        raise ValueError(f"unexpected MNIST image header: {path}")
    images = np.frombuffer(payload, dtype=np.uint8, offset=16)
    if images.size != count * rows * columns:
        raise ValueError(f"unexpected MNIST image length: {path}")
    return images.reshape(count, rows * columns).astype(np.float32) / 255.0


def _labels(path: Path, expected_count: int) -> np.ndarray:
    with gzip.open(path, "rb") as source:
        payload = source.read()
    magic, count = struct.unpack_from(">II", payload)
    if (magic, count) != (2049, expected_count):
        raise ValueError(f"unexpected MNIST label header: {path}")
    labels = np.frombuffer(payload, dtype=np.uint8, offset=8)
    if labels.size != count or np.any(labels > 9):
        raise ValueError(f"unexpected MNIST labels: {path}")
    return labels.copy()


def load_mnist(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = {name: _ensure_file(directory, name, md5) for name, md5 in FILES.items()}
    train_x = _images(paths["train-images-idx3-ubyte.gz"], 60_000)
    train_y = _labels(paths["train-labels-idx1-ubyte.gz"], 60_000)
    test_x = _images(paths["t10k-images-idx3-ubyte.gz"], 10_000)
    test_y = _labels(paths["t10k-labels-idx1-ubyte.gz"], 10_000)
    return train_x, train_y, test_x, test_y
