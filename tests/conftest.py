"""Shared fixtures for holofile tests."""
from __future__ import annotations

import struct

import numpy as np
import pytest

# Binary header layout constants (mirrors _header.py)
MAGIC = b"HOLO"
HEADER_SIZE = 64
VERSION = 7
_STRUCT_FMT = "<4sHHIIIQBB34x"
_STRUCT = struct.Struct(_STRUCT_FMT)


def make_raw_holo(
    path,
    *,
    width: int = 4,
    height: int = 4,
    bit_depth: int = 16,
    endian: int = 0,       # 0=little, 1=big
    data_type: int = 0,
    frames: np.ndarray | None = None,
    footer_json: bytes | None = None,
) -> None:
    """Write a valid .holo file directly with struct.pack (no dependency on writer)."""
    if frames is None:
        rng = np.random.default_rng(42)
        dtype = np.dtype(f"<u{bit_depth // 8}") if endian == 0 else np.dtype(f">u{bit_depth // 8}")
        frames = rng.integers(0, 2 ** bit_depth, (3, height, width), dtype=dtype)

    num_frames = frames.shape[0]
    frame_size = width * height * (bit_depth // 8)
    data_size = num_frames * frame_size

    header = _STRUCT.pack(
        MAGIC,
        VERSION,
        bit_depth,
        width,
        height,
        num_frames,
        data_size,
        endian,
        data_type,
    )

    with open(path, "wb") as f:
        f.write(header)
        f.write(frames.tobytes())
        if footer_json:
            f.write(footer_json)


@pytest.fixture
def holo_path(tmp_path):
    """A simple 3-frame 4×4 uint16 little-endian .holo file."""
    p = tmp_path / "test.holo"
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 65536, (3, 4, 4), dtype=np.uint16)
    make_raw_holo(p, frames=frames)
    return p, frames


@pytest.fixture
def holo_path_with_footer(tmp_path):
    """A 2-frame .holo file that includes a footer."""
    p = tmp_path / "footer.holo"
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 65536, (2, 4, 4), dtype=np.uint16)
    footer = b'{"camera": "HoloVibes", "fps": 30}'
    make_raw_holo(p, frames=frames, footer_json=footer)
    return p, frames, footer
