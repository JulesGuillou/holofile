from __future__ import annotations

import os

from ._exceptions import HoloIOError
from ._footer import HoloFooter
from ._header import HEADER_SIZE, HoloHeader


def read_header(path: str | os.PathLike) -> HoloHeader:
    """Parse and return only the header. Does not read frame data."""
    try:
        with open(path, "rb") as f:
            raw = f.read(HEADER_SIZE)
    except FileNotFoundError:
        raise
    except OSError as e:
        raise HoloIOError(str(e)) from e
    return HoloHeader.from_bytes(raw)


def read_footer(path: str | os.PathLike) -> HoloFooter:
    """Parse and return only the footer."""
    header = read_header(path)
    footer_offset = HEADER_SIZE + header.data_size
    try:
        with open(path, "rb") as f:
            f.seek(footer_offset)
            raw = f.read()
    except OSError as e:
        raise HoloIOError(str(e)) from e
    if not raw:
        return HoloFooter.empty()
    try:
        return HoloFooter.from_json(raw.decode("utf-8"))
    except Exception:
        return HoloFooter.empty()


def inspect(path: str | os.PathLike) -> dict:
    """Return a summary dict suitable for pretty-printing."""
    header = read_header(path)
    footer = read_footer(path)
    try:
        file_size = os.path.getsize(path)
    except OSError as e:
        raise HoloIOError(str(e)) from e
    return {
        "file": str(path),
        "file_size": file_size,
        "version": header.version,
        "bit_depth": header.bit_depth,
        "width": header.width,
        "height": header.height,
        "num_frames": header.num_frames,
        "data_size": header.data_size,
        "frame_size": header.frame_size,
        "endian": header.endian.name.lower(),
        "data_type": header.data_type.name.lower(),
        "footer_keys": list(footer.data.keys()),
        "footer": footer.data,
    }
