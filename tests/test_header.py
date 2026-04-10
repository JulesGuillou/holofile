import struct

import numpy as np
import pytest

from holofile._enums import DataType, Endian
from holofile._exceptions import HoloFormatError
from holofile._header import HEADER_SIZE, VERSION, HoloHeader

_STRUCT_FMT = "<4sHHIIIQBB34x"
_STRUCT = struct.Struct(_STRUCT_FMT)


def _make_raw_header(**kwargs):
    defaults = dict(
        magic=b"HOLO",
        version=VERSION,
        bit_depth=16,
        width=8,
        height=8,
        num_frames=10,
        data_size=8 * 8 * 2 * 10,
        endian=0,
        data_type=0,
    )
    defaults.update(kwargs)
    return _STRUCT.pack(
        defaults["magic"],
        defaults["version"],
        defaults["bit_depth"],
        defaults["width"],
        defaults["height"],
        defaults["num_frames"],
        defaults["data_size"],
        defaults["endian"],
        defaults["data_type"],
    )


def test_round_trip():
    h = HoloHeader(
        version=VERSION,
        bit_depth=16,
        width=640,
        height=480,
        num_frames=100,
        data_size=640 * 480 * 2 * 100,
        endian=Endian.LITTLE,
        data_type=DataType.RAW,
    )
    raw = h.to_bytes()
    assert len(raw) == HEADER_SIZE
    h2 = HoloHeader.from_bytes(raw)
    assert h == h2


def test_bad_magic():
    raw = _make_raw_header(magic=b"NOPE")
    with pytest.raises(HoloFormatError, match="magic"):
        HoloHeader.from_bytes(raw)


def test_wrong_version():
    raw = _make_raw_header(version=6)
    with pytest.raises(HoloFormatError, match="version"):
        HoloHeader.from_bytes(raw)


def test_frame_size():
    h = HoloHeader(
        version=VERSION, bit_depth=16, width=4, height=4,
        num_frames=1, data_size=32, endian=Endian.LITTLE, data_type=DataType.RAW,
    )
    assert h.frame_size == 4 * 4 * 2


def test_dtype_little_endian():
    h = HoloHeader(
        version=VERSION, bit_depth=16, width=4, height=4,
        num_frames=1, data_size=32, endian=Endian.LITTLE, data_type=DataType.RAW,
    )
    assert h.dtype == np.dtype("<u2")


def test_dtype_big_endian():
    h = HoloHeader(
        version=VERSION, bit_depth=16, width=4, height=4,
        num_frames=1, data_size=32, endian=Endian.BIG, data_type=DataType.RAW,
    )
    assert h.dtype == np.dtype(">u2")


def test_header_too_short():
    with pytest.raises(HoloFormatError, match="short"):
        HoloHeader.from_bytes(b"HOLO")


def test_unknown_endian_value():
    raw = _make_raw_header(endian=99)
    with pytest.raises(HoloFormatError, match="endian"):
        HoloHeader.from_bytes(raw)


def test_unknown_data_type_value():
    raw = _make_raw_header(data_type=99)
    with pytest.raises(HoloFormatError, match="data_type"):
        HoloHeader.from_bytes(raw)


def test_dtype_odd_bit_depth_raises():
    # Construct a header with bit_depth=12 by bypassing the dataclass
    # (the validator lives in dtype, not __init__)
    import dataclasses
    h = HoloHeader(
        version=VERSION, bit_depth=12, width=4, height=4,
        num_frames=1, data_size=0, endian=Endian.LITTLE, data_type=DataType.RAW,
    )
    with pytest.raises(HoloFormatError, match="multiple of 8"):
        _ = h.dtype


def test_dtype_32bit():
    h = HoloHeader(
        version=VERSION, bit_depth=32, width=4, height=4,
        num_frames=1, data_size=64, endian=Endian.LITTLE, data_type=DataType.RAW,
    )
    assert h.dtype == np.dtype("<u4")


def test_from_bytes_memoryview():
    """from_bytes should accept a memoryview, not just bytes."""
    h = HoloHeader(
        version=VERSION, bit_depth=16, width=4, height=4,
        num_frames=1, data_size=32, endian=Endian.LITTLE, data_type=DataType.RAW,
    )
    raw = bytearray(h.to_bytes())
    h2 = HoloHeader.from_bytes(memoryview(raw))
    assert h == h2
