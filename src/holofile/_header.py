from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ._enums import DataType, Endian
from ._exceptions import HoloFormatError

MAGIC = b"HOLO"
VERSION = 7
HEADER_SIZE = 64

# Struct layout (all fields packed as little-endian regardless of payload endianness):
# 4s  magic
# H   version      (uint16)
# H   bit_depth    (uint16)
# I   width        (uint32)
# I   height       (uint32)
# I   num_frames   (uint32)
# Q   data_size    (uint64)
# B   endian       (uint8)
# B   data_type    (uint8)
# 34x padding
_STRUCT_FMT = "<4sHHIIIQBB34x"
_STRUCT = struct.Struct(_STRUCT_FMT)
assert _STRUCT.size == HEADER_SIZE


@dataclass(frozen=True)
class HoloHeader:
    version: int
    bit_depth: int
    width: int
    height: int
    num_frames: int
    data_size: int
    endian: Endian
    data_type: DataType

    @property
    def frame_size(self) -> int:
        """Bytes per frame: width * height * (bit_depth // 8)."""
        return self.width * self.height * (self.bit_depth // 8)

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype inferred from bit_depth and endian."""
        bits = self.bit_depth
        if bits % 8 != 0:
            raise HoloFormatError(f"bit_depth {bits} is not a multiple of 8")
        bytes_per_sample = bits // 8
        # Map byte width to numpy kind (unsigned int)
        kind = "u"
        prefix = "<" if self.endian == Endian.LITTLE else ">"
        return np.dtype(f"{prefix}{kind}{bytes_per_sample}")

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> "HoloHeader":
        if len(buf) < HEADER_SIZE:
            raise HoloFormatError(
                f"Header too short: expected {HEADER_SIZE} bytes, got {len(buf)}"
            )
        fields = _STRUCT.unpack_from(buf, 0)
        magic, version, bit_depth, width, height, num_frames, data_size, endian_val, data_type_val = fields
        if magic != MAGIC:
            raise HoloFormatError(
                f"Invalid magic number: expected {MAGIC!r}, got {magic!r}"
            )
        if version != VERSION:
            raise HoloFormatError(
                f"Unsupported version: expected {VERSION}, got {version}"
            )
        try:
            endian = Endian(endian_val)
        except ValueError:
            raise HoloFormatError(f"Unknown endian value: {endian_val}")
        try:
            data_type = DataType(data_type_val)
        except ValueError:
            raise HoloFormatError(f"Unknown data_type value: {data_type_val}")
        return cls(
            version=version,
            bit_depth=bit_depth,
            width=width,
            height=height,
            num_frames=num_frames,
            data_size=data_size,
            endian=endian,
            data_type=data_type,
        )

    def to_bytes(self) -> bytes:
        return _STRUCT.pack(
            MAGIC,
            self.version,
            self.bit_depth,
            self.width,
            self.height,
            self.num_frames,
            self.data_size,
            int(self.endian),
            int(self.data_type),
        )
