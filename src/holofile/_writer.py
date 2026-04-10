from __future__ import annotations

import os
import struct
from typing import Any

import numpy as np

from ._enums import DataType, Endian
from ._exceptions import (
    HoloDTypeError,
    HoloFormatError,
    HoloIOError,
    HoloShapeError,
)
from ._footer import HoloFooter
from ._header import HEADER_SIZE, VERSION, HoloHeader

# Offsets within the header for the fields we patch on close
_NUM_FRAMES_OFFSET = 16   # uint32 at byte 16
_DATA_SIZE_OFFSET = 20    # uint64 at byte 20


class HoloWriter:
    def __init__(
        self,
        path: str | os.PathLike,
        *,
        bit_depth: int,
        width: int,
        height: int,
        endian: Endian = Endian.LITTLE,
        data_type: DataType = DataType.RAW,
        overwrite: bool = False,
        append: bool = False,
    ) -> None:
        if overwrite and append:
            raise ValueError("overwrite and append are mutually exclusive")
        if bit_depth % 8 != 0:
            raise ValueError(f"bit_depth must be a multiple of 8, got {bit_depth}")

        self._bit_depth = bit_depth
        self._width = width
        self._height = height
        self._endian = endian
        self._data_type = data_type
        self._footer: HoloFooter | None = None
        self._frames_written = 0

        if append:
            try:
                self._file = open(path, "r+b")
            except FileNotFoundError:
                raise
            except OSError as e:
                raise HoloIOError(str(e)) from e

            raw = self._file.read(HEADER_SIZE)
            existing = HoloHeader.from_bytes(raw)
            if (
                existing.bit_depth != bit_depth
                or existing.width != width
                or existing.height != height
            ):
                self._file.close()
                raise HoloFormatError(
                    "Existing header is incompatible: "
                    f"bit_depth={existing.bit_depth} width={existing.width} height={existing.height} "
                    f"vs requested bit_depth={bit_depth} width={width} height={height}"
                )
            self._frames_written = existing.num_frames
            self._file.seek(HEADER_SIZE + existing.data_size)
        else:
            mode = "wb" if overwrite else "xb"
            try:
                self._file = open(path, mode)
            except FileExistsError:
                raise
            except OSError as e:
                raise HoloIOError(str(e)) from e

            # Write placeholder header (num_frames=0, data_size=0)
            placeholder = HoloHeader(
                version=VERSION,
                bit_depth=bit_depth,
                width=width,
                height=height,
                num_frames=0,
                data_size=0,
                endian=endian,
                data_type=data_type,
            )
            try:
                self._file.write(placeholder.to_bytes())
            except OSError as e:
                self._file.close()
                raise HoloIOError(str(e)) from e

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "HoloWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._file.closed:
            return
        try:
            self._patch_header()
            if self._footer is not None:
                self._file.write(self._footer.to_json().encode("utf-8"))
        except OSError as e:
            raise HoloIOError(str(e)) from e
        finally:
            self._file.close()

    def _patch_header(self) -> None:
        frame_size = self._width * self._height * (self._bit_depth // 8)
        data_size = self._frames_written * frame_size
        self._file.seek(_NUM_FRAMES_OFFSET)
        self._file.write(struct.pack("<I", self._frames_written))
        self._file.seek(_DATA_SIZE_OFFSET)
        self._file.write(struct.pack("<Q", data_size))
        # Leave file position at end of data section so footer is appended correctly.
        self._file.seek(HEADER_SIZE + data_size)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def header(self) -> HoloHeader:
        frame_size = self._width * self._height * (self._bit_depth // 8)
        return HoloHeader(
            version=VERSION,
            bit_depth=self._bit_depth,
            width=self._width,
            height=self._height,
            num_frames=self._frames_written,
            data_size=self._frames_written * frame_size,
            endian=self._endian,
            data_type=self._data_type,
        )

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #

    def write(self, data: Any) -> int:
        arr = np.asarray(data)
        expected_dtype = self.header.dtype

        if arr.dtype != expected_dtype:
            raise HoloDTypeError(
                f"Expected dtype {expected_dtype}, got {arr.dtype}"
            )

        if arr.ndim == 2:
            # Single frame
            if arr.shape != (self._height, self._width):
                raise HoloShapeError(
                    f"Expected frame shape ({self._height}, {self._width}), got {arr.shape}"
                )
            frames = arr[np.newaxis]
        elif arr.ndim == 3:
            if arr.shape[1:] != (self._height, self._width):
                raise HoloShapeError(
                    f"Expected frame shape ({self._height}, {self._width}), got {arr.shape[1:]}"
                )
            frames = arr
        else:
            raise HoloShapeError(
                f"data must be 2-D (single frame) or 3-D (batch), got {arr.ndim}-D"
            )

        count = frames.shape[0]
        try:
            self._file.write(frames.tobytes())
        except OSError as e:
            raise HoloIOError(str(e)) from e
        self._frames_written += count
        return count

    def set_footer(self, footer: HoloFooter | dict) -> None:
        if isinstance(footer, dict):
            footer = HoloFooter(data=footer)
        self._footer = footer
