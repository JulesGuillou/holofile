from __future__ import annotations

import mmap as _mmap
import os
from typing import Any, Iterator

import numpy as np

from ._exceptions import HoloFormatError, HoloIndexError, HoloIOError
from ._footer import HoloFooter
from ._header import HEADER_SIZE, HoloHeader


class HoloReader:
    def __init__(
        self,
        path: str | os.PathLike,
        *,
        mmap: bool = False,
    ) -> None:
        try:
            self._file = open(path, "rb")
        except FileNotFoundError:
            raise
        except OSError as e:
            raise HoloIOError(str(e)) from e

        try:
            raw = self._file.read(HEADER_SIZE)
        except OSError as e:
            self._file.close()
            raise HoloIOError(str(e)) from e

        try:
            self._header = HoloHeader.from_bytes(raw)
        except HoloFormatError:
            self._file.close()
            raise

        self._mmap: _mmap.mmap | None = None
        self._mmap_view: memoryview | None = None
        if mmap:
            try:
                # Windows requires mmap offsets to be a multiple of
                # mmap.ALLOCATIONGRANULARITY (64 KiB). Map the whole file
                # from offset 0 and expose only the data section as a view.
                total_size = HEADER_SIZE + self._header.data_size
                self._mmap = _mmap.mmap(
                    self._file.fileno(),
                    length=total_size,
                    offset=0,
                    access=_mmap.ACCESS_READ,
                )
                self._mmap_view = memoryview(self._mmap)[HEADER_SIZE:]
            except OSError as e:
                self._file.close()
                raise HoloIOError(str(e)) from e

        self._footer: HoloFooter | None = None
        self._footer_loaded = False

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "HoloReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._mmap_view is not None:
            self._mmap_view.release()
            self._mmap_view = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if not self._file.closed:
            self._file.close()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def header(self) -> HoloHeader:
        return self._header

    @property
    def footer(self) -> HoloFooter:
        if not self._footer_loaded:
            self._footer = self._load_footer()
            self._footer_loaded = True
        return self._footer  # type: ignore[return-value]

    def _load_footer(self) -> HoloFooter:
        footer_offset = HEADER_SIZE + self._header.data_size
        try:
            self._file.seek(footer_offset)
            raw = self._file.read()
        except OSError as e:
            raise HoloIOError(str(e)) from e
        if not raw:
            return HoloFooter.empty()
        try:
            return HoloFooter.from_json(raw.decode("utf-8"))
        except Exception:
            return HoloFooter.empty()

    # ------------------------------------------------------------------ #
    # Length / indexing
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self._header.num_frames

    def __getitem__(self, index: int | slice) -> np.ndarray:
        n = self._header.num_frames
        if isinstance(index, int):
            if index < 0:
                index += n
            if not (0 <= index < n):
                raise HoloIndexError(f"Frame index {index} out of range [0, {n})")
            return self.read(index, index + 1)
        # slice
        start, stop, step = index.indices(n)
        return self.read(start, stop, step=step)

    # ------------------------------------------------------------------ #
    # Core read
    # ------------------------------------------------------------------ #

    def _resolve_range(self, start: int, stop: int | None) -> tuple[int, int]:
        n = self._header.num_frames
        if stop is None:
            stop = n
        if not (0 <= start <= n):
            raise HoloIndexError(f"start={start} out of range [0, {n}]")
        if not (0 <= stop <= n):
            raise HoloIndexError(f"stop={stop} out of range [0, {n}]")
        return start, stop

    def read_into(
        self,
        buf: Any,
        start: int = 0,
        stop: int | None = None,
        *,
        step: int = 1,
    ) -> int:
        start, stop = self._resolve_range(start, stop)
        indices = range(start, stop, step)
        num_frames = len(indices)
        if num_frames == 0:
            return 0

        frame_size = self._header.frame_size
        required = num_frames * frame_size

        mv = memoryview(buf).cast("B")
        if len(mv) < required:
            raise BufferError(
                f"Buffer too small: need {required} bytes, got {len(mv)}"
            )

        try:
            if self._mmap_view is not None:
                dst_offset = 0
                for i in indices:
                    src_offset = i * frame_size
                    mv[dst_offset : dst_offset + frame_size] = self._mmap_view[
                        src_offset : src_offset + frame_size
                    ]
                    dst_offset += frame_size
            elif step == 1:
                self._file.seek(HEADER_SIZE + start * frame_size)
                total = 0
                while total < required:
                    chunk = self._file.readinto(mv[total:required])
                    if not chunk:
                        break
                    total += chunk
            else:
                dst_offset = 0
                for i in indices:
                    self._file.seek(HEADER_SIZE + i * frame_size)
                    total = 0
                    while total < frame_size:
                        chunk = self._file.readinto(
                            mv[dst_offset + total : dst_offset + frame_size]
                        )
                        if not chunk:
                            break
                        total += chunk
                    dst_offset += frame_size
        except OSError as e:
            raise HoloIOError(str(e)) from e

        return num_frames

    def read(
        self,
        start: int = 0,
        stop: int | None = None,
        *,
        step: int = 1,
    ) -> np.ndarray:
        start, stop = self._resolve_range(start, stop)
        indices = range(start, stop, step)
        num_frames = len(indices)
        h = self._header
        arr = np.empty(
            (num_frames, h.height, h.width), dtype=h.dtype
        )
        self.read_into(arr, start, stop, step=step)
        return arr

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #

    def __iter__(self) -> Iterator[np.ndarray]:
        for i in range(len(self)):
            yield self.read(i, i + 1)

    def iter_chunks(
        self,
        chunk_size: int,
        *,
        buf: Any = None,
    ) -> Iterator[np.ndarray | memoryview]:
        n = len(self)
        h = self._header
        if buf is None:
            buf = np.empty(
                (chunk_size, h.height, h.width), dtype=h.dtype
            )

        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            count = stop - start
            self.read_into(buf, start, stop)
            if isinstance(buf, np.ndarray):
                yield buf[:count]
            else:
                yield memoryview(buf)[: count * h.frame_size]
