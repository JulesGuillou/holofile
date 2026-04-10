from unittest.mock import patch

import numpy as np
import pytest

from holofile._enums import DataType, Endian
from holofile._exceptions import HoloFormatError, HoloIndexError
from holofile._reader import HoloReader
from .conftest import make_raw_holo


def test_read_all_frames(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        frames = r.read()
    assert frames.shape == expected.shape
    np.testing.assert_array_equal(frames, expected)


def test_header_fields(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        h = r.header
    assert h.width == 4
    assert h.height == 4
    assert h.num_frames == 3
    assert h.bit_depth == 16
    assert h.endian == Endian.LITTLE


def test_len(holo_path):
    path, _ = holo_path
    with HoloReader(path) as r:
        assert len(r) == 3


def test_read_slice(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        frames = r.read(1, 3)
    np.testing.assert_array_equal(frames, expected[1:3])


def test_read_step(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        frames = r.read(0, 3, step=2)
    np.testing.assert_array_equal(frames, expected[::2])


def test_getitem_int(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        frame = r[1]
    np.testing.assert_array_equal(frame, expected[1:2])


def test_getitem_negative(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        frame = r[-1]
    np.testing.assert_array_equal(frame, expected[-1:])


def test_getitem_slice(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        frames = r[1:]
    np.testing.assert_array_equal(frames, expected[1:])


def test_iter(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        collected = list(r)
    assert len(collected) == 3
    for i, frame in enumerate(collected):
        np.testing.assert_array_equal(frame[0], expected[i])


def test_iter_chunks(holo_path):
    path, expected = holo_path
    # iter_chunks reuses the internal buffer, so copy each chunk before the
    # next iteration overwrites it.
    with HoloReader(path) as r:
        chunks = [c.copy() for c in r.iter_chunks(2)]
    assert chunks[0].shape[0] == 2
    assert chunks[1].shape[0] == 1
    np.testing.assert_array_equal(chunks[0], expected[:2])
    np.testing.assert_array_equal(chunks[1], expected[2:])


def test_iter_chunks_with_buf(holo_path):
    path, expected = holo_path
    with HoloReader(path) as r:
        buf = np.empty((2, 4, 4), dtype=np.dtype("<u2"))
        results = []
        for chunk in r.iter_chunks(2, buf=buf):
            results.append(chunk.copy())
    np.testing.assert_array_equal(results[0], expected[:2])
    np.testing.assert_array_equal(results[1], expected[2:])


def test_read_into(holo_path):
    path, expected = holo_path
    buf = np.empty((3, 4, 4), dtype=np.dtype("<u2"))
    with HoloReader(path) as r:
        n = r.read_into(buf)
    assert n == 3
    np.testing.assert_array_equal(buf, expected)


def test_read_into_buffer_too_small(holo_path):
    path, _ = holo_path
    buf = bytearray(10)
    with HoloReader(path) as r:
        with pytest.raises(BufferError):
            r.read_into(buf)


def test_index_out_of_range(holo_path):
    path, _ = holo_path
    with HoloReader(path) as r:
        with pytest.raises(HoloIndexError):
            r.read(0, 10)


def test_context_manager_closes(holo_path):
    path, _ = holo_path
    r = HoloReader(path)
    r.close()
    assert r._file.closed


def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        HoloReader(tmp_path / "nonexistent.holo")


def test_bad_magic(tmp_path):
    p = tmp_path / "bad.holo"
    p.write_bytes(b"NOPE" + b"\x00" * 60)
    with pytest.raises(HoloFormatError):
        HoloReader(p)


def test_footer_present(holo_path_with_footer):
    path, _, _ = holo_path_with_footer
    with HoloReader(path) as r:
        footer = r.footer
    assert footer.get("camera") == "HoloVibes"
    assert footer.get("fps") == 30


def test_footer_absent(holo_path):
    path, _ = holo_path
    with HoloReader(path) as r:
        footer = r.footer
    assert footer.data == {}


def test_mmap_read(holo_path):
    path, expected = holo_path
    with HoloReader(path, mmap=True) as r:
        frames = r.read()
    np.testing.assert_array_equal(frames, expected)


def test_mmap_read_step(holo_path):
    """mmap path with step > 1 (non-contiguous frame access)."""
    path, expected = holo_path
    with HoloReader(path, mmap=True) as r:
        frames = r.read(0, 3, step=2)
    np.testing.assert_array_equal(frames, expected[::2])


def test_read_into_empty_range(holo_path):
    """read_into with start == stop returns 0 without touching the buffer."""
    path, _ = holo_path
    buf = bytearray(0)
    with HoloReader(path) as r:
        n = r.read_into(buf, 1, 1)
    assert n == 0


def test_read_into_bytearray(holo_path):
    """read_into accepts a plain bytearray (buffer protocol)."""
    path, expected = holo_path
    frame_size = 4 * 4 * 2
    buf = bytearray(frame_size)
    with HoloReader(path) as r:
        n = r.read_into(buf, 0, 1)
    assert n == 1
    np.testing.assert_array_equal(
        np.frombuffer(buf, dtype="<u2").reshape(4, 4),
        expected[0],
    )


def test_start_out_of_range(holo_path):
    path, _ = holo_path
    with HoloReader(path) as r:
        with pytest.raises(HoloIndexError):
            r.read(-1, 2)


def test_getitem_int_out_of_range(holo_path):
    path, _ = holo_path
    with HoloReader(path) as r:
        with pytest.raises(HoloIndexError):
            _ = r[99]


def test_getitem_negative_out_of_range(holo_path):
    path, _ = holo_path
    with HoloReader(path) as r:
        with pytest.raises(HoloIndexError):
            _ = r[-99]


def test_footer_malformed_json(tmp_path):
    """A footer that is not valid JSON is silently ignored."""
    make_raw_holo(tmp_path / "malformed.holo", footer_json=b"not-json{{")
    with HoloReader(tmp_path / "malformed.holo") as r:
        assert r.footer.data == {}


def test_footer_load_io_error(holo_path):
    """An OSError while reading the footer is re-raised as HoloIOError."""
    from unittest.mock import patch
    from holofile._exceptions import HoloIOError
    path, _ = holo_path
    with HoloReader(path) as r:
        with patch.object(r._file, "seek", side_effect=OSError("seek fail")):
            with pytest.raises(HoloIOError, match="seek fail"):
                _ = r.footer


def test_iter_chunks_bytearray_buf(holo_path):
    """iter_chunks with a raw bytearray buffer yields memoryview slices."""
    path, expected = holo_path
    frame_size = 4 * 4 * 2
    buf = bytearray(2 * frame_size)
    results = []
    with HoloReader(path) as r:
        for chunk in r.iter_chunks(2, buf=buf):
            results.append(bytes(chunk))
    first_two = np.frombuffer(results[0], dtype="<u2").reshape(2, 4, 4)
    np.testing.assert_array_equal(first_two, expected[:2])


def test_mmap_io_error(tmp_path):
    """An OSError from mmap construction is wrapped in HoloIOError."""
    import mmap as _mmap
    from unittest.mock import patch
    from holofile._exceptions import HoloIOError
    path = tmp_path / "ok.holo"
    make_raw_holo(path)
    with patch("mmap.mmap", side_effect=OSError("mmap fail")):
        with pytest.raises(HoloIOError, match="mmap fail"):
            HoloReader(path, mmap=True)


def test_open_os_error(tmp_path):
    """A non-FileNotFoundError OSError on open is wrapped in HoloIOError."""
    from unittest.mock import patch
    from holofile._exceptions import HoloIOError
    with patch("builtins.open", side_effect=OSError("permission denied")):
        with pytest.raises(HoloIOError, match="permission denied"):
            HoloReader(tmp_path / "fake.holo")


def test_read_header_bytes_os_error(tmp_path):
    """An OSError while reading the 64-byte header is wrapped in HoloIOError."""
    from unittest.mock import patch, MagicMock
    from holofile._exceptions import HoloIOError
    path = tmp_path / "ok.holo"
    make_raw_holo(path)
    mock_file = MagicMock()
    mock_file.read.side_effect = OSError("read fail")
    mock_file.__enter__ = lambda s: s
    mock_file.__exit__ = MagicMock(return_value=False)
    with patch("builtins.open", return_value=mock_file):
        with pytest.raises(HoloIOError, match="read fail"):
            HoloReader(path)


def test_readinto_returns_zero_step1(tmp_path):
    """If readinto returns 0 mid-read (step=1 path, line 182) the loop exits gracefully."""
    path = tmp_path / "trunc.holo"
    rng = np.random.default_rng(99)
    frames = rng.integers(0, 65536, (2, 4, 4), dtype=np.uint16)
    make_raw_holo(path, frames=frames)
    with HoloReader(path) as r:
        buf = np.empty((2, 4, 4), dtype=np.dtype("<u2"))
        original_readinto = r._file.readinto
        call_count = [0]
        def fake_readinto_partial(b):
            call_count[0] += 1
            if call_count[0] == 1:
                # Return a partial read (half the buffer) to force a second iteration
                half = max(1, len(b) // 2)
                n = original_readinto(memoryview(b)[:half])
                return n
            return 0   # ← triggers line 182: if not chunk: break
        with patch.object(r._file, "readinto", side_effect=fake_readinto_partial):
            r.read_into(buf)  # exits early without raising


def test_readinto_returns_zero_step_gt1(tmp_path):
    """If readinto returns 0 mid-read (step>1 path, line 194) the loop exits gracefully."""
    path = tmp_path / "trunc.holo"
    rng = np.random.default_rng(98)
    frames = rng.integers(0, 65536, (3, 4, 4), dtype=np.uint16)
    make_raw_holo(path, frames=frames)
    with HoloReader(path) as r:
        buf = np.empty((2, 4, 4), dtype=np.dtype("<u2"))
        original_readinto = r._file.readinto
        call_count = [0]
        def fake_readinto_partial(b):
            call_count[0] += 1
            if call_count[0] == 1:
                half = max(1, len(b) // 2)
                n = original_readinto(memoryview(b)[:half])
                return n
            return 0   # ← triggers line 194: if not chunk: break
        with patch.object(r._file, "readinto", side_effect=fake_readinto_partial):
            r.read_into(buf, 0, 3, step=2)  # step > 1 path, exits early without raising


def test_read_into_os_error(tmp_path):
    """An OSError during read_into is wrapped in HoloIOError."""
    from unittest.mock import patch
    from holofile._exceptions import HoloIOError
    path = tmp_path / "ok.holo"
    make_raw_holo(path)
    with HoloReader(path) as r:
        buf = np.empty((3, 4, 4), dtype=np.dtype("<u2"))
        with patch.object(r._file, "seek", side_effect=OSError("seek fail")):
            with pytest.raises(HoloIOError, match="seek fail"):
                r.read_into(buf)
