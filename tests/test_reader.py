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
