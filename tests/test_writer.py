import numpy as np
import pytest

from holofile._enums import DataType, Endian
from holofile._exceptions import HoloDTypeError, HoloFormatError, HoloShapeError
from holofile._reader import HoloReader
from holofile._writer import HoloWriter
from holofile._footer import HoloFooter


def _make_frames(n=3, h=4, w=4, dtype=np.dtype("<u2")):
    rng = np.random.default_rng(7)
    return rng.integers(0, 65536, (n, h, w)).astype(dtype)


def test_write_and_read_roundtrip(tmp_path):
    p = tmp_path / "out.holo"
    frames = _make_frames()
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames)
    with HoloReader(p) as r:
        result = r.read()
    np.testing.assert_array_equal(result, frames)


def test_write_single_frame(tmp_path):
    p = tmp_path / "single.holo"
    frame = _make_frames(1)[0]  # 2-D
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        n = w.write(frame)
    assert n == 1
    with HoloReader(p) as r:
        assert len(r) == 1
        np.testing.assert_array_equal(r[0], frame[np.newaxis])


def test_write_batch(tmp_path):
    p = tmp_path / "batch.holo"
    frames = _make_frames(5)
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames[:2])
        w.write(frames[2:])
    with HoloReader(p) as r:
        np.testing.assert_array_equal(r.read(), frames)


def test_frames_written_property(tmp_path):
    p = tmp_path / "count.holo"
    frames = _make_frames(3)
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        assert w.frames_written == 0
        w.write(frames[:1])
        assert w.frames_written == 1
        w.write(frames[1:])
        assert w.frames_written == 3


def test_header_patched_on_close(tmp_path):
    p = tmp_path / "patched.holo"
    frames = _make_frames(4)
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames)
    with HoloReader(p) as r:
        assert r.header.num_frames == 4
        assert r.header.data_size == 4 * 4 * 4 * 2


def test_file_exists_error(tmp_path):
    p = tmp_path / "exists.holo"
    p.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        HoloWriter(p, bit_depth=16, width=4, height=4)


def test_overwrite(tmp_path):
    p = tmp_path / "ow.holo"
    frames = _make_frames(2)
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames)
    new_frames = _make_frames(1)
    with HoloWriter(p, bit_depth=16, width=4, height=4, overwrite=True) as w:
        w.write(new_frames)
    with HoloReader(p) as r:
        assert len(r) == 1
        np.testing.assert_array_equal(r.read(), new_frames)


def test_append(tmp_path):
    p = tmp_path / "append.holo"
    frames_a = _make_frames(2)
    frames_b = _make_frames(3)
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames_a)
    with HoloWriter(p, bit_depth=16, width=4, height=4, append=True) as w:
        w.write(frames_b)
    with HoloReader(p) as r:
        assert len(r) == 5
        np.testing.assert_array_equal(r.read()[:2], frames_a)
        np.testing.assert_array_equal(r.read()[2:], frames_b)


def test_append_incompatible_header(tmp_path):
    p = tmp_path / "incompat.holo"
    frames = _make_frames(1)
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames)
    with pytest.raises(HoloFormatError):
        HoloWriter(p, bit_depth=8, width=4, height=4, append=True)


def test_overwrite_and_append_exclusive():
    with pytest.raises(ValueError):
        HoloWriter("/tmp/x.holo", bit_depth=16, width=4, height=4, overwrite=True, append=True)


def test_bit_depth_not_multiple_of_8():
    with pytest.raises(ValueError):
        HoloWriter("/tmp/x.holo", bit_depth=12, width=4, height=4)


def test_shape_mismatch(tmp_path):
    p = tmp_path / "shape.holo"
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        bad = np.zeros((3, 8), dtype=np.dtype("<u2"))
        with pytest.raises(HoloShapeError):
            w.write(bad)


def test_dtype_mismatch(tmp_path):
    p = tmp_path / "dtype.holo"
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        bad = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(HoloDTypeError):
            w.write(bad)


def test_footer_written(tmp_path):
    p = tmp_path / "footer.holo"
    frames = _make_frames(1)
    footer = HoloFooter(data={"camera": "test", "fps": 50})
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames)
        w.set_footer(footer)
    with HoloReader(p) as r:
        f = r.footer
    assert f.get("camera") == "test"
    assert f.get("fps") == 50


def test_set_footer_dict(tmp_path):
    p = tmp_path / "footer_dict.holo"
    frames = _make_frames(1)
    with HoloWriter(p, bit_depth=16, width=4, height=4) as w:
        w.write(frames)
        w.set_footer({"key": "value"})
    with HoloReader(p) as r:
        assert r.footer.get("key") == "value"
