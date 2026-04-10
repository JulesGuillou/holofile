import numpy as np
import pytest

from holofile._enums import DataType, Endian
from holofile._helpers import inspect, read_footer, read_header
from .conftest import make_raw_holo


def test_read_header(holo_path):
    path, _ = holo_path
    h = read_header(path)
    assert h.width == 4
    assert h.height == 4
    assert h.num_frames == 3
    assert h.bit_depth == 16
    assert h.endian == Endian.LITTLE


def test_read_footer_present(holo_path_with_footer):
    path, _, _ = holo_path_with_footer
    f = read_footer(path)
    assert f.get("camera") == "HoloVibes"


def test_read_footer_absent(holo_path):
    path, _ = holo_path
    f = read_footer(path)
    assert f.data == {}


def test_inspect_keys(holo_path):
    path, _ = holo_path
    info = inspect(path)
    assert info["num_frames"] == 3
    assert info["width"] == 4
    assert info["height"] == 4
    assert info["bit_depth"] == 16
    assert info["endian"] == "little"
    assert info["data_type"] == "raw"
    assert "file_size" in info
    assert "frame_size" in info
    assert info["frame_size"] == 4 * 4 * 2


def test_read_header_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_header(tmp_path / "no.holo")


def test_read_header_io_error(tmp_path):
    from unittest.mock import patch, mock_open
    from holofile._exceptions import HoloIOError
    with patch("builtins.open", side_effect=OSError("disk error")):
        with pytest.raises(HoloIOError, match="disk error"):
            read_header(tmp_path / "fake.holo")


def test_read_footer_io_error(holo_path):
    from unittest.mock import patch
    from holofile._exceptions import HoloIOError
    path, _ = holo_path
    original_open = open
    call_count = 0

    def patched_open(p, mode="r", *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:   # second open is the footer read
            raise OSError("seek error")
        return original_open(p, mode, *args, **kwargs)

    with patch("holofile._helpers.open", patched_open):
        with pytest.raises(HoloIOError, match="seek error"):
            read_footer(path)


def test_read_footer_malformed_json(tmp_path):
    """A footer that is not valid JSON returns an empty footer instead of raising."""
    make_raw_holo(tmp_path / "malformed.holo", footer_json=b"not-json{{{")
    f = read_footer(tmp_path / "malformed.holo")
    assert f.data == {}


def test_inspect_io_error_on_getsize(holo_path):
    from unittest.mock import patch
    from holofile._exceptions import HoloIOError
    path, _ = holo_path
    with patch("os.path.getsize", side_effect=OSError("stat fail")):
        with pytest.raises(HoloIOError, match="stat fail"):
            inspect(path)
