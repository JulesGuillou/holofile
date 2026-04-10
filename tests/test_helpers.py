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
