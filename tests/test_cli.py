import json
import struct
import subprocess
import sys

import numpy as np
import pytest

from holofile._writer import HoloWriter
from holofile._reader import HoloReader
from .conftest import make_raw_holo


def run_holo(*args, input_bytes=None):
    """Run the holo CLI via subprocess, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "holofile._cli"] + list(args),
        capture_output=True,
        input=input_bytes,
    )
    return result.returncode, result.stdout, result.stderr


def test_inspect_human_readable(holo_path):
    path, _ = holo_path
    code, out, _ = run_holo("inspect", str(path))
    assert code == 0
    text = out.decode()
    assert "Frames" in text
    assert "3" in text


def test_inspect_json(holo_path):
    path, _ = holo_path
    code, out, _ = run_holo("inspect", str(path), "--json")
    assert code == 0
    data = json.loads(out)
    assert data["num_frames"] == 3
    assert data["width"] == 4


def test_inspect_nonexistent(tmp_path):
    code, _, _ = run_holo("inspect", str(tmp_path / "no.holo"))
    assert code in (1, 2)


def test_read_raw_to_stdout(holo_path):
    path, expected = holo_path
    code, out, _ = run_holo("read", str(path))
    assert code == 0
    result = np.frombuffer(out, dtype=np.dtype("<u2")).reshape(3, 4, 4)
    np.testing.assert_array_equal(result, expected)


def test_read_slice(holo_path):
    path, expected = holo_path
    code, out, _ = run_holo("read", str(path), "--start", "1", "--stop", "3")
    assert code == 0
    result = np.frombuffer(out, dtype=np.dtype("<u2")).reshape(2, 4, 4)
    np.testing.assert_array_equal(result, expected[1:3])


def test_read_npy_to_file(holo_path, tmp_path):
    path, expected = holo_path
    out_path = tmp_path / "out.npy"
    code, _, _ = run_holo("read", str(path), "--format", "npy", "--out", str(out_path))
    assert code == 0
    result = np.load(out_path)
    np.testing.assert_array_equal(result, expected)


def test_write_from_stdin(tmp_path):
    out_path = tmp_path / "written.holo"
    rng = np.random.default_rng(5)
    frames = rng.integers(0, 65536, (2, 4, 4), dtype=np.uint16)
    raw = frames.tobytes()
    code, _, _ = run_holo(
        "write", str(out_path),
        "--width", "4", "--height", "4", "--bit-depth", "16",
        input_bytes=raw,
    )
    assert code == 0
    with HoloReader(out_path) as r:
        result = r.read()
    np.testing.assert_array_equal(result, frames)


def test_append_from_stdin(tmp_path):
    path = tmp_path / "append.holo"
    rng = np.random.default_rng(6)
    frames_a = rng.integers(0, 65536, (2, 4, 4), dtype=np.uint16)
    frames_b = rng.integers(0, 65536, (1, 4, 4), dtype=np.uint16)

    with HoloWriter(path, bit_depth=16, width=4, height=4) as w:
        w.write(frames_a)

    code, _, _ = run_holo("append", str(path), input_bytes=frames_b.tobytes())
    assert code == 0

    with HoloReader(path) as r:
        assert len(r) == 3
        np.testing.assert_array_equal(r.read()[2:], frames_b)


def test_footer_get(holo_path_with_footer):
    path, _, _ = holo_path_with_footer
    code, out, _ = run_holo("footer", "get", str(path))
    assert code == 0
    data = json.loads(out)
    assert data["camera"] == "HoloVibes"


def test_footer_set(holo_path):
    path, _ = holo_path
    new_footer = json.dumps({"new_key": "hello"})
    code, _, _ = run_holo("footer", "set", str(path), "--json", new_footer)
    assert code == 0
    code2, out, _ = run_holo("footer", "get", str(path))
    assert code2 == 0
    assert json.loads(out)["new_key"] == "hello"
