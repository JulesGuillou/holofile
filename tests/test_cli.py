"""CLI tests.

Two layers:
- Direct calls into _cmd_* functions with a mocked argparse.Namespace
  → counts toward _cli.py coverage
- A handful of subprocess smoke tests to verify the entry-point works end-to-end
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from holofile._cli import (
    _cmd_append,
    _cmd_footer_get,
    _cmd_footer_set,
    _cmd_inspect,
    _cmd_read,
    _cmd_write,
    main,
)
from holofile._reader import HoloReader
from holofile._writer import HoloWriter
from .conftest import make_raw_holo


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def ns(**kwargs) -> SimpleNamespace:
    """Build a SimpleNamespace (acts like argparse.Namespace) with defaults."""
    defaults = dict(
        file=None,
        json=False,
        header_only=False,
        start=None,
        stop=None,
        step=None,
        out=None,
        format="raw",
        width=None,
        height=None,
        bit_depth=None,
        data_type="raw",
        endian="little",
        overwrite=False,
        json_str=None,
        footer_command=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def run_subprocess(*args, input_bytes=None):
    result = subprocess.run(
        [sys.executable, "-m", "holofile._cli"] + list(args),
        capture_output=True,
        input=input_bytes,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# inspect — direct
# ---------------------------------------------------------------------------

class TestInspectDirect:
    def test_human_readable(self, holo_path):
        path, _ = holo_path
        code = _cmd_inspect(ns(file=str(path)))
        assert code == 0

    def test_json_output(self, holo_path, capsys):
        path, _ = holo_path
        code = _cmd_inspect(ns(file=str(path), json=True))
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["num_frames"] == 3

    def test_header_only_human(self, holo_path, capsys):
        path, _ = holo_path
        code = _cmd_inspect(ns(file=str(path), header_only=True))
        assert code == 0
        captured = capsys.readouterr()
        assert "Frames" in captured.out
        assert "Footer" not in captured.out

    def test_header_only_json(self, holo_path, capsys):
        path, _ = holo_path
        code = _cmd_inspect(ns(file=str(path), header_only=True, json=True))
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["num_frames"] == 3
        assert "footer" not in data

    def test_format_error(self, tmp_path):
        p = tmp_path / "bad.holo"
        p.write_bytes(b"NOPE" + b"\x00" * 60)
        assert _cmd_inspect(ns(file=str(p))) == 1

    def test_format_error_header_only(self, tmp_path):
        p = tmp_path / "bad.holo"
        p.write_bytes(b"NOPE" + b"\x00" * 60)
        assert _cmd_inspect(ns(file=str(p), header_only=True)) == 1

    def test_file_not_found(self, tmp_path):
        assert _cmd_inspect(ns(file=str(tmp_path / "no.holo"))) == 2

    def test_file_not_found_header_only(self, tmp_path):
        assert _cmd_inspect(ns(file=str(tmp_path / "no.holo"), header_only=True)) == 2

    def test_with_footer(self, holo_path_with_footer, capsys):
        path, _, _ = holo_path_with_footer
        code = _cmd_inspect(ns(file=str(path)))
        assert code == 0
        assert "camera" in capsys.readouterr().out

    def test_no_footer(self, holo_path, capsys):
        path, _ = holo_path
        code = _cmd_inspect(ns(file=str(path)))
        assert code == 0
        assert "(none)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# read — direct
# ---------------------------------------------------------------------------

class TestReadDirect:
    def test_raw_to_stdout(self, holo_path, capsys):
        path, expected = holo_path
        with patch("sys.stdout", new=io.RawIOBase()) as mock_stdout:
            buf = io.BytesIO()
            with patch("sys.stdout") as mock_out:
                mock_out.buffer = buf
                code = _cmd_read(ns(file=str(path)))
        assert code == 0
        result = np.frombuffer(buf.getvalue(), dtype=np.dtype("<u2")).reshape(3, 4, 4)
        np.testing.assert_array_equal(result, expected)

    def test_raw_to_file(self, holo_path, tmp_path):
        path, expected = holo_path
        out = tmp_path / "out.bin"
        code = _cmd_read(ns(file=str(path), out=str(out)))
        assert code == 0
        result = np.frombuffer(out.read_bytes(), dtype=np.dtype("<u2")).reshape(3, 4, 4)
        np.testing.assert_array_equal(result, expected)

    def test_raw_slice_to_file(self, holo_path, tmp_path):
        path, expected = holo_path
        out = tmp_path / "slice.bin"
        code = _cmd_read(ns(file=str(path), start=1, stop=3, out=str(out)))
        assert code == 0
        result = np.frombuffer(out.read_bytes(), dtype=np.dtype("<u2")).reshape(2, 4, 4)
        np.testing.assert_array_equal(result, expected[1:3])

    def test_npy_to_file(self, holo_path, tmp_path):
        path, expected = holo_path
        out = tmp_path / "out.npy"
        code = _cmd_read(ns(file=str(path), format="npy", out=str(out)))
        assert code == 0
        np.testing.assert_array_equal(np.load(out), expected)

    def test_npy_to_stdout(self, holo_path, capsys):
        path, expected = holo_path
        buf = io.BytesIO()
        with patch("sys.stdout") as mock_out:
            mock_out.buffer = buf
            code = _cmd_read(ns(file=str(path), format="npy"))
        assert code == 0
        result = np.load(io.BytesIO(buf.getvalue()))
        np.testing.assert_array_equal(result, expected)

    def test_tiff_requires_out(self, holo_path):
        path, _ = holo_path
        code = _cmd_read(ns(file=str(path), format="tiff"))
        assert code == 3

    def test_tiff_to_file(self, holo_path, tmp_path):
        pytest.importorskip("tifffile")
        path, _ = holo_path
        out = tmp_path / "out.tiff"
        code = _cmd_read(ns(file=str(path), format="tiff", out=str(out)))
        assert code == 0
        assert out.exists()

    def test_raw_output_os_error(self, holo_path, tmp_path):
        """An OSError while writing the output file returns EXIT_IO."""
        from unittest.mock import patch, MagicMock
        path, _ = holo_path
        out = tmp_path / "out.bin"
        # Let the HoloReader open succeed, but fail on the output write open.
        original_open = open
        call_count = [0]
        def selective_open(p, mode="r", *args, **kwargs):
            call_count[0] += 1
            if "wb" in str(mode):
                raise OSError("write fail")
            return original_open(p, mode, *args, **kwargs)
        with patch("builtins.open", selective_open):
            code = _cmd_read(ns(file=str(path), out=str(out)))
        assert code == 2

    def test_unknown_format(self, holo_path):
        path, _ = holo_path
        code = _cmd_read(ns(file=str(path), format="xyz"))
        assert code == 3

    def test_format_error(self, tmp_path):
        p = tmp_path / "bad.holo"
        p.write_bytes(b"NOPE" + b"\x00" * 60)
        assert _cmd_read(ns(file=str(p))) == 1

    def test_io_error(self, tmp_path):
        assert _cmd_read(ns(file=str(tmp_path / "no.holo"))) == 2

    def test_with_step(self, holo_path, tmp_path):
        path, expected = holo_path
        out = tmp_path / "step.bin"
        code = _cmd_read(ns(file=str(path), step=2, out=str(out)))
        assert code == 0
        result = np.frombuffer(out.read_bytes(), dtype=np.dtype("<u2")).reshape(2, 4, 4)
        np.testing.assert_array_equal(result, expected[::2])


# ---------------------------------------------------------------------------
# write — direct
# ---------------------------------------------------------------------------

class TestWriteDirect:
    def test_write_from_stdin(self, tmp_path):
        out = tmp_path / "out.holo"
        rng = np.random.default_rng(10)
        frames = rng.integers(0, 65536, (2, 4, 4), dtype=np.uint16)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(frames.tobytes())
            code = _cmd_write(ns(
                file=str(out), width=4, height=4, bit_depth=16,
                data_type="raw", endian="little",
            ))
        assert code == 0
        with HoloReader(out) as r:
            np.testing.assert_array_equal(r.read(), frames)

    def test_write_overwrite(self, tmp_path):
        out = tmp_path / "out.holo"
        out.write_bytes(b"x")
        rng = np.random.default_rng(11)
        frames = rng.integers(0, 65536, (1, 4, 4), dtype=np.uint16)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(frames.tobytes())
            code = _cmd_write(ns(
                file=str(out), width=4, height=4, bit_depth=16, overwrite=True,
            ))
        assert code == 0

    def test_write_file_exists_no_overwrite(self, tmp_path):
        out = tmp_path / "out.holo"
        out.write_bytes(b"x")
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(b"")
            code = _cmd_write(ns(file=str(out), width=4, height=4, bit_depth=16))
        assert code == 1

    def test_write_incomplete_frame_warning(self, tmp_path, capsys):
        out = tmp_path / "out.holo"
        # send only 3 bytes (less than frame_size=32)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(b"\x00\x00\x00")
            code = _cmd_write(ns(file=str(out), width=4, height=4, bit_depth=16))
        assert code == 0
        assert "incomplete" in capsys.readouterr().err

    def test_write_io_error(self, tmp_path):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(b"")
            # pass a directory as path to trigger OS error
            code = _cmd_write(ns(
                file=str(tmp_path),  # directory, not a file
                width=4, height=4, bit_depth=16,
            ))
        assert code in (1, 2)

    def test_write_format_error_from_writer(self, tmp_path):
        """HoloFormatError raised inside HoloWriter (e.g. append mismatch) → EXIT_FORMAT."""
        from unittest.mock import patch
        from holofile._exceptions import HoloFormatError as _HFE
        out = tmp_path / "fe.holo"
        with patch("holofile._cli.HoloWriter", side_effect=_HFE("bad")):
            with patch("sys.stdin") as mock_stdin:
                mock_stdin.buffer = io.BytesIO(b"")
                code = _cmd_write(ns(file=str(out), width=4, height=4, bit_depth=16))
        assert code == 1

    def test_write_data_type_processed(self, tmp_path):
        out = tmp_path / "proc.holo"
        rng = np.random.default_rng(12)
        frames = rng.integers(0, 65536, (1, 4, 4), dtype=np.uint16)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(frames.tobytes())
            code = _cmd_write(ns(
                file=str(out), width=4, height=4, bit_depth=16, data_type="processed",
            ))
        assert code == 0
        with HoloReader(out) as r:
            from holofile._enums import DataType
            assert r.header.data_type == DataType.PROCESSED

    def test_write_big_endian(self, tmp_path):
        out = tmp_path / "be.holo"
        frames = np.zeros((1, 4, 4), dtype=np.dtype(">u2"))
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(frames.tobytes())
            code = _cmd_write(ns(
                file=str(out), width=4, height=4, bit_depth=16, endian="big",
            ))
        assert code == 0
        with HoloReader(out) as r:
            from holofile._enums import Endian
            assert r.header.endian == Endian.BIG


# ---------------------------------------------------------------------------
# append — direct
# ---------------------------------------------------------------------------

class TestAppendDirect:
    def test_append(self, tmp_path):
        path = tmp_path / "a.holo"
        rng = np.random.default_rng(20)
        frames_a = rng.integers(0, 65536, (2, 4, 4), dtype=np.uint16)
        frames_b = rng.integers(0, 65536, (1, 4, 4), dtype=np.uint16)
        with HoloWriter(path, bit_depth=16, width=4, height=4) as w:
            w.write(frames_a)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(frames_b.tobytes())
            code = _cmd_append(ns(file=str(path)))
        assert code == 0
        with HoloReader(path) as r:
            assert len(r) == 3

    def test_append_format_error(self, tmp_path):
        p = tmp_path / "bad.holo"
        p.write_bytes(b"NOPE" + b"\x00" * 60)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(b"")
            assert _cmd_append(ns(file=str(p))) == 1

    def test_append_io_error(self, tmp_path):
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(b"")
            assert _cmd_append(ns(file=str(tmp_path / "no.holo"))) == 2

    def test_append_incomplete_frame(self, tmp_path, capsys):
        path = tmp_path / "a.holo"
        frames = np.zeros((1, 4, 4), dtype=np.uint16)
        with HoloWriter(path, bit_depth=16, width=4, height=4) as w:
            w.write(frames)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.buffer = io.BytesIO(b"\x00\x00\x00")
            code = _cmd_append(ns(file=str(path)))
        assert code == 0
        assert "incomplete" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# footer — direct
# ---------------------------------------------------------------------------

class TestFooterDirect:
    def test_footer_get(self, holo_path_with_footer, capsys):
        path, _, _ = holo_path_with_footer
        code = _cmd_footer_get(ns(file=str(path)))
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["camera"] == "HoloVibes"

    def test_footer_get_format_error(self, tmp_path):
        p = tmp_path / "bad.holo"
        p.write_bytes(b"NOPE" + b"\x00" * 60)
        assert _cmd_footer_get(ns(file=str(p))) == 1

    def test_footer_get_io_error(self, tmp_path):
        assert _cmd_footer_get(ns(file=str(tmp_path / "no.holo"))) == 2

    def test_footer_set_via_arg(self, holo_path):
        path, _ = holo_path
        code = _cmd_footer_set(ns(file=str(path), json_str='{"key": "val"}'))
        assert code == 0
        with HoloReader(path) as r:
            assert r.footer.get("key") == "val"

    def test_footer_set_via_stdin(self, holo_path):
        path, _ = holo_path
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.read = lambda: '{"stdin_key": 1}'
            code = _cmd_footer_set(ns(file=str(path), json_str=None))
        assert code == 0
        with HoloReader(path) as r:
            assert r.footer.get("stdin_key") == 1

    def test_footer_set_bad_json(self, holo_path):
        path, _ = holo_path
        assert _cmd_footer_set(ns(file=str(path), json_str="not json")) == 3

    def test_footer_set_format_error(self, tmp_path):
        p = tmp_path / "bad.holo"
        p.write_bytes(b"NOPE" + b"\x00" * 60)
        assert _cmd_footer_set(ns(file=str(p), json_str="{}")) == 1

    def test_footer_set_io_error(self, tmp_path):
        assert _cmd_footer_set(
            ns(file=str(tmp_path / "no.holo"), json_str="{}")
        ) == 2


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------

class TestMainDispatch:
    def test_inspect_dispatch(self, holo_path):
        path, _ = holo_path
        with patch("sys.argv", ["holo", "inspect", str(path)]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_footer_dispatch(self, holo_path):
        path, _ = holo_path
        with patch("sys.argv", ["holo", "footer", "get", str(path)]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_bad_args(self):
        with patch("sys.argv", ["holo"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code != 0


# ---------------------------------------------------------------------------
# subprocess smoke tests (entry-point wiring)
# ---------------------------------------------------------------------------

def test_subprocess_inspect(holo_path):
    path, _ = holo_path
    code, out, _ = run_subprocess("inspect", str(path))
    assert code == 0
    assert b"Frames" in out


def test_subprocess_header_only(holo_path):
    path, _ = holo_path
    code, out, _ = run_subprocess("inspect", str(path), "--header-only", "--json")
    assert code == 0
    assert json.loads(out)["num_frames"] == 3


def test_subprocess_inspect_nonexistent(tmp_path):
    code, _, _ = run_subprocess("inspect", str(tmp_path / "no.holo"))
    assert code in (1, 2)
