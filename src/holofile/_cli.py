from __future__ import annotations

import argparse
import json
import sys

from ._enums import DataType, Endian
from ._exceptions import HoloError, HoloFormatError, HoloIOError
from ._footer import HoloFooter
from ._helpers import inspect as holo_inspect
from ._reader import HoloReader
from ._writer import HoloWriter

_EXIT_OK = 0
_EXIT_FORMAT = 1
_EXIT_IO = 2
_EXIT_ARGS = 3


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        info = holo_inspect(args.file)
    except HoloFormatError as e:
        print(f"Format error: {e}", file=sys.stderr)
        return _EXIT_FORMAT
    except (HoloIOError, OSError) as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return _EXIT_IO

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"File      : {info['file']}")
        print(f"File size : {info['file_size']} bytes")
        print(f"Version   : {info['version']}")
        print(f"Bit depth : {info['bit_depth']}")
        print(f"Width     : {info['width']}")
        print(f"Height    : {info['height']}")
        print(f"Frames    : {info['num_frames']}")
        print(f"Data size : {info['data_size']} bytes")
        print(f"Frame size: {info['frame_size']} bytes")
        print(f"Endian    : {info['endian']}")
        print(f"Data type : {info['data_type']}")
        if info["footer_keys"]:
            print(f"Footer    : {info['footer_keys']}")
        else:
            print("Footer    : (none)")
    return _EXIT_OK


def _cmd_read(args: argparse.Namespace) -> int:
    try:
        with HoloReader(args.file) as r:
            start = args.start or 0
            stop = args.stop
            step = args.step or 1
            frames = r.read(start, stop, step=step)
    except HoloFormatError as e:
        print(f"Format error: {e}", file=sys.stderr)
        return _EXIT_FORMAT
    except (HoloIOError, OSError) as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return _EXIT_IO

    fmt = args.format or "raw"
    out = args.out

    try:
        if fmt == "raw":
            data = frames.tobytes()
            if out:
                with open(out, "wb") as f:
                    f.write(data)
            else:
                sys.stdout.buffer.write(data)
        elif fmt == "npy":
            import numpy as np
            if out:
                np.save(out, frames)
            else:
                import io
                buf = io.BytesIO()
                np.save(buf, frames)
                sys.stdout.buffer.write(buf.getvalue())
        elif fmt == "tiff":
            try:
                import tifffile
            except ImportError:
                print("tifffile is required for TIFF output: pip install tifffile", file=sys.stderr)
                return _EXIT_ARGS
            if out:
                tifffile.imwrite(out, frames)
            else:
                print("--out is required for TIFF format", file=sys.stderr)
                return _EXIT_ARGS
        else:
            print(f"Unknown format: {fmt}", file=sys.stderr)
            return _EXIT_ARGS
    except OSError as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return _EXIT_IO

    return _EXIT_OK


def _cmd_write(args: argparse.Namespace) -> int:
    data_type_map = {"raw": DataType.RAW, "processed": DataType.PROCESSED, "moments": DataType.MOMENTS}
    endian_map = {"little": Endian.LITTLE, "big": Endian.BIG}

    try:
        with HoloWriter(
            args.file,
            bit_depth=args.bit_depth,
            width=args.width,
            height=args.height,
            data_type=data_type_map.get(args.data_type or "raw", DataType.RAW),
            endian=endian_map.get(args.endian or "little", Endian.LITTLE),
            overwrite=args.overwrite,
        ) as w:
            frame_size = args.width * args.height * (args.bit_depth // 8)
            import numpy as np
            dtype = w.header.dtype
            while True:
                chunk = sys.stdin.buffer.read(frame_size)
                if not chunk:
                    break
                if len(chunk) < frame_size:
                    print("Warning: incomplete final frame, skipping", file=sys.stderr)
                    break
                arr = np.frombuffer(chunk, dtype=dtype).reshape(args.height, args.width)
                w.write(arr)
    except FileExistsError:
        print(f"File already exists: {args.file}. Use --overwrite to replace.", file=sys.stderr)
        return _EXIT_FORMAT
    except HoloFormatError as e:
        print(f"Format error: {e}", file=sys.stderr)
        return _EXIT_FORMAT
    except (HoloIOError, OSError) as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return _EXIT_IO

    return _EXIT_OK


def _cmd_append(args: argparse.Namespace) -> int:
    try:
        with HoloReader(args.file) as r:
            h = r.header
        with HoloWriter(
            args.file,
            bit_depth=h.bit_depth,
            width=h.width,
            height=h.height,
            endian=h.endian,
            data_type=h.data_type,
            append=True,
        ) as w:
            frame_size = h.frame_size
            import numpy as np
            dtype = h.dtype
            while True:
                chunk = sys.stdin.buffer.read(frame_size)
                if not chunk:
                    break
                if len(chunk) < frame_size:
                    print("Warning: incomplete final frame, skipping", file=sys.stderr)
                    break
                arr = np.frombuffer(chunk, dtype=dtype).reshape(h.height, h.width)
                w.write(arr)
    except HoloFormatError as e:
        print(f"Format error: {e}", file=sys.stderr)
        return _EXIT_FORMAT
    except (HoloIOError, OSError) as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return _EXIT_IO

    return _EXIT_OK


def _cmd_footer_get(args: argparse.Namespace) -> int:
    from ._helpers import read_footer
    try:
        footer = read_footer(args.file)
    except HoloFormatError as e:
        print(f"Format error: {e}", file=sys.stderr)
        return _EXIT_FORMAT
    except (HoloIOError, OSError) as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return _EXIT_IO
    print(footer.to_json())
    return _EXIT_OK


def _cmd_footer_set(args: argparse.Namespace) -> int:
    if args.json_str:
        json_str = args.json_str
    else:
        json_str = sys.stdin.read()

    try:
        footer = HoloFooter.from_json(json_str)
    except Exception as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return _EXIT_ARGS

    try:
        with HoloReader(args.file) as r:
            h = r.header
            frames = r.read()

        with HoloWriter(
            args.file,
            bit_depth=h.bit_depth,
            width=h.width,
            height=h.height,
            endian=h.endian,
            data_type=h.data_type,
            overwrite=True,
        ) as w:
            w.write(frames)
            w.set_footer(footer)
    except HoloFormatError as e:
        print(f"Format error: {e}", file=sys.stderr)
        return _EXIT_FORMAT
    except (HoloIOError, OSError) as e:
        print(f"I/O error: {e}", file=sys.stderr)
        return _EXIT_IO

    return _EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holo",
        description="Read and write Holovibes .holo files",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Print header + footer summary")
    p_inspect.add_argument("file")
    p_inspect.add_argument("--json", action="store_true", help="Machine-readable JSON output")

    # read
    p_read = sub.add_parser("read", help="Extract frame range to stdout or file")
    p_read.add_argument("file")
    p_read.add_argument("--start", type=int)
    p_read.add_argument("--stop", type=int)
    p_read.add_argument("--step", type=int)
    p_read.add_argument("--out", help="Output path (default: stdout)")
    p_read.add_argument("--format", choices=["raw", "npy", "tiff"], default="raw")

    # write
    p_write = sub.add_parser("write", help="Write raw stdin stream into a new .holo file")
    p_write.add_argument("file")
    p_write.add_argument("--width", type=int, required=True)
    p_write.add_argument("--height", type=int, required=True)
    p_write.add_argument("--bit-depth", type=int, required=True, dest="bit_depth")
    p_write.add_argument("--data-type", choices=["raw", "processed", "moments"], default="raw", dest="data_type")
    p_write.add_argument("--endian", choices=["little", "big"], default="little")
    p_write.add_argument("--overwrite", action="store_true")

    # append
    p_append = sub.add_parser("append", help="Append raw stdin frames to existing file")
    p_append.add_argument("file")

    # footer
    p_footer = sub.add_parser("footer", help="Footer operations")
    footer_sub = p_footer.add_subparsers(dest="footer_command", required=True)

    p_footer_get = footer_sub.add_parser("get", help="Print footer JSON")
    p_footer_get.add_argument("file")

    p_footer_set = footer_sub.add_parser("set", help="Replace footer from stdin or --json")
    p_footer_set.add_argument("file")
    p_footer_set.add_argument("--json", dest="json_str", help="JSON string (default: read from stdin)")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "inspect": _cmd_inspect,
        "read": _cmd_read,
        "write": _cmd_write,
        "append": _cmd_append,
    }

    if args.command == "footer":
        footer_dispatch = {
            "get": _cmd_footer_get,
            "set": _cmd_footer_set,
        }
        code = footer_dispatch[args.footer_command](args)
    else:
        code = dispatch[args.command](args)

    sys.exit(code)


if __name__ == "__main__":
    main()
