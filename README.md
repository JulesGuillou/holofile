# holofile

Python I/O library for Holovibes `.holo` files (format version 7).

Provides a `HoloReader` / `HoloWriter` API and a `holo` CLI for reading, writing, inspecting, and manipulating `.holo` acquisition files produced by [Holovibes](https://holovibes.com).

---

## Installation

`holofile` is not yet published on PyPI. You can install it directly from the GitHub repository.

### Requirements
* **Python** $\ge$ 3.11
* [**uv**](https://docs.astral.sh/uv/) (recommended) or **pip**

### Using uv (Recommended)
To add `holofile` to your current project:
```bash
uv add git+https://github.com/JulesGuillou/holofile.git@main
```

### Using pip
To install the package into your environment:
```bash
pip install git+https://github.com/JulesGuillou/holofile.git@main
```

---

## Quick start

### Reading

```python
import holofile

# Read all frames as a NumPy array
with holofile.HoloReader("acquisition.holo") as r:
    print(r.header)          # HoloHeader(version=7, bit_depth=16, width=1920, ...)
    frames = r.read()        # shape: (num_frames, height, width), dtype: uint16

# Slice a range
with holofile.HoloReader("acquisition.holo") as r:
    batch = r.read(100, 200)      # frames 100–199
    every_other = r.read(step=2)  # every second frame
    frame = r[42]                 # single frame

# Memory-mapped access (recommended for large files / random access)
with holofile.HoloReader("acquisition.holo", mmap=True) as r:
    frame = r[1000]
```

### Writing

```python
import numpy as np
import holofile

frames = np.random.randint(0, 65536, (100, 1080, 1920), dtype=np.uint16)

with holofile.HoloWriter("output.holo", bit_depth=16, width=1920, height=1080) as w:
    w.write(frames)
    w.set_footer({"camera": "MyCamera", "fps": 30})
```

### Appending

```python
with holofile.HoloWriter("output.holo", bit_depth=16, width=1920, height=1080, append=True) as w:
    w.write(new_frames)
```

### Streaming (zero-copy chunks)

```python
import numpy as np

with holofile.HoloReader("large.holo") as r:
    buf = np.empty((64, r.header.height, r.header.width), dtype=r.header.dtype)
    for chunk in r.iter_chunks(64, buf=buf):
        process(chunk)   # chunk is a view into buf — no allocation per iteration
```

### Caller-owned buffer (`read_into`)

Place frames directly into any PEP 3118 buffer — NumPy arrays, `bytearray`, CuPy pinned host arrays, JAX host buffers, etc.:

```python
import numpy as np

with holofile.HoloReader("acquisition.holo") as r:
    buf = np.empty((50, r.header.height, r.header.width), dtype=r.header.dtype)
    n = r.read_into(buf, start=0, stop=50)
```

---

## API reference

### `HoloHeader`

Immutable dataclass representing the 64-byte file header.

| Field | Type | Description |
|---|---|---|
| `version` | `int` | Always `7` |
| `bit_depth` | `int` | Bits per pixel (multiple of 8) |
| `width` | `int` | Frame width in pixels |
| `height` | `int` | Frame height in pixels |
| `num_frames` | `int` | Number of frames |
| `data_size` | `int` | Total data section size in bytes |
| `endian` | `Endian` | `Endian.LITTLE` or `Endian.BIG` |
| `data_type` | `DataType` | `RAW`, `PROCESSED`, or `MOMENTS` |

Computed properties: `frame_size`, `dtype` (NumPy dtype).

### `HoloReader`

```python
HoloReader(path, *, mmap=False)
```

| Method / property | Description |
|---|---|
| `header` | Parsed `HoloHeader` |
| `footer` | Parsed `HoloFooter` (lazy, returns empty if absent) |
| `read(start, stop, *, step)` | Allocate and return frames as `ndarray` |
| `read_into(buf, start, stop, *, step)` | Fill caller-owned buffer; returns frame count |
| `iter_chunks(chunk_size, *, buf)` | Yield successive chunks; reuses `buf` if provided |
| `__iter__` | Yield one frame at a time |
| `__len__` | `header.num_frames` |
| `__getitem__` | Integer or slice indexing |

### `HoloWriter`

```python
HoloWriter(path, *, bit_depth, width, height,
           endian=Endian.LITTLE, data_type=DataType.RAW,
           overwrite=False, append=False)
```

`overwrite` and `append` are mutually exclusive. By default the writer raises `FileExistsError` if the file already exists.

| Method / property | Description |
|---|---|
| `write(data)` | Write a 2-D frame or 3-D batch; returns frame count |
| `set_footer(footer)` | Attach a footer (written on `close()`) |
| `frames_written` | Number of frames written so far |
| `header` | Live `HoloHeader` reflecting current state |

`num_frames` and `data_size` in the header are patched on `close()`, so streaming writes work without knowing the total frame count upfront.

### `HoloFooter`

```python
HoloFooter(data: dict)
HoloFooter.from_json(s)   # parse
HoloFooter.empty()        # {}
footer.get(key, default)
footer.to_json()
```

### Module-level helpers

```python
holofile.read_header(path) -> HoloHeader
holofile.read_footer(path) -> HoloFooter
holofile.inspect(path)     -> dict          # summary of header + footer + file size
```

### Exceptions

```
HoloError                 base class
├── HoloFormatError       bad magic, unsupported version, corrupt header
├── HoloShapeError        frame dimensions mismatch on write
├── HoloDTypeError        dtype / bit_depth mismatch on write
├── HoloIndexError        frame index out of range
└── HoloIOError           OS-level I/O failure
```

---

## CLI

The `holo` command is installed automatically.

### `inspect`

```bash
holo inspect acquisition.holo
holo inspect acquisition.holo --json
holo inspect acquisition.holo --header-only        # skip footer, faster on large files
holo inspect acquisition.holo --header-only --json
```

### `read`

```bash
# Raw binary to stdout
holo read acquisition.holo --start 0 --stop 100 | your_pipeline

# Save as NumPy
holo read acquisition.holo --format npy --out frames.npy

# Save as TIFF (requires tifffile)
holo read acquisition.holo --format tiff --out frames.tiff
```

### `write` / `append`

```bash
# Write from stdin
cat raw_frames.bin | holo write output.holo --width 1920 --height 1080 --bit-depth 16

# Append from stdin
cat more_frames.bin | holo append output.holo
```

### `footer`

```bash
# Print footer
holo footer get acquisition.holo

# Replace footer
holo footer set acquisition.holo --json '{"camera": "MyCamera", "fps": 30}'

# Or from stdin
echo '{"fps": 60}' | holo footer set acquisition.holo
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Format / validation error |
| 2 | I/O error |
| 3 | Bad arguments |

---

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=holofile
```

---

## File format

`.holo` files consist of three sections:

```
[ 64-byte header ][ frame data ][ optional JSON footer ]
```

The header is always little-endian regardless of payload byte order. `num_frames` and `data_size` are written as placeholders at open-time and patched on close, enabling streaming writes.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
