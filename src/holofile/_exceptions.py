class HoloError(Exception):
    """Base exception for all holofile errors."""


class HoloFormatError(HoloError):
    """Magic number mismatch, unsupported version, or corrupt header."""


class HoloShapeError(HoloError):
    """Frame dimensions do not match the header on write."""


class HoloDTypeError(HoloError):
    """dtype / bit_depth mismatch on write."""


class HoloIndexError(HoloError):
    """Frame index out of range."""


class HoloIOError(HoloError):
    """OS-level I/O failure."""
