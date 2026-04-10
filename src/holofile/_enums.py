import enum


class Endian(enum.IntEnum):
    LITTLE = 0
    BIG = 1


class DataType(enum.IntEnum):
    RAW = 0
    PROCESSED = 1
    MOMENTS = 2
