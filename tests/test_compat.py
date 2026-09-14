# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
# 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022
# Python Software Foundation; All Rights Reserved

# This file is part of python-isal which is distributed under the
# PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2.

# This file does not include original code from CPython. It is used to ensure
# that compression and decompression between CPython's zlib and isal_zlib
# is compatible.

import gzip
import itertools
import sys
import zlib
from pathlib import Path

from isal import igzip, isal_zlib

import pytest

DATA_DIR = Path(__file__).parent / "data"
COMPRESSED_FILE = DATA_DIR / "test.fastq.gz"
with gzip.open(str(COMPRESSED_FILE), mode="rb") as file_h:
    DATA = file_h.read()

DATA_SIZES = [2**i for i in range(3, 20)]
# 100 seeds generated with random.randint(0, 2**32-1)
SEEDS_FILE = DATA_DIR / "seeds.txt"
INT_OVERFLOW = 211928379812738912738917238971289378912379823871932719823798123
# Get some negative ints and some really big ints into the mix.
SEEDS = [-INT_OVERFLOW, -3, -1, 0, 1, INT_OVERFLOW] + [
    int(seed) for seed in SEEDS_FILE.read_text().splitlines()]

# Wbits for ZLIB compression, GZIP compression, and RAW compressed streams
WBITS_RANGE = list(range(9, 16)) + list(range(25, 32)) + list(range(-15, -8))


@pytest.mark.parametrize(["data_size", "value"],
                         itertools.product(DATA_SIZES, SEEDS))
def test_crc32(data_size, value):
    data = DATA[:data_size]
    assert zlib.crc32(data, value) == isal_zlib.crc32(data, value)


@pytest.mark.parametrize(["data_size", "value"],
                         itertools.product(DATA_SIZES, SEEDS))
def test_adler32(data_size, value):
    data = DATA[:data_size]
    assert zlib.adler32(data, value) == isal_zlib.adler32(data, value)


@pytest.mark.parametrize(["data_size", "level"],
                         itertools.product(DATA_SIZES, range(4)))
def test_compress(data_size, level):
    data = DATA[:data_size]
    compressed = isal_zlib.compress(data, level=level)
    assert zlib.decompress(compressed) == data


@pytest.mark.parametrize(["data_size", "level"],
                         itertools.product(DATA_SIZES, range(10)))
def test_decompress_zlib(data_size, level):
    data = DATA[:data_size]
    compressed = zlib.compress(data, level=level)
    decompressed = isal_zlib.decompress(compressed)
    assert decompressed == data


@pytest.mark.parametrize(["data_size", "level", "wbits", "memLevel"],
                         itertools.product([128 * 1024], range(4),
                                           WBITS_RANGE, range(1, 10)))
def test_decompress_wbits(data_size, level, wbits, memLevel):
    data = DATA[:data_size]
    compressobj = zlib.compressobj(level=level, wbits=wbits, memLevel=memLevel)
    compressed = compressobj.compress(data) + compressobj.flush()
    decompressed = isal_zlib.decompress(compressed, wbits=wbits)
    assert data == decompressed


@pytest.mark.parametrize(["data_size", "level"],
                         itertools.product(DATA_SIZES, range(4)))
def test_decompress_isal_zlib(data_size, level):
    data = DATA[:data_size]
    compressed = isal_zlib.compress(data, level=level)
    decompressed = isal_zlib.decompress(compressed)
    print(len(decompressed))
    assert decompressed == data


@pytest.mark.parametrize(["data_size", "level", "wbits", "memLevel"],
                         itertools.product([128 * 1024], range(4),
                                           WBITS_RANGE, range(1, 10)))
def test_compress_compressobj(data_size, level, wbits, memLevel):
    data = DATA[:data_size]
    compressobj = isal_zlib.compressobj(level=level,
                                        wbits=wbits,
                                        memLevel=memLevel)
    compressed = compressobj.compress(data) + compressobj.flush()
    decompressed = zlib.decompress(compressed, wbits=wbits)
    assert data == decompressed


@pytest.mark.parametrize(["data_size", "level", "wbits", "memLevel"],
                         itertools.product([128 * 1024], range(4),
                                           WBITS_RANGE, range(1, 10)))
def test_decompress_decompressobj(data_size, level, wbits, memLevel):
    data = DATA[:data_size]
    compressobj = zlib.compressobj(level=level, wbits=wbits, memLevel=memLevel)
    compressed = compressobj.compress(data) + compressobj.flush()
    decompressobj = isal_zlib.decompressobj(wbits=wbits)
    decompressed = decompressobj.decompress(compressed) + decompressobj.flush()
    assert data == decompressed
    assert decompressobj.unused_data == b""
    assert decompressobj.unconsumed_tail == b""


def test_decompressobj_unconsumed_tail():
    data = DATA[:128*1024]
    compressed = zlib.compress(data)
    decompressobj = isal_zlib.decompressobj()
    output = decompressobj.decompress(compressed, 2048)
    assert len(output) == 2048


@pytest.mark.parametrize(["data_size", "level"],
                         itertools.product(DATA_SIZES, range(4)))
def test_igzip_compress(data_size, level):
    data = DATA[:data_size]
    compressed = igzip.compress(data, compresslevel=level)
    assert gzip.decompress(compressed) == data


@pytest.mark.parametrize(["data_size", "level"],
                         itertools.product(DATA_SIZES, range(10)))
def test_decompress_gzip(data_size, level):
    data = DATA[:data_size]
    compressed = gzip.compress(data, compresslevel=level)
    decompressed = igzip.decompress(compressed)
    assert decompressed == data


@pytest.mark.parametrize(["data_size", "level"],
                         itertools.product(DATA_SIZES, range(4)))
def test_decompress_igzip(data_size, level):
    data = DATA[:data_size]
    compressed = igzip.compress(data, compresslevel=level)
    decompressed = igzip.decompress(compressed)
    print(len(decompressed))
    assert decompressed == data


@pytest.mark.parametrize(["unused_size", "wbits"],
                         itertools.product([26], [-15, 15, 31]))
def test_unused_data(unused_size, wbits):
    unused_data = b"abcdefghijklmnopqrstuvwxyz"[:unused_size]
    compressor = zlib.compressobj(wbits=wbits)
    data = b"A meaningful sentence starts with a capital and ends with a."
    compressed = compressor.compress(data) + compressor.flush()
    decompressor = isal_zlib.decompressobj(wbits=wbits)
    result = decompressor.decompress(compressed + unused_data)
    assert result == data
    assert decompressor.unused_data == unused_data


def test_zlib_dictionary_decompress():
    dictionary = b"bla"
    data = b"bladiebla"
    compobj = zlib.compressobj(zdict=dictionary)
    compressed = compobj.compress(data) + compobj.flush()
    decompobj = isal_zlib.decompressobj(zdict=dictionary)
    assert decompobj.decompress(compressed) == data


def test_isal_zlib_dictionary_decompress():
    dictionary = b"bla"
    data = b"bladiebla"
    compobj = isal_zlib.compressobj(zdict=dictionary)
    compressed = compobj.compress(data) + compobj.flush()
    decompobj = zlib.decompressobj(zdict=dictionary)
    assert decompobj.decompress(compressed) == data


# Sizes around the 16 KiB initial buffer, the 128 KiB block size used for
# file streaming, and the 4 MiB message limit used by websocket libraries.
MAX_LENGTHS = [1, 100, 16 * 1024 - 1, 16 * 1024, 16 * 1024 + 1, 128 * 1024,
               512 * 1024 + 1]


@pytest.mark.parametrize(["max_length", "wbits"],
                         itertools.product(MAX_LENGTHS, [15, 31, -15]))
def test_decompressobj_max_length_fixed_blocks(max_length, wbits):
    # All input is always available, so every chunk except the last must be
    # exactly max_length bytes.
    compressobj = zlib.compressobj(wbits=wbits)
    compressed = compressobj.compress(DATA) + compressobj.flush()
    decompressobj = isal_zlib.decompressobj(wbits=wbits)
    chunks = []
    buf = compressed
    while not decompressobj.eof:
        chunks.append(decompressobj.decompress(buf, max_length))
        buf = decompressobj.unconsumed_tail
    assert all(len(chunk) == max_length for chunk in chunks[:-1])
    assert len(chunks[-1]) <= max_length
    assert b"".join(chunks) == DATA
    assert decompressobj.unconsumed_tail == b""
    assert decompressobj.unused_data == b""


@pytest.mark.parametrize(["size", "max_length"],
                         itertools.product(
                             [0, 1, 100, 512 * 1024, 3 * 1024 * 1024],
                             [4 * 1024 * 1024 + 1, 16 * 1024 * 1024 + 1,
                              sys.maxsize]))
def test_decompressobj_max_length_larger_than_output(size, max_length):
    data = DATA[:size]
    compressed = zlib.compress(data)
    decompressobj = isal_zlib.decompressobj()
    assert decompressobj.decompress(compressed, max_length) == data
    assert decompressobj.eof
    assert decompressobj.unconsumed_tail == b""


def test_decompressobj_max_length_zero_is_unbounded():
    compressed = zlib.compress(DATA)
    decompressobj = isal_zlib.decompressobj()
    assert decompressobj.decompress(compressed, 0) == DATA
    assert decompressobj.eof


def test_decompressobj_max_length_exact_fill():
    data = DATA[:200_000]
    compressed = zlib.compress(data)
    for factory in (isal_zlib.decompressobj, zlib.decompressobj):
        decompressobj = factory()
        assert decompressobj.decompress(compressed, len(data)) == data
        assert decompressobj.decompress(decompressobj.unconsumed_tail,
                                        len(data)) == b""
        assert decompressobj.eof
        assert decompressobj.unconsumed_tail == b""


def test_decompressobj_max_length_empty_input_repeated():
    compressed = zlib.compress(DATA)
    decompressobj = isal_zlib.decompressobj()
    max_length = 4 * 1024 * 1024 + 1
    for _ in range(3):
        assert decompressobj.decompress(b"", max_length) == b""
    assert decompressobj.decompress(compressed[:10], max_length) == b""
    assert decompressobj.decompress(compressed[10:], max_length) == DATA
    assert decompressobj.eof


def test_decompressobj_websocket_like():
    # permessage-deflate with context takeover: one raw deflate stream, one
    # sync flush per message, decompressed with a message size limit.
    max_length = 4 * 1024 * 1024 + 1
    messages = [DATA[i * 1024:i * 1024 + 512 * 1024] for i in range(20)]
    compressobj = isal_zlib.compressobj(wbits=-15)
    decompressobj = isal_zlib.decompressobj(wbits=-15)
    for message in messages:
        payload = (compressobj.compress(message) +
                   compressobj.flush(isal_zlib.Z_SYNC_FLUSH))
        assert decompressobj.decompress(payload, max_length) == message
        assert decompressobj.unconsumed_tail == b""
