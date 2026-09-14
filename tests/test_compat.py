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
import os
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


@pytest.mark.parametrize(["size", "level"],
                         itertools.product([0, 1, 100, 4095, 4096, 4097,
                                            16 * 1024 - 1, 16 * 1024],
                                           range(4)))
def test_compressobj_small_inputs(size, level):
    # Small inputs and flushes are compressed into a stack buffer first.
    data = DATA[:size]
    compressobj = isal_zlib.compressobj(level)
    compressed = compressobj.compress(data) + compressobj.flush()
    assert zlib.decompress(compressed) == data


def test_compressobj_compress_output_fills_stack_buffer():
    # Level 0 emits incompressible input right away, so a single small
    # compress call fills the 4 KiB stack buffer and has to continue in a
    # bytes object. The remaining output is returned by later calls.
    data = os.urandom(16 * 1024 - 1)
    compressobj = isal_zlib.compressobj(0)
    first = compressobj.compress(data)
    assert len(first) >= 4096
    assert zlib.decompress(first + compressobj.flush()) == data


@pytest.mark.parametrize("level", range(1, 4))
def test_compressobj_flush_output_larger_than_stack_buffer(level):
    # Levels 1-3 hold input back until the flush, so the flush output of
    # many small messages does not fit the 4 KiB stack buffer.
    messages = [os.urandom(1000) for _ in range(30)]
    compressobj = isal_zlib.compressobj(level, wbits=-15)
    decompressobj = zlib.decompressobj(wbits=-15)
    compressed = b"".join(compressobj.compress(m) for m in messages)
    flushed = compressobj.flush(isal_zlib.Z_SYNC_FLUSH)
    assert len(flushed) > 4096
    assert decompressobj.decompress(compressed + flushed) == b"".join(messages)


@pytest.mark.parametrize(["level", "flush_mode"],
                         itertools.product(range(4),
                                           ["Z_SYNC_FLUSH", "Z_FULL_FLUSH"]))
def test_compressobj_websocket_like_small_messages(level, flush_mode):
    # permessage-deflate: one raw deflate stream, one flush per message.
    compressobj = isal_zlib.compressobj(level, wbits=-15)
    decompressobj = zlib.decompressobj(wbits=-15)
    mode = getattr(isal_zlib, flush_mode)
    for i in range(200):
        message = DATA[i * 997:i * 997 + (i * 37) % 6000]
        payload = compressobj.compress(message) + compressobj.flush(mode)
        assert decompressobj.decompress(payload) == message
