"""Benchmark output buffer allocation behaviour of the decompress and
compress paths.

Reports min/median/max wall-clock per case over several rounds, plus the
spread (max / min). A large spread on the same machine indicates that the
run time depends on heap layout (for example because output buffers are
grown with realloc and sometimes have to be copied).

The ``aiohttp-*`` cases mirror how aiohttp calls the zlib backend:

- websocket receive: raw deflate, one decompressobj per connection,
  ``decompress(unconsumed_tail + payload + b"\\x00\\x00\\xff\\xff",
  max_msg_size + 1)`` with a 4 MiB default message size.
- websocket send: ``compress(message) + flush(Z_SYNC_FLUSH)`` at best speed.
- HTTP body: gzip, network chunks of 256 KiB, ``decompress(unconsumed_tail +
  chunk, max_length=256 KiB)``, draining the tail before the next chunk.

Each case builds its input data only when it runs, so ``--cases`` skips the
setup of other cases and one case's data does not affect the next case's heap.
Every case returns the number of uncompressed bytes it processed, and is run
once and checked against the expected size before it is timed, so truncated
output fails instead of showing up as a speed-up.

Usage:
    python benchmark_scripts/benchmark_output_buffer.py
    python benchmark_scripts/benchmark_output_buffer.py \
        --cases aiohttp-ws-recv-512K-x,aiohttp-ws-recv-100B
"""

import argparse
import functools
import gc
import gzip
import os
import platform
import statistics
import sys
import timeit
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from isal import igzip, igzip_lib, isal_zlib

DATA_DIR = Path(__file__).parent.parent / "tests" / "data"
COMPRESSED_FILE = DATA_DIR / "test.fastq.gz"
with gzip.open(str(COMPRESSED_FILE), mode="rb") as file_h:
    FASTQ = file_h.read()

KIB = 1024
MIB = 1024 * 1024
# aiohttp defaults
WS_MAX_MSG_SIZE = 4 * MIB
WS_MAX_LENGTH = WS_MAX_MSG_SIZE + 1
WS_DEFLATE_TRAILING = b"\x00\x00\xff\xff"
HTTP_CHUNK_SIZE = 256 * KIB


def compressible(size: int) -> bytes:
    if size <= len(FASTQ):
        return FASTQ[:size]
    data = bytearray(FASTQ * (size // len(FASTQ)))
    data += FASTQ[:size - len(data)]
    return bytes(data)


def sync_flushed_messages(messages: List[bytes]) -> List[bytes]:
    """Compress messages the way permessage-deflate does: one raw deflate
    stream with a sync flush after every message (context takeover). The
    trailing 00 00 ff ff is stripped, as on the wire."""
    compressor = isal_zlib.compressobj(isal_zlib.Z_BEST_SPEED, wbits=-15)
    return [(compressor.compress(msg) +
             compressor.flush(isal_zlib.Z_SYNC_FLUSH)
             ).removesuffix(WS_DEFLATE_TRAILING)
            for msg in messages]


def stream_decompressobj(compressed: bytes, block_size: int) -> int:
    """gzip-module style read loop: feed input in slices, always ask for at
    most block_size bytes of output."""
    decompressor = isal_zlib.decompressobj()
    total = 0
    position = 0
    while not decompressor.eof:
        tail = decompressor.unconsumed_tail
        if tail:
            buf = tail
        else:
            buf = compressed[position:position + block_size]
            position += block_size
            if not buf:
                break
        total += len(decompressor.decompress(buf, block_size))
    if not decompressor.eof:
        raise RuntimeError("input ended before the end of the stream")
    return total


def stream_igzipdecompressor(compressed: bytes, block_size: int) -> int:
    decompressor = igzip_lib.IgzipDecompressor(flag=igzip_lib.DECOMP_ZLIB)
    total = 0
    position = 0
    while not decompressor.eof:
        if decompressor.needs_input:
            buf = compressed[position:position + block_size]
            position += block_size
            if not buf:
                break
        else:
            buf = b""
        total += len(decompressor.decompress(buf, block_size))
    if not decompressor.eof:
        raise RuntimeError("input ended before the end of the stream")
    return total


def websocket_receive(payloads: List[bytes], max_length: int) -> int:
    decompressor = isal_zlib.decompressobj(wbits=-15)
    total = 0
    for payload in payloads:
        out = decompressor.decompress(
            decompressor.unconsumed_tail + payload + WS_DEFLATE_TRAILING,
            max_length)
        if len(out) > WS_MAX_MSG_SIZE:
            raise RuntimeError("message exceeded max_msg_size")
        total += len(out)
    return total


def websocket_send(messages: List[bytes]) -> int:
    compressor = isal_zlib.compressobj(isal_zlib.Z_BEST_SPEED, wbits=-15)
    total = 0
    for message in messages:
        compressor.compress(message) + compressor.flush(isal_zlib.Z_SYNC_FLUSH)
        total += len(message)
    return total


def http_body_decompress(body: bytes, chunk_size: int,
                         max_length: int) -> int:
    decompressor = isal_zlib.decompressobj(wbits=31)
    total = 0
    for position in range(0, len(body), chunk_size):
        chunk = body[position:position + chunk_size]
        total += len(decompressor.decompress(
            decompressor.unconsumed_tail + chunk, max_length))
        while decompressor.unconsumed_tail:
            total += len(decompressor.decompress(
                decompressor.unconsumed_tail, max_length))
    if not decompressor.eof or decompressor.unused_data:
        raise RuntimeError("body did not end exactly at the end of the stream")
    return total


TimedCall = Callable[[], int]
# name -> (build the data and return the call to time, bytes per call)
Cases = Dict[str, Tuple[Callable[[], TimedCall], int]]


def ws_x_messages() -> List[bytes]:
    # Issue #256 / aiohttp benchmark: 200 messages of b"x" * 512 KiB.
    return [b"x" * (512 * KIB)] * 200


def ws_mixed_messages() -> List[bytes]:
    # 200 messages of 512 KiB, half incompressible and half text.
    return [os.urandom(256 * KIB) + compressible(256 * KIB)
            for _ in range(8)] * 25


def ws_tiny_messages() -> List[bytes]:
    # 20000 messages of 100 bytes.
    return [os.urandom(50) + compressible(50) for _ in range(8)] * 2500


def build_cases(size_mib: int) -> Cases:
    size = size_mib * MIB
    http_body_size = min(size_mib, 64) * MIB
    ws_bytes = 200 * 512 * KIB
    tiny_bytes = 20000 * 100
    partial = functools.partial

    def stream(func: Callable[[bytes, int], int]) -> TimedCall:
        return partial(func, isal_zlib.compress(compressible(size), 1),
                       128 * KIB)

    def ws_recv(messages: Callable[[], List[bytes]]) -> TimedCall:
        return partial(websocket_receive, sync_flushed_messages(messages()),
                       WS_MAX_LENGTH)

    def ws_send(messages: Callable[[], List[bytes]]) -> TimedCall:
        return partial(websocket_send, messages())

    def http_body() -> TimedCall:
        body = igzip.compress(compressible(http_body_size), 1)
        return partial(http_body_decompress, body, HTTP_CHUNK_SIZE,
                       HTTP_CHUNK_SIZE)

    def oneshot() -> TimedCall:
        compressed = isal_zlib.compress(compressible(size), 1)
        return lambda: len(isal_zlib.decompress(compressed))

    def compressobj_incompressible() -> TimedCall:
        data = os.urandom(128 * MIB)

        def run() -> int:
            compressor = isal_zlib.compressobj()
            compressor.compress(data) + compressor.flush()
            return len(data)
        return run

    def compress_chunks() -> TimedCall:
        chunks = [compressible(128 * KIB)] * 200

        def run() -> int:
            for chunk in chunks:
                isal_zlib.compress(chunk, 1)
            return len(chunks) * len(chunks[0])
        return run

    return {
        "stream-128K-decompressobj": (
            partial(stream, stream_decompressobj), size),
        "stream-128K-igzipdecompressor": (
            partial(stream, stream_igzipdecompressor), size),
        "aiohttp-ws-recv-512K-x": (partial(ws_recv, ws_x_messages), ws_bytes),
        "aiohttp-ws-recv-512K-mixed": (
            partial(ws_recv, ws_mixed_messages), ws_bytes),
        "aiohttp-ws-recv-100B": (partial(ws_recv, ws_tiny_messages),
                                 tiny_bytes),
        "aiohttp-ws-send-512K-x": (partial(ws_send, ws_x_messages), ws_bytes),
        "aiohttp-ws-send-512K-mixed": (
            partial(ws_send, ws_mixed_messages), ws_bytes),
        "aiohttp-ws-send-100B": (partial(ws_send, ws_tiny_messages),
                                 tiny_bytes),
        f"aiohttp-http-body-{http_body_size // MIB}M": (
            http_body, http_body_size),
        f"oneshot-{size_mib}M": (oneshot, size),
        "compressobj-128M-incompressible": (compressobj_incompressible,
                                            128 * MIB),
        "compress-128K-chunks": (compress_chunks, 200 * 128 * KIB),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, default=256,
                        help="Size of the large decompressed payload.")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated subset of case names.")
    args = parser.parse_args()

    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"# {platform.platform()} {platform.machine()}")
    print(f"# Python {sys.version.split()[0]} gil={gil_enabled} "
          f"zlib={zlib.ZLIB_RUNTIME_VERSION}")

    cases = build_cases(args.size_mib)
    if args.cases:
        wanted = args.cases.split(",")
        unknown = set(wanted) - set(cases)
        if unknown:
            parser.error(f"unknown cases: {sorted(unknown)}; "
                         f"available: {sorted(cases)}")
        cases = {name: cases[name] for name in wanted}

    print("case\tmin_ms\tmedian_ms\tmax_ms\tspread\tMiB/s")
    for name, (setup, nbytes) in cases.items():
        func = setup()
        processed = func()
        if processed != nbytes:
            raise RuntimeError(f"{name} processed {processed} bytes, "
                               f"expected {nbytes}")
        gc.collect()
        # timeit disables the garbage collector while timing.
        timings = timeit.repeat(func, number=1, repeat=args.rounds)
        del func
        fastest = min(timings)
        print("{0}\t{1:.2f}\t{2:.2f}\t{3:.2f}\t{4:.3f}\t{5:.0f}".format(
            name,
            fastest * 1000,
            statistics.median(timings) * 1000,
            max(timings) * 1000,
            max(timings) / fastest,
            nbytes / MIB / fastest,
        ))


if __name__ == "__main__":
    main()
