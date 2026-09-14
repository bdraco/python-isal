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

Usage:
    python benchmark_scripts/benchmark_output_buffer.py
    python benchmark_scripts/benchmark_output_buffer.py --cases ws-512K,ws-tiny
"""

import argparse
import gc
import gzip
import os
import platform
import statistics
import sys
import time
import zlib
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from isal import igzip_lib, isal_zlib

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
    repeats = size // len(FASTQ) + 1
    return (FASTQ * repeats)[:size]


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
        total += len(compressor.compress(message) +
                     compressor.flush(isal_zlib.Z_SYNC_FLUSH))
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
    return total


def build_cases(size_mib: int) -> Dict[str, Tuple[Callable[[], int], int]]:
    """Return {name: (callable, bytes processed per call)}."""
    big = compressible(size_mib * MIB)
    big_compressed = isal_zlib.compress(big, 1)
    del big

    # Issue #256 / aiohttp benchmark: 512 KiB messages of b"x".
    ws_x_messages = [b"x" * (512 * KIB)] * 200
    ws_x_payloads = sync_flushed_messages(ws_x_messages)
    # Realistic payloads: half incompressible, half text.
    ws_mixed_messages = [os.urandom(256 * KIB) + compressible(256 * KIB)
                         for _ in range(8)] * 25  # 200 messages
    ws_mixed_payloads = sync_flushed_messages(ws_mixed_messages)
    ws_bytes = 200 * 512 * KIB

    tiny_messages = [os.urandom(50) + compressible(50)
                     for _ in range(8)] * 2500  # 20000 messages
    tiny_payloads = sync_flushed_messages(tiny_messages)
    tiny_bytes = 100 * 20000

    http_body_size = min(size_mib, 64) * MIB
    http_body = gzip.compress(compressible(http_body_size), 6)

    incompressible = os.urandom(128 * MIB)
    chunks = [compressible(128 * KIB)] * 200

    def stream_128k_decompressobj() -> int:
        return stream_decompressobj(big_compressed, 128 * KIB)

    def stream_128k_igzipdecompressor() -> int:
        return stream_igzipdecompressor(big_compressed, 128 * KIB)

    def ws_recv_512k_x() -> int:
        return websocket_receive(ws_x_payloads, WS_MAX_LENGTH)

    def ws_recv_512k_mixed() -> int:
        return websocket_receive(ws_mixed_payloads, WS_MAX_LENGTH)

    def ws_recv_tiny() -> int:
        return websocket_receive(tiny_payloads, WS_MAX_LENGTH)

    def ws_send_512k_x() -> int:
        return websocket_send(ws_x_messages)

    def ws_send_512k_mixed() -> int:
        return websocket_send(ws_mixed_messages)

    def http_body_256k_chunks() -> int:
        return http_body_decompress(http_body, HTTP_CHUNK_SIZE,
                                    HTTP_CHUNK_SIZE)

    def oneshot() -> int:
        return len(isal_zlib.decompress(big_compressed))

    def compressobj_incompressible() -> int:
        compressor = isal_zlib.compressobj()
        return len(compressor.compress(incompressible)) + len(
            compressor.flush())

    def compress_chunks() -> int:
        return sum(len(isal_zlib.compress(chunk, 1)) for chunk in chunks)

    return {
        "stream-128K-decompressobj": (stream_128k_decompressobj,
                                      size_mib * MIB),
        "stream-128K-igzipdecompressor": (stream_128k_igzipdecompressor,
                                          size_mib * MIB),
        "aiohttp-ws-recv-512K-x": (ws_recv_512k_x, ws_bytes),
        "aiohttp-ws-recv-512K-mixed": (ws_recv_512k_mixed, ws_bytes),
        "aiohttp-ws-recv-100B": (ws_recv_tiny, tiny_bytes),
        "aiohttp-ws-send-512K-x": (ws_send_512k_x, ws_bytes),
        "aiohttp-ws-send-512K-mixed": (ws_send_512k_mixed, ws_bytes),
        f"aiohttp-http-body-{http_body_size // MIB}M": (
            http_body_256k_chunks, http_body_size),
        f"oneshot-{size_mib}M": (oneshot, size_mib * MIB),
        "compressobj-128M-incompressible": (compressobj_incompressible,
                                            128 * MIB),
        "compress-128K-chunks": (compress_chunks, 200 * 128 * KIB),
    }


def time_case(func: Callable[[], int], rounds: int) -> List[float]:
    timings = []
    for _ in range(rounds):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter()
            func()
            timings.append(time.perf_counter() - start)
        finally:
            gc.enable()
    return timings


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
    for name, (func, nbytes) in cases.items():
        timings = time_case(func, args.rounds)
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
