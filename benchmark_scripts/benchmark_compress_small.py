"""Benchmark compressing small messages the way websocket permessage-deflate
does: one compressobj per connection, ``compress(message)`` followed by
``flush(Z_SYNC_FLUSH)`` for every message.

Usage:
    python benchmark_scripts/benchmark_compress_small.py
"""

import argparse
import json
import platform
import random
import sys
import timeit
import zlib

from isal import isal_zlib

SIZES = [100, 1024, 2048, 4096, 8192, 12288, 65536]


def json_messages(size: int, count: int) -> list:
    """Deterministic JSON-like messages of exactly size bytes."""
    rng = random.Random(size)
    items = [{"id": rng.randrange(10 ** 6),
              "name": f"user{rng.randrange(1000)}",
              "active": rng.random() < 0.5,
              "score": round(rng.random() * 100, 2),
              "tags": ["a", "bb", "ccc"][:rng.randrange(4)]}
             for _ in range(size // 20 + 64)]
    blob = json.dumps(items, separators=(",", ":")).encode()
    step = max(1, (len(blob) - size) // count)
    return [blob[i * step:i * step + size] for i in range(count)]


def send(module, messages: list, level: int) -> None:
    compressor = module.compressobj(level, module.DEFLATED, -15)
    for message in messages:
        compressor.compress(message) + compressor.flush(module.Z_SYNC_FLUSH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000,
                        help="Messages per round.")
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    print(f"# {platform.platform()} {platform.machine()}")
    print(f"# Python {sys.version.split()[0]} "
          f"zlib={zlib.ZLIB_RUNTIME_VERSION}")
    print("size_bytes\tisal_ns_per_msg\tzlib_ns_per_msg")
    for size in SIZES:
        messages = json_messages(size, args.count)
        results = []
        for module, level in ((isal_zlib, isal_zlib.Z_BEST_SPEED),
                              (zlib, zlib.Z_BEST_SPEED)):
            best = min(timeit.repeat(
                lambda: send(module, messages, level),
                number=1, repeat=args.rounds))
            results.append(best / len(messages) * 1e9)
        print(f"{size}\t{results[0]:.0f}\t{results[1]:.0f}")


if __name__ == "__main__":
    main()
