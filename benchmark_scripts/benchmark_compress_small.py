"""Benchmark compressing small messages the way websocket permessage-deflate
does: one compressobj per connection, ``compress(message)`` followed by
``flush(Z_SYNC_FLUSH)`` for every message.

Besides synthetic JSON of fixed sizes, it sends two real Home Assistant
websocket frames: a state change of about 180 bytes and a two-event batch of
about 440 bytes. Their timestamps, ids and states change in every message,
as in a live event stream.

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

SIZES = [100, 256, 512, 1024, 2048, 4096, 8192, 12288, 65536]

HA_FRAMES = {
    "ha-state-change": (
        '{"type":"event","event":{"c":{"sensor.apollo_r_pro_1_eth_5938e0_'
        'ld2450_target_1_direction":{"+":{"s":"Stationary",'
        '"lc":1789415917.2674696,"c":"01M2GR01PK0X6HGGF5BJP78FSY"}}}},'
        '"id":3}'),
    "ha-event-batch": (
        '[{"type":"event","event":{"c":{"event.north_lpr_smart_detection":'
        '{"+":{"s":"2026-09-14T19:58:39.545+00:00","lc":1789415919.545507,'
        '"c":"01M2GR03XSFDPGFSZ88814257S","a":{"event_type":"vehicle",'
        '"event_id":"635d03e7-e0bc-4372-8419-a45de3016421",'
        '"smart_detect_types":["vehicle"]}}}}},"id":3},'
        '{"type":"event","event":{"c":{"binary_sensor.north_lpr_vehicle_'
        'detected":{"+":{"s":"on","lc":1789415919.5461612,'
        '"c":"01M2GR03XTXACPD2RTDDKEAP9T"}}}},"id":3}]'),
}
ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
STATES = ["Stationary", "Moving away", "Approaching", "on", "off"]


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


def ha_messages(frame: str, count: int) -> list:
    """count copies of a Home Assistant frame with a new timestamp, context
    id, event id and state in every message."""
    rng = random.Random(frame)
    obj = json.loads(frame)

    def vary(node) -> None:
        items = (node.items() if isinstance(node, dict)
                 else enumerate(node) if isinstance(node, list) else ())
        for key, value in list(items):
            if key == "lc":
                node[key] = 1789415917 + rng.random() * 1000
            elif key == "c" and isinstance(value, str):
                node[key] = "01M2GR" + "".join(
                    rng.choice(ULID_CHARS) for _ in range(20))
            elif key == "s" and isinstance(value, str):
                node[key] = rng.choice(STATES)
            elif key == "event_id":
                node[key] = f"{rng.getrandbits(128):032x}"
            else:
                vary(value)

    messages = []
    for _ in range(count):
        vary(obj)
        messages.append(json.dumps(obj, separators=(",", ":")).encode())
    return messages


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
    cases = [(str(size), json_messages(size, args.count)) for size in SIZES]
    cases += [(name, ha_messages(frame, args.count))
              for name, frame in HA_FRAMES.items()]
    print("message\tisal_ns_per_msg\tzlib_ns_per_msg")
    for name, messages in cases:
        results = []
        for module, level in ((isal_zlib, isal_zlib.Z_BEST_SPEED),
                              (zlib, zlib.Z_BEST_SPEED)):
            best = min(timeit.repeat(
                lambda: send(module, messages, level),
                number=1, repeat=args.rounds))
            results.append(best / len(messages) * 1e9)
        print(f"{name}\t{results[0]:.0f}\t{results[1]:.0f}")


if __name__ == "__main__":
    main()
