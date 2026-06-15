# Обробка параметрів запуску.

import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument(
        "--pattern",
        type=int,
        default=1
    )

    parser.add_argument(
        "--size",
        type=float,
        default=0.010
    )

    parser.add_argument(
        "--flip",
        action="store_true",
        default=True
    )

    parser.add_argument(
        "--win",
        action="store_true",
        default=True
    )

    return parser.parse_args()
