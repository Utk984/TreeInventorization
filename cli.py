import argparse
from config import Config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Street Level Tree Detection Pipeline"
    )

    parser.add_argument("--input_csv", "-i", default=None,
                        help="Path to panorama ID CSV (default: check config.py)")
    parser.add_argument("--output_csv", "-o", default=None,
                        help="Where to save tree data CSV (default: check config.py)")
    parser.add_argument("--fov", type=int, default=90,
                        help="Horizontal field of view in degrees (default: 90)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Perspective view width (pixels) (default: 1024)")
    parser.add_argument("--height", type=int, default=720,
                        help="Perspective view height (pixels) (default: 720)")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()

    if args.input_csv:
        cfg.PANORAMA_CSV = args.input_csv
    if args.output_csv:
        cfg.OUTPUT_CSV = args.output_csv 

    cfg.FOV = args.fov
    cfg.WIDTH = args.width
    cfg.HEIGHT = args.height

    return cfg
