import argparse
from config import Config

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Street Level Tree Detection Pipeline"
    )

    # --- required ---
    parser.add_argument(
        "--input_csv", "-i", required=True,
        help="Path to panorama ID CSV (required)"
    )

    # --- optional overrides (defaults match Config) ---
    parser.add_argument("--output_csv", "-o", default=None,
                        help="Where to save tree data CSV (default: tree_data.csv in project root)")
    parser.add_argument("--fov", type=int, default=90,
                        help="Horizontal field of view in degrees (default: 90)")
    parser.add_argument("--width", type=int, default=1024,
                        help="Perspective view width (pixels) (default: 1024)")
    parser.add_argument("--height", type=int, default=720,
                        help="Perspective view height (pixels) (default: 720)")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()

    cfg.PANORAMA_CSV = args.input_csv
    if args.output_csv:
        cfg.STREET_OUTPUT_CSV = args.output_csv 
    else:
        cfg.STREET_OUTPUT_CSV = cfg.OUTPUT_CSV

    cfg.FOV = args.fov
    cfg.WIDTH = args.width
    cfg.HEIGHT = args.height

    return cfg
